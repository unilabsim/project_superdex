#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package the SuperDex Unreal plugins as drop-in archives.

Produces `superdex_physics_unrealplugin_windows.zip` and
`superdex_robotics_unrealplugin_windows.zip`. Each unzips into the `Plugins/` directory of a
vanilla UE 5.5.4 C++ project on Windows, where UBT compiles the plugin's Unreal modules from
source the first time the project is opened.

Internally the native libraries are built by another build system and staged into each
plugin's ThirdParty tree by a separate fetch step. This builds the same libraries with CMake
and stages them at the same paths, so `Build.cs` does not have to know which produced them.

Stages:

  1. BUILD     the native libraries with CMake, from the superdex root.
  2. EXTRACT   a clean copy of each plugin, minus build output and internal-only files.
  3. LIBS      stage the built shared libraries into ThirdParty/<lib>/lib/Win64.
  4. HEADERS   copy the mochi/superdex include trees in as real files. In a source checkout
               these are committed symlinks whose targets resolve nowhere else; in the
               published tree they are stripped entirely. Either way the archive needs real
               files.
  5. UPLUGIN   flip EnabledByDefault so a consumer does not have to enable by hand.
  6. VALIDATE  nothing dropped from Source/, no symlink survived, nothing excluded leaked.
  7. ARCHIVE   zip with the plugin directory as the single top-level entry.

The archives are not independent: SuperDexRobotics links
`../SuperDexPhysics/ThirdParty/mochi_physics/lib/Win64/mochi_physics.dll.imp.lib` and
deliberately does not carry its own copy, so both must be installed side by side. Its
`.uplugin` declares the dependency.

Requirements: CMake >= 3.25, Ninja, and clang-cl 17 or newer. Mochi's CMake selects Clang and
fails outright on anything else. clang-cl finds an installed MSVC toolchain on its own, so an
x64 Native Tools Command Prompt is a convenient way to get everything on PATH rather than a
requirement.

Note for internal Windows use: invoke with `fbpython`, not `python`/`python3`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


logger: logging.Logger = logging.getLogger(__name__)

# Normalised but NOT symlink-resolved. The internal staging tool builds an export-shaped tree
# out of symlinks back into the source repository, and resolve() would follow this file's own
# link and report the source tree while running from the staged one.
SCRIPT_DIR = Path(os.path.abspath(__file__)).parent

# Directories and files that must never reach an archive. Build output, editor scratch, and
# the internal artifact-fetch manifest that this script replaces.
EXCLUDED_NAMES = frozenset(
    {
        "Binaries",
        "Intermediate",
        "Saved",
        "DerivedDataCache",
        ".vs",
        "OSIRIS",
        "BUCK",
        "TARGETS",
        "PACKAGE",
        "__pycache__",
    }
)

# `.umap.txt` files are T3D text exports, not loadable maps; shipping them gives a consumer
# levels that silently fail to open. `_BuiltData` is regenerated lighting data.
EXCLUDED_SUFFIXES = ("_BuiltData.uasset", ".umap.txt", ".pdb", ".pyc")


def _ignore_excluded(_directory: str, names: list[str]) -> set[str]:
    """copytree filter for content that must never reach an archive.

    Shared by stages 2 and 4 because stage 6 rejects this content wherever it came from, and
    nothing after stage 4 removes it: a stage that copied it in would fail the run and still
    write it into the archive.
    """
    return {n for n in names if n in EXCLUDED_NAMES or n.endswith(EXCLUDED_SUFFIXES)}


@dataclass(frozen=True)
class Lib:
    """`stem` is the base name shared by the CMake output and the name `Build.cs` expects."""

    stem: str
    needs_import_lib: bool


@dataclass(frozen=True)
class HeaderTree:
    """A native include tree copied into the plugin, replacing whatever was there."""

    # Destination, relative to the plugin root.
    dest: str
    # Which of Roots' two trees the source lives in.
    root: Literal["mochi", "superdex"]
    # Source, relative to that root.
    source: str


@dataclass(frozen=True)
class Plugin:
    name: str
    # Where the plugin lives, relative to the superdex root.
    export_subdir: str
    # Base name of the archive this plugin produces.
    archive_stem: str
    # Destination for staged libraries, relative to the plugin root.
    lib_dest: str
    libs: tuple[Lib, ...]
    headers: tuple[HeaderTree, ...] = ()

    @property
    def synthesized(self) -> tuple[Path, ...]:
        """Destinations stage 4 rebuilds, so stage 2 is not expected to preserve them."""
        return tuple(Path(tree.dest) for tree in self.headers)


PHYSICS = Plugin(
    name="SuperDexPhysics",
    export_subdir="unreal/SuperDexPhysics",
    archive_stem="superdex_physics_unrealplugin_windows",
    lib_dest="ThirdParty/mochi_physics/lib/Win64",
    libs=(
        Lib("mochi_physics", needs_import_lib=True),
        Lib("marl", needs_import_lib=False),
    ),
    headers=(
        HeaderTree(
            dest="Source/SuperDexPhysics/ThirdParty/mochi_core/include",
            root="mochi",
            source="mochi_core/include",
        ),
        HeaderTree(
            dest="Source/SuperDexPhysics/ThirdParty/mochi_physics/include",
            root="mochi",
            source="mochi_physics/include",
        ),
        # No SuperDexPhysics source includes superdex_physics.h, but Build.cs publishes this
        # path and SuperDexRobotics inherits it through its module dependency, so the Physics
        # archive has to carry it for the Robotics archive to compile.
        HeaderTree(
            dest="Source/SuperDexPhysics/ThirdParty/superdex_physics/include",
            root="superdex",
            source="superdex_physics/include",
        ),
    ),
)

ROBOTICS = Plugin(
    name="SuperDexRobotics",
    export_subdir="unreal/SuperDexRobotics",
    archive_stem="superdex_robotics_unrealplugin_windows",
    lib_dest="ThirdParty/superdex_robotics/lib/Win64",
    libs=(Lib("superdex_robotics", needs_import_lib=True),),
    headers=(
        HeaderTree(
            dest="Source/SuperDexRobotics/ThirdParty/superdex_robotics/include",
            root="superdex",
            source="superdex_robotics/include",
        ),
    ),
)

# Physics first: Robotics links against the import library staged into the Physics tree.
PLUGINS = (PHYSICS, ROBOTICS)

CMAKE_TARGETS = ("mochi_physics", "marl", "superdex_robotics")

# Stripped wherever they appear, matching the `(.*/)?` anchoring in Configerator's
# `project_superdex.cconf`.
# Depth matters: SuperDexRobotics.Build.cs decides SUPERDEXROBOTICS_WITH_BOT_SCENE by probing for
# one of these headers, so a missed one turns the API on against a DLL built without it.
STRIPPED_DIR_NAMES = frozenset({"internal", "experimental"})


def log(msg: str) -> None:
    logger.info(msg)


def run(cmd: list[str | Path]) -> None:
    argv = [str(c) for c in cmd]
    # Quoted so the echoed line can be pasted back; Windows paths have spaces in them.
    log("$ " + (subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)))
    subprocess.run(argv, check=True)


# --- Root resolution -------------------------------------------------------------------
# Only the published layout is built in, and it is spelled out rather than searched for, so
# an unfamiliar tree raises with every path that was tried instead of resolving somewhere
# unintended. A tree arranged some other way is named explicitly with --plugin-root.


@dataclass(frozen=True)
class Roots:
    superdex: Path
    mochi: Path
    # Plugin name -> plugin source directory.
    plugins: dict[str, Path]
    # "export" or "explicit"; reported so a surprising run is obvious from the first line.
    layout: str

    def source_of(self, tree: HeaderTree) -> Path:
        return (self.mochi if tree.root == "mochi" else self.superdex) / tree.source


def _abspath(path: Path) -> Path:
    """Normalise without following symlinks; see the note on SCRIPT_DIR."""
    return Path(os.path.abspath(path))


def _looks_like_mochi(path: Path) -> bool:
    return (path / "CMakeLists.txt").is_file() and (path / "mochi_core").is_dir()


def parse_plugin_roots(values: list[str]) -> dict[str, Path]:
    """Turn `--plugin-root Name=Path` arguments into a name -> directory mapping."""
    known = {plugin.name for plugin in PLUGINS}
    roots: dict[str, Path] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not raw:
            raise RuntimeError(f"--plugin-root wants NAME=PATH, got {value!r}.")
        if name not in known:
            raise RuntimeError(
                f"--plugin-root {name!r} is not a plugin. "
                f"Known: {', '.join(sorted(known))}."
            )
        roots[name] = _abspath(Path(raw))
    return roots


def resolve_roots(
    superdex_override: Path | None,
    mochi_override: Path | None,
    plugin_overrides: dict[str, Path],
) -> Roots:
    superdex = _abspath(superdex_override or SCRIPT_DIR.parent)
    if not (superdex / "CMakeLists.txt").is_file():
        raise RuntimeError(f"{superdex} is not the superdex root (no CMakeLists.txt).")

    # In the published tree mochi is mapped under superdex_physics/libraries/mochi, which is
    # what superdex_physics/cmake/paths.cmake defaults to. A tree that puts it anywhere else
    # names it with --mochi-root, the same way it names the plugins with --plugin-root.
    if mochi_override is not None:
        mochi = _abspath(mochi_override)
        if not _looks_like_mochi(mochi):
            raise RuntimeError(f"--mochi-root {mochi} is not a mochi root.")
    else:
        mochi = superdex / "superdex_physics" / "libraries" / "mochi"
        if not _looks_like_mochi(mochi):
            raise RuntimeError(
                f"Could not locate the mochi root. Tried:\n  {mochi}\n"
                "Pass --mochi-root to point at it explicitly."
            )

    found = {plugin.name: superdex / plugin.export_subdir for plugin in PLUGINS}
    found.update(plugin_overrides)
    if all((d / f"{n}.uplugin").is_file() for n, d in found.items()):
        return Roots(
            superdex=superdex,
            mochi=mochi,
            plugins=found,
            layout="explicit" if plugin_overrides else "export",
        )

    raise RuntimeError(
        "Could not locate both plugins. Tried:\n  "
        + "\n  ".join(f"{n}: {d}" for n, d in sorted(found.items()))
        + "\nPass --plugin-root NAME=PATH to point at a plugin explicitly."
    )


# --- Preflight -------------------------------------------------------------------------


def preflight(generator: str) -> str | None:
    """Report which toolchain PATH resolved to, and reject a missing CMake.

    With more than one CMake or LLVM installed the choice is otherwise invisible, and stage 1
    inherits it silently.
    """
    log("build toolchain")
    tools = ["cmake"]
    if generator.startswith("Ninja"):
        tools.append("ninja")
    if platform.system() == "Windows":
        tools.append("clang-cl")

    resolved = {name: shutil.which(name) for name in tools}
    for name, path in resolved.items():
        log(f"  {name}: {path or 'not found'}")

    # Only CMake is checked, because only it fails as a Python traceback: the ENOENT out of
    # subprocess does not say what was missing. Everything past a successful cmake invocation
    # names its own problem. CMake reports a missing Ninja, and mochi's cmake/compiler.cmake
    # rejects a wrong or too-old compiler with install instructions.
    return None if resolved["cmake"] else "cmake not found on PATH."


# --- Stage 1: build --------------------------------------------------------------------


def build_natives(roots: Roots, build_dir: Path, config: str, generator: str) -> None:
    log("STAGE 1: building the native libraries with CMake")
    run(
        [
            "cmake",
            "-S",
            roots.superdex,
            "-B",
            build_dir,
            "-G",
            generator,
            f"-DCMAKE_BUILD_TYPE={config}",
            "-DMOCHI_BUILD_SHARED=ON",
            "-DSUPERDEX_BUILD_ROBOTICS=ON",
            # Without this the DLLs disagree with UBT on struct packing; see settings.cmake.
            "-DMOCHI_UNREAL_ABI=ON",
            # Profiling, Python bindings, and applications that the plugins do not use.
            "-DMOCHI_USE_TRACY=OFF",
            "-DMOCHI_USE_PYBIND=OFF",
            "-DMOCHI_BUILD_DEBUGGER=OFF",
            "-DMOCHI_BUILD_TESTS=OFF",
            # MOCHI_USE_HDF5 stays at its default of ON. UMochiModel reads and writes .h5
            # through the DLL, and the internal build enables HDF5 too, so turning it off
            # here would ship a plugin that cannot open its own assets and does not match
            # the libraries the same plugins link against internally.
        ]
    )
    run(
        [
            "cmake",
            "--build",
            build_dir,
            "--config",
            config,
            "--parallel",
            str(os.cpu_count() or 1),
            "--target",
            *CMAKE_TARGETS,
        ]
    )


# --- Stage 2: extract ------------------------------------------------------------------


def _make_ignore(
    plugin: Plugin, plugin_root: Path
) -> Callable[[str, list[str]], set[str]]:
    """copytree filter. Scoped by directory, because ThirdParty means three different things.

    At the plugin root it is the staging tree stage 3 refills, so it goes. Under
    Source/<Module> it is either a stage 4 destination, which stage 4 refills, or vendored
    source such as the fTetWild tree SuperDexPhysicsEditor keeps under Private/ -- and
    dropping that would silently remove code the plugin needs to compile.
    """
    synthesized = frozenset(plugin.synthesized)

    def _ignore(directory: str, names: list[str]) -> set[str]:
        here = Path(directory)
        dropped = _ignore_excluded(directory, names)
        if here == plugin_root:
            dropped.add("ThirdParty")
        # Stage 4 deletes and rebuilds these, so copying them costs a large tree twice. In a
        # source checkout they are also the committed symlinks, where one dead target would
        # abort the copy outright.
        rel = here.relative_to(plugin_root)
        dropped |= {n for n in names if rel / n in synthesized}
        return dropped

    return _ignore


def extract_plugin(plugin: Plugin, source: Path, staging_root: Path) -> Path:
    dest = staging_root / plugin.name
    log(f"STAGE 2: extracting {plugin.name} -> {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=_make_ignore(plugin, source), symlinks=False)
    return dest


# --- Stage 3: libraries ----------------------------------------------------------------


def _find_candidates(build_dir: Path, pattern: str, config: str) -> list[Path]:
    """Every match for `pattern`, narrowed to `config` only when that leaves exactly one.

    A multi-config generator emits the same name under Debug/ and Release/; anything else
    that matches more than once is a build tree this script cannot reason about.
    """
    hits = sorted(h for h in build_dir.rglob(pattern) if h.is_file())
    if len(hits) <= 1:
        return hits
    configured = [
        hit
        for hit in hits
        if config.casefold()
        in {part.casefold() for part in hit.relative_to(build_dir).parts[:-1]}
    ]
    return configured if len(configured) == 1 else hits


def _stage(build_dir: Path, pattern: str, dest: Path, config: str) -> str | None:
    """Copy the build tree's one match for `pattern` to `dest`. Returns a problem, or None.

    Absent and ambiguous are reported apart: calling an artifact missing when the tree holds
    two of it sends the reader looking for something that is already there.
    """
    hits = _find_candidates(build_dir, pattern, config)
    if not hits:
        return f"{pattern}: not found under {build_dir}"
    if len(hits) > 1:
        found = ", ".join(str(h.relative_to(build_dir)) for h in hits)
        return f"{pattern}: {len(hits)} matches, none unique to {config}: {found}"
    shutil.copy2(hits[0], dest)
    log(f"  staged {dest.name}")
    return None


def stage_libs(plugin: Plugin, staged: Path, build_dir: Path, config: str) -> list[str]:
    dest = staged / plugin.lib_dest
    dest.mkdir(parents=True, exist_ok=True)
    log(f"STAGE 3: staging libraries into {plugin.lib_dest}")
    # Off Windows CMake emits .so, so there is nothing here to find and every library would
    # be reported missing. Noted instead of failed, so --allow-non-windows still gives a
    # trustworthy answer about the stages it does exercise.
    required = platform.system() == "Windows"
    problems: list[str] = []
    for lib in plugin.libs:
        wanted = [(f"{lib.stem}.dll", dest / f"{lib.stem}.dll")]
        if lib.needs_import_lib:
            wanted.append((f"{lib.stem}.lib", dest / f"{lib.stem}.dll.imp.lib"))
        for pattern, target in wanted:
            problem = _stage(build_dir, pattern, target, config)
            if problem is None:
                continue
            if required:
                problems.append(problem)
            else:
                log(f"  not staged: {pattern}")
    return problems


# --- Stage 4: headers ------------------------------------------------------------------


def stage_headers(plugin: Plugin, staged: Path, roots: Roots) -> list[str]:
    if not plugin.headers:
        return []
    log("STAGE 4: copying the native include trees")
    missing: list[str] = []
    for tree in plugin.headers:
        source = roots.source_of(tree)
        dest = staged / tree.dest
        if not source.is_dir():
            # In the export layout this usually means Configerator's `project_superdex.cconf`
            # does not map the library this header tree comes from.
            missing.append(f"header source not found: {source}")
            continue
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # symlinks=False dereferences: a source checkout reaches these through symlinks and
        # the archive must hold the files themselves.
        shutil.copytree(source, dest, symlinks=False, ignore=_ignore_excluded)
        stripped = _strip_internal_headers(dest)
        suffix = f" (stripped {stripped} internal dir(s))" if stripped else ""
        log(f"  copied {tree.source} -> {tree.dest}{suffix}")
    return missing


def _strip_internal_headers(dest: Path) -> int:
    """Drop every internal/ and experimental/ directory, at any depth.

    Returns the number of directories removed.
    """
    removed = 0
    # topdown so pruning a match also prunes everything beneath it.
    for parent, dirs, _files in os.walk(dest, topdown=True):
        for name in [d for d in dirs if d in STRIPPED_DIR_NAMES]:
            shutil.rmtree(Path(parent) / name)
            dirs.remove(name)
            removed += 1
    return removed


# --- Stage 5: .uplugin -----------------------------------------------------------------


def patch_uplugin(plugin: Plugin, staged: Path) -> None:
    log("STAGE 5: setting EnabledByDefault=true")
    path = staged / f"{plugin.name}.uplugin"
    data = json.loads(path.read_text(encoding="utf-8"))
    # Left false in the repository so internal projects that pick these plugins up without
    # listing them in their .uproject keep their current behaviour.
    data["EnabledByDefault"] = True
    path.write_text(json.dumps(data, indent="\t") + "\n", encoding="utf-8")


# --- Stage 6: validate -----------------------------------------------------------------


def _dropped_sources(plugin: Plugin, source: Path, staged: Path) -> list[str]:
    """Every file under Source/ must survive extraction.

    The include trees in plugin.headers are the exception, since stage 4 strips and rebuilds
    them. Everything else is code the plugin needs to compile, including module-private
    vendored third-party source.
    """
    dropped = []
    for path in (source / "Source").rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        posix = rel.as_posix()
        if any(
            rel.is_relative_to(prefix) for prefix in plugin.synthesized
        ) or posix.endswith(EXCLUDED_SUFFIXES):
            continue
        if any(part in EXCLUDED_NAMES for part in rel.parts):
            continue
        if not (staged / rel).exists():
            dropped.append(posix)
    return dropped


def validate(plugin: Plugin, source: Path, staged: Path) -> list[str]:
    """Only what the earlier stages cannot see: they report their own missing artifacts."""
    log("STAGE 6: validating")
    problems: list[str] = [
        f"source file dropped: {p}" for p in _dropped_sources(plugin, source, staged)
    ]

    for parent, dirs, files in os.walk(staged, followlinks=False):
        for name in dirs + files:
            entry = Path(parent) / name
            rel = entry.relative_to(staged)
            # Cannot fire while every copy dereferences -- stages 2 and 4 pass symlinks=False,
            # stage 3 uses copy2 -- which is what actually resolves a source checkout's header
            # links. This guards that invariant rather than those links: one that did survive
            # would point out of the archive, at a target no consumer has.
            if entry.is_symlink():
                problems.append(f"symlink survived: {rel}")
            if name in EXCLUDED_NAMES or name.endswith(EXCLUDED_SUFFIXES):
                problems.append(f"excluded content leaked: {rel}")
            # These nest several levels down, and one that stage 4 missed ships internal
            # headers against a DLL that does not export them. Scoped to the trees stage 4
            # strips, since a plugin's own internal/ directory is its own business.
            if (
                name in STRIPPED_DIR_NAMES
                and entry.is_dir()
                and any(rel.is_relative_to(prefix) for prefix in plugin.synthesized)
            ):
                problems.append(f"internal directory survived: {rel}")

    if problems:
        for p in problems:
            log(f"  FAIL {p}")
    else:
        log("  ok: no source dropped, no symlinks, nothing excluded leaked")
    return problems


# --- Stage 7: archive ------------------------------------------------------------------


def archive(plugin: Plugin, staged: Path, out_dir: Path) -> Path:
    out = out_dir / f"{plugin.archive_stem}.zip"
    log(f"STAGE 7: writing {out.name}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staged.rglob("*")):
            if path.is_file():
                # The plugin directory is the single top-level entry, so unzipping into
                # Plugins/ lands as Plugins/<Name>/... rather than scattering the contents.
                zf.write(path, Path(plugin.name) / path.relative_to(staged))
    size_mb = out.stat().st_size / (1024 * 1024)
    log(f"  {out.name}: {size_mb:.1f} MiB")
    return out


# --- Driver ----------------------------------------------------------------------------


def package_one(
    plugin: Plugin, roots: Roots, args: argparse.Namespace
) -> tuple[Path, list[str]]:
    """Stages 2 to 7 for one plugin. Returns its archive and any problems found.

    The archive is written even when there are problems, so a failed run can be inspected.
    """
    log("")
    log(f"=== {plugin.name} ===")
    source = roots.plugins[plugin.name]
    staged = extract_plugin(plugin, source, args.staging_dir)
    problems = stage_libs(plugin, staged, args.build_dir, args.config)
    problems += stage_headers(plugin, staged, roots)
    patch_uplugin(plugin, staged)
    problems += validate(plugin, source, staged)
    return archive(plugin, staged, args.out_dir), [
        f"{plugin.name}: {problem}" for problem in problems
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--superdex-root", type=Path, default=None)
    parser.add_argument("--mochi-root", type=Path, default=None)
    parser.add_argument(
        "--plugin-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="locate a plugin explicitly instead of under --superdex-root, e.g. "
        "--plugin-root SuperDexPhysics=/path/to/SuperDexPhysics. Repeatable. Only needed "
        "for a tree that does not lay the plugins out the way the repository does.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "superdex-unreal-build",
    )
    # Not the working directory: an x64 Native Tools Command Prompt starts in the Visual
    # Studio install, which is not writable, and the failure would land after the build.
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "superdex-unreal-dist",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "superdex-unreal-staging",
    )
    parser.add_argument(
        "--config", default="Release", choices=["Debug", "Release", "RelWithDebInfo"]
    )
    parser.add_argument(
        "--generator",
        default="Ninja",
        help="CMake generator. Ninja (the default) needs a compiler already on PATH. A "
        "Visual Studio generator would need -T ClangCL, which this script does not "
        "pass, so Ninja is the supported Windows configuration.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse an existing --build-dir instead of running CMake",
    )
    parser.add_argument(
        "--allow-non-windows",
        action="store_true",
        help="run the packaging stages off Windows. The archives are Win64-only, so this is "
        "for exercising the pipeline, not for producing a release.",
    )
    return parser.parse_args()


def package(args: argparse.Namespace) -> int:
    if platform.system() != "Windows" and not args.allow_non_windows:
        log(f"these archives are Win64-only and this is {platform.system()}.")
        log("pass --allow-non-windows to exercise the packaging stages anyway.")
        return 2

    roots = resolve_roots(
        args.superdex_root, args.mochi_root, parse_plugin_roots(args.plugin_root)
    )
    log(f"layout   : {roots.layout}")
    log(f"superdex : {roots.superdex}")
    log(f"mochi    : {roots.mochi}")
    log(f"output   : {args.out_dir}")

    # Up front, so an unwritable destination costs a second rather than a whole build.
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        blocked = preflight(args.generator)
        if blocked:
            log(f"cannot build the native libraries: {blocked}")
            log("pass --skip-build to reuse an existing --build-dir instead.")
            return 2
        build_natives(roots, args.build_dir, args.config, args.generator)

    # extract_plugin removes and re-creates each plugin's own subdirectory, so this only
    # needs to exist. Nothing here deletes a directory the script does not own, so a
    # caller-supplied --staging-dir keeps any unrelated content it already holds.
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    produced: list[Path] = []
    for plugin in PLUGINS:
        out, found = package_one(plugin, roots, args)
        produced.append(out)
        problems += found

    log("")
    if problems:
        log(f"FAILED with {len(problems)} problem(s):")
        for p in problems:
            log(f"  {p}")
        return 1

    log("=" * 70)
    for out in produced:
        log(f"Done: {out}")
    if platform.system() == "Windows":
        log(
            "Install both: SuperDexRobotics links an import library from SuperDexPhysics."
        )
    else:
        # Success here means stages 2 and 4-7 held, not that the archives are usable.
        log("No libraries staged off Windows: these archives will not install.")
    log("=" * 70)
    return 0


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[package-plugins] %(message)s", stream=sys.stdout
    )
    try:
        return package(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        # These are all "the environment is wrong" failures, which a traceback buries.
        log(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
