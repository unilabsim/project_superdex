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

"""Validate and install an explicit SuperDex Lab wheel/sdist pair."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile
from collections.abc import Sequence
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

DISTRIBUTION = "superdex-lab"


def _project_root(project_root: Path | None = None) -> Path:
    return (project_root or Path(__file__).resolve().parents[2]).resolve()


def _safe_pattern(value: str, source: str) -> PurePosixPath:
    parts = value.split("/")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or Path(value).is_absolute()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"{source}: unsafe relative path pattern {value!r}")
    return path


def _expand_source_pattern(root: Path, pattern: PurePosixPath, source: str) -> set[str]:
    matches: set[str] = set()
    for match in root.glob(pattern.as_posix()):
        if not match.is_file():
            continue
        resolved = match.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{source}: path escapes project root: {match}") from error
        matches.add(relative.as_posix())
    if not matches:
        raise ValueError(f"{source}: pattern matches no source files: {pattern}")
    return matches


def package_data_paths(project_root: Path | None = None) -> tuple[str, ...]:
    """Return concrete package-data paths declared by pyproject.toml."""
    root = _project_root(project_root)
    with (root / "pyproject.toml").open("rb") as stream:
        package_data = tomllib.load(stream)["tool"]["setuptools"]["package-data"]
    paths: set[str] = set()
    for package, entries in package_data.items():
        components = package.split(".")
        if (
            not components
            or any(not part or part in {".", ".."} for part in components)
            or "/" in package
            or "\\" in package
        ):
            raise ValueError(f"pyproject.toml: unsafe package name {package!r}")
        for entry in entries:
            pattern = PurePosixPath(*components) / _safe_pattern(
                entry, "pyproject.toml"
            )
            paths.update(
                _expand_source_pattern(
                    root,
                    pattern,
                    f"pyproject.toml package-data {package!r} entry {entry!r}",
                )
            )
    return tuple(sorted(paths))


def _metadata(content: bytes, source: str) -> tuple[str, str]:
    parsed = BytesParser(policy=policy.default).parsebytes(content)
    name, version = parsed.get("Name"), parsed.get("Version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError(f"{source}: missing Name or Version metadata")
    return name, version


def _wheel(path: Path) -> tuple[str, set[str]]:
    match = re.fullmatch(r"superdex_lab-([^-]+)-[^-]+-[^-]+-[^-]+\.whl", path.name)
    if match is None:
        raise ValueError(f"{path.name}: expected a superdex_lab wheel filename")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(
                    f"{path.name}: contains duplicate members {duplicates!r}"
                )
            metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError(f"{path.name}: expected exactly one METADATA file")
            expected_dist_info = f"superdex_lab-{match.group(1)}.dist-info"
            if PurePosixPath(metadata[0]).parts[0] != expected_dist_info:
                raise ValueError(
                    f"{path.name}: METADATA is outside {expected_dist_info}"
                )
            identity = _metadata(archive.read(metadata[0]), path.name)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"{path.name}: cannot read wheel: {error}") from error
    if identity != (DISTRIBUTION, match.group(1)):
        raise ValueError(f"{path.name}: filename and METADATA identity differ")
    return match.group(1), set(names)


def _sdist(path: Path) -> tuple[str, set[str], set[str]]:
    match = re.fullmatch(r"superdex_lab-([^-]+)\.tar\.gz", path.name)
    if match is None:
        raise ValueError(f"{path.name}: expected a superdex_lab sdist filename")
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            roots = {PurePosixPath(member.name).parts[0] for member in members}
            expected_root = f"superdex_lab-{match.group(1)}"
            if roots != {expected_root}:
                raise ValueError(f"{path.name}: expected one root {expected_root!r}")
            pkg_info_members = [
                member
                for member in members
                if member.name == f"{expected_root}/PKG-INFO" and member.isfile()
            ]
            if len(pkg_info_members) != 1:
                raise ValueError(f"{path.name}: expected exactly one root PKG-INFO")
            pkg_info = archive.extractfile(pkg_info_members[0])
            assert pkg_info is not None
            identity = _metadata(pkg_info.read(), path.name)
            names = {
                PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
                for member in members
                if len(PurePosixPath(member.name).parts) > 1
            }
            files = {
                PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
                for member in members
                if member.isfile() and len(PurePosixPath(member.name).parts) > 1
            }
    except (OSError, tarfile.TarError, KeyError) as error:
        raise ValueError(f"{path.name}: cannot read sdist: {error}") from error
    if identity != (DISTRIBUTION, match.group(1)):
        raise ValueError(f"{path.name}: filename and PKG-INFO identity differ")
    return match.group(1), names, files


def _check_no_internal_paths(source: str, names: set[str]) -> None:
    internal = sorted(
        name
        for name in names
        if "internal" in (part.casefold() for part in PurePosixPath(name).parts)
    )
    if internal:
        raise ValueError(f"{source}: contains internal paths {internal!r}")


def _is_relevant_data(name: str) -> bool:
    path = PurePosixPath(name)
    return name.startswith("superdex/") and (
        path.suffix == ".json" or path.name == "py.typed"
    )


def check_archives(
    wheel: Path,
    sdist: Path,
    project_root: Path | None = None,
) -> None:
    wheel_version, wheel_names = _wheel(wheel)
    sdist_version, sdist_names, sdist_files = _sdist(sdist)
    if wheel_version != sdist_version:
        raise ValueError(
            f"artifact version mismatch: wheel={wheel_version}, sdist={sdist_version}"
        )

    root = _project_root(project_root)
    expected_data = {
        name for name in package_data_paths(root) if _is_relevant_data(name)
    }
    actual_data = {name for name in wheel_names if _is_relevant_data(name)}
    _check_no_internal_paths(wheel.name, wheel_names)
    _check_no_internal_paths(sdist.name, sdist_names)
    if actual_data != expected_data:
        raise ValueError(
            f"{wheel.name}: package data inventory differs: "
            f"expected={sorted(expected_data)!r}, actual={sorted(actual_data)!r}"
        )
    missing_sdist = sorted(expected_data - sdist_files)
    if missing_sdist:
        raise ValueError(f"{sdist.name}: missing package data {missing_sdist!r}")


def _venv_python(root: Path) -> Path:
    venv.EnvBuilder(with_pip=True).create(root)
    for relative in (Path("bin/python"), Path("Scripts/python.exe")):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"cannot find virtualenv interpreter below {root}")


def _probe_source(expected_package_data: Sequence[str]) -> str:
    return f"""
import importlib.util
import json
import sys
import sysconfig
from pathlib import Path, PurePosixPath, PureWindowsPath

from gymnasium.envs.registration import registry

expected_package_data = {tuple(expected_package_data)!r}
venv_root = Path(sys.prefix).resolve()
root = Path(sysconfig.get_path("purelib")).resolve()
registration_path = (root / "superdex/lab/gym/registration.py").resolve()
assert registration_path.is_relative_to(venv_root) and registration_path.is_file()
spec = importlib.util.spec_from_file_location("superdex_lab_registration_probe", registration_path)
assert spec is not None and spec.loader is not None
registration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registration)
registration.register_envs()
env_specs = registration.get_env_specs()
assert env_specs
ids = tuple(env_spec.id for env_spec in env_specs)
assert ids == tuple(sorted(ids))
assert ids == tuple(env_spec.id for env_spec in registration.get_env_specs())
assert len(ids) == len(set(ids))
for env_spec in env_specs:
    assert env_spec.namespace == "superdex_gym"
    registered = registry[env_spec.id]
    assert env_spec is registered or env_spec == registered
    entry_points = (env_spec.entry_point, env_spec.kwargs["env_cls"])
    for entry_point in entry_points:
        assert isinstance(entry_point, str) and entry_point
        module_name = entry_point.partition(":")[0]
        assert module_name == "superdex.lab.gym" or module_name.startswith("superdex.lab.gym.")
        module_base = root.joinpath(*module_name.split("."))
        module_paths = (module_base.with_suffix(".py"), module_base / "__init__.py")
        assert any(path.resolve().is_relative_to(venv_root) and path.is_file() for path in module_paths)
actual_package_data = tuple(sorted(
    path.relative_to(root).as_posix()
    for path in (root / "superdex").rglob("*")
    if path.is_file() and (path.suffix == ".json" or path.name == "py.typed")
))
assert actual_package_data == expected_package_data, (actual_package_data, expected_package_data)

recipe_path = root / "superdex/lab/rllib/recipe_manifest.py"
manifest_path = recipe_path.with_name("recipes") / "manifest.json"
if recipe_path.is_file() and manifest_path.is_file():
    recipe_spec = importlib.util.spec_from_file_location(
        "installed_superdex_lab_recipe_manifest", recipe_path
    )
    assert recipe_spec is not None and recipe_spec.loader is not None
    recipe_manifest = importlib.util.module_from_spec(recipe_spec)
    sys.modules[recipe_spec.name] = recipe_manifest
    recipe_spec.loader.exec_module(recipe_manifest)

    manifest_path = recipe_manifest.PUBLIC_RECIPE_MANIFEST
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(raw_manifest, dict) and raw_manifest
    recipe_manifest.load_recipe_manifest(manifest_path, "__packaging_probe__")
    kinds = {{
        kind
        for record in raw_manifest.values()
        for kind in record.get("recipes", {{}})
    }}
    for kind in kinds:
        recipe_manifest.load_recipe_manifest(manifest_path, kind)

    recipes = recipe_manifest.load_recipes("train")
    expected_train = tuple(
        (env_id, record["slug"])
        for env_id, record in raw_manifest.items()
        if "train" in record.get("recipes", {{}})
    )
    actual_train = tuple(
        (env_id, recipe.slug) for env_id, recipe in recipes.items()
    )
    assert actual_train == expected_train, (actual_train, expected_train)

    recipe_root = manifest_path.parent
    expected_recipe_json = {{"manifest.json"}}
    for record in raw_manifest.values():
        for relative_path in record.get("recipes", {{}}).values():
            assert isinstance(relative_path, str) and relative_path
            posix_path = PurePosixPath(relative_path)
            windows_path = PureWindowsPath(relative_path)
            assert not posix_path.is_absolute()
            assert not windows_path.is_absolute() and not windows_path.drive
            assert "\\\\" not in relative_path
            assert all(part not in {{"", ".", ".."}} for part in posix_path.parts)
            assert posix_path.as_posix() == relative_path
            expected_recipe_json.add(relative_path)
    actual_recipe_json = {{
        path.relative_to(recipe_root).as_posix()
        for path in recipe_root.rglob("*.json")
    }}
    assert actual_recipe_json == expected_recipe_json, (
        actual_recipe_json,
        expected_recipe_json,
    )
"""


def check_install(wheel: Path, project_root: Path | None = None) -> None:
    expected_package_data = tuple(
        name for name in package_data_paths(project_root) if _is_relevant_data(name)
    )
    with tempfile.TemporaryDirectory(prefix="superdex-lab-wheel-") as temporary:
        root = Path(temporary)
        python = _venv_python(root / "venv")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        pip = [str(python), "-m", "pip", "--isolated", "--disable-pip-version-check"]
        subprocess.run(
            [*pip, "install", "gymnasium>=1.1.1"],
            check=True,
            cwd=root,
            env=environment,
        )
        subprocess.run(
            [*pip, "install", "--no-deps", str(wheel.resolve())],
            check=True,
            cwd=root,
            env=environment,
        )
        subprocess.run(
            [str(python), "-I", "-P", "-c", _probe_source(expected_package_data)],
            check=True,
            cwd=root,
            env=environment,
        )


def _artifact(value: str, suffix: str) -> Path:
    path = Path(value)
    if not path.is_file() or not path.name.endswith(suffix):
        raise argparse.ArgumentTypeError(
            f"expected an existing {suffix} artifact: {path}"
        )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel", required=True, type=lambda value: _artifact(value, ".whl")
    )
    parser.add_argument(
        "--sdist", required=True, type=lambda value: _artifact(value, ".tar.gz")
    )
    args = parser.parse_args(argv)
    try:
        check_archives(args.wheel, args.sdist)
        check_install(args.wheel)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Checked {args.wheel} and {args.sdist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
