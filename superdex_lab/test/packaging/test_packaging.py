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

from __future__ import annotations

import io
import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from . import check_artifacts


def _source() -> Path:
    return Path(os.environ["SUPERDEX_LAB_SOURCE"])


def _members(project: Path) -> dict[str, bytes]:
    return dict.fromkeys(check_artifacts.package_data_paths(project), b"fixture\n")


def _write_pair(
    root: Path,
    project: Path,
    version: str = "1.0.0",
    *,
    dist_info: str | None = None,
    duplicate_pkg_info: bool = False,
) -> tuple[Path, Path]:
    members = _members(project)
    wheel = root / f"superdex_lab-{version}-py3-none-any.whl"
    metadata = (
        f"Metadata-Version: 2.4\nName: superdex-lab\nVersion: {version}\n\n"
    ).encode()
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        archive.writestr(
            f"{dist_info or f'superdex_lab-{version}.dist-info'}/METADATA", metadata
        )

    sdist = root / f"superdex_lab-{version}.tar.gz"
    entries = [*members.items(), ("PKG-INFO", metadata)]
    if duplicate_pkg_info:
        entries.append(("PKG-INFO", metadata))
    with tarfile.open(sdist, "w:gz") as archive:
        for name, content in entries:
            info = tarfile.TarInfo(f"superdex_lab-{version}/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return wheel, sdist


class PackagingTest(unittest.TestCase):
    def test_package_data_parser_derives_existing_concrete_paths(self) -> None:
        project = _source()
        package_data = check_artifacts.package_data_paths(project)

        self.assertTrue(package_data)
        self.assertEqual(tuple(sorted(package_data)), package_data)
        for member in package_data:
            self.assertTrue((project / member).is_file(), member)

    def test_package_data_parser_expands_globs_and_rejects_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            data = project / "superdex" / "lab" / "data"
            data.mkdir(parents=True)
            (data / "one.json").write_text("{}")
            (data / "two.json").write_text("{}")
            (project / "pyproject.toml").write_text(
                '[tool.setuptools.package-data]\n"superdex.lab" = ["data/*.json"]\n'
            )
            self.assertEqual(
                (
                    "superdex/lab/data/one.json",
                    "superdex/lab/data/two.json",
                ),
                check_artifacts.package_data_paths(project),
            )
            (project / "pyproject.toml").write_text(
                '[tool.setuptools.package-data]\n"superdex.lab" = ["../secret.json"]\n'
            )
            with self.assertRaisesRegex(ValueError, "unsafe relative path"):
                check_artifacts.package_data_paths(project)

    def test_matching_archives_satisfy_the_public_contract(self) -> None:
        project = _source()
        with tempfile.TemporaryDirectory() as temporary:
            wheel, sdist = _write_pair(Path(temporary), project)
            check_artifacts.check_archives(wheel, sdist, project)

    def test_archive_contract_rejects_internal_and_undeclared_data(self) -> None:
        project = _source()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel, sdist = _write_pair(root, project)
            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr("superdex/lab/example/internal/marker.py", b"fixture")
            with self.assertRaisesRegex(ValueError, "internal paths"):
                check_artifacts.check_archives(wheel, sdist, project)

            for version, filename in (
                ("1.0.1", "extra.json"),
                ("1.0.2", "extra_test.json"),
            ):
                wheel, sdist = _write_pair(root, project, version)
                with zipfile.ZipFile(wheel, "a") as archive:
                    archive.writestr(
                        f"superdex/lab/gym/envs/benchmarks/{filename}", b"{}"
                    )
                with self.assertRaisesRegex(ValueError, "package data inventory"):
                    check_artifacts.check_archives(wheel, sdist, project)

    def test_archive_contract_requires_all_package_data_in_sdist(self) -> None:
        project = _source()
        missing = check_artifacts.package_data_paths(project)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel, sdist = _write_pair(root, project)
            replacement = root / "missing.tar.gz"
            with (
                tarfile.open(sdist, "r:gz") as source,
                tarfile.open(replacement, "w:gz") as target,
            ):
                for member in source.getmembers():
                    if member.name.endswith(f"/{missing}"):
                        continue
                    content = source.extractfile(member) if member.isfile() else None
                    target.addfile(member, content)
            replacement.replace(sdist)
            with self.assertRaisesRegex(ValueError, "missing package data"):
                check_artifacts.check_archives(wheel, sdist, project)

    def test_archive_contract_rejects_duplicate_wheel_members(self) -> None:
        project = _source()
        with tempfile.TemporaryDirectory() as temporary:
            wheel, sdist = _write_pair(Path(temporary), project)
            duplicate = check_artifacts.package_data_paths(project)[0]
            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr(duplicate, b"duplicate")
            with self.assertRaisesRegex(ValueError, "duplicate members"):
                check_artifacts.check_archives(wheel, sdist, project)

    def test_archive_contract_rejects_undeclared_recipe_json(self) -> None:
        project = _source()
        declared_recipe = next(
            Path(path)
            for path in check_artifacts.package_data_paths(project)
            if "/rllib/recipes/" in path and path.endswith(".json")
        )
        undeclared_recipe = (declared_recipe.parent / "undeclared.json").as_posix()
        with tempfile.TemporaryDirectory() as temporary:
            wheel, sdist = _write_pair(Path(temporary), project)
            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr(undeclared_recipe, b"{}")
            with self.assertRaisesRegex(ValueError, "package data inventory"):
                check_artifacts.check_archives(wheel, sdist, project)

    def test_archive_contract_rejects_misplaced_metadata(self) -> None:
        project = _source()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel, sdist = _write_pair(root, project, dist_info="wrong-1.0.0.dist-info")
            with self.assertRaisesRegex(ValueError, "METADATA is outside"):
                check_artifacts.check_archives(wheel, sdist, project)

            wheel, sdist = _write_pair(root, project, duplicate_pkg_info=True)
            with self.assertRaisesRegex(ValueError, "exactly one root PKG-INFO"):
                check_artifacts.check_archives(wheel, sdist, project)

    def test_archive_contract_rejects_version_mismatch(self) -> None:
        project = _source()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel, _ = _write_pair(root, project, "1.0.0")
            _, sdist = _write_pair(root, project, "2.0.0")
            with self.assertRaisesRegex(ValueError, "artifact version mismatch"):
                check_artifacts.check_archives(wheel, sdist, project)

    def test_install_uses_gymnasium_then_exact_wheel_without_dependencies(self) -> None:
        project = _source()
        wheel = Path("superdex_lab-1.0.0-py3-none-any.whl")
        python = Path("venv") / "bin" / "python"
        with (
            mock.patch.object(check_artifacts, "_venv_python", return_value=python),
            mock.patch.object(check_artifacts.subprocess, "run") as run,
        ):
            check_artifacts.check_install(wheel, project)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual("gymnasium>=1.1.1", commands[0][-1])
        self.assertEqual(["install", "--no-deps"], commands[1][-3:-1])
        self.assertEqual(str(wheel.resolve()), commands[1][-1])
        self.assertEqual([str(python), "-I", "-P", "-c"], commands[2][:4])
        for member in check_artifacts.package_data_paths(project):
            self.assertIn(member, commands[2][-1])

    def test_probe_derives_installed_registries_without_concrete_values(self) -> None:
        probe = check_artifacts._probe_source(
            check_artifacts.package_data_paths(_source())
        )
        self.assertIn("spec_from_file_location", probe)
        self.assertIn("superdex/lab/gym/registration.py", probe)
        self.assertIn("registration.register_envs()", probe)
        self.assertIn("registration.get_env_specs()", probe)
        self.assertIn('env_spec.namespace == "superdex_gym"', probe)
        self.assertIn("installed_superdex_lab_recipe_manifest", probe)
        self.assertIn("recipe_manifest.PUBLIC_RECIPE_MANIFEST", probe)
        self.assertIn("json.loads", probe)
        self.assertIn("recipe_manifest.load_recipe_manifest", probe)
        self.assertIn('recipe_manifest.load_recipes("train")', probe)
        self.assertIn("actual_train == expected_train", probe)
        self.assertIn("actual_recipe_json == expected_recipe_json", probe)
        self.assertNotIn("import superdex.lab", probe)
        self.assertNotIn("from superdex.lab.rllib", probe)
        self.assertNotRegex(probe, r"superdex_gym/[A-Za-z0-9_-]+-v[0-9]+")
        for symbol in (
            "PUBLIC_ENV_IDS",
            "PUBLIC_JSON",
            "PUBLIC_RECIPE_JSON",
            "PUBLIC_RECIPE_SLUGS",
            "_SDIST_ONLY",
            "_REQUIRED",
        ):
            self.assertFalse(hasattr(check_artifacts, symbol), symbol)
