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

"""Exact canonical-ID recipe lookup for RLlib applications."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

PUBLIC_RECIPE_MANIFEST = Path(__file__).with_name("recipes") / "manifest.json"

_INVALID_WINDOWS_SLUG_CHARACTERS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
    | {f"com{index}" for index in "\u00b9\u00b2\u00b3"}
    | {f"lpt{index}" for index in "\u00b9\u00b2\u00b3"}
)


@dataclass(frozen=True)
class Recipe:
    """A recipe selected by canonical environment ID."""

    env_id: str
    slug: str
    kind: str
    config: dict[str, Any]


def _path_escape_error(manifest_path: Path, relative_path: str) -> ValueError:
    return ValueError(
        f"Recipe path {relative_path!r} in {manifest_path} must stay below its "
        "manifest directory."
    )


def _resolve_recipe_path(manifest_path: Path, relative_path: str) -> Path:
    recipe_path = PurePosixPath(relative_path)
    windows_path = PureWindowsPath(relative_path)
    if (
        recipe_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in recipe_path.parts
        or ".." in windows_path.parts
    ):
        raise _path_escape_error(manifest_path, relative_path)

    manifest_directory = manifest_path.parent.resolve()
    candidate = manifest_directory.joinpath(*recipe_path.parts).resolve()
    if not candidate.is_relative_to(manifest_directory):
        raise _path_escape_error(manifest_path, relative_path)
    return candidate


def _validate_slug(env_id: str, slug: object) -> str:
    if not isinstance(slug, str) or not slug:
        raise TypeError(f"Recipe entry {env_id!r} must define a non-empty slug.")

    posix_path = PurePosixPath(slug)
    windows_path = PureWindowsPath(slug)
    windows_basename = slug.partition(".")[0].rstrip(" .").casefold()
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or slug in {".", ".."}
        or "/" in slug
        or "\\" in slug
        or slug[-1] in {".", " "}
        or any(character in _INVALID_WINDOWS_SLUG_CHARACTERS for character in slug)
        or any(unicodedata.category(character) == "Cc" for character in slug)
        or windows_basename in _WINDOWS_RESERVED_BASENAMES
    ):
        raise ValueError(
            f"Recipe entry {env_id!r} slug {slug!r} must be one filesystem-safe "
            "path segment."
        )
    return slug


def load_recipe_manifest(
    manifest_path: Path,
    kind: str,
) -> dict[str, Recipe]:
    """Load recipes of ``kind`` keyed by exact canonical environment ID.

    Manifest trees must not be concurrently mutated while loading. Paths are resolved and
    checked before opening, which rejects existing symlink and junction escapes but cannot
    make a mutable filesystem tree race-free across platforms.
    """
    manifest_path = manifest_path.resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"Recipe manifest {manifest_path} must contain a JSON object.")

    recipes: dict[str, Recipe] = {}
    for env_id, record in manifest.items():
        if not isinstance(env_id, str) or not isinstance(record, dict):
            raise TypeError(
                f"Recipe manifest {manifest_path} must map string IDs to objects."
            )
        slug = _validate_slug(env_id, record.get("slug"))
        recipe_paths = record.get("recipes", {})
        if not isinstance(recipe_paths, dict):
            raise TypeError(f"Recipe entry {env_id!r} must define a recipes object.")
        relative_path = recipe_paths.get(kind)
        if relative_path is None:
            continue
        if not isinstance(relative_path, str):
            raise TypeError(
                f"Recipe path for {env_id!r} kind {kind!r} must be a string."
            )
        path = _resolve_recipe_path(manifest_path, relative_path)
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise TypeError(f"Recipe {path} must contain a JSON object.")
        recipes[env_id] = Recipe(env_id, slug, kind, config)
    return recipes


def load_recipes(
    kind: str,
    manifest_paths: tuple[Path, ...] = (PUBLIC_RECIPE_MANIFEST,),
) -> dict[str, Recipe]:
    """Combine disjoint explicit manifests without allowing ID or slug shadowing."""
    recipes: dict[str, Recipe] = {}
    slug_owners: dict[str, str] = {}
    for manifest_path in manifest_paths:
        for env_id, recipe in load_recipe_manifest(manifest_path, kind).items():
            if env_id in recipes:
                raise ValueError(
                    f"Duplicate {kind!r} recipe for canonical environment ID {env_id!r}."
                )
            slug_owner = slug_owners.get(recipe.slug)
            if slug_owner is not None:
                raise ValueError(
                    f"Duplicate {kind!r} recipe slug {recipe.slug!r} for canonical "
                    f"environment IDs {slug_owner!r} and {env_id!r}; slugs must be unique "
                    "so experiment output directories do not collide."
                )
            recipes[env_id] = recipe
            slug_owners[recipe.slug] = env_id
    return recipes
