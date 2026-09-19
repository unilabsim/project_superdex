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

"""Environment reconstruction from RLlib checkpoint metadata."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from superdex.lab.gym.registration import get_env_specs


def resolve_checkpoint_env_spec(env_id: str) -> EnvSpec:
    """Resolve a persisted canonical ID with migration-specific diagnostics."""
    if not isinstance(env_id, str):
        raise TypeError(
            f"Checkpoint environment ID must be a string, not {type(env_id).__name__}."
        )
    try:
        return gym.spec(env_id)
    except gym.error.Error as error:
        if "/" in env_id and not env_id.startswith("superdex_gym/"):
            # Another project's namespaced id: surface Gymnasium's own lookup error.
            raise
        available = ", ".join(spec.id for spec in get_env_specs())
        raise ValueError(
            f"Checkpoint environment ID {env_id!r} is not a registered canonical "
            f"SuperDex ID. Checkpoints must use a versioned 'superdex_gym/...' ID; legacy "
            f"discovery short names and unregistered variants are not supported. "
            f"Available IDs: {available}"
        ) from error


def prepare_checkpoint_env_config(
    env_id: str,
    env_config: Mapping[str, Any],
    *,
    render_mode: str,
) -> tuple[EnvSpec, dict[str, Any]]:
    """Validate a checkpoint ID and return its spec plus isolated rendering overrides."""
    spec = resolve_checkpoint_env_spec(env_id)
    prepared = copy.deepcopy(dict(env_config))
    prepared["render_mode"] = render_mode
    return spec, prepared


def make_checkpoint_env(
    env_id: str,
    env_config: Mapping[str, Any],
) -> gym.Env:
    """Rebuild an env while preserving registered defaults and persisted overrides."""
    spec = resolve_checkpoint_env_spec(env_id)
    return spec.make(**copy.deepcopy(dict(env_config)))
