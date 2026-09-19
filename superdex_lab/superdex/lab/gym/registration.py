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

"""Explicit Gymnasium registration for public SuperDex environments."""

from __future__ import annotations

import copy
import dataclasses
import importlib
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from gymnasium.envs import registration as gym_registration
from gymnasium.envs.registration import EnvSpec

_NAMESPACE = "superdex_gym"
_FACTORY_ENTRY_POINT = "superdex.lab.gym.registration:make_superdex_env"

_PUBLIC_ENV_SPECS: tuple[EnvSpec, ...] = (
    EnvSpec(
        id=f"{_NAMESPACE}/Ant-v0",
        entry_point=_FACTORY_ENTRY_POINT,
        kwargs={
            "env_cls": "superdex.lab.gym.envs.benchmarks.ant_env:AntEnv",
            "cfg": {},
        },
    ),
    EnvSpec(
        id=f"{_NAMESPACE}/AntFullObservation-v0",
        entry_point=_FACTORY_ENTRY_POINT,
        kwargs={
            "env_cls": "superdex.lab.gym.envs.benchmarks.ant_env:AntEnv",
            "cfg": {"exclude_current_positions_from_observation": False},
        },
    ),
    EnvSpec(
        id=f"{_NAMESPACE}/AntNoContact-v0",
        entry_point=_FACTORY_ENTRY_POINT,
        kwargs={
            "env_cls": "superdex.lab.gym.envs.benchmarks.ant_env:AntEnv",
            "cfg": {"include_contact_in_observation": False},
        },
    ),
    EnvSpec(
        id=f"{_NAMESPACE}/AntRotationVector-v0",
        entry_point=_FACTORY_ENTRY_POINT,
        kwargs={
            "env_cls": "superdex.lab.gym.envs.benchmarks.ant_env:AntEnv",
            "cfg": {"use_rotation_vector": True},
        },
    ),
    EnvSpec(
        id=f"{_NAMESPACE}/CartPole-v0",
        entry_point=_FACTORY_ENTRY_POINT,
        kwargs={
            "env_cls": "superdex.lab.gym.envs.benchmarks.cartpole_env:CartPoleEnv",
            "cfg": {},
        },
    ),
    EnvSpec(
        id=f"{_NAMESPACE}/CartPoleActuateOnPole-v0",
        entry_point=_FACTORY_ENTRY_POINT,
        kwargs={
            "env_cls": "superdex.lab.gym.envs.benchmarks.cartpole_env:CartPoleEnv",
            "cfg": {"actuate_on_pole": True},
        },
    ),
    EnvSpec(
        id=f"{_NAMESPACE}/HalfCheetah-v0",
        entry_point=_FACTORY_ENTRY_POINT,
        kwargs={
            "env_cls": (
                "superdex.lab.gym.envs.benchmarks.halfcheetah_env:HalfCheetahEnv"
            ),
            "cfg": {},
        },
    ),
    EnvSpec(
        id=f"{_NAMESPACE}/HalfCheetahFullObservation-v0",
        entry_point=_FACTORY_ENTRY_POINT,
        kwargs={
            "env_cls": (
                "superdex.lab.gym.envs.benchmarks.halfcheetah_env:HalfCheetahEnv"
            ),
            "cfg": {"exclude_current_position_from_observation": False},
        },
    ),
)


def _load_env_type(env_cls: str) -> type[gym.Env]:
    """Import and return the concrete environment class named by ``env_cls``."""
    try:
        module_name, class_name = env_cls.split(":", maxsplit=1)
    except ValueError as e:
        raise ValueError(
            f"Invalid environment entry point {env_cls!r}: "
            "expected the form 'package.module:ClassName'."
        ) from e
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Could not load environment class from entry point {env_cls!r}."
        ) from e


def _cfg_as_dict(cfg: Any, env_cls: str) -> dict[str, Any]:
    """Return a fresh mapping from a registered ``cfg`` so overrides can be merged."""
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return copy.deepcopy(dict(cfg))
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        # Shallow field copy (not ``dataclasses.asdict``) so nested typed/dataclass values
        # are preserved for the env constructor instead of being flattened into dicts.
        return {
            field.name: copy.deepcopy(getattr(cfg, field.name))
            for field in dataclasses.fields(cfg)
        }
    raise TypeError(
        f"Configuration for environment {env_cls!r} must be a mapping or a dataclass "
        f"instance, not {type(cfg).__name__}."
    )


def make_superdex_env(
    *,
    env_cls: str,
    cfg: Any = None,
    **overrides: Any,
) -> gym.Env:
    """Load an environment class and merge Gymnasium keyword arguments into its config.

    ``cfg`` is the configuration registered on the spec; ``overrides`` are standard
    Gymnasium keyword arguments that take precedence over matching ``cfg`` fields. The
    registered ``cfg`` is always deep-copied so the spec is never mutated. When no
    overrides are given the ``cfg`` is passed through unchanged (a typed config object is
    preserved); when overrides are present the ``cfg`` is normalized to a plain mapping
    and the overrides are recursively merged into it, and the concrete environment
    rebuilds its typed config from the result.
    """
    env_type = _load_env_type(env_cls)

    if not overrides:
        # Normalize ``None`` to ``{}`` so this path matches the override path (which routes
        # through ``_cfg_as_dict``); env constructors build their default typed config from
        # an empty mapping but cannot accept ``None``.
        return env_type({} if cfg is None else copy.deepcopy(cfg))

    def _deep_merge(
        defaults: Mapping[str, Any], updates: Mapping[str, Any]
    ) -> dict[str, Any]:
        merged = copy.deepcopy(dict(defaults))
        for key, value in updates.items():
            existing = merged.get(key)
            if isinstance(existing, Mapping) and isinstance(value, Mapping):
                merged[key] = _deep_merge(existing, value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    merged_cfg = _deep_merge(_cfg_as_dict(cfg, env_cls), overrides)
    return env_type(merged_cfg)


def _different_env_spec_fields(
    existing: EnvSpec,
    expected: EnvSpec,
) -> tuple[str, ...]:
    """Return every dataclass field whose value differs between two specs."""
    return tuple(
        field.name
        for field in dataclasses.fields(EnvSpec)
        if getattr(existing, field.name) != getattr(expected, field.name)
    )


def register_env_spec(spec: EnvSpec) -> None:
    """Register ``spec``, accepting an existing entry only when fully equivalent."""
    expected = copy.deepcopy(spec)
    existing = gym.registry.get(expected.id)
    if existing is not None:
        different_fields = _different_env_spec_fields(existing, expected)
        if not different_fields:
            return
        differences = ", ".join(
            f"{field_name}: existing={getattr(existing, field_name)!r}, "
            f"expected={getattr(expected, field_name)!r}"
            for field_name in different_fields
        )
        raise ValueError(
            f"Cannot register environment {expected.id!r}: an incompatible EnvSpec "
            f"is already registered ({differences})."
        )

    # These are canonical IDs, so plugin namespace context must not rewrite them.
    # Gymnasium 1.2.0's namespace context does not restore state when registration raises.
    previous_namespace = gym_registration.current_namespace
    try:
        with gym_registration.namespace(None):
            gym.register(
                id=expected.id,
                entry_point=expected.entry_point,
                reward_threshold=expected.reward_threshold,
                nondeterministic=expected.nondeterministic,
                max_episode_steps=expected.max_episode_steps,
                order_enforce=expected.order_enforce,
                disable_env_checker=expected.disable_env_checker,
                additional_wrappers=expected.additional_wrappers,
                vector_entry_point=expected.vector_entry_point,
                kwargs=expected.kwargs,
            )
    finally:
        gym_registration.current_namespace = previous_namespace


def register_envs() -> None:
    """Register the public SuperDex environments with strict idempotency."""
    for spec in _PUBLIC_ENV_SPECS:
        register_env_spec(spec)


def get_env_specs() -> tuple[EnvSpec, ...]:
    """Return all registered SuperDex specs in deterministic (name, numeric version) order."""
    return tuple(
        sorted(
            (spec for spec in gym.registry.values() if spec.namespace == _NAMESPACE),
            key=lambda spec: (spec.name, -1 if spec.version is None else spec.version),
        )
    )
