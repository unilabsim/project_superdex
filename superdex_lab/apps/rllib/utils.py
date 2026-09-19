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

import logging
from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from ray import tune
from superdex.lab.gym.registration import (
    get_env_specs,
    register_envs as register_gym_envs,
)

logger = logging.getLogger(__name__)

########################################################################################


def _superdex_env_creator(
    env_spec: EnvSpec,
) -> Callable[[Mapping[str, Any]], gym.Env]:
    """Create an RLlib factory that preserves a SuperDex spec's config defaults."""

    def create(env_config: Mapping[str, Any]) -> gym.Env:
        return env_spec.make(**env_config)

    return create


def register_envs() -> None:
    """Register the public SuperDex specs with Gymnasium and expose them to Ray Tune.

    Only declared SuperDex specs receive a Tune creator. Other custom environments must
    be registered explicitly by their owners, so this helper never overwrites unrelated
    Tune registrations nor applies the single-``cfg`` creator to envs whose constructors
    take keyword arguments.
    """
    register_gym_envs()

    # Registration order matters: register_gym_envs() has populated the Gymnasium
    # registry above, so every spec from get_env_specs() is a live SuperDex spec.
    for env_spec in get_env_specs():
        tune.register_env(env_spec.id, _superdex_env_creator(env_spec))
