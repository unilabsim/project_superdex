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

"""Helpers for working with registered SuperDex Gymnasium environments."""

from __future__ import annotations

from typing import Any, TypeAlias

import gymnasium as gym
from superdex.lab.gym.envs.mochi_env import MochiEnv

MochiGymEnv: TypeAlias = gym.Env[Any, Any]


def unwrap_mochi_env(env: gym.Env) -> MochiEnv:
    """Return the checked Mochi base environment without taking ownership."""
    unwrapped = env.unwrapped
    if not isinstance(unwrapped, MochiEnv):
        raise TypeError(
            f"Environment must unwrap to MochiEnv, not {type(unwrapped).__name__}."
        )
    return unwrapped
