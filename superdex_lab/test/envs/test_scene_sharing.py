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

"""Scene-sharing and batched-stepping tests for registered base environments."""

from __future__ import annotations

import re
import unittest
from collections.abc import Iterable
from typing import Callable

import gymnasium as gym
import superdex.physics as sdp
from gymnasium.envs.registration import EnvSpec
from numpy.testing import assert_array_equal
from superdex.lab.gym.registration import get_env_specs
from superdex.lab.gym.utils.registry import unwrap_mochi_env

########################################################################################

_NUM_STEPS_TOTAL = 100
_NUM_STEPS_BATCH = 5
_SEED = 42
_POSE_KEY = "agent_pose"


class TestSceneSharing(unittest.TestCase):
    """Scene-sharing and batched-stepping tests generated per base EnvSpec."""

    def tearDown(self) -> None:
        if sdp.is_initialized():
            sdp.shutdown()

    def test_rejects_duplicate_generated_test_names(self) -> None:
        class DuplicateSceneSharingTests(unittest.TestCase):
            pass

        specs = (EnvSpec("Example-A-v0"), EnvSpec("Example_A-v0"))
        with self.assertRaisesRegex(
            ValueError, "Duplicate generated scene-sharing test"
        ):
            register_scene_sharing_tests(DuplicateSceneSharingTests, specs)
        self.assertFalse(
            hasattr(DuplicateSceneSharingTests, "test_scene_sharing_example_a_v0")
        )

    def test_rejects_existing_generated_test_attribute(self) -> None:
        class ExistingSceneSharingTest(unittest.TestCase):
            test_scene_sharing_example_v0 = None

        with self.assertRaisesRegex(ValueError, "already exists"):
            register_scene_sharing_tests(
                ExistingSceneSharingTest,
                (EnvSpec("Example-v0"),),
            )


def _test_name(env_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", env_id.lower()).strip("_")


def _make_scene_sharing_test(
    spec: EnvSpec,
) -> Callable[[unittest.TestCase], None]:
    def test(self: unittest.TestCase) -> None:
        with (
            gym.make(spec.id, use_shared_scenes=True) as env_1,
            gym.make(spec.id, use_shared_scenes=True) as env_2,
        ):
            mochi_env_1 = unwrap_mochi_env(env_1)
            mochi_env_2 = unwrap_mochi_env(env_2)
            self.assertIsNotNone(mochi_env_1._scene_manager)
            self.assertIsNotNone(mochi_env_2._scene_manager)
            self.assertIs(mochi_env_1._scene, mochi_env_2._scene)

        with (
            gym.make(spec.id, use_shared_scenes=False) as env_1,
            gym.make(spec.id, use_shared_scenes=False) as env_2,
        ):
            mochi_env_1 = unwrap_mochi_env(env_1)
            mochi_env_2 = unwrap_mochi_env(env_2)
            self.assertIsNone(mochi_env_1._scene_manager)
            self.assertIsNone(mochi_env_2._scene_manager)
            self.assertIsNot(mochi_env_1._scene, mochi_env_2._scene)

    test.__doc__ = f"Scene sharing for {spec.id}."
    return test


def _make_batched_stepping_test(
    spec: EnvSpec,
) -> Callable[[unittest.TestCase], None]:
    def test(self: unittest.TestCase) -> None:
        common_cfg = {"num_worker_threads": 0}
        with gym.make(spec.id, **{**common_cfg, "use_shared_scenes": False}) as ref_env:
            mochi_ref_env = unwrap_mochi_env(ref_env)
            ref_env.reset(seed=_SEED)
            if _POSE_KEY not in mochi_ref_env.get_last_step().observation:
                self.skipTest(f"{spec.id} has no {_POSE_KEY!r} observation.")

            ref_env.action_space.seed(_SEED)
            ref_actions = []
            for _ in range(_NUM_STEPS_TOTAL):
                action = ref_env.action_space.sample()
                ref_actions.append(action)
                ref_env.step(action)
            ref_final_pose = mochi_ref_env.get_last_step().observation[_POSE_KEY]

        shared_cfg = {**common_cfg, "use_shared_scenes": True}
        with (
            gym.make(spec.id, **shared_cfg) as env_1,
            gym.make(spec.id, **shared_cfg) as env_2,
        ):
            mochi_env_1 = unwrap_mochi_env(env_1)
            mochi_env_2 = unwrap_mochi_env(env_2)
            env_1.reset(seed=_SEED)
            env_2.reset(seed=_SEED)
            for index in range(0, _NUM_STEPS_TOTAL, _NUM_STEPS_BATCH):
                actions = ref_actions[index : index + _NUM_STEPS_BATCH]
                for action in actions:
                    env_1.step(action)
                for action in actions:
                    env_2.step(action)

            final_pose_1 = mochi_env_1.get_last_step().observation[_POSE_KEY]
            final_pose_2 = mochi_env_2.get_last_step().observation[_POSE_KEY]
            assert_array_equal(
                final_pose_1,
                ref_final_pose,
                err_msg=f"{spec.id}: env_1 batched pose diverged from reference.",
            )
            assert_array_equal(
                final_pose_2,
                ref_final_pose,
                err_msg=f"{spec.id}: env_2 batched pose diverged from reference.",
            )
            self.assertTrue(
                mochi_env_1._scene.is_equal_state(
                    mochi_env_1._state_snapshot, mochi_env_2._state_snapshot
                ),
                msg=f"{spec.id}: env state snapshots diverged.",
            )

    test.__doc__ = f"Shared-scene batched stepping for {spec.id}."
    return test


def _make_survives_creator_close_test(
    spec: EnvSpec,
) -> Callable[[unittest.TestCase], None]:
    def test(self: unittest.TestCase) -> None:
        shared_cfg = {"use_shared_scenes": True}
        creator = gym.make(spec.id, **shared_cfg)
        sibling = gym.make(spec.id, **shared_cfg)
        try:
            mochi_creator = unwrap_mochi_env(creator)
            mochi_sibling = unwrap_mochi_env(sibling)
            self.assertIs(mochi_creator._scene, mochi_sibling._scene)
            creator.close()
            sibling.reset()
            sibling.step(sibling.action_space.sample())
        finally:
            creator.close()
            sibling.close()

    test.__doc__ = f"Shared scene survives creator close for {spec.id}."
    return test


def register_scene_sharing_tests(
    test_case: type[unittest.TestCase],
    specs: Iterable[EnvSpec],
) -> None:
    """Attach deterministic scene-sharing tests for base environment specs."""
    generated_tests = []
    for spec in sorted(specs, key=lambda item: item.id):
        if spec.kwargs.get("cfg"):
            continue
        suffix = _test_name(spec.id)
        generated_tests.extend(
            (
                (
                    f"test_scene_sharing_{suffix}",
                    _make_scene_sharing_test(spec),
                ),
                (
                    f"test_batched_stepping_{suffix}",
                    _make_batched_stepping_test(spec),
                ),
                (
                    f"test_shared_scene_survives_creator_close_{suffix}",
                    _make_survives_creator_close_test(spec),
                ),
            )
        )

    generated_names: set[str] = set()
    for method_name, _ in generated_tests:
        if method_name in generated_names:
            raise ValueError(
                f"Duplicate generated scene-sharing test method name: {method_name}"
            )
        if hasattr(test_case, method_name):
            raise ValueError(
                f"Scene-sharing test method already exists on "
                f"{test_case.__name__}: {method_name}"
            )
        generated_names.add(method_name)

    for method_name, test in generated_tests:
        setattr(test_case, method_name, test)


register_scene_sharing_tests(TestSceneSharing, get_env_specs())

########################################################################################

if __name__ == "__main__":
    unittest.main()
