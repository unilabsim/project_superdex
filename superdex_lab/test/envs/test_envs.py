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

"""Instantiate-and-step smoke tests for registered public environments."""

from __future__ import annotations

import re
import time
import unittest
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from superdex.lab.gym.registration import get_env_specs

########################################################################################

_MAX_STEPS = 20
_MAX_TIME = 5.0


@dataclass(frozen=True)
class TestOnlyEnvCase:
    """An explicit crash-only configuration that must not be registered."""

    name: str
    """Stable snake_case label used as the smoke-test name."""
    base_env_id: str
    test_only_env_id: str
    cfg: Mapping[str, Any]


PUBLIC_TEST_ONLY_ENV_CASES = (
    TestOnlyEnvCase(
        name="ant_test_no_dynamics",
        base_env_id="superdex_gym/Ant-v0",
        test_only_env_id="superdex_gym/AntTestNoDynamics-v0",
        cfg={
            "use_damping": False,
            "use_gravity": False,
            "use_low_friction": True,
        },
    ),
    TestOnlyEnvCase(
        name="cart_pole_test_damped_free_pole",
        base_env_id="superdex_gym/CartPole-v0",
        test_only_env_id="superdex_gym/CartPoleTestDampedFreePole-v0",
        cfg={
            "use_damping": True,
            "use_gravity": False,
            "free_pole": True,
        },
    ),
    TestOnlyEnvCase(
        name="half_cheetah_test_no_gravity_no_springs",
        base_env_id="superdex_gym/HalfCheetah-v0",
        test_only_env_id="superdex_gym/HalfCheetahTestNoGravityNoSprings-v0",
        cfg={
            "use_gravity": False,
            "use_rest_springs": False,
        },
    ),
)


class TestEnvs(unittest.TestCase):
    """Instantiate-and-step smoke tests for public registry entries."""

    def test_registry_is_non_empty(self) -> None:
        self.assertTrue(
            get_env_specs(),
            "the SuperDex Gymnasium registry contains no environments",
        )

    def test_test_only_ids_are_not_registered(self) -> None:
        for case in PUBLIC_TEST_ONLY_ENV_CASES:
            with self.subTest(env_id=case.test_only_env_id):
                self.assertNotIn(case.test_only_env_id, gym.registry)

    def test_rejects_duplicate_registered_test_names(self) -> None:
        class DuplicateRegisteredTests(unittest.TestCase):
            pass

        specs = (EnvSpec("Example-A-v0"), EnvSpec("Example_A-v0"))
        with self.assertRaisesRegex(ValueError, "Duplicate generated smoke-test"):
            register_env_smoke_tests(DuplicateRegisteredTests, specs)
        self.assertFalse(
            hasattr(DuplicateRegisteredTests, "test_registered_example_a_v0")
        )

    def test_rejects_duplicate_explicit_test_names(self) -> None:
        class DuplicateExplicitTests(unittest.TestCase):
            pass

        case = TestOnlyEnvCase("duplicate", "Base-v0", "Test-v0", {})
        with self.assertRaisesRegex(ValueError, "Duplicate generated smoke-test"):
            register_env_smoke_tests(DuplicateExplicitTests, (), (case, case))
        self.assertFalse(hasattr(DuplicateExplicitTests, "test_config_duplicate"))

    def test_rejects_existing_registered_test_attribute(self) -> None:
        class ExistingRegisteredTest(unittest.TestCase):
            test_registered_example_v0 = None

        with self.assertRaisesRegex(ValueError, "already exists"):
            register_env_smoke_tests(
                ExistingRegisteredTest,
                (EnvSpec("Example-v0"),),
            )

    def test_rejects_existing_explicit_test_attribute(self) -> None:
        class ExistingExplicitTest(unittest.TestCase):
            test_config_example = None

        case = TestOnlyEnvCase("example", "Base-v0", "Test-v0", {})
        with self.assertRaisesRegex(ValueError, "already exists"):
            register_env_smoke_tests(ExistingExplicitTest, (), (case,))


def _run_env(env_id: str, cfg: Mapping[str, Any] | None = None) -> None:
    """Reset and step an environment a few times, bounded by wall-clock time."""
    with gym.make(env_id, **(cfg or {})) as env:
        env.reset()
        start_time = time.time()
        for _ in range(_MAX_STEPS):
            env.step(env.action_space.sample())
            if time.time() - start_time > _MAX_TIME:
                break


def _test_name(env_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", env_id.lower()).strip("_")


def _make_registered_env_test(spec: EnvSpec) -> Callable[[unittest.TestCase], None]:
    def test(self: unittest.TestCase) -> None:
        _run_env(spec.id)

    test.__doc__ = f"Instantiate and step {spec.id}."
    return test


def _make_test_only_env_test(
    case: TestOnlyEnvCase,
) -> Callable[[unittest.TestCase], None]:
    def test(self: unittest.TestCase) -> None:
        self.assertNotIn(case.test_only_env_id, gym.registry)
        _run_env(case.base_env_id, case.cfg)

    test.__doc__ = f"Instantiate and step crash-only configuration {case.name}."
    return test


def register_env_smoke_tests(
    test_case: type[unittest.TestCase],
    specs: Iterable[EnvSpec],
    test_only_cases: Iterable[TestOnlyEnvCase] = (),
) -> None:
    """Attach deterministic smoke tests for registry specs and explicit configs."""
    generated_tests = [
        (
            f"test_registered_{_test_name(spec.id)}",
            _make_registered_env_test(spec),
        )
        for spec in sorted(specs, key=lambda item: item.id)
    ]
    generated_tests.extend(
        (
            f"test_config_{case.name}",
            _make_test_only_env_test(case),
        )
        for case in sorted(test_only_cases, key=lambda item: item.name)
    )

    generated_names: set[str] = set()
    for method_name, _ in generated_tests:
        if method_name in generated_names:
            raise ValueError(
                f"Duplicate generated smoke-test method name: {method_name}"
            )
        if hasattr(test_case, method_name):
            raise ValueError(
                f"Smoke-test method already exists on {test_case.__name__}: {method_name}"
            )
        generated_names.add(method_name)

    for method_name, test in generated_tests:
        setattr(test_case, method_name, test)


register_env_smoke_tests(
    TestEnvs,
    get_env_specs(),
    PUBLIC_TEST_ONLY_ENV_CASES,
)

########################################################################################

if __name__ == "__main__":
    unittest.main()
