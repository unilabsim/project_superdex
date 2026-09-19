# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest

from test.conftest import mochi, np


def test_scene_batch_executor_steps_and_refreshes_each_scene():
    scenes = [mochi.create_scene(f"batch-{i}") for i in range(4)]
    actors = []
    try:
        for scene in scenes:
            shape = mochi.create_tet_mesh_shape(
                np.array(
                    [0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1], dtype=np.float32
                ),
                np.array([0, 1, 2, 3], dtype=np.int32),
            )
            actor = scene.create_articulated_actor(
                mochi.ArticulatedActorParams(
                    joints=[mochi.ArticulatedJointParams(type=mochi.ArticulatedJointType.FREE)],
                    links=[mochi.ArticulatedLinkParams(name="base", shape=shape)],
                )
            )
            mochi.release_shape(shape)
            actors.append(actor)
        dofs = actors[0].get_num_dofs()
        links = [
            [scene.get_actor(handle) for handle in actor.get_nested_link_actors()]
            for scene, actor in zip(scenes, actors, strict=True)
        ]
        forces = np.zeros((len(scenes), dofs), dtype=np.float32)
        qpos = np.empty_like(forces)
        qvel = np.empty_like(forces)
        link_state = np.empty((len(scenes), len(links[0]), 16), dtype=np.float32)
        contact = np.empty((len(scenes), 0, 3), dtype=np.float32)
        diverged = np.empty(len(scenes), dtype=np.uint8)
        with mochi.SceneBatchExecutor(
            scenes,
            actors,
            links,
            [[] for _ in scenes],
            [[] for _ in scenes],
            [],
            [],
            num_workers=2,
        ) as executor:
            assert executor.num_scenes == 4
            assert executor.num_dofs == dofs
            assert executor.num_links == len(links[0])
            assert executor.num_contacts == 0
            executor.step(0.002, forces, qpos, qvel, link_state, contact, diverged, 31)
            assert np.isfinite(qpos).all()
            assert np.isfinite(qvel).all()
            assert np.isfinite(link_state).all()
            assert not diverged.any()
            for scene in scenes:
                assert abs(scene.get_total_simulation_time() - 0.002) < 1e-8
        assert executor.closed
    finally:
        for scene in scenes:
            mochi.destroy_scene(scene)


def test_scene_batch_executor_shutdown_closes_live_workers():
    scene = mochi.create_scene("shutdown")
    shape = mochi.create_tet_mesh_shape(
        np.array([0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1], dtype=np.float32),
        np.array([0, 1, 2, 3], dtype=np.int32),
    )
    actor = scene.create_articulated_actor(
        mochi.ArticulatedActorParams(
            joints=[mochi.ArticulatedJointParams(type=mochi.ArticulatedJointType.FREE)],
            links=[mochi.ArticulatedLinkParams(name="base", shape=shape)],
        )
    )
    mochi.release_shape(shape)
    links = [[scene.get_actor(handle) for handle in actor.get_nested_link_actors()]]
    executor = mochi.SceneBatchExecutor(
        [scene],
        [actor],
        links,
        [[]],
        [[]],
        [],
        [],
        num_workers=1,
    )
    with pytest.raises(RuntimeError, match="close SceneBatchExecutor"):
        mochi.destroy_scene(scene)
    mochi.shutdown()
    assert executor.closed
    mochi.initialize(num_worker_threads=0)


def test_scene_batch_executor_selective_readback_skips_optional_buffers():
    scene = mochi.create_scene("selective-readback")
    shape = mochi.create_tet_mesh_shape(
        np.array([0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1], dtype=np.float32),
        np.array([0, 1, 2, 3], dtype=np.int32),
    )
    actor = scene.create_articulated_actor(
        mochi.ArticulatedActorParams(
            joints=[mochi.ArticulatedJointParams(type=mochi.ArticulatedJointType.FREE)],
            links=[mochi.ArticulatedLinkParams(name="base", shape=shape)],
        )
    )
    mochi.release_shape(shape)
    try:
        links = [[scene.get_actor(handle) for handle in actor.get_nested_link_actors()]]
        forces = np.zeros((1, actor.get_num_dofs()), dtype=np.float32)
        qpos = np.empty_like(forces)
        with mochi.SceneBatchExecutor([scene], [actor], links, [[]], [[]], [], [], 1) as executor:
            executor.step(0.002, forces, qpos, None, None, None, None, 1)
            assert np.isfinite(qpos).all()
            with pytest.raises(ValueError, match="readback_mask"):
                executor.step(0.002, forces, qpos, None, None, None, None, 2)
    finally:
        mochi.destroy_scene(scene)


def test_scene_batch_executor_control_step_matches_two_serial_steps():
    scenes = [mochi.create_scene(f"control-{i}") for i in range(2)]
    actors = []
    try:
        for scene in scenes:
            shape = mochi.create_tet_mesh_shape(
                np.array([0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1], dtype=np.float32),
                np.array([0, 1, 2, 3], dtype=np.int32),
            )
            actors.append(scene.create_articulated_actor(
                mochi.ArticulatedActorParams(
                    joints=[mochi.ArticulatedJointParams(type=mochi.ArticulatedJointType.FREE)],
                    links=[mochi.ArticulatedLinkParams(name="base", shape=shape)],
                )
            ))
            mochi.release_shape(shape)
        links = [[scene.get_actor(h) for h in actor.get_nested_link_actors()]
                 for scene, actor in zip(scenes, actors, strict=True)]
        dofs = actors[0].get_num_dofs()
        controls = np.zeros((2, dofs), dtype=np.float32)
        indices = np.arange(dofs, dtype=np.int32)
        zeros = np.zeros(dofs, dtype=np.float32)
        gears = np.ones(dofs, dtype=np.float32)
        ranges = np.full((dofs, 2), [-1e6, 1e6], dtype=np.float32)
        qpos = np.empty_like(controls)
        qvel = np.empty_like(controls)
        link_state = np.empty((2, len(links[0]), 16), dtype=np.float32)
        contact = np.empty((2, 0, 3), dtype=np.float32)
        diverged = np.empty(2, dtype=np.uint8)
        with mochi.SceneBatchExecutor(scenes, actors, links, [[] for _ in scenes], [[] for _ in scenes], [], [], 2) as executor:
            executor.step_control(0.002, controls, indices, indices, zeros, zeros, gears, ranges, 2,
                                  qpos, qvel, link_state, contact, diverged, 31)
            assert np.isfinite(qpos).all()
            assert not diverged.any()
    finally:
        for scene in scenes:
            mochi.destroy_scene(scene)


def _create_free_articulated_actor(scene, name, translation):
    shape = mochi.create_tet_mesh_shape(
        np.array(
            [0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1], dtype=np.float32
        ),
        np.array([0, 1, 2, 3], dtype=np.int32),
    )
    params = mochi.ArticulatedActorParams(
        joints=[mochi.ArticulatedJointParams(type=mochi.ArticulatedJointType.FREE)],
        links=[mochi.ArticulatedLinkParams(name="base", shape=shape)],
    )
    params.name = name
    params.world_from_root = mochi.TransformRT(translation=translation)
    actor = scene.create_articulated_actor(params)
    mochi.release_shape(shape)
    return actor


def _create_static_box_actor(scene):
    params = mochi.RigidActorParams()
    params.shape = mochi.create_tet_mesh_shape(
        np.array(
            [0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1], dtype=np.float32
        ),
        np.array([0, 1, 2, 3], dtype=np.int32),
    )
    params.is_static = True
    actor = scene.create_rigid_actor(params)
    mochi.release_shape(params.shape)
    return actor


def test_scene_batch_executor_v2_steps_two_actors_in_one_scene():
    scene = mochi.create_scene("v2-multi-actor")
    first = _create_free_articulated_actor(scene, "first", [-1, 0, 0])
    second = _create_free_articulated_actor(scene, "second", [1, 0, 0])
    static = _create_static_box_actor(scene)
    try:
        links = [
            [scene.get_actor(handle) for handle in actor.get_nested_link_actors()]
            for actor in (first, second)
        ]
        contact_source = links[1][0]
        contact_query = contact_source.register_query(mochi.QueryType.CONTACT_POINTS)
        assert contact_query.is_valid()
        forces = np.zeros(
            (1, first.get_num_dofs() + second.get_num_dofs()), dtype=np.float32
        )
        forces[0, first.get_num_dofs()] = 1.0
        qpos = np.empty_like(forces)
        qvel = np.empty_like(forces)
        link_state = np.empty((1, len(links[0]) + len(links[1]), 16), dtype=np.float32)
        contact = np.empty((1, 1, 3), dtype=np.float32)
        diverged = np.empty(1, dtype=np.uint8)

        with mochi.SceneBatchExecutorV2(
            [scene],
            [[first, second, static]],
            [links + [[]]],
            [first.get_num_dofs(), second.get_num_dofs(), 0],
            [[contact_source]],
            [[static]],
            [0],
            [0.1],
            num_workers=1,
        ) as executor:
            assert executor.abi_version == 2
            assert mochi.SCENE_BATCH_EXECUTOR_ABI_VERSION == 2
            assert executor.num_scenes == 1
            assert executor.num_actors == 3
            dofs = first.get_num_dofs()
            assert dofs == 6
            assert executor.num_dofs == 12
            assert executor.num_links == 2
            assert executor.num_actuators == 12
            assert executor.num_contacts == 1
            assert executor.actor_dof_counts == [dofs, dofs, 0]
            assert executor.actor_dof_offsets == [0, dofs, 2 * dofs]
            assert executor.actor_link_counts == [1, 1, 0]
            assert executor.actor_link_offsets == [0, 1, 2]
            assert executor.actor_actuator_counts == [dofs, dofs, 0]
            assert executor.actor_actuator_offsets == [0, dofs, 2 * dofs]

            controls = np.zeros((1, dofs * 2), dtype=np.float32)
            controls[0, dofs] = -1.0
            control_qpos = np.arange(dofs * 2, dtype=np.int32)
            control_qvel = np.arange(dofs * 2, dtype=np.int32)
            control_kp = np.zeros(dofs * 2, dtype=np.float32)
            control_kp[dofs:] = 1.0
            control_kd = np.zeros(dofs * 2, dtype=np.float32)
            control_gear = np.ones(dofs * 2, dtype=np.float32)
            control_ranges = np.full(
                (dofs * 2, 2), [-1e6, 1e6], dtype=np.float32
            )
            executor.step_control(
                0.002,
                controls,
                control_qpos,
                control_qvel,
                control_kp,
                control_kd,
                control_gear,
                control_ranges,
                1,
                None,
                qvel,
                None,
                None,
                None,
                2,
            )
            assert abs(scene.get_total_simulation_time() - 0.002) < 1e-8
            assert qvel[0, 0] == pytest.approx(0.0, abs=1e-6)
            assert qvel[0, dofs] < 0.0

            qvel_before_force_step = qvel.copy()
            executor.step(0.002, forces, qpos, qvel, link_state, contact, diverged, 31)
            assert np.isfinite(qpos).all()
            assert np.isfinite(qvel).all()
            assert np.isfinite(link_state).all()
            assert np.isfinite(contact).all()
            assert not diverged.any()
            assert abs(scene.get_total_simulation_time() - 0.004) < 1e-8
            assert qvel[0, 0] == pytest.approx(0.0, abs=1e-6)
            assert qvel[0, dofs] > qvel_before_force_step[0, dofs]
        assert executor.closed
        contact_source.cancel_query(contact_query)
    finally:
        mochi.destroy_scene(scene)


def test_scene_batch_executor_v2_selective_write_state_updates_only_masked_channels():
    scenes = [mochi.create_scene(f"v2-write-{i}") for i in range(2)]
    actors = [
        [
            _create_free_articulated_actor(scene, "first", [-1, 0, 0]),
            _create_free_articulated_actor(scene, "second", [1, 0, 0]),
        ]
        for scene in scenes
    ]
    try:
        links = [
            [
                [scene.get_actor(handle) for handle in actor.get_nested_link_actors()]
                for actor in scene_actors
            ]
            for scene, scene_actors in zip(scenes, actors, strict=True)
        ]
        dofs = actors[0][0].get_num_dofs()
        total_dofs = dofs * 2
        qpos = np.zeros((2, total_dofs), dtype=np.float32)
        qpos[0, 0] = 0.125
        qpos[1, dofs + 1] = 0.375
        qpos_mask = np.zeros_like(qpos, dtype=np.uint8)
        qpos_mask[0, 0] = 1
        qpos_mask[1, dofs + 1] = 1

        with mochi.SceneBatchExecutorV2(
            scenes,
            actors,
            links,
            [dofs, dofs],
            [[] for _ in scenes],
            [[] for _ in scenes],
            [],
            [],
            num_workers=2,
        ) as executor:
            executor.write_state(qpos, None, qpos_mask, None)

            actual = []
            for scene_actors in actors:
                for actor in scene_actors:
                    pose = np.empty(dofs, dtype=np.float32)
                    actor.get_articulated_pose(pose)
                    actual.append(pose.copy())
            assert actual[0][0] == pytest.approx(0.125)
            assert actual[1][0] == pytest.approx(0.0)
            assert actual[2][0] == pytest.approx(0.0)
            assert actual[3][1] == pytest.approx(0.375)
            assert actual[3][0] == pytest.approx(0.0)
    finally:
        for scene in scenes:
            mochi.destroy_scene(scene)


def test_scene_batch_executor_v2_rejects_invalid_layout_control_and_write_inputs():
    mismatch_scenes = [mochi.create_scene(f"v2-layout-{i}") for i in range(2)]
    mismatch_actors = [
        [
            _create_free_articulated_actor(mismatch_scenes[0], "first", [-1, 0, 0]),
            _create_static_box_actor(mismatch_scenes[0]),
        ],
        [
            _create_free_articulated_actor(mismatch_scenes[1], "first", [-1, 0, 0]),
            _create_free_articulated_actor(mismatch_scenes[1], "second", [1, 0, 0]),
        ],
    ]
    mismatch_links = []
    for scene, scene_actors in zip(mismatch_scenes, mismatch_actors, strict=True):
        scene_links = []
        for actor in scene_actors:
            if actor.get_num_dofs() == 0:
                scene_links.append([])
            else:
                scene_links.append(
                    [
                        scene.get_actor(handle)
                        for handle in actor.get_nested_link_actors()
                    ]
                )
        mismatch_links.append(scene_links)
    with pytest.raises(ValueError, match="same actor slot DoF/link layout"):
        mochi.SceneBatchExecutorV2(
            mismatch_scenes,
            mismatch_actors,
            mismatch_links,
            [6, 0],
            [[] for _ in mismatch_scenes],
            [[] for _ in mismatch_scenes],
            [],
            [],
            num_workers=1,
        )

    scene = mochi.create_scene("v2-validation")
    first = _create_free_articulated_actor(scene, "first", [-1, 0, 0])
    second = _create_free_articulated_actor(scene, "second", [1, 0, 0])
    try:
        links = [
            [scene.get_actor(handle) for handle in actor.get_nested_link_actors()]
            for actor in (first, second)
        ]
        dofs = first.get_num_dofs()
        with mochi.SceneBatchExecutorV2(
            [scene],
            [[first, second]],
            [links],
            [dofs, dofs],
            [[]],
            [[]],
            [],
            [],
            num_workers=1,
        ) as executor:
            controls = np.zeros((1, dofs * 2), dtype=np.float32)
            indices = np.arange(dofs * 2, dtype=np.int32)
            zeros = np.zeros(dofs * 2, dtype=np.float32)
            gears = np.ones(dofs * 2, dtype=np.float32)
            ranges = np.full((dofs * 2, 2), [-1e6, 1e6], dtype=np.float32)
            cross_actor = indices.copy()
            cross_actor[dofs] = 0
            with pytest.raises(ValueError, match="qpos/qvel indices"):
                executor.step_control(
                    0.002,
                    controls,
                    cross_actor,
                    indices,
                    zeros,
                    zeros,
                    gears,
                    ranges,
                    1,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                )

            invalid_values = np.zeros((1, dofs * 2), dtype=np.float32)
            invalid_values[0, 0] = np.nan
            invalid_mask = np.zeros((1, dofs * 2), dtype=np.uint8)
            invalid_mask[0, 0] = 1
            initial_pose = np.empty(dofs, dtype=np.float32)
            first.get_articulated_pose(initial_pose)
            with pytest.raises(ValueError, match="finite"):
                executor.write_state(
                    invalid_values, None, invalid_mask, None
                )
            unchanged_pose = np.empty(dofs, dtype=np.float32)
            first.get_articulated_pose(unchanged_pose)
            np.testing.assert_array_equal(initial_pose, unchanged_pose)

            swapped_links = [links[1], links[0]]
            with pytest.raises(ValueError, match="nested links of their actor slot"):
                mochi.SceneBatchExecutorV2(
                    [scene],
                    [[first, second]],
                    [swapped_links],
                    [dofs, dofs],
                    [[]],
                    [[]],
                    [],
                    [],
                    num_workers=1,
                )

            with pytest.raises(ValueError, match="write_mask"):
                executor.write_state(
                    np.zeros((1, dofs), dtype=np.float32),
                    None,
                    np.zeros((1, dofs), dtype=np.uint8),
                    None,
                )
            with pytest.raises(ValueError, match="readback is disabled"):
                executor.step(
                    0.002,
                    np.zeros((1, dofs * 2), dtype=np.float32),
                    np.empty((1, dofs * 2), dtype=np.float32),
                    None,
                    None,
                    None,
                    None,
                    0,
                )
            with pytest.raises(RuntimeError, match="closed"):
                executor.close()
                executor.step(
                    0.002,
                    np.zeros((1, dofs * 2), dtype=np.float32),
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                )
    finally:
        for scene in [*mismatch_scenes, scene]:
            mochi.destroy_scene(scene)


def test_scene_batch_executor_v2_native_failure_closes_executor():
    scenes = [mochi.create_scene(f"v2-native-failure-{i}") for i in range(2)]
    actors = [
        _create_free_articulated_actor(scene, "first", [0, 0, 0])
        for scene in scenes
    ]
    contact_sources = []
    contact_query = None
    executor = None
    try:
        for scene, actor in zip(scenes, actors, strict=True):
            links = [
                scene.get_actor(handle) for handle in actor.get_nested_link_actors()
            ]
            contact_sources.append(links[0])
        contact_query = contact_sources[0].register_query(
            mochi.QueryType.CONTACT_POINTS
        )
        dofs = actors[0].get_num_dofs()
        links = [
            [[scene.get_actor(handle) for handle in actor.get_nested_link_actors()]]
            for scene, actor in zip(scenes, actors, strict=True)
        ]
        forces = np.zeros((len(scenes), dofs), dtype=np.float32)
        contacts = np.empty((len(scenes), 1, 3), dtype=np.float32)

        executor = mochi.SceneBatchExecutorV2(
            scenes,
            [[actor] for actor in actors],
            links,
            [dofs],
            [[source] for source in contact_sources],
            [[None] for _ in scenes],
            [0],
            [0.1],
            num_workers=1,
        )
        with pytest.raises(Exception, match="registered the query"):
            executor.step(
                0.002,
                forces,
                None,
                None,
                None,
                contacts,
                None,
                8,
            )

        assert executor.closed
        with pytest.raises(RuntimeError, match="closed"):
            executor.step(
                0.002,
                forces,
                None,
                None,
                None,
                contacts,
                None,
                8,
            )
        for scene in scenes:
            assert abs(scene.get_total_simulation_time() - 0.002) < 1e-8
    finally:
        if executor is not None and not executor.closed:
            executor.close()
        if contact_query is not None and contact_query.is_valid():
            contact_sources[0].cancel_query(contact_query)
        for scene in scenes:
            mochi.destroy_scene(scene)


def test_scene_batch_executor_v2_lease_blocks_v1_scene_and_actor_destruction():
    scene = mochi.create_scene("v2-lease")
    actor = _create_free_articulated_actor(scene, "first", [0, 0, 0])
    static = _create_static_box_actor(scene)
    links = [[scene.get_actor(handle) for handle in actor.get_nested_link_actors()]]
    executor = None
    try:
        executor = mochi.SceneBatchExecutorV2(
            [scene],
            [[actor, static]],
            [links + [[]]],
            [actor.get_num_dofs(), 0],
            [[]],
            [[]],
            [],
            [],
            num_workers=1,
        )
        with pytest.raises(RuntimeError, match="already owned"):
            mochi.SceneBatchExecutor(
                [scene],
                [actor],
                links,
                [[]],
                [[]],
                [],
                [],
                num_workers=1,
            )
        with pytest.raises(RuntimeError, match="destroying one of its scenes"):
            mochi.destroy_scene(scene)
        with pytest.raises(RuntimeError, match="destroying one of its actors"):
            scene.destroy_actor(static)
        with pytest.raises(RuntimeError, match="destroying one of its actors"):
            scene.destroy_actor(static.get_handle())
        executor.close()
        assert executor.closed
        static_handle = static.get_handle()
        scene.destroy_actor(static_handle)
        assert scene.get_actor(static_handle) is None
        mochi.destroy_scene(scene)
    finally:
        if executor is not None and not executor.closed:
            executor.close()
        if mochi.is_valid_scene(scene):
            mochi.destroy_scene(scene)
