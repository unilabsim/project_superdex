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
        with mochi.SceneBatchExecutor(scenes, actors, num_workers=2) as executor:
            assert executor.num_scenes == 4
            assert executor.num_dofs == dofs
            assert executor.num_links == len(links[0])
            assert executor.num_contacts == 0
            executor.step(0.002, forces, qpos, qvel, link_state, contact)
            assert np.isfinite(qpos).all()
            assert np.isfinite(qvel).all()
            assert np.isfinite(link_state).all()
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
    executor = mochi.SceneBatchExecutor([scene], [actor], num_workers=1)
    with pytest.raises(RuntimeError, match="close SceneBatchExecutor"):
        mochi.destroy_scene(scene)
    mochi.shutdown()
    assert executor.closed
    mochi.initialize(num_worker_threads=0)
