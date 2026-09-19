/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/pose_stabilizer.h>

#include <array>

using namespace mochi;

TEST(PoseStabilizer, LastValid) {
  std::array<TransformRT, 1> const initialPose = {TransformRT::Identity()};
  LastValidPoseStabilizer stabilizer{0.5_r, initialPose};

  TransformRT const validPose{Real3{1_r, 2_r, 3_r}};
  std::array<TransformRT, 1> pose = {validPose};
  stabilizer.Process(1_r, pose, pose);
  EXPECT_NEAR_EQ(pose[0], validPose);

  pose[0] = TransformRT{Real3{4_r, 5_r, 6_r}};
  stabilizer.Process(0_r, pose, pose);
  EXPECT_NEAR_EQ(pose[0], validPose);
}

TEST(PoseStabilizer, LinearInterpolation) {
  std::array<TransformRT, 1> const initialPose = {TransformRT::Identity()};
  LinearInterpolationPoseStabilizer stabilizer{0.5_r, 2, initialPose};
  std::array<TransformRT, 1> const targetPose = {TransformRT{Real3{4_r, 0_r, 0_r}}};
  std::array<TransformRT, 1> outputPose;

  stabilizer.Process(0_r, targetPose, outputPose);
  EXPECT_NEAR_EQ(outputPose[0], initialPose[0]);

  stabilizer.Process(1_r, targetPose, outputPose);
  TransformRT const midpointPose{Real3{2_r, 0_r, 0_r}};
  EXPECT_NEAR_EQ(outputPose[0], midpointPose);

  stabilizer.Process(1_r, targetPose, outputPose);
  EXPECT_NEAR_EQ(outputPose[0], targetPose[0]);
}
