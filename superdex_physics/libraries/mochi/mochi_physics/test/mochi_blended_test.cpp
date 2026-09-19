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

#include "mochi_physics_test_fixture.h"

#include <mochi_core/utils/constants.h>
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_blended.h>
#include <mochi_physics/src/mochi_common_components.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_soft_skinned.h>

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <utility>

using namespace mochi;

namespace {

constexpr real kFiniteDifferenceStep = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 2e-3_r;
constexpr real kFiniteDifferenceTolerance = MOCHI_USE_DOUBLE_PRECISION ? 2e-7_r : 2e-3_r;
constexpr real kCompositionTolerance = MOCHI_USE_DOUBLE_PRECISION ? 1e-12_r : 1e-6_r;
constexpr real kSoftWeight = 0.35_r;
constexpr Real3 kFreeJointLinearVelocity{1_r, 2_r, 0_r};

using CCurrentSkinnedVelocity = CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>;

class SkinnedVelocityTest : public test::MochiSceneTestBase {
 protected:
  ArticulatedActorParams MakeRevoluteSkeletonParams() {
    ShapeHandle const linkShape = test::CreateUnitCubeTetMeshShape(_mochiContext);
    ArticulatedActorParams params;
    params.worldFromRoot = TransformRT{
        Quaternion::FromRotationVector(Real3{0.2_r, -0.3_r, 0.4_r}), Real3{1_r, 2_r, 3_r}};
    params.joints = {
        {.type = ArticulatedJointType::Hard},
        {.type = ArticulatedJointType::Revolute, .axis = Real3{0_r, 0_r, 1_r}},
    };
    params.links = {
        {.parentLink = -1,
         .shape = linkShape,
         .colliderType = ColliderType::None,
         .hasGravity = false},
        {.parentLink = 0,
         .shape = linkShape,
         .colliderType = ColliderType::None,
         .hasGravity = false},
    };
    return params;
  }

  ShapeHandle CreateSkinnedUnitCube(
      bool constrainNodeZero,
      std::shared_ptr<BlendingDataMap const> blending = nullptr,
      int boneIndex = 1) {
    auto&& [coords, connectivity] = test::CreateMinimalTetMeshUnitCube();
    auto mesh = std::make_shared<TetrahedralMesh const>(coords, connectivity);
    auto skinning = std::make_shared<SkinningData const>(
        test::MakeSingleBoneSkinning(mesh->GetNumNodes(), boneIndex));

    std::shared_ptr<ConstrainedNodesData const> constrained;
    if (constrainNodeZero) {
      constrained = std::make_shared<ConstrainedNodesData const>(DynamicArray<int>{0});
    }

    auto shape =
        std::make_shared<TetrahedralMeshShape>(mesh, skinning, constrained, std::move(blending));
    return assert_cast<ContextImpl*>(_mochiContext)->RegisterShape(shape, test::ExpectOK{});
  }

  SoftSkinnedActorParams MakeFreeJointSoftSkinnedParams() {
    SoftSkinnedActorParams params;
    params.hasInertia = true;
    params.skeletonParams.joints = {{.type = ArticulatedJointType::Free}};
    params.skeletonParams.links = {
        {.parentLink = -1,
         .shape = test::CreateUnitCubeTetMeshShape(_mochiContext),
         .colliderType = ColliderType::None,
         .hasGravity = false},
    };

    SoftActorParams soft;
    soft.name = "soft";
    soft.shape = CreateSkinnedUnitCube(
        /*constrainNodeZero=*/true, /*blending=*/nullptr, /*boneIndex=*/0);
    soft.hasGravity = false;
    soft.hasInertia = false;
    soft.hasStress = false;
    params.softParams = {soft};
    return params;
  }

  static DynamicArray<real> MakeFreeJointVelocity() {
    return {
        kFreeJointLinearVelocity[0],
        kFreeJointLinearVelocity[1],
        kFreeJointLinearVelocity[2],
        0_r,
        0_r,
        0_r};
  }

  Actor* CreateMovingFreeJointSoftSkinnedActor() {
    Actor* actor =
        _scene->CreateSoftSkinnedActor(MakeFreeJointSoftSkinnedParams(), test::ExpectOK{});
    actor->SetArticulatedJointVelocities(MakeFreeJointVelocity(), test::ExpectOK{});
    return actor;
  }

  static Quaternion RotateRootByQuarterTurn(Actor* actor) {
    TransformRT root = actor->GetRootTransform();
    Quaternion const rotation = Quaternion::FromRotationVector(Real3{0_r, 0_r, 0.5_r * kPI});
    root.SetRotation(rotation);
    actor->SetRootTransform(root, test::ExpectOK{});
    return rotation;
  }

  SoftSkinnedActorParams MakeSoftSkinnedParams(bool blended = false) {
    SoftSkinnedActorParams params;
    params.skeletonParams = MakeRevoluteSkeletonParams();

    SoftActorParams soft;
    soft.name = "soft";
    soft.shape = CreateSkinnedUnitCube(/*constrainNodeZero=*/true);
    soft.hasGravity = false;
    soft.hasInertia = true;
    soft.hasStress = false;
    params.softParams = {soft};

    if (blended) {
      auto const cube = test::CreateMinimalTetMeshUnitCube();
      params.skeletonParams.skin = ArticulatedSkinParams{
          .shape = CreateSkinnedUnitCube(
              /*constrainNodeZero=*/false,
              MakeOneToOneBlendingMap(DynamicString{"soft"}, isize(cube.first))),
      };
    }
    return params;
  }

  static std::shared_ptr<BlendingDataMap const> MakeOneToOneBlendingMap(
      DynamicString const& softName,
      int numNodes) {
    BlendingDataTargetMesh target;
    target.indices.resize(2 * numNodes, 0);
    target.weights.resize(2 * numNodes, 0_r);
    for (int node = 0; node < numNodes; ++node) {
      target.indices[2 * node + 1] = node;
      target.weights[2 * node + 1] = kSoftWeight;
    }

    auto map = std::make_shared<BlendingDataMap>();
    map->perSourceShapeData.emplace(softName, std::move(target));
    return map;
  }

  static ColumnVector<real> GetFinalSkinPositionsWorld(
      entt::registry const& reg,
      entt::entity entity) {
    auto const& rest = reg.get<CArticulatedSkinningData const>(entity).restCoords;
    auto const& displacement =
        reg.get<CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned> const>(
               entity)
            .value;
    ColumnVector<real> positions(rest.Rows());
    positions = rest + displacement;
    auto const& worldFromLocal = reg.get<CRootTransform const>(entity).worldFromLocal;
    for (Real3& position : Unflatten<Real3>(positions.GetSpan())) {
      position = worldFromLocal.TransformPoint(position);
    }
    return positions;
  }

  static DynamicArray<real> PerturbPose(
      Actor const* actor,
      DynamicArray<real> const& pose,
      DynamicArray<real> const& velocity,
      real scale) {
    DynamicArray<real> delta(velocity.size());
    for (int i = 0; i < isize(velocity); ++i) {
      delta[i] = scale * velocity[i];
    }
    DynamicArray<real> perturbed(pose.size());
    actor->AddArticulatedDeltaToPose(pose, delta, perturbed, test::ExpectOK{});
    return perturbed;
  }
};

} // namespace

TEST_F(SkinnedVelocityTest, ArticulatedVelocityMatchesFiniteDifferenceAndOverwritesOutput) {
  ArticulatedActorParams params = MakeRevoluteSkeletonParams();
  params.skin = ArticulatedSkinParams{.shape = CreateSkinnedUnitCube(false)};

  Actor* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  ASSERT_EQ(1, actor->GetNumDofs());

  auto& reg = GetRegistry();
  entt::entity const entity = GetEntity(actor);
  DynamicArray<real> pose = {0.37_r};
  DynamicArray<real> velocity = {1.7_r};

  actor->SetArticulatedPoseFromJoints(
      PerturbPose(actor, pose, velocity, kFiniteDifferenceStep), test::ExpectOK{});
  ColumnVector<real> const positionsPlus = GetFinalSkinPositionsWorld(reg, entity);
  actor->SetArticulatedPoseFromJoints(
      PerturbPose(actor, pose, velocity, -kFiniteDifferenceStep), test::ExpectOK{});
  ColumnVector<real> const positionsMinus = GetFinalSkinPositionsWorld(reg, entity);
  ColumnVector<real> const expected =
      (positionsPlus - positionsMinus) * (0.5_r / kFiniteDifferenceStep);
  EXPECT_GT(expected.Norm(), 0_r);

  actor->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});
  actor->SetArticulatedJointVelocities(velocity, test::ExpectOK{});
  auto& output = reg.get<CCurrentSkinnedVelocity>(entity).value;

  output.SetConstant(123_r);
  ecs::InvokeForEach(
      &articulated::compound::UpdateSkinningVelocity, reg, MakeSingletonConstSpan(entity));
  ColumnVector<real> const firstResult = output;
  EXPECT_TRUE(test::NearEqualMatrices(expected, firstResult, kFiniteDifferenceTolerance));

  output.SetConstant(-456_r);
  ecs::InvokeForEach(
      &articulated::compound::UpdateSkinningVelocity, reg, MakeSingletonConstSpan(entity));
  EXPECT_TRUE(test::NearEqualMatrices(expected, output, kFiniteDifferenceTolerance));
  EXPECT_TRUE(test::NearEqualMatrices(firstResult, output, kCompositionTolerance));
}

TEST_F(SkinnedVelocityTest, PoseChangeUpdatesLinkVelocity) {
  Actor* parent = CreateMovingFreeJointSoftSkinnedActor();
  auto const linkHandles = parent->GetNestedLinkActors(test::ExpectOK{});
  ASSERT_EQ(1, isize(linkHandles));
  Actor* link = _scene->GetActor(linkHandles[0]);
  ASSERT_NE(nullptr, link);

  EXPECT_NEAR_EQ(kFreeJointLinearVelocity, link->GetLinearVelocity(test::ExpectOK{}));

  Quaternion const rotation = RotateRootByQuarterTurn(parent);

  EXPECT_NEAR_EQ(rotation * kFreeJointLinearVelocity, link->GetLinearVelocity(test::ExpectOK{}));
  DynamicArray<real> const jointVelocity = MakeFreeJointVelocity();
  DynamicArray<real> jointVelocityAfter(jointVelocity.size());
  parent->GetArticulatedJointVelocities(jointVelocityAfter, test::ExpectOK{});
  EXPECT_SPAN_EQ(jointVelocity, jointVelocityAfter);
}

TEST_F(SkinnedVelocityTest, PoseChangeUpdatesNestedSoftVelocity) {
  Actor* parent = CreateMovingFreeJointSoftSkinnedActor();
  auto const softHandles = parent->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(1, isize(softHandles));

  auto& reg = GetRegistry();
  entt::entity const nestedEntity = GetEntity(_scene->GetActor(softHandles[0]));
  ASSERT_TRUE((reg.all_of<CIntegrationVelocitySlices<DisplacementLayer::Skinned>>(nestedEntity)));
  Quaternion const rotation = RotateRootByQuarterTurn(parent);

  auto const velocity = Unflatten<Real3 const>(
      reg.get<CCurrentSkinnedVelocity const>(nestedEntity).value.GetConstSpan());
  Real3 const expected = rotation * kFreeJointLinearVelocity;
  for (Real3 const& nodeVelocity : velocity) {
    EXPECT_NEAR_EQ(expected, nodeVelocity);
  }
}

TEST_F(SkinnedVelocityTest, CurrentVelocityIsUniversalWhileIntegrationHistoryIsConditional) {
  SoftSkinnedActorParams params = MakeSoftSkinnedParams();
  params.hasInertia = false;
  params.softParams[0].hasInertia = true;

  Actor* parent = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  auto const softHandles = parent->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(1, isize(softHandles));

  auto& reg = GetRegistry();
  entt::entity const nestedEntity = GetEntity(_scene->GetActor(softHandles[0]));
  EXPECT_TRUE(reg.all_of<CCurrentSkinnedVelocity>(nestedEntity));
  EXPECT_FALSE((reg.all_of<CVelocitySlice<real, TimeStep::Previous, DisplacementLayer::Skinned>>(
      nestedEntity)));
  EXPECT_FALSE((reg.all_of<CVelocitySlice<real, TimeStep::StageStart, DisplacementLayer::Skinned>>(
      nestedEntity)));
  EXPECT_FALSE((reg.all_of<CIntegrationVelocitySlices<DisplacementLayer::Skinned>>(nestedEntity)));
  EXPECT_FALSE(ecs::CanInvokeOnEntity(&skinned::UpdateSkinningVelocity<true>, reg, nestedEntity));
  EXPECT_TRUE(ecs::CanInvokeOnEntity(&skinned::UpdateSkinningVelocity<false>, reg, nestedEntity));

  SoftSkinnedActorParams stateParams = MakeSoftSkinnedParams();
  stateParams.hasInertia = true;
  stateParams.softParams[0].hasInertia = false;
  Actor* stateParent = _scene->CreateSoftSkinnedActor(stateParams, test::ExpectOK{});
  auto const stateSoftHandles = stateParent->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(1, isize(stateSoftHandles));
  entt::entity const stateNestedEntity = GetEntity(_scene->GetActor(stateSoftHandles[0]));
  EXPECT_TRUE(
      (reg.all_of<CIntegrationVelocitySlices<DisplacementLayer::Skinned>>(stateNestedEntity)));
  EXPECT_TRUE(
      ecs::CanInvokeOnEntity(&skinned::UpdateSkinningVelocity<true>, reg, stateNestedEntity));
  EXPECT_FALSE(
      ecs::CanInvokeOnEntity(&skinned::UpdateSkinningVelocity<false>, reg, stateNestedEntity));

  auto& currentVelocity = reg.get<CCurrentSkinnedVelocity>(nestedEntity).value;
  currentVelocity.SetConstant(123_r);
  ColumnVector<real> const expected = currentVelocity;
  skinned::UpdateDerivedStatePipeline(reg, MakeSingletonConstSpan(nestedEntity));
  EXPECT_TRUE(test::NearEqualMatrices(expected, currentVelocity, kCompositionTolerance));
}

TEST_F(
    SkinnedVelocityTest,
    CaptureRestoreRebuildsFullVelocityAndRestoresStateBackedSkinnedVelocity) {
  SoftSkinnedActorParams params = MakeSoftSkinnedParams();
  params.hasInertia = true;
  params.softParams[0].hasInertia = false;
  Actor* parent = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});

  auto const softHandles = parent->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(1, isize(softHandles));

  auto& reg = GetRegistry();
  entt::entity const parentEntity = GetEntity(parent);
  entt::entity const nestedEntity = GetEntity(_scene->GetActor(softHandles[0]));
  auto const& links = reg.get<CGroupMembers const>(parentEntity).actors;
  ASSERT_EQ(2, isize(links));
  ASSERT_TRUE((reg.all_of<CIntegrationVelocitySlices<DisplacementLayer::Skinned>>(nestedEntity)));

  constexpr std::array kExpectedFullVelocity{
      1_r,
      2_r,
      3_r,
      4_r,
      5_r,
      6_r,
      -7_r,
      8_r,
      -9_r,
      -10_r,
      11_r,
      12_r,
  };
  ASSERT_EQ(isize(kExpectedFullVelocity), isize(links) * RigidSize::kDAll);
  auto const expectedFullVelocity = MakeConstSpan(kExpectedFullVelocity);
  for (int i = 0; i < isize(links); ++i) {
    auto const packedVelocity =
        expectedFullVelocity.subspan(i * RigidSize::kDAll, RigidSize::kDAll);
    auto& linkVelocity = reg.get<CRigidVel<TimeStep::Current>>(links[i]).value;
    linkVelocity.SetVCom(Load<RigidSize::kDTrans, Vec4r>(packedVelocity.data()));
    linkVelocity.SetOmega(
        Load<RigidSize::kDRot, Vec4r>(packedVelocity.data() + RigidSize::kDTrans));
  }

  auto& skinnedVelocity = reg.get<CCurrentSkinnedVelocity>(nestedEntity).value;
  for (int i = 0; i < skinnedVelocity.Rows(); ++i) {
    skinnedVelocity[i] = StaticCast<real>(i + 1) * 0.125_r;
  }
  ColumnVector<real> const expectedSkinnedVelocity = skinnedVelocity;
  StateHandle const state = _scene->CaptureState(test::ExpectOK{});

  reg.get<CArticulatedFullVel>(parentEntity).value.SetConstant(123_r);
  skinnedVelocity.SetConstant(456_r);
  for (entt::entity const link : links) {
    auto& linkVelocity = reg.get<CRigidVel<TimeStep::Current>>(link).value;
    linkVelocity.SetVCom({-1_r, -1_r, -1_r});
    linkVelocity.SetOmega({-1_r, -1_r, -1_r});
  }

  _scene->RestoreState(state, /*releaseImmediately=*/true, test::ExpectOK{});

  EXPECT_SPAN_EQ(
      expectedFullVelocity, reg.get<CArticulatedFullVel const>(parentEntity).value.GetConstSpan());
  EXPECT_SPAN_EQ(
      expectedSkinnedVelocity.GetConstSpan(),
      reg.get<CCurrentSkinnedVelocity const>(nestedEntity).value.GetConstSpan());
}

TEST_F(SkinnedVelocityTest, NestedSoftVelocityMatchesSimultaneousFiniteDifference) {
  Actor* parent = _scene->CreateSoftSkinnedActor(MakeSoftSkinnedParams(), test::ExpectOK{});
  ASSERT_EQ(1, parent->GetNumDofs());
  auto const softHandles = parent->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(1, isize(softHandles));

  auto& reg = GetRegistry();
  entt::entity const nestedEntity = GetEntity(_scene->GetActor(softHandles[0]));
  DynamicArray<real> pose = {-0.29_r};
  DynamicArray<real> articulatedVelocity = {1.25_r};

  auto& rawDisplacement = reg.get<CDisplacementSlice<real, TimeStep::Current>>(nestedEntity).value;
  ASSERT_GE(rawDisplacement.Rows(), 6);
  rawDisplacement[3] = 0.20_r;
  rawDisplacement[4] = 0.10_r;
  rawDisplacement[5] = -0.15_r;
  skinned::ResolveAllNodeSkinningDisplacementsPipeline(reg, MakeSingletonConstSpan(nestedEntity));
  ColumnVector<real> const displacement = rawDisplacement;

  auto& rawVelocity = reg.get<CVelocitySlice<real, TimeStep::Current>>(nestedEntity).value;
  rawVelocity.SetZero();
  rawVelocity[3] = 0.30_r;
  rawVelocity[4] = -0.40_r;
  rawVelocity[5] = 0.20_r;
  ColumnVector<real> const softVelocity = rawVelocity;

  auto const evaluatePerturbedPositions = [&](real sign) {
    parent->SetArticulatedPoseFromJoints(
        PerturbPose(parent, pose, articulatedVelocity, sign * kFiniteDifferenceStep),
        test::ExpectOK{});
    rawDisplacement = displacement + softVelocity * (sign * kFiniteDifferenceStep);
    skinned::ResolveAllNodeSkinningDisplacementsPipeline(reg, MakeSingletonConstSpan(nestedEntity));
    return GetFinalSkinPositionsWorld(reg, nestedEntity);
  };

  ColumnVector<real> const positionsPlus = evaluatePerturbedPositions(1_r);
  ColumnVector<real> const positionsMinus = evaluatePerturbedPositions(-1_r);
  ColumnVector<real> const expected =
      (positionsPlus - positionsMinus) * (0.5_r / kFiniteDifferenceStep);
  EXPECT_GT(expected.Norm(), 0_r);

  parent->SetArticulatedPoseFromJoints(pose, test::ExpectOK{});
  rawDisplacement = displacement;
  rawVelocity = softVelocity;
  skinned::ResolveAllNodeSkinningDisplacementsPipeline(reg, MakeSingletonConstSpan(nestedEntity));
  parent->SetArticulatedJointVelocities(articulatedVelocity, test::ExpectOK{});

  EXPECT_FALSE(ecs::CanInvokeOnEntity(&skinned::UpdateSkinningVelocity<true>, reg, nestedEntity));
  EXPECT_TRUE(ecs::CanInvokeOnEntity(&skinned::UpdateSkinningVelocity<false>, reg, nestedEntity));
  ecs::InvokeForEach<ecs::policy::AllowReadWriteSameComponent>(
      &skinned::UpdateSkinningVelocity<false>, reg, MakeSingletonConstSpan(nestedEntity));
  EXPECT_TRUE(
      test::NearEqualMatrices(
          expected,
          reg.get<CCurrentSkinnedVelocity const>(nestedEntity).value,
          kFiniteDifferenceTolerance));
}

TEST_F(SkinnedVelocityTest, BlendedVelocityIsExactAffineCompositionAfterSkinning) {
  Actor* parent = _scene->CreateSoftSkinnedActor(MakeSoftSkinnedParams(true), test::ExpectOK{});
  auto const softHandles = parent->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(1, isize(softHandles));
  Actor* unselectedParent =
      _scene->CreateSoftSkinnedActor(MakeSoftSkinnedParams(true), test::ExpectOK{});
  auto const unselectedSoftHandles = unselectedParent->GetNestedSoftActors(test::ExpectOK{});
  ASSERT_EQ(1, isize(unselectedSoftHandles));

  auto& reg = GetRegistry();
  entt::entity const parentEntity = GetEntity(parent);
  entt::entity const nestedEntity = GetEntity(_scene->GetActor(softHandles[0]));
  entt::entity const unselectedParentEntity = GetEntity(unselectedParent);
  entt::entity const unselectedNestedEntity = GetEntity(_scene->GetActor(unselectedSoftHandles[0]));
  parent->SetArticulatedPoseFromJoints(DynamicArray<real>{0.41_r}, test::ExpectOK{});

  auto& softVelocity = reg.get<CVelocitySlice<real, TimeStep::Current>>(nestedEntity).value;
  softVelocity.SetZero();
  ASSERT_GE(softVelocity.Rows(), 9);
  softVelocity[3] = 0.45_r;
  softVelocity[4] = -0.15_r;
  softVelocity[7] = 0.30_r;
  parent->SetArticulatedJointVelocities(DynamicArray<real>{-0.85_r}, test::ExpectOK{});

  auto& unselectedParentVelocity = reg.get<CCurrentSkinnedVelocity>(unselectedParentEntity).value;
  unselectedParentVelocity.SetConstant(123_r);
  ColumnVector<real> const expectedUnselectedParentVelocity = unselectedParentVelocity;
  auto& unselectedNestedVelocity = reg.get<CCurrentSkinnedVelocity>(unselectedNestedEntity).value;
  unselectedNestedVelocity.SetConstant(-456_r);
  ColumnVector<real> const expectedUnselectedNestedVelocity = unselectedNestedVelocity;

  ecs::InvokeForEach(
      &articulated::compound::UpdateSkinningVelocity, reg, MakeSingletonConstSpan(parentEntity));
  ColumnVector<real> const skeletonVelocity =
      reg.get<CCurrentSkinnedVelocity const>(parentEntity).value;
  EXPECT_TRUE(
      test::NearEqualMatrices(
          expectedUnselectedParentVelocity, unselectedParentVelocity, kCompositionTolerance));

  ecs::InvokeForEach<ecs::policy::AllowReadWriteSameComponent>(
      &skinned::UpdateSkinningVelocity<false>, reg, MakeSingletonConstSpan(nestedEntity));
  ColumnVector<real> const nestedVelocity =
      reg.get<CCurrentSkinnedVelocity const>(nestedEntity).value;
  EXPECT_TRUE(
      test::NearEqualMatrices(
          expectedUnselectedNestedVelocity, unselectedNestedVelocity, kCompositionTolerance));

  ColumnVector<real> expected(skeletonVelocity.Rows());
  for (int i = 0; i < expected.Rows(); ++i) {
    expected[i] = skeletonVelocity[i] + kSoftWeight * (nestedVelocity[i] - skeletonVelocity[i]);
  }

  ecs::InvokeForEach<ecs::policy::AllowReadWriteSameComponent>(
      &blended::UpdateBlendingVelocity, reg, MakeSingletonConstSpan(parentEntity));
  auto const& parentVelocity = reg.get<CCurrentSkinnedVelocity const>(parentEntity).value;
  EXPECT_TRUE(test::NearEqualMatrices(expected, parentVelocity, kCompositionTolerance));
  EXPECT_TRUE(
      test::NearEqualMatrices(
          expectedUnselectedParentVelocity, unselectedParentVelocity, kCompositionTolerance));

  ecs::InvokeForEach(
      &articulated::compound::UpdateSkinningVelocity, reg, MakeSingletonConstSpan(parentEntity));
  ecs::InvokeForEach<ecs::policy::AllowReadWriteSameComponent>(
      &skinned::UpdateSkinningVelocity<false>, reg, MakeSingletonConstSpan(nestedEntity));
  ecs::InvokeForEach<ecs::policy::AllowReadWriteSameComponent>(
      &blended::UpdateBlendingVelocity, reg, MakeSingletonConstSpan(parentEntity));
  EXPECT_TRUE(test::NearEqualMatrices(expected, parentVelocity, kCompositionTolerance));
}
