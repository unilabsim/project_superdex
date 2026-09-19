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

#pragma once

/*
    A utility file for performing differentiable skinning operations. These utilities are
    used for articulated model reduction.
*/

/*
    A differentiable skinning transform takes the following form:

    y(i) = sum_{k = 1}^N w(k, i) T_{j_{i,k}}(P_{j_{i,k}}(x(i))),            eq. (1)

    where:
        x(i) is the input vector (in reference configuration),
        y(i) is the output skinned vector,
        w(k, i) are the skinning weights. Every vertex has only N nonzero
            weights for nearby bones.
        j_{i,1}, ..., j_{i,N} denote the indices of those bones for which
            vertex i has nonzero weight.
        P_j denotes the fixed rigid pre-transform into bone j's reference frame.
        T_j denotes the current rigid bone transform.

    /////////////////////////////////////////////////////////////
    //  Input derivatives
    /////////////////////////////////////////////////////////////

    If x is a function of a reduced parameter set z (i.e., x = x(z)), then the derivative of dy/dz
    has the form:

        dy(i)/dz = sum_{k = 1}^N w(k, i) R(T_{j_{i,k}}) R(P_{j_{i,k}}) dx(i)/dz

    where R(T) denotes the rotation of transform T. Computing dy/dz from dx/dz is handled with the
    DTransform function and the process is essentially the same as the Transform operation.

    /////////////////////////////////////////////////////////////
    //  Bone derivatives
    /////////////////////////////////////////////////////////////

    Suppose that theta is the vector of bone transform parameters. That is, theta is a block vector
    where each block theta_j holds the translation and Lie rotation parameters of T_j. Then the
    derivative of y with respect to theta is

        dy(i)/dtheta = sum_{k = 1}^N
            w(k, i) dT_{j_{i,k}}/dtheta (P_{j_{i,k}}(x(i)))

    In particular, this means dy/dtheta is a sparse matrix. Moreover, the (i, j) block of dy/dtheta,
    corresponding to the i-th vertex and j-th bone is zero if j is not in the set of skinning bones
    for vertex i. Therefore, we only need to compute the blocks

        dy(i)/dtheta_{j_{i,k}} =
            w(j_{i,k}, i) dT_{j_{i,k}}/dtheta_j (P_{j_{i,k}}(x(i)))

    All other blocks for vertex i will be zero.
*/

#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rigid_body_size.h>
#include <mochi_core/utils/transform_rt.h>

#include <utility>

namespace mochi {

constexpr int kDSkinningDofsPerVertex = 3;

// A differentiable skinning transform
// See notes at top of file for complete explanation.
struct DSkinningTransform {
  using VertexBones = DynamicArray<std::pair<int, real>>;

  DynamicArray<VertexBones> perVertexBones;
  // Fixed transforms from skin reference coordinates into each bone's reference frame.
  DynamicArray<TransformRT> preTransforms;
  int totalPairs = 0;

  DSkinningTransform() = default;

  /**
   * @brief Constructs a differentiable skinning transform from vertex-major skinning weights.
   *
   * @param[in] skinIndices Bone index for each skinning weight.
   * @param[in] skinWeights Skinning weights, grouped by vertex.
   * @param[in] weightsPerNode Number of consecutive weights stored for each vertex.
   * @param[in] bonePreTransforms Fixed transform into each bone's reference frame.
   *
   * @note Zero weights are dropped and repeated bone indices within a vertex are summed. Bones are
   * stored in ascending index order. Weights are not normalized.
   */
  explicit DSkinningTransform(
      Span<int const> skinIndices,
      Span<real const> skinWeights,
      int weightsPerNode,
      DynamicArray<TransformRT> bonePreTransforms);
  MOCHI_DECLARE_MOVE_ONLY(DSkinningTransform);

  // Compute the forward map
  inline void Transform(
      Span<TransformRT const> worldFromBoneTransforms,
      ColumnVectorView<real const> input,
      ColumnVectorView<real> output,
      Span<int const> activeVertices = {}) const;

  // Apply the derivative of the forward map with respect to the inputs to a packed vector.
  // The input and output may be the same view. When activeVertices is non-empty, only the
  // corresponding output entries are written; all other entries are left unchanged.
  inline void DTransform(
      Span<TransformRT const> worldFromBoneTransforms,
      ColumnVectorView<real const> input,
      ColumnVectorView<real> output,
      Span<int const> activeVertices = {}) const;

  // Compute the derivative of the forward map with respect to the inputs.
  inline void DTransform(
      Span<TransformRT const> worldFromBoneTransforms,
      RowMatrixView<real const, krylov::kDynamic, RigidSize::kDim> input,
      RowMatrixView<real, krylov::kDynamic, RigidSize::kDim> output,
      Span<int const> activeVertices = {}) const;

  // Multiply the derivative of the forward map with respect to the bone parameters by a packed
  // vector. Input blocks contain translation followed by Lie rotation parameters. When
  // activeVertices is non-empty, only the corresponding output entries are written.
  inline void DTransformDBonesTimesVector(
      Span<TransformRT const> worldFromBoneTransforms,
      ColumnVectorView<real const> unposedPositions,
      ColumnVectorView<real const> input,
      ColumnVectorView<real> output,
      Span<int const> activeVertices = {}) const;

  // Compute the derivative of the forward map with respect to the bone parameters.
  // Use CreateDBones() to create storage for the output parameter.
  inline void DTransformDBones(
      Span<TransformRT const> worldFromBoneTransforms,
      ColumnVectorView<real const, krylov::kDynamic> input,
      SparseMatrix<real>& outputDBones,
      Span<int const> activeVertices = {}) const;

  // Creates storage for DTransformDBones
  inline SparseMatrix<real> CreateDBones() const;

  TransformRT const& GetBonePreTransform(int boneId) const {
    MOCHI_ASSERT_VERBOSE(boneId >= 0 && boneId < GetBoneCount(), "Invalid bone index.");
    return preTransforms[boneId];
  }
  Span<TransformRT const> GetPreTransforms() const {
    return preTransforms;
  }

  int GetNumVertices() const {
    return isize(perVertexBones);
  }
  int GetBoneCount() const {
    return isize(preTransforms);
  }
};

} // namespace mochi

#include "dskinning_inl.h"
