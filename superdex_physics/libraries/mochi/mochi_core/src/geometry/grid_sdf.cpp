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

#include <mochi_core/geometry/grid_sdf.h>

#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/profile.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

namespace mochi {

// Forced inlining causes pathological MSVC compile times. Let MSVC decide.
#if MOCHI_COMPILER_MSVC
#define MOCHI_FLUSH_INLINE_LAMBDA
#else
#define MOCHI_FLUSH_INLINE_LAMBDA MOCHI_FORCE_INLINE_LAMBDA
#endif

static Real3
ComputeAxisBasedResolution(TriangularMesh const* mesh, GridSdfParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_PROFILE_SCOPE();

  // Compute axis size
  auto const& aabb = mesh->GetAabb();
  Real3 range = aabb.GetMax() - aabb.GetMin();
  real axisSize = 0_r;

  switch (params.resolutionMode) {
    case GridSdfResolutionMode::LargestAxis:
      axisSize = Max(range);
      break;

    case GridSdfResolutionMode::SmallestAxis:
      axisSize = Min(range);
      break;

    case GridSdfResolutionMode::MeanAxis:
      axisSize = Mean(range);
      break;

    default:
      MOCHI_ASSERT_VERBOSE(false, "Unreachable code");
  }

  MOCHI_ERROR_IF(
      !IsFinite(axisSize) || (axisSize <= 0_r),
      error,
      "Mesh AABB side length must be finite and positive for SDF computation.");

  // Compute delta
  return params.resolutionDelta * axisSize;
}

static Real3
ComputeEdgeBasedResolution(TriangularMesh const* mesh, GridSdfParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_PROFILE_SCOPE();

  // Compute edge size
  auto nodes = mesh->GetNodeCoordinates();
  auto edges = mesh->GetEdges();
  MOCHI_ERROR_IF(edges.empty(), error, "Empty triangle mesh cannot be used to compute an SDF.");
  MOCHI_ERROR_RETURN(error, {});
  real edgeSize = 0_r;

  switch (params.resolutionMode) {
    case GridSdfResolutionMode::LargestEdge:
      edgeSize = std::numeric_limits<real>::min();
      for (int i = 0; i < mesh->GetNumEdges(); ++i) {
        edgeSize = Max(edgeSize, Norm(nodes[edges[i][0]] - nodes[edges[i][1]]));
      }
      break;

    case GridSdfResolutionMode::SmallestEdge:
      edgeSize = std::numeric_limits<real>::max();
      for (int i = 0; i < mesh->GetNumEdges(); ++i) {
        edgeSize = Min(edgeSize, Norm(nodes[edges[i][0]] - nodes[edges[i][1]]));
      }
      break;

    case GridSdfResolutionMode::MeanEdge:
      edgeSize = 0_r;
      for (int i = 0; i < mesh->GetNumEdges(); ++i) {
        edgeSize += Norm(nodes[edges[i][0]] - nodes[edges[i][1]]);
      }
      edgeSize /= isize(edges);
      break;

    default:
      MOCHI_ASSERT_VERBOSE(false, "Unreachable code");
  }

  MOCHI_ERROR_IF(
      !IsFinite(edgeSize) || (edgeSize <= 0_r),
      error,
      "Mesh edge length must be finite and non-zero to compute an SDF.");

  // Compute delta
  return params.resolutionDelta * edgeSize;
}

GridSdf::GridSdf(
    std::shared_ptr<TriangularMesh const> const& mesh,
    GridSdfParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();
  Initialize(mesh, params, error);
}

GridSdf::GridSdf(
    std::shared_ptr<DenseGrid3D<real> const> const& grid,
    VMatrix4x4r const& actorFromGrid)
    : _distanceGrid(grid) {
  MOCHI_ASSERT(_distanceGrid != nullptr);
  MOCHI_ASSERT(
      (actorFromGrid[3] == Vec4r{0_r, 0_r, 0_r, 1_r}),
      "GridSdf has invalid transformation matrix. It may only contain scale, rotation (or mirroring), and translation.");

  // Compute scale of the first 3 basis vectors
  auto actorFromGridCols = Transpose4x4(actorFromGrid);
  auto scaleVec =
      Sqrt(Sqr(actorFromGridCols[0]) + Sqr(actorFromGridCols[1]) + Sqr(actorFromGridCols[2]));
  MOCHI_ASSERT(
      AllTrue<3>(scaleVec > std::numeric_limits<real>::epsilon()),
      "GridSdf transformation matrix has degenerate scale");

  // Store uniform scale (X, Y, and Z scale should be equal)
  _actorFromGridScale = scaleVec[0];

  // Remove scale from the matrix
  auto actorFromGridNoScale = Dot4x4(actorFromGrid, VDiagonalMatrix<4>(1_r / scaleVec));

  // Store the 3x3 grid-from-actor rotation matrix (transpose of a rotation matrix is its inverse)
  // The 4th component of each SIMD vector should be ignored, but clamp it to zero just in case.
  _gridFromActorRotation = Transpose3x3(actorFromGridNoScale);
  _gridFromActorRotation[0] = ToSimdDirection(_gridFromActorRotation[0]);
  _gridFromActorRotation[1] = ToSimdDirection(_gridFromActorRotation[1]);
  _gridFromActorRotation[2] = ToSimdDirection(_gridFromActorRotation[2]);

  // Store the full grid-from-actor transform
  _actorFromGridMatT = Transpose4x4(actorFromGrid);
  _gridFromActorMatT = InvertTransformationTransposed(_actorFromGridMatT);

  // Transform the collider bounds
  _colliderBoundsInActorSpace =
      TransformShape(actorFromGrid, _distanceGrid->GetNegativeValueBounds());
}

GridSdf GridSdf::Create(GridSdfData&& data) {
  auto grid = std::make_unique<DenseGrid3D<real>>(
      data.dims, data.bounds, data.negativeValueBounds, std::move(data.values));
  Real3 scale = data.scale.value_or(Real3{1_r, 1_r, 1_r});
  Quaternion rotation = data.rotation.value_or(Quaternion::Identity());
  Real3 translation = data.translation.value_or(Real3{});
  auto rt = TransformRT{rotation, translation};
  VMatrix4x4r actorFromGrid = Dot4x4(ToVMatrix4x4(rt), VDiagonalMatrix<4>(ToSimd(scale, 1_r)));
  return GridSdf{std::move(grid), actorFromGrid};
}

void GridSdf::Initialize(
    std::shared_ptr<TriangularMesh const> const& mesh,
    GridSdfParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  MOCHI_ERROR_IF_NOT(mesh, error, "Must provide a triangle mesh");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.boundaryPaddingDist) && (params.boundaryPaddingDist >= 0_r),
      error,
      "Boundary padding distance must be finite and non-negative.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.resolutionDelta) && (params.resolutionDelta[0] > 0_r) &&
          (params.resolutionDelta[1] > 0_r) && (params.resolutionDelta[2] > 0_r),
      error,
      "Resolution delta must be finite and positive.");
  MOCHI_ERROR_RETURN(error);

  // Determine delta
  Real3 delta{};
  switch (params.resolutionMode) {
    case GridSdfResolutionMode::LargestAxis:
    case GridSdfResolutionMode::SmallestAxis:
    case GridSdfResolutionMode::MeanAxis:
      delta = ComputeAxisBasedResolution(mesh.get(), params, error);
      break;

    case GridSdfResolutionMode::LargestEdge:
    case GridSdfResolutionMode::SmallestEdge:
    case GridSdfResolutionMode::MeanEdge:
      delta = ComputeEdgeBasedResolution(mesh.get(), params, error);
      break;

    case GridSdfResolutionMode::Explicit:
      delta = params.resolutionDelta;
      break;

    default:
      MOCHI_ERROR_SET(error, "Invalid GridSdfResolutionMode");
  }
  MOCHI_ERROR_RETURN(error);

  // Grid bounds
  Aabb colliderBounds = mesh->GetAabb();
  Aabb gridBounds = ExpandShape(colliderBounds, params.boundaryPaddingDist);

  // Make sure the grid bounds are strictly larger than the mesh abounds.
  Real3 epsilon = gridBounds.GetSize() * std::numeric_limits<real>::epsilon();
  gridBounds = Aabb{gridBounds.GetMin() - epsilon, gridBounds.GetMax() + epsilon};

  // Grid resolution
  auto gridSize = gridBounds.GetSize();
  auto cellDims64 = StaticCast<NdArray<int64_t, 3>>(Ceil(gridSize / delta));

  // Sanity check in case of excessively small delta
  for (int i = 0; i < 3; ++i) {
    MOCHI_ERROR_IF(
        cellDims64[i] >= std::numeric_limits<int>::max(),
        error,
        "Computed SDF grid resolution would be too large. Your mesh may have degenerate bounds or degenerate "
        "triangles. See GridSdfParams for resolution options.");
  }
  MOCHI_ERROR_IF(
      static_cast<double>(cellDims64[0]) * cellDims64[1] * cellDims64[2] >= 1e9,
      error,
      "Computed SDF resolution would include more than a billion cells. Your mesh may have degenerate bounds "
      "or degenerate triangles. See GridSdfParams for resolution options.");
  MOCHI_ERROR_RETURN(error);

  auto cellDims = StaticCast<Int3>(cellDims64);
  for (int i = 0; i < 3; ++i) {
    MOCHI_ASSERT_VERBOSE(
        (gridSize[i] / static_cast<real>(cellDims[i])) <= delta[i],
        "This algorithm should result in a cell size no larger than the delta computed based on GridSdfParams.");
  }

  // Every grid needs at least 1 cell (voxel), but that isn't enough for good results.
  // In fact, mochi_contact.cpp will log a warning if there are less than 6 cells per axis.
  for (int i = 0; i < 3; ++i) {
    cellDims[i] = Max(cellDims[i], Max(1, params.minGridResolution[i]));
  }

  // If there are N cells in a particular direction, then the scalar field will have (N + 1) values.
  auto gridDims = cellDims + Int3{1, 1, 1};

  // Initialize dense grid
  auto newGrid = std::make_shared<DenseGrid3D<real>>(gridDims, gridBounds, colliderBounds);

  // WARNING: This is very slow. It could be made faster if that's important enough.
  _isMeshClosed = InitializeBruteForce(mesh, *newGrid);

  // Same bounds as the grid because transform is identity
  _colliderBoundsInActorSpace = colliderBounds;

  // Store pointer-to-const
  _distanceGrid = newGrid;
}

bool GridSdf::InitializeBruteForce(
    std::shared_ptr<TriangularMesh const> const& mesh,
    DenseGrid3D<real>& outDistanceGrid) {
  MOCHI_PROFILE_SCOPE();

  Int3 const gridDims = outDistanceGrid.GetDimensions();
  std::vector<Real3> samples;

  // Generate a list of sample points
  for (int x = 0; x < gridDims[0]; ++x) {
    for (int y = 0; y < gridDims[1]; ++y) {
      for (int z = 0; z < gridDims[2]; ++z) {
        samples.push_back(outDistanceGrid.GetPointOf(Int3{x, y, z}));
      }
    }
  }

  // Use MeshCollider to find the signed distance and closest face
  MeshCollider meshCollider(mesh);
  meshCollider.Initialize();
  DynamicArray<int> indices;
  DynamicArray<Real3> contacts;
  SdfInfo sdf;
  bool isSdfGradUnitary = {};
  ContactDetectionParams params;
  params.tolerance = std::numeric_limits<real>::infinity(); // Include all points
  params.useAccelerationStructures = true;
  mochi::FindPointContactsParallel(
      samples, &meshCollider, params, TransformRT{}, indices, contacts, sdf, isSdfGradUnitary);
  MOCHI_ASSERT(contacts.size() == samples.size());

  // Fill in the grid
  int i = 0;
  for (int x = 0; x < gridDims[0]; ++x) {
    for (int y = 0; y < gridDims[1]; ++y) {
      for (int z = 0; z < gridDims[2]; ++z) {
        Int3 indexXYZ{x, y, z};
        outDistanceGrid(indexXYZ) = sdf.val[i];
        ++i;
      }
    }
  }

  return meshCollider.IsMeshClosed();
}

template <GridExtrapolation kExtrapolationType>
void GridSdf::FindPointContactsImpl(
    Span<Real3 const> points,
    TransformRT const& pointsFromActor,
    ContactDetectionParams const& params,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf) const {
  // Coordinate spaces:
  // - Point-space is the coordinate space of the input points.
  // - Actor-space is the collider's local space. pointsFromActor maps actor-space to point-space.
  // - Grid-space is the local space in which the DenseGrid3D was computed. This may be different
  //   from actor-space if the SDF was computed offline and a transform was applied on load. In
  //   that case, the GridSdf will store transformation matrices to convert between actor-space and
  //   grid-space.
  //
  // The algorithm uses these coordinate spaces in a three-stage pipeline:
  // 1. Cull against the point- and grid-space AABBs, compacting candidates into a distance queue.
  // 2. Evaluate GridSDF distances and gradients together, compacting accepted contacts into an
  // output queue.
  // 3. Transform contacts to actor-space and append them to the outputs.
  // Compaction preserves input order between stages.
  //
  // NOTE: Potential performance improvements:
  // - Completed contacts are buffered and copied instead of written directly to pre-sized outputs.
  // - Inputs smaller than kDistanceBatchSize still pay the SIMD setup and tail-handling costs.
  // - AABB culling processes one batch at a time, limiting opportunities to hide instruction
  //   latency by interleaving independent batches.
  // - Calls already culled by a BSH may benefit from skipping the point-space AABB culling.

  MOCHI_ASSERT_VERBOSE(outIndices.empty(), "Expected empty contact detection result.");
  MOCHI_ASSERT_VERBOSE(outContacts.empty(), "Expected empty contact detection result.");
  MOCHI_ASSERT_VERBOSE(outSdf.empty(), "Expected empty contact detection result.");
  int const numPoints = isize(points);
  real const toleranceInGridSpace = params.tolerance / _actorFromGridScale;
  bool hasReserved = false;

  // 4x4 matrices used to transform points between point-space and grid-space.
  // They are transposed because DotVecMat(vec, matT) is faster than DotMatVec(mat, vec).
  auto const actorFromPointsMatT = ToVMatrix4x4Transpose(Invert(pointsFromActor));
  auto const gridFromPointsMatT = Dot4x4(actorFromPointsMatT, _gridFromActorMatT);
  auto const& actorFromGridRotT = _gridFromActorRotation;

  // Transform the SDF's AABB into point-space. This gives us a quick way to reject points that are
  // outside the volume (often the majority) before we transform them into SDF-space. If the
  // requested tolerance is infinite, set these bounds to infinity to avoid any culling.
  Aabb boundsInGridSpace{-kInf3, kInf3};
  Aabb boundsInPointSpace{-kInf3, kInf3};
  if (IsFinite(toleranceInGridSpace)) {
    auto const pointsFromGridT = InvertTransformationTransposed(gridFromPointsMatT);
    boundsInGridSpace = ExpandShape(_distanceGrid->GetNegativeValueBounds(), toleranceInGridSpace);
    boundsInPointSpace = TransformShape_Transposed(pointsFromGridT, boundsInGridSpace);
  }

  constexpr int kOutputBatchSize = Simd<real>::kSize;
  constexpr int kDistanceBatchSize = Max(4, kOutputBatchSize); // BatchInt<2> is unsupported.
  constexpr int kQueueFlushThreshold = 64 * kOutputBatchSize;
  static_assert(kQueueFlushThreshold % kDistanceBatchSize == 0);

  // The queue is flushed after insertion, so StoreSelected may start at
  // kQueueFlushThreshold - 1 and write a full SIMD batch.
  constexpr int kQueueCapacity = kQueueFlushThreshold + kDistanceBatchSize - 1;

  using DistanceV = BatchReal<kDistanceBatchSize>;
  using DistanceV3 = BatchReal3<kDistanceBatchSize>;
  using DistanceIndexV = BatchInt<kDistanceBatchSize>;
  using DistanceMaskV = Simd<DenseGrid3D<real>::IType, kDistanceBatchSize>;
  using OutputV = BatchReal<kOutputBatchSize>;
  using OutputV3 = BatchReal3<kOutputBatchSize>;
  static_assert(
      DistanceV::kIsSupported && DistanceIndexV::kIsSupported && DistanceMaskV::kIsSupported);

  // Points that passed both AABB tests and await GridSDF sampling, stored in SoA layout.
  struct DistanceQueue {
    int size;
    alignas(DistanceV) real pointsInGridSpace[3][kQueueCapacity];
    alignas(DistanceIndexV) int pointIndices[kQueueCapacity];
  };

  // Contacts that passed the distance test and await actor-space transformation, stored in SoA
  // layout.
  struct OutputQueue {
    int size;
    alignas(DistanceV) real pointsInGridSpace[3][kQueueCapacity];
    alignas(DistanceV) real gradientsInGridSpace[3][kQueueCapacity];
    alignas(DistanceV) real distancesInGridSpace[kQueueCapacity];
    alignas(DistanceIndexV) int pointIndices[kQueueCapacity];
  };

  DistanceQueue distanceQueue MOCHI_NO_INIT;
  distanceQueue.size = 0;

  OutputQueue outputQueue MOCHI_NO_INIT;
  outputQueue.size = 0;

  auto flushOutputQueue = [&]() MOCHI_FLUSH_INLINE_LAMBDA {
    MOCHI_ASSERT_VERBOSE(outputQueue.size > 0, "Cannot flush an empty output queue.");
    int const flushSize = outputQueue.size;
    if (!hasReserved) {
      // Reserve memory the first time we know we have points to add.
      // Reserve the max size so we don't have to allocate again.
      outIndices.reserve(numPoints);
      int const paddedOutputSize = numPoints + kOutputBatchSize - 1;
      outContacts.reserve(paddedOutputSize);
      outSdf.reserve(paddedOutputSize);
      hasReserved = true;
    }
    int const outputOffset = isize(outIndices);
    int const newOutputSize = outputOffset + flushSize;
    int const paddedOutputSize = outputOffset + RoundUp(flushSize, kOutputBatchSize);
    outIndices.resize_noinit(newOutputSize);
    outContacts.resize_noinit(paddedOutputSize);
    outSdf.resize_noinit(paddedOutputSize);
    std::copy_n(&outputQueue.pointIndices[0], flushSize, &outIndices[outputOffset]);

    auto const actorFromGridLinearBatchT = Broadcast3x3<OutputV>(_actorFromGridMatT);
    auto const actorFromGridTranslationBatch = Broadcast3<OutputV>(_actorFromGridMatT[3]);
    auto const actorFromGridRotBatchT = Broadcast3x3<OutputV>(actorFromGridRotT);

    int i = 0;
    for (; i + kOutputBatchSize <= flushSize; i += kOutputBatchSize) {
      OutputV3 pointsInGridSpace{
          Load<OutputV>(&outputQueue.pointsInGridSpace[0][i]),
          Load<OutputV>(&outputQueue.pointsInGridSpace[1][i]),
          Load<OutputV>(&outputQueue.pointsInGridSpace[2][i])};

      OutputV3 gradientsInGridSpace{
          Load<OutputV>(&outputQueue.gradientsInGridSpace[0][i]),
          Load<OutputV>(&outputQueue.gradientsInGridSpace[1][i]),
          Load<OutputV>(&outputQueue.gradientsInGridSpace[2][i])};

      // The caller expects results in actor-space.
      auto const pointsInActorSpace =
          DotVecMat(pointsInGridSpace, actorFromGridLinearBatchT) + actorFromGridTranslationBatch;

      // Rotate gradients using the transpose of the rotation matrix on the right (the transpose is
      // the inverse in this case).
      auto const gradientsInActorSpace = DotVecMat(gradientsInGridSpace, actorFromGridRotBatchT);
      OutputV const distances =
          Load<OutputV>(&outputQueue.distancesInGridSpace[i]) * _actorFromGridScale;

      StoreTransposed(&outContacts[outputOffset + i][0], pointsInActorSpace);
      Store(&outSdf.val[outputOffset + i], distances);
      StoreTransposed(&outSdf.grad[outputOffset + i][0], gradientsInActorSpace);
    }

    // Avoid reading unspecified StoreSelected padding, then use the reserved output padding
    // for full-width stores.
    if (i < flushSize) {
      int const tailSize = flushSize - i;
      OutputV3 pointsInGridSpace{
          Load<OutputV>(&outputQueue.pointsInGridSpace[0][i], tailSize),
          Load<OutputV>(&outputQueue.pointsInGridSpace[1][i], tailSize),
          Load<OutputV>(&outputQueue.pointsInGridSpace[2][i], tailSize)};
      OutputV3 gradientsInGridSpace{
          Load<OutputV>(&outputQueue.gradientsInGridSpace[0][i], tailSize),
          Load<OutputV>(&outputQueue.gradientsInGridSpace[1][i], tailSize),
          Load<OutputV>(&outputQueue.gradientsInGridSpace[2][i], tailSize)};

      auto const pointsInActorSpace =
          DotVecMat(pointsInGridSpace, actorFromGridLinearBatchT) + actorFromGridTranslationBatch;
      auto const gradientsInActorSpace = DotVecMat(gradientsInGridSpace, actorFromGridRotBatchT);
      OutputV const distances =
          Load<OutputV>(&outputQueue.distancesInGridSpace[i], tailSize) * _actorFromGridScale;

      StoreTransposed(&outContacts[outputOffset + i][0], pointsInActorSpace);
      Store(&outSdf.val[outputOffset + i], distances);
      StoreTransposed(&outSdf.grad[outputOffset + i][0], gradientsInActorSpace);
      outContacts.resize_noinit(newOutputSize);
      outSdf.resize_noinit(newOutputSize);
    }

    outputQueue.size = 0;
  };

  auto const laneSequence = Sequence<DistanceMaskV>();
  auto const pointIndexSequence = Sequence<DistanceIndexV>();
  auto processDistanceBatch = [&](DistanceV3 const& pointsInGridSpace,
                                  DistanceIndexV pointIndices,
                                  auto validLaneMask) MOCHI_FORCE_INLINE_LAMBDA {
    DistanceV distancesInGridSpace MOCHI_NO_INIT;
    DistanceV3 gradientsInGridSpace MOCHI_NO_INIT;
    _distanceGrid->template TrilinearSampleBatch<
        kDistanceBatchSize,
        kExtrapolationType,
        /*kComputeValues*/ true,
        /*kComputeGradients*/ true>(
        pointsInGridSpace, &distancesInGridSpace, &gradientsInGridSpace);

    DistanceV hitMask = distancesInGridSpace <= toleranceInGridSpace;
    if constexpr (IsSimd<decltype(validLaneMask)>) {
      hitMask &= ReinterpretCast<DistanceV>(validLaneMask);
    }
    if (!AnyTrue(hitMask)) {
      return;
    }

    MOCHI_ASSERT_VERBOSE(outputQueue.size < kQueueFlushThreshold);
    int const scratchBegin = outputQueue.size;
    StoreSelected(&outputQueue.pointsInGridSpace[0][scratchBegin], hitMask, pointsInGridSpace[0]);
    StoreSelected(&outputQueue.pointsInGridSpace[1][scratchBegin], hitMask, pointsInGridSpace[1]);
    StoreSelected(&outputQueue.pointsInGridSpace[2][scratchBegin], hitMask, pointsInGridSpace[2]);
    StoreSelected(
        &outputQueue.gradientsInGridSpace[0][scratchBegin], hitMask, gradientsInGridSpace[0]);
    StoreSelected(
        &outputQueue.gradientsInGridSpace[1][scratchBegin], hitMask, gradientsInGridSpace[1]);
    StoreSelected(
        &outputQueue.gradientsInGridSpace[2][scratchBegin], hitMask, gradientsInGridSpace[2]);
    StoreSelected(&outputQueue.pointIndices[scratchBegin], hitMask, pointIndices);
    outputQueue.size += StoreSelected(
        &outputQueue.distancesInGridSpace[scratchBegin], hitMask, distancesInGridSpace);

    if (outputQueue.size >= kQueueFlushThreshold)
      MOCHI_UNLIKELY {
        flushOutputQueue();
      }
  };

  auto flushDistanceQueue = [&](int flushSize) MOCHI_FORCE_INLINE_LAMBDA {
    MOCHI_ASSERT_VERBOSE(
        flushSize > 0 && flushSize <= distanceQueue.size, "Invalid distance queue flush size.");
    int i = 0;
    for (; i + kDistanceBatchSize <= flushSize; i += kDistanceBatchSize) {
      DistanceV3 pointsInGridSpace{
          Load<DistanceV>(&distanceQueue.pointsInGridSpace[0][i]),
          Load<DistanceV>(&distanceQueue.pointsInGridSpace[1][i]),
          Load<DistanceV>(&distanceQueue.pointsInGridSpace[2][i])};
      processDistanceBatch(
          pointsInGridSpace, Load<DistanceIndexV>(&distanceQueue.pointIndices[i]), false);
    }

    int const tailSize = flushSize - i;
    if (tailSize > 0) {
      MOCHI_ASSERT_VERBOSE(
          flushSize == distanceQueue.size,
          "A partial distance queue cannot precede carried values.");
      for (int axis = 0; axis < 3; ++axis) {
        std::fill_n(
            &distanceQueue.pointsInGridSpace[axis][i + tailSize],
            kDistanceBatchSize - tailSize,
            distanceQueue.pointsInGridSpace[axis][i]);
      }
      DistanceV3 pointsInGridSpace{
          Load<DistanceV>(&distanceQueue.pointsInGridSpace[0][i]),
          Load<DistanceV>(&distanceQueue.pointsInGridSpace[1][i]),
          Load<DistanceV>(&distanceQueue.pointsInGridSpace[2][i])};
      auto const validLaneMask = laneSequence < tailSize;
      processDistanceBatch(
          pointsInGridSpace,
          Load<DistanceIndexV>(&distanceQueue.pointIndices[i], tailSize),
          validLaneMask);
    }

    int const carrySize = distanceQueue.size - flushSize;
    for (int axis = 0; axis < 3; ++axis) {
      std::copy_n(
          &distanceQueue.pointsInGridSpace[axis][flushSize],
          carrySize,
          &distanceQueue.pointsInGridSpace[axis][0]);
    }
    std::copy_n(&distanceQueue.pointIndices[flushSize], carrySize, &distanceQueue.pointIndices[0]);
    distanceQueue.size = carrySize;
  };

  auto const boundsInPointSpaceMin = Broadcast3<DistanceV>(boundsInPointSpace.VGetMin());
  auto const boundsInPointSpaceMax = Broadcast3<DistanceV>(boundsInPointSpace.VGetMax());
  auto const boundsInGridSpaceMin = Broadcast3<DistanceV>(boundsInGridSpace.VGetMin());
  auto const boundsInGridSpaceMax = Broadcast3<DistanceV>(boundsInGridSpace.VGetMax());
  auto const gridFromPointsLinearBatchT = Broadcast3x3<DistanceV>(gridFromPointsMatT);
  auto const gridFromPointsTranslationBatch = Broadcast3<DistanceV>(gridFromPointsMatT[3]);

  auto processPointBatch = [&](DistanceV3 const& pointsInPointSpace,
                               DistanceIndexV pointIndices,
                               auto validLaneMask) MOCHI_FORCE_INLINE_LAMBDA {
    // AABB culling in point space.
    DistanceV pointSpaceMask = (pointsInPointSpace[0] >= boundsInPointSpaceMin[0]) &
        (pointsInPointSpace[0] <= boundsInPointSpaceMax[0]) &
        (pointsInPointSpace[1] >= boundsInPointSpaceMin[1]) &
        (pointsInPointSpace[1] <= boundsInPointSpaceMax[1]) &
        (pointsInPointSpace[2] >= boundsInPointSpaceMin[2]) &
        (pointsInPointSpace[2] <= boundsInPointSpaceMax[2]);
    if constexpr (IsSimd<decltype(validLaneMask)>) {
      pointSpaceMask &= ReinterpretCast<DistanceV>(validLaneMask);
    }
    if (!AnyTrue(pointSpaceMask)) {
      return;
    }

    // AABB culling in grid space.
    DistanceV3 const pointsInGridSpace =
        DotVecMat(pointsInPointSpace, gridFromPointsLinearBatchT) + gridFromPointsTranslationBatch;
    DistanceV const gridSpaceMask = (pointsInGridSpace[0] >= boundsInGridSpaceMin[0]) &
        (pointsInGridSpace[0] <= boundsInGridSpaceMax[0]) &
        (pointsInGridSpace[1] >= boundsInGridSpaceMin[1]) &
        (pointsInGridSpace[1] <= boundsInGridSpaceMax[1]) &
        (pointsInGridSpace[2] >= boundsInGridSpaceMin[2]) &
        (pointsInGridSpace[2] <= boundsInGridSpaceMax[2]);
    DistanceV const hitMask = pointSpaceMask & gridSpaceMask;
    if (!AnyTrue(hitMask)) {
      return;
    }

    MOCHI_ASSERT_VERBOSE(distanceQueue.size < kQueueFlushThreshold);
    int const scratchBegin = distanceQueue.size;
    StoreSelected(&distanceQueue.pointsInGridSpace[0][scratchBegin], hitMask, pointsInGridSpace[0]);
    StoreSelected(&distanceQueue.pointsInGridSpace[1][scratchBegin], hitMask, pointsInGridSpace[1]);
    StoreSelected(&distanceQueue.pointsInGridSpace[2][scratchBegin], hitMask, pointsInGridSpace[2]);
    distanceQueue.size +=
        StoreSelected(&distanceQueue.pointIndices[scratchBegin], hitMask, pointIndices);

    if (distanceQueue.size >= kQueueFlushThreshold)
      MOCHI_UNLIKELY {
        flushDistanceQueue(kQueueFlushThreshold);
      }
  };

  int i = 0;
  for (; i + kDistanceBatchSize <= numPoints; i += kDistanceBatchSize) {
    DistanceV3 pointsInPointSpace MOCHI_NO_INIT;
    LoadTransposed(&points[i][0], pointsInPointSpace);
    processPointBatch(pointsInPointSpace, pointIndexSequence + i, false);
  }

  int const tailSize = numPoints - i;
  if (tailSize > 0) {
    Real3 tailPoints[kDistanceBatchSize] MOCHI_NO_INIT;
    std::copy_n(&points[i], tailSize, &tailPoints[0]);
    std::fill_n(&tailPoints[tailSize], kDistanceBatchSize - tailSize, tailPoints[tailSize - 1]);
    DistanceV3 pointsInPointSpace MOCHI_NO_INIT;
    LoadTransposed(&tailPoints[0][0], pointsInPointSpace);
    processPointBatch(pointsInPointSpace, pointIndexSequence + i, laneSequence < tailSize);
  }

  if (distanceQueue.size > 0) {
    flushDistanceQueue(distanceQueue.size);
  }

  if (outputQueue.size > 0) {
    flushOutputQueue();
  }
}

void GridSdf::FindPointContacts(
    Span<Real3 const> points,
    TransformRT const& pointsFromActor,
    ContactDetectionParams const& params,
    DynamicArray<int>& outIndices,
    DynamicArray<Real3>& outContacts,
    SdfInfo& outSdf,
    bool& outIsSdfGradUnitary) const {
  MOCHI_PROFILE_SCOPE();

  outIsSdfGradUnitary = false; // Grid SDF gradient may not be unitary.

  // If the grid bounds are at least as large as the collider bounds + tolerance, then
  // we can use the faster method of sampling interior points.
  auto colliderBounds = _distanceGrid->GetNegativeValueBounds();
  auto gridBounds = _distanceGrid->GetBounds();
  real const minGridPadding =
      Min(Min(colliderBounds.GetMin() - gridBounds.GetMin()),
          Min(gridBounds.GetMax() - colliderBounds.GetMax()));
  MOCHI_ASSERT_VERBOSE(
      IsFinite(minGridPadding) && minGridPadding >= 0_r,
      "Grid SDF negative-value bounds must be finite and contained within grid bounds.");
  real const toleranceInGridSpace = params.tolerance / _actorFromGridScale;
  if (minGridPadding >= toleranceInGridSpace) {
    FindPointContactsImpl<GridExtrapolation::Unsupported>(
        points, pointsFromActor, params, outIndices, outContacts, outSdf);
  } else {
    FindPointContactsImpl<GridExtrapolation::UpperBound>(
        points, pointsFromActor, params, outIndices, outContacts, outSdf);
  }
}

void GridSdf::GetGridSdfData(GridSdfData& outData) const {
  outData = {};
  outData.dims = _distanceGrid->GetDimensions();
  outData.values = _distanceGrid->GetData();
  outData.bounds = _distanceGrid->GetBounds();
  outData.negativeValueBounds = GetAabb(_distanceGrid->GetNegativeValueBounds());

  VMatrix4x4r actorFromGrid = Transpose4x4(_actorFromGridMatT);
  auto [scale, rt] = DecomposeMatrixTransform(actorFromGrid);
  if (!NearEqual(scale, Real3{1_r, 1_r, 1_r})) {
    outData.scale = scale;
  }
  if (!EquivalentRotation(rt.GetRotation(), Quaternion::Identity())) {
    outData.rotation = rt.GetRotation();
  }
  if (!NearEqual(rt.GetTranslation(), Real3{})) {
    outData.translation = rt.GetTranslation();
  }
}

} // namespace mochi
