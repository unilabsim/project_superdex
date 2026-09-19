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
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>

using namespace mochi;

// Non-uniform box mesh: AABB min=(0,0,0), max=(1,2,3), range=(1,2,3)
// 18 unique edges: 4x1, 4x2, 4x3, 2xsqrt(5), 2xsqrt(10), 2xsqrt(13)
// Topologically closed.
static auto CreateTestMesh() {
  return std::make_shared<TriangularMesh>(test::CreateMinimalTriMeshUnitCube(Real3{1_r, 2_r, 3_r}));
}

// Base params that disable padding and min-resolution clamping so that
// the cell resolution is determined solely by the resolution mode and delta.
static GridSdfParams CreateBaseParams() {
  GridSdfParams params;
  params.boundaryPaddingDist = 0_r;
  params.minGridResolution = Int3{1, 1, 1};
  return params;
}

// Independently compute the expected cell resolution from a mesh AABB and delta,
// using the same well-known formula: Ceil(gridSize / delta), clamped to minGridResolution.
// This mirrors the public specification in grid_sdf_params.h without depending on GridSdf
// internals.
static Int3 ExpectedCellResolution(
    Aabb const& meshAabb,
    Real3 const& delta,
    real boundaryPaddingDist,
    Int3 const& minGridResolution) {
  Aabb gridBounds = ExpandShape(meshAabb, boundaryPaddingDist);
  Real3 epsilon = gridBounds.GetSize() * std::numeric_limits<real>::epsilon();
  gridBounds = Aabb{gridBounds.GetMin() - epsilon, gridBounds.GetMax() + epsilon};
  Int3 cellDims = StaticCast<Int3>(Ceil(gridBounds.GetSize() / delta));
  for (int i = 0; i < 3; ++i) {
    cellDims[i] = Max(cellDims[i], Max(1, minGridResolution[i]));
  }
  return cellDims;
}

// ---------------------------------------------------------------------------
// Axis-Based Resolution Mode Tests
// ---------------------------------------------------------------------------

TEST(GridSdf, Resolution_LargestAxis) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::LargestAxis;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  // Reference = max(1,2,3) = 3, delta = 0.5 * 3 = 1.5
  real const reference = Max(mesh->GetAabb().GetSize());
  Real3 const delta = params.resolutionDelta * reference;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

TEST(GridSdf, Resolution_SmallestAxis) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::SmallestAxis;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  // Reference = min(1,2,3) = 1, delta = 0.5 * 1 = 0.5
  real const reference = Min(mesh->GetAabb().GetSize());
  Real3 const delta = params.resolutionDelta * reference;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

TEST(GridSdf, Resolution_MeanAxis) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::MeanAxis;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  // Reference = mean(1,2,3) = 2, delta = 0.5 * 2 = 1.0
  real const reference = Mean(mesh->GetAabb().GetSize());
  Real3 const delta = params.resolutionDelta * reference;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

// ---------------------------------------------------------------------------
// Edge-Based Resolution Mode Tests
// ---------------------------------------------------------------------------

TEST(GridSdf, Resolution_LargestEdge) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::LargestEdge;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  // Independently find the largest edge length: sqrt(13) ~ 3.606
  auto nodes = mesh->GetNodeCoordinates();
  auto edges = mesh->GetEdges();
  real largestEdge = 0_r;
  for (int i = 0; i < mesh->GetNumEdges(); ++i) {
    largestEdge = Max(largestEdge, Norm(nodes[edges[i][0]] - nodes[edges[i][1]]));
  }

  Real3 const delta = params.resolutionDelta * largestEdge;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

TEST(GridSdf, Resolution_SmallestEdge) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::SmallestEdge;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  // Independently find the smallest edge length: 1.0
  auto nodes = mesh->GetNodeCoordinates();
  auto edges = mesh->GetEdges();
  real smallestEdge = std::numeric_limits<real>::max();
  for (int i = 0; i < mesh->GetNumEdges(); ++i) {
    smallestEdge = Min(smallestEdge, Norm(nodes[edges[i][0]] - nodes[edges[i][1]]));
  }

  Real3 const delta = params.resolutionDelta * smallestEdge;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

TEST(GridSdf, Resolution_MeanEdge) {
  auto mesh = CreateTestMesh();

  // Independently compute the mean edge length from mesh data
  auto nodes = mesh->GetNodeCoordinates();
  auto edges = mesh->GetEdges();
  real totalEdgeLength = 0_r;
  for (int i = 0; i < mesh->GetNumEdges(); ++i) {
    totalEdgeLength += Norm(nodes[edges[i][0]] - nodes[edges[i][1]]);
  }
  real const meanEdge = totalEdgeLength / static_cast<real>(mesh->GetNumEdges());

  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::MeanEdge;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  Real3 const delta = params.resolutionDelta * meanEdge;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

// ---------------------------------------------------------------------------
// Explicit Resolution Mode Test
// ---------------------------------------------------------------------------

TEST(GridSdf, Resolution_Explicit) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::Explicit;
  params.resolutionDelta = Real3{1.0_r, 1.0_r, 1.0_r};

  // In Explicit mode, delta = resolutionDelta directly
  Real3 const delta = params.resolutionDelta;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

// ---------------------------------------------------------------------------
// Boundary Padding Test
// ---------------------------------------------------------------------------

TEST(GridSdf, BoundaryPadding) {
  auto mesh = CreateTestMesh();

  // Baseline: zero padding
  auto baseParams = CreateBaseParams();
  baseParams.resolutionMode = GridSdfResolutionMode::MeanAxis;
  baseParams.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};
  GridSdf baseline(mesh, baseParams, test::ExpectOK{});

  // With boundary padding
  auto paddedParams = baseParams;
  paddedParams.boundaryPaddingDist = 0.5_r;
  GridSdf padded(mesh, paddedParams, test::ExpectOK{});

  real const tol = 1e-5_r;

  // Collider bounds (negative-value bounds) should approximate the mesh AABB
  auto const& colliderBounds = padded.GetDistanceGrid().GetNegativeValueBounds();
  EXPECT_NEAR_TOL(colliderBounds.GetMin(), (Real3{0_r, 0_r, 0_r}), tol);
  EXPECT_NEAR_TOL(colliderBounds.GetMax(), (Real3{1_r, 2_r, 3_r}), tol);

  // Grid bounds should be expanded by ~boundaryPaddingDist in each direction
  auto const& gridBounds = padded.GetDistanceGrid().GetBounds();
  EXPECT_NEAR_TOL(gridBounds.GetMin(), (Real3{-0.5_r, -0.5_r, -0.5_r}), tol);
  EXPECT_NEAR_TOL(gridBounds.GetMax(), (Real3{1.5_r, 2.5_r, 3.5_r}), tol);

  // Cell resolution should increase with padding vs zero-padding baseline
  Int3 const baseRes = baseline.GetCellResolution();
  Int3 const paddedRes = padded.GetCellResolution();
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(paddedRes[i], baseRes[i]);
  }
}

// Off-center geometry makes actor-from-grid rotation and reflection observable.
static constexpr Real3 kSphereCenter{0.25_r, -0.25_r, 0_r};
static constexpr real kSphereRadius = 0.25_r;
static constexpr real kInteriorTolerance = 0.125_r;
static constexpr real kExteriorTolerance = 0.75_r;
static constexpr int kNativeWidth = Simd<real>::kSize;
// Derive pattern sizes from configured SIMD types rather than a fixed lane count.
static constexpr int kPatternWidth = Max(kNativeWidth, Simd<int>::kSize);

enum class SampleKind { Hit, ZeroLevelHit, ExtrapolatedHit, DistanceMiss, SpatialMiss };

// Mix both rejection causes with retained points to verify filtering and payload association.
static constexpr auto kMixedPattern = std::array{
    SampleKind::Hit,
    SampleKind::SpatialMiss,
    SampleKind::Hit,
    SampleKind::DistanceMiss,
    SampleKind::Hit};
static constexpr auto kExtrapolatedMixedPattern = std::array{
    SampleKind::Hit,
    SampleKind::SpatialMiss,
    SampleKind::ExtrapolatedHit,
    SampleKind::DistanceMiss,
    SampleKind::Hit};

[[nodiscard]] static SampleKind GetMixedSampleKind(int index) {
  return kMixedPattern[index % isize(kMixedPattern)];
}

[[nodiscard]] static SampleKind GetExtrapolatedMixedSampleKind(int index) {
  return kExtrapolatedMixedPattern[index % isize(kExtrapolatedMixedPattern)];
}

[[nodiscard]] static bool IsHit(SampleKind kind) {
  return kind == SampleKind::Hit || kind == SampleKind::ZeroLevelHit ||
      kind == SampleKind::ExtrapolatedHit;
}

[[nodiscard]] static std::shared_ptr<DenseGrid3D<real> const> CreateSphereGrid() {
  constexpr Int3 kDimensions{9, 9, 9};
  Aabb const gridBounds{Real3{-1_r, -1_r, -1_r}, Real3{1_r, 1_r, 1_r}};
  Aabb const negativeValueBounds{kSphereCenter - kSphereRadius, kSphereCenter + kSphereRadius};
  auto grid = std::make_shared<DenseGrid3D<real>>(kDimensions, gridBounds, negativeValueBounds);
  for (int x = 0; x < kDimensions[0]; ++x) {
    for (int y = 0; y < kDimensions[1]; ++y) {
      for (int z = 0; z < kDimensions[2]; ++z) {
        Int3 const index{x, y, z};
        (*grid)(index) = Norm(grid->GetPointOf(index) - kSphereCenter) - kSphereRadius;
      }
    }
  }
  return grid;
}

[[nodiscard]] static Real3 TransformPoint(VMatrix4x4r const& transform, Real3 const& point) {
  return ToReal3(DotMatVec4x4(transform, ToSimd(point, 1_r)));
}

template <GridExtrapolation kMode>
[[nodiscard]] static bool QueryMatches(
    std::shared_ptr<DenseGrid3D<real> const> const& grid,
    VMatrix4x4r const& actorFromGrid,
    TransformRT const& pointsFromActor,
    Span<Real3 const> pointsInGridSpace,
    Span<int const> expectedIndices,
    real toleranceInGridSpace) {
  GridSdf const sdf{grid, actorFromGrid};
  DynamicArray<Real3> points(pointsInGridSpace.size());
  for (int i = 0; i < isize(points); ++i) {
    points[i] = pointsFromActor.TransformPoint(TransformPoint(actorFromGrid, pointsInGridSpace[i]));
  }

  DynamicArray<int> indices;
  DynamicArray<Real3> contacts;
  SdfInfo sdfInfo;
  bool isSdfGradUnitary = true;
  ContactDetectionParams const params{
      .tolerance = toleranceInGridSpace * sdf.GetActorFromGridScale()};
  sdf.FindPointContacts(
      MakeConstSpan(points), pointsFromActor, params, indices, contacts, sdfInfo, isSdfGradUnitary);
  if (isSdfGradUnitary || indices.size() != expectedIndices.size() ||
      contacts.size() != expectedIndices.size() || sdfInfo.size() != expectedIndices.size()) {
    return false;
  }

  real const scale = sdf.GetActorFromGridScale();
  for (int i = 0; i < isize(expectedIndices); ++i) {
    int const index = expectedIndices[i];
    if (indices[i] != index) {
      return false;
    }
    Real3 point = pointsInGridSpace[index];
    real value = {};
    Real3 gradient = {};
    grid->TrilinearSample(
        MakeSingletonSpan(point), MakeSingletonSpan(value), TrilinearSamplerOptions<kMode>{});
    grid->TrilinearSampleGradient(
        MakeSingletonSpan(point), MakeSingletonSpan(gradient), TrilinearSamplerOptions<kMode>{});
    Real3 const expectedGradient =
        ToReal3(DotMatVec3x3(actorFromGrid, ToSimd(gradient, 0_r))) / scale;
    if (!NearEqual(TransformPoint(actorFromGrid, point), contacts[i], 2e-5_r) ||
        !NearEqual(value * scale, sdfInfo.val[i], 2e-5_r) ||
        !NearEqual(expectedGradient, sdfInfo.grad[i], 2e-5_r)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] static real PatternOffset(int index, int multiplier, int modulus) {
  return 0.001_r *
      (static_cast<real>((multiplier * index) % modulus) - static_cast<real>(modulus) / 2_r);
}

[[nodiscard]] static Real3 MakePatternPoint(SampleKind kind, int index, int count, real tolerance) {
  if (kind == SampleKind::Hit) {
    // Well-separated payloads make reordered or mismatched results observable.
    return kSphereCenter +
        Real3{
            PatternOffset(index, 37, 101),
            PatternOffset(index, 53, 103),
            PatternOffset(index, 71, 107)};
  }
  if (kind == SampleKind::ZeroLevelHit) {
    Real3 point = kSphereCenter;
    point[(index / 2) % 3] += (index & 1) == 0 ? -kSphereRadius : kSphereRadius;
    return point;
  }
  if (kind == SampleKind::ExtrapolatedHit) {
    return Real3{
        1.0625_r,
        kSphereCenter[1] + PatternOffset(index, 37, 101),
        kSphereCenter[2] + PatternOffset(index, 53, 103)};
  }
  real const t = static_cast<real>(index + 1) / static_cast<real>(count + 1);
  real const extent = kSphereRadius + tolerance;
  Real3 offset{0.75_r * extent, 0.75_r * extent, 0.02_r * t};
  int const axis = index % 3;
  offset = Real3{offset[axis], offset[(axis + 1) % 3], offset[(axis + 2) % 3]};
  if ((index & 1) != 0) {
    offset[axis] = -offset[axis];
  }
  if (kind == SampleKind::SpatialMiss) {
    offset[axis] = Sign(offset[axis]) * (extent + kSphereRadius);
  }
  return kSphereCenter + offset;
}

template <GridExtrapolation kMode, typename GetSampleKind>
static void ExpectPattern(
    std::shared_ptr<DenseGrid3D<real> const> const& grid,
    int count,
    real tolerance,
    GetSampleKind const& getSampleKind,
    VMatrix4x4r const& actorFromGrid = VEye<4>(),
    TransformRT const& pointsFromActor = TransformRT{}) {
  DynamicArray<Real3> points;
  DynamicArray<int> expectedIndices;
  points.reserve(count);
  expectedIndices.reserve(count);
  for (int i = 0; i < count; ++i) {
    SampleKind const kind = getSampleKind(i);
    points.push_back(MakePatternPoint(kind, i, count, tolerance));
    if (IsHit(kind)) {
      expectedIndices.push_back(i);
    }
  }
  EXPECT_TRUE(
      QueryMatches<kMode>(
          grid,
          actorFromGrid,
          pointsFromActor,
          MakeConstSpan(points),
          MakeConstSpan(expectedIndices),
          tolerance));
}

template <GridExtrapolation kMode>
[[nodiscard]] static DynamicArray<int>
ReferenceIndices(DenseGrid3D<real> const& grid, Span<Real3 const> points, real tolerance) {
  DynamicArray<int> indices;
  for (int i = 0; i < isize(points); ++i) {
    Real3 point = points[i];
    real value = {};
    grid.TrilinearSample(
        MakeSingletonSpan(point), MakeSingletonSpan(value), TrilinearSamplerOptions<kMode>{});
    if (value <= tolerance) {
      indices.push_back(i);
    }
  }
  return indices;
}

static void TestAffineQuery() {
  constexpr Int3 kDimensions{2, 2, 2};
  Aabb const bounds{Real3{-1_r, -1_r, -1_r}, Real3{1_r, 1_r, 1_r}};
  auto grid = std::make_shared<DenseGrid3D<real>>(kDimensions, bounds, bounds);
  auto const valueAt = [](Real3 const& point) {
    return 0.25_r * point[0] + 0.5_r * point[1] - 0.75_r * point[2] - 0.5_r;
  };
  for (int x = 0; x < kDimensions[0]; ++x) {
    for (int y = 0; y < kDimensions[1]; ++y) {
      for (int z = 0; z < kDimensions[2]; ++z) {
        Int3 const index{x, y, z};
        (*grid)(index) = valueAt(grid->GetPointOf(index));
      }
    }
  }

  GridSdf const sdf{grid, VEye<4>()};
  DynamicArray<Real3> const points{Real3{-0.25_r, 0.25_r, -0.25_r}, Real3{0.25_r, -0.25_r, 0.25_r}};
  DynamicArray<int> indices;
  DynamicArray<Real3> contacts;
  SdfInfo sdfInfo;
  bool isSdfGradUnitary = true;
  sdf.FindPointContacts(
      MakeConstSpan(points),
      TransformRT{},
      ContactDetectionParams{.tolerance = 0_r},
      indices,
      contacts,
      sdfInfo,
      isSdfGradUnitary);
  ASSERT_EQ(points.size(), indices.size());
  ASSERT_EQ(points.size(), contacts.size());
  ASSERT_EQ(points.size(), sdfInfo.size());
  EXPECT_FALSE(isSdfGradUnitary);
  for (int i = 0; i < isize(points); ++i) {
    EXPECT_EQ(i, indices[i]);
    EXPECT_NEAR_EQ(points[i], contacts[i]);
    EXPECT_NEAR_EQ(valueAt(points[i]), sdfInfo.val[i]);
    EXPECT_NEAR_EQ((Real3{0.25_r, 0.5_r, -0.75_r}), sdfInfo.grad[i]);
  }
}

template <GridExtrapolation kMode>
static void TestStructuredPatterns(
    std::shared_ptr<DenseGrid3D<real> const> const& grid,
    real tolerance) {
  ExpectPattern<kMode>(grid, kPatternWidth + 1, tolerance, [](int) { return SampleKind::Hit; });
  for (SampleKind const miss : {SampleKind::SpatialMiss, SampleKind::DistanceMiss}) {
    for (int spans = 2; spans <= 3; ++spans) {
      int const count = spans * kNativeWidth;
      for (int phase = 0; phase < 2; ++phase) {
        ExpectPattern<kMode>(grid, count, tolerance, [miss, phase](int i) {
          return ((i + phase) & 1) == 0 ? SampleKind::Hit : miss;
        });
      }
      for (int split = 0; split < count; ++split) {
        ExpectPattern<kMode>(grid, count, tolerance, [miss, split](int i) {
          return i < split ? SampleKind::Hit : miss;
        });
        ExpectPattern<kMode>(grid, count, tolerance, [miss, split](int i) {
          return i >= split ? SampleKind::Hit : miss;
        });
      }
      for (int selected = 1; selected + 1 < count; ++selected) {
        ExpectPattern<kMode>(grid, count, tolerance, [miss, selected](int i) {
          return i == selected ? SampleKind::Hit : miss;
        });
        ExpectPattern<kMode>(grid, count, tolerance, [miss, selected](int i) {
          return i == selected ? miss : SampleKind::Hit;
        });
      }
    }
    // A nonzero prefix verifies every suffix length at multiple input offsets.
    for (int tail = 1; tail <= kPatternWidth; ++tail) {
      ExpectPattern<kMode>(grid, 3 * kNativeWidth + tail, tolerance, [miss](int i) {
        return (i % 3) == 1 ? miss : SampleKind::Hit;
      });
    }
  }
}

static void TestZeroLevelSetHits(std::shared_ptr<DenseGrid3D<real> const> const& grid) {
  auto const zeroLevelHit = [](int) { return SampleKind::ZeroLevelHit; };
  for (int count = 1; count <= kPatternWidth + 1; ++count) {
    ExpectPattern<GridExtrapolation::Unsupported>(grid, count, 0_r, zeroLevelHit);
  }
  ExpectPattern<GridExtrapolation::Unsupported>(grid, 1024 * kPatternWidth + 1, 0_r, zeroLevelHit);
}

template <GridExtrapolation kMode>
static void TestAllMasks(std::shared_ptr<DenseGrid3D<real> const> const& grid, real tolerance) {
  static_assert(kPatternWidth < 63);
  // Use the same exhaustive masks for spatial and distance eligibility.
  for (SampleKind const miss : {SampleKind::SpatialMiss, SampleKind::DistanceMiss}) {
    for (int live = 1; live <= kPatternWidth; ++live) {
      uint64_t const endMask = uint64_t{1} << live;
      for (uint64_t mask = 0; mask < endMask; ++mask) {
        ExpectPattern<kMode>(grid, live, tolerance, [miss, mask](int i) {
          return ((mask >> i) & 1) != 0 ? SampleKind::Hit : miss;
        });
      }
    }
  }
}

template <GridExtrapolation kMode>
static void TestLargeQueries(std::shared_ptr<DenseGrid3D<real> const> const& grid, real tolerance) {
  constexpr int kCount = 1024 * kPatternWidth + 1;
  // Large streams expose dropped, duplicated, reordered, or mismatched outputs across internal
  // processing boundaries without depending on how those boundaries are implemented.
  ExpectPattern<kMode>(grid, kCount, tolerance, [](int) { return SampleKind::Hit; });
  ExpectPattern<kMode>(grid, kCount, tolerance, GetMixedSampleKind);
  // A long rejected prefix must not hide a later contact.
  ExpectPattern<kMode>(grid, kCount, tolerance, [](int i) {
    return i == kCount - 1 ? SampleKind::Hit : SampleKind::DistanceMiss;
  });
}

// ---------------------------------------------------------------------------
// Point Contact Query Tests
// ---------------------------------------------------------------------------

TEST(GridSdf, FindPointContactsMatchesReference) {
  TestAffineQuery();
  auto const grid = CreateSphereGrid();
  DynamicArray<Real3> points;
  DynamicArray<int> expectedIndices;

  EXPECT_TRUE(
      QueryMatches<GridExtrapolation::Unsupported>(
          grid,
          VEye<4>(),
          TransformRT{},
          MakeConstSpan(points),
          MakeConstSpan(expectedIndices),
          kInteriorTolerance));

  // Cover strict, equal, and rejected samples at negative, zero, and positive tolerances.
  points = {
      kSphereCenter,
      kSphereCenter + Real3{kSphereRadius / 2_r, 0_r, 0_r},
      kSphereCenter + Real3{0.1_r, 0.1_r, 0_r},
      kSphereCenter + Real3{kSphereRadius, 0_r, 0_r},
      kSphereCenter + Real3{kSphereRadius + kInteriorTolerance, 0_r, 0_r},
      kSphereCenter + Real3{kSphereRadius + kInteriorTolerance + 0.03125_r, 0_r, 0_r}};
  expectedIndices = {0, 1};
  EXPECT_TRUE(
      QueryMatches<GridExtrapolation::Unsupported>(
          grid,
          VEye<4>(),
          TransformRT{},
          MakeConstSpan(points),
          MakeConstSpan(expectedIndices),
          -kInteriorTolerance));
  expectedIndices = {0, 1, 2, 3};
  EXPECT_TRUE(
      QueryMatches<GridExtrapolation::Unsupported>(
          grid,
          VEye<4>(),
          TransformRT{},
          MakeConstSpan(points),
          MakeConstSpan(expectedIndices),
          0_r));
  expectedIndices = {0, 1, 2, 3, 4};
  EXPECT_TRUE(
      QueryMatches<GridExtrapolation::Unsupported>(
          grid,
          VEye<4>(),
          TransformRT{},
          MakeConstSpan(points),
          MakeConstSpan(expectedIndices),
          kInteriorTolerance));

  points = {
      MakePatternPoint(SampleKind::Hit, 0, 4, kInteriorTolerance),
      MakePatternPoint(SampleKind::Hit, 0, 4, kInteriorTolerance),
      MakePatternPoint(SampleKind::DistanceMiss, 2, 4, kInteriorTolerance),
      MakePatternPoint(SampleKind::Hit, 3, 4, kInteriorTolerance),
      MakePatternPoint(SampleKind::Hit, 0, 4, kInteriorTolerance)};
  expectedIndices = {0, 1, 3, 4};
  EXPECT_TRUE(
      QueryMatches<GridExtrapolation::Unsupported>(
          grid,
          VEye<4>(),
          TransformRT{},
          MakeConstSpan(points),
          MakeConstSpan(expectedIndices),
          kInteriorTolerance));

  points.clear();
  expectedIndices.clear();
  constexpr real kBoundaryStep = 0.03125_r;
  for (int axis = 0; axis < 3; ++axis) {
    for (real sign : {-1_r, 1_r}) {
      Real3 direction{};
      direction[axis] = sign;
      points.push_back(
          kSphereCenter + direction * (kSphereRadius + kInteriorTolerance - kBoundaryStep));
      expectedIndices.push_back(isize(points) - 1);
      points.push_back(kSphereCenter + direction * (kSphereRadius + kInteriorTolerance));
      expectedIndices.push_back(isize(points) - 1);
      points.push_back(
          kSphereCenter + direction * (kSphereRadius + kInteriorTolerance + kBoundaryStep));
    }
  }
  EXPECT_TRUE(
      QueryMatches<GridExtrapolation::Unsupported>(
          grid,
          VEye<4>(),
          TransformRT{},
          MakeConstSpan(points),
          MakeConstSpan(expectedIndices),
          kInteriorTolerance));

  TestZeroLevelSetHits(grid);

  TestAllMasks<GridExtrapolation::Unsupported>(grid, kInteriorTolerance);
  TestAllMasks<GridExtrapolation::UpperBound>(grid, kExteriorTolerance);

  TestStructuredPatterns<GridExtrapolation::Unsupported>(grid, kInteriorTolerance);
  TestStructuredPatterns<GridExtrapolation::UpperBound>(grid, kExteriorTolerance);

  constexpr int kTransformCount = 3 * kPatternWidth + 1;
  Quaternion const gridRotation = Quaternion::FromAxisAngle(Real3{1_r, 2_r, -1_r}, 0.47_r);
  TransformRT const pointsFromActor{
      Quaternion::FromAxisAngle(Real3{-2_r, 1_r, 3_r}, -0.31_r), Real3{-0.7_r, 0.4_r, 0.6_r}};
  VMatrix4x4r const transformed = Dot4x4(
      ToVMatrix4x4(TransformRT{gridRotation, Real3{0.3_r, -0.6_r, 0.2_r}}),
      VDiagonalMatrix<4>(Vec4r{2_r, 2_r, 2_r, 1_r}));
  ExpectPattern<GridExtrapolation::Unsupported>(
      grid, kTransformCount, kInteriorTolerance, GetMixedSampleKind, transformed, pointsFromActor);
  VMatrix4x4r const reflected = Dot4x4(
      ToVMatrix4x4(TransformRT{gridRotation, Real3{-0.2_r, 0.5_r, -0.4_r}}),
      VDiagonalMatrix<4>(Vec4r{-1.5_r, 1.5_r, 1.5_r, 1_r}));
  ExpectPattern<GridExtrapolation::Unsupported>(
      grid, kTransformCount, kInteriorTolerance, GetMixedSampleKind, reflected, pointsFromActor);
  ExpectPattern<GridExtrapolation::UpperBound>(
      grid,
      kTransformCount,
      kExteriorTolerance,
      GetExtrapolatedMixedSampleKind,
      transformed,
      pointsFromActor);
  ExpectPattern<GridExtrapolation::UpperBound>(
      grid,
      kTransformCount,
      kExteriorTolerance,
      GetExtrapolatedMixedSampleKind,
      reflected,
      pointsFromActor);

  TestLargeQueries<GridExtrapolation::Unsupported>(grid, kInteriorTolerance);
  TestLargeQueries<GridExtrapolation::UpperBound>(grid, kExteriorTolerance);
}

TEST(GridSdf, FindPointContactsExtrapolates) {
  auto const grid = CreateSphereGrid();
  DynamicArray<Real3> boundaryPoints;
  DynamicArray<Real3> exteriorPoints;
  DynamicArray<Real3> farPoints;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) {
        if (x == 0 && y == 0 && z == 0) {
          continue;
        }
        Real3 boundary = kSphereCenter;
        Real3 exterior = kSphereCenter;
        Real3 farPoint = kSphereCenter;
        int const signs[] = {x, y, z};
        for (int axis = 0; axis < 3; ++axis) {
          if (signs[axis] != 0) {
            real const sign = static_cast<real>(signs[axis]);
            boundary[axis] = sign;
            exterior[axis] = 1.125_r * sign;
            farPoint[axis] = 5_r * sign;
          }
        }
        boundaryPoints.push_back(boundary);
        exteriorPoints.push_back(exterior);
        farPoints.push_back(farPoint);
      }
    }
  }

  // Cover all 26 face, edge, and corner directions at and beyond the grid boundary.
  // Equality with the smallest grid padding keeps every sampled point within the physical grid.
  constexpr real kGridPadding = 0.5_r;
  auto expectedIndices = ReferenceIndices<GridExtrapolation::Unsupported>(
      *grid, MakeConstSpan(boundaryPoints), kGridPadding);
  EXPECT_TRUE(
      QueryMatches<GridExtrapolation::Unsupported>(
          grid,
          VEye<4>(),
          TransformRT{},
          MakeConstSpan(boundaryPoints),
          MakeConstSpan(expectedIndices),
          kGridPadding));

  expectedIndices.resize(exteriorPoints.size());
  for (int i = 0; i < isize(expectedIndices); ++i) {
    expectedIndices[i] = i;
  }
  EXPECT_TRUE(
      QueryMatches<GridExtrapolation::UpperBound>(
          grid,
          VEye<4>(),
          TransformRT{},
          MakeConstSpan(exteriorPoints),
          MakeConstSpan(expectedIndices),
          3_r));
  EXPECT_TRUE(
      QueryMatches<GridExtrapolation::UpperBound>(
          grid,
          VEye<4>(),
          TransformRT{},
          MakeConstSpan(farPoints),
          MakeConstSpan(expectedIndices),
          std::numeric_limits<real>::infinity()));
}
