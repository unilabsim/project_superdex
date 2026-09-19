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

#include <mochi_core/ai/mlp.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <cmath>
#include <type_traits>
#include <vector>

using namespace mochi;

// TODO(T186384485): Extend test coverage to cover all specializations.

constexpr real kTolerance = std::is_same_v<real, float> ? 1e-3 : 1e-10;

template <typename ActivationType>
void ActivationTestImpl(
    ActivationType f,
    std::vector<real> const& z,
    std::vector<real> const& gold_z,
    std::vector<real> const& gold_dfdz) {
  MOCHI_ASSERT_VERBOSE((z.size() == gold_z.size()) && (z.size() == gold_dfdz.size()));
  for (int i = 0; i < z.size(); ++i) {
    {
      auto curr_z = z[i];
      f(curr_z); // modifies in place
      EXPECT_NEAR(curr_z, gold_z[i], kTolerance);
    }
    {
      auto curr_z = z[i];
      real dfdz = 0_r;
      f(curr_z, dfdz); // modifies in place
      EXPECT_NEAR(curr_z, gold_z[i], kTolerance);
      EXPECT_NEAR(dfdz, gold_dfdz[i], kTolerance);
    }
  }
}

template <
    int kRowsAtCompile,
    int kColsAtCompile,
    krylov::Direction kMajorDir,
    int kLeadDim = krylov::kAutomaticLeadDim,
    typename ActivationFunctor>
void ActivationSIMDApply(ActivationFunctor f) {
  static_assert((kRowsAtCompile > 0) && (kColsAtCompile > 0) && (kLeadDim >= 0));
  constexpr int kC = (kLeadDim > 0) ? krylov::kDynamic : kColsAtCompile;
  constexpr int kR = (kLeadDim > 0) ? krylov::kDynamic : kRowsAtCompile;
  constexpr int kLD = (kLeadDim > 0) ? krylov::kDynamic : kLeadDim;
  using MatType = Matrix<real, kR, kC, kMajorDir, krylov::Ownership::Owner, kLD>;
  int newLD = kLeadDim;
  if (kLeadDim == krylov::kAutomaticLeadDim) {
    newLD = (kMajorDir == krylov::Direction::ColMajor) ? kRowsAtCompile : kColsAtCompile;
  }
  MatType w(kRowsAtCompile, kColsAtCompile, newLD);
  w.SetRandom(123, -2.3_r, 2.0_r);
  MatType w0(w), gold_w(w);
  for (int i = 0; i < gold_w.Rows(); ++i) {
    for (int j = 0; j < gold_w.Cols(); ++j) {
      f(gold_w(i, j));
    }
  }
  ai::details::ActivationInPlace(f, w);
  EXPECT_TRUE(test::NearEqualMatrices(w, gold_w));
  //
  w = w0;
  MatType dfdw(w0), gold_dfdw(w0);
  gold_w = w0;
  for (int i = 0; i < w0.Rows(); ++i) {
    for (int j = 0; j < w0.Cols(); ++j) {
      f(gold_w(i, j), gold_dfdw(i, j));
    }
  }
  ai::details::ActivationInPlace(f, w, dfdw);
  EXPECT_TRUE(test::NearEqualMatrices(w, gold_w));
  EXPECT_TRUE(test::NearEqualMatrices(dfdw, gold_dfdw));
}

TEST(MlpActivation, Identity) {
  ai::IdentityActivation<real> f;
  std::vector<real> z{-1.2_r, 0_r, 2_r};
  std::vector<real> gold_z{-1.2_r, 0_r, 2_r};
  std::vector<real> gold_dfdz{1_r, 1_r, 1_r};
  ActivationTestImpl(f, z, gold_z, gold_dfdz);
  //
  ActivationSIMDApply<19, 3, krylov::Direction::ColMajor>(f);
  ActivationSIMDApply<7, 17, krylov::Direction::RowMajor>(f);
  ActivationSIMDApply<19, 3, krylov::Direction::ColMajor, 32>(f);
  ActivationSIMDApply<7, 17, krylov::Direction::RowMajor, 32>(f);
}

TEST(MlpActivation, ELU) {
  for (real alpha : {-0.5_r, 0_r, 2_r}) {
    ai::ELUActivation<real> f{alpha};
    std::vector<real> z{-1.2_r, 0_r, 2_r};
    std::vector<real> gold_z{alpha * std::exp(-1.2_r) - alpha, 0_r, 2_r};
    std::vector<real> gold_dfdz{alpha * std::exp(-1.2_r), 1_r, 1_r};
    ActivationTestImpl(f, z, gold_z, gold_dfdz);
    //
    ActivationSIMDApply<19, 3, krylov::Direction::ColMajor>(f);
    ActivationSIMDApply<7, 17, krylov::Direction::RowMajor>(f);
    ActivationSIMDApply<19, 3, krylov::Direction::ColMajor, 32>(f);
    ActivationSIMDApply<7, 17, krylov::Direction::RowMajor, 32>(f);
  }
}

TEST(MlpActivation, ReLU) {
  ai::ReLUActivation<real> f;
  std::vector<real> z{-1.2_r, 0_r, 2_r};
  std::vector<real> gold_z{0_r, 0_r, 2_r};
  std::vector<real> gold_dfdz{0_r, 1_r, 1_r};
  ActivationTestImpl(f, z, gold_z, gold_dfdz);
  //
  ActivationSIMDApply<19, 3, krylov::Direction::ColMajor>(f);
  ActivationSIMDApply<7, 17, krylov::Direction::RowMajor>(f);
  ActivationSIMDApply<19, 3, krylov::Direction::ColMajor, 32>(f);
  ActivationSIMDApply<7, 17, krylov::Direction::RowMajor, 32>(f);
}

TEST(MlpActivation, SiLU) {
  ai::SiLUActivation<real> f;
  // Goldens derived from the definition f(z) = z * sigmoid(z),
  // f'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z))). The +-30 entries exercise
  // both saturated tails.
  auto const sigmoid = [](real x) { return 1_r / (1_r + std::exp(-x)); };
  std::vector<real> z{-30_r, -1.2_r, 0_r, 2_r, 30_r};
  std::vector<real> gold_z, gold_dfdz;
  for (real const v : z) {
    real const s = sigmoid(v);
    gold_z.push_back(v * s);
    gold_dfdz.push_back(s * (1_r + v * (1_r - s)));
  }
  ActivationTestImpl(f, z, gold_z, gold_dfdz);

  // Independently-computed references (constants, NOT derived from the impl or
  // the sigmoid lambda above) so the formula itself is validated, not just
  // self-consistency. silu(1) = 1/(1+e^-1); silu(-1) = -1/(1+e^1); silu'(1).
  // Explicit tolerance: these are 7-digit reference constants, so validate to ~1e-5
  // (not the double-precision kTolerance of 1e-10).
  {
    real v = 1_r, d = 0_r;
    f(v, d);
    EXPECT_NEAR(v, 0.7310586_r, 1e-5_r);
    EXPECT_NEAR(d, 0.9276705_r, 1e-5_r);
  }
  {
    real v = -1_r;
    f(v);
    EXPECT_NEAR(v, -0.2689414_r, 1e-5_r);
  }

  ActivationSIMDApply<19, 3, krylov::Direction::ColMajor>(f);
  ActivationSIMDApply<7, 17, krylov::Direction::RowMajor>(f);
  ActivationSIMDApply<19, 3, krylov::Direction::ColMajor, 32>(f);
  ActivationSIMDApply<7, 17, krylov::Direction::RowMajor, 32>(f);
}

// Check the saturated limits: f(z)->0, f'(z)->0 as z->-inf, and
// f(z)->z, f'(z)->1 as z->+inf.
TEST(MlpActivation, SiLUSaturation) {
  ai::SiLUActivation<real> f;

  for (real const z0 : std::vector<real>{-1000_r, -750_r, -100_r}) {
    real z = z0;
    real dfdz = 1_r; // sentinel that must be overwritten
    f(z, dfdz);
    EXPECT_TRUE(std::isfinite(z) && std::isfinite(dfdz)) << "non-finite at z=" << z0;
    EXPECT_NEAR(z, 0_r, 1e-30_r); // f(z) -> 0
    EXPECT_NEAR(dfdz, 0_r, 1e-30_r); // f'(z) -> 0
  }

  for (real const z0 : std::vector<real>{100_r, 750_r, 1000_r}) {
    real z = z0;
    real dfdz = 0_r;
    f(z, dfdz);
    EXPECT_TRUE(std::isfinite(z) && std::isfinite(dfdz)) << "non-finite at z=" << z0;
    EXPECT_NEAR(z, z0, 1e-3_r * z0); // f(z) -> z
    EXPECT_NEAR(dfdz, 1_r, 1e-6_r); // f'(z) -> 1
  }

  // Check the value-only overload too.
  real z = -1000_r;
  f(z);
  EXPECT_TRUE(std::isfinite(z));
}

// The SIMD path uses Mochi's vectorized Exp (not std::exp), so verify the whole
// vectorized tail stays finite and matches the scalar path for both the value
// and value+derivative overloads. ActivationSIMDApply only feeds SetRandom
// values in [-2.3, 2], so the vectorized tail would otherwise be untested.
TEST(MlpActivation, SiLUSimdSaturation) {
  ai::SiLUActivation<real> f;
  std::vector<real> const vals{-100_r, -88_r, -30_r, -5_r, -1_r, 0_r, 1_r, 5_r, 30_r, 88_r, 100_r};
  // 41x4 = 164 contiguous entries -> SIMD bulk (width 8) + a 4-element remainder.
  constexpr int kRows = 41, kCols = 4;
  Matrix<real> z0(kRows, kCols);
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kCols; ++j) {
      z0(i, j) = vals[static_cast<size_t>(i * kCols + j) % vals.size()];
    }
  }

  // Scalar (std::exp) goldens for both value and derivative.
  Matrix<real> goldZ(z0), goldD(z0);
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kCols; ++j) {
      f(goldZ(i, j), goldD(i, j));
    }
  }

  // The SIMD Exp is a ~1-ULP approximation, not bit-identical to std::exp, so
  // agreement uses a sanity tolerance.
  {
    Matrix<real> z(z0);
    ai::details::ActivationInPlace(f, z); // value-only SIMD path
    for (int i = 0; i < kRows; ++i) {
      for (int j = 0; j < kCols; ++j) {
        EXPECT_TRUE(std::isfinite(z(i, j)));
        EXPECT_NEAR(z(i, j), goldZ(i, j), 2e-3_r);
      }
    }
  }
  {
    Matrix<real> z(z0), d(z0);
    ai::details::ActivationInPlace(f, z, d); // value+derivative SIMD path
    for (int i = 0; i < kRows; ++i) {
      for (int j = 0; j < kCols; ++j) {
        EXPECT_TRUE(std::isfinite(z(i, j)) && std::isfinite(d(i, j)));
        EXPECT_NEAR(z(i, j), goldZ(i, j), 2e-3_r);
        EXPECT_NEAR(d(i, j), goldD(i, j), 2e-3_r);
      }
    }
  }
}
