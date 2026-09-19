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

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/block_view_vector.h>
#include <mochi_core/linear_algebra/krylov/incomplete_cholesky_prec_completion.h>
#include <mochi_core/linear_algebra/krylov/preconditioner.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph_views.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/sparsity_utils.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace mochi::krylov {

/// @brief Select Cholesky completion without dropped fill in natural elimination order.
struct NaturalOrderCompletion {};

/// @brief Empty definition of incomplete Cholesky for non-supported input matrix.
template <typename MatrixType>
struct IncompleteCholeskyPrec {
  IncompleteCholeskyPrec() = delete;
};

/// @brief Incomplete or natural-order-completed Cholesky preconditioner for
/// @ref BlockSparseMatrix input.
///
/// @pre The input matrix must be symmetric positive definite (SPD).
///
/// @warning Factorization breakdown is not reported to the caller. Block pivots must remain finite,
/// positive definite, and numerically safe to invert.
template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
struct IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>
    : Preconditioner<std::remove_const_t<Scalar>> {
  using MatrixType = BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>;
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  static constexpr auto kType = PreconditionerType::IC0;

  /// @brief Constructor
  ///
  /// @param[in] A Input block sparse matrix
  /// @param[in] level Level of fill-in
  /// @param[in] alphaShift Scaling factor for the shift
  ///
  /// @pre @p alphaShift must be nonnegative.
  ///
  /// @note When shifting the input matrix, the incomplete factor is the one for A + alphaShift *
  /// (trace(A) / N) * I where N is the dimension of A.
  explicit IncompleteCholeskyPrec(MatrixType const& A, int level, Scalar alphaShift);

  /**
   * @brief Construct a Cholesky preconditioner using natural-order symbolic completion.
   *
   * @param[in] A Input block sparse matrix.
   * @param[in] completion Selects symbolic completion without dropped fill.
   * @param[in] alphaShift Scaling factor for the diagonal shift.
   *
   * @pre The block sparsity is square and structurally symmetric, with sorted unique rows and a
   * diagonal block in every row.
   * @pre @p alphaShift must be nonnegative.
   *
   * @note Only the diagonal and upper-triangular numerical blocks are used.
   * @note Completion currently requires `int` block indices and pointers.
   * @note In exact arithmetic, the preconditioner exactly factors the shifted matrix A +
   * alphaShift * (trace(A) / N) * I, where N is the dimension of A.
   *
   * @warning Worst-case storage is O(n^2) factor blocks and O(n^3) cached product indices, where n
   * is the number of block rows. Callers must bound n accordingly.
   */
  explicit IncompleteCholeskyPrec(
      MatrixType const& A,
      NaturalOrderCompletion completion,
      Scalar alphaShift);

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors).
  /// @param[in] x Input column vector(s)
  /// @param[out] y Output column vector(s)
  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& y) const;

  /// @brief Apply the preconditioner to a column vector.
  /// @param[in] x Input column vector.
  /// @param[out] Px Output column vector.
  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<NonConstScalar> Px) const override {
    operator()(x, Px);
  }

  /// @brief Concurrent solve for parallel execution.
  /// @todo Implement when linear solvers refactor is complete.
  void ConcurrentSolve(
      [[maybe_unused]] ColumnVectorView<Scalar const> x,
      [[maybe_unused]] ColumnVectorView<NonConstScalar> Px,
      [[maybe_unused]] ParallelWorkerInfo const& data) const override {
    MOCHI_ASSERT(false, "ConcurrentSolve not supported for incomplete Cholesky preconditioner.");
  }

  constexpr PreconditionerType GetType() const override {
    MOCHI_ASSERT(
        _level == 0 || _level == kNaturalOrderCompletionLevel,
        "IncompleteCholeskyPrec::GetType assumes IC0 or exact completion.");
    return kType;
  }

  /// @brief Update the preconditioner from a block sparse matrix with the same sparsity pattern.
  /// @param[in] A Input block sparse matrix.
  /// @note The factor sparsity is not recomputed. Natural-order completion requires the same source
  /// sparsity used at construction.
  template <
      typename ScalarA,
      typename CRIdxA,
      typename PtrA,
      template <typename, typename...> typename StorageA>
  void Update(BlockSparseMatrix<ScalarA, kBlockSize, CRIdxA, PtrA, StorageA> const& A);

 private:
  static constexpr int kNaturalOrderCompletionLevel = -1;
  static constexpr auto kInvalidPosition = static_cast<NonConstPtr>(-1);

  struct NaturalOrderData {
    /// Global block positions in the completed factor.
    DynamicArray<NonConstPtr> upperFillPositions;
    /// Row-local source offsets for transposed factor blocks.
    DynamicArray<NonConstPtr> transposeOffsets;
    /// Row-local target offsets for Schur products.
    DynamicArray<NonConstPtr> productTargets;
  };

  void InitializeNaturalOrderData(MatrixType const& A);

  template <
      typename ScalarA,
      typename CRIdxA,
      typename PtrA,
      template <typename, typename...> typename StorageA>
  void UpdateNaturalOrder(BlockSparseMatrix<ScalarA, kBlockSize, CRIdxA, PtrA, StorageA> const& A);

  void FactorizeNaturalOrder(
      NonConstScalar alpha,
      Span<NonConstPtr const> transposeOffsets,
      Span<NonConstPtr const> productTargets);

  /// @brief Cache data that depends only on the factor sparsity pattern.
  void InitializeFactorData();

  /// @brief Perform the incomplete Cholesky factorization.
  /// @param[in] alpha Scalar diagonal shift.
  /// @warning Assumes that _factorization has already been initialized with the matrix values.
  void Factorize(NonConstScalar alpha);

  /// @brief In-place block LDLT factorization and numerical workspace.
  /// - The strict lower part stores L; its unit diagonal is implicit.
  /// - The diagonal stores D.
  /// - The strict upper part stores W = D * L^T.
  BlockSparseMatrix<NonConstScalar, kBlockSize, NonConstIdx, NonConstPtr> _factorization;

  /// @brief Inverses of the diagonal D blocks.
  DynamicArray<RowMatrix<NonConstScalar, kBlockSize, kBlockSize>> _inverseDiagBlocks;

  /// @brief Start of the upper part, including the diagonal, as a global block index.
  DynamicArray<NonConstPtr> _uStart;

  /// @brief Column map for ordinary factorization, or source-to-factor map for natural-order
  /// completion.
  DynamicArray<NonConstPtr> _factorizationWorkspace;

  /// @brief Original shift factor passed to the constructor.
  NonConstScalar _alphaShift;

  /// @brief Nonnegative fill level, or @ref kNaturalOrderCompletionLevel.
  int _level = 0;

  /// @brief Immutable symbolic schedules allocated only for natural-order completion.
  std::shared_ptr<NaturalOrderData> _naturalOrderData;
};

/// @brief Implementation of incomplete Cholesky for Matrix input.
///
/// @tparam Scalar  Scalar type for input matrix
/// @tparam kRowsAtCompileTime
/// @tparam kColsAtCompileTime
/// @tparam kMajorDir Storage direction for input matrix
/// @tparam kOwnership
/// @tparam kMajorDim
///
/// @note Only the fill-in level 0 is implemented.
/// @note The incomplete factor is identical to the one obtained from the sparse representation of
/// the pruned matrix (i.e. by removing the zeros).
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
struct IncompleteCholeskyPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>
    : Preconditioner<std::remove_const_t<Scalar>> {
  using NonConstScalar = std::remove_const_t<Scalar>;
  static constexpr auto kType = PreconditionerType::IC0;

  /// @brief Constructor
  ///
  /// @param[in] A Input sparse matrix
  /// @param[in] level Level of fill-in
  /// @param[in] alphaShift Scaling factor for the shift
  ///
  /// @note When shifting the input matrix, the incomplete factor is the one for A + alphaShift *
  /// (trace(A) / N) * I where N is the dimension of A.
  /// @note The incomplete factor is identical to the one obtained from the sparse representation of
  /// the pruned matrix (i.e. by removing the zeros).
  /// @note Only the upper triangular part of the input matrix is used.
  explicit IncompleteCholeskyPrec(
      Matrix<
          Scalar,
          kRowsAtCompileTime,
          kColsAtCompileTime,
          kMajorDir,
          kOwnership,
          kMajorDim> const& A,
      int level,
      Scalar alphaShift);

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors).
  /// @param[in] xin Input column vector(s)
  /// @param[out] yout Output column vector(s)
  template <typename Input, typename Output>
  void operator()(Input const& xin, Output&& yout) const;

  /// @brief Apply the preconditioner to a column vector.
  /// @param[in] x Input column vector.
  /// @param[out] Px Output column vector.
  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<NonConstScalar> Px) const override {
    operator()(x, Px);
  }

  /// @brief Concurrent solve for parallel execution.
  /// @todo Implement when linear solvers refactor is complete.
  void ConcurrentSolve(
      [[maybe_unused]] ColumnVectorView<Scalar const> x,
      [[maybe_unused]] ColumnVectorView<NonConstScalar> Px,
      [[maybe_unused]] ParallelWorkerInfo const& data) const override {
    MOCHI_ASSERT(false, "ConcurrentSolve not supported for incomplete Cholesky preconditioner.");
  }

  constexpr PreconditionerType GetType() const override {
    MOCHI_ASSERT(_level == 0, "IncompleteCholeskyPrec::GetType assumes IC0.");
    return kType;
  }

  /// @brief Update the preconditioner from a matrix with the same size.
  /// @param[in] A Input matrix.
  template <typename ScalarA, Direction kMajorDirA, Ownership kOwnershipA, int kMajorDimA>
  void Update(
      Matrix<
          ScalarA,
          kRowsAtCompileTime,
          kColsAtCompileTime,
          kMajorDirA,
          kOwnershipA,
          kMajorDimA> const& A);

 protected:
  /// @brief Perform the incomplete Cholesky factorization.
  /// @warning Assumes that _rIC has already been initialized with the matrix values.
  void Factorize();

  /// @brief Storage for the incomplete factors R^T and R
  /// - The lower triangular matrix R^T is stored in the lower part of _rIC
  /// - The upper triangular matrix R is stored in the upper part of _rIC
  RowMatrix<NonConstScalar, kRowsAtCompileTime, kColsAtCompileTime> _rIC;

  /// @brief Scalar for diagonal shift (A + _alpha * trace(A) / N * I)
  NonConstScalar _alpha;

  /// @brief Original shift factor passed to the constructor.
  NonConstScalar _alphaShift;

  /// @brief Fill-in level.
  int _level = 0;
};

/// @brief Implementation of incomplete Cholesky for SparseMatrix input
///
/// @note We use the CSR implementation from Saad (as a starting point)
/// https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf
/// (Section 10.3.2 -- p. 309 Fortran Code)
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
struct IncompleteCholeskyPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>
    : Preconditioner<std::remove_const_t<Scalar>> {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  static constexpr auto kType = PreconditionerType::IC0;

  /// @brief Constructor
  ///
  /// @param[in] A Input sparse matrix
  /// @param[in] level Level of fill-in
  /// @param[in] alphaShift Scaling factor for the shift
  ///
  /// @note When shifting the input matrix, the incomplete factor is the one for A + alphaShift *
  /// (trace(A) / N) * I where N is the dimension of A.
  /// @note Only the upper triangular part of the input matrix is used.
  explicit IncompleteCholeskyPrec(
      SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& A,
      int level,
      Scalar alphaShift);

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors).
  /// @param[in] x Input column vector(s)
  /// @param[out] y Output column vector(s)
  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& y) const;

  /// @brief Apply the preconditioner to a column vector.
  /// @param[in] x Input column vector.
  /// @param[out] Px Output column vector.
  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<NonConstScalar> Px) const override {
    operator()(x, Px);
  }

  /// @brief Concurrent solve for parallel execution.
  /// @todo Implement when linear solvers refactor is complete.
  void ConcurrentSolve(
      [[maybe_unused]] ColumnVectorView<Scalar const> x,
      [[maybe_unused]] ColumnVectorView<NonConstScalar> Px,
      [[maybe_unused]] ParallelWorkerInfo const& data) const override {
    MOCHI_ASSERT(false, "ConcurrentSolve not supported for incomplete Cholesky preconditioner.");
  }

  constexpr PreconditionerType GetType() const override {
    MOCHI_ASSERT(_level == 0, "IncompleteCholeskyPrec::GetType assumes IC0.");
    return kType;
  }

  /// @brief Update the preconditioner from a sparse matrix with the same sparsity pattern.
  /// @param[in] A Input sparse matrix.
  /// @note The sparsity of the fill level is NOT recomputed. Only the values are updated.
  template <
      typename ScalarA,
      typename CRIdxA,
      typename PtrA,
      template <typename, typename...> typename StorageA>
  void Update(SparseMatrix<ScalarA, CRIdxA, PtrA, StorageA> const& A);

 protected:
  /// @brief Perform the incomplete Cholesky factorization.
  /// @warning Assumes that _rChol has already been initialized with the matrix values.
  void Factorize();

  /// @brief Storage for the incomplete factor R
  SparseMatrix<NonConstScalar, NonConstIdx, NonConstPtr> _rChol;

  /// @brief Array pointing to the start of the upper part (inc. diagonal) for each row
  /// The value for each row is a "global" integer
  DynamicArray<NonConstPtr> _uStart;

  /// @brief Scalar for diagonal shift (A + _alpha * trace(A) / N * I)
  NonConstScalar _alpha;

  /// @brief Original shift factor passed to the constructor.
  NonConstScalar _alphaShift;

  /// @brief Fill-in level.
  int _level = 0;
};

} // namespace mochi::krylov

//
//--- Implementation of class member functions
//

namespace mochi::krylov {

//
//--- Specialization for BlockSparseMatrix
//

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::
    IncompleteCholeskyPrec(
        BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage> const& A,
        int level,
        Scalar alphaShift)
    : _alphaShift(alphaShift), _level(level) {
  static_assert(
      std::signed_integral<CRIdx> && sizeof(CRIdx) <= sizeof(int),
      "IncompleteCholeskyPrec requires signed matrix indices representable as int.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(_level >= 0, "Fill-in level must not be negative.");
  MOCHI_ASSERT_VERBOSE(_alphaShift >= Scalar{0}, "Out-of-range shifting factor.");
  mochi::details::ConvertToFillLevel(_level, A, _factorization);
  InitializeFactorData();
  auto const alpha = _alphaShift > Scalar{0} && A.Rows() > 0 ? _alphaShift * (Trace(A) / A.Rows())
                                                             : NonConstScalar{0};
  Factorize(alpha);
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::
    IncompleteCholeskyPrec(MatrixType const& A, NaturalOrderCompletion, Scalar alphaShift)
    : _alphaShift(alphaShift),
      _level(kNaturalOrderCompletionLevel),
      _naturalOrderData(std::make_shared<NaturalOrderData>()) {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(_alphaShift >= Scalar{0}, "Out-of-range shifting factor.");
  auto factorPattern = details::CompleteCholeskyPatternNaturalOrder(AsGraphView(A));
  DynamicArray<NonConstScalar> factorValues;
  auto const valueCount = static_cast<size_t>(kBlockSize) * kBlockSize * factorPattern.NumTargets();
  if constexpr (kIsResizeNoInitSafe<NonConstScalar>) {
    factorValues.resize_noinit(valueCount);
  } else {
    factorValues.resize(valueCount);
  }
  _factorization.Reset(
      A.BlockCols(),
      std::move(factorPattern.GetMovablePointers()),
      std::move(factorPattern.GetMovableTargets()),
      std::move(factorValues));
  _factorizationWorkspace.reserve(static_cast<size_t>(A.NumNonZeroBlocks()));
  InitializeFactorData();
  InitializeNaturalOrderData(A);
  UpdateNaturalOrder(A);
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <
    typename ScalarA,
    typename CRIdxA,
    typename PtrA,
    template <typename, typename...> typename StorageA>
void IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::Update(
    BlockSparseMatrix<ScalarA, kBlockSize, CRIdxA, PtrA, StorageA> const& A) {
  static_assert(std::is_same_v<Scalar const, ScalarA const>, "Inconsistent scalar types");
  static_assert(std::is_same_v<CRIdx const, CRIdxA const>, "Inconsistent integer types");
  static_assert(std::is_same_v<Ptr const, PtrA const>, "Inconsistent integer types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == _factorization.Rows(), "Matrix size mismatch.");
  if (_level == 0) {
    auto srcValues = A.Values();
    auto dstValues = _factorization.Values();
    MOCHI_ASSERT_VERBOSE(srcValues.size() == dstValues.size(), "Value array size mismatch.");
    MOCHI_ASSERT_VERBOSE(_factorization.Pointers() == A.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(_factorization.Indices() == A.Indices(), "Sparsity pattern mismatch.");
    std::copy(srcValues.begin(), srcValues.end(), dstValues.begin());
  } else if (_level == kNaturalOrderCompletionLevel) {
    UpdateNaturalOrder(A);
    return;
  } else {
    MOCHI_ASSERT_VERBOSE(_level > 0, "Invalid fill-in level.");
    _factorization.SetZero();
    _factorization += A;
  }
  auto const alpha = _alphaShift > Scalar{0} && A.Rows() > 0 ? _alphaShift * (Trace(A) / A.Rows())
                                                             : NonConstScalar{0};
  Factorize(alpha);
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::
    InitializeNaturalOrderData(MatrixType const& A) {
  auto const sourceCount = static_cast<size_t>(A.NumNonZeroBlocks());
  auto const factorCount = static_cast<size_t>(_factorization.NumNonZeroBlocks());
  auto const rowCount = static_cast<size_t>(_factorization.BlockRows());
  MOCHI_ASSERT_VERBOSE(
      factorCount >= sourceCount && (factorCount - sourceCount) % 2 == 0,
      "Invalid source and completed factor patterns.");
  MOCHI_ASSERT_VERBOSE(
      factorCount >= rowCount && (factorCount - rowCount) % 2 == 0,
      "Invalid completed factor pattern.");
  MOCHI_ASSERT_VERBOSE(_naturalOrderData != nullptr, "Missing natural-order data.");

  auto& data = *_naturalOrderData;
  _factorizationWorkspace.resize_noinit(sourceCount);
  data.upperFillPositions.reserve((factorCount - sourceCount) / 2);
  data.transposeOffsets.reserve((factorCount - rowCount) / 2);

  auto const srcPtr = A.Pointers();
  auto const srcIdx = A.Indices();
  auto const dstPtr = _factorization.Pointers();
  auto const dstIdx = _factorization.Indices();
  size_t productCount = 0;
  for (NonConstIdx row = 0; row < A.BlockRows(); ++row) {
    NonConstPtr source = srcPtr[row];
    for (NonConstPtr target = dstPtr[row]; target < dstPtr[row + 1]; ++target) {
      if (source < srcPtr[row + 1] && srcIdx[source] == dstIdx[target]) {
        _factorizationWorkspace[source++] = target;
      } else {
        MOCHI_ASSERT_VERBOSE(
            source == srcPtr[row + 1] || dstIdx[target] < srcIdx[source],
            "Completed factor pattern misses a source block.");
        if (dstIdx[target] >= row) {
          data.upperFillPositions.push_back(target);
        }
      }
    }
    MOCHI_ASSERT_VERBOSE(
        source == srcPtr[row + 1], "Completed factor pattern misses source blocks.");
    auto const upperCount = static_cast<size_t>(dstPtr[row + 1] - _uStart[row] - 1);
    productCount += upperCount * (upperCount + 1) / 2;
  }
  MOCHI_ASSERT_VERBOSE(
      data.upperFillPositions.size() == (factorCount - sourceCount) / 2,
      "Incomplete upper-fill positions.");
  data.productTargets.reserve(productCount);

  // Flatten transpose and product positions in (i, k, j) left-looking LDLT order:
  // L_ik = W_ki^T D_k^-1 and S_ij -= L_ik W_kj.
  for (NonConstIdx row = 0; row < _factorization.BlockRows(); ++row) {
    auto const rowIndices = _factorization.Indices(row);
    auto const localDiag = _uStart[row] - dstPtr[row];
    for (NonConstPtr p = 0; p < localDiag; ++p) {
      auto const sourceRow = rowIndices[p];
      auto const sourceIndices = _factorization.Indices(sourceRow);
      auto const sourceDiag = _uStart[sourceRow] - dstPtr[sourceRow];
      auto source =
          std::lower_bound(sourceIndices.begin() + sourceDiag + 1, sourceIndices.end(), row);
      MOCHI_ASSERT_VERBOSE(
          source != sourceIndices.end() && *source == row,
          "Completed factor pattern must be symmetric.");
      data.transposeOffsets.push_back(static_cast<NonConstPtr>(source - sourceIndices.begin()));

      auto target = rowIndices.begin() + localDiag;
      for (; source != sourceIndices.end(); ++source) {
        target = std::lower_bound(target, rowIndices.end(), *source);
        MOCHI_ASSERT_VERBOSE(
            target != rowIndices.end() && *target == *source,
            "Completed factor pattern is not closed under left-looking updates.");
        data.productTargets.push_back(static_cast<NonConstPtr>(target - rowIndices.begin()));
        ++target;
      }
    }
  }
  MOCHI_ASSERT_VERBOSE(
      data.transposeOffsets.size() == (factorCount - rowCount) / 2,
      "Incomplete transpose offsets.");
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <
    typename ScalarA,
    typename CRIdxA,
    typename PtrA,
    template <typename, typename...> typename StorageA>
void IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::
    UpdateNaturalOrder(BlockSparseMatrix<ScalarA, kBlockSize, CRIdxA, PtrA, StorageA> const& A) {
  MOCHI_ASSERT_VERBOSE(_naturalOrderData != nullptr, "Missing natural-order data.");
  auto const& data = *_naturalOrderData;
  MOCHI_ASSERT_VERBOSE(
      static_cast<size_t>(A.NumNonZeroBlocks()) == _factorizationWorkspace.size(),
      "Sparsity pattern mismatch.");
  auto const srcPtr = A.Pointers();
  auto const srcIdx = A.Indices();
  auto const dstPtr = _factorization.Pointers();
  [[maybe_unused]] auto const dstIdx = _factorization.Indices();

  auto fill = data.upperFillPositions.begin();
  for (NonConstIdx row = 0; row < A.BlockRows(); ++row) {
    auto dstValues = _factorization.Values(row);
    while (fill != data.upperFillPositions.end() && *fill < dstPtr[row + 1]) {
      MOCHI_ASSERT_VERBOSE(*fill >= dstPtr[row], "Invalid fill position.");
      dstValues[*fill - dstPtr[row]].SetZero();
      ++fill;
    }
    auto const srcValues = A.Values(row);

    // Copy only source blocks on or above the diagonal. FactorizeNaturalOrder overwrites every
    // strict-lower factor block before use.
    for (NonConstPtr p = srcPtr[row]; p < srcPtr[row + 1]; ++p) {
      auto const target = _factorizationWorkspace[p];
      MOCHI_ASSERT_VERBOSE(
          target >= dstPtr[row] && target < dstPtr[row + 1] && srcIdx[p] == dstIdx[target],
          "Sparsity pattern mismatch.");
      if (srcIdx[p] >= row) {
        dstValues[target - dstPtr[row]] = srcValues[p - srcPtr[row]];
      }
    }
  }
  MOCHI_ASSERT_VERBOSE(fill == data.upperFillPositions.end(), "Invalid fill position.");

  auto const alpha =
      _alphaShift > Scalar{0} && A.Rows() > 0 ? _alphaShift * (Trace(A) / A.Rows()) : Scalar{0};
  FactorizeNaturalOrder(
      alpha, MakeConstSpan(data.transposeOffsets), MakeConstSpan(data.productTargets));
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::
    FactorizeNaturalOrder(
        NonConstScalar alpha,
        Span<NonConstPtr const> transposeOffsets,
        Span<NonConstPtr const> productTargets) {
  auto const pointers = _factorization.Pointers();
  if (alpha > Scalar{0}) {
    for (NonConstIdx row = 0; row < _factorization.BlockRows(); ++row) {
      auto diagonal = _factorization.Values(row)[_uStart[row] - pointers[row]];
      for (int j = 0; j < kBlockSize; ++j) {
        diagonal(j, j) += alpha;
      }
    }
  }

  size_t scheduleIndex = 0;
  size_t productIndex = 0;
  for (NonConstIdx row = 0; row < _factorization.BlockRows(); ++row) {
    auto const rowIndices = _factorization.Indices(row);
    auto values = _factorization.Values(row);
    auto const localDiag = _uStart[row] - pointers[row];
    for (NonConstPtr p = 0; p < localDiag; ++p, ++scheduleIndex) {
      auto const sourceRow = rowIndices[p];
      auto const sourceValues = _factorization.Values(sourceRow);
      MOCHI_ASSERT_VERBOSE(scheduleIndex < transposeOffsets.size(), "Invalid transpose schedule.");
      auto const sourceOffset = transposeOffsets[scheduleIndex];
      RowMatrix<NonConstScalar, kBlockSize, kBlockSize> const left =
          sourceValues[sourceOffset].Transpose() * _inverseDiagBlocks[sourceRow];
      values[p] = left;
      for (NonConstPtr source = sourceOffset; source < sourceValues.NumBlocks(); ++source) {
        MOCHI_ASSERT_VERBOSE(productIndex < productTargets.size(), "Invalid product schedule.");
        if constexpr (kBlockSize <= 4) {
          // Materialize products to avoid redundant source reloads: the compiler cannot prove that
          // these disjoint views into one allocation do not alias. Larger products may spill,
          // making fused evaluation faster.
          RowMatrix<NonConstScalar, kBlockSize, kBlockSize> const product =
              left * sourceValues[source];
          values[productTargets[productIndex]] -= product;
        } else {
          values[productTargets[productIndex]] -= left * sourceValues[source];
        }
        ++productIndex;
      }
    }
    _inverseDiagBlocks[row] = SymInverse(values[localDiag]);
  }
  MOCHI_ASSERT_VERBOSE(productIndex == productTargets.size(), "Incomplete product schedule.");
  MOCHI_ASSERT_VERBOSE(scheduleIndex == transposeOffsets.size(), "Incomplete transpose schedule.");
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void IncompleteCholeskyPrec<
    BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::InitializeFactorData() {
  auto const rPtr = _factorization.Pointers();
  _uStart.resize_noinit(_factorization.BlockRows());
  _factorizationWorkspace.clear();
  _factorizationWorkspace.resize(_factorization.BlockRows(), kInvalidPosition);
  _inverseDiagBlocks.resize(_factorization.BlockRows());
  for (NonConstIdx i = 0; i < _factorization.BlockRows(); ++i) {
    auto const cIdx = _factorization.Indices(i);
    MOCHI_ASSERT_VERBOSE(
        std::is_sorted(cIdx.begin(), cIdx.end()), "Column entries need to be sorted");
    auto const diagonal = std::lower_bound(cIdx.begin(), cIdx.end(), i);
    MOCHI_ASSERT_VERBOSE(diagonal != cIdx.end() && *diagonal == i, "Missing diagonal block");
    _uStart[i] = rPtr[i] + static_cast<NonConstPtr>(diagonal - cIdx.begin());
  }
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::Factorize(
    NonConstScalar alpha) {
  auto* const columnToLocal = _factorizationWorkspace.data();
  MOCHI_ASSERT_VERBOSE(
      std::ranges::all_of(
          _factorizationWorkspace,
          [](NonConstPtr position) { return position == kInvalidPosition; }),
      "Factorization workspace is not clear.");
  auto const rPtr = _factorization.Pointers();
  RowMatrix<NonConstScalar, kBlockSize, kBlockSize> scaling;
  if (alpha > NonConstScalar{0}) {
    for (NonConstIdx i = 0; i < _factorization.BlockRows(); ++i) {
      auto diagonal = _factorization.Values(i)[_uStart[i] - rPtr[i]];
      for (int j = 0; j < kBlockSize; ++j) {
        diagonal(j, j) += alpha;
      }
    }
  }

  // For the current Schur row S, store L_ik = S_ik D_k^-1 and update S_ij -= L_ik W_kj.
  // Intended for bounded-degree FEM sparsity and small fixed fill. High-degree rows may be
  // superlinear.
  for (NonConstIdx i = 0; i < _factorization.BlockRows(); ++i) {
    auto const colIdx = _factorization.Indices(i);
    auto values = _factorization.Values(i);
    for (NonConstPtr p = 0; p < colIdx.size(); ++p) {
      columnToLocal[colIdx[p]] = p;
    }
    auto const localDiag = _uStart[i] - rPtr[i];
    for (NonConstPtr p = 0; p < localDiag; ++p) {
      auto const k = colIdx[p];
      auto const sourceIdx = _factorization.Indices(k);
      auto const sourceValues = _factorization.Values(k);
      scaling = values[p] * _inverseDiagBlocks[k];
      values[p] = scaling;
      auto const sourceDiag = _uStart[k] - rPtr[k];
      for (NonConstPtr q = sourceDiag + 1; q < sourceIdx.size(); ++q) {
        auto const targetPosition = columnToLocal[sourceIdx[q]];
        if (targetPosition == kInvalidPosition) {
          continue;
        }
        values[targetPosition] -= scaling * sourceValues[q];
      }
    }
    _inverseDiagBlocks[i] = SymInverse(values[localDiag]);
    for (NonConstPtr p = 0; p < colIdx.size(); ++p) {
      columnToLocal[colIdx[p]] = kInvalidPosition;
    }
  }
  MOCHI_ASSERT_VERBOSE(
      std::ranges::all_of(
          _factorizationWorkspace,
          [](NonConstPtr position) { return position == kInvalidPosition; }),
      "Factorization workspace was not restored.");
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename Input, typename Output>
void IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::operator()(
    Input const& xin,
    Output&& yout) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_factorization.Rows(), xin, yout);
  MatrixView<
      typename details::MatTraits<Input>::Scalar const,
      details::MatTraits<Input>::kNumRows,
      details::MatTraits<Input>::kNumCols,
      details::MatTraits<Input>::kMajorDir,
      krylov::kDynamic>
      x(xin.data(), xin.Rows(), xin.Cols(), xin.LeadDim());
  MatrixView<
      std::remove_const_t<typename details::MatTraits<Output>::Scalar>,
      details::MatTraits<Output>::kNumRows,
      details::MatTraits<Output>::kNumCols,
      details::MatTraits<Output>::kMajorDir,
      krylov::kDynamic>
      y(yout.data(), yout.Rows(), yout.Cols(), yout.LeadDim());
  y = x;

  [[maybe_unused]] ColumnVector<NonConstScalar> xTmp;
  [[maybe_unused]] auto const xTmpRequiredSize = static_cast<NonConstIdx>(
      mochi::details::RowMultiplier<NonConstScalar, kBlockSize>::GetWorkspaceSize(
          _factorization.MaxNnzPerRow()));
  ColumnVector<NonConstScalar, kBlockSize> product;
  [[maybe_unused]] auto productAccessor = mochi::details::GetAccessor(product);
  auto const rPtr = _factorization.Pointers();
  for (int c = 0; c < y.Cols(); ++c) {
    auto yc = y.Col(c);
    [[maybe_unused]] auto yAccessor = mochi::details::GetAccessor(yc);
    for (int i = 0; i < _factorization.BlockRows(); ++i) {
      auto const colIdx = _factorization.Indices(i);
      auto const values = _factorization.Values(i);
      auto yi = yc.template MiddleRows<kBlockSize>(i * kBlockSize, kBlockSize);
      auto const localDiag = _uStart[i] - rPtr[i];
      if constexpr (mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
        mochi::details::RowMultiplier<NonConstScalar, kBlockSize>::ApplyToColVector(
            colIdx.data(),
            values.data(),
            localDiag,
            values.LeadDim(),
            yAccessor,
            productAccessor,
            /*br*/ 0,
            /*c*/ 0,
            xTmp,
            xTmpRequiredSize);
        yi -= product;
      } else {
        static_assert(mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::RowMajor);
        for (int p = 0; p < localDiag; ++p) {
          yi -= values[p] * yc.template MiddleRows<kBlockSize>(colIdx[p] * kBlockSize, kBlockSize);
        }
      }
    }

    for (int i = _factorization.BlockRows(); i-- > 0;) {
      auto const colIdx = _factorization.Indices(i);
      auto const values = _factorization.Values(i);
      auto yi = yc.template MiddleRows<kBlockSize>(i * kBlockSize, kBlockSize);
      auto const localDiag = _uStart[i] - rPtr[i];
      if constexpr (mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
        mochi::details::RowMultiplier<NonConstScalar, kBlockSize>::ApplyToColVector(
            colIdx.data() + localDiag + 1,
            values.data() + (localDiag + 1) * kBlockSize,
            isize(colIdx) - localDiag - 1,
            values.LeadDim(),
            yAccessor,
            productAccessor,
            /*br*/ 0,
            /*c*/ 0,
            xTmp,
            xTmpRequiredSize);
        yi -= product;
      } else {
        static_assert(mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::RowMajor);
        for (int p = localDiag + 1; p < isize(colIdx); ++p) {
          yi -= values[p] * yc.template MiddleRows<kBlockSize>(colIdx[p] * kBlockSize, kBlockSize);
        }
      }
      product = _inverseDiagBlocks[i] * yi;
      yi = product;
    }
  }
}

//
//--- Specialization for Matrix
//

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
IncompleteCholeskyPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
    IncompleteCholeskyPrec(
        Matrix<
            Scalar,
            kRowsAtCompileTime,
            kColsAtCompileTime,
            kMajorDir,
            kOwnership,
            kMajorDim> const& A,
        int level,
        Scalar alphaShift)
    : _alpha(alphaShift), _alphaShift(alphaShift), _level(level) {
  MOCHI_ASSERT_VERBOSE(_level >= 0, "Fill-in level must not be negative.");
  MOCHI_ASSERT_VERBOSE(_alphaShift >= Scalar{0}, "Out-of-range shifting factor.")
  if (_alphaShift > Scalar{0}) {
    _alpha = _alphaShift * (Trace(A) / A.Rows());
  }
  mochi::details::ConvertToFillLevel(_level, A, _rIC);
  Factorize();
}

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
template <typename ScalarA, Direction kMajorDirA, Ownership kOwnershipA, int kMajorDimA>
void IncompleteCholeskyPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
    Update(
        Matrix<
            ScalarA,
            kRowsAtCompileTime,
            kColsAtCompileTime,
            kMajorDirA,
            kOwnershipA,
            kMajorDimA> const& A) {
  static_assert(std::is_same_v<Scalar const, ScalarA const>, "Inconsistent scalar types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == _rIC.Rows(), "Matrix size mismatch.");
  _rIC = A;
  if (_alphaShift > Scalar{0}) {
    _alpha = _alphaShift * (Trace(A) / A.Rows());
  }
  Factorize();
}

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
void IncompleteCholeskyPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
    Factorize() {
  if (_alpha > Scalar{0}) {
    for (int i = 0; i < _rIC.Rows(); ++i) {
      _rIC(i, i) += _alpha;
    }
  }
  //
  int info = 0;
  auto const n = _rIC.Rows();
  for (int i = 0; i < n; ++i) {
    // Test for zero or negative entry on the diagonal
    if (_rIC(i, i) < std::numeric_limits<NonConstScalar>::min())
      MOCHI_UNLIKELY {
        info = i + 1;
        break;
      }
    //
    _rIC(i, i) = Sqrt(_rIC(i, i));
    //
    auto const scaling = NonConstScalar(1.0) / _rIC(i, i);
    _rIC.template Block<1, krylov::kDynamic>(i, i + 1, 1, n - i - 1) *= scaling;
    //
    for (int k = i + 1; k < n; ++k) {
      if (_rIC(i, k) == Scalar{0})
        MOCHI_UNLIKELY {
          continue;
        }
      auto const rik_t = _rIC(i, k);
      for (int j = k; j < n; ++j) {
        if (Abs(_rIC(k, j)) < std::numeric_limits<NonConstScalar>::min())
          MOCHI_UNLIKELY {
            continue;
          }
        _rIC(k, j) -= rik_t * _rIC(i, j);
      }
    }
  }
  if (info != 0) {
    MOCHI_LOG_ERROR("Non-positive pivot at position %d", info - 1);
  }
  //
  // Symmetrize _rIC
  //
  for (int i = 0; i < _rIC.Rows(); ++i) {
    _rIC.template Block<1, krylov::kDynamic>(i, 0, 1, i) =
        _rIC.template Block<krylov::kDynamic, 1>(0, i, i, 1).Transpose();
  }
  //
  for (int i = 0; i < _rIC.Rows(); ++i) {
    _rIC(i, i) = Scalar(1) / _rIC(i, i);
  }
}

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
template <typename Input, typename Output>
void IncompleteCholeskyPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
operator()(Input const& xin, Output&& yout) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_rIC.Rows(), xin, yout);
  //
  //--- Convert input and output to MatrixView
  //
  MatrixView<
      typename details::MatTraits<Input>::Scalar const,
      details::MatTraits<Input>::kNumRows,
      details::MatTraits<Input>::kNumCols,
      details::MatTraits<Input>::kMajorDir,
      krylov::kDynamic>
      x(xin.data(), xin.Rows(), xin.Cols(), xin.LeadDim());
  MatrixView<
      std::remove_const_t<typename details::MatTraits<Output>::Scalar>,
      details::MatTraits<Output>::kNumRows,
      details::MatTraits<Output>::kNumCols,
      details::MatTraits<Output>::kMajorDir,
      krylov::kDynamic>
      y(yout.data(), yout.Rows(), yout.Cols(), yout.LeadDim());
  y = x;
  for (int ii = 0; ii < y.Rows(); ++ii) {
    y.Row(ii) -= _rIC.template Block<1, krylov::kDynamic>(ii, 0, 1, ii) * y.TopRows(ii);
    y.Row(ii) *= _rIC(ii, ii);
  }
  y.Row(y.Rows() - 1) *= _rIC(y.Rows() - 1, y.Rows() - 1);
  for (int ii = y.Rows() - 2; ii >= 0; --ii) {
    y.Row(ii) -= _rIC.template Block<1, krylov::kDynamic>(ii, ii + 1, 1, y.Rows() - ii - 1) *
        y.MiddleRows(ii + 1, y.Rows() - ii - 1);
    y.Row(ii) *= _rIC(ii, ii);
  }
}

//
//--- Specialization for SparseMatrix
//

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
IncompleteCholeskyPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::IncompleteCholeskyPrec(
    SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& A,
    int level,
    Scalar alphaShift)
    : _uStart(A.Rows()), _alpha(alphaShift), _alphaShift(alphaShift), _level(level) {
  static_assert(
      std::signed_integral<CRIdx> && sizeof(CRIdx) <= sizeof(int),
      "IncompleteCholeskyPrec requires signed matrix indices representable as int.");
  MOCHI_ASSERT_VERBOSE(_level >= 0, "Fill-in level must not be negative.");
  MOCHI_ASSERT_VERBOSE(_alphaShift >= Scalar{0}, "Out-of-range shifting factor.")
  if (_alphaShift > Scalar{0}) {
    _alpha = _alphaShift * (Trace(A) / A.Rows());
  }
  mochi::details::ConvertToFillLevel(_level, A, _rChol);
  Factorize();
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <
    typename ScalarA,
    typename CRIdxA,
    typename PtrA,
    template <typename, typename...> typename StorageA>
void IncompleteCholeskyPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::Update(
    SparseMatrix<ScalarA, CRIdxA, PtrA, StorageA> const& A) {
  static_assert(std::is_same_v<Scalar const, ScalarA const>, "Inconsistent scalar types");
  static_assert(std::is_same_v<CRIdx const, CRIdxA const>, "Inconsistent integer types");
  static_assert(std::is_same_v<Ptr const, PtrA const>, "Inconsistent integer types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == _rChol.Rows(), "Matrix size mismatch.");
  if (_level == 0) {
    auto srcValues = A.Values();
    auto dstValues = _rChol.Values();
    MOCHI_ASSERT_VERBOSE(srcValues.size() == dstValues.size(), "Value array size mismatch.");
    MOCHI_ASSERT_VERBOSE(_rChol.Pointers() == A.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(_rChol.Indices() == A.Indices(), "Sparsity pattern mismatch.");
    std::copy(srcValues.begin(), srcValues.end(), dstValues.begin());
  } else {
    _rChol.SetZero();
    _rChol += A;
  }
  if (_alphaShift > Scalar{0}) {
    _alpha = _alphaShift * (Trace(A) / A.Rows());
  }
  Factorize();
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void IncompleteCholeskyPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::Factorize() {
  auto const rPtr = _rChol.Pointers();
  auto const rIdx = _rChol.Indices();
  auto rVal = _rChol.Values();
  //
  // Extract position of "diagonal" & shift diagonal if needed
  //
  for (NonConstIdx i = 0; i < _rChol.Rows(); ++i) {
    auto const cIdx = _rChol.Indices(i);
    auto rValues = _rChol.Values(i);
    MOCHI_ASSERT_VERBOSE(
        std::is_sorted(cIdx.begin(), cIdx.end()), "Column entries need to be sorted");
    auto ptr = std::lower_bound(cIdx.begin(), cIdx.end(), i);
    MOCHI_ASSERT_VERBOSE((ptr != cIdx.end()) && (*ptr == i), "Diagonal entry not found");
    auto const p = static_cast<int>(ptr - cIdx.begin());
    _uStart[i] = p + rPtr[i];
    if (_alpha > Scalar{0}) {
      rValues[p] += _alpha;
    }
  }
  //
  // Algorithm IKJ
  // I = row index
  // K = col index (K < I)
  // J = col index (J > I)
  //
  auto const dummyFlag = static_cast<NonConstPtr>(-1);
  DynamicArray<NonConstPtr> iw(_rChol.Rows(), dummyFlag);
  NonConstIdx info = 0;
  for (NonConstIdx i = 0; i < _rChol.Rows(); ++i) {
    // Test for zero or negative entry on the diagonal
    if (rVal[_uStart[i]] < std::numeric_limits<NonConstScalar>::min())
      MOCHI_UNLIKELY {
        info = i + 1;
        break;
      }
    //--- Get inverse map of rIdx only for the upper "triangular" part
    for (NonConstPtr p = _uStart[i]; p < rPtr[i + 1]; ++p) {
      iw[rIdx[p]] = p;
    }
    //
    RowVectorView<NonConstScalar> ri(&rVal[_uStart[i]], rPtr[i + 1] - _uStart[i]);
    ri *= NonConstScalar(1.0) / Sqrt(rVal[_uStart[i]]);
    //--- Use 'p' as pointer integer along the row 'i'
    //--- TODO Explore whether a `ParallelFor` can accelerate the computation
    NonConstPtr p = _uStart[i] + 1;
    // Extraction of _uStart guarantees that the row has, at least, one entry
    auto const lastCol = rIdx[rPtr[i + 1] - 1];
    for (; p < rPtr[i + 1]; ++p) {
      auto const k = rIdx[p];
      auto const rik_t = rVal[p];
      //--- Use 'pos' as pointer integer along the row 'k'
      NonConstPtr pos = _uStart[k];
      //--- Explore the "upper part" block row 'k'
      //--- only for the blocks that could be updated by block row 'i'
      for (; ((pos < rPtr[k + 1]) && (rIdx[pos] <= lastCol)); ++pos) {
        NonConstPtr jpos = iw[rIdx[pos]];
        // r_{kj} <- r_{kj} - r_{ik}^T (1 / r_{ii}) r_{ij}
        rVal[pos] -= (jpos == dummyFlag) ? Scalar(0) : rik_t * rVal[jpos];
      }
    }
    //--- Reset entries of iw
    for (p = _uStart[i]; p < rPtr[i + 1]; ++p) {
      iw[rIdx[p]] = dummyFlag;
    }
  } // for (NonConstIdx i = 0; i < _rChol.Rows(); ++i)
  //
  if (info != 0) {
    MOCHI_LOG_ERROR("Non-positive pivot at position %d", info - 1);
  }
  //
  // Symmetrize R
  //
  DynamicArray<int> countEntries(_rChol.Rows(), 0);
  for (int i = 0; i < _rChol.Rows(); ++i) {
    auto const rColIdx = _rChol.Indices(i);
    for (int k = int(_uStart[i] - rPtr[i]) + 1; k < isize(rColIdx); ++k) {
      countEntries[rColIdx[k]] += 1;
    }
  }
  DynamicArray<DynamicArray<NonConstScalar>> transposeGraph(_rChol.Cols());
  for (int i = 0; i < _rChol.Cols(); ++i) {
    transposeGraph[i].reserve(countEntries[i]);
  }
  for (int i = 0; i < _rChol.Rows(); ++i) {
    auto const rColIdx = _rChol.Indices(i);
    auto rValues = _rChol.Values(i);
    for (int k = int(_uStart[i] - rPtr[i]) + 1; k < isize(rColIdx); ++k) {
      transposeGraph[rColIdx[k]].push_back(rValues[k]);
    }
  }
  //
  // Fill the lower triangular part
  //
  for (int i = 0; i < _rChol.Rows(); ++i) {
    auto const& list = transposeGraph[i];
    MOCHI_ASSERT_VERBOSE(list.size() == _uStart[i] - rPtr[i], "Incompatible size");
    int k = 0;
    [[maybe_unused]] auto const rColIdx = _rChol.Indices(i);
    auto rValues = _rChol.Values(i);
    for (auto rval : list) {
      MOCHI_ASSERT_VERBOSE(rColIdx[k] <= i, "Incorrect location");
      rValues[k++] = rval;
    }
    //--- Invert the diagonal entry.
    rValues[_uStart[i] - rPtr[i]] = NonConstScalar(1) / rValues[_uStart[i] - rPtr[i]];
  }
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename Input, typename Output>
void IncompleteCholeskyPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::operator()(
    Input const& xin,
    Output&& yout) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_rChol.Rows(), xin, yout);
  //
  //--- Convert input and output to MatrixView
  //
  MatrixView<
      typename details::MatTraits<Input>::Scalar const,
      details::MatTraits<Input>::kNumRows,
      details::MatTraits<Input>::kNumCols,
      details::MatTraits<Input>::kMajorDir,
      krylov::kDynamic>
      x(xin.data(), xin.Rows(), xin.Cols(), xin.LeadDim());
  MatrixView<
      std::remove_const_t<typename details::MatTraits<Output>::Scalar>,
      details::MatTraits<Output>::kNumRows,
      details::MatTraits<Output>::kNumCols,
      details::MatTraits<Output>::kMajorDir,
      krylov::kDynamic>
      y(yout.data(), yout.Rows(), yout.Cols(), yout.LeadDim());
  //
  y = x;
  // Solve "R^T z = x"
  auto rPtr = _rChol.Pointers();
  auto rVal = _rChol.Values();
  auto rCol = _rChol.Indices();
  //
  for (int ir = 0; ir < y.Rows(); ++ir) {
    auto spanCol = rCol.subspan(rPtr[ir], int(_uStart[ir] - rPtr[ir]));
    auto spanVal = rVal.subspan(rPtr[ir], isize(spanCol));
    // We assume that `x` and `y` have the same orientation.
    // Call appropriate kernel depending on the orientation of `y`
    if constexpr (details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
      for (int jc = 0; jc < y.Cols(); ++jc) {
        NonConstScalar sum = 0;
        mochi::details::AddRowSparseTimesColumnVector<NonConstScalar, NonConstIdx>(
            spanCol, spanVal, y.Col(jc).Data(), sum);
        y(ir, jc) -= sum;
      }
    } else {
      for (int jc = 0; jc < y.Cols(); ++jc) {
        NonConstScalar sum = 0;
        mochi::details::AddRowSparseTimesVector<NonConstScalar, NonConstIdx>(
            spanCol, spanVal, y.Col(jc), sum);
        y(ir, jc) -= sum;
      }
    }
    y.Row(ir) *= rVal[_uStart[ir]];
  }
  // Solve "R y = z"
  int ir = _rChol.Rows() - 1;
  y.Row(ir) *= rVal[_uStart[ir]];
  for (ir = _rChol.Rows() - 2; ir >= 0; --ir) {
    auto const upperPos = _uStart[ir] + 1;
    auto const len = int(rPtr[ir + 1] - upperPos);
    auto spanCol = rCol.subspan(upperPos, len);
    auto spanVal = rVal.subspan(upperPos, len);
    // We assume that `x` and `y` have the same orientation.
    // Call appropriate kernel depending on the orientation of `x`
    if constexpr (details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
      for (int jc = 0; jc < y.Cols(); ++jc) {
        NonConstScalar sum = 0;
        mochi::details::AddRowSparseTimesColumnVector<NonConstScalar, NonConstIdx>(
            spanCol, spanVal, y.Col(jc).Data(), sum);
        y(ir, jc) -= sum;
      }
    } else {
      for (int jc = 0; jc < y.Cols(); ++jc) {
        NonConstScalar sum = 0;
        mochi::details::AddRowSparseTimesVector<NonConstScalar, NonConstIdx>(
            spanCol, spanVal, y.Col(jc), sum);
        y(ir, jc) -= sum;
      }
    }
    y.Row(ir) *= rVal[_uStart[ir]];
  }
}

} // namespace mochi::krylov
