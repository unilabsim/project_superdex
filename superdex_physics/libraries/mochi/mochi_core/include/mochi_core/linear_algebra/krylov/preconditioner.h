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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/linear_solver_params.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/task_scheduler.h>

#include <type_traits>

namespace mochi {

struct ParallelWorkerInfo {
  /// @brief Zero-based index of the current worker within this worker group.
  int workerId;
  /// @brief Number of workers in this worker group.
  int numWorkers;
  /// @brief First assigned row, relative to the vectors passed to the operation.
  int rBegin;
  /// @brief One past the last assigned row, relative to the vectors passed to the operation.
  int rEnd;
  /// @brief Barrier shared by this worker group.
  ParallelBarrier const& barrier;

  /// @brief Wait for all the workers to reach the barrier point.
  void BarrierWait() const {
    barrier.Wait();
  }
};

template <typename Scalar>
struct Preconditioner {
  static_assert(!std::is_const_v<Scalar>, "Scalar type must be non-const");

  virtual ~Preconditioner() = default;

  /// @brief Application of the preconditioner to a column vector.
  virtual void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<Scalar> Px) const = 0;

  /// @brief Concurrent application of the preconditioner to a column vector by a pool of workers.
  /// The calling worker is responsible for applying its own preconditioner contribution.
  /// @param[in] x Input column vector.
  /// @param[out] Px Output column vector.
  /// @param[in] data Parallel information for the calling worker.
  /// @note It's the responsibility of the caller to ensure each worker is allocated to a different
  /// thread, which can be accomplished through TaskScheduler::BatchEnqueueOnAvailableWorkers.
  /// @note `[data.rBegin, data.rEnd)` is the matrix-vector-product row range assigned to the
  /// calling worker. If it writes a row assigned to another worker, the implementation must ensure
  /// the write is complete and visible before that other worker returns from @ref ConcurrentSolve.
  virtual void ConcurrentSolve(
      ColumnVectorView<Scalar const> /*x*/,
      ColumnVectorView<Scalar> /*Px*/,
      ParallelWorkerInfo const& /*data*/) const {
    MOCHI_ASSERT(false, "Parallel solve not supported for this preconditioner.");
  }

  /// @brief Get the preconditioner type.
  virtual constexpr PreconditionerType GetType() const = 0;

  /// @brief Prepare for concurrent application by a pool of workers.
  /// @param[in] workerRowRanges Boundaries of the workers' matrix-vector-product row partitions.
  /// For N > 0 workers, the span contains N + 1 nondecreasing entries. The first is zero, the last
  /// is the number of rows processed by the preconditioner, and equal adjacent entries represent
  /// empty partitions.
  /// @note The partitions are locality hints. Preconditioners may distribute rows differently.
  /// @note The caller must invoke this method once on a single thread and synchronize the workers
  /// before calling @ref ConcurrentSolve.
  /// @note Implementations must not retain @p workerRowRanges.
  /// @note Prepared state remains valid until this function is called again, the configuration
  /// changes, or the instance is moved from.
  /// @warning An instance must not participate in multiple linear solves concurrently.
  virtual void PrepareConcurrentSolve(Span<int const> /*workerRowRanges*/) const {}

  /// @brief Validates the input and output vectors for preconditioner application.
  ///
  /// @tparam InputType Type of the input vector/matrix (must provide data(), Rows(), Cols()).
  /// @tparam OutputType Type of the output vector/matrix (must provide data(), Rows(), Cols()).
  /// @param[in] precN Expected dimension of the preconditioner
  /// @param[in] x Input column vector(s) to validate.
  /// @param[in] Px Output column vector(s) to validate.
  /// @note Parameters are marked [[maybe_unused]] because checks are only active in debug builds.
  template <typename InputType, typename OutputType>
  static void ValidateInputOutput(
      [[maybe_unused]] int precN,
      [[maybe_unused]] InputType const& x,
      [[maybe_unused]] OutputType& Px) {
    static_assert(
        std::is_same_v<std::remove_pointer_t<decltype(x.data())> const, Scalar const>,
        "Inconsistent scalar types");
    static_assert(
        std::is_same_v<std::remove_pointer_t<decltype(Px.data())>, Scalar>,
        "Inconsistent scalar types");
    MOCHI_ASSERT_VERBOSE(x.Rows() == precN, "Incompatible size of input vector");
    MOCHI_ASSERT_VERBOSE(Px.Rows() == precN, "Incompatible size of output vector");
    MOCHI_ASSERT_VERBOSE(x.Cols() == Px.Cols(), "Incompatible number of columns");
  }
};

} // namespace mochi
