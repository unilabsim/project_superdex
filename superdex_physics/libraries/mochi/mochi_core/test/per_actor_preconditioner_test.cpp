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

#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/solvers/island_operators.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <utility>

using namespace mochi;

namespace {

struct ActorOrderLog {
  std::mutex mutex;
  DynamicArray<int> actorIds;
};

struct ConcurrentCall {
  int workerId;
  int numWorkers;
  int rowBegin;
  int rowEnd;

  friend bool operator==(ConcurrentCall const&, ConcurrentCall const&) = default;
};

class RecordingActorPreconditioner final : public ActorPreconditioner<real> {
 public:
  RecordingActorPreconditioner(
      ActorPreconditionerParallelMode mode,
      int rowBlockSize,
      real scale,
      int barrierWaits = 0,
      ActorOrderLog* orderLog = nullptr,
      int actorId = 0)
      : _parallelism{mode, rowBlockSize},
        _scale(scale),
        _barrierWaits(barrierWaits),
        _orderLog(orderLog),
        _actorId(actorId) {}

  [[nodiscard]] ActorPreconditionerParallelism GetConcurrentSolveRequirements() const override {
    return _parallelism;
  }

  void Solve(ColumnVectorView<real const> x, ColumnVectorView<real> Px) const override {
    ++solveCalls;
    for (int row = 0; row < x.Rows(); ++row) {
      Px(row, 0) = _scale * x(row, 0);
    }
  }

  void ConcurrentSolve(
      ColumnVectorView<real const> x,
      ColumnVectorView<real> Px,
      ParallelWorkerInfo const& data) const override {
    {
      std::lock_guard lock(_callsMutex);
      _calls.push_back({data.workerId, data.numWorkers, data.rBegin, data.rEnd});
    }
    if (_orderLog != nullptr && data.workerId == 0) {
      std::lock_guard lock(_orderLog->mutex);
      _orderLog->actorIds.push_back(_actorId);
    }
    for (int wait = 0; wait < _barrierWaits; ++wait) {
      data.BarrierWait();
    }
    for (int row = data.rBegin; row < data.rEnd; ++row) {
      Px(row, 0) = _scale * x(row, 0);
    }
  }

  void Update(ActorPseudoMatrix<real> const&) override {}

  PreconditionerType GetType() const override {
    return PreconditionerType::Jacobi;
  }

  DynamicArray<ConcurrentCall> Calls() const {
    std::lock_guard lock(_callsMutex);
    return _calls;
  }

  mutable std::atomic<int> solveCalls{0};

 private:
  ActorPreconditionerParallelism _parallelism;
  real _scale;
  int _barrierWaits;
  ActorOrderLog* _orderLog;
  int _actorId;
  mutable std::mutex _callsMutex;
  mutable DynamicArray<ConcurrentCall> _calls;
};

} // namespace

template <typename Fn>
static void RunWithExactWorkers(int numWorkers, Fn const& fn) {
  DynamicArray<std::thread> threads;
  threads.reserve(numWorkers);
  for (int workerId = 0; workerId < numWorkers; ++workerId) {
    threads.emplace_back([&fn, workerId] { fn(workerId); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

static void RunConcurrentSolve(
    PerActorPrec<real> const& prec,
    ColumnVector<real> const& x,
    ColumnVector<real>& Px,
    DynamicArray<int> const& workerRowRanges,
    int repetitions = 1) {
  int const numWorkers = static_cast<int>(workerRowRanges.size()) - 1;
  ParallelBarrier barrier(numWorkers);
  RunWithExactWorkers(numWorkers, [&](int workerId) {
    auto workerBarrier = barrier;
    if (workerId == 0) {
      prec.PrepareConcurrentSolve(MakeConstSpan(workerRowRanges));
    }
    workerBarrier.Wait();
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      prec.ConcurrentSolve(
          x,
          Px,
          {workerId,
           numWorkers,
           workerRowRanges[workerId],
           workerRowRanges[workerId + 1],
           workerBarrier});
    }
  });
}

static DynamicArray<ConcurrentCall> SortedCalls(RecordingActorPreconditioner const& prec) {
  auto calls = prec.Calls();
  std::sort(calls.begin(), calls.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.rowBegin < rhs.rowBegin;
  });
  return calls;
}

TEST(PerActorPreconditioner, IndependentRowsPreserveAlignedMatVecRangesAcrossActors) {
  RecordingActorPreconditioner actor0(ActorPreconditionerParallelMode::IndependentRows, 2, 2_r);
  RecordingActorPreconditioner actor1(ActorPreconditionerParallelMode::IndependentRows, 2, 2_r);
  PerActorPrec<real> prec({
      {0, 6, actor0},
      {6, 6, actor1},
  });
  ColumnVector<real> x(12), Px(12);
  x.SetRandom(11);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 0, 4, 8, 12});

  EXPECT_EQ((DynamicArray<ConcurrentCall>{{1, 4, 0, 4}, {2, 4, 4, 6}}), SortedCalls(actor0));
  EXPECT_EQ((DynamicArray<ConcurrentCall>{{2, 4, 0, 2}, {3, 4, 2, 6}}), SortedCalls(actor1));
  ColumnVector<real> const expected = 2_r * x;
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
  EXPECT_EQ(0, actor0.solveCalls);
  EXPECT_EQ(0, actor1.solveCalls);
}

TEST(PerActorPreconditioner, IndependentRowsPreserveRowRangeLocality) {
  RecordingActorPreconditioner prefix(ActorPreconditionerParallelMode::IndependentRows, 1, 1_r);
  RecordingActorPreconditioner actor(ActorPreconditionerParallelMode::IndependentRows, 2, 2_r);
  PerActorPrec<real> prec({
      {0, 3, prefix},
      {3, 12, actor},
  });
  ColumnVector<real> x(15), Px(15), expected(15);
  x.SetRandom(18);
  Px.SetZero();
  expected.TopRows(3) = x.TopRows(3);
  expected.BottomRows(12) = 2_r * x.BottomRows(12);

  RunConcurrentSolve(prec, x, Px, {0, 3, 3, 8, 11, 15});

  EXPECT_EQ((DynamicArray<ConcurrentCall>{{0, 5, 0, 3}}), prefix.Calls());
  EXPECT_EQ(
      (DynamicArray<ConcurrentCall>{{2, 5, 0, 4}, {3, 5, 4, 8}, {4, 5, 8, 12}}),
      SortedCalls(actor));
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, IndependentRowsDoNotScheduleEmptyRanges) {
  RecordingActorPreconditioner actor(ActorPreconditionerParallelMode::IndependentRows, 3, 2_r);
  PerActorPrec<real> prec({ActorPrecApplyer<real>{0, 9, actor}});
  ColumnVector<real> x(9), Px(9), expected(9);
  x.SetRandom(19);
  Px.SetZero();
  expected = 2_r * x;

  RunConcurrentSolve(prec, x, Px, {0, 4, 4, 9});

  EXPECT_EQ((DynamicArray<ConcurrentCall>{{0, 3, 0, 3}, {2, 3, 3, 9}}), SortedCalls(actor));
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, IndependentRowsPlanningDoesNotScaleWithRowCount) {
  // A per-row or per-block planning loop would make this input impractical.
  int constexpr kNumRows = 2147483646;
  RecordingActorPreconditioner actor(ActorPreconditionerParallelMode::IndependentRows, 6, 1_r);
  PerActorPrec<real> prec({ActorPrecApplyer<real>{0, kNumRows, actor}});
  DynamicArray<int> const workerRowRanges{0, 536870911, 1073741823, 1610612734, kNumRows};

  prec.PrepareConcurrentSolve(MakeConstSpan(workerRowRanges));

  EXPECT_TRUE(actor.Calls().empty());
}

TEST(PerActorPreconditioner, SingleWorkerCallsOneWholeSolve) {
  RecordingActorPreconditioner actor(ActorPreconditionerParallelMode::SingleWorker, 0, 3_r);
  PerActorPrec<real> prec({ActorPrecApplyer<real>{0, 6, actor}});
  ColumnVector<real> x(6), Px(6);
  x.SetRandom(12);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 2, 4, 6});

  EXPECT_EQ(1, actor.solveCalls);
  EXPECT_TRUE(actor.Calls().empty());
  ColumnVector<real> const expected = 3_r * x;
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, SynchronizedTeamUsesDenseIdsAndReusableBarrier) {
  RecordingActorPreconditioner actor(
      ActorPreconditionerParallelMode::SynchronizedTeam,
      2,
      4_r,
      /*barrierWaits*/ 2);
  RecordingActorPreconditioner singleWorkerTeam(
      ActorPreconditionerParallelMode::SynchronizedTeam,
      2,
      5_r,
      /*barrierWaits*/ 2);
  PerActorPrec<real> prec({
      {0, 10, actor},
      {10, 2, singleWorkerTeam},
  });
  ColumnVector<real> x(12), Px(12), expected(12);
  x.SetRandom(13);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 4, 12}, /*repetitions*/ 8);

  auto const calls = actor.Calls();
  ASSERT_FALSE(calls.empty());
  int const teamSize = calls.front().numWorkers;
  ASSERT_GT(teamSize, 0);
  EXPECT_EQ(2, teamSize);
  EXPECT_EQ(static_cast<size_t>(teamSize * 8), calls.size());
  DynamicArray<ConcurrentCall> laneCalls;
  for (int workerId = 0; workerId < teamSize; ++workerId) {
    auto const call = std::find_if(calls.begin(), calls.end(), [&](ConcurrentCall const& value) {
      return value.workerId == workerId;
    });
    ASSERT_NE(calls.end(), call);
    EXPECT_EQ(8, std::count(calls.begin(), calls.end(), *call));
    laneCalls.push_back(*call);
  }
  std::sort(laneCalls.begin(), laneCalls.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.rowBegin < rhs.rowBegin;
  });
  EXPECT_EQ(0, laneCalls.front().rowBegin);
  EXPECT_EQ(10, laneCalls.back().rowEnd);
  for (int i = 0; i < static_cast<int>(laneCalls.size()); ++i) {
    EXPECT_EQ(teamSize, laneCalls[i].numWorkers);
    EXPECT_EQ(0, laneCalls[i].rowBegin % 2);
    EXPECT_EQ(0, laneCalls[i].rowEnd % 2);
    EXPECT_LT(laneCalls[i].rowBegin, laneCalls[i].rowEnd);
    if (i > 0) {
      EXPECT_EQ(laneCalls[i - 1].rowEnd, laneCalls[i].rowBegin);
    }
  }
  auto const singleWorkerCalls = singleWorkerTeam.Calls();
  EXPECT_EQ(8, singleWorkerCalls.size());
  for (auto const& call : singleWorkerCalls) {
    EXPECT_EQ((ConcurrentCall{0, 1, 0, 2}), call);
  }
  expected.TopRows(10) = 4_r * x.TopRows(10);
  expected.BottomRows(2) = 5_r * x.BottomRows(2);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, OverlappingSynchronizedTeamsCompleteWithoutDeadlock) {
  ActorOrderLog orderLog;
  RecordingActorPreconditioner actor0(
      ActorPreconditionerParallelMode::SynchronizedTeam, 2, 2_r, 2, &orderLog, 0);
  RecordingActorPreconditioner actor1(
      ActorPreconditionerParallelMode::SynchronizedTeam, 2, 3_r, 2, &orderLog, 1);
  RecordingActorPreconditioner actor2(
      ActorPreconditionerParallelMode::SynchronizedTeam, 2, 4_r, 2, &orderLog, 2);
  PerActorPrec<real> prec({
      {0, 4, actor0},
      {4, 4, actor1},
      {8, 4, actor2},
  });
  ColumnVector<real> x(12), Px(12), expected(12);
  x.SetRandom(14);
  Px.SetZero();
  expected.TopRows(4) = 2_r * x.TopRows(4);
  expected.MiddleRows(4, 4) = 3_r * x.MiddleRows(4, 4);
  expected.BottomRows(4) = 4_r * x.BottomRows(4);

  RunConcurrentSolve(prec, x, Px, {0, 3, 6, 9, 12}, /*repetitions*/ 8);

  EXPECT_EQ(16, actor0.Calls().size());
  EXPECT_EQ(16, actor1.Calls().size());
  EXPECT_EQ(16, actor2.Calls().size());
  DynamicArray<int> const expectedOrder{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                                        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  EXPECT_EQ(expectedOrder, orderLog.actorIds);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, MixedModesMatchSerialSolve) {
  RecordingActorPreconditioner independent(
      ActorPreconditionerParallelMode::IndependentRows, 2, 2_r);
  RecordingActorPreconditioner single(ActorPreconditionerParallelMode::SingleWorker, 1, 3_r);
  RecordingActorPreconditioner synchronized(
      ActorPreconditionerParallelMode::SynchronizedTeam, 2, 4_r, 2);
  PerActorPrec<real> prec({
      {0, 4, independent},
      {4, 2, single},
      {6, 6, synchronized},
  });
  ColumnVector<real> x(12), parallelResult(12), serialResult(12);
  x.SetRandom(15);
  parallelResult.SetZero();
  serialResult.SetZero();

  prec.Solve(x, serialResult);
  RunConcurrentSolve(prec, x, parallelResult, {0, 3, 6, 9, 12});

  EXPECT_TRUE(mochi::test::NearEqualMatrices(parallelResult, serialResult, real{0}));
  EXPECT_EQ(2, single.solveCalls);
  EXPECT_FALSE(independent.Calls().empty());
  EXPECT_FALSE(synchronized.Calls().empty());
}

TEST(PerActorPreconditioner, RepreparesForDifferentWorkerCount) {
  RecordingActorPreconditioner actor(ActorPreconditionerParallelMode::IndependentRows, 2, 2_r);
  PerActorPrec<real> prec({ActorPrecApplyer<real>{0, 8, actor}});
  ColumnVector<real> x(8), Px(8);
  x.SetRandom(16);
  ColumnVector<real> const expected = 2_r * x;

  Px.SetZero();
  RunConcurrentSolve(prec, x, Px, {0, 4, 8});
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));

  Px.SetZero();
  RunConcurrentSolve(prec, x, Px, {0, 0, 4, 8});
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));

  auto const calls = actor.Calls();
  EXPECT_TRUE(std::any_of(calls.begin(), calls.end(), [](ConcurrentCall const& call) {
    return call.numWorkers == 2;
  }));
  EXPECT_TRUE(std::any_of(calls.begin(), calls.end(), [](ConcurrentCall const& call) {
    return call.numWorkers == 3;
  }));
}

TEST(PerActorPreconditioner, BuiltInPreconditionersMatchSerialExecution) {
  int constexpr kBlockSize = 3;
  int constexpr kActorSize = 2 * kBlockSize;

  auto denseActorMatrix = Matrix<real>::Zero(kActorSize, kActorSize);
  for (int i = 0; i < kBlockSize; ++i) {
    denseActorMatrix(i, i) = 4_r;
    denseActorMatrix(i + kBlockSize, i + kBlockSize) = 4_r;
    denseActorMatrix(i, i + kBlockSize) = -1_r;
    denseActorMatrix(i + kBlockSize, i) = -1_r;
  }
  auto actorMatrix = ToBlockSparseMatrix<kBlockSize>(denseActorMatrix, true);
  AnyMatrixView<real const> const actorMatrixView{AsConstView(actorMatrix)};

  // ILU0 uses one worker. Colored SSOR and AMG use synchronized teams.
  ActorPseudoMatrix<real> const ilu0Matrix{0, actorMatrixView, {}};
  ActorPseudoMatrix<real> const coloredSSORMatrix{kActorSize, actorMatrixView, {}};
  ActorPseudoMatrix<real> const amgMatrix{2 * kActorSize, actorMatrixView, {}};
  ILU0ActorPrec<real, kBlockSize> ilu0Prec{ilu0Matrix};
  ColoredSSORActorPrec<real, kBlockSize> coloredSSORPrec{coloredSSORMatrix};
  AMGActorPrec<real, kBlockSize> amgPrec{amgMatrix};

  ASSERT_EQ(
      ActorPreconditionerParallelMode::SingleWorker,
      ilu0Prec.GetConcurrentSolveRequirements().mode);
  ASSERT_EQ(
      ActorPreconditionerParallelMode::SynchronizedTeam,
      coloredSSORPrec.GetConcurrentSolveRequirements().mode);
  ASSERT_EQ(kBlockSize, coloredSSORPrec.GetConcurrentSolveRequirements().rowBlockSize);
  ASSERT_EQ(
      ActorPreconditionerParallelMode::SynchronizedTeam,
      amgPrec.GetConcurrentSolveRequirements().mode);
  ASSERT_EQ(kBlockSize, amgPrec.GetConcurrentSolveRequirements().rowBlockSize);

  PerActorPrec<real> prec({
      {0, kActorSize, ilu0Prec},
      {kActorSize, kActorSize, coloredSSORPrec},
      {2 * kActorSize, kActorSize, amgPrec},
  });

  ColumnVector<real> x(3 * kActorSize);
  ColumnVector<real> serialResult(3 * kActorSize);
  ColumnVector<real> parallelResult(3 * kActorSize);
  x.SetRandom(17);
  serialResult.SetZero();
  prec.Solve(x, serialResult);

  auto expectPlanMatchesSerial = [&](DynamicArray<int> const& workerRowRanges) {
    parallelResult.SetZero();
    RunConcurrentSolve(prec, x, parallelResult, workerRowRanges);
    EXPECT_TRUE(mochi::test::NearEqualMatrices(serialResult, parallelResult, 1e-5_r));
  };

  expectPlanMatchesSerial({0, 3, 6, 9, 12, 15, 18});
  expectPlanMatchesSerial({0, 6, 12, 18});
}
