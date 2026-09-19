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

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace mochi::krylov::details {

/**
 * @brief Compute exact natural-order Cholesky completion of a symmetric graph.
 *
 * @pre The graph is square and symmetric, with sorted unique rows and a diagonal entry in every
 * row.
 * @warning Exact completion can be dense, so callers must establish suitable problem-specific size
 * bounds.
 */
template <typename Tg, typename Ptr, template <typename, typename...> typename Storage>
Graph<int, int> CompleteCholeskyPatternNaturalOrder(Graph<Tg, Ptr, Storage> const& sparsity) {
  static_assert(
      std::is_same_v<std::remove_const_t<Tg>, int> && std::is_same_v<std::remove_const_t<Ptr>, int>,
      "Cholesky completion currently requires int graph indices and pointers.");
  auto const pointers = sparsity.GetPointers();
  auto const targets = sparsity.GetTargets();
  if (pointers.empty()) {
    MOCHI_ASSERT_VERBOSE(targets.empty(), "A graph without row pointers cannot contain targets.");
    return {};
  }

  [[maybe_unused]] auto constexpr kMaxCount = static_cast<size_t>(std::numeric_limits<int>::max());
  MOCHI_ASSERT_VERBOSE(
      pointers.size() - 1 <= kMaxCount && targets.size() <= kMaxCount,
      "Cholesky completion exceeds the supported int graph size.");
  int const numRows = static_cast<int>(pointers.size() - 1);
  [[maybe_unused]] int const numTargets = static_cast<int>(targets.size());
  MOCHI_ASSERT_VERBOSE(
      pointers.front() == 0 && pointers.back() == numTargets, "Invalid graph pointers.");

  DynamicArray<DynamicArray<int>> rows(static_cast<size_t>(numRows));
  for (int row = 0; row < numRows; ++row) {
    int const begin = pointers[row];
    int const end = pointers[row + 1];
    MOCHI_ASSERT_VERBOSE(
        begin >= 0 && begin <= end && end <= numTargets, "Invalid graph row pointers.");
    rows[row].assign(targets.begin() + begin, targets.begin() + end);
    MOCHI_ASSERT_VERBOSE(
        std::ranges::is_sorted(rows[row]) &&
            std::ranges::adjacent_find(rows[row]) == rows[row].end(),
        "Graph rows must be sorted and contain no duplicates.");
    MOCHI_ASSERT_VERBOSE(
        !rows[row].empty() && rows[row].front() >= 0 && rows[row].back() < numRows,
        "Graph targets must be valid row indices.");
    MOCHI_ASSERT_VERBOSE(
        std::ranges::binary_search(rows[row], row), "Every graph row must contain its diagonal.");
  }

#if MOCHI_ASSERT_VERBOSE_ENABLED
  for (int row = 0; row < numRows; ++row) {
    for (int col : rows[row]) {
      MOCHI_ASSERT_VERBOSE(
          std::ranges::binary_search(rows[col], row),
          "The Cholesky input graph must be symmetric.");
    }
  }
#endif

  size_t targetCount = targets.size();
  for (int eliminated = 0; eliminated < numRows; ++eliminated) {
    auto const& eliminatedRow = rows[eliminated];
    int const firstHigher = static_cast<int>(
        std::ranges::upper_bound(eliminatedRow, eliminated) - eliminatedRow.begin());
    for (int a = firstHigher; a < isize(eliminatedRow); ++a) {
      int const row = eliminatedRow[a];
      for (int b = a + 1; b < isize(eliminatedRow); ++b) {
        int const col = eliminatedRow[b];
        auto* rowPosition = std::ranges::lower_bound(rows[row], col);
        if (rowPosition == rows[row].end() || *rowPosition != col) {
          auto* colPosition = std::ranges::lower_bound(rows[col], row);
          auto const rowOffset = rowPosition - rows[row].begin();
          auto const colOffset = colPosition - rows[col].begin();
          MOCHI_ASSERT_VERBOSE(
              colPosition == rows[col].end() || *colPosition != row,
              "Internal Cholesky completion symmetry failure.");
          MOCHI_ASSERT_VERBOSE(
              targetCount <= kMaxCount - 2, "Completed Cholesky graph exceeds int capacity.");
          rows[row].push_back(col);
          std::rotate(rows[row].begin() + rowOffset, rows[row].end() - 1, rows[row].end());
          rows[col].push_back(row);
          std::rotate(rows[col].begin() + colOffset, rows[col].end() - 1, rows[col].end());
          targetCount += 2;
        }
      }
    }
  }

  DynamicArray<int> completedPointers;
  completedPointers.reserve(static_cast<size_t>(numRows) + 1);
  completedPointers.push_back(0);
  DynamicArray<int> completedTargets;
  completedTargets.reserve(targetCount);
  for (auto const& row : rows) {
    completedTargets.append(row);
    completedPointers.push_back(static_cast<int>(completedTargets.size()));
  }
  return Graph<int, int>{std::move(completedPointers), std::move(completedTargets)};
}

} // namespace mochi::krylov::details
