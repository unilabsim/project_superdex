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

#include "pybind_core.h"

#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/log.h>

#include <exception>
#include <utility>
#include <vector>

namespace mochi {

// Out-of-line key function: anchors MochiErrorException's vtable and typeinfo to this TU, so
// the exported RTTI lives in the shared pybind-core library rather than being duplicated per
// extension.
MochiErrorException::~MochiErrorException() = default;

namespace {
Guarded<Context*> g_context{nullptr};

void DestroyGlobalContextLocked(Context*& context) {
  if (!context) {
    throw std::runtime_error("Mochi is not currently initialized.");
  }
  // Before destroying the context, clear the log callback. It may hold a reference to a
  // Python function that the user didn't release; otherwise, the interpreter can crash on exit
  // with: "Fatal Python error: gilstate_tss_set: failed to set current tstate (TSS)".
  context->SetLogCallback(nullptr);
  mochi::DestroyContext(context);
  context = nullptr;
}
} // namespace

void InitGlobalContext(int numWorkerThreads) {
  g_context.Mutate([numWorkerThreads](Context*& context) {
    if (context) {
      throw std::runtime_error("Mochi has already been initialized.");
    }
    context = mochi::CreateContext(numWorkerThreads);
  });
}

Context* GetContext() {
  // Shutdown holds this guard while dependent teardown callbacks run. Those callbacks may query
  // the still-live context, so this snapshot remains lock-free under the documented requirement
  // that lifecycle operations do not race ordinary binding calls.
  return g_context.UnsafeRead([](Context* context) { return context; });
}

void DestroyGlobalContext() {
  g_context.Mutate(DestroyGlobalContextLocked);
}

void CheckContext() {
  if (!GetContext()) {
    throw std::runtime_error("Please call mochi.initialize(num_worker_threads) first.");
  }
}

namespace {
// Function-local static so registration works regardless of static-initialization order across
// the shared-library boundary. Intentionally never cleared (see RegisterContextDependent).
Guarded<std::vector<std::function<void()>>>& ContextDependents() {
  static Guarded<std::vector<std::function<void()>>> dependents;
  return dependents;
}
} // namespace

void RegisterContextDependent(std::function<void()> teardown) {
  ContextDependents().Mutate([&teardown](std::vector<std::function<void()>>& dependents) {
    dependents.push_back(std::move(teardown));
  });
}

void RunContextDependentTeardowns() {
  auto dependents = ContextDependents().Load();
  for (auto it = dependents.rbegin(); it != dependents.rend(); ++it) {
    try {
      (*it)();
    } catch (std::exception const& e) {
      MOCHI_LOG_ERROR("A context-dependent teardown raised an exception: %s", e.what());
    } catch (...) {
      MOCHI_LOG_ERROR("A context-dependent teardown raised an unknown exception.");
    }
  }
}

void ShutdownGlobalContext() {
  g_context.Mutate([](Context*& context) {
    RunContextDependentTeardowns();
    DestroyGlobalContextLocked(context);
  });
}

} // namespace mochi
