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

#include <mochi_physics/pybind/core/pybind_core.h>

#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace mochi::detail {

inline std::mutex &LeasedScenesMutex() {
  static std::mutex mutex;
  return mutex;
}

inline std::unordered_set<Scene *> &LeasedScenes() {
  static std::unordered_set<Scene *> scenes;
  return scenes;
}

inline void RegisterLeasedScenes(std::vector<Scene *> const &scenes) {
  std::lock_guard lock(LeasedScenesMutex());
  for (auto *scene : scenes) {
    if (LeasedScenes().contains(scene)) {
      throw std::runtime_error(
          "a scene is already owned by a SceneBatchExecutor");
    }
  }
  LeasedScenes().insert(scenes.begin(), scenes.end());
}

inline void ReleaseLeasedScenes(std::vector<Scene *> const &scenes) {
  std::lock_guard lock(LeasedScenesMutex());
  for (auto *scene : scenes) {
    LeasedScenes().erase(scene);
  }
}

inline bool IsLeasedScene(Scene const *scene) {
  std::lock_guard lock(LeasedScenesMutex());
  return LeasedScenes().contains(const_cast<Scene *>(scene));
}

} // namespace mochi::detail
