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

#include "pybind_helpers.h"

#include <mochi_physics/pybind/core/pybind_core.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mochi {
namespace {

namespace py = pybind11;

std::mutex &LeasedScenesMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_set<Scene *> &LeasedScenes() {
  static std::unordered_set<Scene *> scenes;
  return scenes;
}

void RegisterLeasedScenes(std::vector<Scene *> const &scenes) {
  std::lock_guard lock(LeasedScenesMutex());
  for (auto *scene : scenes) {
    if (LeasedScenes().contains(scene)) {
      throw std::runtime_error(
          "a scene is already owned by a SceneBatchExecutor");
    }
  }
  LeasedScenes().insert(scenes.begin(), scenes.end());
}

void ReleaseLeasedScenes(std::vector<Scene *> const &scenes) {
  std::lock_guard lock(LeasedScenesMutex());
  for (auto *scene : scenes) {
    LeasedScenes().erase(scene);
  }
}

bool IsLeasedScene(Scene const *scene) {
  std::lock_guard lock(LeasedScenesMutex());
  return LeasedScenes().contains(const_cast<Scene *>(scene));
}

class SceneBatchExecutor
    : public std::enable_shared_from_this<SceneBatchExecutor> {
public:
  SceneBatchExecutor(py::sequence scenes, py::sequence actors,
                     size_t numWorkers) {
    CheckContext();
    if (numWorkers == 0) {
      throw std::invalid_argument(
          "SceneBatchExecutor requires num_workers >= 1");
    }
    if (py::len(scenes) == 0 || py::len(scenes) != py::len(actors)) {
      throw std::invalid_argument(
          "scenes and actors must be non-empty and have equal length");
    }
    numWorkers = std::min(numWorkers, static_cast<size_t>(py::len(scenes)));
    if (GetContext()->GetNumThreads() != 0) {
      throw std::invalid_argument(
          "SceneBatchExecutor requires initialize(num_worker_threads=0); "
          "do not combine outer scene workers with SDK worker threads");
    }

    std::unordered_set<Scene *> uniqueScenes;
    std::unordered_set<Actor *> uniqueActors;
    int dofCount = -1;
    for (size_t i = 0; i < static_cast<size_t>(py::len(scenes)); ++i) {
      auto scene = py::cast<Scene *>(scenes[i]);
      auto actor = py::cast<Actor *>(actors[i]);
      if (!scene || !actor || scene->GetContext() != GetContext() ||
          actor->GetContext() != GetContext() || actor->GetScene() != scene) {
        throw std::invalid_argument(
            "each actor must belong to its corresponding live scene");
      }
      if (!uniqueScenes.insert(scene).second ||
          !uniqueActors.insert(actor).second) {
        throw std::invalid_argument(
            "SceneBatchExecutor requires unique scenes and actors");
      }
      int const thisDofCount = actor->GetNumDofs();
      if (thisDofCount <= 0) {
        throw std::invalid_argument(
            "SceneBatchExecutor requires articulated actors with DoFs");
      }
      if (dofCount >= 0 && thisDofCount != dofCount) {
        throw std::invalid_argument(
            "all actors must have the same number of DoFs");
      }
      dofCount = thisDofCount;
      _scenes.push_back(scene);
      _actors.push_back(actor);
    }
    _dofCount = dofCount;
    _dofIndices.resize(static_cast<size_t>(_dofCount));
    for (int i = 0; i < _dofCount; ++i) {
      _dofIndices[static_cast<size_t>(i)] = i;
    }

    _workers.reserve(numWorkers);
    RegisterLeasedScenes(_scenes);
    _leasesRegistered = true;
    try {
      for (size_t i = 0; i < numWorkers; ++i) {
        _workers.emplace_back(&SceneBatchExecutor::WorkerMain, this);
      }
    } catch (...) {
      {
        std::lock_guard lock(_mutex);
        _stopping = true;
      }
      _workAvailable.notify_all();
      for (auto &worker : _workers) {
        if (worker.joinable()) {
          worker.join();
        }
      }
      ReleaseLeasedScenes(_scenes);
      _leasesRegistered = false;
      throw;
    }
  }

  ~SceneBatchExecutor() { Close(); }

  SceneBatchExecutor(SceneBatchExecutor const &) = delete;
  SceneBatchExecutor &operator=(SceneBatchExecutor const &) = delete;

  [[nodiscard]] size_t GetNumWorkers() const { return _workers.size(); }
  [[nodiscard]] size_t GetNumScenes() const { return _scenes.size(); }
  [[nodiscard]] int GetNumDofs() const { return _dofCount; }
  [[nodiscard]] bool IsClosed() const { return _closed; }

  void Step(double timeStepSec,
            py::array_t<real, py::array::c_style> generalizedForces,
            py::array_t<real, py::array::c_style> qposOut,
            py::array_t<real, py::array::c_style> qvelOut) {
    CheckContext();
    if (!std::isfinite(timeStepSec) || timeStepSec < 0) {
      throw std::invalid_argument(
          "time_step_sec must be finite and non-negative");
    }
    ValidateArray("generalized_forces", generalizedForces);
    ValidateArray("qpos_out", qposOut);
    ValidateArray("qvel_out", qvelOut);

    std::unique_lock callLock(_callMutex);
    if (_closed) {
      throw std::runtime_error("SceneBatchExecutor is closed");
    }
    for (size_t i = 0; i < _scenes.size(); ++i) {
      if (!GetContext()->IsValidScene(_scenes[i]) ||
          _actors[i]->GetScene() != _scenes[i]) {
        throw std::runtime_error(
            "a scene or actor was destroyed while its executor is active");
      }
    }

    auto const *forceData = generalizedForces.data();
    auto *qposData = qposOut.mutable_data();
    auto *qvelData = qvelOut.mutable_data();
    std::exception_ptr failure;
    {
      // Keep all pybind arrays alive with the GIL held. Only the pure C++
      // scheduler barrier may execute without it.
      py::gil_scoped_release release;
      failure = DispatchAndWait(timeStepSec, forceData, qposData, qvelData);
    }
    if (failure) {
      std::rethrow_exception(failure);
    }
  }

  void Close() {
    std::unique_lock callLock(_callMutex);
    if (_closed) {
      return;
    }
    {
      std::lock_guard lock(_mutex);
      _stopping = true;
    }
    _workAvailable.notify_all();
    for (auto &worker : _workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    _workers.clear();
    if (_leasesRegistered) {
      ReleaseLeasedScenes(_scenes);
      _leasesRegistered = false;
    }
    _closed = true;
  }

private:
  std::exception_ptr DispatchAndWait(double timeStepSec, real const* forceData,
                                     real* qposData, real* qvelData) {
    {
      std::lock_guard lock(_mutex);
      _failure = nullptr;
      _pending = _scenes.size();
      for (size_t i = 0; i < _scenes.size(); ++i) {
        _jobs.emplace_back(
            [this, i, timeStepSec, forceData, qposData, qvelData]() {
              auto const offset = i * static_cast<size_t>(_dofCount);
              Error error;
              _actors[i]->SetExternalForcesOnDofs(
                  MakeConstSpan(_dofIndices),
                  Span<real const>(forceData + offset,
                                   static_cast<size_t>(_dofCount)),
                  error);
              if (!error.IsOK()) {
                throw MochiErrorException(error);
              }
              _scenes[i]->Step(timeStepSec);
              _actors[i]->GetArticulatedPose(
                  Span<real>(qposData + offset, static_cast<size_t>(_dofCount)),
                  error);
              if (!error.IsOK()) {
                throw MochiErrorException(error);
              }
              _actors[i]->GetArticulatedJointVelocities(
                  Span<real>(qvelData + offset, static_cast<size_t>(_dofCount)),
                  error);
              if (!error.IsOK()) {
                throw MochiErrorException(error);
              }
            });
      }
    }
    _workAvailable.notify_all();
    std::unique_lock lock(_mutex);
    _allDone.wait(lock, [this]() { return _pending == 0; });
    return _failure;
  }

  void ValidateArray(char const *name,
                     py::array_t<real, py::array::c_style> const &array) const {
    if (array.ndim() != 2 ||
        array.shape(0) != static_cast<py::ssize_t>(_scenes.size()) ||
        array.shape(1) != _dofCount) {
      throw std::invalid_argument(std::string(name) +
                                  " must have shape [num_scenes, num_dofs]");
    }
  }

  void WorkerMain() {
    auto *context = GetContext();
    context->BindThisThread();
    while (true) {
      std::function<void()> job;
      {
        std::unique_lock lock(_mutex);
        _workAvailable.wait(lock,
                            [this]() { return _stopping || !_jobs.empty(); });
        if (_stopping && _jobs.empty()) {
          break;
        }
        job = std::move(_jobs.front());
        _jobs.pop_front();
      }
      try {
        job();
      } catch (...) {
        std::lock_guard lock(_mutex);
        if (!_failure) {
          _failure = std::current_exception();
        }
      }
      {
        std::lock_guard lock(_mutex);
        --_pending;
        if (_pending == 0) {
          _allDone.notify_one();
        }
      }
    }
    context->UnbindThisThread();
  }

  std::vector<Scene *> _scenes;
  std::vector<Actor *> _actors;
  std::vector<int> _dofIndices;
  int _dofCount = 0;
  std::vector<std::thread> _workers;
  std::deque<std::function<void()>> _jobs;
  mutable std::mutex _mutex;
  std::mutex _callMutex;
  std::condition_variable _workAvailable;
  std::condition_variable _allDone;
  std::exception_ptr _failure;
  size_t _pending = 0;
  bool _stopping = false;
  bool _closed = false;
  bool _leasesRegistered = false;
};

} // namespace

void DefineSceneBatchExecutor(py::module_ &m) {
  py::class_<SceneBatchExecutor, std::shared_ptr<SceneBatchExecutor>>(
      m, "SceneBatchExecutor")
      .def(py::init(
               [](py::sequence scenes, py::sequence actors, size_t numWorkers) {
                 auto executor = std::make_shared<SceneBatchExecutor>(
                     scenes, actors, numWorkers);
                 RegisterContextDependent(
                     [weak = std::weak_ptr<SceneBatchExecutor>(executor)]() {
                       if (auto active = weak.lock()) {
                         active->Close();
                       }
                     });
                 return executor;
               }),
           py::arg("scenes"), py::arg("actors"), py::arg("num_workers"))
      .def_property_readonly("num_workers", &SceneBatchExecutor::GetNumWorkers)
      .def_property_readonly("num_scenes", &SceneBatchExecutor::GetNumScenes)
      .def_property_readonly("num_dofs", &SceneBatchExecutor::GetNumDofs)
      .def_property_readonly("closed", &SceneBatchExecutor::IsClosed)
      .def(
          "step",
          [](SceneBatchExecutor& self, double timeStepSec,
             py::array_t<real, py::array::c_style> generalizedForces,
             py::array_t<real, py::array::c_style> qposOut,
             py::array_t<real, py::array::c_style> qvelOut) {
            self.Step(timeStepSec, generalizedForces, qposOut, qvelOut);
          },
          py::arg("time_step_sec"), py::arg("generalized_forces"),
          py::arg("qpos_out"), py::arg("qvel_out"))
      .def("close", [](SceneBatchExecutor& self) {
        py::gil_scoped_release release;
        self.Close();
      })
      .def(
          "__enter__",
          [](SceneBatchExecutor &self) -> SceneBatchExecutor & { return self; })
      .def("__exit__", [](SceneBatchExecutor& self, py::object, py::object,
                          py::object) {
        py::gil_scoped_release release;
        self.Close();
      });
}

void OverrideLeasedSceneDestroy(py::module_ &m) {
  py::delattr(m, "destroy_scene");
  m.def(
      "destroy_scene",
      [](Scene *scene) {
        CheckContext();
        if (scene && IsLeasedScene(scene)) {
          throw std::runtime_error(
              "close SceneBatchExecutor before destroying one of its scenes");
        }
        GetContext()->DestroyScene(scene);
      },
      py::arg("scene"));
}

void OverrideLeasedSceneCallbacks(py::module_& m) {
  auto sceneClass = m.attr("Scene");
  auto registerCallback = [](Scene& scene, std::string_view debugName,
                             std::function<void(StepInfo const&)> callback, int priority,
                             bool preStep) {
    if (IsLeasedScene(&scene)) {
      throw std::runtime_error(
          "SceneBatchExecutor scenes do not support Python step callbacks");
    }
    return preStep ? scene.RegisterPreStepCallback(debugName, std::move(callback), priority)
                   : scene.RegisterPostStepCallback(debugName, std::move(callback), priority);
  };
  py::delattr(sceneClass, "register_pre_step_callback");
  py::delattr(sceneClass, "register_post_step_callback");
  sceneClass.attr("register_pre_step_callback") = py::cpp_function(
      [registerCallback](Scene& scene, std::string_view debugName,
                         std::function<void(StepInfo const&)> callback, int priority) {
        return registerCallback(scene, debugName, std::move(callback), priority, true);
      },
      py::is_method(sceneClass));
  sceneClass.attr("register_post_step_callback") = py::cpp_function(
      [registerCallback](Scene& scene, std::string_view debugName,
                         std::function<void(StepInfo const&)> callback, int priority) {
        return registerCallback(scene, debugName, std::move(callback), priority, false);
      },
      py::is_method(sceneClass));
}

} // namespace mochi
