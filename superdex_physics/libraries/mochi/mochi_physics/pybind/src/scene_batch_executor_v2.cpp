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
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "scene_batch_executor_state.h"

namespace mochi {
namespace {

namespace nb = nanobind;

using RealArray = nb::ndarray<nb::numpy, real, nb::c_contig, nb::device::cpu>;
using RealConstArray =
    nb::ndarray<nb::numpy, real const, nb::c_contig, nb::device::cpu>;
using IntConstArray =
    nb::ndarray<nb::numpy, int const, nb::c_contig, nb::device::cpu>;
using Uint8Array =
    nb::ndarray<nb::numpy, uint8_t, nb::c_contig, nb::device::cpu>;
using Uint8ConstArray =
    nb::ndarray<nb::numpy, uint8_t const, nb::c_contig, nb::device::cpu>;

class SceneBatchExecutorV2 {
public:
  static constexpr uint32_t kReadQpos = 1u << 0;
  static constexpr uint32_t kReadQvel = 1u << 1;
  static constexpr uint32_t kReadLinks = 1u << 2;
  static constexpr uint32_t kReadContacts = 1u << 3;
  static constexpr uint32_t kReadDiverged = 1u << 4;
  static constexpr uint32_t kReadAll = kReadQpos | kReadQvel | kReadLinks |
                                        kReadContacts | kReadDiverged;
  static constexpr int kAbiVersion = 2;

  SceneBatchExecutorV2(nb::sequence scenes, nb::sequence actors,
                       nb::sequence links, nb::sequence actuatorCounts,
                       nb::sequence contactSources, nb::sequence contactOthers,
                       nb::sequence contactKinds, nb::sequence contactDistances,
                       size_t numWorkers) {
    CheckContext();
    if (numWorkers == 0) {
      throw std::invalid_argument(
          "SceneBatchExecutorV2 requires num_workers >= 1");
    }
    if (nb::len(scenes) == 0 || nb::len(scenes) != nb::len(actors) ||
        nb::len(scenes) != nb::len(links) ||
        nb::len(scenes) != nb::len(contactSources) ||
        nb::len(scenes) != nb::len(contactOthers)) {
      throw std::invalid_argument(
          "scenes, actors, links, contact_sources, and contact_others must be "
          "non-empty and have equal length");
    }
    numWorkers = std::min(numWorkers, static_cast<size_t>(nb::len(scenes)));
    if (GetContext()->GetNumThreads() != 0) {
      throw std::invalid_argument(
          "SceneBatchExecutorV2 requires initialize(num_worker_threads=0); "
          "do not combine outer scene workers with SDK worker threads");
    }
    if (nb::len(contactKinds) != nb::len(contactDistances)) {
      throw std::invalid_argument(
          "contact_kinds and contact_distances must have equal length");
    }
    auto const maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
    if (nb::len(contactKinds) > maxInt) {
      throw std::invalid_argument("contact count exceeds the native int range");
    }
    for (size_t i = 0; i < static_cast<size_t>(nb::len(contactKinds)); ++i) {
      auto const kind = nb::cast<int>(contactKinds[i]);
      auto const distance = nb::cast<double>(contactDistances[i]);
      if (kind < 0 || kind > 2 || !std::isfinite(distance) || distance < 0) {
        throw std::invalid_argument(
            "contact kinds must be 0..2 and distances finite/non-negative");
      }
      _contactKinds.push_back(kind);
      _contactDistances.push_back(static_cast<real>(distance));
    }

    std::unordered_set<Scene *> uniqueScenes;
    std::unordered_set<Actor *> uniqueActors;
    int actorCount = -1;
    for (size_t i = 0; i < static_cast<size_t>(nb::len(scenes)); ++i) {
      auto *scene = nb::cast<Scene *>(scenes[i]);
      if (!scene || scene->GetContext() != GetContext() ||
          !uniqueScenes.insert(scene).second) {
        throw std::invalid_argument(
            "SceneBatchExecutorV2 requires unique live scenes");
      }
      _scenes.push_back(scene);
      _actors.emplace_back();
      _links.emplace_back();

      auto sceneActors = nb::cast<nb::sequence>(actors[i]);
      auto sceneLinks = nb::cast<nb::sequence>(links[i]);
      if (nb::len(sceneActors) == 0 ||
          nb::len(sceneActors) != nb::len(sceneLinks)) {
        throw std::invalid_argument(
            "each actors/links entry must be a non-empty equal-length pair");
      }
      if (actorCount < 0) {
        if (nb::len(sceneActors) > maxInt) {
          throw std::invalid_argument(
              "actor slot count exceeds the native int range");
        }
        actorCount = static_cast<int>(nb::len(sceneActors));
      } else if (static_cast<int>(nb::len(sceneActors)) != actorCount) {
        throw std::invalid_argument(
            "all scenes must use the same actor slot count");
      }

      for (size_t actorIndex = 0;
           actorIndex < static_cast<size_t>(nb::len(sceneActors)); ++actorIndex) {
        auto *actor = nb::cast<Actor *>(sceneActors[actorIndex]);
        if (!actor || actor->GetContext() != GetContext() ||
            actor->GetScene() != scene ||
            !uniqueActors.insert(actor).second) {
          throw std::invalid_argument(
              "each actor slot must contain a unique live actor in its scene");
        }
        _actors.back().push_back(actor);

        auto actorLinks =
            nb::cast<nb::sequence>(sceneLinks[actorIndex]);
        if (nb::len(actorLinks) > maxInt) {
          throw std::invalid_argument(
              "actor link count exceeds the native int range");
        }
        _links.back().emplace_back();
        for (size_t linkIndex = 0;
             linkIndex < static_cast<size_t>(nb::len(actorLinks)); ++linkIndex) {
          auto *link = nb::cast<Actor *>(actorLinks[linkIndex]);
          Error linkError;
          auto const parent =
              link ? link->GetArticulatedActor(linkError) : ActorHandle{};
          if (!link || !linkError.IsOK() || link->GetScene() != scene ||
              parent != actor->GetHandle()) {
            throw std::invalid_argument(
                "link actors must be nested links of their actor slot");
          }
          _links.back().back().push_back(link);
        }
      }
    }

    if (actorCount < 1 ||
        nb::len(actuatorCounts) != static_cast<size_t>(actorCount)) {
      throw std::invalid_argument(
          "actuator_counts must have one entry per actor slot");
    }
    int dofOffset = 0;
    int linkOffset = 0;
    int actuatorOffset = 0;
    bool hasArticulatedActor = false;
    for (int actorIndex = 0; actorIndex < actorCount; ++actorIndex) {
      int const dofCount = _actors[0][static_cast<size_t>(actorIndex)]
                               ->GetNumDofs();
      int const linkCount = static_cast<int>(
          _links[0][static_cast<size_t>(actorIndex)].size());
      int const actuatorCount =
          nb::cast<int>(actuatorCounts[static_cast<size_t>(actorIndex)]);
      if (dofCount < 0 || linkCount < 0 || actuatorCount < 0) {
        throw std::invalid_argument(
            "actor DoF, link, and actuator counts must be non-negative");
      }
      if (dofCount == 0 && (linkCount != 0 || actuatorCount != 0)) {
        throw std::invalid_argument(
            "zero-DoF actors cannot expose fake link or actuator state");
      }
      if (static_cast<size_t>(dofOffset) + static_cast<size_t>(dofCount) >
              maxInt ||
          static_cast<size_t>(linkOffset) + static_cast<size_t>(linkCount) >
              maxInt ||
          static_cast<size_t>(actuatorOffset) +
                  static_cast<size_t>(actuatorCount) >
              maxInt) {
        throw std::invalid_argument(
            "actor layout offsets exceed the native int range");
      }
      for (size_t sceneIndex = 0; sceneIndex < _scenes.size(); ++sceneIndex) {
        auto *actor = _actors[sceneIndex][static_cast<size_t>(actorIndex)];
        if (actor->GetNumDofs() != dofCount ||
            static_cast<int>(_links[sceneIndex][static_cast<size_t>(
                                 actorIndex)]
                                 .size()) != linkCount) {
          throw std::invalid_argument(
              "all scenes must use the same actor slot DoF/link layout");
        }
      }
      hasArticulatedActor = hasArticulatedActor || dofCount > 0;
      _actorDofCounts.push_back(dofCount);
      _actorDofOffsets.push_back(dofOffset);
      _actorLinkCounts.push_back(linkCount);
      _actorLinkOffsets.push_back(linkOffset);
      _actorActuatorCounts.push_back(actuatorCount);
      _actorActuatorOffsets.push_back(actuatorOffset);
      // Native actor APIs use actor-local DoF indices. The flattened global
      // address is represented by actor_dof_offsets and state/actuator arrays.
      for (int dof = 0; dof < dofCount; ++dof) {
        _dofIndices.push_back(dof);
        _dofActor.push_back(actorIndex);
      }
      dofOffset += dofCount;
      linkOffset += linkCount;
      actuatorOffset += actuatorCount;
    }
    if (!hasArticulatedActor) {
      throw std::invalid_argument(
          "SceneBatchExecutorV2 requires at least one articulated actor");
    }
    _actorCount = actorCount;
    _dofCount = dofOffset;
    _linkCount = linkOffset;
    _actuatorCount = actuatorOffset;

    for (size_t i = 0; i < _scenes.size(); ++i) {
      auto sceneSources = nb::cast<nb::sequence>(contactSources[i]);
      auto sceneOthers = nb::cast<nb::sequence>(contactOthers[i]);
      if (static_cast<int>(nb::len(sceneSources)) !=
              static_cast<int>(_contactKinds.size()) ||
          nb::len(sceneSources) != nb::len(sceneOthers)) {
        throw std::invalid_argument(
            "each contact source/other list must match contact_kinds");
      }
      _contactSources.emplace_back();
      _contactOthers.emplace_back();
      for (size_t contactIndex = 0; contactIndex < _contactKinds.size();
           ++contactIndex) {
        auto *source =
            nb::cast<Actor *>(sceneSources[contactIndex]);
        if (!source || source->GetScene() != _scenes[i]) {
          throw std::invalid_argument(
              "contact sources must belong to their scene");
        }
        _contactSources.back().push_back(source);
        if (sceneOthers[contactIndex].is_none()) {
          _contactOthers.back().push_back(nullptr);
        } else {
          auto *other = nb::cast<Actor *>(sceneOthers[contactIndex]);
          if (!other || other->GetScene() != _scenes[i]) {
            throw std::invalid_argument(
                "contact actors must belong to their scene");
          }
          _contactOthers.back().push_back(other);
        }
      }
    }
    _contactCount = static_cast<int>(_contactKinds.size());

    _workers.reserve(numWorkers);
    detail::RegisterLeasedScenes(_scenes);
    _leasesRegistered = true;
    try {
      for (size_t i = 0; i < numWorkers; ++i) {
        _workers.emplace_back(&SceneBatchExecutorV2::WorkerMain, this);
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
      detail::ReleaseLeasedScenes(_scenes);
      _leasesRegistered = false;
      throw;
    }
  }

  ~SceneBatchExecutorV2() { Close(); }

  SceneBatchExecutorV2(SceneBatchExecutorV2 const &) = delete;
  SceneBatchExecutorV2 &operator=(SceneBatchExecutorV2 const &) = delete;

  [[nodiscard]] size_t GetNumWorkers() const { return _workers.size(); }
  [[nodiscard]] size_t GetNumScenes() const { return _scenes.size(); }
  [[nodiscard]] int GetNumActors() const { return _actorCount; }
  [[nodiscard]] int GetNumDofs() const { return _dofCount; }
  [[nodiscard]] int GetNumLinks() const { return _linkCount; }
  [[nodiscard]] int GetNumActuators() const { return _actuatorCount; }
  [[nodiscard]] int GetNumContacts() const { return _contactCount; }
  [[nodiscard]] bool IsClosed() const { return _closed; }
  [[nodiscard]] std::vector<int> const &GetActorDofCounts() const {
    return _actorDofCounts;
  }
  [[nodiscard]] std::vector<int> const &GetActorDofOffsets() const {
    return _actorDofOffsets;
  }
  [[nodiscard]] std::vector<int> const &GetActorLinkCounts() const {
    return _actorLinkCounts;
  }
  [[nodiscard]] std::vector<int> const &GetActorLinkOffsets() const {
    return _actorLinkOffsets;
  }
  [[nodiscard]] std::vector<int> const &GetActorActuatorCounts() const {
    return _actorActuatorCounts;
  }
  [[nodiscard]] std::vector<int> const &GetActorActuatorOffsets() const {
    return _actorActuatorOffsets;
  }

  void Step(double timeStepSec,
            RealConstArray generalizedForces,
            nb::object qposOut, nb::object qvelOut, nb::object linkStateOut,
            nb::object contactOut, nb::object divergedOut,
            uint32_t readbackMask) {
    CheckContext();
    ValidateTimeStep("step", timeStepSec);
    ValidateStateArray("generalized_forces", generalizedForces);
    RequireFinite("generalized_forces", generalizedForces);

    std::unique_lock callLock(_callMutex);
    if (_closed) {
      throw std::runtime_error("SceneBatchExecutorV2 is closed");
    }
    CheckLiveActors();
    auto outputs = ValidateReadback(readbackMask, qposOut, qvelOut,
                                    linkStateOut, contactOut, divergedOut);
    std::exception_ptr failure;
    {
      nb::gil_scoped_release release;
      failure = DispatchAndWait(
          timeStepSec, generalizedForces.data(), nullptr,
          outputs.qpos ? outputs.qpos->data() : nullptr,
          outputs.qvel ? outputs.qvel->data() : nullptr,
          outputs.links ? outputs.links->data() : nullptr,
          outputs.contacts ? outputs.contacts->data() : nullptr,
          outputs.diverged ? outputs.diverged->data() : nullptr,
          readbackMask);
    }
    if (failure) {
      CloseLocked();
      std::rethrow_exception(failure);
    }
  }

  void StepControl(
      double timeStepSec, RealConstArray controls,
      IntConstArray qposIndices,
      IntConstArray qvelIndices,
      RealConstArray kp,
      RealConstArray kd,
      RealConstArray gear,
      RealConstArray forceRanges, int numSteps,
      nb::object qposOut, nb::object qvelOut, nb::object linkStateOut,
      nb::object contactOut, nb::object divergedOut,
      uint32_t readbackMask) {
    CheckContext();
    if (numSteps < 1) {
      throw std::invalid_argument(
          "step_control requires positive num_steps");
    }
    ValidateTimeStep("step_control", timeStepSec);
    ValidateControlArray("controls", controls);
    RequireFinite("controls", controls);
    if (qposIndices.ndim() != 1 ||
        qposIndices.shape(0) != _actuatorCount ||
        qvelIndices.ndim() != 1 ||
        qvelIndices.shape(0) != _actuatorCount || kp.ndim() != 1 ||
        kp.shape(0) != _actuatorCount || kd.ndim() != 1 ||
        kd.shape(0) != _actuatorCount || gear.ndim() != 1 ||
        gear.shape(0) != _actuatorCount || forceRanges.ndim() != 2 ||
        forceRanges.shape(0) != _actuatorCount ||
        forceRanges.shape(1) != 2) {
      throw std::invalid_argument(
          "control metadata must match the flattened actuator layout");
    }
    RequireFinite("kp", kp);
    RequireFinite("kd", kd);
    RequireFinite("gear", gear);
    RequireFinite("force_ranges", forceRanges);
    std::vector<int> controlActor(static_cast<size_t>(_actuatorCount), -1);
    for (int i = 0; i < _actuatorCount; ++i) {
      int const q = qposIndices.data()[i];
      int const v = qvelIndices.data()[i];
      if (q < 0 || q >= _dofCount || v < 0 || v >= _dofCount ||
          _dofActor[static_cast<size_t>(q)] !=
              _dofActor[static_cast<size_t>(v)] ||
          forceRanges.data()[2 * i + 0] > forceRanges.data()[2 * i + 1]) {
        throw std::invalid_argument(
            "each actuator's qpos/qvel indices must address one actor");
      }
      controlActor[static_cast<size_t>(i)] =
          _dofActor[static_cast<size_t>(q)];
    }
    auto outputs = ValidateReadback(readbackMask, qposOut, qvelOut,
                                    linkStateOut, contactOut, divergedOut);

    std::unique_lock callLock(_callMutex);
    if (_closed) {
      throw std::runtime_error("SceneBatchExecutorV2 is closed");
    }
    CheckLiveActors();
    std::exception_ptr failure;
    {
      std::lock_guard lock(_mutex);
      _controlMode = true;
      _controlSteps = numSteps;
      _controlData = controls.data();
      _controlQposIndices = qposIndices.data();
      _controlQvelIndices = qvelIndices.data();
      _controlKp = kp.data();
      _controlKd = kd.data();
      _controlGear = gear.data();
      _controlForceRanges = forceRanges.data();
    }
    {
      nb::gil_scoped_release release;
      failure = DispatchAndWait(
          timeStepSec, nullptr, controlActor.data(),
          outputs.qpos ? outputs.qpos->data() : nullptr,
          outputs.qvel ? outputs.qvel->data() : nullptr,
          outputs.links ? outputs.links->data() : nullptr,
          outputs.contacts ? outputs.contacts->data() : nullptr,
          outputs.diverged ? outputs.diverged->data() : nullptr,
          readbackMask);
    }
    if (failure) {
      CloseLocked();
      std::rethrow_exception(failure);
    }
    std::lock_guard lock(_mutex);
    _controlMode = false;
  }

  void WriteState(nb::object qpos, nb::object qvel, nb::object qposMask,
                  nb::object qvelMask) {
    CheckContext();
    auto qposWrite = RequireWritePayload("qpos", qpos, qposMask);
    auto qvelWrite = RequireWritePayload("qvel", qvel, qvelMask);
    if (!qposWrite.has_value() && !qvelWrite.has_value()) {
      return;
    }

    std::unique_lock callLock(_callMutex);
    if (_closed) {
      throw std::runtime_error("SceneBatchExecutorV2 is closed");
    }
    CheckLiveActors();
    try {
      auto const *qposValues =
          qposWrite ? qposWrite->values.data() : nullptr;
      auto const *qposMask = qposWrite ? qposWrite->mask.data() : nullptr;
      auto const *qvelValues =
          qvelWrite ? qvelWrite->values.data() : nullptr;
      auto const *qvelMask = qvelWrite ? qvelWrite->mask.data() : nullptr;
      for (size_t sceneIndex = 0; sceneIndex < _scenes.size(); ++sceneIndex) {
        for (int actorIndex = 0; actorIndex < _actorCount; ++actorIndex) {
          int const dofCount =
              _actorDofCounts[static_cast<size_t>(actorIndex)];
          if (dofCount == 0) {
            continue;
          }
          size_t const actor = static_cast<size_t>(actorIndex);
          size_t const sceneOffset =
              sceneIndex * static_cast<size_t>(_dofCount) +
              static_cast<size_t>(_actorDofOffsets[actor]);
          bool writePose = false;
          bool writeVelocity = false;
          thread_local std::vector<real> pose;
          thread_local std::vector<real> velocity;
          pose.resize(static_cast<size_t>(dofCount));
          velocity.resize(static_cast<size_t>(dofCount));
          auto *actorPtr = _actors[sceneIndex][actor];
          Error error;
          actorPtr->GetArticulatedPose(Span<real>(pose), error);
          if (!error.IsOK()) {
            throw MochiErrorException(error);
          }
          actorPtr->GetArticulatedJointVelocities(Span<real>(velocity), error);
          if (!error.IsOK()) {
            throw MochiErrorException(error);
          }
          for (int dof = 0; dof < dofCount; ++dof) {
            size_t const flat = sceneOffset + static_cast<size_t>(dof);
            if (qposMask && qposMask[flat]) {
              pose[static_cast<size_t>(dof)] = qposValues[flat];
              writePose = true;
            }
            if (qvelMask && qvelMask[flat]) {
              velocity[static_cast<size_t>(dof)] = qvelValues[flat];
              writeVelocity = true;
            }
          }
          if (writePose) {
            actorPtr->SetArticulatedPoseFromJoints(
                Span<real const>(pose), error);
            if (!error.IsOK()) {
              throw MochiErrorException(error);
            }
          }
          if (writeVelocity) {
            actorPtr->SetArticulatedJointVelocities(
                Span<real const>(velocity), error);
            if (!error.IsOK()) {
              throw MochiErrorException(error);
            }
          }
        }
      }
    } catch (...) {
      CloseLocked();
      throw;
    }
  }

  void WriteBoundaryConditions(nb::object values, nb::object mask) {
    CheckContext();
    auto write = RequireWritePayload("boundary_condition_values", values, mask);
    if (!write.has_value()) {
      return;
    }

    std::unique_lock callLock(_callMutex);
    if (_closed) {
      throw std::runtime_error("SceneBatchExecutorV3 is closed");
    }
    CheckLiveActors();
    try {
      auto const *valueData = write->values.data();
      auto const *maskData = write->mask.data();
      for (size_t sceneIndex = 0; sceneIndex < _scenes.size(); ++sceneIndex) {
        for (int actorIndex = 0; actorIndex < _actorCount; ++actorIndex) {
          size_t const actor = static_cast<size_t>(actorIndex);
          int const dofCount = _actorDofCounts[actor];
          int const dofOffset = _actorDofOffsets[actor];
          size_t const sceneOffset =
              sceneIndex * static_cast<size_t>(_dofCount) +
              static_cast<size_t>(dofOffset);
          bool selected = false;
          for (int dof = 0; dof < dofCount; ++dof) {
            selected = selected ||
                       maskData[sceneOffset + static_cast<size_t>(dof)] != 0;
          }
          if (!selected) {
            continue;
          }

          // Boundary-condition APIs append entries. Replacing the complete
          // clearable set keeps repeated selected writes bounded and prevents a
          // new target from competing with an older one.
          auto *actorPtr = _actors[sceneIndex][actor];
          actorPtr->ClearBoundaryConditions();
          Error error;
          actorPtr->AddBoundaryConditionDofsWorld(
              Span<int const>(_dofIndices.data() + dofOffset,
                              static_cast<size_t>(dofCount)),
              Span<real const>(valueData + sceneOffset,
                               static_cast<size_t>(dofCount)),
              error);
          if (!error.IsOK()) {
            throw MochiErrorException(error);
          }
        }
      }
    } catch (...) {
      CloseLocked();
      throw;
    }
  }

  void Close() {
    std::unique_lock callLock(_callMutex);
    CloseLocked();
  }

private:
  struct ReadbackBuffers {
    std::optional<RealArray> qpos;
    std::optional<RealArray> qvel;
    std::optional<RealArray> links;
    std::optional<RealArray> contacts;
    std::optional<Uint8Array> diverged;
  };

  struct WritePayload {
    RealConstArray values;
    Uint8ConstArray mask;
  };

  static void ValidateTimeStep(char const *name, double value) {
    if (!std::isfinite(value) || value < 0) {
      throw std::invalid_argument(
          std::string(name) +
          " requires a finite, non-negative time_step_sec");
    }
  }

  template <typename Array>
  void ValidateStateArray(char const *name, Array const &array) const {
    if (array.ndim() != 2 ||
        array.shape(0) != static_cast<size_t>(_scenes.size()) ||
        array.shape(1) != _dofCount) {
      throw std::invalid_argument(
          std::string(name) + " must have shape [num_scenes, total_dofs]");
    }
  }

  void ValidateControlArray(
      char const *name, RealConstArray const &array) const {
    if (array.ndim() != 2 ||
        array.shape(0) != static_cast<size_t>(_scenes.size()) ||
        array.shape(1) != _actuatorCount) {
      throw std::invalid_argument(
          std::string(name) + " must have shape [num_scenes, num_actuators]");
    }
  }

  template <typename Array>
  static void RequireFinite(char const *name, Array const &array) {
    auto const *data = array.data();
    for (size_t i = 0; i < array.size(); ++i) {
      if (!std::isfinite(data[i])) {
        throw std::invalid_argument(
            std::string(name) + " values must be finite");
      }
    }
  }

  std::optional<RealArray> RequireStateOutput(
      char const *name, nb::object const &object, bool requested) const {
    if (!requested) {
      if (!object.is_none()) {
        throw std::invalid_argument(
            std::string(name) +
            " must be None when readback is disabled");
      }
      return std::nullopt;
    }
    if (object.is_none()) {
      throw std::invalid_argument(
          std::string(name) + " is required by readback_mask");
    }
    auto array = nb::cast<RealArray>(object);
    ValidateStateArray(name, array);
    return array;
  }

  ReadbackBuffers ValidateReadback(uint32_t readbackMask,
                                    nb::object const &qposOut,
                                    nb::object const &qvelOut,
                                    nb::object const &linkStateOut,
                                    nb::object const &contactOut,
                                    nb::object const &divergedOut) const {
    if ((readbackMask & ~kReadAll) != 0) {
      throw std::invalid_argument("readback_mask contains unknown fields");
    }
    auto qpos = RequireStateOutput("qpos_out", qposOut,
                                   readbackMask & kReadQpos);
    auto qvel = RequireStateOutput("qvel_out", qvelOut,
                                   readbackMask & kReadQvel);
    ReadbackBuffers buffers;
    buffers.qpos = std::move(qpos);
    buffers.qvel = std::move(qvel);
    if ((readbackMask & kReadLinks) == 0) {
      if (!linkStateOut.is_none()) {
        throw std::invalid_argument(
            "link_state_out must be None when readback is disabled");
      }
    } else {
      if (linkStateOut.is_none()) {
        throw std::invalid_argument(
            "link_state_out is required by readback_mask");
      }
      buffers.links = nb::cast<RealArray>(linkStateOut);
      if (buffers.links->ndim() != 3 ||
          buffers.links->shape(0) !=
              static_cast<size_t>(_scenes.size()) ||
          buffers.links->shape(1) != _linkCount ||
          buffers.links->shape(2) != 16) {
        throw std::invalid_argument(
            "link_state_out must have shape [num_scenes, total_links, 16]");
      }
    }
    if ((readbackMask & kReadContacts) == 0) {
      if (!contactOut.is_none()) {
        throw std::invalid_argument(
            "contact_out must be None when readback is disabled");
      }
    } else {
      if (contactOut.is_none()) {
        throw std::invalid_argument(
            "contact_out is required by readback_mask");
      }
      buffers.contacts = nb::cast<RealArray>(contactOut);
      if (buffers.contacts->ndim() != 3 ||
          buffers.contacts->shape(0) !=
              static_cast<size_t>(_scenes.size()) ||
          buffers.contacts->shape(1) != _contactCount ||
          buffers.contacts->shape(2) != 3) {
        throw std::invalid_argument(
            "contact_out must have shape [num_scenes, num_contacts, 3]");
      }
    }
    if ((readbackMask & kReadDiverged) == 0) {
      if (!divergedOut.is_none()) {
        throw std::invalid_argument(
            "diverged_out must be None when readback is disabled");
      }
    } else {
      if (divergedOut.is_none()) {
        throw std::invalid_argument(
            "diverged_out is required by readback_mask");
      }
      buffers.diverged = nb::cast<Uint8Array>(divergedOut);
      if (buffers.diverged->ndim() != 1 ||
          buffers.diverged->shape(0) !=
              static_cast<size_t>(_scenes.size())) {
        throw std::invalid_argument(
            "diverged_out must have shape [num_scenes]");
      }
    }
    return buffers;
  }

  std::optional<WritePayload> RequireWritePayload(char const *name,
                                                   nb::object values,
                                                   nb::object mask) const {
    if (mask.is_none()) {
      if (!values.is_none()) {
        throw std::invalid_argument(
            std::string(name) + " must be None when its mask is None");
      }
      return std::nullopt;
    }
    auto maskArray = nb::cast<Uint8ConstArray>(mask);
    if (maskArray.ndim() != 2 ||
        maskArray.shape(0) != static_cast<size_t>(_scenes.size()) ||
        maskArray.shape(1) != _dofCount) {
      throw std::invalid_argument(
          std::string(name) + "_write_mask must have shape "
                              "[num_scenes, total_dofs]");
    }
    if (values.is_none()) {
      throw std::invalid_argument(
          std::string(name) + " is required when its mask is supplied");
    }
    auto valueArray = nb::cast<RealConstArray>(values);
    ValidateStateArray(name, valueArray);
    RequireFinite(name, valueArray);

    bool any = false;
    for (size_t i = 0; i < maskArray.size(); ++i) {
      if (maskArray.data()[i] > 1) {
        throw std::invalid_argument(
            std::string(name) + "_write_mask values must be 0 or 1");
      }
      any = any || maskArray.data()[i] != 0;
    }
    if (!any) {
      return std::nullopt;
    }
    return WritePayload{std::move(valueArray), std::move(maskArray)};
  }

  void CheckLiveActors() const {
    for (size_t sceneIndex = 0; sceneIndex < _scenes.size(); ++sceneIndex) {
      if (!GetContext()->IsValidScene(_scenes[sceneIndex])) {
        throw std::runtime_error(
            "a scene was destroyed while its executor is active");
      }
      for (auto *actor : _actors[sceneIndex]) {
        if (actor->GetScene() != _scenes[sceneIndex]) {
          throw std::runtime_error(
              "an actor was destroyed while its executor is active");
        }
      }
    }
  }

  std::exception_ptr DispatchAndWait(
      double timeStepSec, real const *forceData, int const *controlActor,
      real *qposData, real *qvelData, real *linkStateData,
      real *contactData, uint8_t *divergedData, uint32_t readbackMask) {
    {
      std::lock_guard lock(_mutex);
      _failure = nullptr;
      _nextIndex.store(0, std::memory_order_relaxed);
      _completedWorkers = 0;
      _timeStepSec = timeStepSec;
      _forceData = forceData;
      _controlActorData = controlActor;
      _qposData = qposData;
      _qvelData = qvelData;
      _linkStateData = linkStateData;
      _contactData = contactData;
      _divergedData = divergedData;
      _readbackMask = readbackMask;
      ++_dispatchGeneration;
    }
    _workAvailable.notify_all();
    std::unique_lock lock(_mutex);
    _allDone.wait(lock,
                  [this]() { return _completedWorkers == _workers.size(); });
    return _failure;
  }

  void RunScene(size_t i) {
    Error error;
    thread_local std::vector<real> pose;
    thread_local std::vector<real> velocity;
    thread_local std::vector<real> forces;
    thread_local std::vector<TransformRT> transforms;
    for (int controlStep = 0;
         controlStep < (_controlMode ? _controlSteps : 1); ++controlStep) {
      for (int actorIndex = 0; actorIndex < _actorCount; ++actorIndex) {
        size_t const actor = static_cast<size_t>(actorIndex);
        int const dofCount = _actorDofCounts[actor];
        if (dofCount == 0) {
          continue;
        }
        int const dofOffset = _actorDofOffsets[actor];
        auto *actorPtr = _actors[i][actor];
        real const *forceData = nullptr;
        if (_controlMode) {
          pose.resize(static_cast<size_t>(dofCount));
          velocity.resize(static_cast<size_t>(dofCount));
          forces.assign(static_cast<size_t>(dofCount), 0);
          actorPtr->GetArticulatedPose(Span<real>(pose), error);
          if (!error.IsOK()) {
            throw MochiErrorException(error);
          }
          actorPtr->GetArticulatedJointVelocities(Span<real>(velocity),
                                                  error);
          if (!error.IsOK()) {
            throw MochiErrorException(error);
          }
          for (int channel = 0; channel < _actuatorCount; ++channel) {
            size_t const actuator = static_cast<size_t>(channel);
            if (_controlActorData[actuator] != actorIndex) {
              continue;
            }
            int const q = _controlQposIndices[actuator] - dofOffset;
            int const v = _controlQvelIndices[actuator] - dofOffset;
            auto value =
                _controlKp[actuator] > 0
                    ? _controlKp[actuator] *
                          (_controlData[i * _actuatorCount + channel] -
                           pose[static_cast<size_t>(q)] *
                               _controlGear[actuator]) -
                          _controlKd[actuator] *
                              velocity[static_cast<size_t>(v)] *
                              _controlGear[actuator]
                    : _controlData[i * _actuatorCount + channel];
            value = std::clamp(value,
                               _controlForceRanges[2 * actuator],
                               _controlForceRanges[2 * actuator + 1]);
            forces[static_cast<size_t>(v)] += value * _controlGear[actuator];
          }
          forceData = forces.data();
        } else {
          forceData =
              _forceData + i * static_cast<size_t>(_dofCount) + dofOffset;
        }
        actorPtr->SetExternalForcesOnDofs(
            Span<int const>(_dofIndices.data() + dofOffset,
                            static_cast<size_t>(dofCount)),
            Span<real const>(forceData, static_cast<size_t>(dofCount)),
            error);
        if (!error.IsOK()) {
          throw MochiErrorException(error);
        }
      }
      _scenes[i]->Step(_timeStepSec);
    }

    if (_readbackMask & kReadDiverged) {
      _divergedData[i] =
          _scenes[i]->GetSolverStats().convergenceStatus ==
                  ConvergenceStatus::Diverged
              ? 1
              : 0;
    }
    for (int actorIndex = 0; actorIndex < _actorCount; ++actorIndex) {
      size_t const actor = static_cast<size_t>(actorIndex);
      int const dofCount = _actorDofCounts[actor];
      if (dofCount == 0) {
        continue;
      }
      auto const stateOffset =
          i * static_cast<size_t>(_dofCount) +
          static_cast<size_t>(_actorDofOffsets[actor]);
      if (_readbackMask & kReadQpos) {
        _actors[i][actor]->GetArticulatedPose(
            Span<real>(_qposData + stateOffset,
                      static_cast<size_t>(dofCount)),
            error);
        if (!error.IsOK()) {
          throw MochiErrorException(error);
        }
      }
      if (_readbackMask & kReadQvel) {
        _actors[i][actor]->GetArticulatedJointVelocities(
            Span<real>(_qvelData + stateOffset,
                      static_cast<size_t>(dofCount)),
            error);
        if (!error.IsOK()) {
          throw MochiErrorException(error);
        }
      }
      if ((_readbackMask & kReadLinks) == 0) {
        continue;
      }
      int const linkCount = _actorLinkCounts[actor];
      transforms.resize(static_cast<size_t>(linkCount));
      _actors[i][actor]->GetArticulatedLinkTransforms(MakeSpan(transforms),
                                                      error);
      if (!error.IsOK()) {
        throw MochiErrorException(error);
      }
      auto linkOffset = i * static_cast<size_t>(_linkCount) * 16 +
                        static_cast<size_t>(_actorLinkOffsets[actor]) * 16;
      for (int linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
        auto const &transform =
            transforms[static_cast<size_t>(linkIndex)];
        auto const position = transform.GetTranslation();
        auto const rotation = transform.GetRotation().ToReal4();
        auto *out =
            _linkStateData + linkOffset +
            static_cast<size_t>(linkIndex) * 16;
        out[0] = position[0];
        out[1] = position[1];
        out[2] = position[2];
        out[3] = rotation[3];
        out[4] = rotation[0];
        out[5] = rotation[1];
        out[6] = rotation[2];
        auto *link =
            _links[i][actor][static_cast<size_t>(linkIndex)];
        auto const com = link->GetCenterOfMassTransform(error);
        if (!error.IsOK()) {
          throw MochiErrorException(error);
        }
        auto const comPosition = com.GetTranslation();
        auto const linear = link->GetLinearVelocity(error);
        auto const angular = link->GetAngularVelocity(error);
        if (!error.IsOK()) {
          throw MochiErrorException(error);
        }
        out[7] = comPosition[0];
        out[8] = comPosition[1];
        out[9] = comPosition[2];
        out[10] = linear[0];
        out[11] = linear[1];
        out[12] = linear[2];
        out[13] = angular[0];
        out[14] = angular[1];
        out[15] = angular[2];
      }
    }
    if (_readbackMask & kReadContacts) {
      auto const contactOffset =
          i * static_cast<size_t>(_contactCount) * 3;
      for (int contactIndex = 0; contactIndex < _contactCount;
           ++contactIndex) {
        auto *out = _contactData + contactOffset +
                    static_cast<size_t>(contactIndex) * 3;
        out[0] = 0;
        out[1] = 0;
        out[2] = 0;
        auto *source =
            _contactSources[i][static_cast<size_t>(contactIndex)];
        auto *other = _contactOthers[i][static_cast<size_t>(contactIndex)];
        if (_contactKinds[static_cast<size_t>(contactIndex)] == 0) {
          auto points = source->GetContactPointsWorld(error);
          if (!error.IsOK()) {
            throw MochiErrorException(error);
          }
          auto const own = source->GetHandle();
          auto const otherHandle = other ? other->GetHandle()
                                         : ActorHandle{};
          for (auto const &point : points) {
            if (point.distance <=
                    _contactDistances[static_cast<size_t>(contactIndex)] &&
                (!other ||
                 ((point.actorA == own && point.actorB == otherHandle) ||
                  (point.actorA == otherHandle && point.actorB == own)))) {
              out[0] = 1;
              break;
            }
          }
        } else {
          auto value =
              _contactKinds[static_cast<size_t>(contactIndex)] == 1
                  ? source->GetContactForceWorld(error)
                  : source->GetContactTorqueWorld(error);
          if (!error.IsOK()) {
            throw MochiErrorException(error);
          }
          out[0] = value[0];
          out[1] = value[1];
          out[2] = value[2];
        }
      }
    }
  }

  void WorkerMain() {
    auto *context = GetContext();
    context->BindThisThread();
    size_t seenGeneration = 0;
    while (true) {
      {
        std::unique_lock lock(_mutex);
        _workAvailable.wait(lock, [this, seenGeneration]() {
          return _stopping || _dispatchGeneration != seenGeneration;
        });
        if (_stopping) {
          break;
        }
        seenGeneration = _dispatchGeneration;
      }
      while (true) {
        auto const index = _nextIndex.fetch_add(1, std::memory_order_relaxed);
        if (index >= _scenes.size()) {
          break;
        }
        try {
          RunScene(index);
        } catch (...) {
          std::lock_guard lock(_mutex);
          if (!_failure) {
            _failure = std::current_exception();
          }
        }
      }
      {
        std::lock_guard lock(_mutex);
        ++_completedWorkers;
        if (_completedWorkers == _workers.size()) {
          _allDone.notify_one();
        }
      }
    }
    context->UnbindThisThread();
  }

  void CloseLocked() {
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
      detail::ReleaseLeasedScenes(_scenes);
      _leasesRegistered = false;
    }
    _closed = true;
  }

  std::vector<Scene *> _scenes;
  std::vector<std::vector<Actor *>> _actors;
  std::vector<std::vector<std::vector<Actor *>>> _links;
  std::vector<std::vector<Actor *>> _contactSources;
  std::vector<std::vector<Actor *>> _contactOthers;
  std::vector<int> _contactKinds;
  std::vector<real> _contactDistances;
  std::vector<int> _dofIndices;
  std::vector<int> _dofActor;
  std::vector<int> _actorDofCounts;
  std::vector<int> _actorDofOffsets;
  std::vector<int> _actorLinkCounts;
  std::vector<int> _actorLinkOffsets;
  std::vector<int> _actorActuatorCounts;
  std::vector<int> _actorActuatorOffsets;
  int _actorCount = 0;
  int _dofCount = 0;
  int _linkCount = 0;
  int _actuatorCount = 0;
  int _contactCount = 0;
  std::vector<std::thread> _workers;
  std::atomic<size_t> _nextIndex{0};
  double _timeStepSec = 0;
  real const *_forceData = nullptr;
  int const *_controlActorData = nullptr;
  real *_qposData = nullptr;
  real *_qvelData = nullptr;
  real *_linkStateData = nullptr;
  real *_contactData = nullptr;
  uint8_t *_divergedData = nullptr;
  uint32_t _readbackMask = kReadAll;
  bool _controlMode = false;
  int _controlSteps = 1;
  real const *_controlData = nullptr;
  int const *_controlQposIndices = nullptr;
  int const *_controlQvelIndices = nullptr;
  real const *_controlKp = nullptr;
  real const *_controlKd = nullptr;
  real const *_controlGear = nullptr;
  real const *_controlForceRanges = nullptr;
  mutable std::mutex _mutex;
  std::mutex _callMutex;
  std::condition_variable _workAvailable;
  std::condition_variable _allDone;
  std::exception_ptr _failure;
  size_t _completedWorkers = 0;
  size_t _dispatchGeneration = 0;
  bool _stopping = false;
  bool _closed = false;
  bool _leasesRegistered = false;
};

class SceneBatchExecutorV3 : public SceneBatchExecutorV2 {
public:
  using SceneBatchExecutorV2::SceneBatchExecutorV2;

  static constexpr int kAbiVersion = 3;
};

} // namespace

namespace {

template <typename Executor>
void RegisterExecutorTeardown(nb::handle executorObject) {
  // Keep the permanent context callback weak so Python controls executor
  // lifetime while mochi.shutdown() can still close live executors first.
  PyObject *executorWeak = PyWeakref_NewRef(executorObject.ptr(), nullptr);
  if (!executorWeak) {
    throw std::runtime_error("failed to create SceneBatchExecutor weakref");
  }
  RegisterContextDependent([executorWeak]() {
    PyObject *executorPy = PyWeakref_GetObject(executorWeak);
    if (!executorPy || executorPy == Py_None) {
      return;
    }
    if (auto *executor = nb::cast<Executor *>(nb::handle(executorPy))) {
      executor->Close();
    }
  });
}

}  // namespace

void DefineSceneBatchExecutorV2(nb::module_ &m) {
  m.attr("SCENE_BATCH_EXECUTOR_ABI_VERSION") = nb::int_(3);
  nb::class_<SceneBatchExecutorV2>(m, "SceneBatchExecutorV2",
                                   nb::is_weak_referenceable())
      .def("__init__",
           [](nb::pointer_and_handle<SceneBatchExecutorV2> executor,
              nb::sequence scenes, nb::sequence actors, nb::sequence links,
              nb::sequence actuatorCounts, nb::sequence contactSources,
              nb::sequence contactOthers, nb::sequence contactKinds,
              nb::sequence contactDistances, size_t numWorkers) {
             new (executor.p) SceneBatchExecutorV2(
                 scenes, actors, links, actuatorCounts, contactSources,
                 contactOthers, contactKinds, contactDistances, numWorkers);
             RegisterExecutorTeardown<SceneBatchExecutorV2>(executor.h);
           },
           nb::arg("scenes"), nb::arg("actors"), nb::arg("links"),
           nb::arg("actuator_counts"), nb::arg("contact_sources"),
           nb::arg("contact_others"), nb::arg("contact_kinds"),
           nb::arg("contact_distances"), nb::arg("num_workers"))
      .def_prop_ro("abi_version",
                             [](SceneBatchExecutorV2 const &) {
                               return SceneBatchExecutorV2::kAbiVersion;
                             })
      .def_prop_ro("num_workers", &SceneBatchExecutorV2::GetNumWorkers)
      .def_prop_ro("num_scenes", &SceneBatchExecutorV2::GetNumScenes)
      .def_prop_ro("num_actors", &SceneBatchExecutorV2::GetNumActors)
      .def_prop_ro("num_dofs", &SceneBatchExecutorV2::GetNumDofs)
      .def_prop_ro("num_links", &SceneBatchExecutorV2::GetNumLinks)
      .def_prop_ro("num_actuators",
                             &SceneBatchExecutorV2::GetNumActuators)
      .def_prop_ro("num_contacts",
                             &SceneBatchExecutorV2::GetNumContacts)
      .def_prop_ro(
          "actor_dof_counts",
          [](SceneBatchExecutorV2 const &self) {
            return self.GetActorDofCounts();
          })
      .def_prop_ro(
          "actor_dof_offsets",
          [](SceneBatchExecutorV2 const &self) {
            return self.GetActorDofOffsets();
          })
      .def_prop_ro(
          "actor_qpos_offsets",
          [](SceneBatchExecutorV2 const &self) {
            return self.GetActorDofOffsets();
          })
      .def_prop_ro(
          "actor_qvel_offsets",
          [](SceneBatchExecutorV2 const &self) {
            return self.GetActorDofOffsets();
          })
      .def_prop_ro(
          "actor_link_counts",
          [](SceneBatchExecutorV2 const &self) {
            return self.GetActorLinkCounts();
          })
      .def_prop_ro(
          "actor_link_offsets",
          [](SceneBatchExecutorV2 const &self) {
            return self.GetActorLinkOffsets();
          })
      .def_prop_ro(
          "actor_actuator_counts",
          [](SceneBatchExecutorV2 const &self) {
            return self.GetActorActuatorCounts();
          })
      .def_prop_ro(
          "actor_actuator_offsets",
          [](SceneBatchExecutorV2 const &self) {
            return self.GetActorActuatorOffsets();
          })
      .def_prop_ro("closed", &SceneBatchExecutorV2::IsClosed)
      .def("write_state", &SceneBatchExecutorV2::WriteState,
           nb::arg("qpos").none(), nb::arg("qvel").none(),
           nb::arg("qpos_write_mask").none(),
           nb::arg("qvel_write_mask").none())
      .def("step", &SceneBatchExecutorV2::Step, nb::arg("time_step_sec"),
           nb::arg("generalized_forces"), nb::arg("qpos_out").none(),
           nb::arg("qvel_out").none(), nb::arg("link_state_out").none(),
           nb::arg("contact_out").none(), nb::arg("diverged_out").none(),
           nb::arg("readback_mask"))
      .def("step_control", &SceneBatchExecutorV2::StepControl,
           nb::arg("time_step_sec"), nb::arg("controls"),
           nb::arg("qpos_indices"), nb::arg("qvel_indices"), nb::arg("kp"),
           nb::arg("kd"), nb::arg("gear"), nb::arg("force_ranges"),
           nb::arg("num_steps"), nb::arg("qpos_out").none(),
           nb::arg("qvel_out").none(), nb::arg("link_state_out").none(),
           nb::arg("contact_out").none(), nb::arg("diverged_out").none(),
           nb::arg("readback_mask"))
      .def("close", [](SceneBatchExecutorV2 &self) {
        nb::gil_scoped_release release;
        self.Close();
      })
      .def("__enter__",
           [](SceneBatchExecutorV2 &self) -> SceneBatchExecutorV2 & {
             return self;
           },
           nb::rv_policy::reference)
      .def("__exit__", [](SceneBatchExecutorV2 &self, nb::handle,
                          nb::handle, nb::handle) {
        nb::gil_scoped_release release;
        self.Close();
      },
           nb::arg().none(), nb::arg().none(), nb::arg().none());

  nb::class_<SceneBatchExecutorV3, SceneBatchExecutorV2>(
      m, "SceneBatchExecutorV3", nb::is_weak_referenceable())
      .def("__init__",
           [](nb::pointer_and_handle<SceneBatchExecutorV3> executor,
              nb::sequence scenes, nb::sequence actors, nb::sequence links,
              nb::sequence actuatorCounts, nb::sequence contactSources,
              nb::sequence contactOthers, nb::sequence contactKinds,
              nb::sequence contactDistances, size_t numWorkers) {
             new (executor.p) SceneBatchExecutorV3(
                 scenes, actors, links, actuatorCounts, contactSources,
                 contactOthers, contactKinds, contactDistances, numWorkers);
             RegisterExecutorTeardown<SceneBatchExecutorV3>(executor.h);
           },
           nb::arg("scenes"), nb::arg("actors"), nb::arg("links"),
           nb::arg("actuator_counts"), nb::arg("contact_sources"),
           nb::arg("contact_others"), nb::arg("contact_kinds"),
           nb::arg("contact_distances"), nb::arg("num_workers"))
      .def_prop_ro("abi_version",
                             [](SceneBatchExecutorV3 const &) {
                               return SceneBatchExecutorV3::kAbiVersion;
                             })
      // ABI 2 methods and properties are inherited through the registered
      // SceneBatchExecutorV2 base. Only the constructor, ABI identity, and ABI 3
      // transaction are V3-specific.
      .def("write_boundary_conditions",
           &SceneBatchExecutorV3::WriteBoundaryConditions,
           nb::arg("values").none(), nb::arg("write_mask").none())
      .def("__enter__",
           [](SceneBatchExecutorV3 &self) -> SceneBatchExecutorV3 & {
             return self;
           },
           nb::rv_policy::reference);
}

} // namespace mochi
