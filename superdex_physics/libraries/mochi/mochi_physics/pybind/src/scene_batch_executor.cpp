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
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
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

class SceneBatchExecutor {
public:
  static constexpr uint32_t kReadQpos = 1u << 0;
  static constexpr uint32_t kReadQvel = 1u << 1;
  static constexpr uint32_t kReadLinks = 1u << 2;
  static constexpr uint32_t kReadContacts = 1u << 3;
  static constexpr uint32_t kReadDiverged = 1u << 4;
  static constexpr uint32_t kReadAll = kReadQpos | kReadQvel | kReadLinks |
                                        kReadContacts | kReadDiverged;

  SceneBatchExecutor(nb::sequence scenes, nb::sequence actors,
                     nb::sequence links, nb::sequence contactSources,
                     nb::sequence contactOthers, nb::sequence contactKinds,
                     nb::sequence contactDistances, size_t numWorkers) {
    CheckContext();
    if (numWorkers == 0) {
      throw std::invalid_argument(
          "SceneBatchExecutor requires num_workers >= 1");
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
          "SceneBatchExecutor requires initialize(num_worker_threads=0); "
          "do not combine outer scene workers with SDK worker threads");
    }

    std::unordered_set<Scene *> uniqueScenes;
    std::unordered_set<Actor *> uniqueActors;
    int dofCount = -1;
    int linkCount = -1;
    int contactCount = -1;
    if (nb::len(contactKinds) != nb::len(contactDistances)) {
      throw std::invalid_argument(
          "contact_kinds and contact_distances must have equal length");
    }
    for (size_t i = 0; i < static_cast<size_t>(nb::len(contactKinds)); ++i) {
      auto kind = nb::cast<int>(contactKinds[i]);
      auto distance = nb::cast<double>(contactDistances[i]);
      if (kind < 0 || kind > 2 || !std::isfinite(distance) || distance < 0) {
        throw std::invalid_argument(
            "contact kinds must be 0..2 and distances finite/non-negative");
      }
      _contactKinds.push_back(kind);
      _contactDistances.push_back(static_cast<real>(distance));
    }
    for (size_t i = 0; i < static_cast<size_t>(nb::len(scenes)); ++i) {
      auto scene = nb::cast<Scene *>(scenes[i]);
      auto actor = nb::cast<Actor *>(actors[i]);
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

      auto sceneLinks = nb::cast<nb::sequence>(links[i]);
      if (linkCount < 0) {
        linkCount = static_cast<int>(nb::len(sceneLinks));
      } else if (static_cast<int>(nb::len(sceneLinks)) != linkCount) {
        throw std::invalid_argument("all link actor lists must have equal length");
      }
      std::vector<Actor *> nativeLinks;
      nativeLinks.reserve(static_cast<size_t>(nb::len(sceneLinks)));
      for (size_t linkIndex = 0; linkIndex < static_cast<size_t>(nb::len(sceneLinks));
           ++linkIndex) {
        auto link = nb::cast<Actor *>(sceneLinks[linkIndex]);
        if (!link || link->GetScene() != scene) {
          throw std::invalid_argument("link actors must belong to their scene");
        }
        nativeLinks.push_back(link);
      }
      _links.push_back(std::move(nativeLinks));

      auto sceneSources = nb::cast<nb::sequence>(contactSources[i]);
      auto sceneOthers = nb::cast<nb::sequence>(contactOthers[i]);
      if (static_cast<int>(nb::len(sceneSources)) !=
              static_cast<int>(_contactKinds.size()) ||
          nb::len(sceneSources) != nb::len(sceneOthers)) {
        throw std::invalid_argument(
            "each contact source/other list must match contact_kinds");
      }
      std::vector<Actor *> nativeSources;
      std::vector<Actor *> nativeOthers;
      nativeSources.reserve(_contactKinds.size());
      nativeOthers.reserve(_contactKinds.size());
      for (size_t contactIndex = 0; contactIndex < _contactKinds.size();
           ++contactIndex) {
        auto source = nb::cast<Actor *>(sceneSources[contactIndex]);
        if (!source || source->GetScene() != scene) {
          throw std::invalid_argument("contact sources must belong to their scene");
        }
        nativeSources.push_back(source);
        if (sceneOthers[contactIndex].is_none()) {
          nativeOthers.push_back(nullptr);
        } else {
          auto other = nb::cast<Actor *>(sceneOthers[contactIndex]);
          if (!other || other->GetScene() != scene) {
            throw std::invalid_argument("contact actors must belong to their scene");
          }
          nativeOthers.push_back(other);
        }
      }
      if (contactCount < 0) {
        contactCount = static_cast<int>(nativeSources.size());
      } else if (static_cast<int>(nativeSources.size()) != contactCount) {
        throw std::invalid_argument("all contact actor lists must have equal length");
      }
      _contactSources.push_back(std::move(nativeSources));
      _contactOthers.push_back(std::move(nativeOthers));
    }
    _dofCount = dofCount;
    _linkCount = std::max(0, linkCount);
    _contactCount = std::max(0, contactCount);
    _dofIndices.resize(static_cast<size_t>(_dofCount));
    for (int i = 0; i < _dofCount; ++i) {
      _dofIndices[static_cast<size_t>(i)] = i;
    }

    _workers.reserve(numWorkers);
    detail::RegisterLeasedScenes(_scenes);
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
      detail::ReleaseLeasedScenes(_scenes);
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
  [[nodiscard]] int GetNumLinks() const { return _linkCount; }
  [[nodiscard]] int GetNumContacts() const { return _contactCount; }
  [[nodiscard]] bool IsClosed() const { return _closed; }

  void Step(double timeStepSec,
            RealConstArray generalizedForces,
            nb::object qposOut, nb::object qvelOut, nb::object linkStateOut,
            nb::object contactOut, nb::object divergedOut,
            uint32_t readbackMask) {
    CheckContext();
    if (!std::isfinite(timeStepSec) || timeStepSec < 0) {
      throw std::invalid_argument(
          "time_step_sec must be finite and non-negative");
    }
    ValidateArray("generalized_forces", generalizedForces);
    if ((readbackMask & ~kReadAll) != 0) {
      throw std::invalid_argument("readback_mask contains unknown fields");
    }
    auto qpos = RequireArray("qpos_out", qposOut, readbackMask & kReadQpos);
    auto qvel = RequireArray("qvel_out", qvelOut, readbackMask & kReadQvel);
    auto links = RequireArray("link_state_out", linkStateOut,
                              readbackMask & kReadLinks, 3, _linkCount, 16);
    auto contacts = RequireArray("contact_out", contactOut,
                                 readbackMask & kReadContacts, 3, _contactCount, 3);
    auto diverged = RequireVector("diverged_out", divergedOut,
                                  readbackMask & kReadDiverged);

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
    auto *qposData = qpos ? qpos->data() : nullptr;
    auto *qvelData = qvel ? qvel->data() : nullptr;
    auto *linkStateData = links ? links->data() : nullptr;
    auto *contactData = contacts ? contacts->data() : nullptr;
    auto *divergedData = diverged ? diverged->data() : nullptr;
    std::exception_ptr failure;
    {
      // Keep all pybind arrays alive with the GIL held. Only the pure C++
      // scheduler barrier may execute without it.
      nb::gil_scoped_release release;
      failure = DispatchAndWait(timeStepSec, forceData, qposData, qvelData,
                                linkStateData, contactData, divergedData,
                                readbackMask);
    }
    if (failure) {
      std::rethrow_exception(failure);
    }
  }

  void StepControl(double timeStepSec, RealConstArray controls,
                   IntConstArray qposIndices,
                   IntConstArray qvelIndices,
                   RealConstArray kp,
                   RealConstArray kd,
                   RealConstArray gear,
                   RealConstArray forceRanges,
                   int numSteps, nb::object qposOut, nb::object qvelOut,
                   nb::object linkStateOut, nb::object contactOut,
                   nb::object divergedOut, uint32_t readbackMask) {
    CheckContext();
    if (numSteps < 1 || !std::isfinite(timeStepSec) || timeStepSec < 0) {
      throw std::invalid_argument("step_control requires positive num_steps and finite dt");
    }
    if (controls.ndim() != 2 || controls.shape(0) != static_cast<size_t>(_scenes.size())) {
      throw std::invalid_argument("controls must have shape [num_scenes, num_actuators]");
    }
    auto const actuatorCount = static_cast<size_t>(controls.shape(1));
    auto validateVector = [actuatorCount](char const *name, auto const &array) {
      if (array.ndim() != 1 || array.shape(0) != static_cast<size_t>(actuatorCount)) {
        throw std::invalid_argument(std::string(name) + " must have shape [num_actuators]");
      }
    };
    validateVector("qpos_indices", qposIndices);
    validateVector("qvel_indices", qvelIndices);
    validateVector("kp", kp);
    validateVector("kd", kd);
    validateVector("gear", gear);
    if (forceRanges.ndim() != 2 || forceRanges.shape(0) != static_cast<size_t>(actuatorCount) ||
        forceRanges.shape(1) != 2) {
      throw std::invalid_argument("force_ranges must have shape [num_actuators, 2]");
    }
    for (size_t i = 0; i < actuatorCount; ++i) {
      auto const q = qposIndices.data()[i], v = qvelIndices.data()[i];
      if (q < 0 || q >= _dofCount || v < 0 || v >= _dofCount ||
          !std::isfinite(kp.data()[i]) || !std::isfinite(kd.data()[i]) ||
          !std::isfinite(gear.data()[i]) || !std::isfinite(forceRanges.data()[2 * i + 0]) ||
          !std::isfinite(forceRanges.data()[2 * i + 1]) || forceRanges.data()[2 * i + 0] > forceRanges.data()[2 * i + 1]) {
        throw std::invalid_argument("invalid actuator control metadata");
      }
    }
    if (readbackMask & ~kReadAll) throw std::invalid_argument("readback_mask contains unknown fields");
    auto qpos = RequireArray("qpos_out", qposOut, readbackMask & kReadQpos);
    auto qvel = RequireArray("qvel_out", qvelOut, readbackMask & kReadQvel);
    auto links = RequireArray("link_state_out", linkStateOut, readbackMask & kReadLinks, 3, _linkCount, 16);
    auto contacts = RequireArray("contact_out", contactOut, readbackMask & kReadContacts, 3, _contactCount, 3);
    auto diverged = RequireVector("diverged_out", divergedOut, readbackMask & kReadDiverged);
    std::unique_lock callLock(_callMutex);
    if (_closed) throw std::runtime_error("SceneBatchExecutor is closed");
    auto const *controlData = controls.data();
    auto *qposData = qpos ? qpos->data() : nullptr;
    auto *qvelData = qvel ? qvel->data() : nullptr;
    auto *linkData = links ? links->data() : nullptr;
    auto *contactData = contacts ? contacts->data() : nullptr;
    auto *divergedData = diverged ? diverged->data() : nullptr;
    {
      nb::gil_scoped_release release;
      {
        std::lock_guard lock(_mutex);
        _controlMode = true;
        _controlSteps = numSteps;
        _controlData = controlData;
        _controlQposIndices = qposIndices.data(); _controlQvelIndices = qvelIndices.data();
        _controlKp = kp.data(); _controlKd = kd.data(); _controlGear = gear.data();
        _controlForceRanges = forceRanges.data(); _controlActuatorCount = actuatorCount;
      }
      auto failure = DispatchAndWait(timeStepSec, nullptr, qposData, qvelData, linkData,
                                     contactData, divergedData, readbackMask);
      if (failure) std::rethrow_exception(failure);
      std::lock_guard lock(_mutex);
      _controlMode = false;
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
      detail::ReleaseLeasedScenes(_scenes);
      _leasesRegistered = false;
    }
    _closed = true;
  }

private:
  std::exception_ptr DispatchAndWait(double timeStepSec, real const *forceData,
                                     real *qposData, real *qvelData,
                                     real *linkStateData, real *contactData,
                                     uint8_t *divergedData,
                                     uint32_t readbackMask) {
    {
      std::lock_guard lock(_mutex);
      _failure = nullptr;
      _nextIndex.store(0, std::memory_order_relaxed);
      _completedWorkers = 0;
      _timeStepSec = timeStepSec;
      _forceData = forceData;
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
    _allDone.wait(lock, [this]() {
      return _completedWorkers == _workers.size();
    });
    return _failure;
  }

  void RunScene(size_t i) {
    auto const offset = i * static_cast<size_t>(_dofCount);
    Error error;
    thread_local std::vector<real> pose;
    thread_local std::vector<real> velocity;
    thread_local std::vector<real> forces;
    pose.resize(static_cast<size_t>(_dofCount));
    velocity.resize(static_cast<size_t>(_dofCount));
    forces.resize(static_cast<size_t>(_dofCount));
    for (int controlStep = 0; controlStep < (_controlMode ? _controlSteps : 1); ++controlStep) {
      if (_controlMode) {
        _actors[i]->GetArticulatedPose(Span<real>(pose), error);
        if (!error.IsOK()) throw MochiErrorException(error);
        _actors[i]->GetArticulatedJointVelocities(Span<real>(velocity), error);
        if (!error.IsOK()) throw MochiErrorException(error);
        std::fill(forces.begin(), forces.end(), 0);
        auto const *ctrl = _controlData + i * _controlActuatorCount;
        for (size_t a = 0; a < _controlActuatorCount; ++a) {
          auto value = _controlKp[a] > 0
                           ? _controlKp[a] * (ctrl[a] - pose[_controlQposIndices[a]] * _controlGear[a])
                               - _controlKd[a] * velocity[_controlQvelIndices[a]] * _controlGear[a]
                           : ctrl[a];
          value = std::clamp(value, _controlForceRanges[2 * a], _controlForceRanges[2 * a + 1]);
          forces[static_cast<size_t>(_controlQvelIndices[a])] += value * _controlGear[a];
        }
      }
      auto const *forceData = _controlMode ? forces.data() : _forceData + offset;
      _actors[i]->SetExternalForcesOnDofs(MakeConstSpan(_dofIndices),
          Span<real const>(forceData, static_cast<size_t>(_dofCount)), error);
      if (!error.IsOK()) throw MochiErrorException(error);
      _scenes[i]->Step(_timeStepSec);
    }
    if (_readbackMask & kReadDiverged) {
      _divergedData[i] = _scenes[i]->GetSolverStats().convergenceStatus ==
                                 ConvergenceStatus::Diverged
                             ? 1
                             : 0;
    }
    if (_readbackMask & kReadQpos) {
      _actors[i]->GetArticulatedPose(
          Span<real>(_qposData + offset, static_cast<size_t>(_dofCount)), error);
      if (!error.IsOK()) {
        throw MochiErrorException(error);
      }
    }
    if (_readbackMask & kReadQvel) {
      _actors[i]->GetArticulatedJointVelocities(
          Span<real>(_qvelData + offset, static_cast<size_t>(_dofCount)), error);
      if (!error.IsOK()) {
        throw MochiErrorException(error);
      }
    }
    if (_readbackMask & kReadLinks) {
      auto const linkOffset = i * static_cast<size_t>(_linkCount) * 16;
      thread_local std::vector<TransformRT> transforms;
      transforms.resize(static_cast<size_t>(_linkCount));
      _actors[i]->GetArticulatedLinkTransforms(MakeSpan(transforms), error);
      if (!error.IsOK()) {
        throw MochiErrorException(error);
      }
      for (int linkIndex = 0; linkIndex < _linkCount; ++linkIndex) {
        auto const &transform = transforms[static_cast<size_t>(linkIndex)];
        auto const position = transform.GetTranslation();
        auto const rotation = transform.GetRotation().ToReal4();
        auto *out = _linkStateData + linkOffset +
                    static_cast<size_t>(linkIndex) * 16;
        out[0] = position[0];
        out[1] = position[1];
        out[2] = position[2];
        out[3] = rotation[3];
        out[4] = rotation[0];
        out[5] = rotation[1];
        out[6] = rotation[2];
        auto const com = _links[i][static_cast<size_t>(linkIndex)]
                             ->GetCenterOfMassTransform(error);
        if (!error.IsOK()) {
          throw MochiErrorException(error);
        }
        auto const comPosition = com.GetTranslation();
        auto const linear = _links[i][static_cast<size_t>(linkIndex)]
                                ->GetLinearVelocity(error);
        auto const angular = _links[i][static_cast<size_t>(linkIndex)]
                                 ->GetAngularVelocity(error);
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
      auto const contactOffset = i * static_cast<size_t>(_contactCount) * 3;
      for (int contactIndex = 0; contactIndex < _contactCount; ++contactIndex) {
        auto *out = _contactData + contactOffset +
                    static_cast<size_t>(contactIndex) * 3;
        out[0] = 0;
        out[1] = 0;
        out[2] = 0;
        auto *source = _contactSources[i][static_cast<size_t>(contactIndex)];
        auto *other = _contactOthers[i][static_cast<size_t>(contactIndex)];
        if (_contactKinds[static_cast<size_t>(contactIndex)] == 0) {
          auto points = source->GetContactPointsWorld(error);
          if (!error.IsOK()) {
            throw MochiErrorException(error);
          }
          auto const own = source->GetHandle();
          auto const otherHandle = other ? other->GetHandle() : ActorHandle{};
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
          auto value = _contactKinds[static_cast<size_t>(contactIndex)] == 1
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

  template <typename Array>
  void ValidateArray(char const *name, Array const &array,
                     int ndim = 2, int dim1 = -1, int dim2 = -1) const {
    if (array.ndim() != ndim ||
        array.shape(0) != static_cast<size_t>(_scenes.size()) ||
        (ndim == 2 && array.shape(1) != _dofCount) ||
        (ndim == 3 && (array.shape(1) != dim1 || array.shape(2) != dim2))) {
      throw std::invalid_argument(
          std::string(name) + " has unexpected shape (ndim=" +
          std::to_string(array.ndim()) + ", shape0=" +
          std::to_string(array.shape(0)) + ", expected ndim=" +
          std::to_string(ndim) + ", dim1=" + std::to_string(dim1) +
          ", dim2=" + std::to_string(dim2) + ", links=" +
          std::to_string(_linkCount) + ", actual1=" +
          (array.ndim() > 1 ? std::to_string(array.shape(1)) : "-") +
          ", actual2=" + (array.ndim() > 2 ? std::to_string(array.shape(2)) : "-") + ")");
    }
  }

  template <typename Array>
  void ValidateVector(char const *name, Array const &array) const {
    if (array.ndim() != 1 ||
        array.shape(0) != static_cast<size_t>(_scenes.size())) {
      throw std::invalid_argument(std::string(name) +
                                  " must have shape [num_scenes]");
    }
  }

  std::optional<RealArray> RequireArray(
      char const *name, nb::object const &object, bool requested, int ndim = 2,
      int dim1 = -1, int dim2 = -1) const {
    if (!requested) {
      if (!object.is_none()) {
        throw std::invalid_argument(
            std::string(name) + " must be None when readback_mask does not "
                                "request it");
      }
      return std::nullopt;
    }
    if (object.is_none()) {
      throw std::invalid_argument(std::string(name) +
                                  " is required by readback_mask");
    }
    auto array = nb::cast<RealArray>(object);
    ValidateArray(name, array, ndim, dim1, dim2);
    return array;
  }

  std::optional<Uint8Array> RequireVector(
      char const *name, nb::object const &object, bool requested) const {
    if (!requested) {
      if (!object.is_none()) {
        throw std::invalid_argument(std::string(name) +
                                    " must be None when readback is disabled");
      }
      return std::nullopt;
    }
    if (object.is_none()) {
      throw std::invalid_argument(std::string(name) +
                                  " is required by readback_mask");
    }
    auto array = nb::cast<Uint8Array>(object);
    ValidateVector(name, array);
    return array;
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

  std::vector<Scene *> _scenes;
  std::vector<Actor *> _actors;
  std::vector<std::vector<Actor *>> _links;
  std::vector<std::vector<Actor *>> _contactSources;
  std::vector<std::vector<Actor *>> _contactOthers;
  std::vector<int> _contactKinds;
  std::vector<real> _contactDistances;
  std::vector<int> _dofIndices;
  int _dofCount = 0;
  int _linkCount = 0;
  int _contactCount = 0;
  std::vector<std::thread> _workers;
  std::atomic<size_t> _nextIndex{0};
  double _timeStepSec = 0;
  real const *_forceData = nullptr;
  real *_qposData = nullptr;
  real *_qvelData = nullptr;
  real *_linkStateData = nullptr;
  real *_contactData = nullptr;
  uint8_t *_divergedData = nullptr;
  uint32_t _readbackMask = kReadAll;
  bool _controlMode = false;
  int _controlSteps = 1;
  size_t _controlActuatorCount = 0;
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

} // namespace

namespace {

template <typename Executor>
void RegisterExecutorTeardown(nb::handle executorObject) {
  // The context registry is never cleared. Hold a weak Python reference rather
  // than the C++ object so an executor can be collected before mochi.shutdown().
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

void DefineSceneBatchExecutor(nb::module_ &m) {
  nb::class_<SceneBatchExecutor>(m, "SceneBatchExecutor",
                                 nb::is_weak_referenceable())
      .def("__init__",
           [](nb::pointer_and_handle<SceneBatchExecutor> executor,
              nb::sequence scenes, nb::sequence actors, nb::sequence links,
              nb::sequence contactSources, nb::sequence contactOthers,
              nb::sequence contactKinds, nb::sequence contactDistances,
              size_t numWorkers) {
             new (executor.p) SceneBatchExecutor(
                 scenes, actors, links, contactSources, contactOthers,
                 contactKinds, contactDistances, numWorkers);
             RegisterExecutorTeardown<SceneBatchExecutor>(executor.h);
           },
           nb::arg("scenes"), nb::arg("actors"), nb::arg("links"),
           nb::arg("contact_sources"), nb::arg("contact_others"),
           nb::arg("contact_kinds"), nb::arg("contact_distances"),
           nb::arg("num_workers"))
      .def_prop_ro("num_workers", &SceneBatchExecutor::GetNumWorkers)
      .def_prop_ro("num_scenes", &SceneBatchExecutor::GetNumScenes)
      .def_prop_ro("num_dofs", &SceneBatchExecutor::GetNumDofs)
      .def_prop_ro("num_links", &SceneBatchExecutor::GetNumLinks)
      .def_prop_ro("num_contacts", &SceneBatchExecutor::GetNumContacts)
      .def_prop_ro("closed", &SceneBatchExecutor::IsClosed)
      .def("step_control",
           [](SceneBatchExecutor& self, double dt,
              RealConstArray controls,
              IntConstArray qpos_indices,
              IntConstArray qvel_indices,
              RealConstArray kp,
              RealConstArray kd,
              RealConstArray gear,
              RealConstArray force_ranges,
              int num_steps, nb::object qpos_out, nb::object qvel_out,
              nb::object link_state_out, nb::object contact_out,
              nb::object diverged_out, uint32_t readback_mask) {
             self.StepControl(dt, controls, qpos_indices, qvel_indices, kp, kd,
                              gear, force_ranges, num_steps, qpos_out, qvel_out,
                              link_state_out, contact_out, diverged_out,
                              readback_mask);
           },
           nb::arg("time_step_sec"), nb::arg("controls"),
           nb::arg("qpos_indices"), nb::arg("qvel_indices"), nb::arg("kp"),
           nb::arg("kd"), nb::arg("gear"), nb::arg("force_ranges"),
           nb::arg("num_steps"), nb::arg("qpos_out").none(),
           nb::arg("qvel_out").none(), nb::arg("link_state_out").none(),
           nb::arg("contact_out").none(), nb::arg("diverged_out").none(),
           nb::arg("readback_mask"))
      .def(
          "step",
          [](SceneBatchExecutor& self, double timeStepSec,
             RealConstArray generalizedForces,
             nb::object qposOut, nb::object qvelOut, nb::object linkStateOut,
             nb::object contactOut, nb::object divergedOut,
             uint32_t readbackMask) {
            self.Step(timeStepSec, generalizedForces, qposOut, qvelOut,
                      linkStateOut, contactOut, divergedOut, readbackMask);
          },
          nb::arg("time_step_sec"), nb::arg("generalized_forces"),
          nb::arg("qpos_out").none(), nb::arg("qvel_out").none(),
          nb::arg("link_state_out").none(), nb::arg("contact_out").none(),
          nb::arg("diverged_out").none(),
          nb::arg("readback_mask"))
      .def("close", [](SceneBatchExecutor& self) {
        nb::gil_scoped_release release;
        self.Close();
      })
      .def(
          "__enter__",
          [](SceneBatchExecutor &self) -> SceneBatchExecutor & { return self; },
          nb::rv_policy::reference)
      .def("__exit__", [](SceneBatchExecutor& self, nb::handle, nb::handle,
                          nb::handle) {
        nb::gil_scoped_release release;
        self.Close();
      },
           nb::arg().none(), nb::arg().none(), nb::arg().none());
}

void OverrideLeasedSceneDestroy(nb::module_ &m) {
  nb::delattr(m, "destroy_scene");
  m.def(
      "destroy_scene",
      [](Scene *scene) {
        CheckContext();
        if (scene && detail::IsLeasedScene(scene)) {
          throw std::runtime_error(
              "close SceneBatchExecutor before destroying one of its scenes");
        }
        GetContext()->DestroyScene(scene);
      },
      nb::arg("scene"));
}

void OverrideLeasedActorDestroy(nb::module_ &m) {
  auto sceneClass = nb::borrow<nb::class_<Scene>>(m.attr("Scene"));
  nb::delattr(sceneClass, "destroy_actor");
  sceneClass.def(
      "destroy_actor",
      [](Scene &self, Actor *actor) {
        CheckContext();
        if (actor && detail::IsLeasedScene(actor->GetScene())) {
          throw std::runtime_error(
              "close SceneBatchExecutor before destroying one of its actors");
        }
        self.DestroyActor(actor);
      },
      nb::arg("actor"),
      "Destroy an actor and remove it from the scene.\n\n"
      "Raises:\n"
      "    RuntimeError: If the actor belongs to a leased SceneBatchExecutor "
      "scene. Close the executor first.");
  sceneClass.def(
      "destroy_actor",
      [](Scene &self, ActorHandle actor) {
        CheckContext();
        auto *nativeActor = self.GetActor(actor);
        if (nativeActor && detail::IsLeasedScene(nativeActor->GetScene())) {
          throw std::runtime_error(
              "close SceneBatchExecutor before destroying one of its actors");
        }
        self.DestroyActor(actor);
      },
      nb::arg("actor"),
      "Destroy an actor and remove it from the scene.\n\n"
      "Raises:\n"
      "    RuntimeError: If the actor belongs to a leased SceneBatchExecutor "
      "scene. Close the executor first.");
}

void OverrideLeasedSceneCallbacks(nb::module_& m) {
  auto sceneClass = m.attr("Scene");
  auto registerCallback = [](Scene& scene, std::string_view debugName,
                             std::function<void(StepInfo const&)> callback, int priority,
                             bool preStep) {
    if (detail::IsLeasedScene(&scene)) {
      throw std::runtime_error(
          "SceneBatchExecutor scenes do not support Python step callbacks");
    }
    return preStep ? scene.RegisterPreStepCallback(debugName, std::move(callback), priority)
                   : scene.RegisterPostStepCallback(debugName, std::move(callback), priority);
  };
  nb::delattr(sceneClass, "register_pre_step_callback");
  nb::delattr(sceneClass, "register_post_step_callback");
  sceneClass.attr("register_pre_step_callback") = nb::cpp_function(
      [registerCallback](Scene& scene, std::string_view debugName,
                         std::function<void(StepInfo const&)> callback, int priority) {
        return registerCallback(scene, debugName, std::move(callback), priority, true);
      },
      nb::is_method());
  sceneClass.attr("register_post_step_callback") = nb::cpp_function(
      [registerCallback](Scene& scene, std::string_view debugName,
                         std::function<void(StepInfo const&)> callback, int priority) {
        return registerCallback(scene, debugName, std::move(callback), priority, false);
      },
      nb::is_method());
}

} // namespace mochi
