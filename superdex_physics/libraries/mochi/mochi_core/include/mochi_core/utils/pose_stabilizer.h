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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/transform_rt.h>

namespace mochi {

/** @brief Pose stabilization strategy. */
enum class PoseStabilizerType {
  /** @brief Disable pose stabilization. */
  None,
  /** @brief Hold the last valid pose while tracking confidence is low. */
  LastValid,
  /** @brief Blend from the last valid pose when tracking confidence recovers. */
  LinearInterpolation
};

/** @brief Settings shared by pose stabilizers. */
struct PoseStabilizerParams {
  /** @brief Stabilization strategy to apply. */
  PoseStabilizerType stabilizerType = PoseStabilizerType::LinearInterpolation;
  /** @brief Confidence below which a pose is considered invalid. */
  real confidenceThreshold = 0.7_r;
  /** @brief Number of steps used to blend back to the input pose. Must be positive. */
  int numInterpolationSteps = 10;
};

/**
 * @brief Holds the last valid pose while tracking confidence is below a threshold.
 */
class LastValidPoseStabilizer {
 public:
  /**
   * @brief Construct a stabilizer with an initial valid pose.
   *
   * @param[in] confidenceThreshold Confidence below which a pose is considered invalid.
   * @param[in] initialPose Initial pose to output until the first valid pose is processed.
   */
  LastValidPoseStabilizer(real confidenceThreshold, Span<TransformRT const> initialPose);

  /**
   * @brief Stabilize an input pose.
   *
   * @param[in] confidence Confidence of the input pose.
   * @param[in] pose Input pose.
   * @param[out] outPose Input pose when confidence is at least the threshold; otherwise, the last
   * valid pose.
   *
   * @note @p pose and @p outPose must have the same size as the initial pose.
   */
  void Process(real confidence, Span<TransformRT const> pose, Span<TransformRT> outPose);

  /**
   * @brief Update the confidence threshold without resetting the last valid pose.
   *
   * @param[in] params Settings to apply. Only @ref PoseStabilizerParams::confidenceThreshold is
   * used.
   */
  void UpdateSettings(PoseStabilizerParams const& params);

 private:
  real _confidenceThreshold;
  DynamicArray<TransformRT> _lastValidPose;
};

/**
 * @brief Smoothly resumes pose updates after tracking confidence recovers.
 *
 * @details Holds the last valid pose while confidence is below the threshold. When confidence
 * exceeds the threshold, blends from the held pose toward the input pose over the configured
 * number of steps, then tracks the input pose directly.
 */
class LinearInterpolationPoseStabilizer {
 public:
  /**
   * @brief Construct a stabilizer with an initial valid pose.
   *
   * @param[in] confidenceThreshold Confidence below which a pose is considered invalid. Recovery
   * begins once confidence exceeds this threshold.
   * @param[in] interpolationSteps Number of steps used to resume direct pose updates. Must be
   * positive.
   * @param[in] initialPose Initial pose to output until the first valid pose is processed.
   */
  LinearInterpolationPoseStabilizer(
      real confidenceThreshold,
      int interpolationSteps,
      Span<TransformRT const> initialPose);

  /**
   * @brief Stabilize an input pose.
   *
   * @param[in] confidence Confidence of the input pose.
   * @param[in] pose Input pose.
   * @param[out] outPose Last valid pose while confidence is low, a blend toward the input pose
   * during recovery, or the input pose once recovery is complete.
   *
   * @note @p pose and @p outPose must have the same size as the initial pose.
   */
  void Process(real confidence, Span<TransformRT const> pose, Span<TransformRT> outPose);

  /**
   * @brief Update the confidence threshold and interpolation step count without resetting state.
   *
   * @param[in] params Settings to apply. @ref PoseStabilizerParams::confidenceThreshold and @ref
   * PoseStabilizerParams::numInterpolationSteps are used.
   *
   * @note @ref PoseStabilizerParams::numInterpolationSteps must be positive.
   */
  void UpdateSettings(PoseStabilizerParams const& params);

 private:
  enum StabilizerState { Unstable, Interpolating, Stable };
  StabilizerState _stabilizerState = StabilizerState::Stable;
  real _confidenceThreshold;
  int _interpolationSteps = 0;
  int _interpolationCount = 0;
  DynamicArray<TransformRT> _lastValidPose;
  DynamicArray<TransformRT> _lastReferencePose;

  void UpdateInternalState(real confidence);
  void UpdateInterpolationReference(Span<TransformRT const> pose);
  void ComputeInterpolatedPose(Span<TransformRT const> pose, Span<TransformRT> outPose) const;
};

} // namespace mochi
