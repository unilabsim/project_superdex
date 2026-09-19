# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample runner for registered SuperDex Gymnasium environments.

Usage:
    python run_sample.py <environment_id> [options]

Examples:
    python run_sample.py superdex_gym/CartPole-v0 --action_sampler random --num_episodes 5
    python run_sample.py superdex_gym/HalfCheetah-v0 --action_sampler sweep --video
"""

import argparse
import cProfile
import pathlib
import warnings
from typing import Any

import gymnasium as gym
from superdex.lab.gym.registration import get_env_specs
from superdex.lab.gym.utils.registry import MochiGymEnv, unwrap_mochi_env
from superdex.physics.utils.logging import configure_logger
from superdex.physics.viewer import VIEWER_AVAILABLE
from superdex.physics.viewer.utils import AnimationWriter

# Support both direct-script and package-module execution.
try:
    from sample_runner import random_action, sample_runner, sweep_action, zero_action
except ImportError:
    from .sample_runner import random_action, sample_runner, sweep_action, zero_action

########################################################################################


def make_env(env_id: str, common_env_cfg: dict[str, Any]) -> MochiGymEnv:
    """Build a registered environment, preserving variant defaults."""
    return gym.make(env_id, **common_env_cfg)


def _resolve_render_mode(requested: str, video_recording: bool) -> str | None:
    """Map the --render-mode selection to a Mochi ``render_mode`` value.

    ``none`` runs headless. ``auto`` keeps the historical behavior: use the viewer when
    available (``rgb_array`` when recording, otherwise ``human``) and fall back to no
    rendering otherwise. ``human``/``rgb_array`` are explicit and require the viewer,
    failing with an actionable error when it is unavailable. ``VIEWER_AVAILABLE`` only
    confirms a compatible Polyscope import, not that the display backend will start.
    """
    if requested == "none":
        return None
    if requested == "auto":
        if VIEWER_AVAILABLE:
            return "rgb_array" if video_recording else "human"
        warnings.warn(
            "Polyscope is not installed in the current environment, or it's an "
            "incompatible version. Please install Polyscope >= 2.5.0 to enable the "
            "renderer. Falling back to render mode none...",
            stacklevel=2,
        )
        return None
    if not VIEWER_AVAILABLE:
        raise SystemExit(
            f"--render-mode {requested} requires the Polyscope viewer (>= 2.5.0), which "
            "is not available in this environment. Rerun with --render-mode none for "
            "headless use."
        )
    return requested


def _resolve_render_and_video(
    render_mode: str,
    video_path: pathlib.Path | None,
    video_size: str | None,
) -> tuple[str | None, pathlib.Path | None, tuple[int, ...] | None]:
    """Resolve the renderer mode and normalize the video output settings.

    Returns the effective Mochi ``render_mode`` plus the (possibly disabled) video path
    and parsed size. Headless mode disables video output; ``human`` cannot record.
    """
    effective_render_mode = _resolve_render_mode(
        render_mode, video_recording=video_path is not None
    )
    if effective_render_mode is None:
        if video_path is not None:
            warnings.warn(
                "Rendering is disabled (--render-mode none); ignoring the requested "
                "video output.",
                stacklevel=2,
            )
        return None, None, None
    if effective_render_mode == "human" and video_path is not None:
        raise SystemExit(
            "--render-mode human cannot record video (the human viewer is on-screen). "
            "Use --render-mode rgb_array (or auto) together with --video."
        )
    parsed_size: tuple[int, ...] | None = None
    if video_size is not None:
        parsed_size = tuple(int(part) for part in video_size.split("x"))
        if len(parsed_size) != 2:
            raise ValueError(f"Invalid video size: {video_size}")
    return effective_render_mode, video_path, parsed_size


########################################################################################


def run_sample(
    env_id: str,
    action_sampler: str,
    num_episodes: int,
    render_mode: str,
    video_size: str | None,
    video_path: pathlib.Path | None,
    start_paused: bool,
    profile: bool,
):
    """Run a sample environment with the specified configuration and action sampler.

    This function creates an environment based on the sample name, selects an action
    sampling strategy, and runs the specified number of episodes while profiling
    the execution performance.
    """

    # Resolve renderer mode and normalize video output from the CLI selection.
    effective_render_mode, video_path, video_size = _resolve_render_and_video(
        render_mode, video_path, video_size
    )

    # Setup common parameters. These are runtime/presentation settings owned by the CLI,
    # so they are merged over the entry config and win. Task configuration deliberately
    # does not appear here: steps_per_episode would restate each env class default while
    # silently overriding a horizon supplied by a config variant, and num_worker_threads
    # already defaults to 0.
    common_env_cfg = {
        "render_mode": effective_render_mode,
        "render_size": video_size,
        "start_paused": start_paused,
        "profile": True,
    }

    # Create the environment to run and select the action sampler function.
    # fmt: off
    action_samplers = {
        "zero": zero_action,
        "random": random_action,
        "sweep": sweep_action,
    }
    # fmt: on

    if action_sampler not in action_samplers:
        raise ValueError(f"Unknown action sampler: {action_sampler}")

    # Building the env initializes the viewer backend when a render mode is active; that
    # can still fail on a host without a usable display even when VIEWER_AVAILABLE is
    # True, so turn it into an actionable message.
    try:
        env = make_env(env_id, common_env_cfg)
    except Exception as error:  # noqa: BLE001 -- surface an actionable renderer message
        if effective_render_mode is None:
            raise
        raise SystemExit(
            f"Failed to initialize the renderer backend for --render-mode "
            f"{effective_render_mode!r}: {error}. If this host has no display backend, "
            "rerun with --render-mode none for headless use."
        ) from error
    mochi_env = unwrap_mochi_env(env)
    action_sampler_fn = action_samplers[action_sampler]

    # Initialize animation writer.
    animation_writer = None
    if video_path is not None:
        fps = mochi_env.get_control_frequency()
        animation_writer = AnimationWriter(video_path, fps, "mp4")

    # With the environment instantiated, we can now step it.
    # See the definition of `sample_runner` for more details.
    if profile:
        pr = cProfile.Profile()
        pr.enable()
        sample_runner(env, mochi_env, action_sampler_fn, num_episodes, animation_writer)
        pr.disable()
        print()
        print("cProfile summary")
        pr.print_stats(sort="cumulative")
    else:
        sample_runner(env, mochi_env, action_sampler_fn, num_episodes, animation_writer)

    # Print the environment's profiler summary (if available).
    profiler = mochi_env.get_profiler()
    if profiler.enabled:
        print()
        print("Environment profiler summary")
        profiler.print_summary()


########################################################################################


def main():
    available_env_ids = tuple(spec.id for spec in get_env_specs())

    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Run SuperDex Gym sample environments with different action sampling strategies."
    )
    parser.add_argument(
        "env_id",
        type=str,
        help="Canonical environment ID to run. One of: " + ", ".join(available_env_ids),
    )
    parser.add_argument(
        "--action_sampler",
        type=str,
        default="sweep",
        help="Action sampling strategy to use (zero, random, sweep)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument(
        "--render-mode",
        dest="render_mode",
        type=str,
        default="auto",
        choices=("auto", "human", "rgb_array", "none"),
        help="Renderer mode. 'auto' (default) uses the viewer when available "
        "('human', or 'rgb_array' with --video) and falls back to 'none' otherwise; "
        "'none' runs headless (no display backend required); 'human'/'rgb_array' "
        "require the viewer.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable video recording of the environment. Setting this argument will "
        "make rendering happen offscreen.",
    )
    parser.add_argument(
        "--video_size",
        type=str,
        default=None,
        help=("Size of the video output (e.g., 1280x720, 1920x1080)"),
    )
    parser.add_argument(
        "--video_path",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to save video output (defaults to output/ folder if ony --video is "
            "used). This argument implies --video"
        ),
    )
    parser.add_argument(
        "--start_paused",
        action="store_true",
        help="Start the environment in paused state",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling during execution",
    )
    args = parser.parse_args()
    if args.env_id not in available_env_ids:
        parser.error(
            f"unknown SuperDex environment ID {args.env_id!r}; choose one of: "
            + ", ".join(available_env_ids)
        )

    # Setup logging.
    configure_logger()

    # Retrieve video path.
    # If not specified, then put it into a child "output" folder.
    video_path = args.video_path
    if video_path is not None:
        args.video = True
    if args.video and video_path is None:
        base_path = pathlib.Path(__file__).parent.resolve()
        video_path = base_path / "output"

    # Run the sample.
    run_sample(
        env_id=args.env_id,
        action_sampler=args.action_sampler,
        num_episodes=args.num_episodes,
        render_mode=args.render_mode,
        video_size=args.video_size,
        video_path=video_path,
        start_paused=args.start_paused,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
