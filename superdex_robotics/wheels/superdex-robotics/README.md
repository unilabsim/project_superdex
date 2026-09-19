# SuperDex Robotics

[SuperDex Robotics](https://facebookresearch.github.io/project_superdex/robotics/) is a
robotics SDK that provides robot definitions and composition, controllers, sensors,
actuators, and the framework that aggregates them into complete simulation configs.

Provides the `superdex.robotics` package: robot loading, compositing, control, actuation, and sensing built on top of `superdex-physics`.

```bash
pip install superdex-robotics
```

This wheel carries the FP32 native extension. For FP64, install the `fp64` extra -- which
pulls in `superdex-robotics-fp64` -- and select it at import time:

```bash
pip install 'superdex-robotics[fp64]'
export SUPERDEX_PRECISION=fp64
```

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## License

First-party code in this distribution is Apache-2.0 licensed; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
Third-party code and dependencies retain their own terms.
