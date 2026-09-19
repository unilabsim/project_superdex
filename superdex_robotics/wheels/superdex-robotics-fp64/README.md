# SuperDex Robotics FP64

FP64 extension for
[SuperDex Robotics](https://facebookresearch.github.io/project_superdex/robotics/).

Payload only: this distribution carries the FP64 build of the native robotics
extension. The `superdex.robotics` API lives in `superdex-robotics`, and the
FP64 physics libraries it links against live in `superdex-physics-fp64`; both
are dependencies of this distribution.

You do not normally install this by name. Ask for it through the extra:

```bash
pip install 'superdex-robotics[fp64]'
export SUPERDEX_PRECISION=fp64
```

Precision is process-wide and resolved at import time, so set `SUPERDEX_PRECISION` before
the first `import superdex.robotics`.

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## License

First-party code in this distribution is Apache-2.0 licensed; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
Third-party code and dependencies retain their own terms.
