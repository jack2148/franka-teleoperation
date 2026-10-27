# Source/reference installation and validation

This release supports source inspection and the documented CPU test paths. It does not install or reconstruct the original OMY executable environment. Read the [release contract](README.md) before using historical commands.

## OMY mathematical tests

With Python, NumPy and pytest installed, run from this repository:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

These six tests require neither ROS nor the historical scenes. Four execute selected pure functions from the shipped bridge; two retain earlier mathematical checks. They do not initialize the historical launcher or certify model/contact behavior.

## Full Geomagic CPU regression

Keep the candidate repositories side by side:

```text
workspace/
  franka-teleoperation/
  franka-imitation-learning/
```

Use the companion's `INSTALL.md` and `requirements-core.txt` for the documented Python environment and pinned LeRobot dependency. The recorded environment uses Linux, Python 3.12.14, MuJoCo 3.12.0, NumPy 2.2.6 and pytest 8.3.5. CPU rendering requires OSMesa; native force tests require a C++17 compiler. Do not put the original research workspace on `PYTHONPATH`.

After installing those dependencies, from `franka-imitation-learning/` in Bash:

```sh
. scripts/env.sh
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  ../franka-teleoperation/tests ../franka-teleoperation/geomagic/tests
```

This selects the 678 teleoperation CPU tests. Integration tests use the companion runtime and model dependencies, while device/protocol tests retain their existing synthetic fixtures. It does not start training, a physical device or the OMY ROS launcher. [Validation evidence](docs/VALIDATION.md) distinguishes this test run from installation on a fresh OS/driver stack.

## Device-dependent source

OpenHaptics headers/runtime, device permissions and physical Touch hardware are external prerequisites. The SDK is not redistributed. `hardware_validation/test_hdapi_sdk.py` is an explicit SDK-dependent check, not part of the default CPU suite. Physical haptic acceptance, device timing and hard real-time operation are not certified by this release. The ordinary Touch UI lives in the companion repository and follows its installation instructions.

## Historical OMY reference paths — not an installation recipe

`teleop`, `simul`, `launch/`, `src/` and `config/` preserve the historical layout. Their old execution path expects these absent files at this repository root:

| Historical path | Status |
| --- | --- |
| `mujoco_menagerie/franka_fr3/scene.xml` | Original FR3 scene with custom front wall unavailable |
| `robotis_mujoco_menagerie/robotis_omy/scene.xml` | Exact native OMY bundle/revision not established |

Historical ROS2/OMY bringup dependencies are also external. `simul` reports the missing FR3 scene if invoked; the launcher and controller behavior have not been changed. These paths are reference information, not instructions to download replacement assets. Do not fill them with a modified companion model, current upstream model or reconstructed wall/sensor patch and describe it as the original environment.

Exact asset recovery is stopped. See [historical limitations](docs/HISTORICAL_OMY_LIMITATIONS.md). No physical Franka deployment is provided or claimed.
