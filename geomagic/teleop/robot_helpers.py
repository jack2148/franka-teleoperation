"""Read the existing robot-side functions without importing its ROS runner.

The existing controller-only loader still retains a nested ROS import try-block,
viewer imports and IPC imports. This stricter allowlist compiles ONLY named FR3
math definitions/scalar constants from the trusted local source. No DLS formula
is copied; source filenames/line numbers remain attached to the functions.
"""

import ast
from functools import lru_cache
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace


MIMIC_ROOT = Path(os.environ.get("FRANKA_LEARNING_ROOT", str(Path(__file__).resolve().parents[3] / "franka-imitation-learning"))).expanduser()
SNAPSHOT_ROOT = Path(__file__).resolve().parents[1] / "vendor/fr3_controller_snapshot"
BRIDGE_PATH = SNAPSHOT_ROOT / "FR3_omy_bridge.py"
ADAPTER_PATH = SNAPSHOT_ROOT / "adapter.py"
GEOMAGIC_ORIENTATION_CONDITIONING_GAIN = 8.0  # 1/s; OMY source remains 4.
FUNCTIONS = frozenset({
    "read_site_pose", "skew", "rotvec_to_matrix", "matrix_to_rotvec", "limit_norm",
    "condition_target", "dls_pseudoinverse", "exact_nullspace_projector",
    "compute_joint_target",
})
CONSTANTS = frozenset({
    "MAX_TARGET_LINEAR_SPEED", "MAX_TARGET_LINEAR_ACCEL", "POSITION_SNAP_ERROR",
    "POSITION_SNAP_SPEED", "MAX_TARGET_ANGULAR_SPEED", "MAX_TARGET_ANGULAR_ACCEL",
    "TARGET_ROTATION_KP", "ROTATION_SNAP_ERROR", "POSITION_KP", "ORIENTATION_KP",
    "IK_DAMPING", "ROTATION_LENGTH_SCALE", "MAX_JOINT_SPEED",
    "ENABLE_NULLSPACE_POSTURE", "NULLSPACE_GAIN",
})


@lru_cache(maxsize=8)
def robot_helpers(orientation_conditioning_gain=GEOMAGIC_ORIENTATION_CONDITIONING_GAIN,
                  *, position_gain=8.0, orientation_gain=4.0):
    """Configure a private controller namespace, preserving Touch defaults.

    The optional gain permits a deterministic gain-4 baseline with the same
    controller math. Optional feedback gains let a caller select its own
    response without changing another controller or the OMY module. Each
    cached configuration owns its globals; rate, acceleration, joint limits,
    damping and null-space parameters always come from the existing source.
    """
    for label, gain in (("orientation conditioning gain", orientation_conditioning_gain),
                        ("position gain", position_gain), ("orientation gain", orientation_gain)):
        if isinstance(gain, bool) or not math.isfinite(gain) or gain <= 0:
            raise ValueError(f"{label} must be finite and positive")
    import mujoco
    import numpy as np

    source = ast.parse(BRIDGE_PATH.read_text(), filename=str(BRIDGE_PATH))
    selected, found = [], set()
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in CONSTANTS:
                # Scalars only: fail visibly if upstream configuration changes.
                ast.literal_eval(node.value)
                selected.append(node)
                found.add(target.id)
    if found != FUNCTIONS | CONSTANTS:
        raise RuntimeError(f"FR3 helper contract changed: missing {(FUNCTIONS | CONSTANTS) - found}")
    namespace = {"np": np, "mujoco": mujoco, "__name__": "geomagic_existing_fr3_helpers"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(BRIDGE_PATH), "exec"), namespace)
    # These functions execute in this private namespace, not the OMY runner's.
    # Set the conditioner global once, before profiling; no per-tick override.
    namespace["TARGET_ROTATION_KP"] = float(orientation_conditioning_gain)
    namespace["POSITION_KP"] = float(position_gain)
    namespace["ORIENTATION_KP"] = float(orientation_gain)
    from geomagic.teleop.runtime_profile import get_profiler
    profiler = get_profiler()
    if profiler:
        for name, label in {'condition_target':'controller.condition_target',
                            'compute_joint_target':'controller.ik_total',
                            'dls_pseudoinverse':'controller.dls_solve',
                            'exact_nullspace_projector':'controller.null_space',
                            'matrix_to_rotvec':'controller.so3_log',
                            'rotvec_to_matrix':'controller.so3_exp',
                            'limit_norm':'controller.norm_clip'}.items():
            namespace[name] = profiler.wrap(label, namespace[name])
        namespace['mujoco'] = SimpleNamespace(mj_jacSite=profiler.wrap(
            'controller.jacobian', mujoco.mj_jacSite))
        # bridge helpers use only mj_jacSite from mujoco.
        class ProfileNumpy:
            clip = staticmethod(profiler.wrap('controller.clip', np.clip))
            def __getattr__(self, name):
                return getattr(np, name)
        namespace['np'] = ProfileNumpy()
    return SimpleNamespace(**{name: namespace[name] for name in found})


@lru_cache(maxsize=1)
def command_adapter():
    """Load the actual existing TeleopCommand/adapter, without package startup."""
    name = "_geomagic_existing_mimic_command_adapter"
    spec = importlib.util.spec_from_file_location(name, ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
