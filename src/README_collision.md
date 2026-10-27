# FR3 collision feedback module

> **Historical reference.** The text and commands below describe the earlier OMY system. Its original scene/model bundle, including the front wall, is not included or reconstructed in this source/reference release. Present-tense asset/installation descriptions are historical, not current readiness claims. See [reproduction limits](../docs/HISTORICAL_OMY_LIMITATIONS.md).

`$TELEOP_ROOT/src/fr3_collision_feedback.py` is a bridge-neutral
MuJoCo collision layer. It is deliberately separate from ROS and Dynamixel
current control.

## Runtime flow

```text
FR3 MuJoCo mj_step
        |
        v
Fr3CollisionMonitor.update(model, data)
        |
        +-- CollisionFeedback (JSON/ROS-friendly)
        |     +-- wall_contact
        |     +-- force_world / normal_world
        |     +-- contact list
        |
        +-- resistance_force_world
              |
              v
future bridge: OMY Jacobian transpose -> bounded effort/current command
```

## Current API

- `CollisionConfig`: wall geometry names and force limits.
- `Fr3CollisionMonitor.update()`: reads MuJoCo contacts after each step.
- `CollisionFeedback.as_dict()`: transport payload for a future ROS/ZMQ bridge.
- `cartesian_force_to_joint_torque()`: computes `J.T @ F`; it does not send
  hardware effort/current.

The default wall geometry is `fr3_front_wall_geom`, which is present in the
teleop FR3 scene. Self-collision detection remains a separate OMY package.
