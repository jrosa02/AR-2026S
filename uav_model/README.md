# uav_model

ROS 2 package implementing a 13-state rigid-body dynamics model for the CrazyFlie 2.1 quadrotor. It runs as a standalone Python simulation node — a software substitute for the Gazebo physics engine when full 3D rendering is not needed (e.g. during controller development or rapid iteration).

## State vector

```
x = [ r(3)  v(3)  q(4)  ω(3) ]   # 13 float64 elements
```

| Slice | Symbol | Frame | Description |
|---|---|---|---|
| `x[0:3]`  | r | World | Position (m) |
| `x[3:6]`  | v | World | Linear velocity (m/s) |
| `x[6:10]` | q | —     | Unit quaternion (w, x, y, z) |
| `x[10:13]`| ω | Body  | Angular velocity (rad/s) |

Initial condition (hover at origin, upright): `x₀ = [0,0,0, 0,0,0, 1,0,0,0, 0,0,0]`

## Control input

The node subscribes to `cf_control_msgs/ThrustAndTorque` — the same message the physical `cf_control::Mixer` node consumes:

```
u = [ T  τx  τy  τz ]   # collective thrust (N) + body torques (N·m)
```

Motor mixing (motor speeds → thrust/torque) is handled by the `cf_control::Mixer` node upstream. This model receives the **abstract** wrench, not raw motor speeds.

## Running

```bash
# Default: reads SDF from ros_gz_crazyflie_bringup, params from config/sim_params.yaml
ros2 launch uav_model sim.launch.py

# Override SDF or params
ros2 launch uav_model sim.launch.py \
    sdf_path:=/path/to/model.sdf \
    params_file:=/path/to/sim_params.yaml
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sdf_path` | string | *(from bringup package)* | Path to `model.sdf` — required |
| `sim_rate_hz` | double | `500.0` | Simulation timer frequency |
| `control_topic` | string | `/cf_control/control_command` | Incoming `ThrustAndTorque` topic |
| `state_topic` | string | `/uav_model/odom` | Outgoing `nav_msgs/Odometry` topic |
| `initial_state` | double[] | zeros + `qw=1` | Override initial 13-state vector |

## Published state

State is published as `nav_msgs/Odometry` on `/uav_model/odom`:

- `pose.pose.position` → position r
- `pose.pose.orientation` → quaternion q (w, x, y, z)
- `twist.twist.linear` → velocity v (world frame)
- `twist.twist.angular` → angular velocity ω (body frame)

## Physical constants

All constants are extracted at startup from the Gazebo SDF file via `SDFAdapter`:

| Constant | SDF source | Symbol |
|---|---|---|
| Body mass | `<inertial>/<mass>` | m |
| Inertia tensor | `<inertial>/<inertia>` (full 3×3) | J |
| Thrust coefficient | `MulticopterMotorModel/<motorConstant>` | cT |
| Drag coefficient | `MulticopterMotorModel/<rotorDragCoefficient>` | cD |
| Rotor positions | Motor link `<pose>` | r_i |
| Turning directions | `<turningDirection>` (ccw → −1, cw → +1) | d_i |

`J⁻¹` is precomputed once at startup in `UAVModel.__init__` and cached as an instance attribute — it is never inverted inside the integration loop.

## Tests

```bash
colcon test --merge-install --packages-select uav_model
colcon test-result --verbose
```

20 tests covering: SDF extraction correctness, quaternion normalisation, rotor-to-wrench conversion, hover equilibrium, free-fall trajectory against analytic solution, and long-run quaternion norm preservation.

## Acklnowledgments
Documentation and code formatting done using LLM.
