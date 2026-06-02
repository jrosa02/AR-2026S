"""Standalone simulation + trajectory plot for the CrazyFlie UAV model.

Runs a closed-loop MPC+Mellinger (or Mellinger-only) simulation for a
configurable duration, records the drone's trajectory, and saves a 2×3
matplotlib figure to a PNG file.

Usage::

    python3 simulate_and_plot.py [options]
    ros2 run uav_model simulate_and_plot [options]

Options::

    --controller {mpc,mellinger}   Controller stack (default: mpc)
    --path-file PATH               YAML waypoint file (default: example_path.yaml)
    --sim-time FLOAT               Simulation duration in seconds (default: 120)
    --output PATH                  Output image path (default: drone_path.png)
    --mpc-params PATH              MPC parameter YAML file (default: mpc_params.yaml)
"""

import argparse
import os

import matplotlib

matplotlib.use('Agg')  # must precede pyplot — non-interactive, works without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from ament_index_python.packages import get_package_share_directory  # noqa: E402

from uav_model.config_loader.sdf_adapter import SDFAdapter  # noqa: E402
from uav_model.model.mellinger import MellingerController, MellingerGains  # noqa: E402
from uav_model.model.mpc_controller import MPCController, MPCParams  # noqa: E402
from uav_model.model.uav_model import UAVFlatState, UAVModel  # noqa: E402

# ---------------------------------------------------------------------------
# Simulation timing constants (match ROS node rates)
# ---------------------------------------------------------------------------
_DT = 0.002  # physics step: 500 Hz
_CTRL_EVERY = 5  # Mellinger at 100 Hz
_MPC_EVERY = 25  # MPC at 20 Hz

_DEFAULT_SDF = os.path.join(
    get_package_share_directory('ros_gz_crazyflie_bringup'),
    'gazebo_files',
    'gazebo',
    'crazyflie',
    'model.sdf',
)
_DEFAULT_PATH = os.path.join(
    get_package_share_directory('uav_model'),
    'config',
    'example_path.yaml',
)
_DEFAULT_MPC_PARAMS = os.path.join(
    get_package_share_directory('uav_model'),
    'config',
    'mpc_params.yaml',
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_waypoints(path: str) -> list:
    with open(path) as f:
        data = yaml.safe_load(f)
    wps = data['waypoints']
    for wp in wps:
        wp.setdefault('yaw', 0.0)
        wp.setdefault('tolerance', 0.5)
    return wps


def _load_mpc_params(path: str) -> MPCParams:
    with open(path) as f:
        data = yaml.safe_load(f)
    # Support both bare dict and ROS2 param file format
    p = data.get('mpc_controller', {}).get('ros__parameters', data)
    return MPCParams(
        horizon=int(p.get('mpc_horizon', 10)),
        dt=float(p.get('mpc_dt', 0.05)),
        Q_pos=np.array(p.get('mpc_q_pos', [10.0, 10.0, 10.0])),
        Q_vel=np.array(p.get('mpc_q_vel', [1.0, 1.0, 1.0])),
        R_acc=np.array(p.get('mpc_r_acc', [0.1, 0.1, 0.1])),
        Q_terminal=np.array(p.get('mpc_q_terminal', [50.0, 50.0, 50.0])),
        v_max=float(p.get('mpc_v_max', 3.0)),
        a_max=float(p.get('mpc_a_max', 5.0)),
    )


def _load_mellinger_gains(path: str) -> MellingerGains:
    with open(path) as f:
        data = yaml.safe_load(f)
    p = data.get('mpc_controller', {}).get('ros__parameters', data)
    return MellingerGains(
        kx=np.array(p.get('kx', [6.0, 6.0, 6.0])),
        kv=np.array(p.get('kv', [4.0, 4.0, 4.0])),
        kR=np.array(p.get('kR', [8.0e-3, 8.0e-3, 2.0e-3])),
        kOmega=np.array(p.get('kOmega', [1.1e-3, 1.1e-3, 5.0e-4])),
    )


def _waypoint_to_flat(wp: dict) -> UAVFlatState:
    data = np.zeros(18)
    data[0] = wp['x']
    data[1] = wp['y']
    data[2] = wp['z']
    data[15] = float(wp.get('yaw', 0.0))
    return UAVFlatState(data)


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------


def run_simulation(
    ctrl: str,
    waypoints: list,
    sim_time: float,
    cf_params,
    mpc_params: MPCParams,
    mellinger_gains: MellingerGains,
) -> dict:
    """Run the closed-loop simulation and return recorded data arrays."""
    model = UAVModel(cf_params)
    model.reset()

    mellinger = MellingerController(cf_params, mellinger_gains)
    mpc = MPCController(mpc_params) if ctrl == 'mpc' else None

    # Initial state: origin, level attitude
    init = np.zeros(13)
    init[6] = 1.0  # quaternion w = 1
    model.reset(init)

    # Active waypoint tracking
    wp_idx = 0
    wp = waypoints[wp_idx]
    goal = np.array([wp['x'], wp['y'], wp['z']])
    yaw = float(wp.get('yaw', 0.0))
    tolerance = float(wp['tolerance'])

    flat_setpoint = _waypoint_to_flat(wp)
    hover_thrust = float(cf_params.mass) * 9.81
    u = np.array([hover_thrust, 0.0, 0.0, 0.0])

    times, positions, speeds, thrusts, errors = [], [], [], [], []

    total_steps = int(sim_time / _DT)
    print_every = max(1, total_steps // 100)  # print ~100 progress updates

    for i in range(total_steps):
        t = i * _DT

        # --- Waypoint advancement (at Mellinger rate to avoid redundant checks) ---
        if i % _CTRL_EVERY == 0:
            dist = float(np.linalg.norm(model.state.position - goal))
            if dist < tolerance:
                wp_idx = (wp_idx + 1) % len(waypoints)
                wp = waypoints[wp_idx]
                goal = np.array([wp['x'], wp['y'], wp['z']])
                yaw = float(wp.get('yaw', 0.0))
                tolerance = float(wp['tolerance'])
                if ctrl == 'mellinger':
                    flat_setpoint = _waypoint_to_flat(wp)

        # --- MPC outer loop (20 Hz) ---
        if ctrl == 'mpc' and i % _MPC_EVERY == 0:
            flat_setpoint = mpc.compute(model.state, goal, yaw)

        # --- Mellinger inner loop (100 Hz) ---
        if i % _CTRL_EVERY == 0:
            u = mellinger.compute(flat_setpoint, model.state)

        # --- Physics step (500 Hz) ---
        model.step(u, _DT)

        # --- Record at 100 Hz (after step) ---
        if i % _CTRL_EVERY == 0:
            times.append(t)
            positions.append(model.state.position.copy())
            speeds.append(float(np.linalg.norm(model.state.velocity)))
            thrusts.append(float(u[0]))
            errors.append(float(np.linalg.norm(model.state.position - goal)))

        if i % print_every == 0:
            print(
                f'\r  t={t:6.1f}/{sim_time:.0f}s  wp={wp_idx + 1}/{len(waypoints)}'
                f'  pos=[{model.state.position[0]:5.2f},{model.state.position[1]:5.2f},'
                f'{model.state.position[2]:5.2f}]  err={errors[-1]:.2f}m',
                end='',
                flush=True,
            )

    print()  # newline after progress line

    return {
        'times': np.array(times),
        'positions': np.array(positions),  # (N, 3)
        'speeds': np.array(speeds),
        'thrusts': np.array(thrusts),
        'errors': np.array(errors),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def save_plot(
    data: dict, waypoints: list, ctrl: str, sim_time: float, output: str, hover_thrust: float
) -> None:
    """Generate and save the 2×3 figure."""
    times = data['times']
    pos = data['positions']
    speeds = data['speeds']
    thrusts = data['thrusts']
    errors = data['errors']

    wp_xyz = np.array([[w['x'], w['y'], w['z']] for w in waypoints])
    tolerance = waypoints[0]['tolerance']

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f'CrazyFlie simulation — controller: {ctrl},  duration: {sim_time:.0f} s,  '
        f'waypoints: {len(waypoints)}',
        fontsize=11,
    )

    # ---- ax0: XY top-down ---------------------------------------------------
    ax0 = fig.add_subplot(2, 3, 1)
    ax0.plot(pos[:, 0], pos[:, 1], 'b-', lw=0.6)
    ax0.scatter(wp_xyz[:, 0], wp_xyz[:, 1], c='red', s=35, zorder=5, label='waypoints')
    ax0.scatter(pos[0, 0], pos[0, 1], c='lime', s=70, zorder=6, label='start')
    ax0.set_xlabel('X (m)')
    ax0.set_ylabel('Y (m)')
    ax0.set_title('XY — top-down')
    ax0.set_aspect('equal')
    ax0.grid(True, lw=0.4)
    ax0.legend(fontsize=7)

    # ---- ax1: XZ side -------------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 2)
    ax1.plot(pos[:, 0], pos[:, 2], 'b-', lw=0.6)
    ax1.scatter(wp_xyz[:, 0], wp_xyz[:, 2], c='red', s=35, zorder=5, label='waypoints')
    ax1.scatter(pos[0, 0], pos[0, 2], c='lime', s=70, zorder=6, label='start')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('XZ — side view')
    ax1.grid(True, lw=0.4)
    ax1.legend(fontsize=7)

    # ---- ax2: YZ front ------------------------------------------------------
    ax2 = fig.add_subplot(2, 3, 3)
    ax2.plot(pos[:, 1], pos[:, 2], 'b-', lw=0.6)
    ax2.scatter(wp_xyz[:, 1], wp_xyz[:, 2], c='red', s=35, zorder=5, label='waypoints')
    ax2.scatter(pos[0, 1], pos[0, 2], c='lime', s=70, zorder=6, label='start')
    ax2.set_xlabel('Y (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('YZ — front view')
    ax2.grid(True, lw=0.4)
    ax2.legend(fontsize=7)

    # ---- ax3: position error ------------------------------------------------
    ax3 = fig.add_subplot(2, 3, 4)
    ax3.plot(times, errors, color='steelblue', lw=0.7)
    ax3.axhline(
        tolerance, color='red', linestyle='--', lw=0.9, label=f'tolerance ({tolerance:.2f} m)'
    )
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Distance to waypoint (m)')
    ax3.set_title('Position Error')
    ax3.grid(True, lw=0.4)
    ax3.legend(fontsize=7)

    # ---- ax4: speed ---------------------------------------------------------
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(times, speeds, color='darkorchid', lw=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Speed')
    ax4.grid(True, lw=0.4)

    # ---- ax5: thrust --------------------------------------------------------
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.plot(times, thrusts, color='darkcyan', lw=0.7)
    ax5.axhline(
        hover_thrust, color='red', linestyle='--', lw=0.9, label=f'hover ({hover_thrust:.4f} N)'
    )
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Thrust (N)')
    ax5.set_title('Collective Thrust')
    ax5.grid(True, lw=0.4)
    ax5.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot saved → {os.path.abspath(output)}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Parse CLI arguments, run simulation, and save trajectory plot."""
    parser = argparse.ArgumentParser(
        description='Simulate the CrazyFlie drone and save a trajectory plot.',
    )
    parser.add_argument(
        '--controller',
        choices=['mpc', 'mellinger'],
        default='mpc',
        help='Controller stack (default: mpc)',
    )
    parser.add_argument(
        '--path-file',
        default=_DEFAULT_PATH,
        metavar='PATH',
        help='YAML waypoint file (default: installed example_path.yaml)',
    )
    parser.add_argument(
        '--sim-time',
        type=float,
        default=120.0,
        metavar='SECONDS',
        help='Simulation duration in seconds (default: 120)',
    )
    parser.add_argument(
        '--output',
        default='drone_path.png',
        metavar='PATH',
        help='Output image file (default: drone_path.png)',
    )
    parser.add_argument(
        '--mpc-params',
        default=_DEFAULT_MPC_PARAMS,
        metavar='PATH',
        help='MPC parameter YAML file (default: installed mpc_params.yaml)',
    )
    args = parser.parse_args()

    print(f'Controller : {args.controller}')
    print(f'Path file  : {args.path_file}')
    print(f'Sim time   : {args.sim_time} s')
    print(f'Output     : {args.output}')

    waypoints = _load_waypoints(args.path_file)
    print(f'Waypoints  : {len(waypoints)} loaded')

    cf_params = SDFAdapter(_DEFAULT_SDF).extract()
    mpc_params = _load_mpc_params(args.mpc_params)
    mellinger_gains = _load_mellinger_gains(args.mpc_params)
    hover_thrust = float(cf_params.mass) * 9.81

    print(f'Simulating ({int(args.sim_time / _DT):,} physics steps) …')
    data = run_simulation(
        ctrl=args.controller,
        waypoints=waypoints,
        sim_time=args.sim_time,
        cf_params=cf_params,
        mpc_params=mpc_params,
        mellinger_gains=mellinger_gains,
    )

    print('Generating plot …')
    save_plot(
        data=data,
        waypoints=waypoints,
        ctrl=args.controller,
        sim_time=args.sim_time,
        output=args.output,
        hover_thrust=hover_thrust,
    )


if __name__ == '__main__':
    main()
