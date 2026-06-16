#!/usr/bin/env python3
"""Plot a waypoint path YAML with velocity quiver arrows. Saves PNG, no display."""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load(path_file: str) -> list[dict]:
    with open(path_file) as f:
        data = yaml.safe_load(f)
    wps = data.get('waypoints')
    if not wps:
        sys.exit(f'No waypoints in {path_file}')
    return wps


def plot(wps: list[dict], out: str) -> None:
    xs  = np.array([w['x']          for w in wps])
    ys  = np.array([w['y']          for w in wps])
    zs  = np.array([w['z']          for w in wps])
    us  = np.array([w.get('vx', 0.) for w in wps])
    vs  = np.array([w.get('vy', 0.) for w in wps])
    ws  = np.array([w.get('vz', 0.) for w in wps])

    speeds = np.sqrt(us**2 + vs**2 + ws**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Path preview — waypoints + velocity vectors', fontsize=12)

    # ── XY top-down ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(xs, ys, '-', color='tab:blue', linewidth=1.0, alpha=0.5, zorder=1)
    ax.scatter(xs, ys, s=18, color='tab:blue', zorder=3)
    ax.plot(xs[0], ys[0], 'go', markersize=9, zorder=5, label='start')
    ax.plot(xs[-1], ys[-1], 'rs', markersize=9, zorder=5, label='end')

    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(str(i), (x, y), textcoords='offset points',
                    xytext=(4, 3), fontsize=6, color='dimgray')

    # Quiver: angles/scale in data coordinates so arrow length = velocity magnitude
    max_spd = speeds.max() if speeds.max() > 0 else 1.0
    ax.quiver(xs, ys, us, vs,
              angles='xy', scale_units='xy', scale=max_spd * 2.5,
              color='tab:orange', alpha=0.85, width=0.006,
              headwidth=4, headlength=5, zorder=4, label='velocity')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Top-down (XY)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── XZ side view ───────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(xs, zs, '-', color='tab:blue', linewidth=1.0, alpha=0.5, zorder=1)
    ax2.scatter(xs, zs, s=18, color='tab:blue', zorder=3)
    ax2.plot(xs[0], zs[0], 'go', markersize=9, zorder=5, label='start')
    ax2.plot(xs[-1], zs[-1], 'rs', markersize=9, zorder=5, label='end')

    for i, (x, z) in enumerate(zip(xs, zs)):
        ax2.annotate(str(i), (x, z), textcoords='offset points',
                     xytext=(4, 3), fontsize=6, color='dimgray')

    ax2.quiver(xs, zs, us, ws,
               angles='xy', scale_units='xy', scale=max_spd * 2.5,
               color='tab:orange', alpha=0.85, width=0.006,
               headwidth=4, headlength=5, zorder=4, label='velocity')

    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('z [m]')
    ax2.set_title('Side view (XZ)')
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}  ({len(wps)} waypoints, max |v|={max_spd:.2f} m/s)')


def main() -> None:
    default_yaml = (
        '/home/developer/ros2_ws/install/share/uav_model/config/example_path.yaml'
    )
    default_out = '/home/developer/ros2_ws/src/plots/path_preview.png'

    ap = argparse.ArgumentParser(description='Preview a waypoint path with velocity vectors.')
    ap.add_argument('path_file', nargs='?', default=default_yaml)
    ap.add_argument('--out', default=default_out)
    args = ap.parse_args()

    wps = load(args.path_file)
    print(f'Loaded {len(wps)} waypoints from {args.path_file}')
    plot(wps, args.out)


if __name__ == '__main__':
    main()
