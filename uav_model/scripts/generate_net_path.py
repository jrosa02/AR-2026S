#!/usr/bin/env python3
"""Generate a two-phase net-pattern survey path YAML.

Phase 1 – boustrophedon rows: sweep along X for each Y line.
Phase 2 – boustrophedon columns: sweep along Y for each X line.

The two sets of flight lines cross at right angles, forming a net.

Velocity at each waypoint uses a bisector-blend of the incoming and outgoing
directions so the arrows form a continuous flowing pattern (no abrupt 90° jumps
at corners). At angles > ~96° (cos < -0.1) the outgoing direction is used
directly to avoid backward-pointing arrows.

Speed classification:
  V_SURVEY — outgoing segment is a full-length axis-aligned row/column pass.
  V_TRANS  — short row-to-row steps and diagonal repositioning moves.
"""

import math
import numpy as np

# ── Parameters ──────────────────────────────────────────────────────────────
ALT      = 1.5        # survey altitude [m]
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = -1.5, 1.5
SPACING  = 0.5        # row/column spacing [m]
V_SURVEY = 1.5        # speed along survey passes [m/s]
V_TRANS  = 0.8        # speed on transitions [m/s]
TOL      = 0.25       # position tolerance for all waypoints [m]

OUT = '/home/developer/ros2_ws/src/uav_model/config/example_path.yaml'

_X_FULL = X_MAX - X_MIN  # 3.0 m
_Y_FULL = Y_MAX - Y_MIN  # 3.0 m


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def _is_survey_segment(p0: np.ndarray, p1: np.ndarray) -> bool:
    """True when p0→p1 is a full-length axis-aligned survey pass."""
    d   = p1 - p0
    seg = np.linalg.norm(d)
    ud  = unit(d)
    if abs(ud[0]) > 0.99 and abs(seg - _X_FULL) < 0.05:
        return True
    if abs(ud[1]) > 0.99 and abs(seg - _Y_FULL) < 0.05:
        return True
    return False


def _smooth_vel(i: int, pts: list[tuple]) -> np.ndarray:
    """Continuous-flow velocity at waypoint i.

    Direction: normalised bisector of in/out unit-vectors for angles ≤ 96°,
               outgoing direction for more obtuse turns.
    Speed:     V_SURVEY when the outgoing segment is a full survey pass,
               V_TRANS  otherwise.
    """
    n = len(pts)
    p = [np.array(pt[:3], dtype=float) for pt in pts]

    if i == n - 1:
        return np.zeros(3)

    out_d = unit(p[i + 1] - p[i])
    speed = V_SURVEY if _is_survey_segment(p[i], p[i + 1]) else V_TRANS

    if i == 0:
        return out_d * speed

    in_d  = unit(p[i] - p[i - 1])
    cos_a = float(np.clip(np.dot(in_d, out_d), -1.0, 1.0))

    # Bisector blend for angles ≤ ~96° — produces 45° diagonals at right-angle
    # corners so the flow looks smooth. For more obtuse turns use out_d directly.
    if cos_a > -0.1:
        s = in_d + out_d
        direction = unit(s) if np.linalg.norm(s) > 1e-6 else out_d
    else:
        direction = out_d

    return direction * speed


def grid(lo: float, hi: float) -> list[float]:
    n = round((hi - lo) / SPACING) + 1
    return [round(float(v), 4) for v in np.linspace(lo, hi, n)]


def boustrophedon_x(ys: list[float]) -> list[tuple]:
    """Horizontal passes: one full x-sweep per y value, alternating direction."""
    pts = []
    for i, y in enumerate(ys):
        if i % 2 == 0:
            pts += [(X_MIN, y, ALT), (X_MAX, y, ALT)]
        else:
            pts += [(X_MAX, y, ALT), (X_MIN, y, ALT)]
    return pts


def boustrophedon_y(xs: list[float]) -> list[tuple]:
    """Vertical passes: one full y-sweep per x value, alternating direction."""
    pts = []
    for i, x in enumerate(xs):
        if i % 2 == 0:
            pts += [(x, Y_MIN, ALT), (x, Y_MAX, ALT)]
        else:
            pts += [(x, Y_MAX, ALT), (x, Y_MIN, ALT)]
    return pts


def build_waypoints(pts: list[tuple]) -> list[dict]:
    wps = []
    for i, pt in enumerate(pts):
        v = _smooth_vel(i, pts)
        wps.append({
            'x':  round(pt[0], 3), 'y': round(pt[1], 3), 'z': round(pt[2], 3),
            'vx': round(float(v[0]), 3),
            'vy': round(float(v[1]), 3),
            'vz': round(float(v[2]), 3),
            'yaw': 0.0, 'tolerance': TOL,
        })
    return wps


def fmt(wp: dict) -> str:
    return (
        f"  - {{x: {wp['x']:6.3f}, y: {wp['y']:6.3f}, z: {wp['z']:5.3f},"
        f" vx: {wp['vx']:6.3f}, vy: {wp['vy']:6.3f}, vz: {wp['vz']:6.3f},"
        f" yaw: 0.0, tolerance: {wp['tolerance']}}}"
    )


def main():
    ys = grid(Y_MIN, Y_MAX)
    xs = grid(X_MIN, X_MAX)

    phase1 = boustrophedon_x(ys)
    phase2 = boustrophedon_y(xs)

    # Climb waypoint above origin, then phase 1, then phase 2
    all_pts = [(0.0, 0.0, ALT)] + phase1 + phase2

    wps = build_waypoints(all_pts)

    n1, n2 = len(phase1), len(phase2)
    header = (
        f'# Two-phase net-pattern survey path (auto-generated).\n'
        f'# Phase 1: {n1 // 2} horizontal rows  ({n1} corners)\n'
        f'# Phase 2: {n2 // 2} vertical columns ({n2} corners)\n'
        f'# Total: {len(wps)} waypoints | '
        f'ALT={ALT} m | SPACING={SPACING} m | '
        f'V_survey={V_SURVEY} m/s | V_trans={V_TRANS} m/s | TOL={TOL} m\n'
        f'# Velocities: bisector-blend of in/out directions (continuous flow).\n'
    )

    with open(OUT, 'w') as f:
        f.write(header)
        f.write('waypoints:\n')
        for wp in wps:
            f.write(fmt(wp) + '\n')

    print(f'Wrote {len(wps)} waypoints → {OUT}')
    for i, wp in enumerate(wps):
        spd = math.sqrt(wp['vx']**2 + wp['vy']**2 + wp['vz']**2)
        print(f'  [{i:2d}] ({wp["x"]:6.3f}, {wp["y"]:6.3f}, {wp["z"]:.3f})'
              f'  v=({wp["vx"]:6.3f},{wp["vy"]:6.3f},{wp["vz"]:6.3f})  |v|={spd:.2f}')


if __name__ == '__main__':
    main()
