"""
extract_2d_offset.py
────────────────────
Reads mocap CSV with 3D position + quaternion for a gripper and probe,
then outputs the 2D position offset, rotation, and their derivatives of the
probe relative to the gripper, all expressed in the probe's own frame.

All quantities are computed in the probe (object) frame
────────────────────────────────────────────────────────
  Position offset  : (probe_pos − gripper_pos) rotated into the probe's frame,
                     then projected onto PLANE.

  Why probe frame?
    • Probe orbiting the gripper (both co-rotating as a rigid body): the
      probe-frame offset is constant  →  translational velocity = 0.
    • Probe spinning about its own centre (fixed position relative to gripper,
      but changing orientation): the probe-frame offset rotates  →
      translational velocity ≠ 0.

  Rotation offset  : swing-twist angle of q_gripper⁻¹ ⊗ q_probe around the
                     axis normal to PLANE.  Frame-invariant scalar.

  Angular velocity : ω = (r × v) / ‖r‖²  where r and v are the 2-D
                     probe-frame offset and its time derivative respectively.
                     This is the angular velocity of the gripper-to-probe
                     direction as seen from the probe's own frame.
"""

import numpy as np
import pandas as pd

# ── USER CONFIGURATION ───────────────────────────────────────────────────────
PLANE      = "xy"                  # Projection plane: "xy" | "yz" | "xz"
INPUT_CSV  = "mocap_raw.csv"
OUTPUT_CSV = "offset_2d.csv"
# ─────────────────────────────────────────────────────────────────────────────

_PLANE_CFG = {
    "xy": {"axes": ("x", "y"), "normal_idx": 2},   # normal = +Z
    "yz": {"axes": ("y", "z"), "normal_idx": 0},   # normal = +X
    "xz": {"axes": ("x", "z"), "normal_idx": 1},   # normal = +Y
}

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


# ── Quaternion helpers (convention: w, x, y, z) ──────────────────────────────

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions stored as (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quaternion) of q = (w, x, y, z)."""
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _rotate_vec_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate 3-vector v by unit quaternion q = (w, x, y, z).
    Uses the sandwich product: v' = q ⊗ [0, v] ⊗ q*.
    To express v in the frame of q, pass q* (conjugate) as q.
    """
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]


def _swing_twist_angle(q: np.ndarray, normal_idx: int) -> float:
    """
    Swing-twist decomposition: extract the twist angle around a cardinal axis.

    Given q = (w, x, y, z) and the index of the twist axis (0=X, 1=Y, 2=Z),
    returns the signed twist angle in radians ∈ (−π, π].
    """
    q_vec = q[1:]
    proj = np.zeros(3)
    proj[normal_idx] = q_vec[normal_idx]
    twist = np.array([q[0], proj[0], proj[1], proj[2]])
    norm = np.linalg.norm(twist)
    if norm < 1e-10:
        return 0.0
    twist /= norm
    return 2.0 * np.arctan2(twist[normal_idx + 1], twist[0])


# ── Main processing ───────────────────────────────────────────────────────────

def process(input_csv: str, output_csv: str, plane: str) -> None:
    plane = plane.lower()
    if plane not in _PLANE_CFG:
        raise ValueError(f"PLANE must be one of {list(_PLANE_CFG)}, got '{plane}'")

    cfg        = _PLANE_CFG[plane]
    ax0, ax1   = cfg["axes"]
    normal_idx = cfg["normal_idx"]
    normal_lbl = "xyz"[normal_idx].upper()

    df = pd.read_csv(input_csv)

    rows = []
    for _, r in df.iterrows():
        g_pos = np.array([r["gripper_x_m"], r["gripper_y_m"], r["gripper_z_m"]])
        p_pos = np.array([r["probe_x_m"],   r["probe_y_m"],   r["probe_z_m"]])

        g_q = np.array([r["gripper_qw"], r["gripper_qx"], r["gripper_qy"], r["gripper_qz"]])
        p_q = np.array([r["probe_qw"],   r["probe_qx"],   r["probe_qy"],   r["probe_qz"]])

        # ── Position offset in probe's frame ────────────────────────────────
        # Rotating the world-frame offset by q_probe⁻¹ expresses it in the
        # probe's own coordinate system.  When the probe and gripper move as
        # a rigid body (co-rotation), this vector is constant.  When the probe
        # spins about its own centre at a fixed position, this vector rotates.
        delta       = p_pos - g_pos
        delta_probe = _rotate_vec_by_quat(delta, _quat_conj(p_q))

        offset_u = delta_probe[_AXIS_IDX[ax0]]
        offset_v = delta_probe[_AXIS_IDX[ax1]]

        # ── Relative rotation: swing-twist angle of q_gripper⁻¹ ⊗ q_probe ──
        # Frame-invariant scalar — unaffected by the frame choice above.
        q_rel   = _quat_mul(_quat_conj(g_q), p_q)
        rot_rad = _swing_twist_angle(q_rel, normal_idx)

        rows.append({
            "timestamp_s":     r["timestamp_s"],
            "frame":           int(r["frame"]),
            f"offset_{ax0}_m": offset_u,
            f"offset_{ax1}_m": offset_v,
            "rotation_rad":    rot_rad,
            "rotation_deg":    np.degrees(rot_rad),
        })

    out = pd.DataFrame(rows)

    # ── Derivatives ───────────────────────────────────────────────────────────
    # np.gradient uses central differences for interior points and one-sided
    # differences at the boundaries, respecting non-uniform sample intervals.
    t  = out["timestamp_s"].to_numpy()
    ru = out[f"offset_{ax0}_m"].to_numpy()
    rv = out[f"offset_{ax1}_m"].to_numpy()

    vu = np.gradient(ru, t)
    vv = np.gradient(rv, t)

    out[f"vel_offset_{ax0}_m/s"]   = vu
    out[f"vel_offset_{ax1}_m/s"]   = vv
    out[f"accel_offset_{ax0}_m/s2"] = np.gradient(vu, t)
    out[f"accel_offset_{ax1}_m/s2"] = np.gradient(vv, t)

    # ── Angular velocity: ω = (r × v) / ‖r‖² ────────────────────────────────
    # r and v are already in the same frame (probe frame), so this is
    # consistent.  Gives the angular velocity of the gripper-to-probe direction
    # vector as observed from the probe's own frame.
    r_sq  = ru**2 + rv**2
    omega = np.where(r_sq > 1e-12, (ru * vv - rv * vu) / r_sq, 0.0)

    out["vel_rotation_rad/s"]      = omega
    out["accel_rotation_rad/s2"]   = np.gradient(omega, t)

    out.to_csv(output_csv, index=False, float_format="%.6f")

    print(f"Plane        : {plane.upper()}  (rotation around {normal_lbl}-axis)")
    print(f"Position axes: {ax0.upper()}, {ax1.upper()}  —  expressed in probe frame")
    print(f"Frames       : {len(out)}")
    print(f"Output       : {output_csv}")
    print()
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    process(INPUT_CSV, OUTPUT_CSV, PLANE)
