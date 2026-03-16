"""
extract_2d_offset.py
────────────────────
Reads mocap CSV with 3D position + quaternion for a gripper and probe,
then outputs the 2D position offset and rotation of the probe relative to
the gripper, projected onto a user-selected plane.

Position offset  : probe_pos - gripper_pos, projected onto PLANE.
Rotation offset  : relative rotation (q_gripper⁻¹ × q_probe), decomposed
                   into the twist angle around the axis normal to PLANE
                   via swing-twist decomposition.
"""

import numpy as np
import pandas as pd

# ── USER CONFIGURATION ───────────────────────────────────────────────────────
PLANE      = "xy"                  # Projection plane: "xy" | "yz" | "xz"
INPUT_CSV  = "mocap_raw.csv"
OUTPUT_CSV = "offset_2d.csv"
# ─────────────────────────────────────────────────────────────────────────────

# Maps each plane to the two in-plane axes and the index of the normal axis
# (0=X, 1=Y, 2=Z) used to extract the twist rotation component.
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


def _swing_twist_angle(q: np.ndarray, normal_idx: int) -> float:
    """
    Swing-twist decomposition: extract the twist angle around a cardinal axis.

    Given q = (w, x, y, z) and the index of the twist axis (0=X, 1=Y, 2=Z),
    returns the signed twist angle in radians ∈ (−π, π].

    The twist quaternion is: t = normalize(w,  n̂ · (x,y,z) · n̂)
    i.e. only the component of the imaginary part along the axis is kept.
    """
    q_vec = q[1:]                          # (x, y, z)
    proj = np.zeros(3)
    proj[normal_idx] = q_vec[normal_idx]   # project onto normal axis
    twist = np.array([q[0], proj[0], proj[1], proj[2]])
    norm = np.linalg.norm(twist)
    if norm < 1e-10:
        return 0.0
    twist /= norm
    # angle = 2 * atan2( sin(θ/2), cos(θ/2) )  where the imaginary component
    # along the normal axis is sin(θ/2).
    return 2.0 * np.arctan2(twist[normal_idx + 1], twist[0])


# ── Main processing ───────────────────────────────────────────────────────────

def process(input_csv: str, output_csv: str, plane: str) -> None:
    plane = plane.lower()
    if plane not in _PLANE_CFG:
        raise ValueError(f"PLANE must be one of {list(_PLANE_CFG)}, got '{plane}'")

    cfg        = _PLANE_CFG[plane]
    ax0, ax1   = cfg["axes"]        # e.g. "x", "y"
    normal_idx = cfg["normal_idx"]  # index of axis normal to the plane
    normal_lbl = "xyz"[normal_idx].upper()

    df = pd.read_csv(input_csv)

    rows = []
    for _, r in df.iterrows():
        # ── 3D positions ────────────────────────────────────────────────────
        g_pos = np.array([r["gripper_x_m"], r["gripper_y_m"], r["gripper_z_m"]])
        p_pos = np.array([r["probe_x_m"],   r["probe_y_m"],   r["probe_z_m"]])

        # ── Quaternions as (w, x, y, z) ─────────────────────────────────────
        g_q = np.array([r["gripper_qw"], r["gripper_qx"], r["gripper_qy"], r["gripper_qz"]])
        p_q = np.array([r["probe_qw"],   r["probe_qx"],   r["probe_qy"],   r["probe_qz"]])

        # ── 2D position offset: project (probe - gripper) onto plane ────────
        delta = p_pos - g_pos
        offset_u = delta[_AXIS_IDX[ax0]]
        offset_v = delta[_AXIS_IDX[ax1]]

        # ── Relative rotation: q_rel = q_gripper⁻¹ ⊗ q_probe ───────────────
        # Expresses the probe orientation in the gripper's local frame.
        q_rel = _quat_mul(_quat_conj(g_q), p_q)

        # ── 2D rotation: twist angle around the plane's normal axis ─────────
        rot_rad = _swing_twist_angle(q_rel, normal_idx)

        rows.append({
            "timestamp_s":            r["timestamp_s"],
            "frame":                  int(r["frame"]),
            f"offset_{ax0}_m":        offset_u,
            f"offset_{ax1}_m":        offset_v,
            "rotation_rad":           rot_rad,
            "rotation_deg":           np.degrees(rot_rad),
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False, float_format="%.6f")

    print(f"Plane        : {plane.upper()}  (rotation extracted around {normal_lbl}-axis)")
    print(f"Position axes: {ax0.upper()}, {ax1.upper()}")
    print(f"Frames       : {len(out)}")
    print(f"Output       : {output_csv}")
    print()
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    process(INPUT_CSV, OUTPUT_CSV, PLANE)
