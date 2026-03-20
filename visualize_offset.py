"""
visualize_offset.py
───────────────────
Interactive viewer for the 2D offset data produced by extract_2d_offset.py.

Layout
──────
  Left  : 2D spatial view — full trajectory (faded), recent trail, current
           position dot, rotation indicator, velocity arrow, acceleration arrow.
  Right : Three stacked time-series plots — position, velocity, acceleration —
           with a moving vertical cursor.
  Bottom: Transport controls  ◀◀  ◀  ▶/⏸  ▶  ▶▶  +  frame slider.

Playback
────────
  ▶ / ⏸   Play / pause at 1× real-time speed.
  ◀ / ▶   Step one frame at a time.
  ◀◀/ ▶▶  Jump 10 frames at a time.
  Slider   Drag to any frame (pauses playback).
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_CSV = "offset_2d.csv"
TRAIL_LEN = 60          # number of past frames shown as the coloured trail


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ── Main viewer ───────────────────────────────────────────────────────────────

def run(df: pd.DataFrame) -> animation.FuncAnimation:

    # ── Discover column names from whatever plane was used ───────────────────
    pos_cols = [c for c in df.columns if c.startswith("offset_") and c.endswith("_m")]
    vel_cols = [c for c in df.columns if c.startswith("vel_offset_")]
    acc_cols = [c for c in df.columns if c.startswith("accel_offset_")]

    ax0 = pos_cols[0].replace("offset_", "").replace("_m", "").upper()  # e.g. "X"
    ax1 = pos_cols[1].replace("offset_", "").replace("_m", "").upper()  # e.g. "Y"

    t  = df["timestamp_s"].to_numpy();  t = t - t[0]   # relative seconds
    px = df[pos_cols[0]].to_numpy()
    py = df[pos_cols[1]].to_numpy()
    pr = df["rotation_rad"].to_numpy()
    vx = df[vel_cols[0]].to_numpy()
    vy = df[vel_cols[1]].to_numpy()
    vr = df["vel_rotation_rad/s"].to_numpy()
    bx = df[acc_cols[0]].to_numpy()    # b = acceleration (avoids 'ax' clash)
    by = df[acc_cols[1]].to_numpy()
    br = df["accel_rotation_rad/s2"].to_numpy()
    N  = len(df)

    # Arrow scale: max arrow = 20 % of spatial extent
    extent  = max(px.max() - px.min(), py.max() - py.min(), 1e-9)
    vel_max = max(np.abs(vx).max(), np.abs(vy).max(), 1e-9)
    acc_max = max(np.abs(bx).max(), np.abs(by).max(), 1e-9)
    VEL_SC  = extent * 0.20 / vel_max
    ACC_SC  = extent * 0.20 / acc_max
    ROT_LEN = extent * 0.18   # rotation indicator length

    # ── Colour palette (Catppuccin Mocha) ────────────────────────────────────
    BG      = "#1e1e2e"
    PANEL   = "#2a2a3e"
    TXT     = "#cdd6f4"
    GRID_C  = "#3a3a5e"
    C_TRAIL = "#89b4fa"   # blue
    C_POS   = "#f38ba8"   # red
    C_ROT   = "#a6e3a1"   # green
    C_VEL   = "#89dceb"   # teal
    C_ACC   = "#fab387"   # peach
    C_S0    = "#89b4fa"   # series colour 0
    C_S1    = "#cba6f7"   # series colour 1
    C_S2    = "#f38ba8"   # series colour 2
    C_CUR   = "#f9e2af"   # cursor yellow

    def _style(a, title=""):
        a.set_facecolor(PANEL)
        for sp in a.spines.values():
            sp.set_color(GRID_C)
        a.tick_params(colors=TXT, labelsize=8)
        a.xaxis.label.set_color(TXT)
        a.yaxis.label.set_color(TXT)
        if title:
            a.set_title(title, color=TXT, fontsize=9, pad=4)

    # ── Figure & outer gridspec ───────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs_outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[8.8, 1.2], hspace=0.12
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, gs_outer[0], width_ratios=[1, 1.5], wspace=0.32
    )
    gs_ts   = gridspec.GridSpecFromSubplotSpec(3, 1, gs_top[1],   hspace=0.10)
    gs_ctrl = gridspec.GridSpecFromSubplotSpec(1, 9, gs_outer[1], wspace=0.15)

    # ── 2D spatial view ───────────────────────────────────────────────────────
    sp = fig.add_subplot(gs_top[0])
    _style(sp, f"2D Position Offset  ({ax0} / {ax1}  plane)")
    sp.set_aspect("equal", adjustable="datalim")
    sp.set_xlabel(f"Δ{ax0}  (m)")
    sp.set_ylabel(f"Δ{ax1}  (m)")
    sp.grid(True, color=GRID_C, lw=0.4, zorder=0)

    margin = extent * 0.12
    sp.set_xlim(px.min() - margin, px.max() + margin)
    sp.set_ylim(py.min() - margin, py.max() + margin)

    # Gripper origin
    sp.scatter([0], [0], s=140, c=C_ROT, zorder=7, marker="+",
               linewidths=2.5, label="Gripper (origin)")
    # Full trajectory (very faded background)
    sp.plot(px, py, color="#353550", lw=0.6, zorder=1)

    # Animated artists on the spatial view
    trail_ln, = sp.plot([], [], lw=2.0, color=C_TRAIL,  zorder=2)
    pos_dot,  = sp.plot([], [], "o",   color=C_POS,     zorder=8,
                        ms=8, markeredgecolor="white", markeredgewidth=0.5)
    rot_ln,   = sp.plot([], [], "-",   color=C_ROT,     lw=1.8, zorder=4)
    vel_patch = FancyArrowPatch(
        (0, 0), (0, 0), arrowstyle="->",
        color=C_VEL, lw=2.0, mutation_scale=13, zorder=5
    )
    acc_patch = FancyArrowPatch(
        (0, 0), (0, 0), arrowstyle="->",
        color=C_ACC, lw=2.0, mutation_scale=13, zorder=5
    )
    sp.add_patch(vel_patch)
    sp.add_patch(acc_patch)

    sp.legend(
        handles=[
            Line2D([0],[0], color=C_TRAIL, lw=2,    label=f"Trail  (last {TRAIL_LEN})"),
            Line2D([0],[0], marker="o", color=C_POS, ms=7, lw=0, label="Position"),
            Line2D([0],[0], color=C_ROT,  lw=1.8,   label="Rotation angle"),
            Line2D([0],[0], color=C_VEL,  lw=2,     label="Velocity"),
            Line2D([0],[0], color=C_ACC,  lw=2,     label="Acceleration"),
        ],
        loc="upper right", fontsize=7.5,
        facecolor=PANEL, labelcolor=TXT, edgecolor=GRID_C,
    )

    # ── Three stacked time-series plots ──────────────────────────────────────
    ts_specs = [
        ("Position offset",             "m  /  rad",
         [px, py, pr],
         [f"Δ{ax0} (m)", f"Δ{ax1} (m)", "θ (rad)"]),
        ("Velocity  —  gripper frame",  "m s⁻¹  /  rad s⁻¹",
         [vx, vy, vr],
         [f"v{ax0.lower()} (m/s)", f"v{ax1.lower()} (m/s)", "ω (rad/s)"]),
        ("Acceleration  —  gripper frame", "m s⁻²  /  rad s⁻²",
         [bx, by, br],
         [f"a{ax0.lower()} (m/s²)", f"a{ax1.lower()} (m/s²)", "α (rad/s²)"]),
    ]
    S_COLS = [C_S0, C_S1, C_S2]

    ts_axes = []
    cursors = []
    for row, (title, ylabel, ys, lbls) in enumerate(ts_specs):
        a = fig.add_subplot(gs_ts[row])
        _style(a, title)
        a.set_xlim(t[0], t[-1])
        a.set_ylabel(ylabel, fontsize=7)
        a.grid(True, color=GRID_C, lw=0.4)
        for y, lbl, c in zip(ys, lbls, S_COLS):
            a.plot(t, y, color=c, lw=0.75, label=lbl)
        a.legend(loc="upper right", fontsize=6.5, ncol=3,
                 facecolor=PANEL, labelcolor=TXT, edgecolor=GRID_C)
        cursors.append(a.axvline(t[0], color=C_CUR, lw=1.3, zorder=10))
        if row < 2:
            a.set_xticklabels([])
        else:
            a.set_xlabel("Time  (s)", fontsize=8)
        ts_axes.append(a)

    # ── Control bar ───────────────────────────────────────────────────────────
    # Columns 0-4: five buttons | column 5: frame/time text | columns 6-8: slider
    btn_specs = [("◀◀", 0), ("◀", 1), ("▶", 2), ("▶", 3), ("▶▶", 4)]
    ctrl_axes = [fig.add_subplot(gs_ctrl[i]) for i in range(9)]
    for ca in ctrl_axes:
        ca.set_facecolor(BG)
        for s in ca.spines.values():
            s.set_visible(False)

    BC, BH = PANEL, "#313244"
    btn_rr = Button(ctrl_axes[0], "◀◀", color=BC, hovercolor=BH)
    btn_r  = Button(ctrl_axes[1], "◀",  color=BC, hovercolor=BH)
    btn_pp = Button(ctrl_axes[2], "▶",  color=BC, hovercolor=BH)
    btn_f  = Button(ctrl_axes[3], "▶",  color=BC, hovercolor=BH)
    btn_ff = Button(ctrl_axes[4], "▶▶", color=BC, hovercolor=BH)
    for b in [btn_rr, btn_r, btn_pp, btn_f, btn_ff]:
        b.label.set_color(TXT)
        b.label.set_fontsize(11)

    # Re-label the step buttons more clearly
    btn_r.label.set_text("◀")
    btn_f.label.set_text("▶")

    info_txt = ctrl_axes[5].text(
        0.5, 0.5, f"Frame 0 / {N-1}\nt = 0.000 s",
        ha="center", va="center", color=TXT, fontsize=8,
        transform=ctrl_axes[5].transAxes,
    )

    # Merge columns 6-8 for the slider using a SubplotSpec
    sld_ax = fig.add_subplot(gs_ctrl[6:9])
    sld_ax.set_facecolor(BG)
    for s in sld_ax.spines.values():
        s.set_visible(False)
    slider = Slider(sld_ax, "Frame", 0, N - 1, valinit=0, valstep=1,
                    color=C_S0)
    slider.label.set_color(TXT)
    slider.valtext.set_color(TXT)

    fig.suptitle("MoCap Offset Viewer", color=TXT, fontsize=13, y=0.998)

    # ── State ─────────────────────────────────────────────────────────────────
    state = {
        "idx":          0,
        "playing":      False,
        "t0_real":      0.0,   # perf_counter value when play was pressed
        "t0_data":      0.0,   # data-time at play start
    }

    # ── Artist update ─────────────────────────────────────────────────────────
    def _draw(i: int):
        # Spatial view
        lo = max(0, i - TRAIL_LEN)
        trail_ln.set_data(px[lo:i+1], py[lo:i+1])
        pos_dot.set_data([px[i]], [py[i]])
        rot_ln.set_data(
            [px[i], px[i] + ROT_LEN * np.cos(pr[i])],
            [py[i], py[i] + ROT_LEN * np.sin(pr[i])],
        )
        vel_patch.set_positions(
            (px[i], py[i]),
            (px[i] + vx[i] * VEL_SC, py[i] + vy[i] * VEL_SC),
        )
        acc_patch.set_positions(
            (px[i], py[i]),
            (px[i] + bx[i] * ACC_SC, py[i] + by[i] * ACC_SC),
        )
        # Time-series cursors
        for cur in cursors:
            cur.set_xdata([t[i], t[i]])
        # Frame info
        info_txt.set_text(f"Frame {i} / {N-1}\nt = {t[i]:.3f} s")
        # Slider (suppress re-entrant callback)
        slider.eventson = False
        slider.set_val(i)
        slider.eventson = True

    _draw(0)

    # ── Navigation helpers ────────────────────────────────────────────────────
    def _go(i: int):
        state["idx"] = int(np.clip(i, 0, N - 1))
        _draw(state["idx"])
        fig.canvas.draw_idle()

    def on_rr(_):  _go(state["idx"] - 10)
    def on_r(_):   _go(state["idx"] - 1)
    def on_f(_):   _go(state["idx"] + 1)
    def on_ff(_):  _go(state["idx"] + 10)

    def on_pp(_):
        if state["playing"]:
            state["playing"] = False
            btn_pp.label.set_text("▶")
        else:
            state["playing"]  = True
            btn_pp.label.set_text("⏸")
            state["t0_real"]  = time.perf_counter()
            state["t0_data"]  = t[state["idx"]]
        fig.canvas.draw_idle()

    def on_slider(v):
        if state["playing"]:               # dragging slider pauses playback
            state["playing"] = False
            btn_pp.label.set_text("▶")
        _go(int(v))

    btn_rr.on_clicked(on_rr)
    btn_r.on_clicked(on_r)
    btn_pp.on_clicked(on_pp)
    btn_f.on_clicked(on_f)
    btn_ff.on_clicked(on_ff)
    slider.on_changed(on_slider)

    # ── Animation timer (drives 1× real-time playback) ────────────────────────
    def _tick(_):
        if not state["playing"]:
            return
        elapsed = time.perf_counter() - state["t0_real"]
        target  = state["t0_data"] + elapsed
        i = int(np.searchsorted(t, target))
        if i >= N:
            i = N - 1
            state["playing"] = False
            btn_pp.label.set_text("▶")
        state["idx"] = i
        _draw(i)

    anim = animation.FuncAnimation(
        fig, _tick, interval=16, cache_frame_data=False   # ~60 fps poll
    )

    plt.show()
    return anim   # keep alive — GC would stop the animation


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV
    _anim = run(_load(csv_path))
