"""Publication-style diagrams of FACTOR (architecture, frames, estimator, training, real loop, calibration).

Every figure is drawn with Matplotlib primitives so it stays editable and reproducible; output goes
to PDF (for LaTeX), SVG (for the README) and PNG. Width 7 in matches a two-column paper.
FACTOR 도식 생성기. 논문용 PDF, README 용 SVG, PNG 로 내보낸다.

    python docs/figures/make_figures.py            # writes docs/figures/fig*.{pdf,svg,png}
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)

# Palette by role / 역할별 색
INK = "#1f2328"
MUTED = "#57606a"
ROLE = {
    "shared": ("#eef1f4", "#8c959f"),
    "sim": ("#e3edf8", "#3b6ea8"),
    "real": ("#fbeee0", "#b8732d"),
    "learn": ("#efe7f7", "#7451a8"),
    "rule": ("#e6f2ea", "#3f8a5a"),
    "io": ("#ffffff", "#57606a"),
    "warn": ("#fdecec", "#b54747"),
}


def canvas(w_in, h_in, w=100.0):
    """Figure with a data box ``w`` wide, aspect kept. / 가로 w 단위의 그림."""
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    h = w * h_in / w_in
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax, h


def box(ax, x, y, w, h, text, role="io", size=7.6, bold=False, sub=None, r=1.2, lw=0.8):
    """Rounded box centred at (x, y). Returns anchor points. / 중심 (x, y) 둥근 상자."""
    fc, ec = ROLE[role]
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=3
    )
    ax.add_patch(p)
    if sub:
        ax.text(
            x,
            y + h * 0.13,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=INK,
            fontweight="bold" if bold else "normal",
            zorder=4,
        )
        ax.text(
            x,
            y - h * 0.22,
            sub,
            ha="center",
            va="center",
            fontsize=size - 1.4,
            color=MUTED,
            zorder=4,
            style="italic",
        )
    else:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=INK,
            fontweight="bold" if bold else "normal",
            zorder=4,
            linespacing=1.15,
        )
    return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2), "c": (x, y)}


def group(ax, x0, y0, x1, y1, label, role="shared", dash=True, label_side="top-left", size=7.6):
    fc, ec = ROLE[role]
    p = FancyBboxPatch(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.6",
        fc=fc,
        ec=ec,
        lw=0.8,
        ls=(0, (4, 2)) if dash else "-",
        alpha=0.45,
        zorder=1,
    )
    ax.add_patch(p)
    lx, ha = (x0 + 1.2, "left") if label_side == "top-left" else (x1 - 1.2, "right")
    ax.text(lx, y1 - 1.0, label, ha=ha, va="top", fontsize=size, color=ec, fontweight="bold", zorder=2)


def arrow(
    ax,
    a,
    b,
    color=INK,
    lw=0.8,
    style="-|>",
    rad=0.0,
    ls="-",
    label=None,
    lpos=0.5,
    loff=(0, 1.4),
    lsize=6.6,
    shrink=1.0,
):
    ar = FancyArrowPatch(
        a,
        b,
        arrowstyle=style,
        mutation_scale=7,
        color=color,
        lw=lw,
        ls=ls,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=shrink,
        shrinkB=shrink,
        zorder=2,
    )
    ax.add_patch(ar)
    if label:
        mx = a[0] + (b[0] - a[0]) * lpos + loff[0]
        my = a[1] + (b[1] - a[1]) * lpos + loff[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=lsize,
            color=MUTED,
            style="italic",
            zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85),
        )


def poly(ax, pts, color=INK, lw=0.8, ls="-", end=True):
    """Orthogonal connector through points. / 꺾은선 연결."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1], strict=False):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
    arrow(
        ax,
        pts[-2],
        pts[-1],
        color=color,
        lw=lw,
        ls=ls,
        shrink=0.5 if not end else 1.0,
        style="-|>" if end else "-",
    )


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ============================================================================ Fig. 1 overview
def fig_overview():
    fig, ax, top = canvas(7.0, 3.45)
    # shared spec
    s = box(
        ax,
        50,
        top - 4.5,
        64,
        5.2,
        "Scene specification",
        "shared",
        bold=True,
        sub="robot URDF + kinematics · camera intrinsics · fiducial layout (anchor frame) · objects",
    )
    # lanes
    group(ax, 1.5, 11.5, 48.5, top - 9.5, "Simulation  (Isaac Lab digital twin)", "sim")
    group(
        ax,
        51.5,
        11.5,
        98.5,
        top - 9.5,
        "Real robot  (FAIRINO FR3 + AG-95, one RGB webcam)",
        "real",
        label_side="top-right",
    )
    yrow1, yrow2, yrow3 = top - 15.5, top - 24.5, top - 33.5
    a1 = box(ax, 12.5, yrow1, 18, 5.6, "Domain randomisation", "sim", sub="anchor, camera, appearance")
    a2 = box(ax, 36.5, yrow1, 18, 5.6, "Virtual camera aim", "sim", sub="look-at true anchor + jitter")
    a3 = box(ax, 12.5, yrow2, 18, 5.6, "Teacher + DAgger", "sim", sub="privileged belief error")
    a4 = box(ax, 36.5, yrow2, 18, 5.6, "Labels", "sim", sub="(Δx, Δy, Δψ) at peg, contact state")
    b1 = box(ax, 63.5, yrow1, 18, 5.6, "Webcam frame", "real", sub="2592×1944, 2.7 fps")
    b2 = box(ax, 87.5, yrow1, 18, 5.6, "ArUco anchor PnP", "real", sub=r"$T_{CA}$ every frame")
    b3 = box(ax, 63.5, yrow2, 18, 5.6, "Session alignment", "real", sub=r"belief $\hat{T}_{BA}$ (once)")
    b4 = box(ax, 87.5, yrow2, 18, 5.6, "Rotation homography", "real", sub=r"$H=K_vRK^{-1}$, foveated")
    arrow(ax, a1["r"], a2["l"])
    arrow(ax, a1["b"], a3["t"])
    arrow(ax, a3["r"], a4["l"])
    arrow(ax, b1["r"], b2["l"])
    arrow(ax, b2["b"], b4["t"])
    arrow(ax, b3["r"], b4["l"])
    # shared core
    c1 = box(
        ax,
        51,
        yrow3,
        24,
        6.2,
        "Fiducial-anchored canonical view",
        "shared",
        bold=True,
        sub="RGB + belief render + proprioception",
    )
    poly(ax, [a2["r"], (46.8, yrow1), (46.8, yrow3 + 3.1)])
    poly(ax, [b4["b"], (b4["b"][0], yrow3), c1["r"]])
    poly(ax, [a4["b"], (a4["b"][0], 11.3), (42.5, 11.3), (42.5, 9.6)], color=ROLE["sim"][1], ls="--")
    ax.text(31.0, 12.4, "supervision", fontsize=6.6, color=ROLE["sim"][1], style="italic", ha="center")
    e = box(
        ax, 51, 6.5, 22, 6.0, "Belief-conditioned estimator", "learn", bold=True, sub="5-model ensemble, σ"
    )
    arrow(ax, c1["b"], e["t"])
    k = box(
        ax,
        81,
        6.5,
        30,
        6.0,
        r"Funnel skill $\rightarrow$ IK $\rightarrow$ servo stream",
        "rule",
        bold=True,
        sub="height latch · lift-and-retry · safety monitor",
    )
    arrow(ax, e["r"], k["l"])
    ax.text(
        (e["r"][0] + k["l"][0]) / 2,
        2.3,
        "fused\ncorrection",
        fontsize=6.4,
        color=MUTED,
        style="italic",
        ha="center",
        linespacing=1.0,
    )
    t = box(ax, 16, 6.5, 24, 6.0, "Twin rollouts", "sim", bold=True, sub="99.9 % at ±8 mm, camera ±8 cm")
    arrow(ax, e["l"], t["r"], style="<|-|>")
    ax.text(
        (e["l"][0] + t["r"][0]) / 2,
        2.3,
        "train /\nevaluate",
        fontsize=6.4,
        color=MUTED,
        style="italic",
        ha="center",
        linespacing=1.0,
    )
    poly(
        ax,
        [k["t"], (k["t"][0], yrow2 - 4.5), (63.5, yrow2 - 4.5), (63.5, yrow2 - 2.8)],
        color=ROLE["real"][1],
        ls="--",
    )
    ax.text(
        73.5, yrow2 - 6.2, "joint angles", fontsize=6.6, color=ROLE["real"][1], style="italic", ha="center"
    )
    arrow(ax, s["b"], (25, top - 9.6), color=MUTED, lw=0.6)
    arrow(ax, s["b"], (75, top - 9.6), color=MUTED, lw=0.6)
    save(fig, "fig1_overview")


# ============================================================================ Fig. 2 frames
def fig_frames():
    import math

    fig, ax, top = canvas(7.0, 2.25)

    def frame(x, y, name, color, ang=0.0, size=4.2, lx=-1.2, ly=-1.2, ha="right"):
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        arrow(ax, (x, y), (x + size * ca, y + size * sa), color="#c0392b", lw=1.0, shrink=0)
        arrow(ax, (x, y), (x - size * sa, y + size * ca), color="#27864a", lw=1.0, shrink=0)
        ax.plot([x], [y], "o", ms=3.2, color=color, zorder=6)
        ax.text(x + lx, y + ly, name, fontsize=9.5, color=color, ha=ha, va="top", fontweight="bold", zorder=6)

    # work table, ArUco mat, tower / 작업대, 매트, 탑
    ax.add_patch(
        Polygon([[24, 2.5], [80, 2.5], [86, 14], [30, 14]], closed=True, fc="#f3f4f6", ec="#c9ced4", lw=0.7)
    )
    ax.add_patch(
        Polygon([[42, 4.5], [64, 4.5], [67.5, 11.5], [45.5, 11.5]], closed=True, fc="white", ec=INK, lw=0.6)
    )
    for mx, my in ((44.8, 5.5), (54, 5.5), (62.6, 5.5), (48, 10.4), (57, 10.4), (65.6, 10.4)):
        ax.add_patch(Rectangle((mx - 1.0, my - 0.55), 2.0, 1.1, fc=INK, ec=INK, lw=0.2))
    ax.add_patch(
        Polygon(
            [[47, 7.2], [62, 7.2], [62.9, 8.8], [47.9, 8.8]], closed=True, fc="#e8d3b0", ec="#9c7a4b", lw=0.5
        )
    )
    for px in (50.5, 55.5, 60.5):
        ax.add_patch(Rectangle((px - 0.35, 8.0), 0.7, 6.5, fc="#d9b98a", ec="#9c7a4b", lw=0.4, zorder=3))
    for k, (w, c) in enumerate(((4.6, "#7a4fb0"), (3.9, "#2f9e5e"), (3.2, "#e2532d"), (2.5, "#f2c12e"))):
        ax.add_patch(Rectangle((50.5 - w / 2, 8.7 + k * 0.85), w, 0.85, fc=c, ec="none", zorder=4))
    # robot / 로봇
    ax.add_patch(Rectangle((4, 2.5), 10, 3.2, fc="#dfe3e8", ec=MUTED, lw=0.6))
    ax.plot(
        [9, 10.5, 27, 38.5], [5.7, 19.5, 24.5, 21.0], color="#8c959f", lw=5, solid_capstyle="round", zorder=2
    )
    ax.plot([38.5, 38.5], [21.0, 17.6], color=INK, lw=2.4, zorder=2)
    ax.plot([37.6, 37.6], [17.6, 15.8], color=INK, lw=1.0)
    ax.plot([39.4, 39.4], [17.6, 15.8], color=INK, lw=1.0)
    # camera / 카메라
    ax.add_patch(Polygon([[90.5, 26.5], [97, 29.5], [97, 23.5]], closed=True, fc="#2d333b", ec=INK, lw=0.5))
    ax.text(97, 22.2, "webcam", fontsize=6.6, color=MUTED, ha="right", va="top")
    frame(9, 5.7, "B", INK, lx=-1.4, ly=-0.6)
    frame(55.5, 8.0, "A", ROLE["rule"][1], size=3.6, lx=3.2, ly=-0.9, ha="left")
    frame(50.0, 3.3, r"$\hat{A}$", ROLE["learn"][1], ang=-7, size=3.6, lx=-0.9, ly=0.9)
    frame(90.5, 26.5, "C", ROLE["real"][1], ang=205, size=3.6, lx=1.2, ly=-1.6, ha="left")
    # transforms / 변환
    arrow(ax, (12.5, 4.6), (48.8, 3.0), color=ROLE["learn"][1], rad=0.08, ls="--")
    ax.text(
        30,
        1.0,
        r"belief $\hat{T}_{BA}$  (set once per session)",
        fontsize=7,
        color=ROLE["learn"][1],
        ha="center",
        style="italic",
    )
    arrow(ax, (88.6, 25.2), (57.2, 9.8), color=ROLE["real"][1], rad=-0.12)
    ax.text(
        80,
        15.6,
        r"$T_{CA}$ from ArUco PnP," + "\n" + "every frame",
        fontsize=7,
        color=ROLE["real"][1],
        ha="left",
        style="italic",
        linespacing=1.1,
    )
    arrow(ax, (10.8, 8.5), (88.4, 27.6), color=MUTED, rad=-0.22, ls=":")
    ax.text(
        58,
        31.0,
        r"$T_{BC}$: not needed to build the view",
        fontsize=7,
        color=MUTED,
        ha="center",
        style="italic",
    )
    arrow(ax, (50.4, 3.5), (55.0, 7.7), color=ROLE["warn"][1], lw=1.1, shrink=0.3)
    ax.text(
        70,
        3.6,
        r"belief error $(\Delta x,\Delta y,\Delta\psi)$" + "\n" + "= what the estimator learns",
        fontsize=7,
        color=ROLE["warn"][1],
        ha="left",
        style="italic",
        linespacing=1.1,
    )
    save(fig, "fig2_frames")


# ============================================================================ Fig. 3 estimator
def fig_estimator():
    fig, ax, top = canvas(7.0, 2.75)
    i1 = box(ax, 9, top - 6, 15, 7.5, "Canonical RGB", "io", sub="128×128×3")
    i2 = box(ax, 9, top - 16, 15, 7.5, "Belief render", "learn", sub="finger segment, σ=1.5 px")
    i3 = box(ax, 9, top - 26, 15, 7.5, "Proprioception", "io", sub=r"believed grip $\in A$ (×0.1 m)")
    cat = box(ax, 27, top - 11, 7, 17, "concat\n4 ch", "shared", size=7)
    arrow(ax, i1["r"], (23.5, top - 6))
    arrow(ax, i2["r"], (23.5, top - 16))
    enc = box(
        ax, 43, top - 11, 18, 12, "CoordConv encoder", "learn", bold=True, sub="conv ×4, stride 2, + x/y maps"
    )
    arrow(ax, cat["r"], enc["l"])
    # little feature-map stack
    pm = box(ax, 43, top - 26, 18, 6.5, "MLP", "learn", sub="64-d embedding")
    arrow(ax, i3["r"], pm["l"])
    box(ax, 60, top - 15, 9, 15, "fuse\n⊕", "shared", size=7.4)
    arrow(ax, enc["r"], (55.5, top - 11))
    poly(ax, [pm["r"], (57.5, top - 26), (57.5, top - 22.5)])
    heads = [
        ("Correction", r"$(\Delta x,\Delta y,\Delta\psi)$ at peg", "learn", top - 4.5),
        ("Log-variance", r"heteroscedastic $\sigma$", "learn", top - 12.5),
        ("Contact state", "free · between · grasped · seated", "learn", top - 20.5),
        ("In-hand offset", "ablation head", "io", top - 28.5),
    ]
    for name, sub, role, y in heads:
        hb = box(ax, 76, y, 17, 6.4, name, role, sub=sub)
        poly(ax, [(64.5, top - 15), (66.5, top - 15), (66.5, y), hb["l"]])
    box(ax, 93, top - 12, 10.5, 15, "Ensemble\n×5\nmean +\nspread", "rule", size=7)
    arrow(ax, (84.5, top - 4.5), (87.7, top - 8))
    arrow(ax, (84.5, top - 12.5), (87.7, top - 12.5))
    ax.text(
        50,
        1.6,
        "Loss: Gaussian NLL on the correction + cross-entropy on the contact state; labels are "
        "local (exact translation at the peg in view).",
        fontsize=6.8,
        color=MUTED,
        ha="center",
        style="italic",
    )
    save(fig, "fig3_estimator")


# ============================================================================ Fig. 4 training
def fig_training():
    fig, ax, top = canvas(7.0, 2.35)
    group(ax, 1.5, 4.0, 63, top - 1.5, "Isaac Lab twin · 128 parallel envs · 4×RTX 4090", "sim")
    r = box(ax, 11, top - 9, 15, 7, "Randomise", "sim", sub="anchor, camera, look")
    t = box(ax, 30, top - 9, 15, 7, "Teacher rollout", "sim", sub="true belief error")
    b = box(ax, 49, top - 9, 16, 7, "Frame buffer", "shared", sub="1200 + 3×400 steps")
    d = box(ax, 30, 11.5, 15, 7, "Student rollout", "sim", sub="DAgger round k")
    arrow(ax, r["r"], t["l"])
    arrow(ax, t["r"], b["l"])
    poly(ax, [d["r"], (49, 11.5), b["b"]])
    tr = box(ax, 74, top - 9, 15, 7, "Supervised fit", "learn", bold=True, sub="NLL + CE, 60+3×30 ep")
    arrow(ax, b["r"], tr["l"])
    poly(ax, [tr["b"], (74, 6.3), (30, 6.3), d["b"]], color=ROLE["learn"][1], ls="--")
    ax.text(
        58,
        7.4,
        "student policy for the next round",
        fontsize=6.6,
        color=ROLE["learn"][1],
        style="italic",
        ha="center",
    )
    g = box(ax, 91, top - 9, 13, 7, "Warm-start\ngenerations", "learn", size=7)
    arrow(ax, tr["r"], g["l"])
    e = box(ax, 91, 11.5, 13, 7, "5-model\nensemble", "rule", size=7)
    arrow(ax, g["b"], e["t"])
    ax.text(
        50,
        1.6,
        "Evaluation in the twin: 14 s episodes, anchor error × camera displacement grid; "
        "rules-only and teacher as references.",
        fontsize=6.8,
        color=MUTED,
        ha="center",
        style="italic",
    )
    save(fig, "fig4_training")


# ============================================================================ Fig. 5 real loop
def fig_realloop():
    fig, ax, top = canvas(7.0, 2.05)
    group(ax, 1.5, 16.0, 98.5, top - 0.8, "Perception, every tick (10 Hz)", "learn")
    group(ax, 1.5, 0.8, 98.5, 13.6, "Action, every tick", "rule", label_side="top-right")
    y1, y2 = top - 7.6, 5.8
    p = [
        box(ax, 10, y1, 14, 7, "Webcam frame", "real", sub="shared PNG + SHA"),
        box(ax, 27, y1, 15, 7, "Anchor detection", "real", sub=r"ArUco PnP $\rightarrow T_{CA}$"),
        box(ax, 45, y1, 16, 7, "Canonical view", "shared", sub="look-at active peg"),
        box(ax, 63, y1, 15, 7, "Belief channel", "learn", sub="drawn from FK grip"),
        box(ax, 82, y1, 16, 7, "Estimator server", "learn", sub="ensemble, σ, spread"),
    ]
    for a, b in zip(p[:-1], p[1:], strict=False):
        arrow(ax, a["r"], b["l"])
    q = [
        box(ax, 82, y2, 16, 7, "Session fusion", "rule", sub="inverse-variance, gate"),
        box(ax, 63, y2, 15, 7, "Funnel planner", "rule", sub="hover · descend · insert"),
        box(ax, 45, y2, 16, 7, "Workspace clamp + IK", "rule", sub="grasp posture branch"),
        box(ax, 27, y2, 15, 7, "Servo limiter", "real", sub="speed, accel, brake"),
        box(ax, 10, y2, 14, 7, "FR3 ServoJ", "real", sub="16 ms, safety monitor"),
    ]
    arrow(ax, p[-1]["b"], q[0]["t"])
    for a, b in zip(q[:-1], q[1:], strict=False):
        arrow(ax, a["l"], b["r"])
    poly(ax, [q[-1]["t"], (10, 14.8), (63, 14.8), (63, y1 - 3.5)], color=ROLE["real"][1], ls="--")
    ax.text(
        36.5,
        15.6,
        r"joint angles $\rightarrow$ believed grip point",
        fontsize=6.6,
        color=ROLE["real"][1],
        style="italic",
        ha="center",
    )
    ax.text(50, top - 0.1, "", fontsize=1)
    save(fig, "fig5_realloop")


# ============================================================================ Fig. 6 self-calibration
def fig_selfcalib():
    fig, ax, top = canvas(7.0, 2.2)
    c = box(ax, 12, top - 8, 20, 8, "Candidate poses", "shared", sub="~400, tilt ±28°, yaw ±60°")
    p = box(ax, 36, top - 8, 20, 8, "Predict", "learn", sub="visibility + information per pose")
    s = box(
        ax,
        61,
        top - 8,
        22,
        8,
        "Select",
        "rule",
        bold=True,
        sub=r"max $\Delta\log\sigma^2_{\mathrm{peg}}$ / seconds",
    )
    m = box(ax, 86, top - 8, 18, 8, "Move + observe", "real", sub="min-jerk, 2 fresh frames")
    o = box(ax, 86, 9, 18, 8, "Board + mat", "real", sub=r"corners, $T_{CA}$")
    j = box(
        ax, 61, 9, 22, 8, "Joint LM", "learn", bold=True, sub=r"$X=T_{BC},\ Y=T_{FG}$, per-pose whitening"
    )
    u = box(ax, 36, 9, 20, 8, "Uncertainty", "shared", sub=r"$\sigma$ of peg points via $X\,T_{CA}$")
    r = box(ax, 12, 9, 20, 8, r"$\hat{T}_{BA}=X\,T_{CA}$", "rule", bold=True, sub="written for the run")
    arrow(ax, c["r"], p["l"])
    arrow(ax, p["r"], s["l"])
    arrow(ax, s["r"], m["l"])
    arrow(ax, m["b"], o["t"])
    arrow(ax, o["l"], j["r"])
    arrow(ax, j["l"], u["r"])
    arrow(ax, u["t"], p["b"], label="repeat until σ < target or 60 s", loff=(0, 0), lpos=0.5)
    arrow(ax, u["l"], r["r"], label="stop", loff=(0, 1.6))
    ax.text(
        50,
        1.4,
        "Synthetic, 20 seeds: 0.23 mm (active) vs 0.31 mm (random) peg-point error after 60 s; a "
        "detector miss tightens the predicted viewing-angle limit.",
        fontsize=6.6,
        color=MUTED,
        ha="center",
        style="italic",
    )
    save(fig, "fig6_selfcalib")


if __name__ == "__main__":
    fig_overview()
    fig_frames()
    fig_estimator()
    fig_training()
    fig_realloop()
    fig_selfcalib()
