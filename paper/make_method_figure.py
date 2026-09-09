#!/usr/bin/env python3
"""Draw the camera-ready SetHL method overview as a vector PDF."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

NAVY = "#18364A"; TEAL = "#168B8C"; ORANGE = "#E58A2A"
LIGHT = "#F4F7F8"; MID = "#DCE7EB"; TEXT = "#23323A"; GRAY = "#667780"


def box(ax, xy, wh, title, subtitle="", face=LIGHT, edge=MID):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                           linewidth=1.1, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - .038, title, ha="center", va="top",
            fontsize=7.1, color=TEXT, fontweight="bold")
    if subtitle:
        ax.text(x + w / 2, y + .012, subtitle, ha="center", va="bottom",
                fontsize=4.9, color=GRAY, linespacing=1.0)
    return patch


def arrow(ax, start, end, label=""):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                linewidth=1.1, color=NAVY, shrinkA=2, shrinkB=2))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .025,
                label, ha="center", fontsize=6.2, color=GRAY)


def draw_hand(ax, cx, cy, scale, hidden_shift=0.0, lw=1.25):
    wrist = np.array([cx, cy - .34 * scale])
    bases = np.array([[-.25, -.04], [-.13, .04], [0, .06], [.13, .035], [.24, -.02]])
    tips = np.array([[-.42, .22], [-.20, .47], [-.03, .54], [.14, .47], [.30, .34]])
    for i, (base, tip) in enumerate(zip(bases, tips)):
        b = np.array([cx, cy]) + scale * base
        t = np.array([cx, cy]) + scale * tip
        if i == 1:
            t[0] += hidden_shift * scale
        mid1 = b + .38 * (t - b); mid2 = b + .70 * (t - b)
        color = ORANGE if i == 1 else NAVY
        pts = np.vstack((wrist, b, mid1, mid2, t))
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, solid_capstyle="round")
        ax.scatter(pts[1:, 0], pts[1:, 1], s=3.8, color=color, zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(7.05, 2.35))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 1. Partial symbolic program.
    box(ax, (.012, .17), (.17, .72), "Partial HL program", "wrist + palm pose\nalways given")
    gx, gy = .030, .33; cw, ch = .012, .031
    for row in range(10):
        for col in range(10):
            hidden = (3 <= row <= 5) or (col in (7, 8) and row > 1)
            color = "white" if hidden else TEAL
            ax.add_patch(Rectangle((gx + col * cw, gy + row * ch), cw * .82, ch * .76,
                                   facecolor=color, edgecolor=MID, linewidth=.35))
    ax.text(.060, .685, "given", fontsize=6.2, color=TEAL,
            ha="center", fontweight="bold")
    ax.text(.145, .685, "free", fontsize=6.2, color=GRAY,
            ha="center", fontweight="bold")
    ax.text(.097, .286, r"$M,\;M\odot z$", fontsize=6.5, va="center",
            ha="center", color=TEXT)

    # 2. Completion posterior.
    box(ax, (.224, .17), (.17, .72), "Masked completion", r"$q_\phi(z_{t,b}\mid z_{\rm obs},M)$")
    for k, width in enumerate((.108, .124, .096, .118)):
        yy = .67 - k * .083
        ax.add_patch(FancyBboxPatch((.247, yy), width, .050,
                                    boxstyle="round,pad=.002,rounding_size=.006",
                                    facecolor="#E5F2F2", edgecolor=TEAL, linewidth=.75))
        for j in range(4):
            ax.plot([.260 + j * .024, .260 + j * .024], [yy + .010, yy + .040],
                    color=TEAL, lw=.7, alpha=.8)
    bx = .247
    probs = (.055, .018, .036, .074, .025)
    for j, ph in enumerate(probs):
        ax.add_patch(Rectangle((bx + j * .022, .287), .014, ph,
                               facecolor=ORANGE if j == 3 else MID, edgecolor="none"))
    ax.text(.300, .265, "symbol + in-cell sample", fontsize=6.0, color=GRAY, ha="center")

    # 3. Kinematic spatialization.
    box(ax, (.438, .17), (.17, .72), "Differentiable FK", "no joint-coordinate\nbypass")
    draw_hand(ax, .523, .515, .30, hidden_shift=.14, lw=1.35)
    ax.add_patch(Rectangle((.466, .310), .114, .035, facecolor="#DCEEEF",
                           edgecolor="none", alpha=.9))
    ax.text(.523, .327, "16-ch control latent", fontsize=6.0, color=TEAL,
            ha="center", va="center", fontweight="bold")

    # 4. Frozen Wan + LoRA.
    box(ax, (.650, .17), (.155, .72), "Wan-Control", "flow matching", face="#EDF2F5")
    ax.add_patch(FancyBboxPatch((.674, .38), .107, .257,
                                boxstyle="round,pad=.005,rounding_size=.010",
                                facecolor=NAVY, edgecolor=NAVY))
    ax.text(.7275, .535, "Frozen", color="white", fontsize=7.2, ha="center")
    ax.text(.7275, .488, "Wan 1.3B", color="white", fontsize=8.1,
            ha="center", fontweight="bold")
    ax.add_patch(FancyBboxPatch((.682, .346), .091, .055,
                                boxstyle="round,pad=.004,rounding_size=.008",
                                facecolor=ORANGE, edgecolor="white", linewidth=.7))
    ax.text(.7275, .374, "rank-16 LoRA", fontsize=6.0, color="white",
            ha="center", va="center", fontweight="bold")

    # 5. Compatible samples; navy structure is fixed, orange finger is free.
    box(ax, (.847, .17), (.142, .72), "Compatible videos", "same constraint,\nlocal variation")
    for j, shift in enumerate((-.15, 0.0, .18)):
        yy = .700 - j * .150
        ax.add_patch(FancyBboxPatch((.865, yy - .115), .106, .137,
                                    boxstyle="round,pad=.002,rounding_size=.005",
                                    facecolor="white", edgecolor=MID, linewidth=.7))
        draw_hand(ax, .918, yy - .055, .135, hidden_shift=shift, lw=.82)

    for x0, x1 in ((.182, .224), (.394, .438), (.608, .650), (.805, .847)):
        arrow(ax, (x0, .53), (x1, .53))
    ax.text(.50, .075, "specified symbols preserved", color=TEAL, fontsize=6.8,
            ha="right", fontweight="bold")
    ax.text(.515, .075, "•", color=GRAY, fontsize=7, ha="center")
    ax.text(.53, .075, "unspecified cells sampled", color=ORANGE, fontsize=6.8,
            ha="left", fontweight="bold")

    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "method_overview.pdf", bbox_inches="tight", pad_inches=.02)
    fig.savefig(out / "method_overview.png", dpi=220, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
