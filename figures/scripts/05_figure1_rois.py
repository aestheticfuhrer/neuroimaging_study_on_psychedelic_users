#!/usr/bin/env python3
"""Figure 1: 22 ROIs of the DMN, CEN and SN on a glass brain.

Coordinates are in Table S2 of the Supplement (MNI x y z).

Layout rationale
----------------
Labels carry the bare abbreviation with no "L-"/"R-" prefix; hemisphere is read off
the panel (sagittals) or off the L/R markers and the midline (axial). On the axial
view each name therefore appears twice, once per hemisphere, which is intended.

Labels are placed adjacent to their own marker by a candidate-scoring search rather
than by a repulsion solver, because with no leader lines a label is only readable if
it stays next to the sphere it belongs to.

Output: vector PDF + 600 dpi PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from nilearn import plotting
import os
from pathlib import Path

# Anchored on this file so the figure is written inside this repository,
REPO = Path(__file__).resolve().parents[2]
# PSY_ROOT redirects the output tree, matching 00_theme_psychodel.R.
FIG_OUT = Path(os.environ.get("PSY_ROOT", REPO)) / "figures" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

COL = {"DMN": "#7B3F98", "CEN": "#ECB021", "SN": "#7AAC33"}

FIG_W_IN = 190 / 25.4          # double-column width, drawn at final size
# Axial (0.78:1) beside the two sagittals (1.2:1) stacked gives a combined aspect of
# 0.78 + 0.60, so height ~= width / 1.38.
FIG_H_IN = 5.35
LABEL_PT = 8.0                 # minimum 8 pt
MARKER_S = 100                 # points^2; ~11 pt diameter, ~6% of panel width
DODGE_FRAC = 0.62              # min centre separation, as a fraction of sphere diameter
MERGE_MM = 25                  # name a bilateral pair once if it projects this close
CLEAR_PX = 6.0                 # how much nearer a label must be to its own sphere

ROIS = [
    # DMN (9)
    ("L-IFG",      -42,  47,  -7, "DMN"),
    ("L-mPFC",      -4,  51,  -7, "DMN"),
    ("R-mPFC",       5,  53,  -7, "DMN"),
    ("L-aSTG",     -59,  -5,  -6, "DMN"),
    ("R-aSTG",      59,   0,  -8, "DMN"),
    ("L-PREC/PCC",  -9, -49,  38, "DMN"),
    ("R-PREC/PCC",   9, -49,  38, "DMN"),
    ("L-ANG",      -51, -64,  32, "DMN"),
    ("R-ANG",       54, -52,  26, "DMN"),
    # CEN (9)
    ("R-IFG",       48,  47,  -7, "CEN"),
    ("L-SFG",      -15,  20,  62, "CEN"),
    ("R-SFG",       16,  18,  62, "CEN"),
    ("L-pSTG/TPJ", -66, -28,   8, "CEN"),
    ("R-pSTG/TPJ",  66, -30,  11, "CEN"),
    ("L-ITG",      -60, -56, -13, "CEN"),
    ("R-ITG",       63, -54, -13, "CEN"),
    ("L-SUP",      -62, -35,  36, "CEN"),
    ("R-SUP",       61, -40,  41, "CEN"),
    # SN (4)
    ("L-dmACC",     -6,   6,  38, "SN"),
    ("R-dmACC",      3,   6,  41, "SN"),
    ("L-INS",      -37,  17,  -3, "SN"),
    ("R-INS",       37,  18,  -4, "SN"),
]
assert len(ROIS) == 22

# MNI -> 2D projection per glass-brain view
PROJ = {"l": lambda x, y, z: (y, z),   # left sagittal
        "r": lambda x, y, z: (y, z),   # right sagittal
        "z": lambda x, y, z: (x, y)}   # dorsal (axial)

# which ROIs nilearn draws in which panel
IN_PANEL = {"l": lambda x: x < 0, "r": lambda x: x > 0, "z": lambda x: True}


def dodge(points, min_sep, iters=400):
    """Push apart markers whose projected centres are closer than min_sep.

    Several ROI pairs are genuinely within a few mm of each other once a 3D
    coordinate is flattened to 2D (IFG/mPFC differ almost only in x, so they
    collapse on a sagittal projection). No sphere large enough to see can
    resolve a 4 mm gap, so the pair is separated just far enough that both
    outlines stay legible. Displacement is symmetric, so the centroid holds.
    """
    p = np.asarray(points, dtype=float).copy()
    for _ in range(iters):
        moved = False
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                d = p[j] - p[i]
                dist = float(np.hypot(*d))
                if dist >= min_sep:
                    continue
                if dist < 1e-9:
                    d, dist = np.array([1.0, 0.0]), 1e-9
                shift = (min_sep - dist) / 2.0 * (d / dist)
                p[i] -= shift
                p[j] += shift
                moved = True
        if not moved:
            break
    return p


def _overlap(a, b):
    """Intersection area of two display-space bboxes."""
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _gap_to_box(p, bb):
    dx = max(bb.x0 - p[0], 0.0, p[0] - bb.x1)
    dy = max(bb.y0 - p[1], 0.0, p[1] - bb.y1)
    return float(np.hypot(dx, dy))


DIRS = [(float(np.cos(a)), float(np.sin(a)))
        for a in np.deg2rad(np.arange(0, 360, 22.5))]
GAPS = [2.0, 6.0, 11.0, 17.0]   # clearance beyond the sphere edge, in points

# Directions fixed by hand where the automatic pick is correct but reads awkwardly.
# The solver still chooses the distance, so collisions are still avoided.
PINS = {"z": {"PREC/PCC": (0.0, 1.0)}}   # centred directly above the bilateral pair


def place_labels(fig, ax, entries, markers, r_pt, fontsize, fixed=(), pins=()):
    """Put each label beside the sphere(s) it names, with no leader lines.

    entries are (text, anchor_x, anchor_y, owned), where owned indexes into markers.
    A label may own more than one sphere: bilateral homologs sitting almost on the
    midline are named once, between the pair, since two stacked copies of the same
    abbreviation cannot be told apart anyway.

    For every label we score 16 directions x 4 distances around its anchor and keep
    the cheapest. Costs, in order of severity: landing nearer a sphere the label does
    NOT name (that is what makes a leaderless label unreadable), overlapping another
    label, covering a sphere, then drifting from the anchor. Leaving the axes is
    disqualifying. Labels are committed most-crowded-first, then swept again.
    """
    pins = dict(pins)
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent()
    r_px = r_pt * fig.dpi / 72.0
    pt_px = fig.dpi / 72.0

    centres = ax.transData.transform(np.asarray(markers, dtype=float))
    marker_boxes = [matplotlib.transforms.Bbox.from_extents(
        cx - r_px, cy - r_px, cx + r_px, cy + r_px) for cx, cy in centres]

    texts = [ax.annotate(name, xy=(px, py), xycoords="data",
                         textcoords="offset points", xytext=(0, 0),
                         fontsize=fontsize, color="#1a1a1a", zorder=200,
                         annotation_clip=False)
             for name, px, py, _ in entries]

    anchors = ax.transData.transform(
        np.array([[e[1], e[2]] for e in entries], dtype=float))

    def standoff(i, ux, uy):
        u = np.array([ux, uy], dtype=float)
        reach = max(float(np.dot(centres[k] - anchors[i], u)) for k in entries[i][3])
        return max(reach, 0.0) / pt_px + r_pt

    def candidates(i):
        t, out = texts[i], []
        for ux, uy in (pins[entries[i][0]],) if entries[i][0] in pins else DIRS:
            ha = "left" if ux > 0.35 else ("right" if ux < -0.35 else "center")
            va = "bottom" if uy > 0.35 else ("top" if uy < -0.35 else "center")
            base_d = standoff(i, ux, uy)
            for gap in GAPS:
                d = base_d + gap
                t.set_ha(ha); t.set_va(va)
                t.xyann = (ux * d, uy * d)
                bb = t.get_window_extent(renderer).expanded(1.06, 1.20)
                out.append(((ux * d, uy * d), ha, va, bb, gap))
        return out

    # crowding = neighbours within 4 marker radii; densest areas get first pick
    order = sorted(range(len(entries)), key=lambda i: -sum(
        1 for j in range(len(entries))
        if j != i and np.hypot(*(anchors[i] - anchors[j])) < 4 * r_px))

    placed = {}

    def cost_of(i, bb, gap, others):
        c = gap * 10.0                                   # hug your own sphere
        for mb in marker_boxes:
            c += _overlap(bb, mb) * 3.0                  # do not cover a sphere
        for ob in others:
            c += _overlap(bb, ob) * 6.0                  # do not cover a label
        # a leaderless label is readable only if the nearest sphere is one it names,
        # by a clear margin rather than a hair
        owned = entries[i][3]
        own = min(_gap_to_box(centres[k], bb) for k in owned)
        for k in range(len(centres)):
            if k in owned:
                continue
            c += 150.0 * max(0.0, own + CLEAR_PX - _gap_to_box(centres[k], bb))
        return c

    def choose(i, others):
        t = texts[i]
        best, best_cost = None, np.inf
        for attempt in (True, False):   # second pass drops the in-axes requirement
            for off, ha, va, bb, gap in candidates(i):
                if attempt and not (ax_bb.x0 <= bb.x0 and bb.x1 <= ax_bb.x1
                                    and ax_bb.y0 <= bb.y0 and bb.y1 <= ax_bb.y1):
                    continue
                c = cost_of(i, bb, gap, others)
                if c < best_cost:
                    best, best_cost = (off, ha, va, bb), c
            if best is not None:
                break
        off, ha, va, bb = best
        t.set_ha(ha); t.set_va(va); t.xyann = off
        placed[i] = bb

    fixed = list(fixed)    # e.g. nilearn's own "L"/"R" hemisphere markers
    for i in order:
        choose(i, fixed + [placed[k] for k in placed])
    for _ in range(4):     # refinement sweeps against the final layout
        for i in order:
            choose(i, fixed + [placed[k] for k in placed if k != i])

    # Self-check. With no leader lines the figure is only correct if every label's
    # nearest sphere is one it names; anything else silently mislabels an ROI.
    bad = []
    for i, e in enumerate(entries):
        bb = placed[i]
        d = [_gap_to_box(c, bb) for c in centres]
        nearest = int(np.argmin(d))
        if nearest not in e[3]:
            bad.append((e[0], round(min(d[k] for k in e[3]), 1), round(d[nearest], 1)))
    return texts, bad


fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
PANELS = {"z": (0.005, 0.075, 0.545, 0.915),
          "l": (0.565, 0.530, 0.430, 0.460),
          "r": (0.565, 0.070, 0.430, 0.460)}

axes_for = {}
for key, rect in PANELS.items():
    d = plotting.plot_glass_brain(None, display_mode=key, figure=fig, axes=rect,
                                  alpha=0.35, black_bg=False)
    axes_for[key] = list(d.axes.values())[0].ax

# Breathing room inside each panel so labels stay within their own axes. The margin
# is asymmetric in y: the superior ROIs (SFG, SUP, PREC/PCC) need room above them,
# while the inferior brain holds no ROIs at all, so a bottom margin would only add
# an empty band. Signed spans keep this correct on the inverted axes.
for key, ax in axes_for.items():
    x0, x1 = ax.get_xlim()
    mx = 0.11 * (x1 - x0)
    ax.set_xlim(x0 - mx, x1 + mx)
    y0, y1 = ax.get_ylim()
    ry = y1 - y0
    ax.set_ylim(y0 - 0.04 * ry, y1 + 0.13 * ry)

fig.canvas.draw()
n_bad = 0

for key, ax in axes_for.items():
    keep = [(nm, x, y, z, net) for (nm, x, y, z, net) in ROIS if IN_PANEL[key](x)]
    pts = np.array([PROJ[key](x, y, z) for (_, x, y, z, _) in keep], dtype=float)
    cols = [COL[net] for *_, net in keep]

    # sphere radius expressed in this panel's data units
    d_pt = 2.0 * np.sqrt(MARKER_S / np.pi)
    x0, x1 = ax.get_xlim()
    span_px = ax.get_window_extent().width
    data_per_pt = abs(x1 - x0) / span_px * fig.dpi / 72.0
    d_data = d_pt * data_per_pt

    pts = dodge(pts, DODGE_FRAC * d_data)
    ax.scatter(pts[:, 0], pts[:, 1], s=MARKER_S, c=cols,
               edgecolors="white", linewidths=1.0, zorder=100, clip_on=False)

    # nilearn's own "L"/"R" hemisphere markers on the axial must not be written over
    fixed = [t.get_window_extent(fig.canvas.get_renderer())
             for t in ax.texts if t.get_text() in ("L", "R")]

    # Name a structure once where its two hemispheric nodes nearly coincide in this
    # projection. On the axial that folds the midline pairs (mPFC 9 mm, dmACC 9 mm,
    # PREC/PCC 18 mm) into one label each; lateral pairs stay far apart and keep
    # their own. On the sagittals every abbreviation is already unique, so this is
    # a no-op there.
    base = [nm.split("-", 1)[1] for nm, *_ in keep]
    entries, used = [], set()
    for i in range(len(keep)):
        if i in used:
            continue
        group = [i] + [j for j in range(i + 1, len(keep))
                       if j not in used and base[j] == base[i]
                       and np.hypot(*(pts[j] - pts[i])) < MERGE_MM]
        used.update(group)
        ax_, ay_ = pts[group].mean(axis=0)
        entries.append((base[i], ax_, ay_, group))

    _, bad = place_labels(fig, ax, entries, pts, d_pt / 2.0, LABEL_PT,
                          fixed=fixed, pins=PINS.get(key, {}))
    for nm, own_d, near_d in bad:
        print(f"  !! panel {key}: '{nm}' is {near_d}px from another sphere "
              f"but {own_d}px from its own")
    n_bad += len(bad)

handles = [Line2D([], [], marker="o", linestyle="none", markersize=8,
                  markerfacecolor=COL[k], markeredgecolor="white", label=k)
           for k in ("DMN", "CEN", "SN")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, 0.002), handletextpad=0.4,
           columnspacing=2.2)

out = str(FIG_OUT / "figure1_rois")
fig.savefig(out + ".pdf", format="pdf")
fig.savefig(out + ".png", dpi=600)
print("wrote", out + ".pdf (vector) and", out + ".png (600 dpi)")
nets = [n for *_, n in ROIS]
print(f"DMN {nets.count('DMN')}, CEN {nets.count('CEN')}, SN {nets.count('SN')} = {len(ROIS)} ROIs")
print("label/sphere association check:",
      "OK, every label is nearest its own sphere" if n_bad == 0
      else f"{n_bad} AMBIGUOUS")
