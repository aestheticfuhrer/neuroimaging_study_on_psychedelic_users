#!/usr/bin/env python3
"""Figure 4: specification curve. Applies manuscript labels and writes
PNG/PDF/SVG; plot_specification_curve in
effective_connectivity/scripts/visualizations.py does the drawing.

Run 07_spec_curve_extract.py first to write the spec_cache CSVs.
"""
import matplotlib

matplotlib.use("Agg")

# fonttype 42 embeds TrueType; matplotlib's Type 3 default is rejected by several
# journals. svg.fonttype="none" keeps SVG text as text.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

import importlib.util  # noqa: E402
import os  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Anchored on this file so the inputs, the renderer and the output all resolve
# inside this repository, whatever the working directory.
ROOT = Path(__file__).resolve().parents[2]
EC = str(ROOT / "effective_connectivity")
# PSY_ROOT redirects the output tree, matching 00_theme_psychodel.R.
OUTDIR = str(Path(os.environ.get("PSY_ROOT", ROOT)) / "figures" / "figures")

# The ridgeline renderer this script supplies data to.
RENDERER = str(ROOT / "effective_connectivity" / "scripts" / "visualizations.py")

N_PERM = os.environ.get("N_PERM", "10000")
RAW_LABELS = os.environ.get("RAW_LABELS", "") not in ("", "0")
PRETTY_FACTORS = os.environ.get("PRETTY_FACTORS", "1") not in ("", "0")
DPI = int(os.environ.get("DPI", "400"))

# The renderer prints each factor's column name as its block header, so the only
# way to get reader-facing names is to rename the columns on the way in.
FACTOR_NAMES = {"network_relation": "network relation",
                "bands": "frequency band",
                "eyes": "eyes condition"}


def load_renderer(path):
    """Import the renderer module by absolute path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("visualizations_new", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tidy_labels(df):
    """Manuscript naming: anonymised datasets, hyphenated eyes conditions."""
    df = df.copy()
    if "eyes" in df.columns:
        df["eyes"] = df["eyes"].replace({"eyes open": "eyes-open",
                                         "eyes closed": "eyes-closed"})
    if "city" in df.columns and not RAW_LABELS:
        df["city"] = (df["city"].astype(str)
                      .str.replace("Kraków", "I", regex=False)
                      .str.replace("Warszawa", "II", regex=False)
                      .str.replace("merged(I,II)", "merged (I, II)", regex=False))
        df = df.rename(columns={"city": "dataset"})
    if PRETTY_FACTORS:
        df = df.rename(columns={k: v for k, v in FACTOR_NAMES.items()
                                if k in df.columns})
    return df


def render(plot_fn, name, stem, title, note):
    src = f"{EC}/spec_cache_{name}_{N_PERM}.csv"
    if not os.path.exists(src):
        raise SystemExit(
            f"missing {src}\nrun: python3 07_spec_curve_extract.py {name} {N_PERM}")

    table = tidy_labels(pd.read_csv(src)).reset_index(drop=True)
    spec_df = table.drop(columns=["contrast"])
    results_df = table[["contrast"]]

    # With output_file=None the renderer ends on plt.show(), which is a no-op
    # under Agg but warns. We want the figure object back so we can write all
    # three formats, so swallow just that warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*non-interactive.*")
        fig = plot_fn(spec_df, results_df, output_file=None, title=title)
    if fig is None:
        raise SystemExit(f"renderer returned no figure for {name}")

    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=DPI if ext == "png" else None,
                    bbox_inches="tight")

    c = table["contrast"].to_numpy(dtype=float)
    print(f"\n{note}")
    print(f"  n = {len(c):,} | mean = {c.mean():.4f} | median = {np.median(c):.4f}"
          f" | sd = {c.std(ddof=1):.4f}")
    print(f"  factors: {[x for x in spec_df.columns if x != 'node_pair']}")
    ax2 = fig.axes[1]
    print(f"  x ticks: {[t.get_text() for t in ax2.get_xticklabels()]}")
    print(f"  figure size: {fig.get_size_inches()[0]:.1f} x "
          f"{fig.get_size_inches()[1]:.1f} in")
    for ext in ("png", "pdf", "svg"):
        p = f"{stem}.{ext}"
        print(f"  wrote {os.path.relpath(p, ROOT)} "
              f"({os.path.getsize(p) / 1e6:.2f} MB)")


def main():
    renderer = RENDERER
    if not os.path.exists(renderer):
        raise SystemExit(f"renderer not found at {renderer}")
    os.makedirs(OUTDIR, exist_ok=True)
    mod = load_renderer(renderer)
    if not hasattr(mod, "_ridgeline"):
        raise SystemExit(
            f"{renderer} has no _ridgeline: this looks like the retired barcode "
            "renderer, not the author's ridgeline version")
    plot_fn = mod.plot_specification_curve
    print(f"renderer: {os.path.relpath(renderer, ROOT)}")

    variants = [
        ("unmerged", f"{OUTDIR}/figure4_spec_curve",
         "Specification Curve Analysis",
         "MAIN FIGURE 4 -- unmerged; matches the 9,240 comparisons in Results"),
        ("merged", f"{OUTDIR}/figure4_spec_curve_merged",
         "Specification Curve Analysis",
         "COMPANION -- merged; 4,620 comparisons"),
    ]
    for name, stem, title, note in variants:
        render(plot_fn, name, stem, title, note)


if __name__ == "__main__":
    main()
