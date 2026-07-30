#!/usr/bin/env python3
"""Figure S10: permutation-based P value distributions across the four regression models.

The ECDF is exact at every k. The diagonal is the uniform null, the shaded band is the
two-sided Kolmogorov-Smirnov 95% acceptance region, and the rug shows each coefficient.
A curve bowing above the diagonal near zero means an excess of small P values.
"""
import matplotlib
matplotlib.use("Agg")
import sys, os, warnings, io, contextlib, json
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
EC = REPO / "effective_connectivity"
CACHE = os.environ.get("EC_CACHE", str(EC))

# statistical_model.py lives in effective_connectivity/scripts/.
sys.path.insert(0, str(EC / "scripts"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statistical_model import Study

FREQ = ["delta", "theta", "alpha", "beta", "gamma"]
N_PERM = int(os.environ.get("N_PERM", "10000"))
ALPHA = float(os.environ.get("ALPHA", "0.05"))
FORCE = os.environ.get("FORCE", "") not in ("", "0")
BAND = 0.95  # coverage of the KS acceptance band
PVAL_CACHE = f"{CACHE}/figS10_pvalues_{N_PERM}.json"

COL_ECDF = "#4a3f8f"
COL_NULL = "#2e7d32"
COL_BAND = "#2e7d32"
COL_ALPHA = "#c62828"


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k)


def load_study(name):
    """Load the cache matching N_PERM if it exists, else build it from scratch.
    """
    want = f"{CACHE}/study_{name}_{N_PERM}.cdb"
    if os.path.exists(want):
        st = quiet(Study.load, want)
        assert len(st.permuted_results) == N_PERM, \
            f"{want} holds {len(st.permuted_results)} permutations, expected {N_PERM}"
        print(f"loaded {os.path.basename(want)} ({N_PERM} permutations)", flush=True)
        return st

    print(f"no cache for {name} at {N_PERM}; permuting FRESH (this is slow)...", flush=True)
    seed_cache = f"{CACHE}/study_{name}_1000.cdb"
    st = quiet(Study.load, seed_cache) if os.path.exists(seed_cache) else None
    if st is None:
        raise SystemExit(
            f"No Study cache for '{name}' under {CACHE}.\n"
            f"  looked for: {os.path.basename(want)} or {os.path.basename(seed_cache)}\n"
            "The .cdb caches are ~3.4 GB and are not distributed with this repository.\n"
            "Build them first with:  python3 effective_connectivity/scripts/data_preparation.py\n"
            "or point EC_CACHE at the directory that holds them.")
    quiet(st.permute, n_permutations=N_PERM)
    assert len(st.permuted_results) == N_PERM, \
        f"{name}: got {len(st.permuted_results)}, expected {N_PERM}"
    quiet(st.save, want)
    print(f"saved {want} ({len(st.permuted_results)} permutations)", flush=True)
    return st


MODEL_SPECS = [
    ("merged",   "~ network_relation -1",                  "Model 1"),
    ("merged",   "~ network_relation:bands -1",            "Model 2"),
    ("merged",   "~ city + eyes + network_relation:bands", "Model 3"),
    ("unmerged", "~ city + eyes + network_relation:bands", "Model 4"),
]


def extract_pvalues():
    """Return {title: [P values]}, from the JSON cache when available."""
    if os.path.exists(PVAL_CACHE) and not FORCE:
        with open(PVAL_CACHE) as fh:
            blob = json.load(fh)
        assert blob["n_perm"] == N_PERM, \
            f"{PVAL_CACHE} holds n_perm={blob['n_perm']}, expected {N_PERM}"
        print(f"loaded P values from {os.path.basename(PVAL_CACHE)}", flush=True)
        return {t: blob["models"][t]["p"] for _, _, t in MODEL_SPECS}

    studies = {name: load_study(name) for name in ("merged", "unmerged")}
    blob = {"n_perm": N_PERM, "models": {}}
    for name, formula, title in MODEL_SPECS:
        res = quiet(studies[name].regression, formula, add_network_categories=True,
                    n_permutations=N_PERM, band_order=FREQ)
        # Parameter is a column of param_estimates, not its index.
        pe = res[1]
        keep = pe["p-value (perm)"].notna()
        vals = pe.loc[keep, "p-value (perm)"]
        blob["models"][title] = {
            "formula": formula,
            "study": name,
            "terms": [str(t) for t in pe.loc[keep, "Parameter"]],
            "p": [float(v) for v in vals.values],
        }
        print(f"  extracted {title}: k={len(vals)}", flush=True)
    with open(PVAL_CACHE, "w") as fh:
        json.dump(blob, fh, indent=1)
    print(f"wrote {PVAL_CACHE}", flush=True)
    return {t: blob["models"][t]["p"] for _, _, t in MODEL_SPECS}


print(f"N_PERM={N_PERM}  ALPHA={ALPHA}")
pvalues = extract_pvalues()

fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), sharex=True, sharey=True)

for ax, (_, formula, title) in zip(axes.flat, MODEL_SPECS):
    p = np.sort(np.asarray(pvalues[title], dtype=float))
    k = len(p)

    # Two-sided KS acceptance band. kstwo is the exact null distribution of D at
    # sample size k; the asymptotic sqrt(-ln(a/2)/2k) form is unreliable at k = 9.
    d_crit = stats.kstwo.ppf(BAND, k)
    ks = stats.kstest(p, "uniform")

    grid = np.linspace(0, 1, 400)
    ax.fill_between(grid, np.clip(grid - d_crit, 0, 1), np.clip(grid + d_crit, 0, 1),
                    color=COL_BAND, alpha=0.12, lw=0,
                    label=f"{BAND:.0%} KS band" if title == "Model 1" else None)
    ax.plot([0, 1], [0, 1], color=COL_NULL, lw=1.6,
            label="Uniform null" if title == "Model 1" else None)

    # ECDF as a step function, closed at both ends so it spans the full axis.
    ax.step(np.concatenate([[0], p, [1]]),
            np.concatenate([[0], np.arange(1, k + 1) / k, [1]]),
            where="post", color=COL_ECDF, lw=1.8,
            label="Observed ECDF" if title == "Model 1" else None)

    # Rug: every individual coefficient, so nothing is hidden by the step function.
    # Drawn just INSIDE the bottom of the axes. Placing it below the axis with
    # clip_on=False puts it exactly where the x tick labels sit, and the two collide
    # on the bottom row of panels.
    ax.plot(p, np.full(k, 0.016), "|", color=COL_ECDF, ms=6, mew=1.0, alpha=0.8)

    ax.axvline(ALPHA, color=COL_ALPHA, ls="--", lw=1.4,
               label=f"$\\alpha$ = {ALPHA:g}" if title == "Model 1" else None)

    n_sig = int((p < ALPHA).sum())
    ax.text(0.97, 0.06,
            f"$k$ = {k}\n$D$ = {ks.statistic:.3f}, $P$ = {ks.pvalue:.2f}\n"
            f"$P$ < {ALPHA:g}: {n_sig} obs / {k * ALPHA:.1f} exp",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", lw=0.6))

    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.22, lw=0.6)
    ax.set_aspect("equal", adjustable="box")

    print(f"  {title}: k={k}, {n_sig} below alpha (exp {k * ALPHA:.1f}), "
          f"KS D={ks.statistic:.3f} P={ks.pvalue:.3f}, min P={p.min():.4f}", flush=True)

for ax in axes[-1]:
    ax.set_xlabel("P value (permutation-based)")
for ax in axes[:, 0]:
    ax.set_ylabel("Cumulative proportion of coefficients")

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.045, 1, 1))

# PSY_ROOT redirects the output tree, matching 00_theme_psychodel.R.
FIG_OUT = Path(os.environ.get("PSY_ROOT", REPO)) / "figures" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)
OUT = str(FIG_OUT / "figureS10_permutation_pvalues")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
print("wrote figureS10_permutation_pvalues.{pdf,png}")
