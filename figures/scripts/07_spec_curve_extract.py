#!/usr/bin/env python3
"""Extract the specification table from a cached connectivity Study (.cdb).

The specification curve needs one number per measurement -- `between_group_contrast`
-- plus the measurement's condition labels. 

The caches are pickles of the whole Study object (1.1 GB merged, 2.3 GB unmerged),
so they cannot be read partially. This script loads exactly ONE study per process
and writes a ~500 kB CSV, which keeps peak RSS bounded and means the plotting
script never has to touch the pickles again.

Usage:
    python3 07_spec_curve_extract.py merged   [n_perm]
    python3 07_spec_curve_extract.py unmerged [n_perm]

Writes: effective_connectivity/spec_cache_<name>_<n_perm>.csv
"""
import contextlib
import io
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EC = os.environ.get("EC_CACHE", str(REPO / "effective_connectivity"))

# statistical_model.py lives in effective_connectivity/scripts/.
sys.path.insert(0, str(REPO / "effective_connectivity" / "scripts"))

import numpy as np  # noqa: E402


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "merged"
    n_perm = sys.argv[2] if len(sys.argv) > 2 else "10000"
    assert name in ("merged", "unmerged"), name

    src = os.path.join(EC, f"study_{name}_{n_perm}.cdb")
    if not os.path.exists(src):
        sys.exit(f"no cache at {src}")

    from statistical_model import Study  # noqa: E402  (scripts/ added to sys.path above)

    size_gb = os.path.getsize(src) / 1e9
    print(f"loading {os.path.basename(src)} ({size_gb:.2f} GB on disk)...", flush=True)
    study = quiet(Study.load, src)
    if study is None:
        sys.exit(f"failed to load {src}")

    n_perm_actual = len(getattr(study, "permuted_results", []) or [])
    print(f"  loaded: {len(study.data)} measurements, "
          f"{n_perm_actual} permutations in cache", flush=True)

    spec_df, results_df = study.specification_data(add_network_categories=True)
    out = spec_df.copy()
    out["contrast"] = results_df["contrast"].to_numpy()

    # Free the pickle before writing, so a slow disk write does not hold 10+ GB.
    del study

    dest = os.path.join(EC, f"spec_cache_{name}_{n_perm}.csv")
    out.to_csv(dest, index=False, encoding="utf-8")

    c = out["contrast"].to_numpy(dtype=float)
    print(f"  rows={len(out)}  mean={c.mean():.4f}  median={np.median(c):.4f}  "
          f"sd={c.std(ddof=1):.4f}  min={c.min():.4f}  max={c.max():.4f}", flush=True)
    print(f"  condition columns: {[x for x in spec_df.columns if x != 'node_pair']}",
          flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
