from statistical_model import Study
from helpers_functions import *
import os
import glob
from pathlib import Path
from tqdm import tqdm


def extract_conditions_from_path(file_path):
    """Extract measurement conditions from a data file path.
    Expected layout: data/DTF_<CITY>/<eyes-open|eyes-closed>/GLOBAL_..._<band>_....xlsx
    """
    path = Path(file_path)

    # Extract city. The spelling must match the committed caches and the label
    # substitution in figures/scripts/07_spec_curve_ridgeline.py.
    if "DTF_WAW" in str(path):
        city = "Warszawa"
    elif "DTF_KRK" in str(path):
        city = "Kraków"
    else:
        raise ValueError(f"Cannot determine city from path: {file_path}")

    # Extract eye condition. "ec"/"eo" are accepted so older data trees import.
    dirs = {part.lower() for part in path.parts}
    if dirs & {"eyes-closed", "ec"}:
        eyes = "eyes closed"
    elif dirs & {"eyes-open", "eo"}:
        eyes = "eyes open"
    else:
        raise ValueError(f"Cannot determine eye condition from path: {file_path}")

    # Extract frequency band
    frequency_band = {
        "delta": "delta",
        "theta": "theta",
        "alpha": "alpha",
        "beta" : "beta" ,
        "gamma": "gamma",
    }

    band = None
    for key, value in frequency_band.items():
        if key in path.name.lower():
            band = value
            break
    if band is None:
        raise ValueError(f"Cannot determine frequency band from path: {file_path}")

    return {
        "city": city,
        "eyes": eyes,
        "bands": band
    }

node_labels = {
    # Default Mode Network (DMN) nodes
    'L-mPFC': 'DMN',
    'R-mPFC': 'DMN',
    'L-IFG': 'DMN',
    'L-PREC': 'DMN',
    'R-PREC': 'DMN',
    'L-ANG': 'DMN',
    'R-ANG': 'DMN',
    'L-aSTG': 'DMN',
    'R-aSTG': 'DMN',
    
    # Salience Network (SN) nodes
    'L-ACC': 'SN',
    'R-ACC': 'SN',
    'L-INS': 'SN',
    'R-INS': 'SN',
    
    # Central Executive Network (CEN) nodes
    'R-IFG': 'CEN',
    'L-SFG': 'CEN',
    'R-SFG': 'CEN',
    'L-pSTG': 'CEN',
    'R-pSTG': 'CEN',
    'L-ITG': 'CEN',
    'R-ITG': 'CEN',
    'L-SUP': 'CEN',
    'R-SUP': 'CEN'
}


# Anchored on this file, not the working directory, so the script runs from anywhere.
EC_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = EC_ROOT / "data"


def build_studies(base_dir=BASE_DIR, out_dir=EC_ROOT, n_permutations=10000, seed=20240917):
    """Import the 20 DTF measurement tables and write the two study caches.

    Runs 2 x n_permutations permutations and writes ~3.4 GB, so it is guarded by
    __main__ below: importing this module must stay cheap and side-effect free.

    The permutation draw is seeded so that rebuilding the caches reproduces the
    same p-values. Earlier caches were built unseeded, so numbers regenerated
    from this point differ from those in the manuscript by the Monte Carlo error
    of 10,000 permutations -- of order 0.005, third decimal place, changing no
    term's significance.
    """
    study = Study(nodes=node_labels, control_group_name=9)
    # sorted() because glob order is filesystem-dependent and import order must be stable.
    excel_files = sorted(glob.glob(os.path.join(str(base_dir), "**/*.xlsx"), recursive=True))

    if not excel_files:
        raise FileNotFoundError(f"No .xlsx measurements found under {base_dir}")

    for file_path in tqdm(excel_files):
        conditions = extract_conditions_from_path(file_path)
        study.import_measurement_from_excel(file_path,
                                            measurement_conditions=conditions,
                                            independent_samples=["city"])

    study.summary()

    study2 = study.merge_independent_condition(["city"])
    study2.summary()
    study2.permute(n_permutations=n_permutations, seed=seed)
    study2.save(str(Path(out_dir) / f"study_merged_{n_permutations}.cdb"))

    study.permute(n_permutations=n_permutations, seed=seed)
    study.save(str(Path(out_dir) / f"study_unmerged_{n_permutations}.cdb"))
    return study, study2


if __name__ == "__main__":
    build_studies()

