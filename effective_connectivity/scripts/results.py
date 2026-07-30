"""
This script performs data loading, exploration, regression analysis, for effective connectivity in paper
"""

from statistical_model import Study
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Caches are written next to the data by data_preparation.py; anchored on this
# file so the script runs from any working directory.
EC_ROOT = Path(__file__).resolve().parent.parent

# Define frequency band order (from data_preparation.py)
FREQUENCY_BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]


def plot_pvalue_histogram(ax, results, title):
    """Plot p-value histogram on given axes."""
    n_bins = 40
    
    if results[1] is None or 'p-value (perm)' not in results[1].columns:
        ax.text(0.5, 0.5, 'No p-values available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return 0, 0
    
    p_values = results[1]['p-value (perm)'].dropna()
    n_pvalues = len(p_values)
    expected_freq = 1.0 / n_bins
    
    ax.hist(p_values, bins=n_bins, edgecolor='black', alpha=0.7,
            density=False, weights=np.ones(n_pvalues) / n_pvalues)
    ax.axhline(y=expected_freq, color='green', linestyle='-', linewidth=2,
               label=f'Expected under null = {expected_freq:.3f}')
    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax.set_xlabel('p-value (permutation-based)')
    ax.set_ylabel('Relative Frequency')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    return n_pvalues, np.sum(p_values < 0.05)


# =============================================================================
# 1. Loading the Data
# =============================================================================
print("=" * 80)
print("1. Loading the Data")
print("=" * 80)

study = Study.load(str(EC_ROOT / "study_merged_10000.cdb"))
study2 = Study.load(str(EC_ROOT / "study_unmerged_10000.cdb"))

# =============================================================================
# 2. Exploring the Data
# =============================================================================
print("\n" + "=" * 80)
print("2. Exploring the Data")
print("=" * 80)

print("\nMerged study summary:")
study.summary()

print("\nUnmerged study summary:")
study2.summary()

# =============================================================================
# 3. Regression Analysis
# =============================================================================
print("\n" + "=" * 80)
print("3. Regression Analysis")
print("=" * 80)

results_network_only = study.regression(
    "~ network_relation -1",
    add_network_categories=True,
    n_permutations=10000,
    band_order=FREQUENCY_BAND_ORDER
)
print(study.print_apa_format(results_network_only))

# =============================================================================
# 4. Regression Analysis: interactions
# =============================================================================
print("\n" + "=" * 80)
print("4. Regression Analysis: interactions")
print("=" * 80)

results_network_bands = study.regression(
    "~ network_relation:bands -1",
    add_network_categories=True,
    n_permutations=10000,
    band_order=FREQUENCY_BAND_ORDER
)
print(study.print_apa_format(results_network_bands))

# =============================================================================
# 5. Regression Analysis: full specification with interactions
# =============================================================================
print("\n" + "=" * 80)
print("5. Regression Analysis: full specification with interactions")
print("=" * 80)

results_full_merged = study.regression(
    "~ city + eyes +network_relation:bands",
    add_network_categories=True,
    n_permutations=10000,
    band_order=FREQUENCY_BAND_ORDER
)

results_full_unmerged = study2.regression(
    "~ city + eyes +network_relation:bands",
    add_network_categories=True,
    n_permutations=10000,
    band_order=FREQUENCY_BAND_ORDER
)

print(study.print_apa_format(results_full_merged))
print(study2.print_apa_format(results_full_unmerged))

# =============================================================================
# 6. Specification Curve Analysis
# =============================================================================
print("\n" + "=" * 80)
print("6. Specification Curve Analysis")
print("=" * 80)

print("Specification curve for model1 with merged data...")
spec_df, results_df = study.specification_curve(add_network_categories=True, plot_type='density')

print("Specification curve for model2 with unmerged data...")
spec_df, results_df = study2.specification_curve(add_network_categories=True, plot_type='density')

# =============================================================================
# 7. P-Value Histograms
# =============================================================================
print("\n" + "=" * 80)
print("7. P-Value Histograms")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models = [
    (results_network_only, "Model 1"),
    (results_network_bands, "Model 2"),
    (results_full_merged, "Model 3"),
    (results_full_unmerged, "Model 4"),
]

for ax, (results, title) in zip(axes.flat, models):
    n_pvalues, n_sig = plot_pvalue_histogram(ax, results, title)
    if n_pvalues > 0:
        print(f"{title}: {n_pvalues} parameters, {n_sig} significant at α=0.05")

plt.tight_layout()
plt.show()