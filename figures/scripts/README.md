# figures/scripts

One script per manuscript figure. All paths resolve relative to the repository,
so the scripts run from any working directory:

```sh
Rscript figures/scripts/01_lz.R          # or from inside figures/scripts/
python3 figures/scripts/05_figure1_rois.py
```

Output goes to `figures/figures/`. Set the `PSY_ROOT` environment variable to
write somewhere else instead.

## Scripts

| script | figure | inputs |
|---|---|---|
| `00_theme_psychodel.R` | shared ggplot theme, palette and `save_fig()`; sourced by the other R scripts | — |
| `01_lz.R` | Figure 3, Lempel-Ziv | `LZ_Lempel-Ziv_complexity/data/complexity_data.csv`, `beh/beh_data_merged.csv` |
| `02_psd.R` | Figure 2, PSD panels | `PSD_static_spectral/output/dataframes/PSD_trials_sep/`, `beh/beh_data_merged.csv` |
| `03_violins.R` | Figures S2-S9, behavioural violins | `beh/beh_data_merged.csv` |
| `05_figure1_rois.py` | Figure 1, ROIs | none (coordinates are literals); needs `nilearn` |
| `06_figureS10_pvalues.py` | Figure S10, permutation P value ECDFs | the `.cdb` Study caches — **not distributed**, see below |
| `07_spec_curve_extract.py` | (step 1 of Figure 4) | a `.cdb` Study cache — **not distributed** |
| `07_spec_curve_ridgeline.py` | Figure 4, specification curve | `effective_connectivity/spec_cache_<variant>_10000.csv` |
| `figure1_consort.mmd` | Figure S1, CONSORT (mermaid source; see the render command below) | — |

## Figure S1, the CONSORT diagram

Rendered with [mermaid-cli](https://github.com/mermaid-js/mermaid-cli). The PNG is
written at scale 3 to match the other figures' resolution:

```sh
npx @mermaid-js/mermaid-cli -i figure1_consort.mmd -o ../figures/figureS1_consort.png -b transparent -s 3
npx @mermaid-js/mermaid-cli -i figure1_consort.mmd -o ../figures/figureS1_consort.svg -b transparent
npx @mermaid-js/mermaid-cli -i figure1_consort.mmd -o ../figures/figureS1_consort.pdf -b transparent
```

## Figure 4, the specification curve: two steps

| order | script | what it does |
|---|---|---|
| 1 (rarely) | `07_spec_curve_extract.py` | reads a `.cdb` study cache, writes `effective_connectivity/spec_cache_<variant>_10000.csv` |
| 2 (each render) | `07_spec_curve_ridgeline.py` | renders `figure4_spec_curve{,_merged}.{png,pdf,svg}` |

```sh
python3 07_spec_curve_extract.py unmerged 10000   # only if the CSV is missing
python3 07_spec_curve_extract.py merged   10000
python3 07_spec_curve_ridgeline.py
```

Both CSVs are committed, so **step 1 is normally unnecessary** — step 2 runs on a
clean clone.

The split exists because the `.cdb` caches are 3.4 GB of pickle and peak around
21 GB of memory to load; the extracted CSV is under 1 MB and makes a re-render
take seconds. The contrasts the figure needs are computed at import time and
never touched by `permute()`, so the extract only has to be redone when the
underlying `.xlsx` data change, not when the permutation count does.

`07_spec_curve_ridgeline.py` does **no plotting of its own**. It imports
`plot_specification_curve` from `effective_connectivity/scripts/visualizations.py`
and only supplies data, manuscript labels and multi-format saving.

**Which variant is the manuscript figure:** `figure4_spec_curve` is the
*unmerged* one (datasets I and II kept separate, 9,240 specifications, mean
0.0229). That is what Results reports. `figure4_spec_curve_merged` (4,620,
mean 0.0216) is a companion, not the headline figure.

## What is not distributed here

`06_figureS10_pvalues.py` and `07_spec_curve_extract.py` need the
`study_{merged,unmerged}_10000.cdb` Study caches, which are ~3.4 GB and are not
committed (`*.cdb` is gitignored). Rebuild them with:

```sh
python3 effective_connectivity/scripts/data_preparation.py   # slow: 10,000 permutations
```

or point `EC_CACHE` at a directory that already holds them. Both scripts fail
with an explicit message rather than a stack trace when the caches are absent.

## A note on Figure 2 and multiple comparisons

The significance stars in `02_psd.R`, and the decision to draw a band's
reactivity panel, use the **uncorrected** p-value. Tables 1-2 of the manuscript
report BH-adjusted q-values. A band can therefore carry a star in Figure 2 while
the tables report it as non-significant. This is intentional — the figure reports
the uncorrected test — but the tables are the reference for inference.
