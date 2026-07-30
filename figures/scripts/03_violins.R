# =============================================================================
# 03_violins.R  --  Figures S1-S8: questionnaire /substance use split violins
#
# Multi_split_violin_plot to ggplot: split violin (psy-exp left / psy-naiv right) + narrow boxplot + jittered points + Welch
# t-test p + per-group n, now with a visible colour legend. Recoloured to the
# psychodel palette; group 1 -> psy-exp, 9 -> psy-naiv.
#
# =============================================================================

# Locate this script so every path below resolves inside the repository,
HERE <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) return(dirname(normalizePath(sub("^--file=", "", m[1]))))
  for (i in seq_len(sys.nframe())) {
    ofile <- sys.frame(i)$ofile
    if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  }
  normalizePath(getwd())
})()
REPO <- normalizePath(file.path(HERE, "..", ".."))

source(file.path(HERE, "00_theme_psychodel.R"))

beh <- file.path(REPO, "beh")

# beh_data_merged.csv is the single behavioural table for both sites: one row
# per participant, one set of column names, and the questionnaire scores that
# beh_krk.csv and beh_wwa.csv hold per site. Splitting it by city is therefore
# all the "join" these figures need.
BEH <- read.csv(file.path(beh, "beh_data_merged.csv"), stringsAsFactors = FALSE)
BEH$.grp <- fct_group(BEH$group)

# Panels show questionnaire completers, which is fewer than the analysis
# sample: participants with EEG but no questionnaire carry blank scores and
# drop out panel by panel. Per-panel n is printed in each caption.
split_site <- function(city) {
  d <- BEH[BEH$city == city, ]
  # dense-ranked frequency columns, computed within the site
  d$cannabis_rank   <- dplyr::dense_rank(suppressWarnings(as.numeric(d$cannabis_lt)))
  d$meditation_rank <- dplyr::dense_rank(suppressWarnings(as.numeric(d$meditation_h)))
  d
}
krk <- split_site("krk")
wwa <- split_site("wwa")

cat(sprintf("\n-- Behavioural sample: Cracow %d, Warsaw %d (of %d in beh_data_merged.csv) --\n",
            nrow(krk), nrow(wwa), nrow(BEH)))

## canonical key -> (column name per dataset, panel title)
# panel titles match the canonical row labels in appendix Table S1c/S1d verbatim
titles <- c(
  stai_i="STAI-I (state)", stai_ii="STAI-II (trait)", bdi_ii="BDI-II",
  rrq_reflection="RRQ-Reflection", rrq_rumination="RRQ-Rumination",
  cannabis_rank="Lifetime Cannabis Use (ranked)", meditation_rank="Lifetime Meditation Hours (ranked)",
  arsq_discont="ARSQ Discontinuity of mind", arsq_tom="ARSQ Theory of mind", arsq_self="ARSQ Self",
  arsq_plan="ARSQ Planning", arsq_sleep="ARSQ Sleepiness", arsq_comf="ARSQ Comfort",
  arsq_somat="ARSQ Somatic awareness", arsq_health="ARSQ Health concern",
  arsq_visual="ARSQ Visual thought", arsq_verbal="ARSQ Verbal thought")

## canonical key -> column in beh_data_merged.csv (both sites share one schema)
cols <- c(stai_i="stai_i", stai_ii="stai_ii", bdi_ii="bdi_ii",
  rrq_reflection="rrq_reflection", rrq_rumination="rrq_rumination",
  cannabis_rank="cannabis_rank", meditation_rank="meditation_rank",
  arsq_discont="arsq_discontinuity_of_mind", arsq_tom="arsq_theory_of_mind",
  arsq_self="arsq_self", arsq_plan="arsq_planning", arsq_sleep="arsq_sleepiness",
  arsq_comf="arsq_comfort", arsq_somat="arsq_somatic_awareness",
  arsq_health="arsq_health_concern", arsq_visual="arsq_visual_thought",
  arsq_verbal="arsq_verbal_thought")

## --- one split-violin panel -------------------------------------------------
violin_panel <- function(df, key) {
  d <- data.frame(g = df$.grp, y = suppressWarnings(as.numeric(df[[cols[[key]]]])))
  d <- d[!is.na(d$g) & !is.na(d$y), ]
  n  <- table(d$g)
  pv <- tryCatch(t.test(y ~ g, data = d)$p.value, error = function(e) NA_real_)
  plab <- if (is.na(pv)) "" else if (pv < .001) "p < .001" else sprintf("p = %.3f", pv)
  yr <- range(d$y); pad <- diff(yr) * 0.12; if (pad == 0) pad <- 1
  ggplot(d, aes(x = "", y = y, fill = g)) +
    geom_split_violin(scale = "count", trim = TRUE, colour = "grey30",
                      alpha = 0.9, linewidth = 0.3) +
    geom_boxplot(aes(colour = g), width = 0.12, position = position_dodge(0.28),
                 outlier.shape = NA, fill = "white", alpha = 0.5,
                 linewidth = 0.45, show.legend = FALSE) +
    geom_point(aes(colour = g), position = position_jitterdodge(
                 jitter.width = 0.10, dodge.width = 0.30, seed = 1),
               size = 0.9, alpha = 0.6, show.legend = FALSE) +
    scale_fill_psy() + scale_colour_psy() +
    annotate("text", x = 1, y = yr[2] + pad, label = plab, size = 3, colour = "grey30") +
    labs(title = titles[[key]], x = NULL, y = NULL,
         caption = sprintf("psy-exp n = %d      psy-naiv n = %d",
                           n[["psy-exp"]], n[["psy-naiv"]])) +
    coord_cartesian(ylim = c(yr[1] - pad, yr[2] + 2 * pad)) +
    theme_psychodel() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          plot.title = element_text(size = 12, hjust = 0.5))
}

## --- assemble a figure from several keys ------------------------------------
make_fig <- function(df, keys, name, ncol, w, h, title) {
  panels <- lapply(keys, function(k) violin_panel(df, k))
  fig <- patchwork::wrap_plots(panels, ncol = ncol) +
    patchwork::plot_layout(guides = "collect") +
    patchwork::plot_annotation(title = title) &
    theme(legend.position = "bottom", plot.title = element_text(face = "bold"))
  save_fig(fig, name, w = w, h = h)
}
suppressPackageStartupMessages(library(patchwork))

anx  <- c("stai_i","stai_ii","bdi_ii")
rrq  <- c("rrq_reflection","rrq_rumination")
rank <- c("cannabis_rank","meditation_rank")
arsq <- c("arsq_discont","arsq_tom","arsq_self","arsq_plan","arsq_sleep",
          "arsq_comf","arsq_somat","arsq_health","arsq_visual","arsq_verbal")

make_fig(krk, anx,  "figureS2_krk_anxiety_depression", 3, 12, 5.2, "Dataset I: STAI and BDI-II")
make_fig(krk, rrq,  "figureS3_krk_rrq",                2,  8, 5.2, "Dataset I: RRQ")
make_fig(krk, rank, "figureS4_krk_ranked",             2,  8, 5.2, "Dataset I: Ranked cannabis and meditation")
make_fig(krk, arsq, "figureS5_krk_arsq",               5, 18, 9.5, "Dataset I: ARSQ subscales")
make_fig(wwa, anx,  "figureS6_wwa_anxiety_depression", 3, 12, 5.2, "Dataset II: STAI and BDI-II")
make_fig(wwa, rrq,  "figureS7_wwa_rrq",                2,  8, 5.2, "Dataset II: RRQ")
make_fig(wwa, rank, "figureS8_wwa_ranked",             2,  8, 5.2, "Dataset II: Ranked cannabis and meditation")
make_fig(wwa, arsq, "figureS9_wwa_arsq",               5, 18, 9.5, "Dataset II: ARSQ subscales")
