# =============================================================================
# 02_psd.R  --  Figure 2: mean PSD by band, condition, and group
#
# Bars = model-predicted means from lmer(band ~ cond*group + (1|pid))
#     with vcov-based SE (the random intercept handles pseudoreplication);
# Significance from uncorrected post-hoc contrasts (group|condition for
#     the left panel; the condition x group interaction, i.e. the eyes-open
#     minus eyes-closed group difference, for the right panel).
# Recoloured to the psychodel palette; eo/ec -> eyes-open/eyes-closed;
# e/c -> psy-exp/psy-naiv; visible legend.
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
suppressPackageStartupMessages({ library(lmerTest); library(emmeans); library(patchwork) })

# One wide file per site: participant, city, group,
# condition, trial_number, delta..gamma).
PSD_DIR <- file.path(REPO, "PSD_static_spectral", "output", "dataframes", "PSD_trials_sep")
FINAL   <- read.csv(file.path(REPO, "beh", "beh_data_merged.csv"), stringsAsFactors = FALSE)
bands_in <- c("delta", "theta", "alpha", "beta", "gamma")
BP <- NULL
for (bb in bands_in) {
  d <- read.csv(file.path(PSD_DIR, sprintf("_%s_PSD_trials_separately.csv", bb)),
                stringsAsFactors = FALSE)
  d <- d[, c("participant_id", "city", "condition", "trial_number", bb)]
  BP <- if (is.null(BP)) d else merge(BP, d, by = c("participant_id", "city", "condition", "trial_number"))
}
BP <- merge(BP, FINAL[, c("participant_id", "group")], by = "participant_id")
BP$participant <- BP$participant_id

## Sample actually entering the PSD models. Smaller than the N = 113 flow
## diagram: two participants (Cracow 121, Warsaw 26) have no spectra.
psd_ids <- unique(BP$participant_id)
cat(sprintf("\n-- PSD sample: N = %d --\n", length(psd_ids)))
print(table(city  = FINAL$city[match(psd_ids, FINAL$participant_id)],
            group = fct_group(FINAL$group[match(psd_ids, FINAL$participant_id)])))
bands <- c("delta", "theta", "alpha", "beta", "gamma")
ylab_power <- expression("Power [dB " * mu * V^2 * "/Hz]")

build_band <- function(band) {
  df <- BP
  df$y     <- df[[band]]
  df$group <- fct_group(df$group)
  df$cond  <- fct_cond(df$condition)
  df$pid   <- factor(df$participant)
  df <- df[!is.na(df$y) & !is.na(df$group) & !is.na(df$cond), ]

  ## --- stats model (with city): condition RANDOM SLOPE corrects the
  ##     pseudoreplication; contrasts reported as raw emmeans z-tests.
  ##
  ##     NOTE ON MULTIPLICITY: the stars drawn below and the decision to show a
  ##     reactivity panel both use the UNCORRECTED p-value. Tables 1-2 of the
  ##     manuscript report BH-adjusted q-values, so a band can carry a star here
  ##     while the tables report it as non-significant. This is deliberate -
  ##     the figure reports the uncorrected test - but read the tables for
  ##     inference.
  ms   <- lmerTest::lmer(y ~ cond * group + city + (cond | pid), data = df)
  con2 <- as.data.frame(emmeans(ms, pairwise ~ group | cond, lmer.df = "asymptotic")$contrasts)
  con3 <- as.data.frame(contrast(emmeans(ms, ~ cond:group, lmer.df = "asymptotic"),
                                 interaction = "pairwise"))

  ## --- prediction model (without city) + vcov SE (random slope) -----------
  mp   <- lmerTest::lmer(y ~ cond * group + (cond | pid), data = df)
  grid <- expand.grid(cond = factor(cond_levels, cond_levels),
                      group = factor(group_levels, group_levels))
  grid$y   <- predict(mp, grid, re.form = NA)
  mm       <- model.matrix(terms(mp), grid)
  pvar     <- diag(mm %*% tcrossprod(vcov(mp), mm))
  grid$plo <- grid$y - sqrt(pvar)
  grid$phi <- grid$y + sqrt(pvar)

  ## --- eyes-open minus eyes-closed difference per group, propagated SE -----
  eo <- grid$cond == "eyes-open"; ec <- grid$cond == "eyes-closed"
  mm_eo <- mm[eo, ]; mm_ec <- mm[ec, ]
  cov_eo_ec <- diag(mm_eo %*% vcov(mp) %*% t(mm_ec))
  pvar_EO   <- diag(mm_eo %*% tcrossprod(vcov(mp), mm_eo))
  pvar_EC   <- diag(mm_ec %*% tcrossprod(vcov(mp), mm_ec))
  se_diff   <- sqrt(abs(pvar_EO + pvar_EC - 2 * cov_eo_ec))
  diff <- data.frame(group = factor(group_levels, group_levels),
                     d = grid$y[eo] - grid$y[ec], se = se_diff)

  list(band = band, grid = grid, diff = diff, con2 = con2, con3 = con3)
}

## --- panel builders ---------------------------------------------------------
left_panel <- function(b, title = NULL) {
  g <- b$grid; rng <- max(g$phi) - min(g$plo)
  p <- ggplot(g, aes(cond, y, fill = group)) +
    geom_col(position = position_dodge(0.9), width = 0.75, colour = "grey20", linewidth = 0.5) +
    geom_errorbar(aes(ymin = plo, ymax = phi), position = position_dodge(0.9),
                  width = 0.2, linewidth = 0.6, colour = "grey20") +
    scale_fill_psy() +
    coord_cartesian(ylim = c(min(g$plo) - 0.15 * rng, max(g$phi) + 0.30 * rng)) +
    labs(x = NULL, y = ylab_power, title = title) + theme_psychodel() +
    theme(plot.title = element_text(face = "bold", hjust = 0))
  # star per condition where the group contrast (uncorrected p) is significant
  for (i in seq_len(nrow(b$con2))) {
    if (b$con2$p.value[i] < .05) {
      xc <- as.integer(factor(b$con2$cond[i], cond_levels))
      yb <- max(g$phi[g$cond == cond_levels[xc]]) + 0.12 * rng
      p <- p + sig_bracket(xc - 0.22, xc + 0.22, yb, p_stars(b$con2$p.value[i]),
                           tip = 0.04 * rng, textsize = 4.5)
    }
  }
  p
}

right_panel <- function(b) {
  d <- b$diff
  # always include 0 so the difference bars are anchored at a visible baseline
  lo0 <- min(0, min(d$d - d$se)); hi0 <- max(0, max(d$d + d$se))
  rng <- hi0 - lo0; if (rng == 0) rng <- 1
  neg <- mean(d$d) < 0                          # bars point down -> bracket below
  if (neg) {
    yb   <- min(d$d - d$se) - 0.16 * rng
    ylim <- c(yb - 0.16 * rng, hi0 + 0.06 * rng)
    side <- "bottom"
  } else {
    yb   <- max(d$d + d$se) + 0.16 * rng
    ylim <- c(lo0 - 0.06 * rng, yb + 0.16 * rng)
    side <- "top"
  }
  ggplot(d, aes(group, d, fill = group)) +
    geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.5) +
    geom_col(width = 0.65, colour = "grey20", linewidth = 0.5) +
    geom_errorbar(aes(ymin = d - se, ymax = d + se), width = 0.2, linewidth = 0.6, colour = "grey20") +
    scale_fill_psy() +
    coord_cartesian(ylim = ylim) +
    labs(x = "eyes-open minus eyes-closed", y = NULL) +
    theme_psychodel() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
    sig_bracket(1, 2, yb, p_stars(b$con3$p.value[1]), tip = 0.05 * rng, textsize = 5, side = side)
}

## --- assemble per band (band name as LEFT-panel title, so it survives the
##     nesting into the composite; nested plot_annotation() titles do not) -----
FORCE_RIGHT <- c("theta")
show_right <- function(b) b$con3$p.value[1] < .05 || b$band %in% FORCE_RIGHT

band_core <- function(b) {
  title <- paste0(toupper(substr(b$band, 1, 1)), substr(b$band, 2, nchar(b$band)))
  if (show_right(b)) {
    left_panel(b, title) + right_panel(b) + plot_layout(widths = c(1.6, 1))
  } else {
    left_panel(b, title)
  }
}

## --- run --------------------------------------------------------------------
built <- lapply(bands, build_band); names(built) <- bands
cat("\n-- Interaction (eyes-open minus eyes-closed, group difference), uncorrected p --\n")
for (b in built) cat(sprintf("  %-6s p = %.4f  %s\n", b$band, b$con3$p.value[1],
                             ifelse(b$con3$p.value[1] < .05, "-> right panel", "")))

for (b in built) {
  bp <- band_core(b) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
  w  <- if (show_right(b)) 7.5 else 5.2
  save_fig(bp, sprintf("figure2_psd_panel_%s", b$band), w = w, h = 5)
}

## composite of all five bands (2 columns, one shared legend)
comp <- patchwork::wrap_plots(lapply(built, band_core), ncol = 2) +
  plot_layout(guides = "collect") +
  plot_annotation(title = "Power spectral density by band, condition, and group") &
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"))
save_fig(comp, "figure2_psd", w = 15, h = 16)
