# =============================================================================
# 01_lz.R  --  Figure 3: Lempel-Ziv complexity by condition and group
#
# Significance stars come from the trial-level mixed model; the bar
# error bars come from SUBJECT-level means (trials aggregated first) to avoid
# pseudoreplication. Recoloured to the psychodel palette; EO/EC -> eyes-open/
# eyes-closed; Experimental/Control -> psy-exp/psy-naiv.
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
suppressPackageStartupMessages({ library(lmerTest); library(emmeans) })

# Trial-level LZ (Subject_ID, Condition EO/EC, Trial_N, Mean_LZ, City.x).
#
# Group and city are taken from beh/beh_data_merged.csv rather than from the
# demographic columns baked into complexity_data.csv. Those columns are the
# result of an older join that missed two participants (Cracow 111, Warsaw 58),
# leaving their Group as NA; reading membership from the single behavioural
# table keeps one source of truth and picks both of them up.
FINAL <- read.csv(file.path(REPO, "beh", "beh_data_merged.csv"), stringsAsFactors = FALSE)
lz <- read.csv(file.path(REPO, "LZ_Lempel-Ziv_complexity", "data", "complexity_data.csv"),
               stringsAsFactors = FALSE) |>
  mutate(sid_num = suppressWarnings(as.integer(Subject_ID)),
         city    = ifelse(tolower(City.x) == "krk", "krk", "wwa")) |>
  inner_join(FINAL[, c("sid", "city", "group")],
             by = c("sid_num" = "sid", "city" = "city")) |>
  mutate(City  = city,
         group = fct_group(group),
         cond  = fct_cond(Condition),
         sid   = factor(paste(city, Subject_ID, sep = "-"))) |>
  filter(!is.na(group), !is.na(cond), !is.na(Mean_LZ))

cat(sprintf("\n-- LZ sample: N = %d --\n", dplyr::n_distinct(lz$sid)))
print(table(city = lz$City[!duplicated(lz$sid)], group = lz$group[!duplicated(lz$sid)]))

## --- (a) trial-level mixed model -> significance stars ----------------------
m  <- lmerTest::lmer(Mean_LZ ~ cond * group + City + (1 | sid), data = lz)
ct <- as.data.frame(pairs(emmeans(m, ~ group | cond, lmer.df = "asymptotic")))
cat("\n-- Group contrasts within condition (asymptotic) --\n"); print(ct)

star_by_cond <- setNames(p_stars(ct$p.value), as.character(ct$cond))

## --- (b) subject-level means -> group SE (pseudoreplication-safe) ------------
subj <- lz |>
  group_by(sid, group, cond) |>
  summarise(v = mean(Mean_LZ), .groups = "drop")
cell <- subj |>
  group_by(group, cond) |>
  summarise(m = mean(v), se = sd(v) / sqrt(n()), n = n(), .groups = "drop")
cat("\n-- Subject-level cell means +/- SE --\n"); print(cell)

## --- plotting geometry ------------------------------------------------------
lo <- min(cell$m - cell$se); hi <- max(cell$m + cell$se); rng <- hi - lo
ylim <- c(lo - 0.35 * rng, hi + 0.55 * rng)
dodge <- position_dodge(width = 0.8)

# bracket y per condition (above the taller bar+SE), with headroom for the star
brk <- cell |> group_by(cond) |> summarise(top = max(m + se), .groups = "drop") |>
  mutate(xc = as.integer(cond),
         y  = top + 0.18 * rng,
         lab = star_by_cond[as.character(cond)])

p_lz <- ggplot(cell, aes(cond, m, fill = group)) +
  geom_col(position = dodge, width = 0.7, colour = "grey20", linewidth = 0.5) +
  geom_errorbar(aes(ymin = m - se, ymax = m + se),
                position = dodge, width = 0.2, linewidth = 0.6, colour = "grey20") +
  scale_fill_psy() +
  coord_cartesian(ylim = ylim) +
  labs(x = NULL, y = "Lempel-Ziv mean values (LZ)",
       title = "Lempel-Ziv complexity",
       subtitle = "Subject-level means; error bars = SE") +
  theme_psychodel()

# significance brackets per condition
for (i in seq_len(nrow(brk))) {
  p_lz <- p_lz +
    sig_bracket(brk$xc[i] - 0.2, brk$xc[i] + 0.2, brk$y[i], brk$lab[i],
                tip = 0.05 * rng, textsize = 5)
}

save_fig(p_lz, "figure3_lz", w = 6.5, h = 5.2)
