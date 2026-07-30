# =============================================================================
# Shared visual identity for the manuscript figures
#
# Group -> colour mapping is FIXED:
#   psy-exp  = purple   (line #756bb1, fill #bcbddc)
#   psy-naiv = orange   (line #e6550d, fill #fdae6b)
# Condition (eyes-open / eyes-closed) lives on the x-axis and is never coloured.
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(grid)
  library(svglite)
  library(ragg)
})

## --- paths -------------------------------------------------------------------
# Set the PSY_ROOT environment variable to redirect output somewhere else.
psy_script_dir <- function() {
  # When sourced, the calling frame carries the path of THIS file.
  for (i in seq_len(sys.nframe())) {
    ofile <- sys.frame(i)$ofile
    if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  }
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) return(dirname(normalizePath(sub("^--file=", "", m[1]))))
  normalizePath(getwd())
}

PSY_ROOT <- Sys.getenv("PSY_ROOT",
                       unset = normalizePath(file.path(psy_script_dir(), "..", "..")))
FIG_OUT  <- file.path(PSY_ROOT, "figures", "figures")
if (!dir.exists(FIG_OUT)) dir.create(FIG_OUT, recursive = TRUE)

## --- palette -----------------------------------------------------------------
psy_line <- c("psy-exp" = "#756bb1", "psy-naiv" = "#e6550d")  # dark line / point
psy_fill <- c("psy-exp" = "#bcbddc", "psy-naiv" = "#fdae6b")  # violin / bar fill
psy_mid  <- c("psy-exp" = "#9f98c9", "psy-naiv" = "#feca9f")  # mid tint (boxes)

## --- factor levels + label maps ---------------------------------------------
group_levels <- c("psy-exp", "psy-naiv")
cond_levels  <- c("eyes-open", "eyes-closed")

recode_group <- function(x) dplyr::recode(as.character(x),
  "1" = "psy-exp", "9" = "psy-naiv",
  "psy_exp" = "psy-exp", "psy_naive" = "psy-naiv",
  "Experimental" = "psy-exp", "Control" = "psy-naiv",
  "e" = "psy-exp", "c" = "psy-naiv",
  "Users" = "psy-exp", "Non-users" = "psy-naiv",
  .default = as.character(x))

recode_cond <- function(x) dplyr::recode(tolower(as.character(x)),
  "eo" = "eyes-open", "ec" = "eyes-closed",
  .default = as.character(x))

fct_group <- function(x) factor(recode_group(x), levels = group_levels)
fct_cond  <- function(x) factor(recode_cond(x),  levels = cond_levels)

## --- theme -------------------------------------------------------------------
theme_psychodel <- function(base_size = 13) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(colour = "grey90"),
      axis.line          = element_line(colour = "grey20"),
      axis.ticks         = element_line(colour = "grey20"),
      legend.position    = "bottom",
      legend.title       = element_text(face = "bold"),
      plot.title         = element_text(face = "bold", hjust = 0),
      plot.subtitle      = element_text(colour = "grey30"),
      plot.title.position = "plot",
      plot.caption       = element_text(colour = "grey40", hjust = 0.5)
    )
}

# Reusable scales that carry the legend. limits + drop=FALSE force BOTH groups
# to appear in the legend even for single-group panels.
scale_fill_psy <- function(name = "Group", ...)
  scale_fill_manual(values = psy_fill, name = name,
                    limits = group_levels, drop = FALSE, ...)
scale_colour_psy <- function(name = "Group", ...)
  scale_colour_manual(values = psy_line, name = name,
                      limits = group_levels, drop = FALSE, ...)

## --- split-violin geom (self-contained; no external package) ----------------
# Canonical GeomSplitViolin (after jan-glx). Odd group -> left half,
# even group -> right half. With group_levels ordered psy-exp, psy-naiv this
# puts psy-exp (purple) on the LEFT and psy-naiv (orange) on the RIGHT.
GeomSplitViolin <- ggproto("GeomSplitViolin", GeomViolin,
  draw_group = function(self, data, ..., draw_quantiles = NULL) {
    data <- transform(data,
                      xminv = x - violinwidth * (x - xmin),
                      xmaxv = x + violinwidth * (xmax - x))
    grp <- data[1, "group"]
    newdata <- data[order(if (grp %% 2 == 1) data$y else -data$y), ]
    newdata$x <- if (grp %% 2 == 1) newdata$xminv else newdata$xmaxv
    newdata <- rbind(newdata[1, ], newdata, newdata[nrow(newdata), ])
    newdata[c(1, nrow(newdata)), "x"] <- round(newdata[1, "x"])
    if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
      stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <= 1))
      quantiles  <- ggplot2:::create_quantile_segment_frame(data, draw_quantiles)
      aesthetics <- data[rep(1, nrow(quantiles)),
                         setdiff(names(data), c("x", "y")), drop = FALSE]
      aesthetics$alpha <- rep(1, nrow(quantiles))
      both <- cbind(quantiles, aesthetics)
      quantile_grob <- GeomPath$draw_panel(both, ...)
      ggplot2:::ggname("geom_split_violin",
        grid::grobTree(GeomPolygon$draw_panel(newdata, ...), quantile_grob))
    } else {
      ggplot2:::ggname("geom_split_violin", GeomPolygon$draw_panel(newdata, ...))
    }
  })

geom_split_violin <- function(mapping = NULL, data = NULL, stat = "ydensity",
                              position = "identity", ..., draw_quantiles = NULL,
                              trim = TRUE, scale = "area", na.rm = FALSE,
                              show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomSplitViolin,
        position = position, show.legend = show.legend, inherit.aes = inherit.aes,
        params = list(trim = trim, scale = scale,
                      draw_quantiles = draw_quantiles, na.rm = na.rm, ...))
}

## --- significance bracket (replaces ggsignif) -------------------------------
p_stars <- function(p) ifelse(p < .001, "***",
                       ifelse(p < .01,  "**",
                       ifelse(p < .05,  "*", "n.s.")))

# side = "top": bracket sits above the bars, tips point down, label above the
#   line (use over positive bars). side = "bottom": bracket sits below the bars,
#   tips point up, label below the line (use over negative/downward bars).
sig_bracket <- function(x1, x2, y, label, tip = NULL, textsize = 4.4, side = "top") {
  if (is.null(tip)) tip <- abs(y) * 0.01
  ytip <- if (side == "top") y - tip else y + tip
  vj   <- if (side == "top") -0.25 else 1.15
  list(
    annotate("segment", x = x1, xend = x1, y = ytip, yend = y),
    annotate("segment", x = x2, xend = x2, y = ytip, yend = y),
    annotate("segment", x = x1, xend = x2, y = y,    yend = y),
    annotate("text", x = (x1 + x2) / 2, y = y, label = label,
             vjust = vj, size = textsize)
  )
}

## --- three-format writer -----------------------------------------------------
save_fig <- function(plot, name, w = 7, h = 5, dpi = 400) {
  ggsave(file.path(FIG_OUT, paste0(name, ".pdf")), plot,
         device = grDevices::pdf, width = w, height = h, units = "in")
  ggsave(file.path(FIG_OUT, paste0(name, ".png")), plot,
         device = ragg::agg_png, width = w, height = h, units = "in", dpi = dpi)
  ggsave(file.path(FIG_OUT, paste0(name, ".svg")), plot,
         device = svglite::svglite, width = w, height = h, units = "in")
  message(sprintf("  wrote %s.{pdf,png,svg}", name))
  invisible(name)
}
