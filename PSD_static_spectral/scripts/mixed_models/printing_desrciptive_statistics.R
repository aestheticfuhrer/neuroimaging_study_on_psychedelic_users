library(dplyr)
library(plotrix)

# Resolve paths relative to this script; the input CSVs and post_hoc_adamczyk.R
# live in this same directory, so no working directory needs to be set.
script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) return(dirname(normalizePath(sub("^--file=", "", m[1]))))
  for (i in seq_len(sys.nframe())) {
    ofile <- sys.frame(i)$ofile
    if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  }
  normalizePath(getwd())
}
HERE <- script_dir()

source(file.path(HERE, "post_hoc_adamczyk.R"))

##################################################################

bands <- c("delta", "theta", "alpha", "beta", "gamma")
models <- list()
band_dataframes <- list()

#Iterate over bands and add a model to a list of models
for (band in bands) {
  
  file_name <- file.path(HERE, sprintf("_%s_PSD_trials_separately.csv", band))
  df_band <- read.csv(file_name, sep=",", na.strings = "N/A")
  df_band <- df_band[,-1]#remove first column with row numbers
  
  #different conditions
  control <- df_band %>%
    filter(group == 'c') %>%
    pull(band)
  
  experimental <- df_band %>%
    filter(group == 'e') %>%
    pull(band)
  
  eyes_open <- df_band %>%
    filter(condition == 'eo') %>%
    pull(band)
  
  eyes_closed <- df_band %>%
    filter(condition == 'ec') %>%
    pull(band)
  
  cracow <- df_band %>%
    filter(city == 'krk') %>%
    pull(band)
  
  warsaw <- df_band %>%
    filter(city == 'wwa') %>%
    pull(band)
  
  #weird post-hoc
  eyes_open_on_e <- df_band %>%
    filter(condition == 'eo' & group == 'e') %>%
    pull(band)
  
  eyes_closed_on_e <- df_band %>%
    filter(condition == 'ec' & group == 'e') %>%
    pull(band)
  
  eyes_open_on_c <- df_band %>%
    filter(condition == 'eo' & group == 'c') %>%
    pull(band)
  
  eyes_closed_on_c <- df_band %>%
    filter(condition == 'ec' & group == 'c') %>%
    pull(band)
  
  #eo_minus_ec_on_exp = eyes_open_on_e - eyes_closed_on_e
  #eo_minus_ec_on_con = eyes_open_on_c - eyes_closed_on_c
  
  #printing
  print(toupper(band))#so a band name will be displayed on 
  print('--')
  print(paste0('Control: M=', round(mean(control),2), ' ; STD=', round(sd(control),2)))
  print(paste0('Experimental: M=', round(mean(experimental),2), ' ; STD=', round(sd(experimental),2)))
  print(paste0('Eyes open: M=', round(mean(eyes_open),2), ' ; STD=', round(sd(eyes_open),2)))
  print(paste0('Eyes Closed: M=', round(mean(eyes_closed),2), ' ; STD=', round(sd(eyes_closed),2)))
  print(paste0('Krakow: M=', round(mean(cracow),2), ' ; STD=', round(sd(cracow),2)))
  print(paste0('Warsaw: M=', round(mean(warsaw),2), ' ; STD=', round(sd(warsaw),2)))
  #print(paste0('EO-EC among Controls: M=', round(mean(eo_minus_ec_on_con),2), ' ; SE=', round(std.error(eo_minus_ec_on_con),2)))
  #print(paste0('EO-EC among Experimental: M=', round(mean(eo_minus_ec_on_exp),2), ' ; SE=', round(std.error(eo_minus_ec_on_exp),2)))
  print('--------------')
  
}
