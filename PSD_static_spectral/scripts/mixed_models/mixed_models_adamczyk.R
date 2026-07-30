library(tidyverse)
library(emmeans)
library(ggplot2)
library(performance)
library(lmerTest)
library(effectsize)
library(ggsignif)
library(lme4)
library(Matrix)
library(pander)
panderOptions("table.split.table", Inf)#post-hocs will be displayed in a single line
library(patchwork)



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
  print(toupper(band))#so a band name will be displayed on top
  
  file_name <- file.path(HERE, sprintf("_%s_PSD_trials_separately.csv", band))
  df_band <- read.csv(file_name, sep=",", na.strings = "N/A")
  df_band <- df_band[,-1]#remove first column with row numbers
  
  #save dataframe for a future plot
  band_dataframes[[band]] <- df_band

  ### Lmer models #########
  model_formula <- as.formula(sprintf("%s ~ condition*group + city + (1|participant_id)", band))
  models[[band]] <- lmer(model_formula, data = df_band)

  ### Results per band ###
  
  anova_results <- anova(models[[band]])
  print(anova_results)
  print(eta_squared(models[[band]]))
  
  ### Post-hocs only when the interaction is significant ###
  if ("condition:group" %in% rownames(anova_results) && anova_results["condition:group", "Pr(>F)"] < 0.05) {
    print('Post-hoc contrast: ec - eo on c - e')
    post_hoc_results <- post_hocs(models[[band]])
    pander(format(post_hoc_results))
  }
  
  ### Just a visual gap ###
  print(' ')
  print('#################')
  
}

#lapply(models, anova)





######### Plot Models ###############

plot_list <- list()

for (band_name in bands) {

df_band <-  band_dataframes[[band_name]]
  
levels(df_band$group) <- c("Users", "Non-users")
df_band$group[df_band$group == "e"] <-'Users'
df_band$group[df_band$group == "c"] <-'Non-users'

#note that we don't save the model anywhere 
new_data_band <- expand.grid(
  condition=c('eo', 'ec'),
  group=c('Users', 'Non-users'))#, band_mean = 0)!!!!!!

new_data_band[[band_name]] <- 0

formula_band <- as.formula(paste(band_name, "~ condition*group + (1|participant_id)"))
band_lmer2 <- lmer(formula_band, df_band)

new_data_band[[band_name]] = predict(band_lmer2, new_data_band, re.form=NA)#band_name!!!!!!

mm = model.matrix(terms(band_lmer2), new_data_band)
## or new_data_N2$N2 <- mm %*% fixef(N2_lmer_city)
pvar1 = diag(mm %*% tcrossprod(vcov(band_lmer2),mm)) # wariancja bez uwzględnienia random effectów
tvar1 = pvar1+VarCorr(band_lmer2)$participant_id[1]  # wariancja z uwzględnieniem random effectów
# jeżeli chcesz CI zamiast SE to trzeba pomnożyć sqrt(pvar1) i sqrt (tvar1) przez cmult
#cmult = 1.96
new_data_band = data.frame(
  new_data_band, 
  plo = new_data_band[[band_name]]-sqrt(pvar1),#here two times was new_data_band$band_mean
  phi = new_data_band[[band_name]]+sqrt(pvar1)
  # tlo = new_data_N2$N2-sqrt(tvar1), # nie używaj tych wartości
  # thi = new_data_N2$N2+sqrt(tvar1)  # nie używaj tych wartości
)

#post-hocs but we're going to look only at the differences in ec-eo
post_hoc_results <- post_hocs(models[[band_name]])

#filtering with multiple contrasts and thus, multiple rows
specific_row <- post_hoc_results[Contrast == "eo - ec on c - e"]
adjusted_p_value <- specific_row[["Adjusted P value"]]

#filtering with a single contrast
#adjusted_p_value <- post_hoc_results[["Adjusted P value"]]

#also significance has to be on top of the bar so let's find it's maximum value
y_max = max(new_data_band[[band_name]])
y_min = min(new_data_band[[band_name]])


plot_band = ggplot(new_data_band, aes(x = condition, 
                               y = get(band_name), 
                               fill = group)) +
  geom_bar(stat = "identity", lwd = 1.2, color = "black",  position=position_dodge(.9))  +
  #labs(x="Condition", y = "Power [dB mV^2/Hz]") +
  labs(x = "Condition", y = expression("Power [dB" ~ frac(mu*V^2, Hz) ~ "]")) + 
  geom_errorbar(aes(ymin=plo, ymax=phi), width=.2, lwd=1.2, position=position_dodge(.9)) + 
  theme_classic(base_size=14) + 
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))) +
  theme(axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) +
  theme(legend.key.size = unit(1,"cm")) +
  scale_fill_manual(values = c("#6047ff", "#fc6d6d"),name="Group") +
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_cartesian(ylim = c(y_min-1, y_max+1)) + 
  theme(legend.position = "bottom") +
  theme(legend.justification = c("left"))# c(0, 35))

#additional condition for gamma eo condtion
#this is just a temporal solution tho, cause significance should appear whenever the results appear
#it should not be hard-coded for gamma eo
 if (band_name == 'gamma'){
   plot_band = plot_band +
     geom_signif(y_position = y_max+.8,
                 xmin = .75, xmax = 1.25,  # Set the x positions for the comparison
                 annotation = "*",
                 tip_length = 0.05,
                 size = 1.4,
                 textsize = 9,
                 vjust = .4)
 }


plot_list[[band_name]] <- plot_band
#print(plot_band)


##second subplot with substraction contrast##

#second plot appears only if there's any significance

if (adjusted_p_value < .05) {
  mm_eo <- mm[new_data_band$condition == "eo", ]
  mm_ec <- mm[new_data_band$condition == "ec", ]
  
  cov_matrix <- mm_eo %*% vcov(band_lmer2) %*% t(mm_ec)
  cov_eo_ec <- diag(cov_matrix)
  
  pvar_EO = diag(mm[which(new_data_band$condition == "eo"), ] %*% tcrossprod(vcov(band_lmer2), mm[which(new_data_band$condition == "eo"), ]))
  pvar_EC = diag(mm[which(new_data_band$condition == "ec"), ] %*% tcrossprod(vcov(band_lmer2), mm[which(new_data_band$condition == "ec"), ]))
  pvar_diff_correct <- pvar_EO + pvar_EC - 2 * cov_eo_ec
  
  
  se_diff <- sqrt(abs(pvar_diff_correct))
  
  new_data_diff = data.frame(
    Group = c("Non-users", "Users"), # KTÓRA KOLEJNOŚĆ JEST POPRAWNA?
    #Group = c("Users", "Non-users"),
    Difference = new_data_band[[band_name]][new_data_band$condition == "eo"] -
      new_data_band[[band_name]][new_data_band$condition == "ec"],
    SE = se_diff
  )

#difference plot

#for ylim
finding_limits = c(new_data_diff$Difference + new_data_diff$SE, new_data_diff$Difference - new_data_diff$SE)


plot_band_diff = ggplot(new_data_diff, aes(x = Group, y = Difference, fill = Group)) +
  geom_bar(stat = "identity", lwd = 1.2, color = "black",  position=position_dodge(.1), width=.7) +
  geom_errorbar(aes(ymin=Difference - SE, ymax=Difference + SE), width=.2, lwd=1.2, position=position_dodge(.1)) +
  scale_fill_manual(values = c("#6047ff", "#fc6d6d"),name="Group") + 
  theme_classic(base_size=14) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
  theme(legend.key.size = unit(.6,"cm")) +
  xlab("eo - ec") +
  ylab("") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position = "none") +
  theme(axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) +
  coord_cartesian(ylim = c(min(finding_limits)-.1, max(finding_limits)+.6))

}#end of "if there's any significance" statement

#Add significance star if adjusted p-value is less than 0.05
if (adjusted_p_value < .001) {
  plot_band_diff = plot_band_diff +
    geom_signif(y_position = max(finding_limits)+.17,
                xmin = 1, xmax = 2,  # Set the x positions for the comparison
                annotation = "***",
                tip_length = 0.05,
                size = 1.4,
                textsize = 9,
                vjust = .4)
}

#Theta will be trated separately since the significance is lower and plot reaches below 0
if (band_name == 'theta') {
  plot_band_diff = plot_band_diff +
    geom_signif(y_position = min(finding_limits)-.17,
                xmin = 1, xmax = 2,  # Set the x positions for the comparison
                annotation = "**",
                tip_length = -0.05,
                size = 1.4,
                textsize = 9,
                vjust = 1.7) +
    coord_cartesian(ylim = c(min(finding_limits)-.4, max(finding_limits)+.6))#a bit different limits for theta
}



#patchwork library of constructiong subplots

band_name_plot_title <- paste0(toupper(substr(band_name, 1, 1)), substr(band_name, 2, nchar(band_name)))

if (adjusted_p_value < .05) {
print(
  (plot_band + plot_band_diff) +
    plot_layout(widths = c(1.5, 1)) +
  plot_annotation(title = band_name_plot_title) &
  theme(plot.title = element_text(hjust = 0.55, size = 16)) #title in the middle of a plot
)
}

#plot without significance
else {
  print(plot_band +
    plot_annotation(title = band_name_plot_title) &
      theme(plot.title = element_text(hjust = 0.55, size = 16)) #title in the middle of a plot
)
}

}













######### Plot City Comparison ###############

plot_list <- list()

for (band_name in bands) {
  
  df_band <-  band_dataframes[[band_name]]
  
  levels(df_band$city) <- c("krk", "wwa")
  df_band$city[df_band$city == "krk"] <-'Cracow'
  df_band$city[df_band$city == "wwa"] <-'Warsaw'
  
  #note that we don't save the model anywhere 
  new_data_band <- expand.grid(
    condition=c('eo', 'ec'),
    city=c('Cracow', 'Warsaw'))#, band_mean = 0)!!!!!!
  
  new_data_band[[band_name]] <- 0
  
  formula_band <- as.formula(paste(band_name, "~ condition*city + (1|participant_id)"))
  band_lmer2 <- lmer(formula_band, df_band)
  
  new_data_band[[band_name]] = predict(band_lmer2, new_data_band, re.form=NA)#band_name!!!!!!
  
  mm = model.matrix(terms(band_lmer2), new_data_band)
  ## or new_data_N2$N2 <- mm %*% fixef(N2_lmer_city)
  pvar1 = diag(mm %*% tcrossprod(vcov(band_lmer2),mm)) # wariancja bez uwzględnienia random effectów
  tvar1 = pvar1+VarCorr(band_lmer2)$participant_id[1]  # wariancja z uwzględnieniem random effectów
  # jeżeli chcesz CI zamiast SE to trzeba pomnożyć sqrt(pvar1) i sqrt (tvar1) przez cmult
  #cmult = 1.96
  new_data_band = data.frame(
    new_data_band, 
    plo = new_data_band[[band_name]]-sqrt(pvar1),#here two times was new_data_band$band_mean
    phi = new_data_band[[band_name]]+sqrt(pvar1)
    # tlo = new_data_N2$N2-sqrt(tvar1), # nie używaj tych wartości
    # thi = new_data_N2$N2+sqrt(tvar1)  # nie używaj tych wartości
  )
  
  #Check whether city is significant
  anova_results <- anova(models[[band_name]])
  colnames(anova_results) <- trimws(colnames(anova_results))  # Remove extra spaces
  #filter this fella 
  specific_row <- anova_results["city", , drop=FALSE]
  adjusted_p_value <- specific_row[["Pr(>F)"]]#that is the value we should check while adding significance starts on a plot
  #also significance has to be on top of the bar so let's find it's maximum value
  y_max = max(new_data_band[[band_name]])
  y_min = min(new_data_band[[band_name]])
  
  
  plot_band = ggplot(new_data_band, aes(x = condition, 
                                        y = get(band_name), 
                                        fill = city)) +
    geom_bar(stat = "identity", lwd = 1.2, color = "black",  position=position_dodge())  +
    labs(x="Condition", y = "Power [dB mV^2/Hz]") +
    geom_errorbar(aes(ymin=plo, ymax=phi), width=.2, lwd=1.2, position=position_dodge(.9)) + 
    theme_classic(base_size=14) + 
    theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))) +
    theme(axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) +
    theme(legend.key.size = unit(1,"cm")) +
    scale_fill_manual(values = c("#6047ff", "#fc6d6d"),name="City") +
    ggtitle(sprintf(band_name))+
    theme(plot.title = element_text(hjust = 0.5)) +
    coord_cartesian(ylim = c(y_min-1, y_max+1)) #+ c(0, 35))

#Significance dla Citry!!!!!!!!  
  
  # if (adjusted_p_value < .001) {
  #   plot_band = plot_band +
  #     geom_signif(y_position = y_max+.5,
  #                 xmin = .5, xmax = 1.5,  # Set the x positions for the comparison
  #                 annotation = "***",
  #                 tip_length = 0.05,
  #                 size = 1.4,
  #                 textsize = 9,
  #                 vjust = .4)
  # }
  # 
  # else if (adjusted_p_value < .01) {
  #   plot_band = plot_band +
  #     geom_signif(y_position = y_max+.5,
  #                 xmin = .5, xmax = 1.5,  # Set the x positions for the comparison
  #                 annotation = "**",
  #                 tip_length = 0.05,
  #                 size = 1.4,
  #                 textsize = 9,
  #                 vjust = .4)
  # }
  # 
  # else if (adjusted_p_value < .05) {
  #   plot_band = plot_band +
  #     geom_signif(y_position = y_max+.5,
  #                 xmin = .5, xmax = 1.5,  # Set the x positions for the comparison
  #                 annotation = "*",
  #                 tip_length = 0.05,
  #                 size = 1.4,
  #                 textsize = 9,
  #                 vjust = .4)
  # }
  # 
  plot_list[[band_name]] <- plot_band
  print(plot_band)
}

