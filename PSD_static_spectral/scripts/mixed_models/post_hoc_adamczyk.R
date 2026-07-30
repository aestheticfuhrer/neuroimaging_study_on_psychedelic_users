post_hocs <- function(model) {
  require(data.table)
  require(emmeans)
  
  # Tworzenie data.frames z wyników post hoc
  # con1 <- as.data.frame(emmeans(model, pairwise ~ condition | group, type = "response", lmer.df = "asymp")$contrasts)
  con2 <- as.data.frame(emmeans(model, pairwise ~ group | condition, type = "response", lmer.df = "asymp")$contrasts)
  con3 <- as.data.frame(contrast(emmeans(model, ~ condition:group, type = "response", lmer.df = "asymp"), interaction = "pairwise"))
  
  # Dodanie opisowych etykiet dla kontrastów
  # con1$contrast <- with(con1, paste0(contrast, " on ", group))
  # con1$group <- NULL
  con2$contrast <- with(con2, paste0(contrast, " on ", condition))
  con2$condition <- NULL
  con3$contrast <- with(con3, paste0(condition_pairwise, " on ", group_pairwise))
  con3$condition_pairwise <- NULL
  con3$group_pairwise <- NULL
  #w projekcie rs psychodel odejmujemy eo od ec a nie na odwrót
  con3$z.ratio = con3$z.ratio*-1
  
  # Łączenie wyników z różnych kontrastów
  con <- rbindlist(list(con2, con3), fill = TRUE) #rbindlist(list(con1, con2, con3), fill = TRUE)
  
  
  # Transformacja na data.table
  con <- as.data.table(con)
  
  # Usunięcie kolumny 'df', jeśli istnieje
  if ("df" %in% colnames(con)) {
    con <- con[, !c("df"), with = FALSE]
  }
  
  # Dodanie kolumny z poprawionymi wartościami p
  if ("p.value" %in% colnames(con)) {
    con[, p.value.adj := p.adjust(p.value, method = "holm")]
  } else {
    stop("Nie znaleziono kolumny 'p.value' w wynikach kontrastów.")
  }
  
  # Automatyczne generowanie nowych nazw kolumn na podstawie aktualnych danych
  col_names <- colnames(con)
  custom_names <- c("Contrast", 
                    "Estimate", 
                    "Standard Error", 
                    "Z ratio", 
                    "P value", 
                    "Adjusted P value")
  
  # Dopasowanie liczby nazw do liczby kolumn w tabeli
  if (length(col_names) <= length(custom_names)) {
    new_names <- custom_names[1:length(col_names)]
  } else {
    new_names <- c(custom_names, paste0("Extra Column ", seq(length(custom_names) + 1, length(col_names))))
  }
  
  # Ustawienie nowych nazw kolumn
  setnames(con, col_names, new_names)
  
  #w projekcie rs psychodel odejmujemy eo od ec a nie na odwrót
  con$Contrast[con$Contrast == "ec - eo on c - e"] <- "eo - ec on c - e"
  
  return(con)
}
