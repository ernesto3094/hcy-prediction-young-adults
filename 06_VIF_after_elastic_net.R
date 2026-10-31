# ============================================================
# SCRIPT 06
# VIF AFTER ELASTIC NET VARIABLE SELECTION
#
# IMPORTANT:
# - TRAIN data only.
# - TEST data are never read or accessed.
# - Uses the transformed TRAIN representation frozen earlier.
# - Predictor sets come directly from the final Elastic Net
#   selection files produced in Script 05.
# ============================================================


# ============================================================
# 1. PACKAGES
# ============================================================

required_packages <- c(
  "readr",
  "dplyr",
  "car"
)

missing_packages <- required_packages[
  !required_packages %in% rownames(installed.packages())
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(readr)
library(dplyr)
library(car)


# ============================================================
# 2. INPUT DIRECTORY AND FILES
# ============================================================

input_dir <- "05_elastic_net_results"

women_data_file <- file.path(
  input_dir,
  "01_women_TRAIN_transformed.csv"
)

men_data_file <- file.path(
  input_dir,
  "02_men_TRAIN_transformed.csv"
)

women_selection_file <- file.path(
  input_dir,
  "10_women_final_selected_predictors.csv"
)

men_selection_file <- file.path(
  input_dir,
  "20_men_final_selected_predictors.csv"
)


required_files <- c(
  women_data_file,
  men_data_file,
  women_selection_file,
  men_selection_file
)

missing_files <- required_files[
  !file.exists(required_files)
]

if (length(missing_files) > 0) {
  stop(
    paste(
      "Missing input files:",
      paste(
        missing_files,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 3. READ DATA
# ============================================================

women_train <- read_csv(
  women_data_file,
  show_col_types = FALSE
)

men_train <- read_csv(
  men_data_file,
  show_col_types = FALSE
)

women_selected <- read_csv(
  women_selection_file,
  show_col_types = FALSE
)

men_selected <- read_csv(
  men_selection_file,
  show_col_types = FALSE
)


# ============================================================
# 4. EXTRACT FINAL PREDICTOR SETS
# ============================================================

women_predictors <- women_selected$Variable
men_predictors <- men_selected$Variable

cat(
  "\nWomen selected predictors:\n"
)

print(
  women_predictors
)

cat(
  "\nMen selected predictors:\n"
)

print(
  men_predictors
)


# ============================================================
# 5. BASIC CHECKS
# ============================================================

if (nrow(women_train) != 190) {
  stop(
    paste(
      "Unexpected WOMEN TRAIN sample size:",
      nrow(women_train)
    )
  )
}

if (nrow(men_train) != 83) {
  stop(
    paste(
      "Unexpected MEN TRAIN sample size:",
      nrow(men_train)
    )
  )
}


missing_women <- setdiff(
  c(
    "HCY",
    women_predictors
  ),
  names(women_train)
)

missing_men <- setdiff(
  c(
    "HCY",
    men_predictors
  ),
  names(men_train)
)

if (length(missing_women) > 0) {
  stop(
    paste(
      "Missing WOMEN variables:",
      paste(
        missing_women,
        collapse = ", "
      )
    )
  )
}

if (length(missing_men) > 0) {
  stop(
    paste(
      "Missing MEN variables:",
      paste(
        missing_men,
        collapse = ", "
      )
    )
  )
}


women_analysis <- women_train %>%
  select(
    HCY,
    all_of(women_predictors)
  )

men_analysis <- men_train %>%
  select(
    HCY,
    all_of(men_predictors)
  )


if (any(is.na(women_analysis))) {
  stop(
    "Missing values detected in WOMEN analytical data."
  )
}

if (any(is.na(men_analysis))) {
  stop(
    "Missing values detected in MEN analytical data."
  )
}


# ============================================================
# 6. OUTPUT DIRECTORY
# ============================================================

output_dir <- "06_vif_results"

if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}


# ============================================================
# 7. VIF FUNCTION
# ============================================================

calculate_vif <- function(
    data,
    predictors,
    sex_name
) {
  
  model_formula <- as.formula(
    paste(
      "HCY ~",
      paste(
        predictors,
        collapse = " + "
      )
    )
  )
  
  model <- lm(
    model_formula,
    data = data
  )
  
  vif_values <- car::vif(
    model
  )
  
  
  # ==========================================================
  # Handle standard VIF or GVIF output
  # ==========================================================
  
  if (is.matrix(vif_values)) {
    
    vif_table <- data.frame(
      Predictor =
        rownames(vif_values),
      VIF =
        vif_values[
          ,
          "GVIF^(1/(2*Df))"
        ],
      stringsAsFactors = FALSE
    )
    
  } else {
    
    vif_table <- data.frame(
      Predictor =
        names(vif_values),
      VIF =
        as.numeric(vif_values),
      stringsAsFactors = FALSE
    )
  }
  
  
  # ==========================================================
  # Tolerance
  # ==========================================================
  
  vif_table <- vif_table %>%
    mutate(
      Tolerance = 1 / VIF
    )
  
  
  # ==========================================================
  # Descriptive interpretation
  # ==========================================================
  
  vif_table <- vif_table %>%
    mutate(
      Interpretation = case_when(
        VIF < 5 ~
          "Low multicollinearity",
        VIF >= 5 & VIF < 10 ~
          "Potentially relevant multicollinearity",
        VIF >= 10 ~
          "High multicollinearity"
      )
    ) %>%
    arrange(
      desc(VIF)
    )
  
  
  # ==========================================================
  # Correlation matrix among selected predictors
  # ==========================================================
  
  correlation_matrix <- cor(
    data[
      ,
      predictors,
      drop = FALSE
    ],
    use = "pairwise.complete.obs",
    method = "pearson"
  )
  
  correlation_table <- data.frame(
    Predictor =
      rownames(
        correlation_matrix
      ),
    correlation_matrix,
    check.names = FALSE,
    row.names = NULL
  )
  
  
  # ==========================================================
  # Summary
  # ==========================================================
  
  summary_table <- data.frame(
    Sex = sex_name,
    N = nrow(data),
    Number_Predictors =
      length(predictors),
    Maximum_VIF =
      max(vif_table$VIF),
    Mean_VIF =
      mean(vif_table$VIF),
    Minimum_Tolerance =
      min(vif_table$Tolerance),
    Predictors_VIF_5_or_more =
      sum(
        vif_table$VIF >= 5
      ),
    Predictors_VIF_10_or_more =
      sum(
        vif_table$VIF >= 10
      )
  )
  
  
  list(
    model = model,
    vif_table = vif_table,
    correlation_table =
      correlation_table,
    summary_table =
      summary_table
  )
}


# ============================================================
# 8. WOMEN VIF
# ============================================================

women_vif <- calculate_vif(
  data = women_analysis,
  predictors = women_predictors,
  sex_name = "Women"
)


# ============================================================
# 9. MEN VIF
# ============================================================

men_vif <- calculate_vif(
  data = men_analysis,
  predictors = men_predictors,
  sex_name = "Men"
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

write_csv(
  women_vif$vif_table,
  file.path(
    output_dir,
    "01_women_VIF.csv"
  )
)

write_csv(
  women_vif$correlation_table,
  file.path(
    output_dir,
    "02_women_selected_predictor_correlations.csv"
  )
)

write_csv(
  women_vif$summary_table,
  file.path(
    output_dir,
    "03_women_VIF_summary.csv"
  )
)


write_csv(
  men_vif$vif_table,
  file.path(
    output_dir,
    "04_men_VIF.csv"
  )
)

write_csv(
  men_vif$correlation_table,
  file.path(
    output_dir,
    "05_men_selected_predictor_correlations.csv"
  )
)

write_csv(
  men_vif$summary_table,
  file.path(
    output_dir,
    "06_men_VIF_summary.csv"
  )
)


# ============================================================
# 11. DISPLAY WOMEN RESULTS
# ============================================================

cat(
  "\n\n==================================================\n"
)

cat(
  "WOMEN - VIF AFTER ELASTIC NET\n"
)

cat(
  "==================================================\n"
)

print(
  women_vif$vif_table
)

cat(
  "\nWOMEN SUMMARY:\n"
)

print(
  women_vif$summary_table
)


# ============================================================
# 12. DISPLAY MEN RESULTS
# ============================================================

cat(
  "\n\n==================================================\n"
)

cat(
  "MEN - VIF AFTER ELASTIC NET\n"
)

cat(
  "==================================================\n"
)

print(
  men_vif$vif_table
)

cat(
  "\nMEN SUMMARY:\n"
)

print(
  men_vif$summary_table
)


# ============================================================
# 13. FINAL CHECK
# ============================================================

if (
  max(women_vif$vif_table$VIF) >= 10 ||
  max(men_vif$vif_table$VIF) >= 10
) {
  
  warning(
    paste(
      "At least one selected predictor has VIF >= 10.",
      "Review multicollinearity before proceeding."
    )
  )
  
} else {
  
  cat(
    "\nNo selected predictor has VIF >= 10.\n"
  )
}


# ============================================================
# 14. FINAL MESSAGE
# ============================================================

cat(
  "\n==================================================\n",
  "SCRIPT 06 COMPLETED SUCCESSFULLY\n",
  "==================================================\n",
  "Only transformed TRAIN data were used.\n",
  "TEST data were never read or accessed.\n",
  "VIF was calculated after Elastic Net selection.\n",
  "Centering/scaling was not required for VIF.\n",
  "==================================================\n"
)
