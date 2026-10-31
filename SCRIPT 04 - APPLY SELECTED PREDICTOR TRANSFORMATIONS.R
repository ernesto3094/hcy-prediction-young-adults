# ============================================================
# SCRIPT 04 - APPLY SELECTED PREDICTOR TRANSFORMATIONS
#             AND REASSESS PRELIMINARY MLR DIAGNOSTICS
# ============================================================
#
# Study:
# Exploratory Sex-Stratified Modeling of Homocysteine Levels
# in Young Adults Using Regression and Machine Learning
# Approaches
#
# PURPOSE:
#
#   1. Load the sex-specific TRAIN datasets.
#   2. Rename analytical variables to English.
#   3. Apply the predictor transformations selected after
#      reviewing:
#
#         - raw descriptive statistics
#         - raw-data boxplots
#         - preliminary MLR diagnostics
#         - transformation comparison statistics
#         - transformation comparison histograms
#
#   4. Save the final transformation plan.
#   5. Save transformed TRAIN datasets.
#   6. Compare predictor skewness and kurtosis before and
#      after the selected transformations.
#   7. Refit one preliminary MLR per sex using the transformed
#      predictor representation.
#   8. Reassess:
#
#         - residual mean
#         - Lilliefors test
#         - Breusch-Pagan test
#         - NCV test
#         - Durbin-Watson test
#         - Runs test
#         - residual ACF
#         - GVLMA
#
#   9. Generate post-transformation diagnostic figures.
#
#
# IMPORTANT:
#
#   - TRAIN data only.
#   - TEST data are NOT accessed.
#   - HCY remains on its ORIGINAL measurement scale.
#   - No observations are removed.
#   - No centering or scaling is performed.
#   - No Elastic Net is performed.
#   - No final predictive model is fitted.
#   - The preliminary MLR is used for diagnostic purposes only.
#
# ============================================================


# ------------------------------------------------------------
# 1. REQUIRED PACKAGES
# ------------------------------------------------------------

required_packages <- c(
  "dplyr",
  "tidyr",
  "readr",
  "ggplot2",
  "nortest",
  "lmtest",
  "car",
  "gvlma"
)


missing_packages <- required_packages[
  !sapply(
    required_packages,
    requireNamespace,
    quietly = TRUE
  )
]


if (length(missing_packages) > 0) {
  
  stop(
    paste0(
      "\nThe following packages are missing:\n",
      paste(missing_packages, collapse = ", "),
      "\n\nInstall them using:\n\n",
      "install.packages(c(",
      paste(
        paste0('"', missing_packages, '"'),
        collapse = ", "
      ),
      "))\n"
    )
  )
}


library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(nortest)
library(lmtest)
library(car)
library(gvlma)


# ------------------------------------------------------------
# 2. CREATE OUTPUT DIRECTORIES
# ------------------------------------------------------------
#
# The script automatically creates one main folder and
# separate subfolders for women and men.
#
# No computer-specific path is required.
#
# ------------------------------------------------------------

output_dir <- "04_selected_transformations_results"

women_dir <- file.path(
  output_dir,
  "women"
)

men_dir <- file.path(
  output_dir,
  "men"
)

women_figures_dir <- file.path(
  women_dir,
  "figures"
)

men_figures_dir <- file.path(
  men_dir,
  "figures"
)


dir.create(
  women_figures_dir,
  showWarnings = FALSE,
  recursive = TRUE
)

dir.create(
  men_figures_dir,
  showWarnings = FALSE,
  recursive = TRUE
)


cat(
  "\n=============================================\n",
  "OUTPUT DIRECTORY\n",
  "=============================================\n",
  normalizePath(
    output_dir,
    winslash = "/",
    mustWork = FALSE
  ),
  "\n"
)


# ------------------------------------------------------------
# 3. LOAD WOMEN TRAIN DATA
# ------------------------------------------------------------
#
# Select:
# 01_women_TRAIN.csv
#
# ------------------------------------------------------------

cat(
  "\nSelect file: 01_women_TRAIN.csv\n"
)

women_train_original <- read_csv(
  file.choose(),
  show_col_types = FALSE
)


# ------------------------------------------------------------
# 4. LOAD MEN TRAIN DATA
# ------------------------------------------------------------
#
# Select:
# 03_men_TRAIN.csv
#
# ------------------------------------------------------------

cat(
  "\nSelect file: 03_men_TRAIN.csv\n"
)

men_train_original <- read_csv(
  file.choose(),
  show_col_types = FALSE
)


# ------------------------------------------------------------
# 5. RENAME VARIABLES TO ENGLISH
# ------------------------------------------------------------

english_name_map <- c(
  
  "GENERO"   = "Gender",
  "EDAD"     = "Age",
  "TALLA"    = "Height",
  "PESO"     = "Weight",
  "IMC"      = "BMI",
  "GRASAT"   = "TotalFat",
  "GRASAVIS" = "VisceralFat",
  "MUSCULO"  = "MuscleMass",
  "HGB"      = "Hgb",
  "HTO"      = "Hct",
  "VCM"      = "MCV",
  "RDWCV"    = "RDW_CV",
  "COL"      = "TotalCholesterol"
)


rename_to_english <- function(data) {
  
  for (old_name in names(english_name_map)) {
    
    if (old_name %in% names(data)) {
      
      names(data)[
        names(data) == old_name
      ] <- english_name_map[[old_name]]
    }
  }
  
  return(data)
}


women_train_original <- rename_to_english(
  women_train_original
)

men_train_original <- rename_to_english(
  men_train_original
)


# ------------------------------------------------------------
# 6. DEFINE THE 21 ELIGIBLE PREDICTORS
# ------------------------------------------------------------

predictors <- c(
  
  "Age",
  "Height",
  "Weight",
  "BMI",
  "TotalFat",
  "VisceralFat",
  "MuscleMass",
  
  "Hgb",
  "Hct",
  "MCV",
  "MCH",
  "MCHC",
  "RDW_CV",
  "RBC",
  "PLT",
  "WBC",
  
  "GLU",
  "TotalCholesterol",
  "TRG",
  "HDL",
  "LDL"
)


stopifnot(
  length(predictors) == 21
)


# ------------------------------------------------------------
# 7. VERIFY REQUIRED VARIABLES
# ------------------------------------------------------------

required_variables <- c(
  "HCY",
  predictors
)


missing_women <- setdiff(
  required_variables,
  names(women_train_original)
)

missing_men <- setdiff(
  required_variables,
  names(men_train_original)
)


if (length(missing_women) > 0) {
  
  stop(
    paste(
      "Missing variables in WOMEN TRAIN:",
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
      "Missing variables in MEN TRAIN:",
      paste(
        missing_men,
        collapse = ", "
      )
    )
  )
}


# ------------------------------------------------------------
# 8. DEFINE FINAL TRANSFORMATION PLAN
# ------------------------------------------------------------
#
# These decisions were selected after joint review of the
# TRAIN-only descriptive statistics and graphical assessment.
#
# Predictor normality was NOT required.
#
# Transformations were retained only when a simple
# transformation provided a clear and defensible improvement.
#
# HCY is not included because it remains on its original
# measurement scale.
#
# ------------------------------------------------------------

women_transformation_plan <- data.frame(
  
  Sex = "Women",
  
  Variable = predictors,
  
  Transformation = c(
    
    # Age
    "Original",
    
    # Height
    "Original",
    
    # Weight
    "Original",
    
    # BMI
    "Original",
    
    # TotalFat
    "Original",
    
    # VisceralFat
    "Original",
    
    # MuscleMass
    "Original",
    
    # Hgb
    "Original",
    
    # Hct
    "Original",
    
    # MCV
    "Original",
    
    # MCH
    "Original",
    
    # MCHC
    "Original",
    
    # RDW_CV
    "Original",
    
    # RBC
    "Original",
    
    # PLT
    "Log",
    
    # WBC
    "Log",
    
    # GLU
    "Original",
    
    # TotalCholesterol
    "Original",
    
    # TRG
    "Log",
    
    # HDL
    "Log",
    
    # LDL
    "SquareRoot"
  ),
  
  stringsAsFactors = FALSE
)


men_transformation_plan <- data.frame(
  
  Sex = "Men",
  
  Variable = predictors,
  
  Transformation = c(
    
    # Age
    "Original",
    
    # Height
    "Original",
    
    # Weight
    "Original",
    
    # BMI
    "Original",
    
    # TotalFat
    "Original",
    
    # VisceralFat
    "Log",
    
    # MuscleMass
    "Original",
    
    # Hgb
    "Original",
    
    # Hct
    "Original",
    
    # MCV
    "Original",
    
    # MCH
    "Original",
    
    # MCHC
    "Original",
    
    # RDW_CV
    "Original",
    
    # RBC
    "Original",
    
    # PLT
    "Original",
    
    # WBC
    "Log",
    
    # GLU
    "Original",
    
    # TotalCholesterol
    "Original",
    
    # TRG
    "Log",
    
    # HDL
    "Log",
    
    # LDL
    "Original"
  ),
  
  stringsAsFactors = FALSE
)


final_transformation_plan <- bind_rows(
  
  women_transformation_plan,
  
  men_transformation_plan
)


# ------------------------------------------------------------
# 9. ADD RESPONSE-VARIABLE DOCUMENTATION
# ------------------------------------------------------------

response_plan <- data.frame(
  
  Sex = c(
    "Women",
    "Men"
  ),
  
  Variable = c(
    "HCY",
    "HCY"
  ),
  
  Transformation = c(
    "Original",
    "Original"
  ),
  
  stringsAsFactors = FALSE
)


documented_transformation_plan <- bind_rows(
  
  final_transformation_plan,
  
  response_plan
)


# ------------------------------------------------------------
# 10. SAVE FINAL TRANSFORMATION PLAN
# ------------------------------------------------------------

write_csv(
  
  documented_transformation_plan,
  
  file.path(
    output_dir,
    "00_final_transformation_plan.csv"
  )
)


# ------------------------------------------------------------
# 11. FUNCTION TO APPLY SELECTED TRANSFORMATIONS
# ------------------------------------------------------------

apply_selected_transformations <- function(
    data,
    transformation_plan
) {
  
  transformed_data <- data
  
  
  for (
    i in seq_len(
      nrow(
        transformation_plan
      )
    )
  ) {
    
    variable <-
      transformation_plan$Variable[i]
    
    transformation <-
      transformation_plan$Transformation[i]
    
    
    if (
      transformation == "Original"
    ) {
      
      next
    }
    
    
    if (
      transformation == "Log"
    ) {
      
      if (
        any(
          transformed_data[[variable]] <= 0,
          na.rm = TRUE
        )
      ) {
        
        stop(
          paste(
            "Log transformation cannot be applied to",
            variable,
            "because non-positive values were detected."
          )
        )
      }
      
      
      transformed_data[[variable]] <-
        log(
          transformed_data[[variable]]
        )
    }
    
    
    if (
      transformation == "SquareRoot"
    ) {
      
      if (
        any(
          transformed_data[[variable]] < 0,
          na.rm = TRUE
        )
      ) {
        
        stop(
          paste(
            "Square-root transformation cannot be applied to",
            variable,
            "because negative values were detected."
          )
        )
      }
      
      
      transformed_data[[variable]] <-
        sqrt(
          transformed_data[[variable]]
        )
    }
  }
  
  
  return(
    transformed_data
  )
}


# ------------------------------------------------------------
# 12. APPLY FINAL WOMEN TRANSFORMATIONS
# ------------------------------------------------------------

women_train_transformed <-
  
  apply_selected_transformations(
    
    data =
      women_train_original,
    
    transformation_plan =
      women_transformation_plan
  )


# ------------------------------------------------------------
# 13. APPLY FINAL MEN TRANSFORMATIONS
# ------------------------------------------------------------

men_train_transformed <-
  
  apply_selected_transformations(
    
    data =
      men_train_original,
    
    transformation_plan =
      men_transformation_plan
  )


# ------------------------------------------------------------
# 14. VERIFY THAT HCY DID NOT CHANGE
# ------------------------------------------------------------

if (
  !isTRUE(
    all.equal(
      women_train_original$HCY,
      women_train_transformed$HCY
    )
  )
) {
  
  stop(
    "HCY was unexpectedly modified in WOMEN data."
  )
}


if (
  !isTRUE(
    all.equal(
      men_train_original$HCY,
      men_train_transformed$HCY
    )
  )
) {
  
  stop(
    "HCY was unexpectedly modified in MEN data."
  )
}


# ------------------------------------------------------------
# 15. SAVE TRANSFORMED TRAIN DATASETS
# ------------------------------------------------------------
#
# Predictor names are preserved.
#
# Only their numerical representation changes where a final
# transformation was selected.
#
# ------------------------------------------------------------

write_csv(
  
  women_train_transformed,
  
  file.path(
    women_dir,
    "01_women_TRAIN_selected_transformations.csv"
  )
)


write_csv(
  
  men_train_transformed,
  
  file.path(
    men_dir,
    "01_men_TRAIN_selected_transformations.csv"
  )
)


# ------------------------------------------------------------
# 16. SKEWNESS FUNCTION
# ------------------------------------------------------------

skewness_function <- function(x) {
  
  x <- x[
    is.finite(x)
  ]
  
  
  if (
    length(x) < 3 ||
    sd(x) == 0
  ) {
    
    return(
      NA_real_
    )
  }
  
  
  mean(
    (x - mean(x))^3
  ) /
    sd(x)^3
}


# ------------------------------------------------------------
# 17. KURTOSIS FUNCTION
# ------------------------------------------------------------
#
# Pearson kurtosis:
#
# Normal distribution approximately = 3
#
# ------------------------------------------------------------

kurtosis_function <- function(x) {
  
  x <- x[
    is.finite(x)
  ]
  
  
  if (
    length(x) < 4 ||
    sd(x) == 0
  ) {
    
    return(
      NA_real_
    )
  }
  
  
  mean(
    (x - mean(x))^4
  ) /
    sd(x)^4
}


# ------------------------------------------------------------
# 18. BEFORE/AFTER DISTRIBUTION COMPARISON
# ------------------------------------------------------------

compare_before_after_distribution <- function(
    original_data,
    transformed_data,
    transformation_plan,
    sex_name
) {
  
  result_list <- list()
  
  
  for (
    variable in predictors
  ) {
    
    original_values <-
      original_data[[variable]]
    
    transformed_values <-
      transformed_data[[variable]]
    
    
    transformation <-
      
      transformation_plan$Transformation[
        transformation_plan$Variable ==
          variable
      ]
    
    
    result_list[[
      variable
    ]] <- data.frame(
      
      Sex =
        sex_name,
      
      Variable =
        variable,
      
      Transformation =
        transformation,
      
      Original_Skewness =
        skewness_function(
          original_values
        ),
      
      Transformed_Skewness =
        skewness_function(
          transformed_values
        ),
      
      Original_Kurtosis =
        kurtosis_function(
          original_values
        ),
      
      Transformed_Kurtosis =
        kurtosis_function(
          transformed_values
        )
    )
  }
  
  
  bind_rows(
    result_list
  ) %>%
    
    mutate(
      
      Original_Skewness =
        round(
          Original_Skewness,
          4
        ),
      
      Transformed_Skewness =
        round(
          Transformed_Skewness,
          4
        ),
      
      Original_Kurtosis =
        round(
          Original_Kurtosis,
          4
        ),
      
      Transformed_Kurtosis =
        round(
          Transformed_Kurtosis,
          4
        )
    )
}


women_distribution_comparison <-
  
  compare_before_after_distribution(
    
    original_data =
      women_train_original,
    
    transformed_data =
      women_train_transformed,
    
    transformation_plan =
      women_transformation_plan,
    
    sex_name =
      "Women"
  )


men_distribution_comparison <-
  
  compare_before_after_distribution(
    
    original_data =
      men_train_original,
    
    transformed_data =
      men_train_transformed,
    
    transformation_plan =
      men_transformation_plan,
    
    sex_name =
      "Men"
  )


# ------------------------------------------------------------
# 19. SAVE BEFORE/AFTER DISTRIBUTION TABLES
# ------------------------------------------------------------

write_csv(
  
  women_distribution_comparison,
  
  file.path(
    women_dir,
    "02_women_distribution_before_after.csv"
  )
)


write_csv(
  
  men_distribution_comparison,
  
  file.path(
    men_dir,
    "02_men_distribution_before_after.csv"
  )
)


# ------------------------------------------------------------
# 20. BUILD PRELIMINARY MLR FORMULA
# ------------------------------------------------------------

mlr_formula <- as.formula(
  
  paste(
    
    "HCY ~",
    
    paste(
      predictors,
      collapse = " + "
    )
  )
)


# ------------------------------------------------------------
# 21. FIT ORIGINAL AND TRANSFORMED PRELIMINARY MLR MODELS
# ------------------------------------------------------------

women_mlr_before <- lm(
  
  formula =
    mlr_formula,
  
  data =
    women_train_original
)


women_mlr_after <- lm(
  
  formula =
    mlr_formula,
  
  data =
    women_train_transformed
)


men_mlr_before <- lm(
  
  formula =
    mlr_formula,
  
  data =
    men_train_original
)


men_mlr_after <- lm(
  
  formula =
    mlr_formula,
  
  data =
    men_train_transformed
)


# ------------------------------------------------------------
# 22. RUNS TEST FUNCTION
# ------------------------------------------------------------

runs_test_residuals <- function(
    residual_values
) {
  
  signs <- ifelse(
    
    residual_values > 0,
    1,
    
    ifelse(
      residual_values < 0,
      0,
      NA
    )
  )
  
  
  signs <- signs[
    !is.na(signs)
  ]
  
  
  n_positive <-
    sum(
      signs == 1
    )
  
  n_negative <-
    sum(
      signs == 0
    )
  
  
  if (
    n_positive == 0 ||
    n_negative == 0
  ) {
    
    return(
      list(
        
        runs =
          NA_real_,
        
        expected_runs =
          NA_real_,
        
        z =
          NA_real_,
        
        p_value =
          NA_real_
      )
    )
  }
  
  
  number_of_runs <-
    
    1 +
    
    sum(
      signs[-1] !=
        signs[-length(signs)]
    )
  
  
  expected_runs <-
    
    1 +
    
    (
      2 *
        n_positive *
        n_negative
    ) /
    
    (
      n_positive +
        n_negative
    )
  
  
  variance_runs <-
    
    (
      2 *
        n_positive *
        n_negative *
        (
          2 *
            n_positive *
            n_negative -
            n_positive -
            n_negative
        )
    ) /
    
    (
      (
        n_positive +
          n_negative
      )^2 *
        
        (
          n_positive +
            n_negative -
            1
        )
    )
  
  
  z_value <-
    
    (
      number_of_runs -
        expected_runs
    ) /
    
    sqrt(
      variance_runs
    )
  
  
  p_value <-
    
    2 *
    pnorm(
      -abs(
        z_value
      )
    )
  
  
  return(
    
    list(
      
      runs =
        number_of_runs,
      
      expected_runs =
        expected_runs,
      
      z =
        z_value,
      
      p_value =
        p_value
    )
  )
}


# ------------------------------------------------------------
# 23. PRELIMINARY MLR DIAGNOSTIC FUNCTION
# ------------------------------------------------------------

run_mlr_diagnostics <- function(
    model,
    sex_name,
    stage
) {
  
  residual_values <-
    residuals(
      model
    )
  
  
  n <-
    length(
      residual_values
    )
  
  
  residual_mean <-
    mean(
      residual_values
    )
  
  
  residual_sd <-
    sd(
      residual_values
    )
  
  
  # Lilliefors
  
  lillie_result <-
    
    nortest::lillie.test(
      residual_values
    )
  
  
  # Breusch-Pagan
  
  bp_result <-
    
    lmtest::bptest(
      model
    )
  
  
  # Non-Constant Variance test
  
  ncv_result <- tryCatch(
    
    car::ncvTest(
      model
    ),
    
    error = function(e) {
      
      NULL
    }
  )
  
  
  # Durbin-Watson
  
  dw_result <- tryCatch(
    
    lmtest::dwtest(
      model
    ),
    
    error = function(e) {
      
      NULL
    }
  )
  
  
  # Runs test
  
  runs_result <-
    
    runs_test_residuals(
      residual_values
    )
  
  
  # Residual ACF
  
  acf_result <- acf(
    
    residual_values,
    
    plot =
      FALSE,
    
    lag.max =
      min(
        20,
        n - 1
      )
  )
  
  
  acf_values <-
    
    as.numeric(
      acf_result$acf
    )[-1]
  
  
  acf_reference_limit <-
    
    1.96 /
    sqrt(
      n
    )
  
  
  maximum_absolute_acf <-
    
    max(
      abs(
        acf_values
      ),
      na.rm =
        TRUE
    )
  
  
  number_acf_outside_reference <-
    
    sum(
      abs(
        acf_values
      ) >
        acf_reference_limit,
      na.rm =
        TRUE
    )
  
  
  # Prepare NCV output
  
  if (!is.null(ncv_result)) {
    
    ncv_chisquare <-
      unname(
        ncv_result$ChiSquare
      )
    
    ncv_df <-
      unname(
        ncv_result$Df
      )
    
    ncv_p <-
      unname(
        ncv_result$p
      )
    
  } else {
    
    ncv_chisquare <-
      NA_real_
    
    ncv_df <-
      NA_real_
    
    ncv_p <-
      NA_real_
  }
  
  
  # Prepare Durbin-Watson output
  
  if (!is.null(dw_result)) {
    
    dw_statistic <-
      unname(
        dw_result$statistic
      )
    
    dw_p <-
      dw_result$p.value
    
  } else {
    
    dw_statistic <-
      NA_real_
    
    dw_p <-
      NA_real_
  }
  
  
  data.frame(
    
    Sex =
      sex_name,
    
    Stage =
      stage,
    
    N =
      n,
    
    Number_of_predictors =
      length(
        predictors
      ),
    
    Residual_Mean =
      residual_mean,
    
    Residual_SD =
      residual_sd,
    
    Lilliefors_D =
      unname(
        lillie_result$statistic
      ),
    
    Lilliefors_p =
      lillie_result$p.value,
    
    Breusch_Pagan_statistic =
      unname(
        bp_result$statistic
      ),
    
    Breusch_Pagan_df =
      unname(
        bp_result$parameter
      ),
    
    Breusch_Pagan_p =
      bp_result$p.value,
    
    NCV_ChiSquare =
      ncv_chisquare,
    
    NCV_df =
      ncv_df,
    
    NCV_p =
      ncv_p,
    
    Durbin_Watson =
      dw_statistic,
    
    Durbin_Watson_p =
      dw_p,
    
    Runs =
      runs_result$runs,
    
    Expected_Runs =
      runs_result$expected_runs,
    
    Runs_Z =
      runs_result$z,
    
    Runs_p =
      runs_result$p_value,
    
    ACF_95pct_reference =
      acf_reference_limit,
    
    Maximum_absolute_ACF =
      maximum_absolute_acf,
    
    Number_ACF_outside_95pct_reference =
      number_acf_outside_reference
  )
}


# ------------------------------------------------------------
# 24. RUN BEFORE/AFTER MLR DIAGNOSTICS
# ------------------------------------------------------------

women_diagnostics_before <-
  
  run_mlr_diagnostics(
    
    model =
      women_mlr_before,
    
    sex_name =
      "Women",
    
    stage =
      "Before transformation"
  )


women_diagnostics_after <-
  
  run_mlr_diagnostics(
    
    model =
      women_mlr_after,
    
    sex_name =
      "Women",
    
    stage =
      "After selected transformations"
  )


men_diagnostics_before <-
  
  run_mlr_diagnostics(
    
    model =
      men_mlr_before,
    
    sex_name =
      "Men",
    
    stage =
      "Before transformation"
  )


men_diagnostics_after <-
  
  run_mlr_diagnostics(
    
    model =
      men_mlr_after,
    
    sex_name =
      "Men",
    
    stage =
      "After selected transformations"
  )


women_diagnostic_comparison <- bind_rows(
  
  women_diagnostics_before,
  
  women_diagnostics_after
)


men_diagnostic_comparison <- bind_rows(
  
  men_diagnostics_before,
  
  men_diagnostics_after
)


# ------------------------------------------------------------
# 25. SAVE BEFORE/AFTER DIAGNOSTIC TABLES
# ------------------------------------------------------------

write_csv(
  
  women_diagnostic_comparison,
  
  file.path(
    women_dir,
    "03_women_MLR_diagnostics_before_after.csv"
  )
)


write_csv(
  
  men_diagnostic_comparison,
  
  file.path(
    men_dir,
    "03_men_MLR_diagnostics_before_after.csv"
  )
)


# ------------------------------------------------------------
# 26. GVLMA BEFORE AND AFTER - WOMEN
# ------------------------------------------------------------

women_gvlma_before <- tryCatch(
  
  gvlma::gvlma(
    women_mlr_before
  ),
  
  error = function(e) {
    
    e
  }
)


women_gvlma_after <- tryCatch(
  
  gvlma::gvlma(
    women_mlr_after
  ),
  
  error = function(e) {
    
    e
  }
)


capture.output(
  
  {
    
    cat(
      "WOMEN - GVLMA BEFORE AND AFTER SELECTED TRANSFORMATIONS\n\n"
    )
    
    
    cat(
      "=============================================\n"
    )
    
    cat(
      "BEFORE TRANSFORMATION\n"
    )
    
    cat(
      "=============================================\n\n"
    )
    
    
    if (
      inherits(
        women_gvlma_before,
        "error"
      )
    ) {
      
      cat(
        women_gvlma_before$message
      )
      
    } else {
      
      print(
        summary(
          women_gvlma_before
        )
      )
    }
    
    
    cat(
      "\n\n=============================================\n"
    )
    
    cat(
      "AFTER SELECTED TRANSFORMATIONS\n"
    )
    
    cat(
      "=============================================\n\n"
    )
    
    
    if (
      inherits(
        women_gvlma_after,
        "error"
      )
    ) {
      
      cat(
        women_gvlma_after$message
      )
      
    } else {
      
      print(
        summary(
          women_gvlma_after
        )
      )
    }
  },
  
  file =
    file.path(
      women_dir,
      "04_women_GVLMA_before_after.txt"
    )
)


# ------------------------------------------------------------
# 27. GVLMA BEFORE AND AFTER - MEN
# ------------------------------------------------------------

men_gvlma_before <- tryCatch(
  
  gvlma::gvlma(
    men_mlr_before
  ),
  
  error = function(e) {
    
    e
  }
)


men_gvlma_after <- tryCatch(
  
  gvlma::gvlma(
    men_mlr_after
  ),
  
  error = function(e) {
    
    e
  }
)


capture.output(
  
  {
    
    cat(
      "MEN - GVLMA BEFORE AND AFTER SELECTED TRANSFORMATIONS\n\n"
    )
    
    
    cat(
      "=============================================\n"
    )
    
    cat(
      "BEFORE TRANSFORMATION\n"
    )
    
    cat(
      "=============================================\n\n"
    )
    
    
    if (
      inherits(
        men_gvlma_before,
        "error"
      )
    ) {
      
      cat(
        men_gvlma_before$message
      )
      
    } else {
      
      print(
        summary(
          men_gvlma_before
        )
      )
    }
    
    
    cat(
      "\n\n=============================================\n"
    )
    
    cat(
      "AFTER SELECTED TRANSFORMATIONS\n"
    )
    
    cat(
      "=============================================\n\n"
    )
    
    
    if (
      inherits(
        men_gvlma_after,
        "error"
      )
    ) {
      
      cat(
        men_gvlma_after$message
      )
      
    } else {
      
      print(
        summary(
          men_gvlma_after
        )
      )
    }
  },
  
  file =
    file.path(
      men_dir,
      "04_men_GVLMA_before_after.txt"
    )
)


# ------------------------------------------------------------
# 28. SAVE MLR SUMMARIES AFTER TRANSFORMATION
# ------------------------------------------------------------

capture.output(
  
  summary(
    women_mlr_after
  ),
  
  file =
    file.path(
      women_dir,
      "05_women_MLR_after_transformation_summary.txt"
    )
)


capture.output(
  
  summary(
    men_mlr_after
  ),
  
  file =
    file.path(
      men_dir,
      "05_men_MLR_after_transformation_summary.txt"
    )
)


# ------------------------------------------------------------
# 29. POST-TRANSFORMATION RESIDUAL HISTOGRAM
# ------------------------------------------------------------

save_residual_histogram <- function(
    model,
    sex_name,
    output_path
) {
  
  residual_data <- data.frame(
    
    Residual =
      residuals(
        model
      )
  )
  
  
  figure <- ggplot(
    
    residual_data,
    
    aes(
      x = Residual
    )
    
  ) +
    
    geom_histogram(
      bins = 20
    ) +
    
    labs(
      
      title =
        paste(
          "Residual Histogram After Selected Transformations -",
          sex_name
        ),
      
      x =
        "Residual",
      
      y =
        "Frequency"
    ) +
    
    theme_minimal(
      base_size = 11
    )
  
  
  ggsave(
    
    filename =
      output_path,
    
    plot =
      figure,
    
    width =
      7,
    
    height =
      5,
    
    units =
      "in",
    
    dpi =
      300
  )
}


# ------------------------------------------------------------
# 30. POST-TRANSFORMATION Q-Q PLOT
# ------------------------------------------------------------

save_qqplot <- function(
    model,
    sex_name,
    output_path
) {
  
  residual_data <- data.frame(
    
    Residual =
      residuals(
        model
      )
  )
  
  
  figure <- ggplot(
    
    residual_data,
    
    aes(
      sample = Residual
    )
    
  ) +
    
    stat_qq() +
    
    stat_qq_line() +
    
    labs(
      
      title =
        paste(
          "Normal Q-Q Plot After Selected Transformations -",
          sex_name
        ),
      
      x =
        "Theoretical Quantiles",
      
      y =
        "Sample Quantiles"
    ) +
    
    theme_minimal(
      base_size = 11
    )
  
  
  ggsave(
    
    filename =
      output_path,
    
    plot =
      figure,
    
    width =
      7,
    
    height =
      5,
    
    units =
      "in",
    
    dpi =
      300
  )
}


# ------------------------------------------------------------
# 31. POST-TRANSFORMATION SCALE-LOCATION PLOT
# ------------------------------------------------------------

save_scale_location <- function(
    model,
    sex_name,
    output_path
) {
  
  plot_data <- data.frame(
    
    Fitted =
      fitted(
        model
      ),
    
    Sqrt_Abs_Standardized_Residual =
      sqrt(
        abs(
          rstandard(
            model
          )
        )
      )
  )
  
  
  figure <- ggplot(
    
    plot_data,
    
    aes(
      
      x =
        Fitted,
      
      y =
        Sqrt_Abs_Standardized_Residual
    )
    
  ) +
    
    geom_point(
      size = 1.5
    ) +
    
    geom_smooth(
      
      method =
        "loess",
      
      formula =
        y ~ x,
      
      se =
        FALSE
    ) +
    
    labs(
      
      title =
        paste(
          "Scale-Location Plot After Selected Transformations -",
          sex_name
        ),
      
      x =
        "Fitted Values",
      
      y =
        "Square root of absolute standardized residuals"
    ) +
    
    theme_minimal(
      base_size = 11
    )
  
  
  ggsave(
    
    filename =
      output_path,
    
    plot =
      figure,
    
    width =
      7,
    
    height =
      5,
    
    units =
      "in",
    
    dpi =
      300
  )
}


# ------------------------------------------------------------
# 32. POST-TRANSFORMATION RESIDUAL ACF
# ------------------------------------------------------------

save_residual_acf <- function(
    model,
    sex_name,
    output_path
) {
  
  png(
    
    filename =
      output_path,
    
    width =
      7,
    
    height =
      5,
    
    units =
      "in",
    
    res =
      300
  )
  
  
  acf(
    
    residuals(
      model
    ),
    
    main =
      paste(
        "Residual ACF After Selected Transformations -",
        sex_name
      )
  )
  
  
  dev.off()
}


# ------------------------------------------------------------
# 33. GENERATE WOMEN POST-TRANSFORMATION FIGURES
# ------------------------------------------------------------

save_residual_histogram(
  
  women_mlr_after,
  
  "Women",
  
  file.path(
    women_figures_dir,
    "residuals_histogram_women_after.png"
  )
)


save_qqplot(
  
  women_mlr_after,
  
  "Women",
  
  file.path(
    women_figures_dir,
    "qqplot_women_after.png"
  )
)


save_scale_location(
  
  women_mlr_after,
  
  "Women",
  
  file.path(
    women_figures_dir,
    "scale_location_women_after.png"
  )
)


save_residual_acf(
  
  women_mlr_after,
  
  "Women",
  
  file.path(
    women_figures_dir,
    "residuals_acf_women_after.png"
  )
)


# ------------------------------------------------------------
# 34. GENERATE MEN POST-TRANSFORMATION FIGURES
# ------------------------------------------------------------

save_residual_histogram(
  
  men_mlr_after,
  
  "Men",
  
  file.path(
    men_figures_dir,
    "residuals_histogram_men_after.png"
  )
)


save_qqplot(
  
  men_mlr_after,
  
  "Men",
  
  file.path(
    men_figures_dir,
    "qqplot_men_after.png"
  )
)


save_scale_location(
  
  men_mlr_after,
  
  "Men",
  
  file.path(
    men_figures_dir,
    "scale_location_men_after.png"
  )
)


save_residual_acf(
  
  men_mlr_after,
  
  "Men",
  
  file.path(
    men_figures_dir,
    "residuals_acf_men_after.png"
  )
)


# ------------------------------------------------------------
# 35. CREATE ANALYSIS DOCUMENTATION FILE
# ------------------------------------------------------------

documentation <- c(
  
  "SCRIPT 04 - SELECTED PREDICTOR TRANSFORMATIONS",
  
  "",
  
  "TRAIN data only.",
  
  "",
  
  "FINAL WOMEN TRANSFORMATIONS:",
  
  "HDL -> Natural logarithm",
  "PLT -> Natural logarithm",
  "TRG -> Natural logarithm",
  "WBC -> Natural logarithm",
  "LDL -> Square root",
  
  "",
  
  "All other women predictors remain on their original scale.",
  
  "",
  
  "FINAL MEN TRANSFORMATIONS:",
  
  "HDL -> Natural logarithm",
  "TRG -> Natural logarithm",
  "VisceralFat -> Natural logarithm",
  "WBC -> Natural logarithm",
  
  "",
  
  "All other men predictors remain on their original scale.",
  
  "",
  
  "HCY remains on its original measurement scale in both sexes.",
  
  "",
  
  "Transformation decisions were based exclusively on TRAIN data.",
  
  "",
  
  "The transformation assessment considered:",
  "- descriptive skewness and kurtosis",
  "- raw-data boxplots",
  "- preliminary MLR diagnostics",
  "- alternative transformation statistics",
  "- graphical comparison of transformed distributions",
  "- biological plausibility",
  "- preference for simple monotonic transformations",
  
  "",
  
  "No transformation was selected solely to force predictor normality.",
  
  "",
  
  "Yeo-Johnson was evaluated during the assessment stage but was not",
  "retained in the final transformation strategy.",
  
  "",
  
  "TEST data were not accessed in this script.",
  
  "No centering or scaling was performed.",
  
  "No Elastic Net predictor selection was performed.",
  
  "No final predictive performance was estimated."
)


writeLines(
  
  documentation,
  
  con =
    file.path(
      output_dir,
      "00_READ_ME_FIRST.txt"
    )
)


# ------------------------------------------------------------
# 36. DISPLAY MAIN RESULTS
# ------------------------------------------------------------

cat(
  "\n\n=============================================\n",
  "FINAL TRANSFORMATION PLAN\n",
  "=============================================\n"
)


print(
  as_tibble(
    documented_transformation_plan
  ),
  n = Inf
)


cat(
  "\n\n=============================================\n",
  "WOMEN - DISTRIBUTION BEFORE / AFTER\n",
  "=============================================\n"
)


print(
  as_tibble(
    women_distribution_comparison
  ),
  n = Inf
)


cat(
  "\n\n=============================================\n",
  "MEN - DISTRIBUTION BEFORE / AFTER\n",
  "=============================================\n"
)


print(
  as_tibble(
    men_distribution_comparison
  ),
  n = Inf
)


cat(
  "\n\n=============================================\n",
  "WOMEN - MLR DIAGNOSTICS BEFORE / AFTER\n",
  "=============================================\n"
)


print(
  women_diagnostic_comparison
)


cat(
  "\n\n=============================================\n",
  "MEN - MLR DIAGNOSTICS BEFORE / AFTER\n",
  "=============================================\n"
)


print(
  men_diagnostic_comparison
)


# ------------------------------------------------------------
# 37. FINAL MESSAGE
# ------------------------------------------------------------

cat(
  "\n\n=============================================\n",
  "SCRIPT 04 COMPLETED SUCCESSFULLY\n",
  "=============================================\n",
  "\nSelected predictor transformations applied.",
  "\nHCY remained unchanged.",
  "\nTransformed TRAIN datasets saved.",
  "\nBefore/after distribution summaries saved.",
  "\nBefore/after MLR diagnostics saved.",
  "\nBefore/after GVLMA results saved.",
  "\nPost-transformation figures saved as 300-dpi PNG files.",
  "\n",
  "\nTEST data were NOT accessed.",
  "\nNo centering or scaling was performed.",
  "\nNo Elastic Net was performed.",
  "\n",
  "\nAll results were saved in:",
  "\n",
  normalizePath(
    output_dir,
    winslash = "/",
    mustWork = FALSE
  ),
  "\n",
  "\n=============================================\n"
)
