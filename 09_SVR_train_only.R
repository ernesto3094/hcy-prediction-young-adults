# ============================================================
# SCRIPT 09
# SUPPORT VECTOR REGRESSION - TRAIN ONLY
#
# FINAL CLEAN VERSION
#
# Models:
#   1. Linear SVR
#   2. Polynomial SVR
#   3. Radial Basis Function SVR
#
# METHODOLOGICAL DESIGN:
# - TRAIN only.
# - TEST is never read or accessed.
# - Same 10 seeds used in MLR and PR.
# - Same grouped HCY-stratified 10-fold CV procedure.
# - Duplicate IDs remain in the same fold.
# - Elastic Net-selected predictors are used.
# - Transformed predictors are used.
# - Centering/scaling is estimated inside each CV training fold.
# - HCY remains on its original scale.
# - Hyperparameters are selected by minimum mean OOF RMSE
#   across the 10 CV configurations.
# - Official predictive R2 = 1 - SSE/SST.
# - No post hoc expansion of the predefined tuning grids.
# ============================================================


# ============================================================
# 1. PACKAGES
# ============================================================

required_packages <- c(
  "caret",
  "kernlab",
  "readr",
  "dplyr"
)

missing_packages <- required_packages[
  !required_packages %in% rownames(installed.packages())
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(caret)
library(kernlab)
library(readr)
library(dplyr)


# ============================================================
# 2. DIRECTORIES
# ============================================================

input_dir <- "05_elastic_net_results"
output_dir <- "09_SVR_results"

if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}


# ============================================================
# 3. INPUT FILES
# ============================================================

women_train_file <- file.path(
  input_dir,
  "01_women_TRAIN_transformed.csv"
)

men_train_file <- file.path(
  input_dir,
  "02_men_TRAIN_transformed.csv"
)

women_selected_file <- file.path(
  input_dir,
  "10_women_final_selected_predictors.csv"
)

men_selected_file <- file.path(
  input_dir,
  "20_men_final_selected_predictors.csv"
)


required_files <- c(
  women_train_file,
  men_train_file,
  women_selected_file,
  men_selected_file
)

missing_files <- required_files[
  !file.exists(required_files)
]

if (length(missing_files) > 0) {
  
  stop(
    paste(
      "Missing required files:",
      paste(
        missing_files,
        collapse = "\n"
      )
    )
  )
}


# ============================================================
# 4. READ TRAIN DATA
# ============================================================

women_train <- read_csv(
  women_train_file,
  show_col_types = FALSE
)

men_train <- read_csv(
  men_train_file,
  show_col_types = FALSE
)

women_selected <- read_csv(
  women_selected_file,
  show_col_types = FALSE
)

men_selected <- read_csv(
  men_selected_file,
  show_col_types = FALSE
)


# ============================================================
# 5. FINAL ELASTIC NET-SELECTED PREDICTORS
# ============================================================

if (!"Variable" %in% names(women_selected)) {
  stop(
    "Column 'Variable' not found in WOMEN selection file."
  )
}

if (!"Variable" %in% names(men_selected)) {
  stop(
    "Column 'Variable' not found in MEN selection file."
  )
}


women_predictors <- women_selected$Variable
men_predictors <- men_selected$Variable


expected_women <- c(
  "LDL",
  "BMI",
  "GLU",
  "VisceralFat",
  "MCHC"
)

expected_men <- c(
  "BMI",
  "HDL"
)


if (!setequal(women_predictors, expected_women)) {
  
  stop(
    paste(
      "Unexpected WOMEN predictor set:",
      paste(
        women_predictors,
        collapse = ", "
      )
    )
  )
}


if (!setequal(men_predictors, expected_men)) {
  
  stop(
    paste(
      "Unexpected MEN predictor set:",
      paste(
        men_predictors,
        collapse = ", "
      )
    )
  )
}


cat("\n========================================\n")
cat("FINAL PREDICTORS FOR SVR\n")
cat("========================================\n")

cat("\nWOMEN:\n")
print(women_predictors)

cat("\nMEN:\n")
print(men_predictors)


# ============================================================
# 6. VERIFY SAMPLE SIZES
# ============================================================

if (nrow(women_train) != 190) {
  
  stop(
    paste(
      "Unexpected WOMEN TRAIN size:",
      nrow(women_train),
      "- expected 190."
    )
  )
}


if (nrow(men_train) != 83) {
  
  stop(
    paste(
      "Unexpected MEN TRAIN size:",
      nrow(men_train),
      "- expected 83."
    )
  )
}


# ============================================================
# 7. DATA VALIDATION
# ============================================================

check_dataset <- function(
    data,
    predictors,
    sex_name
) {
  
  required_columns <- c(
    "ID",
    "HCY",
    predictors
  )
  
  
  missing_columns <- setdiff(
    required_columns,
    names(data)
  )
  
  
  if (length(missing_columns) > 0) {
    
    stop(
      paste(
        sex_name,
        "is missing columns:",
        paste(
          missing_columns,
          collapse = ", "
        )
      )
    )
  }
  
  
  for (variable in c(
    "HCY",
    predictors
  )) {
    
    if (!is.numeric(data[[variable]])) {
      
      stop(
        paste(
          sex_name,
          variable,
          "is not numeric."
        )
      )
    }
    
    
    if (any(is.na(data[[variable]]))) {
      
      stop(
        paste(
          "Missing values detected in",
          sex_name,
          variable
        )
      )
    }
    
    
    if (any(!is.finite(data[[variable]]))) {
      
      stop(
        paste(
          "Non-finite values detected in",
          sex_name,
          variable
        )
      )
    }
  }
  
  
  invisible(TRUE)
}


check_dataset(
  women_train,
  women_predictors,
  "WOMEN"
)

check_dataset(
  men_train,
  men_predictors,
  "MEN"
)


# ============================================================
# 8. CV SETTINGS
# ============================================================

cv_seeds <- c(
  123,
  456,
  789,
  1024,
  2048,
  4096,
  7777,
  12021,
  31415,
  27182
)

number_folds <- 10


# ============================================================
# 9. PREDEFINED SVR TUNING GRIDS
# ============================================================

# ------------------------------------------------------------
# Linear SVR
#
# C = 2^-4 ... 2^4
# ------------------------------------------------------------

linear_grid <- expand.grid(
  
  C = 2^seq(
    -4,
    4,
    by = 1
  )
)


# ------------------------------------------------------------
# Polynomial SVR
#
# degree = 2 ... 8
# scale  = 0.001, 0.01, 0.1, 1
# C      = 0.25 ... 16
# ------------------------------------------------------------

polynomial_grid <- expand.grid(
  
  degree = 2:8,
  
  scale = c(
    0.001,
    0.01,
    0.1,
    1
  ),
  
  C = c(
    0.25,
    0.5,
    1,
    2,
    4,
    8,
    16
  )
)


# ------------------------------------------------------------
# Radial SVR
#
# sigma = 2^-10 ... 2^-1
# C     = 2^-4 ... 2^8
# ------------------------------------------------------------

radial_grid <- expand.grid(
  
  sigma = 2^seq(
    -10,
    -1,
    by = 1
  ),
  
  C = 2^seq(
    -4,
    8,
    by = 1
  )
)


cat("\n========================================\n")
cat("SVR TUNING GRIDS\n")
cat("========================================\n")

cat(
  "Linear combinations:",
  nrow(linear_grid),
  "\n"
)

cat(
  "Polynomial combinations:",
  nrow(polynomial_grid),
  "\n"
)

cat(
  "Radial combinations:",
  nrow(radial_grid),
  "\n"
)


# ============================================================
# 10. CREATE GROUPED AND HCY-STRATIFIED CV FOLDS
#
# IMPORTANT:
# This is the same deterministic procedure used for MLR and PR.
#
# Folds are generated in memory only.
# ============================================================

create_cv_folds <- function(
    data,
    seeds,
    k = 10
) {
  
  data_internal <- data %>%
    mutate(
      
      Internal_Row =
        seq_len(n()),
      
      ID =
        as.character(ID)
    )
  
  
  # ----------------------------------------------------------
  # Create one grouping unit per unique ID.
  #
  # If an ID appears more than once, all observations with that
  # ID are assigned to the same CV fold.
  # ----------------------------------------------------------
  
  id_groups <- data_internal %>%
    group_by(ID) %>%
    summarise(
      
      Group_HCY =
        mean(HCY),
      
      .groups =
        "drop"
    ) %>%
    arrange(
      Group_HCY
    ) %>%
    mutate(
      
      HCY_Stratum =
        ntile(
          Group_HCY,
          5
        )
    )
  
  
  all_assignments <- list()
  
  
  for (seed_index in seq_along(seeds)) {
    
    seed_value <-
      seeds[seed_index]
    
    
    set.seed(
      seed_value
    )
    
    
    current_groups <-
      id_groups
    
    
    current_groups$Fold <-
      NA_integer_
    
    
    strata <- sort(
      unique(
        current_groups$HCY_Stratum
      )
    )
    
    
    for (stratum_value in strata) {
      
      stratum_rows <- which(
        current_groups$HCY_Stratum ==
          stratum_value
      )
      
      
      shuffled_rows <- sample(
        stratum_rows,
        size =
          length(stratum_rows),
        replace =
          FALSE
      )
      
      
      fold_labels <- rep(
        seq_len(k),
        length.out =
          length(shuffled_rows)
      )
      
      
      current_groups$Fold[
        shuffled_rows
      ] <- fold_labels
    }
    
    
    current_assignment <- data_internal %>%
      select(
        Internal_Row,
        ID
      ) %>%
      left_join(
        
        current_groups %>%
          select(
            ID,
            HCY_Stratum,
            Fold
          ),
        
        by =
          "ID"
      )
    
    
    current_assignment$Seed <-
      seed_value
    
    
    current_assignment <- current_assignment %>%
      select(
        Seed,
        Fold,
        Internal_Row,
        ID,
        HCY_Stratum
      )
    
    
    all_assignments[
      [length(all_assignments) + 1]
    ] <- current_assignment
  }
  
  
  assignments <- bind_rows(
    all_assignments
  )
  
  
  # ----------------------------------------------------------
  # Internal validation
  # ----------------------------------------------------------
  
  expected_rows <-
    nrow(data) *
    length(seeds)
  
  
  if (nrow(assignments) != expected_rows) {
    
    stop(
      "Unexpected number of CV assignment rows."
    )
  }
  
  
  if (any(is.na(assignments$Fold))) {
    
    stop(
      "Missing CV fold assignment detected."
    )
  }
  
  
  duplicate_check <- assignments %>%
    group_by(
      Seed,
      ID
    ) %>%
    summarise(
      
      Number_of_Folds =
        n_distinct(Fold),
      
      .groups =
        "drop"
    )
  
  
  if (
    any(
      duplicate_check$Number_of_Folds != 1
    )
  ) {
    
    stop(
      "A duplicated ID was assigned to different folds."
    )
  }
  
  
  fold_check <- assignments %>%
    group_by(
      Seed
    ) %>%
    summarise(
      
      Number_of_Folds =
        n_distinct(Fold),
      
      .groups =
        "drop"
    )
  
  
  if (
    any(
      fold_check$Number_of_Folds != k
    )
  ) {
    
    stop(
      "At least one seed does not contain exactly 10 folds."
    )
  }
  
  
  return(
    assignments
  )
}


# ============================================================
# 11. CONVERT FOLDS TO CARET INDICES
# ============================================================

create_caret_indices <- function(
    data,
    assignments,
    seeds,
    k = 10
) {
  
  train_indices <- list()
  validation_indices <- list()
  
  
  for (seed_value in seeds) {
    
    seed_assignments <- assignments[
      assignments$Seed ==
        seed_value,
      ,
      drop = FALSE
    ]
    
    
    for (fold_value in seq_len(k)) {
      
      resample_name <- sprintf(
        "S%d_F%02d",
        seed_value,
        fold_value
      )
      
      
      validation_rows <-
        seed_assignments$Internal_Row[
          seed_assignments$Fold ==
            fold_value
        ]
      
      
      validation_rows <-
        as.integer(
          validation_rows
        )
      
      
      training_rows <- setdiff(
        seq_len(
          nrow(data)
        ),
        validation_rows
      )
      
      
      train_indices[[resample_name]] <-
        as.integer(
          training_rows
        )
      
      
      validation_indices[[resample_name]] <-
        as.integer(
          validation_rows
        )
    }
  }
  
  
  return(
    list(
      
      index =
        train_indices,
      
      indexOut =
        validation_indices
    )
  )
}


# ============================================================
# 12. OFFICIAL PERFORMANCE METRICS
# ============================================================

calculate_rmse <- function(
    observed,
    predicted
) {
  
  sqrt(
    mean(
      (
        observed -
          predicted
      )^2
    )
  )
}


calculate_mae <- function(
    observed,
    predicted
) {
  
  mean(
    abs(
      observed -
        predicted
    )
  )
}


calculate_predictive_r2 <- function(
    observed,
    predicted
) {
  
  sse <- sum(
    (
      observed -
        predicted
    )^2
  )
  
  
  sst <- sum(
    (
      observed -
        mean(observed)
    )^2
  )
  
  
  if (sst == 0) {
    
    return(
      NA_real_
    )
  }
  
  
  return(
    1 -
      (
        sse /
          sst
      )
  )
}


# ============================================================
# 13. STANDARDIZE SAVED CARET PREDICTIONS
# ============================================================

prepare_prediction_table <- function(
    fit,
    sex_name,
    kernel_name
) {
  
  predictions <-
    fit$pred
  
  
  if (is.null(predictions)) {
    
    stop(
      paste(
        "No saved CV predictions for",
        sex_name,
        kernel_name
      )
    )
  }
  
  
  predictions$Sex <-
    sex_name
  
  
  predictions$Kernel <-
    kernel_name
  
  
  # ----------------------------------------------------------
  # Standard columns across the three SVR kernels
  # ----------------------------------------------------------
  
  predictions$Degree <-
    NA_real_
  
  predictions$Scale <-
    NA_real_
  
  predictions$Sigma <-
    NA_real_
  
  
  if ("degree" %in% names(predictions)) {
    
    predictions$Degree <-
      predictions$degree
  }
  
  
  if ("scale" %in% names(predictions)) {
    
    predictions$Scale <-
      predictions$scale
  }
  
  
  if ("sigma" %in% names(predictions)) {
    
    predictions$Sigma <-
      predictions$sigma
  }
  
  
  # ----------------------------------------------------------
  # Recover the original CV seed from the resample name
  # ----------------------------------------------------------
  
  predictions$Seed <- as.integer(
    
    sub(
      "^S([0-9]+)_F[0-9]+$",
      "\\1",
      predictions$Resample
    )
  )
  
  
  if (any(is.na(predictions$Seed))) {
    
    stop(
      "Seed extraction failed from CV resample names."
    )
  }
  
  
  return(
    
    predictions %>%
      select(
        Sex,
        Kernel,
        Degree,
        Scale,
        Sigma,
        C,
        Seed,
        Resample,
        rowIndex,
        obs,
        pred
      )
  )
}


# ============================================================
# 14. CALCULATE OOF METRICS FOR EACH HYPERPARAMETER COMBINATION
#
# Each observation receives exactly one out-of-fold prediction
# for each seed and each hyperparameter configuration.
# ============================================================

calculate_tuning_metrics <- function(
    prediction_table
) {
  
  result <- prediction_table %>%
    
    group_by(
      Sex,
      Kernel,
      Degree,
      Scale,
      Sigma,
      C,
      Seed
    ) %>%
    
    summarise(
      
      N =
        n(),
      
      RMSE =
        calculate_rmse(
          observed =
            obs,
          predicted =
            pred
        ),
      
      MAE =
        calculate_mae(
          observed =
            obs,
          predicted =
            pred
        ),
      
      Predictive_R2 =
        calculate_predictive_r2(
          observed =
            obs,
          predicted =
            pred
        ),
      
      .groups =
        "drop"
    )
  
  
  return(
    result
  )
}


# ============================================================
# 15. TUNE ONE SVR KERNEL
# ============================================================

tune_svr_kernel <- function(
    data,
    predictors,
    sex_name,
    method_name,
    kernel_name,
    tuning_grid,
    cv_indices
) {
  
  cat("\n========================================\n")
  cat(
    "TUNING",
    sex_name,
    "-",
    kernel_name,
    "\n"
  )
  cat("========================================\n")
  
  
  model_formula <- as.formula(
    paste(
      "HCY ~",
      paste(
        predictors,
        collapse = " + "
      )
    )
  )
  
  
  control <- trainControl(
    
    method =
      "cv",
    
    index =
      cv_indices$index,
    
    indexOut =
      cv_indices$indexOut,
    
    savePredictions =
      "all",
    
    returnResamp =
      "all",
    
    allowParallel =
      FALSE
  )
  
  
  set.seed(123)
  
  
  fit <- train(
    
    model_formula,
    
    data =
      data,
    
    method =
      method_name,
    
    trControl =
      control,
    
    preProcess = c(
      "center",
      "scale"
    ),
    
    tuneGrid =
      tuning_grid,
    
    metric =
      "RMSE"
  )
  
  
  prediction_table <-
    prepare_prediction_table(
      
      fit =
        fit,
      
      sex_name =
        sex_name,
      
      kernel_name =
        kernel_name
    )
  
  
  tuning_metrics <-
    calculate_tuning_metrics(
      prediction_table
    )
  
  
  rm(
    fit,
    prediction_table
  )
  
  gc()
  
  
  return(
    tuning_metrics
  )
}


# ============================================================
# 16. PREPARE WOMEN CV FOLDS
# ============================================================

women_assignments <- create_cv_folds(
  
  data =
    women_train,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds
)


women_indices <- create_caret_indices(
  
  data =
    women_train,
  
  assignments =
    women_assignments,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds
)


# ============================================================
# 17. TUNE WOMEN SVR - LINEAR
# ============================================================

women_linear <- tune_svr_kernel(
  
  data =
    women_train,
  
  predictors =
    women_predictors,
  
  sex_name =
    "Women",
  
  method_name =
    "svmLinear",
  
  kernel_name =
    "Linear",
  
  tuning_grid =
    linear_grid,
  
  cv_indices =
    women_indices
)


# ============================================================
# 18. TUNE WOMEN SVR - POLYNOMIAL
# ============================================================

women_polynomial <- tune_svr_kernel(
  
  data =
    women_train,
  
  predictors =
    women_predictors,
  
  sex_name =
    "Women",
  
  method_name =
    "svmPoly",
  
  kernel_name =
    "Polynomial",
  
  tuning_grid =
    polynomial_grid,
  
  cv_indices =
    women_indices
)


# ============================================================
# 19. TUNE WOMEN SVR - RADIAL
# ============================================================

women_radial <- tune_svr_kernel(
  
  data =
    women_train,
  
  predictors =
    women_predictors,
  
  sex_name =
    "Women",
  
  method_name =
    "svmRadial",
  
  kernel_name =
    "Radial",
  
  tuning_grid =
    radial_grid,
  
  cv_indices =
    women_indices
)


rm(
  women_assignments,
  women_indices
)

gc()


# ============================================================
# 20. PREPARE MEN CV FOLDS
# ============================================================

men_assignments <- create_cv_folds(
  
  data =
    men_train,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds
)


men_indices <- create_caret_indices(
  
  data =
    men_train,
  
  assignments =
    men_assignments,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds
)


# ============================================================
# 21. TUNE MEN SVR - LINEAR
# ============================================================

men_linear <- tune_svr_kernel(
  
  data =
    men_train,
  
  predictors =
    men_predictors,
  
  sex_name =
    "Men",
  
  method_name =
    "svmLinear",
  
  kernel_name =
    "Linear",
  
  tuning_grid =
    linear_grid,
  
  cv_indices =
    men_indices
)


# ============================================================
# 22. TUNE MEN SVR - POLYNOMIAL
# ============================================================

men_polynomial <- tune_svr_kernel(
  
  data =
    men_train,
  
  predictors =
    men_predictors,
  
  sex_name =
    "Men",
  
  method_name =
    "svmPoly",
  
  kernel_name =
    "Polynomial",
  
  tuning_grid =
    polynomial_grid,
  
  cv_indices =
    men_indices
)


# ============================================================
# 23. TUNE MEN SVR - RADIAL
# ============================================================

men_radial <- tune_svr_kernel(
  
  data =
    men_train,
  
  predictors =
    men_predictors,
  
  sex_name =
    "Men",
  
  method_name =
    "svmRadial",
  
  kernel_name =
    "Radial",
  
  tuning_grid =
    radial_grid,
  
  cv_indices =
    men_indices
)


rm(
  men_assignments,
  men_indices
)

gc()


# ============================================================
# 24. COMBINE ALL SVR TUNING RESULTS
# ============================================================

all_tuning_by_seed <- bind_rows(
  
  women_linear,
  
  women_polynomial,
  
  women_radial,
  
  men_linear,
  
  men_polynomial,
  
  men_radial
)


# ============================================================
# 25. AGGREGATE TUNING RESULTS ACROSS THE 10 SEEDS
# ============================================================

tuning_summary <- all_tuning_by_seed %>%
  
  group_by(
    Sex,
    Kernel,
    Degree,
    Scale,
    Sigma,
    C
  ) %>%
  
  summarise(
    
    Mean_RMSE =
      mean(RMSE),
    
    SD_RMSE =
      sd(RMSE),
    
    Mean_MAE =
      mean(MAE),
    
    SD_MAE =
      sd(MAE),
    
    Mean_Predictive_R2 =
      mean(
        Predictive_R2
      ),
    
    SD_Predictive_R2 =
      sd(
        Predictive_R2
      ),
    
    .groups =
      "drop"
  )


# ============================================================
# 26. BEST HYPERPARAMETERS WITHIN EACH SEED
#
# Descriptive tuning-stability analysis.
# ============================================================

best_by_seed <- all_tuning_by_seed %>%
  
  group_by(
    Sex,
    Kernel,
    Seed
  ) %>%
  
  arrange(
    RMSE,
    Degree,
    C,
    Scale,
    Sigma,
    .by_group = TRUE
  ) %>%
  
  slice(1) %>%
  
  ungroup()


# ============================================================
# 27. SELECT FINAL HYPERPARAMETERS
#
# Primary criterion:
# Lowest mean OOF RMSE across the 10 CV configurations.
#
# The predefined tuning grids are not expanded after examining
# these results.
# ============================================================

selected_hyperparameters <- tuning_summary %>%
  
  group_by(
    Sex,
    Kernel
  ) %>%
  
  arrange(
    Mean_RMSE,
    Degree,
    C,
    Scale,
    Sigma,
    .by_group = TRUE
  ) %>%
  
  slice(1) %>%
  
  ungroup()


cat("\n\n========================================\n")
cat("FINAL SVR HYPERPARAMETERS\n")
cat("========================================\n")

print(
  selected_hyperparameters
)


# ============================================================
# 28. FINAL CV PERFORMANCE BY SEED
# ============================================================

selected_keys <- selected_hyperparameters %>%
  
  select(
    Sex,
    Kernel,
    Degree,
    Scale,
    Sigma,
    C
  )


final_cv_by_seed <- all_tuning_by_seed %>%
  
  inner_join(
    
    selected_keys,
    
    by = c(
      "Sex",
      "Kernel",
      "Degree",
      "Scale",
      "Sigma",
      "C"
    )
  )


# ============================================================
# 29. FINAL INTERNAL CV SUMMARY
# ============================================================

final_cv_summary <- final_cv_by_seed %>%
  
  group_by(
    Sex,
    Kernel,
    Degree,
    Scale,
    Sigma,
    C
  ) %>%
  
  summarise(
    
    Mean_RMSE =
      mean(RMSE),
    
    SD_RMSE =
      sd(RMSE),
    
    Min_RMSE =
      min(RMSE),
    
    Max_RMSE =
      max(RMSE),
    
    Mean_MAE =
      mean(MAE),
    
    SD_MAE =
      sd(MAE),
    
    Min_MAE =
      min(MAE),
    
    Max_MAE =
      max(MAE),
    
    Mean_Predictive_R2 =
      mean(
        Predictive_R2
      ),
    
    SD_Predictive_R2 =
      sd(
        Predictive_R2
      ),
    
    Min_Predictive_R2 =
      min(
        Predictive_R2
      ),
    
    Max_Predictive_R2 =
      max(
        Predictive_R2
      ),
    
    .groups =
      "drop"
  )


# ============================================================
# 30. CREATE FINAL ONE-ROW TUNING GRID
# ============================================================

create_final_grid <- function(
    selected_row,
    kernel_name
) {
  
  if (kernel_name == "Linear") {
    
    return(
      data.frame(
        C =
          selected_row$C
      )
    )
  }
  
  
  if (kernel_name == "Polynomial") {
    
    return(
      data.frame(
        
        degree =
          selected_row$Degree,
        
        scale =
          selected_row$Scale,
        
        C =
          selected_row$C
      )
    )
  }
  
  
  if (kernel_name == "Radial") {
    
    return(
      data.frame(
        
        sigma =
          selected_row$Sigma,
        
        C =
          selected_row$C
      )
    )
  }
  
  
  stop(
    "Unknown SVR kernel."
  )
}


# ============================================================
# 31. FIT FINAL FULL-TRAIN SVR MODEL
# ============================================================

fit_final_svr <- function(
    data,
    predictors,
    sex_name,
    kernel_name,
    selected_row
) {
  
  method_name <- switch(
    
    kernel_name,
    
    "Linear" =
      "svmLinear",
    
    "Polynomial" =
      "svmPoly",
    
    "Radial" =
      "svmRadial"
  )
  
  
  final_grid <- create_final_grid(
    
    selected_row =
      selected_row,
    
    kernel_name =
      kernel_name
  )
  
  
  model_formula <- as.formula(
    paste(
      "HCY ~",
      paste(
        predictors,
        collapse = " + "
      )
    )
  )
  
  
  final_control <- trainControl(
    method =
      "none"
  )
  
  
  set.seed(123)
  
  
  final_model <- train(
    
    model_formula,
    
    data =
      data,
    
    method =
      method_name,
    
    preProcess = c(
      "center",
      "scale"
    ),
    
    tuneGrid =
      final_grid,
    
    trControl =
      final_control
  )
  
  
  train_predictions <- predict(
    
    final_model,
    
    newdata =
      data
  )
  
  
  train_metrics <- data.frame(
    
    Sex =
      sex_name,
    
    Kernel =
      kernel_name,
    
    Degree =
      ifelse(
        kernel_name ==
          "Polynomial",
        selected_row$Degree,
        NA_real_
      ),
    
    Scale =
      ifelse(
        kernel_name ==
          "Polynomial",
        selected_row$Scale,
        NA_real_
      ),
    
    Sigma =
      ifelse(
        kernel_name ==
          "Radial",
        selected_row$Sigma,
        NA_real_
      ),
    
    C =
      selected_row$C,
    
    N =
      nrow(data),
    
    RMSE =
      calculate_rmse(
        observed =
          data$HCY,
        predicted =
          train_predictions
      ),
    
    MAE =
      calculate_mae(
        observed =
          data$HCY,
        predicted =
          train_predictions
      ),
    
    Predictive_R2 =
      calculate_predictive_r2(
        observed =
          data$HCY,
        predicted =
          train_predictions
      )
  )
  
  
  return(
    list(
      
      model =
        final_model,
      
      metrics =
        train_metrics
    )
  )
}


# ============================================================
# 32. FIT FINAL WOMEN SVR MODELS
# ============================================================

women_final_models <- list()
women_train_metrics <- list()


for (
  kernel_name in c(
    "Linear",
    "Polynomial",
    "Radial"
  )
) {
  
  selected_row <- selected_hyperparameters %>%
    
    filter(
      Sex == "Women",
      Kernel == kernel_name
    )
  
  
  if (nrow(selected_row) != 1) {
    
    stop(
      paste(
        "Unexpected number of selected rows for Women",
        kernel_name
      )
    )
  }
  
  
  final_fit <- fit_final_svr(
    
    data =
      women_train,
    
    predictors =
      women_predictors,
    
    sex_name =
      "Women",
    
    kernel_name =
      kernel_name,
    
    selected_row =
      selected_row
  )
  
  
  women_final_models[[kernel_name]] <-
    final_fit$model
  
  
  women_train_metrics[[kernel_name]] <-
    final_fit$metrics
}


# ============================================================
# 33. FIT FINAL MEN SVR MODELS
# ============================================================

men_final_models <- list()
men_train_metrics <- list()


for (
  kernel_name in c(
    "Linear",
    "Polynomial",
    "Radial"
  )
) {
  
  selected_row <- selected_hyperparameters %>%
    
    filter(
      Sex == "Men",
      Kernel == kernel_name
    )
  
  
  if (nrow(selected_row) != 1) {
    
    stop(
      paste(
        "Unexpected number of selected rows for Men",
        kernel_name
      )
    )
  }
  
  
  final_fit <- fit_final_svr(
    
    data =
      men_train,
    
    predictors =
      men_predictors,
    
    sex_name =
      "Men",
    
    kernel_name =
      kernel_name,
    
    selected_row =
      selected_row
  )
  
  
  men_final_models[[kernel_name]] <-
    final_fit$model
  
  
  men_train_metrics[[kernel_name]] <-
    final_fit$metrics
}


# ============================================================
# 34. COMBINE FINAL TRAIN METRICS
#
# IMPORTANT FIX:
# women_train_metrics and men_train_metrics are lists containing
# one data frame for each kernel.
#
# Each list must first be collapsed before combining both sexes.
# ============================================================

women_train_metrics_df <- dplyr::bind_rows(
  women_train_metrics
)

men_train_metrics_df <- dplyr::bind_rows(
  men_train_metrics
)


all_train_metrics <- dplyr::bind_rows(
  women_train_metrics_df,
  men_train_metrics_df
)


# ------------------------------------------------------------
# Verify that exactly six final models are present
# ------------------------------------------------------------

if (nrow(all_train_metrics) != 6) {
  
  stop(
    paste(
      "Unexpected number of final SVR TRAIN metric rows:",
      nrow(all_train_metrics),
      "- expected 6."
    )
  )
}


# ------------------------------------------------------------
# Verify absence of list columns
# ------------------------------------------------------------

list_columns <- vapply(
  all_train_metrics,
  is.list,
  logical(1)
)


if (any(list_columns)) {
  
  stop(
    paste(
      "List columns detected in final SVR metrics:",
      paste(
        names(all_train_metrics)[list_columns],
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 35. CREATE FINAL MODEL BUNDLES
# ============================================================

women_bundle <- list(
  
  sex =
    "Women",
  
  predictors =
    women_predictors,
  
  models =
    women_final_models,
  
  selected_hyperparameters =
    selected_hyperparameters %>%
    filter(
      Sex == "Women"
    ),
  
  cv_seeds =
    cv_seeds,
  
  number_folds =
    number_folds,
  
  preprocessing =
    "Centering and scaling estimated from TRAIN",
  
  tuning_selection =
    "Minimum mean OOF RMSE across 10 predefined CV configurations",
  
  r2_definition =
    "1 - SSE/SST",
  
  test_accessed =
    FALSE
)


men_bundle <- list(
  
  sex =
    "Men",
  
  predictors =
    men_predictors,
  
  models =
    men_final_models,
  
  selected_hyperparameters =
    selected_hyperparameters %>%
    filter(
      Sex == "Men"
    ),
  
  cv_seeds =
    cv_seeds,
  
  number_folds =
    number_folds,
  
  preprocessing =
    "Centering and scaling estimated from TRAIN",
  
  tuning_selection =
    "Minimum mean OOF RMSE across 10 predefined CV configurations",
  
  r2_definition =
    "1 - SSE/SST",
  
  test_accessed =
    FALSE
)


# ============================================================
# 36. SAVE ESSENTIAL OUTPUT FILES
# ============================================================

write_csv(
  
  tuning_summary,
  
  file.path(
    output_dir,
    "01_SVR_tuning_summary.csv"
  )
)


write_csv(
  
  best_by_seed,
  
  file.path(
    output_dir,
    "02_SVR_best_hyperparameters_by_seed.csv"
  )
)


write_csv(
  
  selected_hyperparameters,
  
  file.path(
    output_dir,
    "03_SVR_selected_hyperparameters.csv"
  )
)


write_csv(
  
  final_cv_by_seed,
  
  file.path(
    output_dir,
    "04_SVR_CV_metrics_by_seed.csv"
  )
)


write_csv(
  
  final_cv_summary,
  
  file.path(
    output_dir,
    "05_SVR_CV_summary.csv"
  )
)


write_csv(
  
  all_train_metrics,
  
  file.path(
    output_dir,
    "06_SVR_train_metrics.csv"
  )
)


saveRDS(
  
  women_bundle,
  
  file.path(
    output_dir,
    "women_SVR_bundle.rds"
  )
)


saveRDS(
  
  men_bundle,
  
  file.path(
    output_dir,
    "men_SVR_bundle.rds"
  )
)


# ============================================================
# 37. DISPLAY FINAL RESULTS
# ============================================================

cat("\n\n========================================\n")
cat("SVR SELECTED HYPERPARAMETERS\n")
cat("========================================\n")

print(
  selected_hyperparameters
)


cat("\n========================================\n")
cat("SVR FINAL INTERNAL CV PERFORMANCE\n")
cat("========================================\n")

print(
  final_cv_summary
)


cat("\n========================================\n")
cat("SVR COMPLETE TRAIN PERFORMANCE\n")
cat("========================================\n")

print(
  all_train_metrics
)


# ============================================================
# 38. VERIFY OUTPUT FILES
# ============================================================

expected_files <- c(
  
  "01_SVR_tuning_summary.csv",
  
  "02_SVR_best_hyperparameters_by_seed.csv",
  
  "03_SVR_selected_hyperparameters.csv",
  
  "04_SVR_CV_metrics_by_seed.csv",
  
  "05_SVR_CV_summary.csv",
  
  "06_SVR_train_metrics.csv",
  
  "women_SVR_bundle.rds",
  
  "men_SVR_bundle.rds"
)


created_files <- list.files(
  output_dir
)


missing_outputs <- setdiff(
  expected_files,
  created_files
)


if (length(missing_outputs) > 0) {
  
  stop(
    paste(
      "Missing output files:",
      paste(
        missing_outputs,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 39. FINAL CONSISTENCY CHECK
# ============================================================

if (
  nrow(
    selected_hyperparameters
  ) != 6
) {
  
  stop(
    "Expected six selected SVR configurations."
  )
}


if (
  nrow(
    final_cv_summary
  ) != 6
) {
  
  stop(
    "Expected six final SVR CV summary rows."
  )
}


if (
  nrow(
    all_train_metrics
  ) != 6
) {
  
  stop(
    "Expected six final SVR TRAIN metric rows."
  )
}


# ============================================================
# 40. FINAL MESSAGE
# ============================================================

cat("\n\n========================================\n")
cat("SCRIPT 09 COMPLETED SUCCESSFULLY\n")
cat("========================================\n")

cat(
  "Models: Linear, Polynomial, Radial SVR\n"
)

cat(
  "Women predictors: 5\n"
)

cat(
  "Men predictors: 2\n"
)

cat(
  "CV seeds: 10\n"
)

cat(
  "Folds per seed: 10\n"
)

cat(
  "Primary tuning criterion: mean OOF RMSE\n"
)

cat(
  "Preprocessing: center and scale inside CV\n"
)

cat(
  "HCY: original scale\n"
)

cat(
  "Predictive R2: 1 - SSE/SST\n"
)

cat(
  "Final sex-specific SVR models: 6\n"
)

cat(
  "TEST: NEVER READ OR ACCESSED\n"
)

cat(
  "Output files: 8\n"
)

cat("========================================\n")


print(
  list.files(
    output_dir
  )
)