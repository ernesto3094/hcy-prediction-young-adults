# ============================================================
# SCRIPT 07
# MULTIPLE LINEAR REGRESSION - TRAIN ONLY
#
# FINAL VERSION
#
# Workflow:
# - Uses only transformed TRAIN data.
# - Uses final Elastic Net-selected predictors.
# - Women: LDL, BMI, GLU, VisceralFat, MCHC.
# - Men: BMI, HDL.
# - 10 independent 10-fold CV configurations.
# - CV is stratified by HCY quintiles.
# - Duplicate IDs remain in the same fold.
# - Centering/scaling is estimated inside each CV training fold.
# - HCY remains on its original scale.
# - Predictive R2 = 1 - SSE/SST.
# - Final model is refitted using the complete TRAIN subset.
# - TEST IS NEVER READ OR ACCESSED.
#
# The CV folds are generated in memory and are NOT saved.
# The same fold-generation function and seeds will be reused
# in PR, SVR, RF, and XGBoost.
# ============================================================


# ============================================================
# 1. PACKAGES
# ============================================================

required_packages <- c(
  "readr",
  "dplyr"
)

missing_packages <- required_packages[
  !required_packages %in% rownames(installed.packages())
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(readr)
library(dplyr)


# ============================================================
# 2. DIRECTORIES
# ============================================================

input_dir <- "05_elastic_net_results"
output_dir <- "07_MLR_results"

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
# 5. FINAL PREDICTORS
# ============================================================

if (!"Variable" %in% names(women_selected)) {
  stop(
    "Column 'Variable' was not found in the WOMEN selection file."
  )
}

if (!"Variable" %in% names(men_selected)) {
  stop(
    "Column 'Variable' was not found in the MEN selection file."
  )
}


women_predictors <- women_selected$Variable
men_predictors <- men_selected$Variable


cat("\n========================================\n")
cat("FINAL ELASTIC NET PREDICTORS\n")
cat("========================================\n")

cat("\nWOMEN:\n")
print(women_predictors)

cat("\nMEN:\n")
print(men_predictors)


# ============================================================
# 6. VERIFY EXPECTED FINAL PREDICTORS
# ============================================================

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


# ============================================================
# 7. VERIFY SAMPLE SIZES
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
# 8. DATA CHECK FUNCTION
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
  
  
  for (variable in c("HCY", predictors)) {
    
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
  data = women_train,
  predictors = women_predictors,
  sex_name = "WOMEN"
)

check_dataset(
  data = men_train,
  predictors = men_predictors,
  sex_name = "MEN"
)


# ============================================================
# 9. INTERNAL CV CONFIGURATION
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
# 10. CREATE GROUPED + HCY-STRATIFIED CV FOLDS
#
# IMPORTANT:
# This exact function will be reused in later model scripts.
# Folds exist only in memory.
# ============================================================

create_cv_folds <- function(
    data,
    seeds,
    k = 10
) {
  
  data_internal <- data %>%
    mutate(
      Internal_Row = seq_len(n()),
      ID = as.character(ID)
    )
  
  
  # One group per ID.
  # This guarantees that repeated IDs remain together.
  id_groups <- data_internal %>%
    group_by(ID) %>%
    summarise(
      Group_HCY = mean(HCY),
      .groups = "drop"
    ) %>%
    arrange(Group_HCY) %>%
    mutate(
      HCY_Stratum = ntile(
        Group_HCY,
        5
      )
    )
  
  
  all_assignments <- list()
  
  
  for (seed_index in seq_along(seeds)) {
    
    seed_value <- seeds[seed_index]
    
    set.seed(seed_value)
    
    current_groups <- id_groups
    
    current_groups$Fold <- NA_integer_
    
    
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
        size = length(stratum_rows),
        replace = FALSE
      )
      
      
      fold_labels <- rep(
        seq_len(k),
        length.out = length(shuffled_rows)
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
        by = "ID"
      )
    
    
    current_assignment$Seed <- seed_value
    
    
    current_assignment <- current_assignment %>%
      select(
        Seed,
        Fold,
        Internal_Row,
        ID,
        HCY_Stratum
      )
    
    
    all_assignments[[length(all_assignments) + 1]] <-
      current_assignment
  }
  
  
  assignments <- bind_rows(
    all_assignments
  )
  
  
  # ----------------------------------------------------------
  # Internal validation
  # ----------------------------------------------------------
  
  expected_number_rows <-
    nrow(data) * length(seeds)
  
  
  if (nrow(assignments) != expected_number_rows) {
    stop(
      "Unexpected number of CV assignment rows."
    )
  }
  
  
  if (any(is.na(assignments$Fold))) {
    stop(
      "Missing CV fold assignment detected."
    )
  }
  
  
  id_fold_check <- assignments %>%
    group_by(
      Seed,
      ID
    ) %>%
    summarise(
      Number_of_Folds = n_distinct(Fold),
      .groups = "drop"
    )
  
  
  if (any(id_fold_check$Number_of_Folds != 1)) {
    stop(
      "A duplicated ID was assigned to different folds."
    )
  }
  
  
  fold_number_check <- assignments %>%
    group_by(Seed) %>%
    summarise(
      Number_of_Folds = n_distinct(Fold),
      .groups = "drop"
    )
  
  
  if (any(fold_number_check$Number_of_Folds != k)) {
    stop(
      "At least one seed does not contain exactly 10 folds."
    )
  }
  
  
  return(assignments)
}


# ============================================================
# 11. METRIC FUNCTIONS
# ============================================================

calculate_rmse <- function(
    observed,
    predicted
) {
  
  sqrt(
    mean(
      (observed - predicted)^2
    )
  )
}


calculate_mae <- function(
    observed,
    predicted
) {
  
  mean(
    abs(
      observed - predicted
    )
  )
}


calculate_predictive_r2 <- function(
    observed,
    predicted
) {
  
  sse <- sum(
    (observed - predicted)^2
  )
  
  sst <- sum(
    (observed - mean(observed))^2
  )
  
  
  if (sst == 0) {
    return(NA_real_)
  }
  
  
  r2 <- 1 - (sse / sst)
  
  return(r2)
}


# ============================================================
# 12. SCALING FUNCTIONS
# ============================================================

estimate_scaling <- function(
    data,
    predictors
) {
  
  center_values <- sapply(
    data[
      ,
      predictors,
      drop = FALSE
    ],
    mean
  )
  
  
  scale_values <- sapply(
    data[
      ,
      predictors,
      drop = FALSE
    ],
    sd
  )
  
  
  if (
    any(!is.finite(scale_values)) ||
    any(scale_values <= 0)
  ) {
    
    stop(
      "Invalid scaling parameter detected."
    )
  }
  
  
  return(
    list(
      center = center_values,
      scale = scale_values
    )
  )
}


apply_scaling <- function(
    data,
    predictors,
    scaling_parameters
) {
  
  scaled_data <- data
  
  
  for (variable in predictors) {
    
    scaled_data[[variable]] <-
      (
        data[[variable]] -
          scaling_parameters$center[[variable]]
      ) /
      scaling_parameters$scale[[variable]]
  }
  
  
  return(scaled_data)
}


# ============================================================
# 13. MLR FUNCTION
# ============================================================

run_mlr <- function(
    data,
    predictors,
    sex_name,
    seeds,
    k = 10
) {
  
  cat(
    "\n========================================\n"
  )
  
  cat(
    "CREATING INTERNAL CV FOLDS -",
    sex_name,
    "\n"
  )
  
  cat(
    "========================================\n"
  )
  
  
  folds <- create_cv_folds(
    data = data,
    seeds = seeds,
    k = k
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
  
  
  seed_results <- list()
  
  
  # ==========================================================
  # 10 CV CONFIGURATIONS
  # ==========================================================
  
  for (seed_index in seq_along(seeds)) {
    
    current_seed <- seeds[seed_index]
    
    
    cat(
      sex_name,
      " | CV seed ",
      current_seed,
      " | ",
      seed_index,
      "/",
      length(seeds),
      "\n",
      sep = ""
    )
    
    
    seed_fold_assignments <- folds[
      folds$Seed == current_seed,
      ,
      drop = FALSE
    ]
    
    
    oof_predictions <- rep(
      NA_real_,
      nrow(data)
    )
    
    
    fold_values <- sort(
      unique(
        seed_fold_assignments$Fold
      )
    )
    
    
    # ========================================================
    # 10 FOLDS
    # ========================================================
    
    for (fold_value in fold_values) {
      
      validation_rows <-
        seed_fold_assignments$Internal_Row[
          seed_fold_assignments$Fold ==
            fold_value
        ]
      
      
      validation_rows <- as.integer(
        validation_rows
      )
      
      
      training_rows <- setdiff(
        seq_len(nrow(data)),
        validation_rows
      )
      
      
      cv_train <- data[
        training_rows,
        ,
        drop = FALSE
      ]
      
      
      cv_validation <- data[
        validation_rows,
        ,
        drop = FALSE
      ]
      
      
      # --------------------------------------------------------
      # Estimate center and scale from CV TRAIN only
      # --------------------------------------------------------
      
      fold_scaling <- estimate_scaling(
        data = cv_train,
        predictors = predictors
      )
      
      
      cv_train_scaled <- apply_scaling(
        data = cv_train,
        predictors = predictors,
        scaling_parameters = fold_scaling
      )
      
      
      cv_validation_scaled <- apply_scaling(
        data = cv_validation,
        predictors = predictors,
        scaling_parameters = fold_scaling
      )
      
      
      # --------------------------------------------------------
      # Fit MLR
      # --------------------------------------------------------
      
      fold_model <- lm(
        model_formula,
        data = cv_train_scaled
      )
      
      
      # --------------------------------------------------------
      # Predict validation observations
      # --------------------------------------------------------
      
      fold_predictions <- predict(
        fold_model,
        newdata = cv_validation_scaled
      )
      
      
      oof_predictions[
        validation_rows
      ] <- fold_predictions
    }
    
    
    # ----------------------------------------------------------
    # Ensure every TRAIN observation received an OOF prediction
    # ----------------------------------------------------------
    
    if (any(is.na(oof_predictions))) {
      stop(
        paste(
          "Missing OOF predictions for",
          sex_name,
          "seed",
          current_seed
        )
      )
    }
    
    
    # ----------------------------------------------------------
    # Performance for the complete OOF prediction vector
    # ----------------------------------------------------------
    
    seed_results[[seed_index]] <- data.frame(
      
      Sex = sex_name,
      
      Seed = current_seed,
      
      N = nrow(data),
      
      Number_of_Predictors =
        length(predictors),
      
      RMSE =
        calculate_rmse(
          observed = data$HCY,
          predicted = oof_predictions
        ),
      
      MAE =
        calculate_mae(
          observed = data$HCY,
          predicted = oof_predictions
        ),
      
      Predictive_R2 =
        calculate_predictive_r2(
          observed = data$HCY,
          predicted = oof_predictions
        )
    )
  }
  
  
  # ==========================================================
  # COMBINE SEED RESULTS
  # ==========================================================
  
  seed_metrics <- bind_rows(
    seed_results
  )
  
  
  # ==========================================================
  # SUMMARY ACROSS THE 10 INTERNAL CV SEEDS
  # ==========================================================
  
  cv_summary <- data.frame(
    
    Sex = sex_name,
    
    N = nrow(data),
    
    Number_of_Predictors =
      length(predictors),
    
    Number_of_CV_Seeds =
      length(seeds),
    
    Mean_RMSE =
      mean(
        seed_metrics$RMSE
      ),
    
    SD_RMSE =
      sd(
        seed_metrics$RMSE
      ),
    
    Min_RMSE =
      min(
        seed_metrics$RMSE
      ),
    
    Max_RMSE =
      max(
        seed_metrics$RMSE
      ),
    
    Mean_MAE =
      mean(
        seed_metrics$MAE
      ),
    
    SD_MAE =
      sd(
        seed_metrics$MAE
      ),
    
    Min_MAE =
      min(
        seed_metrics$MAE
      ),
    
    Max_MAE =
      max(
        seed_metrics$MAE
      ),
    
    Mean_Predictive_R2 =
      mean(
        seed_metrics$Predictive_R2
      ),
    
    SD_Predictive_R2 =
      sd(
        seed_metrics$Predictive_R2
      ),
    
    Min_Predictive_R2 =
      min(
        seed_metrics$Predictive_R2
      ),
    
    Max_Predictive_R2 =
      max(
        seed_metrics$Predictive_R2
      )
  )
  
  
  # ==========================================================
  # FINAL MLR USING COMPLETE TRAIN
  # ==========================================================
  
  final_scaling <- estimate_scaling(
    data = data,
    predictors = predictors
  )
  
  
  complete_train_scaled <- apply_scaling(
    data = data,
    predictors = predictors,
    scaling_parameters = final_scaling
  )
  
  
  final_model <- lm(
    model_formula,
    data = complete_train_scaled
  )
  
  
  final_train_predictions <- predict(
    final_model,
    newdata = complete_train_scaled
  )
  
  
  # ==========================================================
  # COMPLETE TRAIN PERFORMANCE
  # ==========================================================
  
  model_summary <- summary(
    final_model
  )
  
  
  train_metrics <- data.frame(
    
    Sex = sex_name,
    
    N = nrow(data),
    
    Number_of_Predictors =
      length(predictors),
    
    RMSE =
      calculate_rmse(
        observed = data$HCY,
        predicted = final_train_predictions
      ),
    
    MAE =
      calculate_mae(
        observed = data$HCY,
        predicted = final_train_predictions
      ),
    
    Predictive_R2 =
      calculate_predictive_r2(
        observed = data$HCY,
        predicted = final_train_predictions
      ),
    
    Adjusted_R2 =
      model_summary$adj.r.squared
  )
  
  
  # ==========================================================
  # FINAL COEFFICIENTS
  # ==========================================================
  
  coefficient_matrix <-
    model_summary$coefficients
  
  
  coefficient_table <- data.frame(
    
    Sex = sex_name,
    
    Term =
      rownames(
        coefficient_matrix
      ),
    
    Estimate =
      coefficient_matrix[
        ,
        "Estimate"
      ],
    
    Std_Error =
      coefficient_matrix[
        ,
        "Std. Error"
      ],
    
    t_value =
      coefficient_matrix[
        ,
        "t value"
      ],
    
    p_value =
      coefficient_matrix[
        ,
        "Pr(>|t|)"
      ],
    
    row.names = NULL
  )
  
  
  # ==========================================================
  # FINAL TRAIN SCALING PARAMETERS
  # ==========================================================
  
  scaling_table <- data.frame(
    
    Sex = sex_name,
    
    Predictor =
      predictors,
    
    Center =
      as.numeric(
        final_scaling$center[
          predictors
        ]
      ),
    
    Scale =
      as.numeric(
        final_scaling$scale[
          predictors
        ]
      )
  )
  
  
  # ==========================================================
  # BUNDLE FOR LATER TEST EVALUATION
  # ==========================================================
  
  model_bundle <- list(
    
    sex = sex_name,
    
    predictors = predictors,
    
    model = final_model,
    
    scaling = final_scaling,
    
    cv_seeds = seeds,
    
    number_folds = k,
    
    r2_definition = "1 - SSE/SST"
  )
  
  
  return(
    list(
      seed_metrics = seed_metrics,
      cv_summary = cv_summary,
      train_metrics = train_metrics,
      coefficients = coefficient_table,
      scaling = scaling_table,
      bundle = model_bundle
    )
  )
}


# ============================================================
# 14. RUN WOMEN MLR
# ============================================================

cat("\n\n========================================\n")
cat("RUNNING WOMEN MLR\n")
cat("========================================\n")


women_mlr <- run_mlr(
  data = women_train,
  predictors = women_predictors,
  sex_name = "Women",
  seeds = cv_seeds,
  k = number_folds
)


# ============================================================
# 15. RUN MEN MLR
# ============================================================

cat("\n\n========================================\n")
cat("RUNNING MEN MLR\n")
cat("========================================\n")


men_mlr <- run_mlr(
  data = men_train,
  predictors = men_predictors,
  sex_name = "Men",
  seeds = cv_seeds,
  k = number_folds
)


# ============================================================
# 16. COMBINE WOMEN + MEN RESULTS
# ============================================================

all_cv_metrics <- bind_rows(
  women_mlr$seed_metrics,
  men_mlr$seed_metrics
)


all_cv_summary <- bind_rows(
  women_mlr$cv_summary,
  men_mlr$cv_summary
)


all_train_metrics <- bind_rows(
  women_mlr$train_metrics,
  men_mlr$train_metrics
)


all_coefficients <- bind_rows(
  women_mlr$coefficients,
  men_mlr$coefficients
)


all_scaling <- bind_rows(
  women_mlr$scaling,
  men_mlr$scaling
)


# ============================================================
# 17. SAVE ONLY ESSENTIAL RESULTS
# ============================================================

write_csv(
  all_cv_metrics,
  file.path(
    output_dir,
    "01_MLR_CV_metrics_by_seed.csv"
  )
)


write_csv(
  all_cv_summary,
  file.path(
    output_dir,
    "02_MLR_CV_summary.csv"
  )
)


write_csv(
  all_train_metrics,
  file.path(
    output_dir,
    "03_MLR_train_metrics.csv"
  )
)


write_csv(
  all_coefficients,
  file.path(
    output_dir,
    "04_MLR_coefficients.csv"
  )
)


write_csv(
  all_scaling,
  file.path(
    output_dir,
    "05_MLR_scaling_parameters.csv"
  )
)


saveRDS(
  women_mlr$bundle,
  file.path(
    output_dir,
    "women_MLR_bundle.rds"
  )
)


saveRDS(
  men_mlr$bundle,
  file.path(
    output_dir,
    "men_MLR_bundle.rds"
  )
)


# ============================================================
# 18. DISPLAY RESULTS
# ============================================================

cat("\n\n========================================\n")
cat("MLR INTERNAL CV SUMMARY\n")
cat("========================================\n")

print(
  all_cv_summary
)


cat("\n========================================\n")
cat("MLR COMPLETE TRAIN METRICS\n")
cat("========================================\n")

print(
  all_train_metrics
)


cat("\n========================================\n")
cat("MLR FINAL COEFFICIENTS\n")
cat("========================================\n")

print(
  all_coefficients
)


# ============================================================
# 19. FINAL VERIFICATION
# ============================================================

expected_output_files <- c(
  "01_MLR_CV_metrics_by_seed.csv",
  "02_MLR_CV_summary.csv",
  "03_MLR_train_metrics.csv",
  "04_MLR_coefficients.csv",
  "05_MLR_scaling_parameters.csv",
  "women_MLR_bundle.rds",
  "men_MLR_bundle.rds"
)


created_files <- list.files(
  output_dir
)


missing_outputs <- setdiff(
  expected_output_files,
  created_files
)


if (length(missing_outputs) > 0) {
  stop(
    paste(
      "The following output files were not created:",
      paste(
        missing_outputs,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

cat("\n\n========================================\n")
cat("SCRIPT 07 COMPLETED SUCCESSFULLY\n")
cat("========================================\n")
cat("Model: Multiple Linear Regression\n")
cat("Women TRAIN: 190 observations\n")
cat("Men TRAIN: 83 observations\n")
cat("Internal CV: 10 seeds x 10 folds\n")
cat("Stratification: HCY quintiles\n")
cat("Duplicate IDs: kept within the same fold\n")
cat("Centering/scaling: inside each CV training fold\n")
cat("Final scaling: complete TRAIN only\n")
cat("HCY: original scale\n")
cat("Predictive R2: 1 - SSE/SST\n")
cat("TEST: NEVER READ OR ACCESSED\n")
cat("Number of output files: 7\n")
cat("========================================\n")

