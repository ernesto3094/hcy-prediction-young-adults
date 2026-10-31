# ============================================================
# SCRIPT 08
# POLYNOMIAL REGRESSION WITH DEGREE TUNING - TRAIN ONLY
#
# FINAL METHODOLOGICAL DESIGN
#
# Candidate polynomial degrees: 2, 3, 4
#
# Degree 1 is not included because it corresponds to the
# Multiple Linear Regression baseline already evaluated.
#
# No predictor-by-predictor interactions are included.
#
# Model selection:
# - 10 independent 10-fold CV configurations.
# - Same predefined seeds used throughout the study.
# - HCY-quintile stratification.
# - Duplicate IDs remain in the same fold.
# - RMSE is the primary tuning metric.
# - A one-standard-error parsimony rule is used:
#   among models within one SE of the minimum mean CV RMSE,
#   the lowest polynomial degree is selected.
#
# Scaling:
# - Estimated within each CV training fold.
# - Applied unchanged to the corresponding validation fold.
#
# HCY:
# - Never transformed.
# - Remains on its original scale.
#
# TEST:
# - NEVER read or accessed.
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
output_dir <- "08_PR_results"

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
# 4. READ DATA
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
# 5. FINAL ELASTIC NET PREDICTORS
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
cat("FINAL ELASTIC NET PREDICTORS\n")
cat("========================================\n")

cat("\nWOMEN:\n")
print(women_predictors)

cat("\nMEN:\n")
print(men_predictors)


# ============================================================
# 6. SAMPLE SIZE CHECK
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
# 8. CV AND TUNING SETTINGS
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


# Polynomial degree 1 is represented by MLR.
# PR therefore evaluates nonlinear expansions only.

candidate_degrees <- c(
  2,
  3,
  4
)


# ============================================================
# 9. CREATE GROUPED AND HCY-STRATIFIED CV FOLDS
#
# This is the same deterministic procedure used for MLR.
# Folds are kept only in memory.
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
  
  
  # One grouping unit per unique ID.
  # Duplicate IDs will always remain together.
  
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
  # Validation checks
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
      .groups = "drop"
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
    group_by(Seed) %>%
    summarise(
      Number_of_Folds =
        n_distinct(Fold),
      .groups = "drop"
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
  
  
  return(assignments)
}


# ============================================================
# 10. PERFORMANCE METRICS
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
    
    return(
      NA_real_
    )
  }
  
  
  return(
    1 - (sse / sst)
  )
}


# ============================================================
# 11. SCALING FUNCTIONS
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
  
  
  return(
    scaled_data
  )
}


# ============================================================
# 12. CREATE POLYNOMIAL FORMULA
#
# Example degree 3:
#
# HCY ~ BMI + I(BMI^2) + I(BMI^3)
#     + HDL + I(HDL^2) + I(HDL^3)
#
# No interaction terms.
# ============================================================

create_polynomial_formula <- function(
    predictors,
    degree
) {
  
  all_terms <- character(0)
  
  
  for (variable in predictors) {
    
    all_terms <- c(
      all_terms,
      variable
    )
    
    
    if (degree >= 2) {
      
      for (
        current_power in 2:degree
      ) {
        
        all_terms <- c(
          all_terms,
          paste0(
            "I(",
            variable,
            "^",
            current_power,
            ")"
          )
        )
      }
    }
  }
  
  
  formula_text <- paste(
    "HCY ~",
    paste(
      all_terms,
      collapse = " + "
    )
  )
  
  
  return(
    as.formula(
      formula_text
    )
  )
}


# ============================================================
# 13. EVALUATE ONE POLYNOMIAL DEGREE
# ============================================================

evaluate_pr_degree <- function(
    data,
    predictors,
    folds,
    sex_name,
    degree,
    seeds
) {
  
  model_formula <- create_polynomial_formula(
    predictors = predictors,
    degree = degree
  )
  
  
  expected_parameters <-
    1 +
    length(predictors) *
    degree
  
  
  seed_results <- list()
  
  
  for (seed_index in seq_along(seeds)) {
    
    current_seed <- seeds[seed_index]
    
    
    cat(
      sex_name,
      " | Degree ",
      degree,
      " | Seed ",
      current_seed,
      " | ",
      seed_index,
      "/",
      length(seeds),
      "\n",
      sep = ""
    )
    
    
    seed_assignments <- folds[
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
        seed_assignments$Fold
      )
    )
    
    
    for (fold_value in fold_values) {
      
      validation_rows <-
        seed_assignments$Internal_Row[
          seed_assignments$Fold ==
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
      # Estimate scaling using current CV training fold only
      # --------------------------------------------------------
      
      fold_scaling <- estimate_scaling(
        data = cv_train,
        predictors = predictors
      )
      
      
      cv_train_scaled <- apply_scaling(
        data = cv_train,
        predictors = predictors,
        scaling_parameters =
          fold_scaling
      )
      
      
      cv_validation_scaled <- apply_scaling(
        data = cv_validation,
        predictors = predictors,
        scaling_parameters =
          fold_scaling
      )
      
      
      # --------------------------------------------------------
      # Fit polynomial regression
      # --------------------------------------------------------
      
      fold_model <- lm(
        model_formula,
        data = cv_train_scaled
      )
      
      
      # --------------------------------------------------------
      # Detect rank-deficient fits
      # --------------------------------------------------------
      
      if (
        fold_model$rank <
        expected_parameters
      ) {
        
        stop(
          paste(
            "Rank-deficient PR model detected.",
            "Sex:",
            sex_name,
            "| Degree:",
            degree,
            "| Seed:",
            current_seed,
            "| Fold:",
            fold_value
          )
        )
      }
      
      
      # --------------------------------------------------------
      # Validation prediction
      # --------------------------------------------------------
      
      fold_predictions <- predict(
        fold_model,
        newdata =
          cv_validation_scaled
      )
      
      
      if (
        any(
          !is.finite(
            fold_predictions
          )
        )
      ) {
        
        stop(
          paste(
            "Non-finite predictions detected.",
            "Sex:",
            sex_name,
            "| Degree:",
            degree,
            "| Seed:",
            current_seed,
            "| Fold:",
            fold_value
          )
        )
      }
      
      
      oof_predictions[
        validation_rows
      ] <- fold_predictions
    }
    
    
    if (
      any(
        is.na(
          oof_predictions
        )
      )
    ) {
      
      stop(
        paste(
          "Missing OOF predictions.",
          "Sex:",
          sex_name,
          "| Degree:",
          degree,
          "| Seed:",
          current_seed
        )
      )
    }
    
    
    seed_results[[seed_index]] <-
      data.frame(
        
        Sex = sex_name,
        
        Degree = degree,
        
        Seed = current_seed,
        
        N = nrow(data),
        
        Original_Predictors =
          length(predictors),
        
        Polynomial_Terms =
          length(predictors) *
          degree,
        
        RMSE =
          calculate_rmse(
            observed = data$HCY,
            predicted =
              oof_predictions
          ),
        
        MAE =
          calculate_mae(
            observed = data$HCY,
            predicted =
              oof_predictions
          ),
        
        Predictive_R2 =
          calculate_predictive_r2(
            observed = data$HCY,
            predicted =
              oof_predictions
          )
      )
  }
  
  
  return(
    bind_rows(
      seed_results
    )
  )
}


# ============================================================
# 14. TUNE POLYNOMIAL DEGREE
# ============================================================

tune_pr <- function(
    data,
    predictors,
    sex_name,
    seeds,
    degrees,
    k = 10
) {
  
  cat("\n========================================\n")
  cat("PR TUNING -", sex_name, "\n")
  cat("========================================\n")
  
  
  folds <- create_cv_folds(
    data = data,
    seeds = seeds,
    k = k
  )
  
  
  degree_results <- list()
  
  
  for (
    degree_index in seq_along(
      degrees
    )
  ) {
    
    current_degree <-
      degrees[degree_index]
    
    
    cat(
      "\nTesting polynomial degree:",
      current_degree,
      "\n"
    )
    
    
    degree_results[[degree_index]] <-
      evaluate_pr_degree(
        
        data = data,
        
        predictors = predictors,
        
        folds = folds,
        
        sex_name = sex_name,
        
        degree = current_degree,
        
        seeds = seeds
      )
  }
  
  
  all_seed_results <- bind_rows(
    degree_results
  )
  
  
  # ==========================================================
  # SUMMARIZE TUNING RESULTS
  # ==========================================================
  
  tuning_summary <-
    all_seed_results %>%
    
    group_by(
      Sex,
      Degree
    ) %>%
    
    summarise(
      
      Mean_RMSE =
        mean(RMSE),
      
      SD_RMSE =
        sd(RMSE),
      
      SE_RMSE =
        sd(RMSE) /
        sqrt(n()),
      
      Min_RMSE =
        min(RMSE),
      
      Max_RMSE =
        max(RMSE),
      
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
      
      Min_Predictive_R2 =
        min(
          Predictive_R2
        ),
      
      Max_Predictive_R2 =
        max(
          Predictive_R2
        ),
      
      .groups = "drop"
    ) %>%
    
    arrange(
      Mean_RMSE,
      Degree
    )
  
  
  # ==========================================================
  # ABSOLUTE BEST DEGREE
  # ==========================================================
  
  best_row <-
    tuning_summary %>%
    arrange(
      Mean_RMSE,
      Degree
    ) %>%
    slice(1)
  
  
  absolute_best_degree <-
    best_row$Degree
  
  
  minimum_mean_rmse <-
    best_row$Mean_RMSE
  
  
  best_degree_se <-
    best_row$SE_RMSE
  
  
  # ==========================================================
  # ONE-STANDARD-ERROR PARSIMONY RULE
  #
  # Choose the simplest degree whose mean RMSE does not exceed
  # the minimum mean RMSE by more than one SE of the best model.
  #
  # This is a model-selection heuristic, not an inferential CI.
  # ==========================================================
  
  selection_threshold <-
    minimum_mean_rmse +
    best_degree_se
  
  
  eligible_degrees <-
    tuning_summary %>%
    
    filter(
      Mean_RMSE <=
        selection_threshold
    ) %>%
    
    arrange(
      Degree
    )
  
  
  selected_degree <-
    min(
      eligible_degrees$Degree
    )
  
  
  selected_row <-
    tuning_summary %>%
    
    filter(
      Degree ==
        selected_degree
    )
  
  
  selection_table <- data.frame(
    
    Sex = sex_name,
    
    Absolute_Best_Degree =
      absolute_best_degree,
    
    Absolute_Best_Mean_RMSE =
      minimum_mean_rmse,
    
    SE_of_Absolute_Best =
      best_degree_se,
    
    One_SE_Threshold =
      selection_threshold,
    
    Selected_Degree =
      selected_degree,
    
    Selected_Degree_Mean_RMSE =
      selected_row$Mean_RMSE,
    
    Selection_Rule =
      "Lowest degree within one SE of minimum mean CV RMSE"
  )
  
  
  # ==========================================================
  # FIT FINAL SELECTED PR USING COMPLETE TRAIN
  # ==========================================================
  
  final_formula <-
    create_polynomial_formula(
      
      predictors = predictors,
      
      degree =
        selected_degree
    )
  
  
  final_scaling <-
    estimate_scaling(
      
      data = data,
      
      predictors =
        predictors
    )
  
  
  complete_train_scaled <-
    apply_scaling(
      
      data = data,
      
      predictors =
        predictors,
      
      scaling_parameters =
        final_scaling
    )
  
  
  final_model <- lm(
    
    final_formula,
    
    data =
      complete_train_scaled
  )
  
  
  expected_final_parameters <-
    1 +
    length(predictors) *
    selected_degree
  
  
  if (
    final_model$rank <
    expected_final_parameters
  ) {
    
    stop(
      paste(
        "Final PR model is rank deficient for",
        sex_name
      )
    )
  }
  
  
  train_predictions <-
    predict(
      
      final_model,
      
      newdata =
        complete_train_scaled
    )
  
  
  final_model_summary <-
    summary(
      final_model
    )
  
  
  # ==========================================================
  # FINAL TRAIN METRICS
  # ==========================================================
  
  train_metrics <- data.frame(
    
    Sex = sex_name,
    
    Selected_Degree =
      selected_degree,
    
    N = nrow(data),
    
    Original_Predictors =
      length(predictors),
    
    Polynomial_Terms =
      length(predictors) *
      selected_degree,
    
    RMSE =
      calculate_rmse(
        data$HCY,
        train_predictions
      ),
    
    MAE =
      calculate_mae(
        data$HCY,
        train_predictions
      ),
    
    Predictive_R2 =
      calculate_predictive_r2(
        data$HCY,
        train_predictions
      ),
    
    Adjusted_R2 =
      final_model_summary$
      adj.r.squared
  )
  
  
  # ==========================================================
  # FINAL COEFFICIENTS
  # ==========================================================
  
  coefficient_matrix <-
    final_model_summary$
    coefficients
  
  
  coefficient_table <- data.frame(
    
    Sex = sex_name,
    
    Degree =
      selected_degree,
    
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
  # SCALING PARAMETERS
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
  # MODEL BUNDLE
  # ==========================================================
  
  model_bundle <- list(
    
    sex =
      sex_name,
    
    model_name =
      "Polynomial Regression",
    
    candidate_degrees =
      degrees,
    
    absolute_best_degree =
      absolute_best_degree,
    
    selected_degree =
      selected_degree,
    
    selection_rule =
      "Lowest degree within one SE of minimum mean CV RMSE",
    
    interaction_terms =
      FALSE,
    
    predictors =
      predictors,
    
    formula =
      final_formula,
    
    model =
      final_model,
    
    scaling =
      final_scaling,
    
    cv_seeds =
      seeds,
    
    number_folds =
      k,
    
    r2_definition =
      "1 - SSE/SST"
  )
  
  
  return(
    list(
      
      seed_metrics =
        all_seed_results,
      
      tuning_summary =
        tuning_summary,
      
      selection =
        selection_table,
      
      train_metrics =
        train_metrics,
      
      coefficients =
        coefficient_table,
      
      scaling =
        scaling_table,
      
      bundle =
        model_bundle
    )
  )
}


# ============================================================
# 15. WOMEN PR
# ============================================================

women_pr <- tune_pr(
  
  data =
    women_train,
  
  predictors =
    women_predictors,
  
  sex_name =
    "Women",
  
  seeds =
    cv_seeds,
  
  degrees =
    candidate_degrees,
  
  k =
    number_folds
)


# ============================================================
# 16. MEN PR
# ============================================================

men_pr <- tune_pr(
  
  data =
    men_train,
  
  predictors =
    men_predictors,
  
  sex_name =
    "Men",
  
  seeds =
    cv_seeds,
  
  degrees =
    candidate_degrees,
  
  k =
    number_folds
)


# ============================================================
# 17. COMBINE WOMEN + MEN
# ============================================================

all_seed_metrics <- bind_rows(
  women_pr$seed_metrics,
  men_pr$seed_metrics
)


all_tuning_summary <- bind_rows(
  women_pr$tuning_summary,
  men_pr$tuning_summary
)


all_selection <- bind_rows(
  women_pr$selection,
  men_pr$selection
)


all_train_metrics <- bind_rows(
  women_pr$train_metrics,
  men_pr$train_metrics
)


all_coefficients <- bind_rows(
  women_pr$coefficients,
  men_pr$coefficients
)


all_scaling <- bind_rows(
  women_pr$scaling,
  men_pr$scaling
)


# ============================================================
# 18. SAVE ESSENTIAL OUTPUTS ONLY
# ============================================================

write_csv(
  all_seed_metrics,
  file.path(
    output_dir,
    "01_PR_tuning_by_seed.csv"
  )
)


write_csv(
  all_tuning_summary,
  file.path(
    output_dir,
    "02_PR_tuning_summary.csv"
  )
)


write_csv(
  all_selection,
  file.path(
    output_dir,
    "03_PR_selected_degree.csv"
  )
)


write_csv(
  all_train_metrics,
  file.path(
    output_dir,
    "04_PR_train_metrics.csv"
  )
)


write_csv(
  all_coefficients,
  file.path(
    output_dir,
    "05_PR_coefficients.csv"
  )
)


write_csv(
  all_scaling,
  file.path(
    output_dir,
    "06_PR_scaling_parameters.csv"
  )
)


saveRDS(
  women_pr$bundle,
  file.path(
    output_dir,
    "women_PR_bundle.rds"
  )
)


saveRDS(
  men_pr$bundle,
  file.path(
    output_dir,
    "men_PR_bundle.rds"
  )
)


# ============================================================
# 19. DISPLAY TUNING RESULTS
# ============================================================

cat("\n\n========================================\n")
cat("PR TUNING SUMMARY\n")
cat("========================================\n")

print(
  all_tuning_summary
)


cat("\n========================================\n")
cat("PR DEGREE SELECTION\n")
cat("========================================\n")

print(
  all_selection
)


cat("\n========================================\n")
cat("PR COMPLETE TRAIN METRICS\n")
cat("========================================\n")

print(
  all_train_metrics
)


# ============================================================
# 20. VERIFY OUTPUT FILES
# ============================================================

expected_output_files <- c(
  
  "01_PR_tuning_by_seed.csv",
  
  "02_PR_tuning_summary.csv",
  
  "03_PR_selected_degree.csv",
  
  "04_PR_train_metrics.csv",
  
  "05_PR_coefficients.csv",
  
  "06_PR_scaling_parameters.csv",
  
  "women_PR_bundle.rds",
  
  "men_PR_bundle.rds"
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
      "Missing output files:",
      paste(
        missing_outputs,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

cat("\n\n========================================\n")
cat("SCRIPT 08 COMPLETED SUCCESSFULLY\n")
cat("========================================\n")
cat("Model: Polynomial Regression\n")
cat("Candidate degrees: 2, 3, 4\n")
cat("Degree 1: represented by MLR baseline\n")
cat("Interactions: NO\n")
cat("Primary tuning metric: RMSE\n")
cat("Selection: one-SE parsimony rule\n")
cat("Internal CV: 10 seeds x 10 folds\n")
cat("Stratification: HCY quintiles\n")
cat("Duplicate IDs: kept in same fold\n")
cat("Scaling: inside each CV training fold\n")
cat("HCY: original scale\n")
cat("Predictive R2: 1 - SSE/SST\n")
cat("TEST: NEVER READ OR ACCESSED\n")
cat("Output files: 8\n")
cat("========================================\n")
