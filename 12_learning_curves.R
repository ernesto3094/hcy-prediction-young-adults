# ============================================================
# SCRIPT 12
# LEARNING CURVES - TRAIN ONLY
# ============================================================
#
# Final frozen workflows:
#
# 1. MLR
# 2. PR
# 3. SVR-Linear
# 4. SVR-Polynomial
# 5. SVR-RBF
# 6. RF
# 7. XGB
#
# TRAIN ONLY.
# TEST is NEVER searched, read, or accessed.
#
# Internal design:
#
# - 10 prespecified seeds
# - 10-fold grouped CV
# - approximate HCY-quintile stratification
# - duplicate IDs remain in the same fold
#
# Learning fractions:
#
# 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
#
# Metrics:
#
# RMSE
# MAE
# Predictive R2 = 1 - SSE/SST
#
# ============================================================





# ============================================================
# 1. PACKAGES
# ============================================================

required_packages <- c(
  "caret",
  "kernlab",
  "readr",
  "dplyr",
  "ggplot2"
)


missing_packages <- required_packages[
  !required_packages %in%
    rownames(installed.packages())
]


if (length(missing_packages) > 0) {
  
  install.packages(
    missing_packages
  )
}


library(caret)
library(kernlab)
library(readr)
library(dplyr)
library(ggplot2)


# ============================================================
# 2. ANALYSIS VERSION
# ============================================================

analysis_version <-
  "SVR_LEARNING_CURVES_CARET_CORRECTION_20261024"


# ============================================================
# 3. DIRECTORIES
# ============================================================

elastic_net_dir <-
  "05_elastic_net_results"


svr_dir <-
  "09_SVR_results"


learning_curve_dir <-
  "12_learning_curves_results"


if (!dir.exists(learning_curve_dir)) {
  
  stop(
    paste(
      learning_curve_dir,
      "does not exist.",
      "Run Script 12 first."
    )
  )
}


# ============================================================
# 4. REQUIRED INPUT FILES
# ============================================================

women_train_file <- file.path(
  elastic_net_dir,
  "01_women_TRAIN_transformed.csv"
)


men_train_file <- file.path(
  elastic_net_dir,
  "02_men_TRAIN_transformed.csv"
)


women_selected_file <- file.path(
  elastic_net_dir,
  "10_women_final_selected_predictors.csv"
)


men_selected_file <- file.path(
  elastic_net_dir,
  "20_men_final_selected_predictors.csv"
)


script09_cv_file <- file.path(
  svr_dir,
  "05_SVR_CV_summary.csv"
)


existing_learning_metrics_file <- file.path(
  learning_curve_dir,
  "02_learning_curve_metrics_by_seed.csv"
)


required_files <- c(
  women_train_file,
  men_train_file,
  women_selected_file,
  men_selected_file,
  script09_cv_file,
  existing_learning_metrics_file
)


missing_files <- required_files[
  !file.exists(required_files)
]


if (length(missing_files) > 0) {
  
  stop(
    paste(
      "Missing required files:\n",
      paste(
        missing_files,
        collapse = "\n"
      )
    )
  )
}


# ============================================================
# 5. READ DATA
# ============================================================

women_train <- readr::read_csv(
  women_train_file,
  show_col_types = FALSE
)


men_train <- readr::read_csv(
  men_train_file,
  show_col_types = FALSE
)


women_selected <- readr::read_csv(
  women_selected_file,
  show_col_types = FALSE
)


men_selected <- readr::read_csv(
  men_selected_file,
  show_col_types = FALSE
)


script09_cv_summary <- readr::read_csv(
  script09_cv_file,
  show_col_types = FALSE
)


existing_learning_metrics <- readr::read_csv(
  existing_learning_metrics_file,
  show_col_types = FALSE
)


# ============================================================
# 6. VALIDATE TRAIN SAMPLE SIZES
# ============================================================

if (nrow(women_train) != 190) {
  
  stop(
    paste(
      "Unexpected WOMEN TRAIN size:",
      nrow(women_train)
    )
  )
}


if (nrow(men_train) != 83) {
  
  stop(
    paste(
      "Unexpected MEN TRAIN size:",
      nrow(men_train)
    )
  )
}


cat("\n========================================\n")
cat("TRAIN SAMPLE SIZES\n")
cat("========================================\n")

cat(
  "Women:",
  nrow(women_train),
  "\n"
)

cat(
  "Men:",
  nrow(men_train),
  "\n"
)


# ============================================================
# 7. FINAL SELECTED PREDICTORS
#
# Use the exact predictor files from Script 05, as Script 09 did.
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


women_predictors <-
  as.character(
    women_selected$Variable
  )


men_predictors <-
  as.character(
    men_selected$Variable
  )


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


if (
  !setequal(
    women_predictors,
    expected_women
  )
) {
  
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


if (
  !setequal(
    men_predictors,
    expected_men
  )
) {
  
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
cat("FINAL SVR PREDICTORS\n")
cat("========================================\n")

cat(
  "Women:",
  paste(
    women_predictors,
    collapse = ", "
  ),
  "\n"
)

cat(
  "Men:",
  paste(
    men_predictors,
    collapse = ", "
  ),
  "\n"
)


# ============================================================
# 8. VALIDATE DATA
# ============================================================

validate_svr_dataset <- function(
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
  
  
  numeric_columns <- c(
    "HCY",
    predictors
  )
  
  
  for (variable in numeric_columns) {
    
    values <- data[
      ,
      variable,
      drop = TRUE
    ]
    
    
    if (!is.numeric(values)) {
      
      stop(
        paste(
          sex_name,
          variable,
          "is not numeric."
        )
      )
    }
    
    
    if (any(is.na(values))) {
      
      stop(
        paste(
          "Missing values detected in",
          sex_name,
          variable
        )
      )
    }
    
    
    if (any(!is.finite(values))) {
      
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


validate_svr_dataset(
  women_train,
  women_predictors,
  "Women"
)


validate_svr_dataset(
  men_train,
  men_predictors,
  "Men"
)


# ============================================================
# 9. CV SETTINGS
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


learning_fractions <- c(
  0.30,
  0.40,
  0.50,
  0.60,
  0.70,
  0.80,
  0.90,
  1.00
)


svr_model_names <- c(
  "SVR-Linear",
  "SVR-Polynomial",
  "SVR-RBF"
)


all_model_names <- c(
  "MLR",
  "PR",
  "SVR-Linear",
  "SVR-Polynomial",
  "SVR-RBF",
  "RF",
  "XGB"
)


# ============================================================
# 10. FROZEN SVR SPECIFICATIONS
#
# These are the selected hyperparameters from Script 09.
# ============================================================

svr_specifications <- data.frame(
  
  Sex = c(
    "Women",
    "Women",
    "Women",
    "Men",
    "Men",
    "Men"
  ),
  
  Model = c(
    "SVR-Linear",
    "SVR-Polynomial",
    "SVR-RBF",
    "SVR-Linear",
    "SVR-Polynomial",
    "SVR-RBF"
  ),
  
  Caret_Method = c(
    "svmLinear",
    "svmPoly",
    "svmRadial",
    "svmLinear",
    "svmPoly",
    "svmRadial"
  ),
  
  C = c(
    0.0625,
    0.25,
    1,
    0.0625,
    0.25,
    256
  ),
  
  Degree = c(
    NA,
    8,
    NA,
    NA,
    4,
    NA
  ),
  
  Scale = c(
    NA,
    0.01,
    NA,
    NA,
    0.1,
    NA
  ),
  
  Sigma = c(
    NA,
    NA,
    0.0078125,
    NA,
    NA,
    0.015625
  ),
  
  stringsAsFactors = FALSE
)


readr::write_csv(
  
  svr_specifications,
  
  file.path(
    learning_curve_dir,
    "09_SVR_learning_curve_frozen_specifications.csv"
  )
)


# ============================================================
# 11. OFFICIAL METRICS
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
  
  
  1 - (sse / sst)
}


# ============================================================
# 12. CREATE GROUPED HCY-STRATIFIED FOLDS
#
# This reproduces the deterministic procedure from Script 09.
# ============================================================

create_cv_folds <- function(
    data,
    seeds,
    k = 10
) {
  
  data_internal <- data %>%
    
    dplyr::mutate(
      
      Internal_Row =
        seq_len(
          dplyr::n()
        ),
      
      ID =
        as.character(ID)
    )
  
  
  id_groups <- data_internal %>%
    
    dplyr::group_by(
      ID
    ) %>%
    
    dplyr::summarise(
      
      Group_HCY =
        mean(HCY),
      
      .groups =
        "drop"
    ) %>%
    
    dplyr::arrange(
      Group_HCY
    ) %>%
    
    dplyr::mutate(
      
      HCY_Stratum =
        dplyr::ntile(
          Group_HCY,
          5
        )
    )
  
  
  all_assignments <- list()
  
  
  for (seed_value in seeds) {
    
    set.seed(
      seed_value
    )
    
    
    current_groups <- id_groups
    
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
        size = length(stratum_rows),
        replace = FALSE
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
      
      dplyr::select(
        Internal_Row,
        ID
      ) %>%
      
      dplyr::left_join(
        
        current_groups %>%
          
          dplyr::select(
            ID,
            HCY_Stratum,
            Fold
          ),
        
        by = "ID"
      )
    
    
    current_assignment$Seed <-
      seed_value
    
    
    current_assignment <- current_assignment %>%
      
      dplyr::select(
        Seed,
        Fold,
        Internal_Row,
        ID,
        HCY_Stratum
      )
    
    
    all_assignments <- append(
      all_assignments,
      list(
        current_assignment
      )
    )
  }
  
  
  assignments <- dplyr::bind_rows(
    all_assignments
  )
  
  
  if (
    nrow(assignments) !=
    nrow(data) *
    length(seeds)
  ) {
    
    stop(
      "Unexpected number of CV assignment rows."
    )
  }
  
  
  if (
    any(
      is.na(
        assignments$Fold
      )
    )
  ) {
    
    stop(
      "Missing CV fold assignment detected."
    )
  }
  
  
  duplicate_check <- assignments %>%
    
    dplyr::group_by(
      Seed,
      ID
    ) %>%
    
    dplyr::summarise(
      
      Number_of_Folds =
        dplyr::n_distinct(
          Fold
        ),
      
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
    
    dplyr::group_by(
      Seed
    ) %>%
    
    dplyr::summarise(
      
      Number_of_Folds =
        dplyr::n_distinct(
          Fold
        ),
      
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
  
  
  assignments
}


# ============================================================
# 13. SAMPLE GROUPED TRAINING SUBSET
#
# This reproduces the nested learning-fraction sampling
# procedure used in Script 12.
# ============================================================

sample_grouped_training_rows <- function(
    data,
    available_rows,
    fraction,
    seed_value,
    fold_value
) {
  
  if (fraction >= 1) {
    
    return(
      sort(
        available_rows
      )
    )
  }
  
  
  available_data <- data[
    available_rows,
    ,
    drop = FALSE
  ]
  
  
  group_table <- available_data %>%
    
    dplyr::mutate(
      ID =
        as.character(ID)
    ) %>%
    
    dplyr::group_by(
      ID
    ) %>%
    
    dplyr::summarise(
      
      Group_HCY =
        mean(HCY),
      
      .groups =
        "drop"
    ) %>%
    
    dplyr::arrange(
      Group_HCY
    ) %>%
    
    dplyr::mutate(
      
      HCY_Stratum =
        dplyr::ntile(
          Group_HCY,
          5
        )
    )
  
  
  sampling_seed <- as.integer(
    
    seed_value +
      fold_value *
      100000 +
      55555
  )
  
  
  set.seed(
    sampling_seed
  )
  
  
  selected_ids <-
    character(0)
  
  
  strata <- sort(
    unique(
      group_table$HCY_Stratum
    )
  )
  
  
  for (stratum_value in strata) {
    
    stratum_ids <- group_table$ID[
      group_table$HCY_Stratum ==
        stratum_value
    ]
    
    
    shuffled_ids <- sample(
      stratum_ids,
      size = length(stratum_ids),
      replace = FALSE
    )
    
    
    number_to_select <- max(
      1,
      round(
        length(shuffled_ids) *
          fraction
      )
    )
    
    
    number_to_select <- min(
      number_to_select,
      length(shuffled_ids)
    )
    
    
    selected_ids <- c(
      selected_ids,
      head(
        shuffled_ids,
        number_to_select
      )
    )
  }
  
  
  selected_rows_local <- which(
    
    as.character(
      available_data$ID
    ) %in%
      selected_ids
  )
  
  
  selected_rows_global <- available_rows[
    selected_rows_local
  ]
  
  
  sort(
    selected_rows_global
  )
}


# ============================================================
# 14. CREATE ONE-ROW CARET TUNING GRID
# ============================================================

create_svr_tune_grid <- function(
    specification
) {
  
  method_name <-
    specification$Caret_Method[1]
  
  
  if (
    method_name ==
    "svmLinear"
  ) {
    
    return(
      
      data.frame(
        C =
          specification$C[1]
      )
    )
  }
  
  
  if (
    method_name ==
    "svmPoly"
  ) {
    
    return(
      
      data.frame(
        
        degree =
          specification$Degree[1],
        
        scale =
          specification$Scale[1],
        
        C =
          specification$C[1]
      )
    )
  }
  
  
  if (
    method_name ==
    "svmRadial"
  ) {
    
    return(
      
      data.frame(
        
        sigma =
          specification$Sigma[1],
        
        C =
          specification$C[1]
      )
    )
  }
  
  
  stop(
    "Unknown caret SVR method."
  )
}


# ============================================================
# 15. FIT ONE FROZEN SVR MODEL WITH CARET
#
# This intentionally mirrors Script 09:
#
# caret::train()
# preProcess = c("center", "scale")
# one-row frozen tuneGrid
# ============================================================

fit_predict_svr_caret <- function(
    training_data,
    validation_data,
    predictors,
    sex_name,
    model_name,
    seed_value
) {
  
  specification <- svr_specifications %>%
    
    dplyr::filter(
      Sex == sex_name,
      Model == model_name
    )
  
  
  if (
    nrow(specification) != 1
  ) {
    
    stop(
      paste(
        "SVR specification error:",
        sex_name,
        model_name
      )
    )
  }
  
  
  method_name <-
    specification$Caret_Method[1]
  
  
  tuning_grid <-
    create_svr_tune_grid(
      specification
    )
  
  
  model_formula <- stats::as.formula(
    
    paste(
      "HCY ~",
      paste(
        predictors,
        collapse = " + "
      )
    )
  )
  
  
  control <- caret::trainControl(
    
    method =
      "none",
    
    allowParallel =
      FALSE
  )
  
  
  set.seed(
    seed_value
  )
  
  
  fit <- caret::train(
    
    model_formula,
    
    data =
      training_data,
    
    method =
      method_name,
    
    trControl =
      control,
    
    preProcess =
      c(
        "center",
        "scale"
      ),
    
    tuneGrid =
      tuning_grid,
    
    metric =
      "RMSE"
  )
  
  
  train_predictions <- as.numeric(
    
    predict(
      fit,
      newdata =
        training_data
    )
  )
  
  
  validation_predictions <- as.numeric(
    
    predict(
      fit,
      newdata =
        validation_data
    )
  )
  
  
  if (
    length(train_predictions) !=
    nrow(training_data)
  ) {
    
    stop(
      "Unexpected SVR training prediction length."
    )
  }
  
  
  if (
    length(validation_predictions) !=
    nrow(validation_data)
  ) {
    
    stop(
      "Unexpected SVR validation prediction length."
    )
  }
  
  
  if (
    any(
      !is.finite(
        train_predictions
      )
    )
  ) {
    
    stop(
      "Non-finite SVR training predictions detected."
    )
  }
  
  
  if (
    any(
      !is.finite(
        validation_predictions
      )
    )
  ) {
    
    stop(
      "Non-finite SVR validation predictions detected."
    )
  }
  
  
  list(
    
    train =
      train_predictions,
    
    validation =
      validation_predictions
  )
}


# ============================================================
# 16. CREATE CV ASSIGNMENTS
# ============================================================

women_assignments <- create_cv_folds(
  
  data =
    women_train,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds
)


men_assignments <- create_cv_folds(
  
  data =
    men_train,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds
)


# ============================================================
# 17. CHECKPOINT SETTINGS
# ============================================================

checkpoint_file <- file.path(
  learning_curve_dir,
  "checkpoint_12B_SVR_caret.rds"
)


checkpoint_signature <- list(
  
  Analysis_Version =
    analysis_version,
  
  CV_Seeds =
    cv_seeds,
  
  Number_Folds =
    number_folds,
  
  Learning_Fractions =
    learning_fractions,
  
  Specifications =
    svr_specifications,
  
  Women_N =
    nrow(women_train),
  
  Men_N =
    nrow(men_train),
  
  Women_Predictors =
    women_predictors,
  
  Men_Predictors =
    men_predictors
)


svr_learning_results <-
  data.frame()


completed_combinations <-
  character(0)


if (
  file.exists(
    checkpoint_file
  )
) {
  
  cat("\n========================================\n")
  cat("12B CHECKPOINT FOUND\n")
  cat("========================================\n")
  
  
  checkpoint <- readRDS(
    checkpoint_file
  )
  
  
  signature_match <- isTRUE(
    
    all.equal(
      checkpoint$signature,
      checkpoint_signature,
      check.attributes = FALSE
    )
  )
  
  
  if (!signature_match) {
    
    stop(
      paste(
        "Existing 12B checkpoint does not match",
        "the current analysis."
      )
    )
  }
  
  
  svr_learning_results <-
    checkpoint$results
  
  
  completed_combinations <-
    checkpoint$completed
  
  
  cat(
    "Recovered combinations:",
    length(
      completed_combinations
    ),
    "\n"
  )
}


# ============================================================
# 18. RUN SVR LEARNING CURVES FOR ONE SEX
# ============================================================

run_svr_learning_curves <- function(
    data,
    predictors,
    assignments,
    sex_name,
    existing_results,
    existing_completed
) {
  
  current_results <-
    existing_results
  
  
  current_completed <-
    existing_completed
  
  
  for (seed_value in cv_seeds) {
    
    seed_assignments <- assignments[
      assignments$Seed ==
        seed_value,
      ,
      drop = FALSE
    ]
    
    
    for (
      fraction_value in
      learning_fractions
    ) {
      
      combination_key <- paste(
        
        sex_name,
        
        seed_value,
        
        sprintf(
          "%.2f",
          fraction_value
        ),
        
        sep = "|"
      )
      
      
      if (
        combination_key %in%
        current_completed
      ) {
        
        next
      }
      
      
      cat("\n========================================\n")
      
      cat(
        sex_name,
        "| Seed:",
        seed_value,
        "| Fraction:",
        fraction_value,
        "\n"
      )
      
      cat("========================================\n")
      
      
      training_records <-
        list()
      
      
      validation_records <-
        list()
      
      
      training_sizes <-
        numeric(
          number_folds
        )
      
      
      # ======================================================
      # INTERNAL 10-FOLD CV
      # ======================================================
      
      for (
        fold_value in
        seq_len(
          number_folds
        )
      ) {
        
        validation_rows <-
          seed_assignments$Internal_Row[
            seed_assignments$Fold ==
              fold_value
          ]
        
        
        available_training_rows <- setdiff(
          
          seq_len(
            nrow(data)
          ),
          
          validation_rows
        )
        
        
        training_rows <-
          sample_grouped_training_rows(
            
            data =
              data,
            
            available_rows =
              available_training_rows,
            
            fraction =
              fraction_value,
            
            seed_value =
              seed_value,
            
            fold_value =
              fold_value
          )
        
        
        training_sizes[
          fold_value
        ] <-
          length(
            training_rows
          )
        
        
        training_data <- data[
          training_rows,
          ,
          drop = FALSE
        ]
        
        
        validation_data <- data[
          validation_rows,
          ,
          drop = FALSE
        ]
        
        
        y_train <-
          training_data$HCY
        
        
        y_validation <-
          validation_data$HCY
        
        
        model_seed <- as.integer(
          
          seed_value +
            fold_value *
            100000
        )
        
        
        for (
          model_name in
          svr_model_names
        ) {
          
          model_result <-
            fit_predict_svr_caret(
              
              training_data =
                training_data,
              
              validation_data =
                validation_data,
              
              predictors =
                predictors,
              
              sex_name =
                sex_name,
              
              model_name =
                model_name,
              
              seed_value =
                model_seed
            )
          
          
          train_record <- data.frame(
            
            Model =
              model_name,
            
            Observed =
              y_train,
            
            Predicted =
              model_result$train,
            
            stringsAsFactors =
              FALSE
          )
          
          
          validation_record <- data.frame(
            
            Model =
              model_name,
            
            Observed =
              y_validation,
            
            Predicted =
              model_result$validation,
            
            stringsAsFactors =
              FALSE
          )
          
          
          training_records <- append(
            
            training_records,
            
            list(
              train_record
            )
          )
          
          
          validation_records <- append(
            
            validation_records,
            
            list(
              validation_record
            )
          )
          
          
          rm(
            model_result,
            train_record,
            validation_record
          )
          
          
          invisible(
            gc()
          )
        }
        
        
        rm(
          training_data,
          validation_data
        )
        
        
        invisible(
          gc()
        )
      }
      
      
      # ======================================================
      # COMBINE FOLD PREDICTIONS
      # ======================================================
      
      all_training_predictions <-
        dplyr::bind_rows(
          training_records
        )
      
      
      all_validation_predictions <-
        dplyr::bind_rows(
          validation_records
        )
      
      
      result_rows <-
        list()
      
      
      # ======================================================
      # CALCULATE METRICS
      # ======================================================
      
      for (
        model_name in
        svr_model_names
      ) {
        
        current_train <-
          all_training_predictions %>%
          
          dplyr::filter(
            Model ==
              model_name
          )
        
        
        current_validation <-
          all_validation_predictions %>%
          
          dplyr::filter(
            Model ==
              model_name
          )
        
        
        if (
          nrow(current_validation) !=
          nrow(data)
        ) {
          
          stop(
            paste(
              "Unexpected OOF validation size:",
              sex_name,
              seed_value,
              fraction_value,
              model_name
            )
          )
        }
        
        
        result_row <- data.frame(
          
          Sex =
            sex_name,
          
          Model =
            model_name,
          
          Seed =
            seed_value,
          
          Training_Fraction =
            fraction_value,
          
          Training_Percent =
            fraction_value *
            100,
          
          Mean_Training_N =
            mean(
              training_sizes
            ),
          
          Min_Training_N =
            min(
              training_sizes
            ),
          
          Max_Training_N =
            max(
              training_sizes
            ),
          
          Training_RMSE =
            calculate_rmse(
              current_train$Observed,
              current_train$Predicted
            ),
          
          Validation_RMSE =
            calculate_rmse(
              current_validation$Observed,
              current_validation$Predicted
            ),
          
          Training_MAE =
            calculate_mae(
              current_train$Observed,
              current_train$Predicted
            ),
          
          Validation_MAE =
            calculate_mae(
              current_validation$Observed,
              current_validation$Predicted
            ),
          
          Training_Predictive_R2 =
            calculate_predictive_r2(
              current_train$Observed,
              current_train$Predicted
            ),
          
          Validation_Predictive_R2 =
            calculate_predictive_r2(
              current_validation$Observed,
              current_validation$Predicted
            ),
          
          stringsAsFactors =
            FALSE
        )
        
        
        result_rows <- append(
          result_rows,
          list(
            result_row
          )
        )
      }
      
      
      combination_results <-
        dplyr::bind_rows(
          result_rows
        )
      
      
      if (
        nrow(combination_results) != 3
      ) {
        
        stop(
          "Each combination must produce exactly three SVR rows."
        )
      }
      
      
      current_results <-
        dplyr::bind_rows(
          current_results,
          combination_results
        )
      
      
      current_completed <- unique(
        
        c(
          current_completed,
          combination_key
        )
      )
      
      
      checkpoint <- list(
        
        signature =
          checkpoint_signature,
        
        completed =
          current_completed,
        
        results =
          current_results
      )
      
      
      saveRDS(
        checkpoint,
        checkpoint_file
      )
      
      
      cat(
        "12B checkpoint saved:",
        length(
          current_completed
        ),
        "of 160 combinations.\n"
      )
      
      
      rm(
        training_records,
        validation_records,
        all_training_predictions,
        all_validation_predictions,
        result_rows,
        combination_results,
        checkpoint
      )
      
      
      invisible(
        gc()
      )
    }
  }
  
  
  list(
    
    results =
      current_results,
    
    completed =
      current_completed
  )
}


# ============================================================
# 19. RUN WOMEN
# ============================================================

women_output <-
  run_svr_learning_curves(
    
    data =
      women_train,
    
    predictors =
      women_predictors,
    
    assignments =
      women_assignments,
    
    sex_name =
      "Women",
    
    existing_results =
      svr_learning_results,
    
    existing_completed =
      completed_combinations
  )


svr_learning_results <-
  women_output$results


completed_combinations <-
  women_output$completed


rm(
  women_output
)


# ============================================================
# 20. RUN MEN
# ============================================================

men_output <-
  run_svr_learning_curves(
    
    data =
      men_train,
    
    predictors =
      men_predictors,
    
    assignments =
      men_assignments,
    
    sex_name =
      "Men",
    
    existing_results =
      svr_learning_results,
    
    existing_completed =
      completed_combinations
  )


svr_learning_results <-
  men_output$results


completed_combinations <-
  men_output$completed


rm(
  men_output
)


# ============================================================
# 21. VALIDATE SVR RESULT SIZE
# ============================================================

expected_svr_combinations <-
  2 *
  length(cv_seeds) *
  length(learning_fractions)


expected_svr_rows <-
  expected_svr_combinations *
  length(svr_model_names)


cat("\n========================================\n")
cat("12B SVR RESULT VALIDATION\n")
cat("========================================\n")

cat(
  "Expected combinations:",
  expected_svr_combinations,
  "\n"
)

cat(
  "Observed combinations:",
  length(
    completed_combinations
  ),
  "\n"
)

cat(
  "Expected SVR rows:",
  expected_svr_rows,
  "\n"
)

cat(
  "Observed SVR rows:",
  nrow(
    svr_learning_results
  ),
  "\n"
)


if (
  length(completed_combinations) !=
  expected_svr_combinations
) {
  
  stop(
    "Incomplete 12B combination count."
  )
}


if (
  nrow(svr_learning_results) !=
  expected_svr_rows
) {
  
  stop(
    "Incomplete 12B SVR metric table."
  )
}


duplicate_check <- svr_learning_results %>%
  
  dplyr::count(
    Sex,
    Model,
    Seed,
    Training_Fraction
  ) %>%
  
  dplyr::filter(
    n != 1
  )


if (
  nrow(duplicate_check) > 0
) {
  
  stop(
    "Duplicated 12B SVR rows detected."
  )
}


# ============================================================
# 22. SAVE STANDALONE CORRECTED SVR RESULTS
# ============================================================

svr_learning_results <- svr_learning_results %>%
  
  dplyr::arrange(
    Sex,
    Model,
    Training_Fraction,
    Seed
  )


readr::write_csv(
  
  svr_learning_results,
  
  file.path(
    learning_curve_dir,
    "10_SVR_learning_curve_metrics_by_seed_caret.csv"
  )
)


# ============================================================
# 23. 100% CONSISTENCY CHECK AGAINST SCRIPT 09
#
# Validation metrics at 100% must reproduce the frozen
# Script 09 CV results.
# ============================================================

svr_100pct <- svr_learning_results %>%
  
  dplyr::filter(
    Training_Fraction == 1
  ) %>%
  
  dplyr::group_by(
    Sex,
    Model
  ) %>%
  
  dplyr::summarise(
    
    LC_Mean_RMSE =
      mean(
        Validation_RMSE
      ),
    
    LC_Mean_MAE =
      mean(
        Validation_MAE
      ),
    
    LC_Mean_Predictive_R2 =
      mean(
        Validation_Predictive_R2
      ),
    
    .groups =
      "drop"
  ) %>%
  
  dplyr::mutate(
    
    Kernel =
      dplyr::case_when(
        
        Model ==
          "SVR-Linear" ~
          "Linear",
        
        Model ==
          "SVR-Polynomial" ~
          "Polynomial",
        
        Model ==
          "SVR-RBF" ~
          "Radial",
        
        TRUE ~
          NA_character_
      )
  )


required_script09_columns <- c(
  "Sex",
  "Kernel",
  "Mean_RMSE",
  "Mean_MAE",
  "Mean_Predictive_R2"
)


missing_script09_columns <- setdiff(
  required_script09_columns,
  names(script09_cv_summary)
)


if (
  length(missing_script09_columns) > 0
) {
  
  stop(
    paste(
      "Script 09 CV summary is missing:",
      paste(
        missing_script09_columns,
        collapse = ", "
      )
    )
  )
}


script09_reference <- script09_cv_summary %>%
  
  dplyr::select(
    Sex,
    Kernel,
    Mean_RMSE,
    Mean_MAE,
    Mean_Predictive_R2
  ) %>%
  
  dplyr::rename(
    
    Script09_Mean_RMSE =
      Mean_RMSE,
    
    Script09_Mean_MAE =
      Mean_MAE,
    
    Script09_Mean_Predictive_R2 =
      Mean_Predictive_R2
  )


consistency_check <- svr_100pct %>%
  
  dplyr::left_join(
    
    script09_reference,
    
    by =
      c(
        "Sex",
        "Kernel"
      )
  ) %>%
  
  dplyr::mutate(
    
    RMSE_Difference =
      LC_Mean_RMSE -
      Script09_Mean_RMSE,
    
    MAE_Difference =
      LC_Mean_MAE -
      Script09_Mean_MAE,
    
    R2_Difference =
      LC_Mean_Predictive_R2 -
      Script09_Mean_Predictive_R2,
    
    Absolute_RMSE_Difference =
      abs(
        RMSE_Difference
      ),
    
    Absolute_MAE_Difference =
      abs(
        MAE_Difference
      ),
    
    Absolute_R2_Difference =
      abs(
        R2_Difference
      )
  ) %>%
  
  dplyr::arrange(
    Sex,
    Kernel
  )


readr::write_csv(
  
  consistency_check,
  
  file.path(
    learning_curve_dir,
    "11_SVR_100pct_consistency_vs_Script09.csv"
  )
)


cat("\n========================================\n")
cat("SVR 100% CONSISTENCY CHECK\n")
cat("========================================\n")

print(
  consistency_check
)


if (
  any(
    is.na(
      consistency_check$Script09_Mean_RMSE
    )
  )
) {
  
  stop(
    "Could not match all 12B SVR models to Script 09."
  )
}


comparison_tolerance <-
  1e-6


max_rmse_difference <- max(
  consistency_check$Absolute_RMSE_Difference,
  na.rm = TRUE
)


max_mae_difference <- max(
  consistency_check$Absolute_MAE_Difference,
  na.rm = TRUE
)


max_r2_difference <- max(
  consistency_check$Absolute_R2_Difference,
  na.rm = TRUE
)


cat(
  "Maximum absolute RMSE difference:",
  max_rmse_difference,
  "\n"
)


cat(
  "Maximum absolute MAE difference:",
  max_mae_difference,
  "\n"
)


cat(
  "Maximum absolute R2 difference:",
  max_r2_difference,
  "\n"
)


if (
  max_rmse_difference >
  comparison_tolerance ||
  max_mae_difference >
  comparison_tolerance ||
  max_r2_difference >
  comparison_tolerance
) {
  
  stop(
    paste(
      "SVR 100% results do not exactly reproduce",
      "Script 09 within tolerance.",
      "Main Script 12 files were NOT overwritten.",
      "Review 11_SVR_100pct_consistency_vs_Script09.csv."
    )
  )
}


cat(
  "SVR consistency with Script 09: PASSED\n"
)


# ============================================================
# 24. VALIDATE EXISTING SCRIPT 12 RESULTS
# ============================================================

required_existing_columns <- c(
  "Sex",
  "Model",
  "Seed",
  "Training_Fraction",
  "Training_Percent",
  "Mean_Training_N",
  "Min_Training_N",
  "Max_Training_N",
  "Training_RMSE",
  "Validation_RMSE",
  "Training_MAE",
  "Validation_MAE",
  "Training_Predictive_R2",
  "Validation_Predictive_R2"
)


missing_existing_columns <- setdiff(
  required_existing_columns,
  names(existing_learning_metrics)
)


if (
  length(missing_existing_columns) > 0
) {
  
  stop(
    paste(
      "Existing Script 12 table is missing:",
      paste(
        missing_existing_columns,
        collapse = ", "
      )
    )
  )
}


if (
  nrow(existing_learning_metrics) !=
  1120
) {
  
  stop(
    paste(
      "Existing Script 12 table should contain 1120 rows.",
      "Observed:",
      nrow(existing_learning_metrics)
    )
  )
}


# ============================================================
# 25. KEEP NON-SVR RESULTS UNCHANGED
# ============================================================

non_svr_metrics <- existing_learning_metrics %>%
  
  dplyr::filter(
    !(Model %in%
        svr_model_names)
  )


if (
  nrow(non_svr_metrics) !=
  640
) {
  
  stop(
    paste(
      "Expected 640 non-SVR rows.",
      "Observed:",
      nrow(non_svr_metrics)
    )
  )
}


# ============================================================
# 26. REPLACE ONLY SVR ROWS
# ============================================================

corrected_learning_metrics <- dplyr::bind_rows(
  
  non_svr_metrics,
  
  svr_learning_results
) %>%
  
  dplyr::arrange(
    Sex,
    Model,
    Training_Fraction,
    Seed
  )


if (
  nrow(corrected_learning_metrics) !=
  1120
) {
  
  stop(
    paste(
      "Corrected table should contain 1120 rows.",
      "Observed:",
      nrow(corrected_learning_metrics)
    )
  )
}


corrected_duplicate_check <-
  corrected_learning_metrics %>%
  
  dplyr::count(
    Sex,
    Model,
    Seed,
    Training_Fraction
  ) %>%
  
  dplyr::filter(
    n != 1
  )


if (
  nrow(
    corrected_duplicate_check
  ) > 0
) {
  
  stop(
    "Duplicated rows detected after SVR replacement."
  )
}


# ============================================================
# 27. RECREATE LEARNING-CURVE SUMMARY
# ============================================================

learning_summary <- corrected_learning_metrics %>%
  
  dplyr::group_by(
    Sex,
    Model,
    Training_Fraction,
    Training_Percent
  ) %>%
  
  dplyr::summarise(
    
    Mean_Training_N =
      mean(
        Mean_Training_N
      ),
    
    Mean_Training_RMSE =
      mean(
        Training_RMSE
      ),
    
    SD_Training_RMSE =
      stats::sd(
        Training_RMSE
      ),
    
    Mean_Validation_RMSE =
      mean(
        Validation_RMSE
      ),
    
    SD_Validation_RMSE =
      stats::sd(
        Validation_RMSE
      ),
    
    Mean_Training_MAE =
      mean(
        Training_MAE
      ),
    
    SD_Training_MAE =
      stats::sd(
        Training_MAE
      ),
    
    Mean_Validation_MAE =
      mean(
        Validation_MAE
      ),
    
    SD_Validation_MAE =
      stats::sd(
        Validation_MAE
      ),
    
    Mean_Training_Predictive_R2 =
      mean(
        Training_Predictive_R2
      ),
    
    SD_Training_Predictive_R2 =
      stats::sd(
        Training_Predictive_R2
      ),
    
    Mean_Validation_Predictive_R2 =
      mean(
        Validation_Predictive_R2
      ),
    
    SD_Validation_Predictive_R2 =
      stats::sd(
        Validation_Predictive_R2
      ),
    
    .groups =
      "drop"
  )


# ============================================================
# 28. RECREATE TRAINING-VALIDATION GAP SUMMARY
# ============================================================

learning_gap_summary <- learning_summary %>%
  
  dplyr::mutate(
    
    RMSE_Gap =
      Mean_Validation_RMSE -
      Mean_Training_RMSE,
    
    MAE_Gap =
      Mean_Validation_MAE -
      Mean_Training_MAE,
    
    R2_Gap =
      Mean_Training_Predictive_R2 -
      Mean_Validation_Predictive_R2
  )


# ============================================================
# 29. RECREATE 100% SUMMARY
# ============================================================

learning_100pct_summary <- learning_gap_summary %>%
  
  dplyr::filter(
    Training_Fraction == 1
  ) %>%
  
  dplyr::arrange(
    Sex,
    Mean_Validation_RMSE
  )


if (
  nrow(
    learning_100pct_summary
  ) != 14
) {
  
  stop(
    "The 100% summary should contain 14 rows."
  )
}


# ============================================================
# 30. OVERWRITE MAIN SCRIPT 12 CSV FILES
#
# This occurs ONLY after the Script 09 consistency check passes.
# ============================================================

readr::write_csv(
  
  corrected_learning_metrics,
  
  file.path(
    learning_curve_dir,
    "02_learning_curve_metrics_by_seed.csv"
  )
)


readr::write_csv(
  
  learning_summary,
  
  file.path(
    learning_curve_dir,
    "03_learning_curve_summary.csv"
  )
)


readr::write_csv(
  
  learning_gap_summary,
  
  file.path(
    learning_curve_dir,
    "04_learning_curve_gap_summary.csv"
  )
)


readr::write_csv(
  
  learning_100pct_summary,
  
  file.path(
    learning_curve_dir,
    "05_learning_curve_100pct_summary.csv"
  )
)


# ============================================================
# 31. PREPARE PLOT DATA
# ============================================================

training_plot_data <- learning_summary %>%
  
  dplyr::transmute(
    
    Sex,
    
    Model,
    
    Training_Percent,
    
    Stage =
      "Training",
    
    Mean_RMSE =
      Mean_Training_RMSE
  )


validation_plot_data <- learning_summary %>%
  
  dplyr::transmute(
    
    Sex,
    
    Model,
    
    Training_Percent,
    
    Stage =
      "Internal validation",
    
    Mean_RMSE =
      Mean_Validation_RMSE
  )


plot_data <- dplyr::bind_rows(
  
  training_plot_data,
  
  validation_plot_data
)


plot_data$Stage <- factor(
  
  plot_data$Stage,
  
  levels = c(
    "Training",
    "Internal validation"
  )
)


plot_data$Model <- factor(
  
  plot_data$Model,
  
  levels =
    all_model_names
)


# ============================================================
# 32. PLOT FUNCTION
#
# NOTE:
# "frozen workflows" has been removed from the figure.
# ============================================================

make_learning_curve_plot <- function(
    sex_name
) {
  
  plot_data %>%
    
    dplyr::filter(
      Sex ==
        sex_name
    ) %>%
    
    ggplot2::ggplot(
      
      ggplot2::aes(
        
        x =
          Training_Percent,
        
        y =
          Mean_RMSE,
        
        linetype =
          Stage,
        
        shape =
          Stage,
        
        group =
          Stage
      )
    ) +
    
    ggplot2::geom_line(
      linewidth = 0.7
    ) +
    
    ggplot2::geom_point(
      size = 2
    ) +
    
    ggplot2::facet_wrap(
      ~ Model,
      scales = "free_y",
      ncol = 3
    ) +
    
    ggplot2::scale_x_continuous(
      
      breaks =
        learning_fractions *
        100
    ) +
    
    ggplot2::labs(
      
      title =
        paste(
          "Learning curves -",
          sex_name
        ),
      
      subtitle =
        "TRAIN-only internal validation",
      
      x =
        "Fraction of internal CV training pool (%)",
      
      y =
        "RMSE of HCY",
      
      linetype =
        NULL,
      
      shape =
        NULL
    ) +
    
    ggplot2::theme_bw(
      base_size = 11
    ) +
    
    ggplot2::theme(
      
      legend.position =
        "bottom",
      
      plot.title =
        ggplot2::element_text(
          face = "bold"
        ),
      
      strip.text =
        ggplot2::element_text(
          face = "bold"
        )
    )
}


# ============================================================
# 33. REGENERATE WOMEN FIGURE
# ============================================================

women_plot <-
  make_learning_curve_plot(
    "Women"
  )


ggplot2::ggsave(
  
  filename =
    file.path(
      learning_curve_dir,
      "06_learning_curves_Women_RMSE.png"
    ),
  
  plot =
    women_plot,
  
  width =
    11,
  
  height =
    8,
  
  units =
    "in",
  
  dpi =
    300
)


# ============================================================
# 34. REGENERATE MEN FIGURE
# ============================================================

men_plot <-
  make_learning_curve_plot(
    "Men"
  )


ggplot2::ggsave(
  
  filename =
    file.path(
      learning_curve_dir,
      "07_learning_curves_Men_RMSE.png"
    ),
  
  plot =
    men_plot,
  
  width =
    11,
  
  height =
    8,
  
  units =
    "in",
  
  dpi =
    300
)


# ============================================================
# 35. SAVE SESSION INFORMATION FOR 12B
# ============================================================

writeLines(
  
  capture.output(
    sessionInfo()
  ),
  
  con =
    file.path(
      learning_curve_dir,
      "12_session_info_12B_SVR_caret.txt"
    )
)


# ============================================================
# 36. FINAL VALIDATION OF CORRECTED OUTPUTS
# ============================================================

if (
  nrow(
    corrected_learning_metrics
  ) != 1120
) {
  
  stop(
    "Final corrected metric table is not 1120 rows."
  )
}


if (
  nrow(
    learning_summary
  ) != 112
) {
  
  stop(
    paste(
      "Final learning summary should contain 112 rows.",
      "Observed:",
      nrow(
        learning_summary
      )
    )
  )
}


if (
  nrow(
    learning_100pct_summary
  ) != 14
) {
  
  stop(
    "Final 100% table is not 14 rows."
  )
}


required_final_files <- c(
  
  "02_learning_curve_metrics_by_seed.csv",
  
  "03_learning_curve_summary.csv",
  
  "04_learning_curve_gap_summary.csv",
  
  "05_learning_curve_100pct_summary.csv",
  
  "06_learning_curves_Women_RMSE.png",
  
  "07_learning_curves_Men_RMSE.png",
  
  "09_SVR_learning_curve_frozen_specifications.csv",
  
  "10_SVR_learning_curve_metrics_by_seed_caret.csv",
  
  "11_SVR_100pct_consistency_vs_Script09.csv",
  
  "12_session_info_12B_SVR_caret.txt"
)


missing_final_files <- setdiff(
  
  required_final_files,
  
  list.files(
    learning_curve_dir
  )
)


if (
  length(
    missing_final_files
  ) > 0
) {
  
  stop(
    paste(
      "Missing final files:",
      paste(
        missing_final_files,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 37. REMOVE 12B CHECKPOINT ONLY AFTER SUCCESS
# ============================================================

if (
  file.exists(
    checkpoint_file
  )
) {
  
  file.remove(
    checkpoint_file
  )
}


# ============================================================
# 38. DISPLAY FINAL 100% RESULTS
# ============================================================

cat("\n\n========================================\n")
cat("CORRECTED 100% LEARNING-CURVE RESULTS\n")
cat("========================================\n")


print(
  learning_100pct_summary
)


cat("\n========================================\n")
cat("WOMEN - CORRECTED 100% RESULTS\n")
cat("========================================\n")


print(
  
  learning_100pct_summary %>%
    
    dplyr::filter(
      Sex ==
        "Women"
    ) %>%
    
    dplyr::select(
      
      Model,
      
      Mean_Training_N,
      
      Mean_Training_RMSE,
      
      Mean_Validation_RMSE,
      
      RMSE_Gap,
      
      Mean_Training_Predictive_R2,
      
      Mean_Validation_Predictive_R2,
      
      R2_Gap
    )
)


cat("\n========================================\n")
cat("MEN - CORRECTED 100% RESULTS\n")
cat("========================================\n")


print(
  
  learning_100pct_summary %>%
    
    dplyr::filter(
      Sex ==
        "Men"
    ) %>%
    
    dplyr::select(
      
      Model,
      
      Mean_Training_N,
      
      Mean_Training_RMSE,
      
      Mean_Validation_RMSE,
      
      RMSE_Gap,
      
      Mean_Training_Predictive_R2,
      
      Mean_Validation_Predictive_R2,
      
      R2_Gap
    )
)


# ============================================================
# 39. FINAL MESSAGE
# ============================================================

cat("\n\n========================================\n")
cat("SCRIPT 12B COMPLETED SUCCESSFULLY\n")
cat("========================================\n")


cat(
  "Analysis version:",
  analysis_version,
  "\n"
)


cat(
  "Women TRAIN:",
  nrow(women_train),
  "\n"
)


cat(
  "Men TRAIN:",
  nrow(men_train),
  "\n"
)


cat(
  "Corrected models:",
  paste(
    svr_model_names,
    collapse = ", "
  ),
  "\n"
)


cat(
  "CV seeds:",
  length(cv_seeds),
  "\n"
)


cat(
  "Folds per seed:",
  number_folds,
  "\n"
)


cat(
  "Learning fractions:",
  paste(
    learning_fractions,
    collapse = ", "
  ),
  "\n"
)


cat(
  "SVR combinations:",
  expected_svr_combinations,
  "\n"
)


cat(
  "SVR metric rows:",
  nrow(
    svr_learning_results
  ),
  "\n"
)


cat(
  "Final combined metric rows:",
  nrow(
    corrected_learning_metrics
  ),
  "\n"
)


cat(
  "Final summary rows:",
  nrow(
    learning_summary
  ),
  "\n"
)


cat(
  "Script 09 consistency: PASSED\n"
)


cat(
  "Predictive R2: 1 - SSE/SST\n"
)


cat(
  "TEST: NEVER READ OR ACCESSED\n"
)


cat(
  "Figure subtitle: TRAIN-only internal validation\n"
)


cat(
  "'frozen workflows' removed from figures\n"
)


cat("========================================\n")


print(
  list.files(
    learning_curve_dir
  )
)