# ============================================================
# SCRIPT 11
# EXTREME GRADIENT BOOSTING (XGBOOST) - TRAIN ONLY
# ============================================================
#
# FINAL XGBOOST ANALYSIS
#
# DESIGN:
# - Fresh XGBoost run.
# - TRAIN only.
# - TEST is NEVER read or accessed.
# - Women TRAIN: n = 190.
# - Men TRAIN: n = 83.
# - All 21 eligible original predictors.
# - No Elastic Net variable selection.
# - No predictor transformations.
# - No centering or scaling.
#
# INTERNAL VALIDATION:
# - 10 prespecified random seeds.
# - 10-fold CV per seed.
# - Approximate HCY-quintile stratification.
# - Duplicate IDs remain in the same fold.
#
# XGBOOST TUNING:
# - nrounds
# - max_depth
# - eta
# - gamma
# - colsample_bytree
# - min_child_weight
# - subsample
# - reg_alpha
# - reg_lambda
#
# SEARCH:
# - Candidate space: 184,320 combinations.
# - Prespecified random search: 150 unique configurations.
# - Same 150 configurations for women and men.
#
# SELECTION:
# - Minimum mean OOF RMSE across the 10 CV configurations.
# - Predictive R2 = 1 - SSE/SST.
#
# CHECKPOINT:
# - Saved after every complete hyperparameter configuration.
# ============================================================


# ============================================================
# 1. PACKAGES
# ============================================================

required_packages <- c(
  "xgboost",
  "readr",
  "dplyr",
  "tibble"
)

missing_packages <- required_packages[
  !required_packages %in% rownames(installed.packages())
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(readr)
library(dplyr)
library(tibble)

# IMPORTANT:
# xgboost is intentionally NOT attached with library(xgboost).
# All XGBoost functions are called with xgboost:: to avoid
# namespace conflicts with dplyr.


# ============================================================
# 2. ANALYSIS VERSION
# ============================================================

analysis_version <- "XGB_FINAL_20261024"


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

output_dir <- "11_XGB_results"

if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}


# ============================================================
# 4. LOCATE ORIGINAL TRAIN FILES
#
# Only TRAIN files are searched.
# TEST files are never requested.
# ============================================================

find_input_file <- function(filename) {
  
  if (file.exists(filename)) {
    return(filename)
  }
  
  all_files <- list.files(
    ".",
    recursive = TRUE,
    full.names = TRUE
  )
  
  matches <- all_files[
    basename(all_files) == filename
  ]
  
  if (length(matches) == 0) {
    
    stop(
      paste(
        "Could not find required file:",
        filename
      )
    )
  }
  
  if (length(matches) > 1) {
    
    stop(
      paste(
        "Multiple copies of",
        filename,
        "were found:\n",
        paste(
          matches,
          collapse = "\n"
        )
      )
    )
  }
  
  return(matches[1])
}


women_train_file <- find_input_file(
  "01_women_TRAIN.csv"
)

men_train_file <- find_input_file(
  "03_men_TRAIN.csv"
)


cat("\n========================================\n")
cat("ORIGINAL TRAIN FILES\n")
cat("========================================\n")

cat(
  "Women:",
  women_train_file,
  "\n"
)

cat(
  "Men:",
  men_train_file,
  "\n"
)


# ============================================================
# 5. READ TRAIN DATA
# ============================================================

women_train <- readr::read_csv(
  women_train_file,
  show_col_types = FALSE
)

men_train <- readr::read_csv(
  men_train_file,
  show_col_types = FALSE
)


# ============================================================
# 6. STANDARDIZE VARIABLE NAMES
#
# Only column names are changed.
# Predictor values remain completely original.
# ============================================================

standardize_variable_names <- function(data) {
  
  rename_map <- c(
    "EDAD" = "Age",
    "TALLA" = "Height",
    "PESO" = "Weight",
    "IMC" = "BMI",
    "GRASAT" = "TotalFat",
    "GRASAVIS" = "VisceralFat",
    "MUSCULO" = "MuscleMass",
    "HGB" = "Hgb",
    "HTO" = "Hct",
    "VCM" = "MCV",
    "RDWCV" = "RDW_CV",
    "COL" = "TotalCholesterol"
  )
  
  for (old_name in names(rename_map)) {
    
    new_name <- unname(
      rename_map[old_name]
    )
    
    if (
      old_name %in% names(data) &&
      !(new_name %in% names(data))
    ) {
      
      names(data)[
        names(data) == old_name
      ] <- new_name
    }
  }
  
  return(data)
}


women_train <- standardize_variable_names(
  women_train
)

men_train <- standardize_variable_names(
  men_train
)


# ============================================================
# 7. FINAL 21 ELIGIBLE XGBOOST PREDICTORS
# ============================================================

xgb_predictors <- c(
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


if (length(xgb_predictors) != 21) {
  
  stop(
    "XGB predictor list must contain exactly 21 variables."
  )
}


cat("\n========================================\n")
cat("XGBOOST PREDICTORS\n")
cat("========================================\n")

print(
  xgb_predictors
)

cat(
  "Number of predictors:",
  length(xgb_predictors),
  "\n"
)


# ============================================================
# 8. VALIDATE TRAIN DATA
# ============================================================

check_xgb_dataset <- function(
    data,
    sex_name
) {
  
  required_columns <- c(
    "ID",
    "HCY",
    xgb_predictors
  )
  
  missing_columns <- setdiff(
    required_columns,
    names(data)
  )
  
  
  if (length(missing_columns) > 0) {
    
    stop(
      paste(
        sex_name,
        "is missing required columns:",
        paste(
          missing_columns,
          collapse = ", "
        )
      )
    )
  }
  
  
  numeric_variables <- c(
    "HCY",
    xgb_predictors
  )
  
  
  for (variable in numeric_variables) {
    
    values <- dplyr::pull(
      data,
      dplyr::all_of(variable)
    )
    
    
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


check_xgb_dataset(
  women_train,
  "WOMEN"
)

check_xgb_dataset(
  men_train,
  "MEN"
)


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


if (length(cv_seeds) != 10) {
  
  stop(
    "Exactly 10 CV seeds are required."
  )
}


# ============================================================
# 10. CPU SETTINGS
# ============================================================

detected_cores <- parallel::detectCores(
  logical = TRUE
)

if (
  is.na(detected_cores) ||
  detected_cores < 2
) {
  
  xgb_threads <- 1
  
} else {
  
  xgb_threads <- min(
    4,
    detected_cores - 1
  )
}


cat("\n========================================\n")
cat("XGBOOST COMPUTATION SETTINGS\n")
cat("========================================\n")

cat(
  "CPU threads:",
  xgb_threads,
  "\n"
)


# ============================================================
# 11. COMPLETE CANDIDATE HYPERPARAMETER SPACE
#
# Total:
#
# 4 x 5 x 4 x 4 x 3 x 4 x 3 x 4 x 4
# = 184,320 combinations
# ============================================================

xgb_candidate_pool <- expand.grid(
  
  nrounds = c(
    100,
    200,
    400,
    600
  ),
  
  max_depth = c(
    1,
    2,
    3,
    4,
    5
  ),
  
  eta = c(
    0.01,
    0.03,
    0.05,
    0.10
  ),
  
  gamma = c(
    0,
    0.1,
    0.5,
    1
  ),
  
  colsample_bytree = c(
    0.60,
    0.80,
    1.00
  ),
  
  min_child_weight = c(
    1,
    3,
    5,
    10
  ),
  
  subsample = c(
    0.60,
    0.80,
    1.00
  ),
  
  reg_alpha = c(
    0,
    0.1,
    0.5,
    1
  ),
  
  reg_lambda = c(
    0.5,
    1,
    5,
    10
  ),
  
  KEEP.OUT.ATTRS = FALSE,
  
  stringsAsFactors = FALSE
)


if (
  nrow(xgb_candidate_pool) != 184320
) {
  
  stop(
    paste(
      "Candidate space should contain 184320 combinations; obtained",
      nrow(xgb_candidate_pool)
    )
  )
}


# ============================================================
# 12. PRESPECIFIED RANDOM SEARCH
#
# The grid is generated BEFORE any model is fitted.
#
# The same 150 configurations are used for both sexes.
# ============================================================

xgb_search_seed <- 20261024

xgb_search_size <- 150


set.seed(
  xgb_search_seed
)


coverage_ok <- FALSE


while (!coverage_ok) {
  
  sampled_rows <- sample.int(
    
    n =
      nrow(xgb_candidate_pool),
    
    size =
      xgb_search_size,
    
    replace =
      FALSE
  )
  
  
  xgb_grid <- xgb_candidate_pool[
    sampled_rows,
    ,
    drop = FALSE
  ]
  
  
  coverage_checks <- c(
    
    setequal(
      unique(xgb_grid$nrounds),
      unique(xgb_candidate_pool$nrounds)
    ),
    
    setequal(
      unique(xgb_grid$max_depth),
      unique(xgb_candidate_pool$max_depth)
    ),
    
    setequal(
      unique(xgb_grid$eta),
      unique(xgb_candidate_pool$eta)
    ),
    
    setequal(
      unique(xgb_grid$gamma),
      unique(xgb_candidate_pool$gamma)
    ),
    
    setequal(
      unique(xgb_grid$colsample_bytree),
      unique(xgb_candidate_pool$colsample_bytree)
    ),
    
    setequal(
      unique(xgb_grid$min_child_weight),
      unique(xgb_candidate_pool$min_child_weight)
    ),
    
    setequal(
      unique(xgb_grid$subsample),
      unique(xgb_candidate_pool$subsample)
    ),
    
    setequal(
      unique(xgb_grid$reg_alpha),
      unique(xgb_candidate_pool$reg_alpha)
    ),
    
    setequal(
      unique(xgb_grid$reg_lambda),
      unique(xgb_candidate_pool$reg_lambda)
    )
  )
  
  
  coverage_ok <- all(
    coverage_checks
  )
}


xgb_grid <- xgb_grid %>%
  
  dplyr::arrange(
    nrounds,
    max_depth,
    eta,
    gamma,
    colsample_bytree,
    min_child_weight,
    subsample,
    reg_alpha,
    reg_lambda
  )


xgb_grid$Config_ID <- seq_len(
  nrow(xgb_grid)
)


xgb_grid <- xgb_grid %>%
  
  dplyr::select(
    Config_ID,
    dplyr::everything()
  )


if (
  nrow(xgb_grid) != 150
) {
  
  stop(
    "The XGB search grid must contain exactly 150 configurations."
  )
}


duplicate_grid_rows <- duplicated(
  
  xgb_grid %>%
    
    dplyr::select(
      -Config_ID
    )
)


if (any(duplicate_grid_rows)) {
  
  stop(
    "Duplicated XGB configurations detected."
  )
}


# Freeze search grid before model fitting.

readr::write_csv(
  
  xgb_grid,
  
  file.path(
    output_dir,
    "00_XGB_search_grid.csv"
  )
)


cat("\n========================================\n")
cat("XGBOOST SEARCH GRID\n")
cat("========================================\n")

cat(
  "Candidate space:",
  nrow(xgb_candidate_pool),
  "\n"
)

cat(
  "Configurations selected:",
  nrow(xgb_grid),
  "\n"
)

cat(
  "Search seed:",
  xgb_search_seed,
  "\n"
)


# ============================================================
# 13. CREATE GROUPED HCY-STRATIFIED CV FOLDS
#
# Duplicate IDs always remain within the same fold.
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
  
  
  assignment_list <- list()
  
  
  for (seed_value in seeds) {
    
    set.seed(
      seed_value
    )
    
    
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
        
        by =
          "ID"
      ) %>%
      
      dplyr::mutate(
        Seed = seed_value
      ) %>%
      
      dplyr::select(
        Seed,
        Fold,
        Internal_Row,
        ID,
        HCY_Stratum
      )
    
    
    assignment_list <- append(
      
      assignment_list,
      
      list(
        current_assignment
      )
    )
  }
  
  
  assignments <- dplyr::bind_rows(
    assignment_list
  )
  
  
  expected_rows <-
    nrow(data) *
    length(seeds)
  
  
  if (
    nrow(assignments) != expected_rows
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
        dplyr::n_distinct(Fold),
      
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
        dplyr::n_distinct(Fold),
      
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
  
  
  return(assignments)
}


# ============================================================
# 14. OFFICIAL PERFORMANCE METRICS
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
# 15. CHECKPOINT DATA SIGNATURE
# ============================================================

create_data_signature <- function(
    data,
    predictors
) {
  
  predictor_sums <- vapply(
    
    predictors,
    
    function(variable) {
      
      values <- dplyr::pull(
        data,
        dplyr::all_of(variable)
      )
      
      sum(values)
    },
    
    numeric(1)
  )
  
  
  predictor_sums_sq <- vapply(
    
    predictors,
    
    function(variable) {
      
      values <- dplyr::pull(
        data,
        dplyr::all_of(variable)
      )
      
      sum(values^2)
    },
    
    numeric(1)
  )
  
  
  signature <- list(
    
    Analysis_Version =
      analysis_version,
    
    N =
      nrow(data),
    
    Unique_IDs =
      dplyr::n_distinct(
        data$ID
      ),
    
    ID_Order =
      paste(
        as.character(data$ID),
        collapse = "|"
      ),
    
    HCY_Sum =
      sum(data$HCY),
    
    HCY_SumSq =
      sum(data$HCY^2),
    
    Predictor_Sums =
      predictor_sums,
    
    Predictor_SumsSq =
      predictor_sums_sq
  )
  
  
  return(signature)
}


# ============================================================
# 16. XGBOOST TUNING FUNCTION
# ============================================================

tune_xgboost <- function(
    data,
    predictors,
    sex_name,
    assignments,
    tuning_grid,
    seeds,
    k,
    checkpoint_file
) {
  
  data_signature <- create_data_signature(
    data,
    predictors
  )
  
  
  xgb_settings <- list(
    
    Analysis_Version =
      analysis_version,
    
    booster =
      "gbtree",
    
    objective =
      "reg:squarederror",
    
    eval_metric =
      "rmse",
    
    tree_method =
      "hist",
    
    search_seed =
      xgb_search_seed,
    
    search_size =
      xgb_search_size,
    
    seeds =
      seeds,
    
    folds =
      k
  )
  
  
  results <- tibble::tibble()
  
  completed_configs <- integer(0)
  
  
  # ----------------------------------------------------------
  # LOAD CURRENT CHECKPOINT
  # ----------------------------------------------------------
  
  if (
    file.exists(
      checkpoint_file
    )
  ) {
    
    cat("\n========================================\n")
    
    cat(
      "CHECKPOINT FOUND FOR",
      sex_name,
      "\n"
    )
    
    cat("========================================\n")
    
    
    checkpoint <- readRDS(
      checkpoint_file
    )
    
    
    grid_match <- isTRUE(
      
      all.equal(
        
        checkpoint$tuning_grid,
        
        tuning_grid,
        
        check.attributes =
          FALSE
      )
    )
    
    
    predictors_match <- identical(
      
      checkpoint$predictors,
      
      predictors
    )
    
    
    signature_match <- isTRUE(
      
      all.equal(
        
        checkpoint$data_signature,
        
        data_signature,
        
        check.attributes =
          FALSE
      )
    )
    
    
    settings_match <- isTRUE(
      
      all.equal(
        
        checkpoint$xgb_settings,
        
        xgb_settings,
        
        check.attributes =
          FALSE
      )
    )
    
    
    if (
      !grid_match ||
      !predictors_match ||
      !signature_match ||
      !settings_match
    ) {
      
      stop(
        paste(
          "Existing checkpoint for",
          sex_name,
          "does not match this analysis."
        )
      )
    }
    
    
    results <- checkpoint$results
    
    completed_configs <-
      checkpoint$completed_configs
    
    
    cat(
      "Recovered",
      length(completed_configs),
      "of",
      nrow(tuning_grid),
      "configurations.\n"
    )
  }
  
  
  # ----------------------------------------------------------
  # LOOP OVER HYPERPARAMETER CONFIGURATIONS
  # ----------------------------------------------------------
  
  for (
    config_index in
    seq_len(
      nrow(tuning_grid)
    )
  ) {
    
    config_id <- tuning_grid$Config_ID[
      config_index
    ]
    
    
    if (
      config_id %in%
      completed_configs
    ) {
      
      next
    }
    
    
    current_config <- tuning_grid[
      config_index,
      ,
      drop = FALSE
    ]
    
    
    cat("\n----------------------------------------\n")
    
    cat(
      sex_name,
      "- XGB configuration",
      config_id,
      "of",
      nrow(tuning_grid),
      "\n"
    )
    
    
    cat(
      "nrounds =",
      current_config$nrounds,
      "| depth =",
      current_config$max_depth,
      "| eta =",
      current_config$eta,
      "| gamma =",
      current_config$gamma,
      "| colsample =",
      current_config$colsample_bytree,
      "| child =",
      current_config$min_child_weight,
      "| subsample =",
      current_config$subsample,
      "| alpha =",
      current_config$reg_alpha,
      "| lambda =",
      current_config$reg_lambda,
      "\n"
    )
    
    
    config_seed_results <- list()
    
    
    # --------------------------------------------------------
    # LOOP OVER 10 CV SEEDS
    # --------------------------------------------------------
    
    for (seed_value in seeds) {
      
      seed_assignments <- assignments[
        assignments$Seed ==
          seed_value,
        ,
        drop = FALSE
      ]
      
      
      oof_predictions <- rep(
        NA_real_,
        nrow(data)
      )
      
      
      # ------------------------------------------------------
      # LOOP OVER 10 CV FOLDS
      # ------------------------------------------------------
      
      for (
        fold_value in
        seq_len(k)
      ) {
        
        validation_rows <-
          seed_assignments$Internal_Row[
            seed_assignments$Fold ==
              fold_value
          ]
        
        
        training_rows <- setdiff(
          
          seq_len(
            nrow(data)
          ),
          
          validation_rows
        )
        
        
        training_matrix <- as.matrix(
          
          data[
            training_rows,
            predictors,
            drop = FALSE
          ]
        )
        
        
        validation_matrix <- as.matrix(
          
          data[
            validation_rows,
            predictors,
            drop = FALSE
          ]
        )
        
        
        training_labels <-
          data$HCY[
            training_rows
          ]
        
        
        dtrain <- xgboost::xgb.DMatrix(
          
          data =
            training_matrix,
          
          label =
            training_labels
        )
        
        
        # Same algorithmic seed for all configurations
        # within the same CV seed/fold pair.
        
        model_seed <- as.integer(
          
          seed_value +
            fold_value * 100000
        )
        
        
        set.seed(
          model_seed
        )
        
        
        xgb_parameters <- list(
          
          booster =
            "gbtree",
          
          objective =
            "reg:squarederror",
          
          eval_metric =
            "rmse",
          
          tree_method =
            "hist",
          
          max_depth =
            as.integer(
              current_config$max_depth
            ),
          
          eta =
            as.numeric(
              current_config$eta
            ),
          
          gamma =
            as.numeric(
              current_config$gamma
            ),
          
          colsample_bytree =
            as.numeric(
              current_config$colsample_bytree
            ),
          
          min_child_weight =
            as.numeric(
              current_config$min_child_weight
            ),
          
          subsample =
            as.numeric(
              current_config$subsample
            ),
          
          alpha =
            as.numeric(
              current_config$reg_alpha
            ),
          
          lambda =
            as.numeric(
              current_config$reg_lambda
            ),
          
          seed =
            model_seed,
          
          nthread =
            xgb_threads,
          
          verbosity =
            0
        )
        
        
        xgb_fit <- xgboost::xgb.train(
          
          params =
            xgb_parameters,
          
          data =
            dtrain,
          
          nrounds =
            as.integer(
              current_config$nrounds
            ),
          
          verbose =
            0
        )
        
        
        fold_predictions <- predict(
          
          xgb_fit,
          
          validation_matrix
        )
        
        
        oof_predictions[
          validation_rows
        ] <- as.numeric(
          fold_predictions
        )
        
        
        rm(
          xgb_fit,
          dtrain,
          training_matrix,
          validation_matrix,
          training_labels,
          fold_predictions
        )
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
            "Missing OOF prediction:",
            sex_name,
            "configuration",
            config_id,
            "seed",
            seed_value
          )
        )
      }
      
      
      seed_result <- data.frame(
        
        Sex =
          sex_name,
        
        Config_ID =
          as.integer(
            config_id
          ),
        
        nrounds =
          as.integer(
            current_config$nrounds
          ),
        
        max_depth =
          as.integer(
            current_config$max_depth
          ),
        
        eta =
          as.numeric(
            current_config$eta
          ),
        
        gamma =
          as.numeric(
            current_config$gamma
          ),
        
        colsample_bytree =
          as.numeric(
            current_config$colsample_bytree
          ),
        
        min_child_weight =
          as.numeric(
            current_config$min_child_weight
          ),
        
        subsample =
          as.numeric(
            current_config$subsample
          ),
        
        reg_alpha =
          as.numeric(
            current_config$reg_alpha
          ),
        
        reg_lambda =
          as.numeric(
            current_config$reg_lambda
          ),
        
        Seed =
          as.integer(
            seed_value
          ),
        
        N =
          nrow(data),
        
        RMSE =
          calculate_rmse(
            data$HCY,
            oof_predictions
          ),
        
        MAE =
          calculate_mae(
            data$HCY,
            oof_predictions
          ),
        
        Predictive_R2 =
          calculate_predictive_r2(
            data$HCY,
            oof_predictions
          )
      )
      
      
      config_seed_results <- append(
        
        config_seed_results,
        
        list(
          seed_result
        )
      )
    }
    
    
    current_results <- dplyr::bind_rows(
      config_seed_results
    )
    
    
    if (
      nrow(current_results) !=
      length(seeds)
    ) {
      
      stop(
        paste(
          "Configuration",
          config_id,
          "did not generate 10 seed results."
        )
      )
    }
    
    
    results <- dplyr::bind_rows(
      
      results,
      
      current_results
    )
    
    
    completed_configs <- sort(
      
      unique(
        
        c(
          completed_configs,
          config_id
        )
      )
    )
    
    
    checkpoint <- list(
      
      sex =
        sex_name,
      
      predictors =
        predictors,
      
      tuning_grid =
        tuning_grid,
      
      data_signature =
        data_signature,
      
      xgb_settings =
        xgb_settings,
      
      completed_configs =
        completed_configs,
      
      results =
        results
    )
    
    
    saveRDS(
      
      checkpoint,
      
      checkpoint_file
    )
    
    
    cat(
      "Checkpoint saved:",
      length(completed_configs),
      "/",
      nrow(tuning_grid),
      "\n"
    )
    
    
    rm(
      current_results,
      config_seed_results,
      checkpoint
    )
    
    
    invisible(
      gc()
    )
  }
  
  
  # ----------------------------------------------------------
  # FINAL TUNING VALIDATION
  # ----------------------------------------------------------
  
  expected_rows <-
    nrow(tuning_grid) *
    length(seeds)
  
  
  if (
    nrow(results) != expected_rows
  ) {
    
    stop(
      paste(
        sex_name,
        "produced",
        nrow(results),
        "rows; expected",
        expected_rows
      )
    )
  }
  
  
  if (
    dplyr::n_distinct(
      results$Config_ID
    ) != nrow(tuning_grid)
  ) {
    
    stop(
      paste(
        sex_name,
        "does not contain all XGB configurations."
      )
    )
  }
  
  
  return(results)
}


# ============================================================
# 17. CREATE CV ASSIGNMENTS
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
# 18. WOMEN XGBOOST TUNING
# ============================================================

women_checkpoint <- file.path(
  
  output_dir,
  
  "checkpoint_women_XGB.rds"
)


women_xgb_tuning <- tune_xgboost(
  
  data =
    women_train,
  
  predictors =
    xgb_predictors,
  
  sex_name =
    "Women",
  
  assignments =
    women_assignments,
  
  tuning_grid =
    xgb_grid,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds,
  
  checkpoint_file =
    women_checkpoint
)


# ============================================================
# 19. MEN XGBOOST TUNING
# ============================================================

men_checkpoint <- file.path(
  
  output_dir,
  
  "checkpoint_men_XGB.rds"
)


men_xgb_tuning <- tune_xgboost(
  
  data =
    men_train,
  
  predictors =
    xgb_predictors,
  
  sex_name =
    "Men",
  
  assignments =
    men_assignments,
  
  tuning_grid =
    xgb_grid,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds,
  
  checkpoint_file =
    men_checkpoint
)


# ============================================================
# 20. COMBINE TUNING RESULTS
# ============================================================

all_xgb_tuning <- dplyr::bind_rows(
  
  women_xgb_tuning,
  
  men_xgb_tuning
)


# ============================================================
# 21. TUNING SUMMARY ACROSS 10 SEEDS
# ============================================================

xgb_tuning_summary <- all_xgb_tuning %>%
  
  dplyr::group_by(
    
    Sex,
    Config_ID,
    nrounds,
    max_depth,
    eta,
    gamma,
    colsample_bytree,
    min_child_weight,
    subsample,
    reg_alpha,
    reg_lambda
  ) %>%
  
  dplyr::summarise(
    
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
# 22. BEST CONFIGURATION WITHIN EACH CV SEED
#
# Descriptive tuning-stability analysis.
# This does NOT determine the final model.
# ============================================================

xgb_best_by_seed <- all_xgb_tuning %>%
  
  dplyr::group_by(
    Sex,
    Seed
  ) %>%
  
  dplyr::arrange(
    RMSE,
    Config_ID,
    .by_group = TRUE
  ) %>%
  
  dplyr::slice_head(
    n = 1
  ) %>%
  
  dplyr::ungroup()


# ============================================================
# 23. SELECT FINAL HYPERPARAMETERS
#
# Primary criterion:
# minimum mean OOF RMSE across the 10 CV configurations.
# ============================================================

selected_xgb_hyperparameters <- xgb_tuning_summary %>%
  
  dplyr::group_by(
    Sex
  ) %>%
  
  dplyr::arrange(
    Mean_RMSE,
    Config_ID,
    .by_group = TRUE
  ) %>%
  
  dplyr::slice_head(
    n = 1
  ) %>%
  
  dplyr::ungroup()


cat("\n\n========================================\n")
cat("SELECTED XGBOOST HYPERPARAMETERS\n")
cat("========================================\n")

print(
  selected_xgb_hyperparameters
)


# ============================================================
# 24. SELECTED CONFIGURATION CV METRICS BY SEED
# ============================================================

selected_keys <- selected_xgb_hyperparameters %>%
  
  dplyr::select(
    
    Sex,
    Config_ID,
    nrounds,
    max_depth,
    eta,
    gamma,
    colsample_bytree,
    min_child_weight,
    subsample,
    reg_alpha,
    reg_lambda
  )


final_xgb_cv_by_seed <- all_xgb_tuning %>%
  
  dplyr::inner_join(
    
    selected_keys,
    
    by = c(
      "Sex",
      "Config_ID",
      "nrounds",
      "max_depth",
      "eta",
      "gamma",
      "colsample_bytree",
      "min_child_weight",
      "subsample",
      "reg_alpha",
      "reg_lambda"
    )
  )


if (
  nrow(final_xgb_cv_by_seed) != 20
) {
  
  stop(
    paste(
      "Expected 20 final XGB CV rows; obtained",
      nrow(final_xgb_cv_by_seed)
    )
  )
}


# ============================================================
# 25. FINAL INTERNAL CV SUMMARY
# ============================================================

final_xgb_cv_summary <- final_xgb_cv_by_seed %>%
  
  dplyr::group_by(
    
    Sex,
    Config_ID,
    nrounds,
    max_depth,
    eta,
    gamma,
    colsample_bytree,
    min_child_weight,
    subsample,
    reg_alpha,
    reg_lambda
  ) %>%
  
  dplyr::summarise(
    
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
# 26. FIT FINAL FULL-TRAIN XGBOOST MODEL
# ============================================================

fit_final_xgb <- function(
    data,
    predictors,
    sex_name,
    selected_row
) {
  
  train_matrix <- as.matrix(
    
    data[
      ,
      predictors,
      drop = FALSE
    ]
  )
  
  
  dtrain <- xgboost::xgb.DMatrix(
    
    data =
      train_matrix,
    
    label =
      data$HCY
  )
  
  
  final_seed <- 123
  
  
  set.seed(
    final_seed
  )
  
  
  final_parameters <- list(
    
    booster =
      "gbtree",
    
    objective =
      "reg:squarederror",
    
    eval_metric =
      "rmse",
    
    tree_method =
      "hist",
    
    max_depth =
      as.integer(
        selected_row$max_depth
      ),
    
    eta =
      as.numeric(
        selected_row$eta
      ),
    
    gamma =
      as.numeric(
        selected_row$gamma
      ),
    
    colsample_bytree =
      as.numeric(
        selected_row$colsample_bytree
      ),
    
    min_child_weight =
      as.numeric(
        selected_row$min_child_weight
      ),
    
    subsample =
      as.numeric(
        selected_row$subsample
      ),
    
    alpha =
      as.numeric(
        selected_row$reg_alpha
      ),
    
    lambda =
      as.numeric(
        selected_row$reg_lambda
      ),
    
    seed =
      final_seed,
    
    nthread =
      xgb_threads,
    
    verbosity =
      0
  )
  
  
  final_model <- xgboost::xgb.train(
    
    params =
      final_parameters,
    
    data =
      dtrain,
    
    nrounds =
      as.integer(
        selected_row$nrounds
      ),
    
    verbose =
      0
  )
  
  
  train_predictions <- predict(
    
    final_model,
    
    train_matrix
  )
  
  
  train_metrics <- data.frame(
    
    Sex =
      sex_name,
    
    N =
      nrow(data),
    
    nrounds =
      as.integer(
        selected_row$nrounds
      ),
    
    max_depth =
      as.integer(
        selected_row$max_depth
      ),
    
    eta =
      as.numeric(
        selected_row$eta
      ),
    
    gamma =
      as.numeric(
        selected_row$gamma
      ),
    
    colsample_bytree =
      as.numeric(
        selected_row$colsample_bytree
      ),
    
    min_child_weight =
      as.numeric(
        selected_row$min_child_weight
      ),
    
    subsample =
      as.numeric(
        selected_row$subsample
      ),
    
    reg_alpha =
      as.numeric(
        selected_row$reg_alpha
      ),
    
    reg_lambda =
      as.numeric(
        selected_row$reg_lambda
      ),
    
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
      )
  )
  
  
  # ----------------------------------------------------------
  # VARIABLE IMPORTANCE
  #
  # Different XGBoost versions may use "Feature" or "Features".
  # Both are supported here.
  # ----------------------------------------------------------
  
  importance_raw <- as.data.frame(
    
    xgboost::xgb.importance(
      
      model =
        final_model,
      
      feature_names =
        predictors
    )
  )
  
  
  importance_table <- data.frame(
    
    Variable =
      predictors,
    
    stringsAsFactors =
      FALSE
  )
  
  
  if (
    nrow(importance_raw) > 0
  ) {
    
    if (
      "Features" %in%
      names(importance_raw)
    ) {
      
      names(importance_raw)[
        names(importance_raw) ==
          "Features"
      ] <- "Variable"
    }
    
    
    if (
      "Feature" %in%
      names(importance_raw)
    ) {
      
      names(importance_raw)[
        names(importance_raw) ==
          "Feature"
      ] <- "Variable"
    }
    
    
    if (
      !("Variable" %in%
        names(importance_raw))
    ) {
      
      stop(
        "Could not identify the XGBoost feature-name column."
      )
    }
    
    
    required_importance_columns <- c(
      "Variable",
      "Gain",
      "Cover",
      "Frequency"
    )
    
    
    missing_importance_columns <- setdiff(
      
      required_importance_columns,
      
      names(importance_raw)
    )
    
    
    if (
      length(
        missing_importance_columns
      ) > 0
    ) {
      
      stop(
        paste(
          "Missing XGBoost importance columns:",
          paste(
            missing_importance_columns,
            collapse = ", "
          )
        )
      )
    }
    
    
    importance_raw <- importance_raw[
      ,
      required_importance_columns,
      drop = FALSE
    ]
    
    
    importance_table <- importance_table %>%
      
      dplyr::left_join(
        
        importance_raw,
        
        by =
          "Variable"
      )
    
  } else {
    
    importance_table$Gain <- 0
    
    importance_table$Cover <- 0
    
    importance_table$Frequency <- 0
  }
  
  
  importance_table$Gain[
    is.na(
      importance_table$Gain
    )
  ] <- 0
  
  
  importance_table$Cover[
    is.na(
      importance_table$Cover
    )
  ] <- 0
  
  
  importance_table$Frequency[
    is.na(
      importance_table$Frequency
    )
  ] <- 0
  
  
  importance_table <- importance_table %>%
    
    dplyr::arrange(
      dplyr::desc(Gain)
    ) %>%
    
    dplyr::mutate(
      
      Sex =
        sex_name,
      
      Rank =
        dplyr::row_number()
    ) %>%
    
    dplyr::select(
      Sex,
      Rank,
      Variable,
      Gain,
      Cover,
      Frequency
    )
  
  
  return(
    
    list(
      
      model =
        final_model,
      
      metrics =
        train_metrics,
      
      importance =
        importance_table,
      
      parameters =
        final_parameters
    )
  )
}


# ============================================================
# 27. SELECT FINAL WOMEN AND MEN CONFIGURATIONS
# ============================================================

women_selected_xgb <- selected_xgb_hyperparameters %>%
  
  dplyr::filter(
    Sex == "Women"
  )


men_selected_xgb <- selected_xgb_hyperparameters %>%
  
  dplyr::filter(
    Sex == "Men"
  )


if (
  nrow(women_selected_xgb) != 1
) {
  
  stop(
    "Expected exactly one WOMEN XGB configuration."
  )
}


if (
  nrow(men_selected_xgb) != 1
) {
  
  stop(
    "Expected exactly one MEN XGB configuration."
  )
}


# ============================================================
# 28. FIT FINAL FULL-TRAIN MODELS
# ============================================================

women_final_xgb <- fit_final_xgb(
  
  data =
    women_train,
  
  predictors =
    xgb_predictors,
  
  sex_name =
    "Women",
  
  selected_row =
    women_selected_xgb
)


men_final_xgb <- fit_final_xgb(
  
  data =
    men_train,
  
  predictors =
    xgb_predictors,
  
  sex_name =
    "Men",
  
  selected_row =
    men_selected_xgb
)


# ============================================================
# 29. COMBINE FINAL TRAIN RESULTS
# ============================================================

xgb_train_metrics <- dplyr::bind_rows(
  
  women_final_xgb$metrics,
  
  men_final_xgb$metrics
)


xgb_variable_importance <- dplyr::bind_rows(
  
  women_final_xgb$importance,
  
  men_final_xgb$importance
)


if (
  nrow(xgb_train_metrics) != 2
) {
  
  stop(
    "Expected exactly two XGB TRAIN metric rows."
  )
}


if (
  nrow(xgb_variable_importance) != 42
) {
  
  stop(
    paste(
      "Expected 42 XGB importance rows; obtained",
      nrow(xgb_variable_importance)
    )
  )
}


# ============================================================
# 30. SAVE FINAL XGBOOST MODELS
# ============================================================

women_model_file <- file.path(
  
  output_dir,
  
  "women_XGB_model.json"
)


men_model_file <- file.path(
  
  output_dir,
  
  "men_XGB_model.json"
)


xgboost::xgb.save(
  
  women_final_xgb$model,
  
  women_model_file
)


xgboost::xgb.save(
  
  men_final_xgb$model,
  
  men_model_file
)


# ============================================================
# 31. CREATE FINAL MODEL BUNDLES
# ============================================================

women_xgb_bundle <- list(
  
  Analysis_Version =
    analysis_version,
  
  sex =
    "Women",
  
  predictors =
    xgb_predictors,
  
  model_file =
    "women_XGB_model.json",
  
  selected_hyperparameters =
    women_selected_xgb,
  
  final_parameters =
    women_final_xgb$parameters,
  
  cv_seeds =
    cv_seeds,
  
  number_folds =
    number_folds,
  
  search_seed =
    xgb_search_seed,
  
  search_size =
    xgb_search_size,
  
  candidate_space_size =
    nrow(xgb_candidate_pool),
  
  preprocessing =
    "Original untransformed predictors; no centering or scaling",
  
  variable_selection =
    "All 21 eligible predictors",
  
  tuning_selection =
    "Minimum mean OOF RMSE across 10 prespecified CV configurations",
  
  r2_definition =
    "1 - SSE/SST",
  
  final_fit_seed =
    123,
  
  test_accessed =
    FALSE
)


men_xgb_bundle <- list(
  
  Analysis_Version =
    analysis_version,
  
  sex =
    "Men",
  
  predictors =
    xgb_predictors,
  
  model_file =
    "men_XGB_model.json",
  
  selected_hyperparameters =
    men_selected_xgb,
  
  final_parameters =
    men_final_xgb$parameters,
  
  cv_seeds =
    cv_seeds,
  
  number_folds =
    number_folds,
  
  search_seed =
    xgb_search_seed,
  
  search_size =
    xgb_search_size,
  
  candidate_space_size =
    nrow(xgb_candidate_pool),
  
  preprocessing =
    "Original untransformed predictors; no centering or scaling",
  
  variable_selection =
    "All 21 eligible predictors",
  
  tuning_selection =
    "Minimum mean OOF RMSE across 10 prespecified CV configurations",
  
  r2_definition =
    "1 - SSE/SST",
  
  final_fit_seed =
    123,
  
  test_accessed =
    FALSE
)


# ============================================================
# 32. SAVE FINAL TABLES
# ============================================================

readr::write_csv(
  
  all_xgb_tuning,
  
  file.path(
    output_dir,
    "01_XGB_tuning_by_seed.csv"
  )
)


readr::write_csv(
  
  xgb_tuning_summary,
  
  file.path(
    output_dir,
    "02_XGB_tuning_summary.csv"
  )
)


readr::write_csv(
  
  selected_xgb_hyperparameters,
  
  file.path(
    output_dir,
    "03_XGB_selected_hyperparameters.csv"
  )
)


readr::write_csv(
  
  final_xgb_cv_by_seed,
  
  file.path(
    output_dir,
    "04_XGB_CV_metrics_by_seed.csv"
  )
)


readr::write_csv(
  
  final_xgb_cv_summary,
  
  file.path(
    output_dir,
    "05_XGB_CV_summary.csv"
  )
)


readr::write_csv(
  
  xgb_train_metrics,
  
  file.path(
    output_dir,
    "06_XGB_train_metrics.csv"
  )
)


readr::write_csv(
  
  xgb_variable_importance,
  
  file.path(
    output_dir,
    "07_XGB_variable_importance.csv"
  )
)


readr::write_csv(
  
  xgb_best_by_seed,
  
  file.path(
    output_dir,
    "08_XGB_best_hyperparameters_by_seed.csv"
  )
)


saveRDS(
  
  women_xgb_bundle,
  
  file.path(
    output_dir,
    "women_XGB_bundle.rds"
  )
)


saveRDS(
  
  men_xgb_bundle,
  
  file.path(
    output_dir,
    "men_XGB_bundle.rds"
  )
)


# ============================================================
# 33. VERIFY FINAL OUTPUT FILES
# ============================================================

expected_files <- c(
  
  "00_XGB_search_grid.csv",
  
  "01_XGB_tuning_by_seed.csv",
  
  "02_XGB_tuning_summary.csv",
  
  "03_XGB_selected_hyperparameters.csv",
  
  "04_XGB_CV_metrics_by_seed.csv",
  
  "05_XGB_CV_summary.csv",
  
  "06_XGB_train_metrics.csv",
  
  "07_XGB_variable_importance.csv",
  
  "08_XGB_best_hyperparameters_by_seed.csv",
  
  "women_XGB_model.json",
  
  "men_XGB_model.json",
  
  "women_XGB_bundle.rds",
  
  "men_XGB_bundle.rds"
)


created_files <- list.files(
  output_dir
)


missing_outputs <- setdiff(
  
  expected_files,
  
  created_files
)


if (
  length(missing_outputs) > 0
) {
  
  stop(
    paste(
      "Missing final XGB files:",
      paste(
        missing_outputs,
        collapse = ", "
      )
    )
  )
}


if (
  nrow(
    selected_xgb_hyperparameters
  ) != 2
) {
  
  stop(
    "Expected exactly two selected XGB configurations."
  )
}


if (
  nrow(
    final_xgb_cv_summary
  ) != 2
) {
  
  stop(
    "Expected exactly two XGB CV summary rows."
  )
}


if (
  nrow(
    xgb_train_metrics
  ) != 2
) {
  
  stop(
    "Expected exactly two XGB TRAIN metric rows."
  )
}


# ============================================================
# 34. REMOVE CHECKPOINTS ONLY AFTER COMPLETE SUCCESS
# ============================================================

if (
  file.exists(
    women_checkpoint
  )
) {
  
  file.remove(
    women_checkpoint
  )
}


if (
  file.exists(
    men_checkpoint
  )
) {
  
  file.remove(
    men_checkpoint
  )
}


# ============================================================
# 35. DISPLAY FINAL RESULTS
# ============================================================

cat("\n\n========================================\n")
cat("FINAL XGBOOST HYPERPARAMETERS\n")
cat("========================================\n")

print(
  selected_xgb_hyperparameters
)


cat("\n========================================\n")
cat("FINAL XGBOOST INTERNAL CV PERFORMANCE\n")
cat("========================================\n")

print(
  final_xgb_cv_summary
)


cat("\n========================================\n")
cat("FINAL XGBOOST TRAIN PERFORMANCE\n")
cat("========================================\n")

print(
  xgb_train_metrics
)


cat("\n========================================\n")
cat("TOP 10 XGB VARIABLES - WOMEN\n")
cat("========================================\n")

print(
  
  xgb_variable_importance %>%
    
    dplyr::filter(
      Sex == "Women"
    ) %>%
    
    dplyr::slice_head(
      n = 10
    )
)


cat("\n========================================\n")
cat("TOP 10 XGB VARIABLES - MEN\n")
cat("========================================\n")

print(
  
  xgb_variable_importance %>%
    
    dplyr::filter(
      Sex == "Men"
    ) %>%
    
    dplyr::slice_head(
      n = 10
    )
)


# ============================================================
# 36. FINAL MESSAGE
# ============================================================

cat("\n\n========================================\n")
cat("SCRIPT 11 COMPLETED SUCCESSFULLY\n")
cat("========================================\n")

cat(
  "Analysis version:",
  analysis_version,
  "\n"
)

cat(
  "Model: Extreme Gradient Boosting\n"
)

cat(
  "Predictors: 21 original eligible predictors\n"
)

cat(
  "Elastic Net: NOT USED\n"
)

cat(
  "Transformations: NONE\n"
)

cat(
  "Centering/scaling: NONE\n"
)

cat(
  "Tuned hyperparameters: 9\n"
)

cat(
  "Candidate space:",
  nrow(xgb_candidate_pool),
  "\n"
)

cat(
  "Random-search configurations:",
  xgb_search_size,
  "\n"
)

cat(
  "Random-search seed:",
  xgb_search_seed,
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
  "Primary tuning criterion: mean OOF RMSE\n"
)

cat(
  "Predictive R2: 1 - SSE/SST\n"
)

cat(
  "TEST: NEVER READ OR ACCESSED\n"
)

cat(
  "Final output files:",
  length(expected_files),
  "\n"
)

cat("========================================\n")


print(
  list.files(
    output_dir
  )
)