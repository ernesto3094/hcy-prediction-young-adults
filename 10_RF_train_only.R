# ============================================================
# SCRIPT 10
# RANDOM FOREST - TRAIN ONLY
#
# FINAL CLEAN VERSION
#
# Hyperparameters tuned:
#   1. mtry
#   2. min.node.size
#   3. sample.fraction
#
# Fixed settings:
#   num.trees = 500
#   replace = TRUE
#   splitrule = "variance"
#
# METHODOLOGICAL DESIGN:
# - TRAIN only.
# - TEST is never read or accessed.
# - All 21 eligible predictors are used.
# - Original, untransformed predictor values are used.
# - No Elastic Net variable selection is used for RF.
# - No centering or scaling is applied.
# - Same 10 prespecified seeds used in previous models.
# - Same grouped HCY-stratified 10-fold CV procedure.
# - Duplicate IDs remain in the same fold.
# - Primary tuning criterion: mean OOF RMSE.
# - Official predictive R2 = 1 - SSE/SST.
# - Checkpoints are saved after every complete configuration.
# ============================================================


# ============================================================
# 1. PACKAGES
# ============================================================

required_packages <- c(
  "ranger",
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

library(ranger)
library(readr)
library(dplyr)
library(tibble)


# ============================================================
# 2. OUTPUT DIRECTORY
# ============================================================

output_dir <- "10_RF_results"

if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}


# ============================================================
# 3. LOCATE ORIGINAL TRAIN FILES
#
# RF must use the original, untransformed TRAIN partitions.
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
        "were found. Specify the correct path before continuing:\n",
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
# 4. READ ORIGINAL TRAIN DATA
# ============================================================

women_train <- read_csv(
  women_train_file,
  show_col_types = FALSE
)

men_train <- read_csv(
  men_train_file,
  show_col_types = FALSE
)


# ============================================================
# 5. STANDARDIZE VARIABLE NAMES
#
# Only column names are changed.
# Predictor values are NOT transformed.
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
# 6. FINAL 21 ELIGIBLE RF PREDICTORS
# ============================================================

rf_predictors <- c(
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


if (length(rf_predictors) != 21) {
  
  stop(
    "RF predictor list must contain exactly 21 variables."
  )
}


cat("\n========================================\n")
cat("RF PREDICTORS\n")
cat("========================================\n")

print(rf_predictors)

cat(
  "\nNumber of predictors:",
  length(rf_predictors),
  "\n"
)


# ============================================================
# 7. VERIFY DATASETS
# ============================================================

check_rf_dataset <- function(
    data,
    sex_name
) {
  
  required_columns <- c(
    "ID",
    "HCY",
    rf_predictors
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
    rf_predictors
  )
  
  
  for (variable in numeric_variables) {
    
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


check_rf_dataset(
  women_train,
  "WOMEN"
)

check_rf_dataset(
  men_train,
  "MEN"
)


# ============================================================
# 8. VERIFY SAMPLE SIZES
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


# ============================================================
# 10. RANDOM FOREST FIXED SETTINGS
# ============================================================

rf_num_trees <- 500

rf_replace <- TRUE

rf_splitrule <- "variance"


# ============================================================
# 11. CPU SETTINGS
#
# ranger is allowed to use up to four threads.
# ============================================================

detected_cores <- parallel::detectCores(
  logical = TRUE
)

if (
  is.na(detected_cores) ||
  detected_cores < 2
) {
  
  rf_threads <- 1
  
} else {
  
  rf_threads <- min(
    4,
    detected_cores - 1
  )
}


cat("\n========================================\n")
cat("RF COMPUTATION SETTINGS\n")
cat("========================================\n")

cat(
  "Trees per forest:",
  rf_num_trees,
  "\n"
)

cat(
  "CPU threads:",
  rf_threads,
  "\n"
)


# ============================================================
# 12. RF TUNING GRID
#
# mtry:
# Number of predictors randomly considered at each split.
#
# min.node.size:
# Minimum terminal-node size.
#
# sample.fraction:
# Fraction of the fold-specific training observations sampled
# for each tree.
#
# Total combinations:
# 6 x 4 x 3 = 72 per sex.
# ============================================================

rf_grid <- expand.grid(
  
  mtry = c(
    2,
    4,
    7,
    10,
    15,
    21
  ),
  
  min.node.size = c(
    3,
    5,
    10,
    15
  ),
  
  sample.fraction = c(
    0.632,
    0.80,
    1.00
  ),
  
  KEEP.OUT.ATTRS = FALSE,
  
  stringsAsFactors = FALSE
)


rf_grid$Config_ID <- seq_len(
  nrow(rf_grid)
)


rf_grid <- rf_grid %>%
  select(
    Config_ID,
    everything()
  )


if (nrow(rf_grid) != 72) {
  
  stop(
    "RF tuning grid should contain exactly 72 configurations."
  )
}


cat("\n========================================\n")
cat("RF TUNING GRID\n")
cat("========================================\n")

cat(
  "mtry values:",
  paste(
    sort(
      unique(
        rf_grid$mtry
      )
    ),
    collapse = ", "
  ),
  "\n"
)

cat(
  "min.node.size values:",
  paste(
    sort(
      unique(
        rf_grid$min.node.size
      )
    ),
    collapse = ", "
  ),
  "\n"
)

cat(
  "sample.fraction values:",
  paste(
    sort(
      unique(
        rf_grid$sample.fraction
      )
    ),
    collapse = ", "
  ),
  "\n"
)

cat(
  "Total configurations:",
  nrow(rf_grid),
  "\n"
)


# ============================================================
# 13. CREATE GROUPED HCY-STRATIFIED CV FOLDS
#
# Same deterministic procedure used for the other final models.
# Duplicate IDs remain within the same fold.
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
    
    
    all_assignments <- append(
      all_assignments,
      list(current_assignment)
    )
  }
  
  
  assignments <- bind_rows(
    all_assignments
  )
  
  
  expected_rows <- nrow(data) *
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
      Number_of_Folds = n_distinct(Fold),
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
      Number_of_Folds = n_distinct(Fold),
      .groups = "drop"
    )
  
  
  if (
    any(
      fold_check$Number_of_Folds != k
    )
  ) {
    
    stop(
      "At least one seed does not contain the expected number of folds."
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
# 15. CREATE CHECKPOINT DATA SIGNATURE
#
# This prevents an old checkpoint from being applied to
# a different dataset.
# ============================================================

create_data_signature <- function(
    data,
    predictors
) {
  
  predictor_sums <- vapply(
    predictors,
    function(variable) {
      sum(data[[variable]])
    },
    numeric(1)
  )
  
  
  predictor_sums_sq <- vapply(
    predictors,
    function(variable) {
      sum(data[[variable]]^2)
    },
    numeric(1)
  )
  
  
  list(
    
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
}


# ============================================================
# 16. RANDOM FOREST TUNING FUNCTION
#
# Checkpoint behavior:
# - A checkpoint is saved after every complete hyperparameter
#   configuration.
# - If the script stops, rerunning it resumes from the next
#   unfinished configuration.
# ============================================================

tune_random_forest <- function(
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
    data = data,
    predictors = predictors
  )
  
  
  rf_settings <- list(
    num.trees = rf_num_trees,
    replace = rf_replace,
    splitrule = rf_splitrule,
    seeds = seeds,
    folds = k
  )
  
  
  results <- tibble()
  
  completed_configs <- integer(0)
  
  
  # ----------------------------------------------------------
  # LOAD CHECKPOINT IF ONE EXISTS
  # ----------------------------------------------------------
  
  if (file.exists(checkpoint_file)) {
    
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
        check.attributes = FALSE
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
        check.attributes = FALSE
      )
    )
    
    
    settings_match <- isTRUE(
      all.equal(
        checkpoint$rf_settings,
        rf_settings,
        check.attributes = FALSE
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
          "does not match the current analysis."
        )
      )
    }
    
    
    results <- checkpoint$results
    
    completed_configs <- checkpoint$completed_configs
    
    
    cat(
      "Recovered",
      length(completed_configs),
      "of",
      nrow(tuning_grid),
      "completed configurations.\n"
    )
  }
  
  
  # ----------------------------------------------------------
  # LOOP OVER HYPERPARAMETER CONFIGURATIONS
  # ----------------------------------------------------------
  
  for (
    config_index in seq_len(
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
      "- RF configuration",
      config_id,
      "of",
      nrow(tuning_grid),
      "\n"
    )
    
    cat(
      "mtry =",
      current_config$mtry,
      "| min.node.size =",
      current_config$min.node.size,
      "| sample.fraction =",
      current_config$sample.fraction,
      "\n"
    )
    
    
    config_seed_results <- list()
    
    
    # --------------------------------------------------------
    # LOOP OVER THE 10 PRESPECIFIED CV SEEDS
    # --------------------------------------------------------
    
    for (
      seed_index in seq_along(seeds)
    ) {
      
      seed_value <- seeds[seed_index]
      
      
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
      # LOOP OVER THE 10 FOLDS
      # ------------------------------------------------------
      
      for (
        fold_value in seq_len(k)
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
        
        
        training_data <- data[
          training_rows,
          c(
            "HCY",
            predictors
          ),
          drop = FALSE
        ]
        
        
        validation_data <- data[
          validation_rows,
          predictors,
          drop = FALSE
        ]
        
        
        # ----------------------------------------------------
        # Forest seed
        #
        # The same forest seed is used for all configurations
        # within the same CV seed/fold combination.
        # ----------------------------------------------------
        
        forest_seed <- as.integer(
          seed_value +
            (fold_value * 100000)
        )
        
        
        rf_fit <- ranger::ranger(
          
          dependent.variable.name =
            "HCY",
          
          data =
            training_data,
          
          num.trees =
            rf_num_trees,
          
          mtry =
            as.integer(
              current_config$mtry
            ),
          
          min.node.size =
            as.integer(
              current_config$min.node.size
            ),
          
          sample.fraction =
            as.numeric(
              current_config$sample.fraction
            ),
          
          replace =
            rf_replace,
          
          splitrule =
            rf_splitrule,
          
          importance =
            "none",
          
          write.forest =
            TRUE,
          
          seed =
            forest_seed,
          
          num.threads =
            rf_threads,
          
          verbose =
            FALSE
        )
        
        
        fold_predictions <- predict(
          rf_fit,
          data = validation_data
        )$predictions
        
        
        oof_predictions[
          validation_rows
        ] <- fold_predictions
        
        
        rm(
          rf_fit,
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
            "Missing OOF prediction detected for",
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
        
        mtry =
          as.integer(
            current_config$mtry
          ),
        
        min.node.size =
          as.integer(
            current_config$min.node.size
          ),
        
        sample.fraction =
          as.numeric(
            current_config$sample.fraction
          ),
        
        Seed =
          as.integer(
            seed_value
          ),
        
        N =
          nrow(data),
        
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
      
      
      config_seed_results <- append(
        config_seed_results,
        list(seed_result)
      )
    }
    
    
    # --------------------------------------------------------
    # COMBINE THE 10 SEED RESULTS FOR THIS CONFIGURATION
    # --------------------------------------------------------
    
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
          "did not produce the expected number of seed results."
        )
      )
    }
    
    
    results <- dplyr::bind_rows(
      results,
      current_results
    )
    
    
    completed_configs <- c(
      completed_configs,
      config_id
    )
    
    
    completed_configs <- sort(
      unique(
        completed_configs
      )
    )
    
    
    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------
    
    checkpoint <- list(
      
      sex =
        sex_name,
      
      predictors =
        predictors,
      
      tuning_grid =
        tuning_grid,
      
      data_signature =
        data_signature,
      
      rf_settings =
        rf_settings,
      
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
      "Checkpoint saved -",
      length(completed_configs),
      "of",
      nrow(tuning_grid),
      "configurations completed.\n"
    )
    
    
    rm(
      current_results,
      config_seed_results,
      checkpoint
    )
    
    gc()
  }
  
  
  # ----------------------------------------------------------
  # FINAL TUNING CHECK
  # ----------------------------------------------------------
  
  expected_rows <- nrow(tuning_grid) *
    length(seeds)
  
  
  if (
    nrow(results) !=
    expected_rows
  ) {
    
    stop(
      paste(
        sex_name,
        "RF tuning produced",
        nrow(results),
        "rows; expected",
        expected_rows
      )
    )
  }
  
  
  unique_config_count <- dplyr::n_distinct(
    results$Config_ID
  )
  
  
  if (
    unique_config_count !=
    nrow(tuning_grid)
  ) {
    
    stop(
      paste(
        sex_name,
        "does not contain all RF configurations."
      )
    )
  }
  
  
  return(results)
}


# ============================================================
# 17. CREATE WOMEN CV FOLDS
# ============================================================

women_assignments <- create_cv_folds(
  
  data =
    women_train,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds
)


# ============================================================
# 18. TUNE WOMEN RANDOM FOREST
# ============================================================

women_checkpoint <- file.path(
  output_dir,
  "checkpoint_women_RF.rds"
)


women_rf_tuning <- tune_random_forest(
  
  data =
    women_train,
  
  predictors =
    rf_predictors,
  
  sex_name =
    "Women",
  
  assignments =
    women_assignments,
  
  tuning_grid =
    rf_grid,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds,
  
  checkpoint_file =
    women_checkpoint
)


# ============================================================
# 19. CREATE MEN CV FOLDS
# ============================================================

men_assignments <- create_cv_folds(
  
  data =
    men_train,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds
)


# ============================================================
# 20. TUNE MEN RANDOM FOREST
# ============================================================

men_checkpoint <- file.path(
  output_dir,
  "checkpoint_men_RF.rds"
)


men_rf_tuning <- tune_random_forest(
  
  data =
    men_train,
  
  predictors =
    rf_predictors,
  
  sex_name =
    "Men",
  
  assignments =
    men_assignments,
  
  tuning_grid =
    rf_grid,
  
  seeds =
    cv_seeds,
  
  k =
    number_folds,
  
  checkpoint_file =
    men_checkpoint
)


# ============================================================
# 21. COMBINE ALL RF TUNING RESULTS
# ============================================================

all_rf_tuning <- dplyr::bind_rows(
  women_rf_tuning,
  men_rf_tuning
)


# ============================================================
# 22. RF TUNING SUMMARY ACROSS THE 10 SEEDS
# ============================================================

rf_tuning_summary <- all_rf_tuning %>%
  
  group_by(
    Sex,
    Config_ID,
    mtry,
    min.node.size,
    sample.fraction
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
# 23. SELECT FINAL RF HYPERPARAMETERS
#
# Primary criterion:
# Lowest mean OOF RMSE across the 10 CV configurations.
#
# Additional columns are used only as deterministic tie-breakers
# in the unlikely event of an exact RMSE tie.
# ============================================================

selected_rf_hyperparameters <- rf_tuning_summary %>%
  
  group_by(Sex) %>%
  
  arrange(
    Mean_RMSE,
    mtry,
    desc(min.node.size),
    sample.fraction,
    .by_group = TRUE
  ) %>%
  
  slice(1) %>%
  
  ungroup()


cat("\n\n========================================\n")
cat("FINAL RF HYPERPARAMETERS\n")
cat("========================================\n")

print(
  selected_rf_hyperparameters
)


# ============================================================
# 24. FINAL RF CV METRICS BY SEED
# ============================================================

selected_rf_keys <- selected_rf_hyperparameters %>%
  
  select(
    Sex,
    Config_ID,
    mtry,
    min.node.size,
    sample.fraction
  )


final_rf_cv_by_seed <- all_rf_tuning %>%
  
  inner_join(
    
    selected_rf_keys,
    
    by = c(
      "Sex",
      "Config_ID",
      "mtry",
      "min.node.size",
      "sample.fraction"
    )
  )


if (
  nrow(final_rf_cv_by_seed) !=
  20
) {
  
  stop(
    paste(
      "Expected 20 final RF CV rows",
      "(10 seeds x 2 sexes), obtained",
      nrow(final_rf_cv_by_seed)
    )
  )
}


# ============================================================
# 25. FINAL RF CV SUMMARY
# ============================================================

final_rf_cv_summary <- final_rf_cv_by_seed %>%
  
  group_by(
    Sex,
    Config_ID,
    mtry,
    min.node.size,
    sample.fraction
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
# 26. FIT FINAL FULL-TRAIN RF MODEL
# ============================================================

fit_final_rf <- function(
    data,
    predictors,
    sex_name,
    selected_row
) {
  
  final_model <- ranger::ranger(
    
    dependent.variable.name =
      "HCY",
    
    data =
      data[
        ,
        c(
          "HCY",
          predictors
        ),
        drop = FALSE
      ],
    
    num.trees =
      rf_num_trees,
    
    mtry =
      as.integer(
        selected_row$mtry
      ),
    
    min.node.size =
      as.integer(
        selected_row$min.node.size
      ),
    
    sample.fraction =
      as.numeric(
        selected_row$sample.fraction
      ),
    
    replace =
      rf_replace,
    
    splitrule =
      rf_splitrule,
    
    importance =
      "permutation",
    
    write.forest =
      TRUE,
    
    seed =
      123,
    
    num.threads =
      rf_threads,
    
    verbose =
      FALSE
  )
  
  
  train_predictions <- predict(
    final_model,
    data =
      data[
        ,
        predictors,
        drop = FALSE
      ]
  )$predictions
  
  
  train_metrics <- data.frame(
    
    Sex =
      sex_name,
    
    N =
      nrow(data),
    
    mtry =
      as.integer(
        selected_row$mtry
      ),
    
    min.node.size =
      as.integer(
        selected_row$min.node.size
      ),
    
    sample.fraction =
      as.numeric(
        selected_row$sample.fraction
      ),
    
    num.trees =
      rf_num_trees,
    
    RMSE =
      calculate_rmse(
        observed = data$HCY,
        predicted = train_predictions
      ),
    
    MAE =
      calculate_mae(
        observed = data$HCY,
        predicted = train_predictions
      ),
    
    Predictive_R2 =
      calculate_predictive_r2(
        observed = data$HCY,
        predicted = train_predictions
      )
  )
  
  
  importance_table <- tibble(
    
    Sex =
      sex_name,
    
    Variable =
      names(
        final_model$variable.importance
      ),
    
    Permutation_Importance =
      as.numeric(
        final_model$variable.importance
      )
  ) %>%
    
    arrange(
      desc(
        Permutation_Importance
      )
    ) %>%
    
    mutate(
      Rank =
        row_number()
    ) %>%
    
    select(
      Sex,
      Rank,
      Variable,
      Permutation_Importance
    )
  
  
  return(
    list(
      model = final_model,
      metrics = train_metrics,
      importance = importance_table
    )
  )
}


# ============================================================
# 27. FIT FINAL WOMEN RF
# ============================================================

women_selected_rf <- selected_rf_hyperparameters %>%
  
  filter(
    Sex == "Women"
  )


if (
  nrow(women_selected_rf) != 1
) {
  
  stop(
    "Expected exactly one WOMEN RF configuration."
  )
}


women_final_rf <- fit_final_rf(
  
  data =
    women_train,
  
  predictors =
    rf_predictors,
  
  sex_name =
    "Women",
  
  selected_row =
    women_selected_rf
)


# ============================================================
# 28. FIT FINAL MEN RF
# ============================================================

men_selected_rf <- selected_rf_hyperparameters %>%
  
  filter(
    Sex == "Men"
  )


if (
  nrow(men_selected_rf) != 1
) {
  
  stop(
    "Expected exactly one MEN RF configuration."
  )
}


men_final_rf <- fit_final_rf(
  
  data =
    men_train,
  
  predictors =
    rf_predictors,
  
  sex_name =
    "Men",
  
  selected_row =
    men_selected_rf
)


# ============================================================
# 29. COMBINE FINAL TRAIN METRICS
# ============================================================

rf_train_metrics <- dplyr::bind_rows(
  women_final_rf$metrics,
  men_final_rf$metrics
)


if (
  nrow(rf_train_metrics) != 2
) {
  
  stop(
    "Expected exactly two RF TRAIN metric rows."
  )
}


# ============================================================
# 30. COMBINE VARIABLE IMPORTANCE
# ============================================================

rf_variable_importance <- dplyr::bind_rows(
  women_final_rf$importance,
  men_final_rf$importance
)


if (
  nrow(rf_variable_importance) != 42
) {
  
  stop(
    paste(
      "Expected 42 RF importance rows;",
      "obtained",
      nrow(rf_variable_importance)
    )
  )
}


# ============================================================
# 31. CREATE FINAL RF MODEL BUNDLES
# ============================================================

women_rf_bundle <- list(
  
  sex =
    "Women",
  
  predictors =
    rf_predictors,
  
  model =
    women_final_rf$model,
  
  selected_hyperparameters =
    women_selected_rf,
  
  cv_seeds =
    cv_seeds,
  
  number_folds =
    number_folds,
  
  num.trees =
    rf_num_trees,
  
  replace =
    rf_replace,
  
  splitrule =
    rf_splitrule,
  
  preprocessing =
    "Original untransformed predictors; no centering or scaling",
  
  variable_selection =
    "All 21 eligible predictors",
  
  tuning_selection =
    "Minimum mean OOF RMSE across 10 prespecified CV configurations",
  
  r2_definition =
    "1 - SSE/SST",
  
  test_accessed =
    FALSE
)


men_rf_bundle <- list(
  
  sex =
    "Men",
  
  predictors =
    rf_predictors,
  
  model =
    men_final_rf$model,
  
  selected_hyperparameters =
    men_selected_rf,
  
  cv_seeds =
    cv_seeds,
  
  number_folds =
    number_folds,
  
  num.trees =
    rf_num_trees,
  
  replace =
    rf_replace,
  
  splitrule =
    rf_splitrule,
  
  preprocessing =
    "Original untransformed predictors; no centering or scaling",
  
  variable_selection =
    "All 21 eligible predictors",
  
  tuning_selection =
    "Minimum mean OOF RMSE across 10 prespecified CV configurations",
  
  r2_definition =
    "1 - SSE/SST",
  
  test_accessed =
    FALSE
)


# ============================================================
# 32. SAVE FINAL OUTPUTS
# ============================================================

write_csv(
  all_rf_tuning,
  file.path(
    output_dir,
    "01_RF_tuning_by_seed.csv"
  )
)


write_csv(
  rf_tuning_summary,
  file.path(
    output_dir,
    "02_RF_tuning_summary.csv"
  )
)


write_csv(
  selected_rf_hyperparameters,
  file.path(
    output_dir,
    "03_RF_selected_hyperparameters.csv"
  )
)


write_csv(
  final_rf_cv_by_seed,
  file.path(
    output_dir,
    "04_RF_CV_metrics_by_seed.csv"
  )
)


write_csv(
  final_rf_cv_summary,
  file.path(
    output_dir,
    "05_RF_CV_summary.csv"
  )
)


write_csv(
  rf_train_metrics,
  file.path(
    output_dir,
    "06_RF_train_metrics.csv"
  )
)


write_csv(
  rf_variable_importance,
  file.path(
    output_dir,
    "07_RF_variable_importance.csv"
  )
)


saveRDS(
  women_rf_bundle,
  file.path(
    output_dir,
    "women_RF_bundle.rds"
  )
)


saveRDS(
  men_rf_bundle,
  file.path(
    output_dir,
    "men_RF_bundle.rds"
  )
)


# ============================================================
# 33. VERIFY FINAL OUTPUT FILES
# ============================================================

expected_files <- c(
  "01_RF_tuning_by_seed.csv",
  "02_RF_tuning_summary.csv",
  "03_RF_selected_hyperparameters.csv",
  "04_RF_CV_metrics_by_seed.csv",
  "05_RF_CV_summary.csv",
  "06_RF_train_metrics.csv",
  "07_RF_variable_importance.csv",
  "women_RF_bundle.rds",
  "men_RF_bundle.rds"
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
      "Missing RF output files:",
      paste(
        missing_outputs,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 34. FINAL CONSISTENCY CHECKS
# ============================================================

if (
  nrow(
    selected_rf_hyperparameters
  ) != 2
) {
  
  stop(
    "Expected exactly two selected RF configurations."
  )
}


if (
  nrow(
    final_rf_cv_summary
  ) != 2
) {
  
  stop(
    "Expected exactly two RF CV summary rows."
  )
}


if (
  nrow(
    rf_train_metrics
  ) != 2
) {
  
  stop(
    "Expected exactly two RF TRAIN metric rows."
  )
}


# ============================================================
# 35. REMOVE TEMPORARY CHECKPOINTS
#
# Only removed after all final files are successfully written.
# ============================================================

if (file.exists(women_checkpoint)) {
  
  file.remove(
    women_checkpoint
  )
}


if (file.exists(men_checkpoint)) {
  
  file.remove(
    men_checkpoint
  )
}


# ============================================================
# 36. DISPLAY SELECTED HYPERPARAMETERS
# ============================================================

cat("\n\n========================================\n")
cat("RF SELECTED HYPERPARAMETERS\n")
cat("========================================\n")

print(
  selected_rf_hyperparameters
)


# ============================================================
# 37. DISPLAY FINAL INTERNAL CV PERFORMANCE
# ============================================================

cat("\n========================================\n")
cat("RF INTERNAL CV PERFORMANCE\n")
cat("========================================\n")

print(
  final_rf_cv_summary
)


# ============================================================
# 38. DISPLAY COMPLETE TRAIN PERFORMANCE
# ============================================================

cat("\n========================================\n")
cat("RF COMPLETE TRAIN PERFORMANCE\n")
cat("========================================\n")

print(
  rf_train_metrics
)


# ============================================================
# 39. DISPLAY VARIABLE IMPORTANCE
# ============================================================

cat("\n========================================\n")
cat("TOP 10 RF VARIABLES - WOMEN\n")
cat("========================================\n")

print(
  rf_variable_importance %>%
    filter(
      Sex == "Women"
    ) %>%
    slice_head(
      n = 10
    )
)


cat("\n========================================\n")
cat("TOP 10 RF VARIABLES - MEN\n")
cat("========================================\n")

print(
  rf_variable_importance %>%
    filter(
      Sex == "Men"
    ) %>%
    slice_head(
      n = 10
    )
)


# ============================================================
# 40. FINAL MESSAGE
# ============================================================

cat("\n\n========================================\n")
cat("SCRIPT 10 COMPLETED SUCCESSFULLY\n")
cat("========================================\n")

cat(
  "Model: Random Forest\n"
)

cat(
  "Predictors: all 21 eligible predictors\n"
)

cat(
  "Predictor transformations: NONE\n"
)

cat(
  "Elastic Net selection: NONE\n"
)

cat(
  "Centering/scaling: NONE\n"
)

cat(
  "Tuned hyperparameters: mtry, min.node.size, sample.fraction\n"
)

cat(
  "Trees per forest:",
  rf_num_trees,
  "\n"
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
  "Predictive R2: 1 - SSE/SST\n"
)

cat(
  "TEST: NEVER READ OR ACCESSED\n"
)

cat(
  "Final output files: 9\n"
)

cat(
  "Temporary checkpoints removed after successful completion.\n"
)

cat("========================================\n")


print(
  list.files(
    output_dir
  )
)