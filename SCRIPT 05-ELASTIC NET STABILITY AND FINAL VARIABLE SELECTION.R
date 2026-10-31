# ============================================================
# SCRIPT 05
# ELASTIC NET STABILITY AND FINAL VARIABLE SELECTION
#
# IMPORTANT:
# - TRAIN data only.
# - TEST data are never read or accessed.
# - HCY remains on its original scale.
# - Previously selected predictor transformations are fixed.
# - Scaling is estimated within each CV training fold.
# - Duplicate IDs are kept in the same internal CV fold.
# - No Stable-60/70/80/90/100 groups are created.
# ============================================================


# ============================================================
# 1. PACKAGES
# ============================================================

required_packages <- c(
  "readr",
  "dplyr",
  "caret",
  "glmnet"
)

missing_packages <- required_packages[
  !required_packages %in% rownames(installed.packages())
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(readr)
library(dplyr)
library(caret)
library(glmnet)


# ============================================================
# 2. INPUT FILES
# ============================================================

women_file <- "01_women_TRAIN.csv"
men_file   <- "03_men_TRAIN.csv"

if (!file.exists(women_file)) {
  stop("File not found: 01_women_TRAIN.csv")
}

if (!file.exists(men_file)) {
  stop("File not found: 03_men_TRAIN.csv")
}

women_train <- read_csv(
  women_file,
  show_col_types = FALSE
)

men_train <- read_csv(
  men_file,
  show_col_types = FALSE
)


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

output_dir <- "05_elastic_net_results"

if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}


# ============================================================
# 4. STANDARDIZE VARIABLE NAMES
# ============================================================

rename_variables <- function(data) {
  
  name_map <- c(
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
  
  for (old_name in names(name_map)) {
    
    new_name <- name_map[[old_name]]
    
    if (
      old_name %in% names(data) &&
      !(new_name %in% names(data))
    ) {
      names(data)[names(data) == old_name] <- new_name
    }
  }
  
  data
}

women_train <- rename_variables(women_train)
men_train   <- rename_variables(men_train)


# ============================================================
# 5. OFFICIAL 21 PREDICTORS
# ============================================================

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


# ============================================================
# 6. BASIC DATA CHECKS
# ============================================================

required_columns <- c(
  "ID",
  "HCY",
  predictors
)

missing_women <- setdiff(
  required_columns,
  names(women_train)
)

missing_men <- setdiff(
  required_columns,
  names(men_train)
)

if (length(missing_women) > 0) {
  stop(
    paste(
      "Missing WOMEN columns:",
      paste(missing_women, collapse = ", ")
    )
  )
}

if (length(missing_men) > 0) {
  stop(
    paste(
      "Missing MEN columns:",
      paste(missing_men, collapse = ", ")
    )
  )
}

if (nrow(women_train) != 190) {
  stop(
    paste(
      "Unexpected WOMEN TRAIN sample size:",
      nrow(women_train),
      "Expected: 190"
    )
  )
}

if (nrow(men_train) != 83) {
  stop(
    paste(
      "Unexpected MEN TRAIN sample size:",
      nrow(men_train),
      "Expected: 83"
    )
  )
}

check_missing <- function(data, sex_name) {
  
  analytical <- data[, c("HCY", predictors)]
  
  total_missing <- sum(is.na(analytical))
  
  if (total_missing > 0) {
    stop(
      paste(
        sex_name,
        "contains",
        total_missing,
        "missing analytical values."
      )
    )
  }
}

check_missing(women_train, "WOMEN")
check_missing(men_train, "MEN")


# ============================================================
# 7. APPLY THE FROZEN TRANSFORMATIONS
# ============================================================

safe_log <- function(x, variable_name) {
  
  if (any(x <= 0, na.rm = TRUE)) {
    stop(
      paste(
        "Cannot apply log transformation to",
        variable_name,
        "because non-positive values were detected."
      )
    )
  }
  
  log(x)
}

safe_sqrt <- function(x, variable_name) {
  
  if (any(x < 0, na.rm = TRUE)) {
    stop(
      paste(
        "Cannot apply square-root transformation to",
        variable_name,
        "because negative values were detected."
      )
    )
  }
  
  sqrt(x)
}


# ------------------------------------------------------------
# WOMEN
#
# Log:
# PLT, WBC, TRG, HDL
#
# Square root:
# LDL
# ------------------------------------------------------------

women_model_data <- women_train

women_model_data$PLT <- safe_log(
  women_model_data$PLT,
  "PLT"
)

women_model_data$WBC <- safe_log(
  women_model_data$WBC,
  "WBC"
)

women_model_data$TRG <- safe_log(
  women_model_data$TRG,
  "TRG"
)

women_model_data$HDL <- safe_log(
  women_model_data$HDL,
  "HDL"
)

women_model_data$LDL <- safe_sqrt(
  women_model_data$LDL,
  "LDL"
)


# ------------------------------------------------------------
# MEN
#
# Log:
# VisceralFat, WBC, TRG, HDL
# ------------------------------------------------------------

men_model_data <- men_train

men_model_data$VisceralFat <- safe_log(
  men_model_data$VisceralFat,
  "VisceralFat"
)

men_model_data$WBC <- safe_log(
  men_model_data$WBC,
  "WBC"
)

men_model_data$TRG <- safe_log(
  men_model_data$TRG,
  "TRG"
)

men_model_data$HDL <- safe_log(
  men_model_data$HDL,
  "HDL"
)


# ============================================================
# 8. SAVE TRANSFORMED TRAIN DATA
# ============================================================

write_csv(
  women_model_data,
  file.path(
    output_dir,
    "01_women_TRAIN_transformed.csv"
  )
)

write_csv(
  men_model_data,
  file.path(
    output_dir,
    "02_men_TRAIN_transformed.csv"
  )
)


# ============================================================
# 9. ELASTIC NET TUNING GRID
#
# alpha = 0 is intentionally excluded because pure Ridge
# regression does not provide sparse variable selection.
# ============================================================

alpha_grid <- seq(
  0.1,
  1.0,
  by = 0.1
)

lambda_grid <- 10^seq(
  -4,
  1,
  length.out = 60
)

tuning_grid <- expand.grid(
  alpha = alpha_grid,
  lambda = lambda_grid
)


# ============================================================
# 10. INTERNAL CV SEEDS
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


# ============================================================
# 11. CREATE GROUPED 10-FOLD CV
#
# Rows sharing the same ID are always assigned to the same
# validation fold.
#
# HCY is used only to approximately stratify the internal folds.
# ============================================================

create_grouped_folds <- function(
    data,
    seed,
    k = 10
) {
  
  group_data <- data %>%
    group_by(ID) %>%
    summarise(
      Group_HCY = mean(HCY),
      .groups = "drop"
    )
  
  set.seed(seed)
  
  validation_group_folds <- caret::createFolds(
    group_data$Group_HCY,
    k = k,
    list = TRUE,
    returnTrain = FALSE
  )
  
  index <- list()
  index_out <- list()
  fold_map <- list()
  
  for (i in seq_along(validation_group_folds)) {
    
    validation_group_rows <-
      validation_group_folds[[i]]
    
    validation_ids <-
      group_data$ID[
        validation_group_rows
      ]
    
    validation_rows <- which(
      data$ID %in% validation_ids
    )
    
    training_rows <- setdiff(
      seq_len(nrow(data)),
      validation_rows
    )
    
    fold_name <- sprintf(
      "Fold%02d",
      i
    )
    
    index[[fold_name]] <- training_rows
    index_out[[fold_name]] <- validation_rows
    
    fold_map[[i]] <- data.frame(
      Seed = seed,
      Fold = i,
      Row = validation_rows,
      ID = data$ID[validation_rows],
      HCY = data$HCY[validation_rows]
    )
  }
  
  list(
    index = index,
    indexOut = index_out,
    fold_map = bind_rows(fold_map)
  )
}


# ============================================================
# 12. FUNCTION FOR REPEATED INTERNAL ELASTIC NET TUNING
# ============================================================

run_elastic_net_stability <- function(
    data,
    sex_name
) {
  
  all_tuning_results <- list()
  best_tuning_results <- list()
  selection_results <- list()
  fold_results <- list()
  fitted_models <- list()
  
  model_data <- data %>%
    select(
      ID,
      HCY,
      all_of(predictors)
    )
  
  formula_en <- as.formula(
    paste(
      "HCY ~",
      paste(
        predictors,
        collapse = " + "
      )
    )
  )
  
  for (i in seq_along(cv_seeds)) {
    
    current_seed <- cv_seeds[i]
    
    cat(
      "\n============================================\n"
    )
    cat(
      sex_name,
      "- Elastic Net run",
      i,
      "of",
      length(cv_seeds),
      "\n"
    )
    cat(
      "CV seed:",
      current_seed,
      "\n"
    )
    cat(
      "============================================\n"
    )
    
    folds <- create_grouped_folds(
      data = model_data,
      seed = current_seed,
      k = 10
    )
    
    control <- trainControl(
      method = "cv",
      number = 10,
      index = folds$index,
      indexOut = folds$indexOut,
      savePredictions = "final",
      returnResamp = "all",
      allowParallel = TRUE
    )
    
    set.seed(current_seed)
    
    fit <- caret::train(
      formula_en,
      data = model_data %>%
        select(
          -ID
        ),
      method = "glmnet",
      trControl = control,
      tuneGrid = tuning_grid,
      preProcess = c(
        "center",
        "scale"
      ),
      metric = "RMSE",
      maximize = FALSE,
      standardize = FALSE
    )
    
    fitted_models[[i]] <- fit
    
    current_results <- fit$results %>%
      mutate(
        Seed = current_seed
      )
    
    all_tuning_results[[i]] <-
      current_results
    
    best_alpha <-
      fit$bestTune$alpha
    
    best_lambda <-
      fit$bestTune$lambda
    
    best_row <- current_results %>%
      filter(
        alpha == best_alpha,
        lambda == best_lambda
      ) %>%
      slice(1)
    
    coefficients <- as.matrix(
      coef(
        fit$finalModel,
        s = best_lambda
      )
    )
    
    coefficient_table <- data.frame(
      Variable = rownames(coefficients),
      Coefficient = coefficients[, 1],
      stringsAsFactors = FALSE
    )
    
    selected_variables <-
      coefficient_table %>%
      filter(
        Variable != "(Intercept)",
        abs(Coefficient) > 1e-10
      ) %>%
      pull(Variable)
    
    best_tuning_results[[i]] <-
      data.frame(
        Sex = sex_name,
        Seed = current_seed,
        Alpha = best_alpha,
        Lambda = best_lambda,
        CV_RMSE = best_row$RMSE,
        CV_R2 = best_row$Rsquared,
        CV_MAE = best_row$MAE,
        Number_Selected =
          length(selected_variables)
      )
    
    selection_results[[i]] <-
      data.frame(
        Sex = sex_name,
        Seed = current_seed,
        Predictor = predictors,
        Selected = as.integer(
          predictors %in%
            selected_variables
        )
      )
    
    fold_results[[i]] <-
      folds$fold_map
  }
  
  all_tuning_results <-
    bind_rows(all_tuning_results)
  
  best_tuning_results <-
    bind_rows(best_tuning_results)
  
  selection_results <-
    bind_rows(selection_results)
  
  fold_results <-
    bind_rows(fold_results)
  
  # ----------------------------------------------------------
  # Aggregate CV performance across the 10 fold configurations
  # ----------------------------------------------------------
  
  aggregated_tuning <-
    all_tuning_results %>%
    group_by(
      alpha,
      lambda
    ) %>%
    summarise(
      Mean_RMSE = mean(RMSE),
      SD_RMSE = sd(RMSE),
      Mean_R2 = mean(Rsquared),
      SD_R2 = sd(Rsquared),
      Mean_MAE = mean(MAE),
      SD_MAE = sd(MAE),
      .groups = "drop"
    ) %>%
    arrange(
      Mean_RMSE,
      SD_RMSE,
      desc(lambda)
    )
  
  final_tuning <-
    aggregated_tuning %>%
    slice(1)
  
  final_alpha <-
    final_tuning$alpha
  
  final_lambda <-
    final_tuning$lambda
  
  # ----------------------------------------------------------
  # Predictor-selection frequency across CV seeds
  # ----------------------------------------------------------
  
  selection_frequency <-
    selection_results %>%
    group_by(
      Sex,
      Predictor
    ) %>%
    summarise(
      Selected_Count = sum(Selected),
      Selection_Frequency =
        mean(Selected),
      Selection_Percent =
        100 * mean(Selected),
      .groups = "drop"
    ) %>%
    arrange(
      desc(Selected_Count),
      Predictor
    )
  
  # ----------------------------------------------------------
  # Hyperparameter stability summary
  # ----------------------------------------------------------
  
  hyperparameter_summary <-
    data.frame(
      Sex = sex_name,
      Measure = c(
        "Number of CV seeds",
        "Mean alpha",
        "SD alpha",
        "Median alpha",
        "Minimum alpha",
        "Maximum alpha",
        "Mean lambda",
        "SD lambda",
        "Median lambda",
        "Minimum lambda",
        "Maximum lambda",
        "Mean selected predictors",
        "SD selected predictors",
        "Minimum selected predictors",
        "Maximum selected predictors"
      ),
      Value = c(
        nrow(best_tuning_results),
        mean(best_tuning_results$Alpha),
        sd(best_tuning_results$Alpha),
        median(best_tuning_results$Alpha),
        min(best_tuning_results$Alpha),
        max(best_tuning_results$Alpha),
        mean(best_tuning_results$Lambda),
        sd(best_tuning_results$Lambda),
        median(best_tuning_results$Lambda),
        min(best_tuning_results$Lambda),
        max(best_tuning_results$Lambda),
        mean(best_tuning_results$Number_Selected),
        sd(best_tuning_results$Number_Selected),
        min(best_tuning_results$Number_Selected),
        max(best_tuning_results$Number_Selected)
      )
    )
  
  # ----------------------------------------------------------
  # Final scaling based on the complete TRAIN subset
  # ----------------------------------------------------------
  
  x_train <- model_data %>%
    select(
      all_of(predictors)
    )
  
  y_train <- model_data$HCY
  
  scaling_model <- caret::preProcess(
    x_train,
    method = c(
      "center",
      "scale"
    )
  )
  
  x_train_scaled <-
    predict(
      scaling_model,
      x_train
    )
  
  # ----------------------------------------------------------
  # Final Elastic Net fit on complete TRAIN
  #
  # The final alpha and lambda are based on aggregated
  # internal CV performance only.
  # ----------------------------------------------------------
  
  final_model <- glmnet::glmnet(
    x = as.matrix(x_train_scaled),
    y = y_train,
    alpha = final_alpha,
    lambda = final_lambda,
    standardize = FALSE,
    intercept = TRUE
  )
  
  final_coefficients_matrix <-
    as.matrix(
      coef(
        final_model,
        s = final_lambda
      )
    )
  
  final_coefficients <-
    data.frame(
      Variable =
        rownames(
          final_coefficients_matrix
        ),
      Coefficient =
        final_coefficients_matrix[, 1],
      stringsAsFactors = FALSE
    )
  
  final_selected_predictors <-
    final_coefficients %>%
    filter(
      Variable != "(Intercept)",
      abs(Coefficient) > 1e-10
    ) %>%
    arrange(
      desc(abs(Coefficient))
    )
  
  # ----------------------------------------------------------
  # Scaling parameters
  # ----------------------------------------------------------
  
  scaling_parameters <-
    data.frame(
      Predictor = predictors,
      Center = as.numeric(
        scaling_model$mean[
          predictors
        ]
      ),
      Scale = as.numeric(
        scaling_model$std[
          predictors
        ]
      )
    )
  
  final_hyperparameters <-
    data.frame(
      Sex = sex_name,
      Alpha = final_alpha,
      Lambda = final_lambda,
      Mean_CV_RMSE =
        final_tuning$Mean_RMSE,
      SD_CV_RMSE =
        final_tuning$SD_RMSE,
      Mean_CV_R2 =
        final_tuning$Mean_R2,
      Mean_CV_MAE =
        final_tuning$Mean_MAE,
      Number_Final_Selected =
        nrow(
          final_selected_predictors
        )
    )
  
  # ----------------------------------------------------------
  # Boundary check for lambda grid
  # ----------------------------------------------------------
  
  lambda_boundary <-
    final_lambda %in%
    range(lambda_grid)
  
  if (lambda_boundary) {
    
    warning(
      paste(
        sex_name,
        "final lambda is located at the boundary",
        "of the tuning grid. The lambda range",
        "should be reviewed before freezing the model."
      )
    )
  }
  
  list(
    all_tuning =
      all_tuning_results,
    best_by_seed =
      best_tuning_results,
    selection_by_seed =
      selection_results,
    selection_frequency =
      selection_frequency,
    aggregated_tuning =
      aggregated_tuning,
    hyperparameter_summary =
      hyperparameter_summary,
    final_hyperparameters =
      final_hyperparameters,
    final_coefficients =
      final_coefficients,
    final_selected_predictors =
      final_selected_predictors,
    scaling_parameters =
      scaling_parameters,
    fold_assignments =
      fold_results,
    final_model =
      final_model,
    scaling_model =
      scaling_model
  )
}


# ============================================================
# 13. RUN WOMEN
# ============================================================

women_en <- run_elastic_net_stability(
  data = women_model_data,
  sex_name = "Women"
)


# ============================================================
# 14. RUN MEN
# ============================================================

men_en <- run_elastic_net_stability(
  data = men_model_data,
  sex_name = "Men"
)


# ============================================================
# 15. SAVE WOMEN RESULTS
# ============================================================

write_csv(
  women_en$best_by_seed,
  file.path(
    output_dir,
    "03_women_best_tuning_by_seed.csv"
  )
)

write_csv(
  women_en$selection_by_seed,
  file.path(
    output_dir,
    "04_women_selection_by_seed.csv"
  )
)

write_csv(
  women_en$selection_frequency,
  file.path(
    output_dir,
    "05_women_selection_frequency.csv"
  )
)

write_csv(
  women_en$aggregated_tuning,
  file.path(
    output_dir,
    "06_women_aggregated_tuning.csv"
  )
)

write_csv(
  women_en$hyperparameter_summary,
  file.path(
    output_dir,
    "07_women_hyperparameter_stability.csv"
  )
)

write_csv(
  women_en$final_hyperparameters,
  file.path(
    output_dir,
    "08_women_final_hyperparameters.csv"
  )
)

write_csv(
  women_en$final_coefficients,
  file.path(
    output_dir,
    "09_women_final_coefficients.csv"
  )
)

write_csv(
  women_en$final_selected_predictors,
  file.path(
    output_dir,
    "10_women_final_selected_predictors.csv"
  )
)

write_csv(
  women_en$scaling_parameters,
  file.path(
    output_dir,
    "11_women_scaling_parameters.csv"
  )
)

write_csv(
  women_en$fold_assignments,
  file.path(
    output_dir,
    "12_women_cv_fold_assignments.csv"
  )
)


# ============================================================
# 16. SAVE MEN RESULTS
# ============================================================

write_csv(
  men_en$best_by_seed,
  file.path(
    output_dir,
    "13_men_best_tuning_by_seed.csv"
  )
)

write_csv(
  men_en$selection_by_seed,
  file.path(
    output_dir,
    "14_men_selection_by_seed.csv"
  )
)

write_csv(
  men_en$selection_frequency,
  file.path(
    output_dir,
    "15_men_selection_frequency.csv"
  )
)

write_csv(
  men_en$aggregated_tuning,
  file.path(
    output_dir,
    "16_men_aggregated_tuning.csv"
  )
)

write_csv(
  men_en$hyperparameter_summary,
  file.path(
    output_dir,
    "17_men_hyperparameter_stability.csv"
  )
)

write_csv(
  men_en$final_hyperparameters,
  file.path(
    output_dir,
    "18_men_final_hyperparameters.csv"
  )
)

write_csv(
  men_en$final_coefficients,
  file.path(
    output_dir,
    "19_men_final_coefficients.csv"
  )
)

write_csv(
  men_en$final_selected_predictors,
  file.path(
    output_dir,
    "20_men_final_selected_predictors.csv"
  )
)

write_csv(
  men_en$scaling_parameters,
  file.path(
    output_dir,
    "21_men_scaling_parameters.csv"
  )
)

write_csv(
  men_en$fold_assignments,
  file.path(
    output_dir,
    "22_men_cv_fold_assignments.csv"
  )
)


# ============================================================
# 17. SAVE FINAL MODELS AND SCALING OBJECTS
# ============================================================

saveRDS(
  women_en$final_model,
  file.path(
    output_dir,
    "women_final_elastic_net_model.rds"
  )
)

saveRDS(
  women_en$scaling_model,
  file.path(
    output_dir,
    "women_scaling_model.rds"
  )
)

saveRDS(
  men_en$final_model,
  file.path(
    output_dir,
    "men_final_elastic_net_model.rds"
  )
)

saveRDS(
  men_en$scaling_model,
  file.path(
    output_dir,
    "men_scaling_model.rds"
  )
)


# ============================================================
# 18. DISPLAY FINAL RESULTS
# ============================================================

cat(
  "\n\n==================================================\n"
)
cat(
  "WOMEN - FINAL ELASTIC NET PARAMETERS\n"
)
cat(
  "==================================================\n"
)

print(
  women_en$final_hyperparameters
)

cat(
  "\nWOMEN - FINAL SELECTED PREDICTORS\n"
)

print(
  women_en$final_selected_predictors
)

cat(
  "\nWOMEN - SELECTION FREQUENCY\n"
)

print(
  women_en$selection_frequency
)


cat(
  "\n\n==================================================\n"
)
cat(
  "MEN - FINAL ELASTIC NET PARAMETERS\n"
)
cat(
  "==================================================\n"
)

print(
  men_en$final_hyperparameters
)

cat(
  "\nMEN - FINAL SELECTED PREDICTORS\n"
)

print(
  men_en$final_selected_predictors
)

cat(
  "\nMEN - SELECTION FREQUENCY\n"
)

print(
  men_en$selection_frequency
)


# ============================================================
# 19. FINAL SAFETY MESSAGE
# ============================================================

cat(
  "\n\n==================================================\n",
  "SCRIPT 05 COMPLETED SUCCESSFULLY\n",
  "==================================================\n",
  "Only the fixed TRAIN subsets were used.\n",
  "TEST data were never read or accessed.\n",
  "HCY remained on its original scale.\n",
  "Predictor transformations were previously frozen.\n",
  "Centering and scaling were estimated inside CV folds.\n",
  "Elastic Net tuning stability was evaluated across 10 CV seeds.\n",
  "No stability-based predictor groups were created.\n",
  "One final predictor set was obtained for each sex.\n",
  "==================================================\n"
)
