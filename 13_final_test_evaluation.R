# ============================================================
# SCRIPT 13
# FINAL HELD-OUT TEST EVALUATION
# CORRECTED COMPLETE VERSION
# ============================================================
#
# FINAL MODELING WORKFLOWS
#
# 1. MLR
# 2. PR
# 3. SVR-Linear
# 4. SVR-Polynomial
# 5. SVR-RBF
# 6. RF
# 7. XGB
#
# IMPORTANT
#
# - Model specifications were frozen before TEST evaluation.
# - TRAIN is used for model fitting.
# - TEST is used only for final evaluation.
# - TEST is not used for variable selection.
# - TEST is not used for hyperparameter tuning.
# - TEST is not used for transformation decisions.
# - TEST is not used for scaling estimation.
# - HCY remains on its original scale.
#
# MLR / PR / SVR
#
# Women:
#   BMI
#   VisceralFat
#   MCHC
#   GLU
#   sqrt(LDL)
#
# Men:
#   BMI
#   log(HDL)
#
# RF / XGB
#
# - All 21 eligible original predictors.
# - No transformations.
# - No centering/scaling.
#
# OFFICIAL METRICS
#
# RMSE
# MAE
# Predictive R2 = 1 - SSE/SST
#
# Confidence intervals and paired model comparisons are NOT
# performed here. They will be obtained in Script 14 using
# paired bootstrap with 2000 replicates.
#
# ============================================================


# ============================================================
# 1. PACKAGES
# ============================================================

required_packages <- c(
  "caret",
  "kernlab",
  "ranger",
  "xgboost",
  "readr",
  "dplyr"
)


missing_packages <- required_packages[
  !required_packages %in%
    rownames(
      installed.packages()
    )
]


if (
  length(
    missing_packages
  ) > 0
) {
  
  install.packages(
    missing_packages
  )
}


library(caret)
library(kernlab)
library(readr)
library(dplyr)


# Load remaining namespaces explicitly.

invisible(
  loadNamespace(
    "ranger"
  )
)


invisible(
  loadNamespace(
    "xgboost"
  )
)


# ============================================================
# 2. ANALYSIS VERSION
# ============================================================

analysis_version <-
  "FINAL_TEST_EVALUATION_20261024_V2"


# ============================================================
# 3. DIRECTORIES
# ============================================================

elastic_net_dir <-
  "05_elastic_net_results"


xgb_dir <-
  "11_XGB_results"


output_dir <-
  "13_final_test_results"


if (
  !dir.exists(
    output_dir
  )
) {
  
  dir.create(
    output_dir
  )
}


# ============================================================
# 4. TRAIN INPUT FILES
#
# TEST files are NOT read in this section.
# ============================================================

women_raw_train_file <-
  "01_women_TRAIN.csv"


men_raw_train_file <-
  "03_men_TRAIN.csv"


women_transformed_train_file <-
  file.path(
    elastic_net_dir,
    "01_women_TRAIN_transformed.csv"
  )


men_transformed_train_file <-
  file.path(
    elastic_net_dir,
    "02_men_TRAIN_transformed.csv"
  )


women_selected_file <-
  file.path(
    elastic_net_dir,
    "10_women_final_selected_predictors.csv"
  )


men_selected_file <-
  file.path(
    elastic_net_dir,
    "20_men_final_selected_predictors.csv"
  )


women_xgb_model_file <-
  file.path(
    xgb_dir,
    "women_XGB_model.json"
  )


men_xgb_model_file <-
  file.path(
    xgb_dir,
    "men_XGB_model.json"
  )


train_required_files <- c(
  
  women_raw_train_file,
  
  men_raw_train_file,
  
  women_transformed_train_file,
  
  men_transformed_train_file,
  
  women_selected_file,
  
  men_selected_file,
  
  women_xgb_model_file,
  
  men_xgb_model_file
)


missing_train_files <- train_required_files[
  !file.exists(
    train_required_files
  )
]


if (
  length(
    missing_train_files
  ) > 0
) {
  
  stop(
    paste(
      "Missing TRAIN/model files:\n",
      paste(
        missing_train_files,
        collapse = "\n"
      )
    )
  )
}


# ============================================================
# 5. READ TRAIN DATA ONLY
# ============================================================

women_raw_train <- readr::read_csv(
  women_raw_train_file,
  show_col_types = FALSE
)


men_raw_train <- readr::read_csv(
  men_raw_train_file,
  show_col_types = FALSE
)


women_transformed_train <- readr::read_csv(
  women_transformed_train_file,
  show_col_types = FALSE
)


men_transformed_train <- readr::read_csv(
  men_transformed_train_file,
  show_col_types = FALSE
)


women_selected_table <- readr::read_csv(
  women_selected_file,
  show_col_types = FALSE
)


men_selected_table <- readr::read_csv(
  men_selected_file,
  show_col_types = FALSE
)


cat("\n========================================\n")
cat("TRAIN DATA LOADED\n")
cat("========================================\n")

cat(
  "Women TRAIN:",
  nrow(
    women_raw_train
  ),
  "\n"
)

cat(
  "Men TRAIN:",
  nrow(
    men_raw_train
  ),
  "\n"
)

cat(
  "TEST has NOT been read in this execution.\n"
)


# ============================================================
# 6. STANDARDIZE VARIABLE NAMES
# ============================================================

standardize_variable_names <- function(
    data
) {
  
  rename_map <- c(
    
    "EDAD" =
      "Age",
    
    "TALLA" =
      "Height",
    
    "PESO" =
      "Weight",
    
    "IMC" =
      "BMI",
    
    "GRASAT" =
      "TotalFat",
    
    "GRASAVIS" =
      "VisceralFat",
    
    "MUSCULO" =
      "MuscleMass",
    
    "HGB" =
      "Hgb",
    
    "HTO" =
      "Hct",
    
    "VCM" =
      "MCV",
    
    "RDWCV" =
      "RDW_CV",
    
    "COL" =
      "TotalCholesterol"
  )
  
  
  for (
    old_name in
    names(rename_map)
  ) {
    
    new_name <- unname(
      rename_map[
        old_name
      ]
    )
    
    
    if (
      old_name %in%
      names(data) &&
      !(new_name %in%
        names(data))
    ) {
      
      names(data)[
        names(data) ==
          old_name
      ] <- new_name
    }
  }
  
  
  return(data)
}


women_raw_train <-
  standardize_variable_names(
    women_raw_train
  )


men_raw_train <-
  standardize_variable_names(
    men_raw_train
  )


women_transformed_train <-
  standardize_variable_names(
    women_transformed_train
  )


men_transformed_train <-
  standardize_variable_names(
    men_transformed_train
  )


# ============================================================
# 7. TRAIN SAMPLE-SIZE CHECKS
# ============================================================

if (
  nrow(
    women_raw_train
  ) != 190
) {
  
  stop(
    paste(
      "Women TRAIN should contain 190 observations.",
      "Observed:",
      nrow(
        women_raw_train
      )
    )
  )
}


if (
  nrow(
    men_raw_train
  ) != 83
) {
  
  stop(
    paste(
      "Men TRAIN should contain 83 observations.",
      "Observed:",
      nrow(
        men_raw_train
      )
    )
  )
}


if (
  nrow(
    women_transformed_train
  ) != 190
) {
  
  stop(
    "Transformed Women TRAIN must contain 190 observations."
  )
}


if (
  nrow(
    men_transformed_train
  ) != 83
) {
  
  stop(
    "Transformed Men TRAIN must contain 83 observations."
  )
}


# ============================================================
# 8. FINAL 21 ELIGIBLE ORIGINAL PREDICTORS
# ============================================================

all_predictors <- c(
  
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


if (
  length(
    all_predictors
  ) != 21
) {
  
  stop(
    "Exactly 21 eligible predictors are required."
  )
}


# ============================================================
# 9. FROZEN EN-SELECTED PREDICTORS
# ============================================================

women_selected_predictors <- c(
  "BMI",
  "VisceralFat",
  "MCHC",
  "GLU",
  "LDL"
)


men_selected_predictors <- c(
  "BMI",
  "HDL"
)


# ============================================================
# 10. VERIFY SELECTION FILES
# ============================================================

if (
  !"Variable" %in%
  names(
    women_selected_table
  )
) {
  
  stop(
    "Column 'Variable' not found in Women selection file."
  )
}


if (
  !"Variable" %in%
  names(
    men_selected_table
  )
) {
  
  stop(
    "Column 'Variable' not found in Men selection file."
  )
}


if (
  !setequal(
    as.character(
      women_selected_table$Variable
    ),
    women_selected_predictors
  )
) {
  
  stop(
    paste(
      "Women selected predictors do not match the frozen set.",
      "Found:",
      paste(
        women_selected_table$Variable,
        collapse = ", "
      )
    )
  )
}


if (
  !setequal(
    as.character(
      men_selected_table$Variable
    ),
    men_selected_predictors
  )
) {
  
  stop(
    paste(
      "Men selected predictors do not match the frozen set.",
      "Found:",
      paste(
        men_selected_table$Variable,
        collapse = ", "
      )
    )
  )
}


cat("\n========================================\n")
cat("FROZEN SELECTED PREDICTORS\n")
cat("========================================\n")

cat(
  "Women:",
  paste(
    women_selected_predictors,
    collapse = ", "
  ),
  "\n"
)

cat(
  "Men:",
  paste(
    men_selected_predictors,
    collapse = ", "
  ),
  "\n"
)


# ============================================================
# 11. VALIDATE RAW DATASET
# ============================================================

validate_raw_dataset <- function(
    data,
    dataset_name
) {
  
  required_columns <- c(
    "ID",
    "HCY",
    all_predictors
  )
  
  
  missing_columns <- setdiff(
    required_columns,
    names(data)
  )
  
  
  if (
    length(
      missing_columns
    ) > 0
  ) {
    
    stop(
      paste(
        dataset_name,
        "is missing:",
        paste(
          missing_columns,
          collapse = ", "
        )
      )
    )
  }
  
  
  numeric_columns <- c(
    "HCY",
    all_predictors
  )
  
  
  for (
    variable in
    numeric_columns
  ) {
    
    values <- data[
      ,
      variable,
      drop = TRUE
    ]
    
    
    if (
      !is.numeric(
        values
      )
    ) {
      
      stop(
        paste(
          dataset_name,
          variable,
          "is not numeric."
        )
      )
    }
    
    
    if (
      any(
        is.na(
          values
        )
      )
    ) {
      
      stop(
        paste(
          dataset_name,
          variable,
          "contains missing values."
        )
      )
    }
    
    
    if (
      any(
        !is.finite(
          values
        )
      )
    ) {
      
      stop(
        paste(
          dataset_name,
          variable,
          "contains non-finite values."
        )
      )
    }
  }
  
  
  invisible(TRUE)
}


validate_raw_dataset(
  women_raw_train,
  "Women TRAIN"
)


validate_raw_dataset(
  men_raw_train,
  "Men TRAIN"
)


# ============================================================
# 12. VERIFY RAW VS TRANSFORMED TRAIN LINEAGE
# ============================================================

if (
  !all(
    as.character(
      women_raw_train$ID
    ) ==
    as.character(
      women_transformed_train$ID
    )
  )
) {
  
  stop(
    "Women raw/transformed TRAIN row order does not match."
  )
}


if (
  !all(
    as.character(
      men_raw_train$ID
    ) ==
    as.character(
      men_transformed_train$ID
    )
  )
) {
  
  stop(
    "Men raw/transformed TRAIN row order does not match."
  )
}


if (
  max(
    abs(
      women_raw_train$HCY -
      women_transformed_train$HCY
    )
  ) > 1e-10
) {
  
  stop(
    "Women HCY differs between raw and transformed TRAIN."
  )
}


if (
  max(
    abs(
      men_raw_train$HCY -
      men_transformed_train$HCY
    )
  ) > 1e-10
) {
  
  stop(
    "Men HCY differs between raw and transformed TRAIN."
  )
}


# ============================================================
# 13. VERIFY FROZEN TRAIN TRANSFORMATIONS
#
# Only transformations used by the final selected predictors
# are checked here.
# ============================================================

if (
  any(
    women_raw_train$LDL < 0
  )
) {
  
  stop(
    "Negative LDL detected in Women TRAIN."
  )
}


if (
  any(
    men_raw_train$HDL <= 0
  )
) {
  
  stop(
    "Non-positive HDL detected in Men TRAIN."
  )
}


women_transformation_check <- c(
  
  BMI =
    max(
      abs(
        women_transformed_train$BMI -
          women_raw_train$BMI
      )
    ),
  
  VisceralFat =
    max(
      abs(
        women_transformed_train$VisceralFat -
          women_raw_train$VisceralFat
      )
    ),
  
  MCHC =
    max(
      abs(
        women_transformed_train$MCHC -
          women_raw_train$MCHC
      )
    ),
  
  GLU =
    max(
      abs(
        women_transformed_train$GLU -
          women_raw_train$GLU
      )
    ),
  
  LDL =
    max(
      abs(
        women_transformed_train$LDL -
          sqrt(
            women_raw_train$LDL
          )
      )
    )
)


men_transformation_check <- c(
  
  BMI =
    max(
      abs(
        men_transformed_train$BMI -
          men_raw_train$BMI
      )
    ),
  
  HDL =
    max(
      abs(
        men_transformed_train$HDL -
          log(
            men_raw_train$HDL
          )
      )
    )
)


if (
  max(
    women_transformation_check
  ) > 1e-8
) {
  
  stop(
    "Women transformed TRAIN does not reproduce the frozen transformations."
  )
}


if (
  max(
    men_transformation_check
  ) > 1e-8
) {
  
  stop(
    "Men transformed TRAIN does not reproduce the frozen transformations."
  )
}


cat("\n========================================\n")
cat("TRAIN TRANSFORMATION CHECK\n")
cat("========================================\n")

cat(
  "Women transformations: PASSED\n"
)

cat(
  "Men transformations: PASSED\n"
)


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
  
  
  if (
    sst == 0
  ) {
    
    return(
      NA_real_
    )
  }
  
  
  return(
    1 -
      sse /
      sst
  )
}


# ============================================================
# 15. CPU SETTINGS
# ============================================================

detected_cores <- parallel::detectCores(
  logical = TRUE
)


if (
  is.na(
    detected_cores
  ) ||
  detected_cores < 2
) {
  
  model_threads <- 1
  
} else {
  
  model_threads <- min(
    4,
    detected_cores - 1
  )
}


cat("\n========================================\n")
cat("COMPUTATION SETTINGS\n")
cat("========================================\n")

cat(
  "Model threads:",
  model_threads,
  "\n"
)


# ============================================================
# 16. SCALING FUNCTIONS
# ============================================================

fit_scaler <- function(
    x
) {
  
  x <- as.matrix(
    x
  )
  
  
  means <- colMeans(
    x
  )
  
  
  sds <- apply(
    x,
    2,
    stats::sd
  )
  
  
  invalid_sd <- which(
    !is.finite(
      sds
    ) |
      sds == 0
  )
  
  
  if (
    length(
      invalid_sd
    ) > 0
  ) {
    
    sds[
      invalid_sd
    ] <- 1
  }
  
  
  return(
    
    list(
      
      mean =
        means,
      
      sd =
        sds
    )
  )
}


apply_scaler <- function(
    x,
    scaler
) {
  
  x <- as.matrix(
    x
  )
  
  
  result <- sweep(
    x,
    2,
    scaler$mean,
    FUN = "-"
  )
  
  
  result <- sweep(
    result,
    2,
    scaler$sd,
    FUN = "/"
  )
  
  
  return(
    as.matrix(
      result
    )
  )
}


# ============================================================
# 17. ADDITIVE RAW POLYNOMIAL FEATURES
#
# Degree 2.
# No interaction terms.
# ============================================================

create_polynomial_matrix <- function(
    x,
    degree = 2
) {
  
  x <- as.matrix(
    x
  )
  
  
  blocks <- lapply(
    
    seq_len(
      ncol(
        x
      )
    ),
    
    function(j) {
      
      values <- x[
        ,
        j
      ]
      
      
      block <- sapply(
        
        seq_len(
          degree
        ),
        
        function(d) {
          
          values^d
        }
      )
      
      
      if (
        degree == 1
      ) {
        
        block <- matrix(
          block,
          ncol = 1
        )
      }
      
      
      colnames(
        block
      ) <- paste0(
        
        colnames(x)[j],
        
        "_degree",
        
        seq_len(
          degree
        )
      )
      
      
      return(
        block
      )
    }
  )
  
  
  return(
    
    as.matrix(
      
      do.call(
        cbind,
        blocks
      )
    )
  )
}


# ============================================================
# 18. FIT SCALED LINEAR MODEL
# ============================================================

fit_scaled_linear_model <- function(
    x_train,
    y_train
) {
  
  scaler <- fit_scaler(
    x_train
  )
  
  
  x_train_scaled <- apply_scaler(
    x_train,
    scaler
  )
  
  
  design <- cbind(
    Intercept = 1,
    x_train_scaled
  )
  
  
  fit <- stats::lm.fit(
    x =
      design,
    y =
      y_train
  )
  
  
  coefficients <-
    fit$coefficients
  
  
  coefficients[
    is.na(
      coefficients
    )
  ] <- 0
  
  
  return(
    
    list(
      
      coefficients =
        coefficients,
      
      scaler =
        scaler,
      
      feature_names =
        colnames(
          x_train
        )
    )
  )
}


# ============================================================
# 19. PREDICT SCALED LINEAR MODEL
# ============================================================

predict_scaled_linear_model <- function(
    model,
    x_new
) {
  
  x_new <- as.matrix(
    x_new
  )
  
  
  if (
    !identical(
      colnames(
        x_new
      ),
      model$feature_names
    )
  ) {
    
    stop(
      "Linear-model feature names/order do not match the fitted model."
    )
  }
  
  
  x_scaled <- apply_scaler(
    x_new,
    model$scaler
  )
  
  
  design <- cbind(
    Intercept = 1,
    x_scaled
  )
  
  
  return(
    
    as.numeric(
      
      design %*%
        model$coefficients
    )
  )
}


# ============================================================
# 20. FROZEN SVR SPECIFICATIONS
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
  
  stringsAsFactors =
    FALSE
)


# ============================================================
# 21. FROZEN RANDOM FOREST SPECIFICATIONS
# ============================================================

rf_specifications <- data.frame(
  
  Sex = c(
    "Women",
    "Men"
  ),
  
  mtry = c(
    2,
    4
  ),
  
  min_node_size = c(
    15,
    15
  ),
  
  sample_fraction = c(
    0.80,
    0.80
  ),
  
  num_trees = c(
    500,
    500
  ),
  
  stringsAsFactors =
    FALSE
)


# ============================================================
# 22. FROZEN XGBOOST SPECIFICATIONS
# ============================================================

xgb_specifications <- data.frame(
  
  Sex = c(
    "Women",
    "Men"
  ),
  
  nrounds = c(
    200,
    600
  ),
  
  max_depth = c(
    1,
    1
  ),
  
  eta = c(
    0.03,
    0.01
  ),
  
  gamma = c(
    0,
    0.5
  ),
  
  colsample_bytree = c(
    0.80,
    0.60
  ),
  
  min_child_weight = c(
    1,
    3
  ),
  
  subsample = c(
    0.60,
    1.00
  ),
  
  reg_alpha = c(
    1,
    1
  ),
  
  reg_lambda = c(
    5,
    0.5
  ),
  
  stringsAsFactors =
    FALSE
)


# ============================================================
# 23. MODEL NAMES
# ============================================================

model_names <- c(
  "MLR",
  "PR",
  "SVR-Linear",
  "SVR-Polynomial",
  "SVR-RBF",
  "RF",
  "XGB"
)


# ============================================================
# 24. CREATE SVR TUNING GRID
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
    "Unknown SVR method."
  )
}


# ============================================================
# 25. FIT FINAL SVR
#
# Exact caret model families used in Script 09 and 12B.
# No tuning occurs here.
# ============================================================

fit_final_svr <- function(
    train_data,
    predictors,
    sex_name,
    model_name
) {
  
  specification <- svr_specifications %>%
    
    dplyr::filter(
      Sex ==
        sex_name,
      Model ==
        model_name
    )
  
  
  if (
    nrow(
      specification
    ) != 1
  ) {
    
    stop(
      paste(
        "SVR specification error:",
        sex_name,
        model_name
      )
    )
  }
  
  
  tuning_grid <-
    create_svr_tune_grid(
      specification
    )
  
  
  formula_object <- stats::as.formula(
    
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
    123
  )
  
  
  model <- caret::train(
    
    formula_object,
    
    data =
      train_data[
        ,
        c(
          "HCY",
          predictors
        ),
        drop = FALSE
      ],
    
    method =
      specification$Caret_Method[1],
    
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
  
  
  return(
    model
  )
}


# ============================================================
# 26. FIT FINAL RANDOM FOREST
# ============================================================

fit_final_rf <- function(
    train_data,
    sex_name
) {
  
  specification <- rf_specifications %>%
    
    dplyr::filter(
      Sex ==
        sex_name
    )
  
  
  if (
    nrow(
      specification
    ) != 1
  ) {
    
    stop(
      paste(
        "RF specification error:",
        sex_name
      )
    )
  }
  
  
  train_frame <- as.data.frame(
    
    train_data[
      ,
      c(
        "HCY",
        all_predictors
      ),
      drop = FALSE
    ]
  )
  
  
  set.seed(
    123
  )
  
  
  model <- ranger::ranger(
    
    dependent.variable.name =
      "HCY",
    
    data =
      train_frame,
    
    num.trees =
      as.integer(
        specification$num_trees[1]
      ),
    
    mtry =
      as.integer(
        specification$mtry[1]
      ),
    
    min.node.size =
      as.integer(
        specification$min_node_size[1]
      ),
    
    sample.fraction =
      specification$sample_fraction[1],
    
    replace =
      TRUE,
    
    splitrule =
      "variance",
    
    importance =
      "none",
    
    seed =
      123,
    
    num.threads =
      model_threads
  )
  
  
  return(
    model
  )
}


# ============================================================
# 27. FIT / LOAD ALL FINAL MODELS USING TRAIN ONLY
#
# TEST IS STILL NOT READ IN THIS EXECUTION.
# ============================================================

cat("\n========================================\n")
cat("FITTING FINAL MODELS USING TRAIN ONLY\n")
cat("========================================\n")


# ------------------------------------------------------------
# WOMEN MLR
# ------------------------------------------------------------

women_mlr_x <- as.matrix(
  
  women_transformed_train[
    ,
    women_selected_predictors,
    drop = FALSE
  ]
)


women_mlr_model <- fit_scaled_linear_model(
  
  women_mlr_x,
  
  women_transformed_train$HCY
)


# ------------------------------------------------------------
# MEN MLR
# ------------------------------------------------------------

men_mlr_x <- as.matrix(
  
  men_transformed_train[
    ,
    men_selected_predictors,
    drop = FALSE
  ]
)


men_mlr_model <- fit_scaled_linear_model(
  
  men_mlr_x,
  
  men_transformed_train$HCY
)


# ------------------------------------------------------------
# WOMEN PR
# ------------------------------------------------------------

women_pr_x <- create_polynomial_matrix(
  
  women_mlr_x,
  
  degree =
    2
)


women_pr_model <- fit_scaled_linear_model(
  
  women_pr_x,
  
  women_transformed_train$HCY
)


# ------------------------------------------------------------
# MEN PR
# ------------------------------------------------------------

men_pr_x <- create_polynomial_matrix(
  
  men_mlr_x,
  
  degree =
    2
)


men_pr_model <- fit_scaled_linear_model(
  
  men_pr_x,
  
  men_transformed_train$HCY
)


# ------------------------------------------------------------
# WOMEN SVR
# ------------------------------------------------------------

women_svr_linear <- fit_final_svr(
  
  train_data =
    women_transformed_train,
  
  predictors =
    women_selected_predictors,
  
  sex_name =
    "Women",
  
  model_name =
    "SVR-Linear"
)


women_svr_polynomial <- fit_final_svr(
  
  train_data =
    women_transformed_train,
  
  predictors =
    women_selected_predictors,
  
  sex_name =
    "Women",
  
  model_name =
    "SVR-Polynomial"
)


women_svr_rbf <- fit_final_svr(
  
  train_data =
    women_transformed_train,
  
  predictors =
    women_selected_predictors,
  
  sex_name =
    "Women",
  
  model_name =
    "SVR-RBF"
)


# ------------------------------------------------------------
# MEN SVR
# ------------------------------------------------------------

men_svr_linear <- fit_final_svr(
  
  train_data =
    men_transformed_train,
  
  predictors =
    men_selected_predictors,
  
  sex_name =
    "Men",
  
  model_name =
    "SVR-Linear"
)


men_svr_polynomial <- fit_final_svr(
  
  train_data =
    men_transformed_train,
  
  predictors =
    men_selected_predictors,
  
  sex_name =
    "Men",
  
  model_name =
    "SVR-Polynomial"
)


men_svr_rbf <- fit_final_svr(
  
  train_data =
    men_transformed_train,
  
  predictors =
    men_selected_predictors,
  
  sex_name =
    "Men",
  
  model_name =
    "SVR-RBF"
)


# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

women_rf_model <- fit_final_rf(
  
  train_data =
    women_raw_train,
  
  sex_name =
    "Women"
)


men_rf_model <- fit_final_rf(
  
  train_data =
    men_raw_train,
  
  sex_name =
    "Men"
)


# ------------------------------------------------------------
# XGBOOST
#
# Use the already frozen final models from Script 11.
# ------------------------------------------------------------

women_xgb_model <- xgboost::xgb.load(
  women_xgb_model_file
)


men_xgb_model <- xgboost::xgb.load(
  men_xgb_model_file
)


cat(
  "All final models fitted/loaded using TRAIN only.\n"
)

cat(
  "TEST has still NOT been read in this execution.\n"
)


# ============================================================
# 28. TRAIN PREDICTIONS
# ============================================================

women_train_predictions <- data.frame(
  
  ID =
    women_raw_train$ID,
  
  Observed_HCY =
    women_raw_train$HCY,
  
  MLR =
    predict_scaled_linear_model(
      women_mlr_model,
      women_mlr_x
    ),
  
  PR =
    predict_scaled_linear_model(
      women_pr_model,
      women_pr_x
    ),
  
  `SVR-Linear` =
    as.numeric(
      
      stats::predict(
        
        women_svr_linear,
        
        newdata =
          women_transformed_train
      )
    ),
  
  `SVR-Polynomial` =
    as.numeric(
      
      stats::predict(
        
        women_svr_polynomial,
        
        newdata =
          women_transformed_train
      )
    ),
  
  `SVR-RBF` =
    as.numeric(
      
      stats::predict(
        
        women_svr_rbf,
        
        newdata =
          women_transformed_train
      )
    ),
  
  RF =
    as.numeric(
      
      stats::predict(
        
        women_rf_model,
        
        data =
          as.data.frame(
            
            women_raw_train[
              ,
              all_predictors,
              drop = FALSE
            ]
          )
      )$predictions
    ),
  
  XGB =
    as.numeric(
      
      stats::predict(
        
        women_xgb_model,
        
        newdata =
          as.matrix(
            
            women_raw_train[
              ,
              all_predictors,
              drop = FALSE
            ]
          )
      )
    ),
  
  check.names =
    FALSE
)


men_train_predictions <- data.frame(
  
  ID =
    men_raw_train$ID,
  
  Observed_HCY =
    men_raw_train$HCY,
  
  MLR =
    predict_scaled_linear_model(
      men_mlr_model,
      men_mlr_x
    ),
  
  PR =
    predict_scaled_linear_model(
      men_pr_model,
      men_pr_x
    ),
  
  `SVR-Linear` =
    as.numeric(
      
      stats::predict(
        
        men_svr_linear,
        
        newdata =
          men_transformed_train
      )
    ),
  
  `SVR-Polynomial` =
    as.numeric(
      
      stats::predict(
        
        men_svr_polynomial,
        
        newdata =
          men_transformed_train
      )
    ),
  
  `SVR-RBF` =
    as.numeric(
      
      stats::predict(
        
        men_svr_rbf,
        
        newdata =
          men_transformed_train
      )
    ),
  
  RF =
    as.numeric(
      
      stats::predict(
        
        men_rf_model,
        
        data =
          as.data.frame(
            
            men_raw_train[
              ,
              all_predictors,
              drop = FALSE
            ]
          )
      )$predictions
    ),
  
  XGB =
    as.numeric(
      
      stats::predict(
        
        men_xgb_model,
        
        newdata =
          as.matrix(
            
            men_raw_train[
              ,
              all_predictors,
              drop = FALSE
            ]
          )
      )
    ),
  
  check.names =
    FALSE
)


# ============================================================
# 29. VALIDATE PREDICTION TABLE
# ============================================================

validate_prediction_table <- function(
    predictions,
    expected_n,
    dataset_name
) {
  
  if (
    nrow(
      predictions
    ) !=
    expected_n
  ) {
    
    stop(
      paste(
        dataset_name,
        "prediction table has incorrect size."
      )
    )
  }
  
  
  for (
    model_name in
    model_names
  ) {
    
    values <- predictions[
      ,
      model_name,
      drop = TRUE
    ]
    
    
    if (
      length(
        values
      ) !=
      expected_n
    ) {
      
      stop(
        paste(
          dataset_name,
          model_name,
          "has incorrect prediction length."
        )
      )
    }
    
    
    if (
      any(
        is.na(
          values
        )
      )
    ) {
      
      stop(
        paste(
          dataset_name,
          model_name,
          "contains NA predictions."
        )
      )
    }
    
    
    if (
      any(
        !is.finite(
          values
        )
      )
    ) {
      
      stop(
        paste(
          dataset_name,
          model_name,
          "contains non-finite predictions."
        )
      )
    }
  }
  
  
  invisible(TRUE)
}


validate_prediction_table(
  women_train_predictions,
  190,
  "Women TRAIN"
)


validate_prediction_table(
  men_train_predictions,
  83,
  "Men TRAIN"
)


# ============================================================
# 30. METRIC TABLE FUNCTION
# ============================================================

calculate_metric_table <- function(
    prediction_table,
    sex_name,
    dataset_name
) {
  
  rows <- list()
  
  
  observed <-
    prediction_table$Observed_HCY
  
  
  for (
    model_name in
    model_names
  ) {
    
    predicted <- prediction_table[
      ,
      model_name,
      drop = TRUE
    ]
    
    
    result_row <- data.frame(
      
      Sex =
        sex_name,
      
      Dataset =
        dataset_name,
      
      Model =
        model_name,
      
      N =
        length(
          observed
        ),
      
      RMSE =
        calculate_rmse(
          observed,
          predicted
        ),
      
      MAE =
        calculate_mae(
          observed,
          predicted
        ),
      
      Predictive_R2 =
        calculate_predictive_r2(
          observed,
          predicted
        ),
      
      stringsAsFactors =
        FALSE
    )
    
    
    rows <- append(
      rows,
      list(
        result_row
      )
    )
  }
  
  
  return(
    
    dplyr::bind_rows(
      rows
    )
  )
}


# ============================================================
# 31. FINAL TRAIN METRICS
# ============================================================

train_metrics <- dplyr::bind_rows(
  
  calculate_metric_table(
    women_train_predictions,
    "Women",
    "TRAIN"
  ),
  
  calculate_metric_table(
    men_train_predictions,
    "Men",
    "TRAIN"
  )
)


if (
  nrow(
    train_metrics
  ) != 14
) {
  
  stop(
    "TRAIN metric table must contain exactly 14 rows."
  )
}


cat("\n========================================\n")
cat("FINAL TRAIN METRICS\n")
cat("TEST STILL NOT READ IN THIS EXECUTION\n")
cat("========================================\n")

print(
  train_metrics
)


# ============================================================
# 32. VERIFY KNOWN TRAIN METRICS
#
# This protects against accidental changes in the final
# workflows before TEST evaluation.
# ============================================================

known_train_reference <- data.frame(
  
  Sex = c(
    rep(
      "Women",
      7
    ),
    rep(
      "Men",
      7
    )
  ),
  
  Model = rep(
    model_names,
    2
  ),
  
  Expected_R2 = c(
    
    0.11836419,
    
    0.15049591,
    
    0.06581199,
    
    0.09709973,
    
    0.07586778,
    
    0.51299488,
    
    0.31164315,
    
    0.26955040,
    
    0.36500244,
    
    0.23379734,
    
    0.35316498,
    
    0.36074432,
    
    0.59018363,
    
    0.54326500
  ),
  
  stringsAsFactors =
    FALSE
)


train_reference_check <- train_metrics %>%
  
  dplyr::select(
    Sex,
    Model,
    Observed_R2 =
      Predictive_R2
  ) %>%
  
  dplyr::left_join(
    
    known_train_reference,
    
    by =
      c(
        "Sex",
        "Model"
      )
  ) %>%
  
  dplyr::mutate(
    
    Absolute_Difference =
      abs(
        Observed_R2 -
          Expected_R2
      )
  )


if (
  max(
    train_reference_check$Absolute_Difference,
    na.rm = TRUE
  ) > 1e-6
) {
  
  stop(
    paste(
      "Final TRAIN metrics no longer reproduce the frozen",
      "workflows. TEST evaluation was stopped."
    )
  )
}


cat(
  "Frozen TRAIN workflow check: PASSED\n"
)


# ============================================================
# 33. SAVE FROZEN WORKFLOW SPECIFICATIONS
#
# This is completed before reading TEST.
# ============================================================

workflow_specifications <- data.frame(
  
  Sex = c(
    rep(
      "Women",
      7
    ),
    rep(
      "Men",
      7
    )
  ),
  
  Model = rep(
    model_names,
    2
  ),
  
  Specification = c(
    
    paste(
      "5 EN-selected transformed predictors;",
      "full-TRAIN centering/scaling"
    ),
    
    paste(
      "5 EN-selected transformed predictors;",
      "additive degree-2 raw polynomial terms;",
      "full-TRAIN centering/scaling"
    ),
    
    paste(
      "caret svmLinear;",
      "C=0.0625;",
      "full-TRAIN center/scale"
    ),
    
    paste(
      "caret svmPoly;",
      "degree=8;",
      "scale=0.01;",
      "C=0.25;",
      "full-TRAIN center/scale"
    ),
    
    paste(
      "caret svmRadial;",
      "sigma=0.0078125;",
      "C=1;",
      "full-TRAIN center/scale"
    ),
    
    paste(
      "21 original predictors;",
      "mtry=2;",
      "min.node.size=15;",
      "sample.fraction=0.80;",
      "trees=500"
    ),
    
    paste(
      "21 original predictors;",
      "nrounds=200;",
      "depth=1;",
      "eta=0.03;",
      "gamma=0;",
      "colsample=0.80;",
      "min.child.weight=1;",
      "subsample=0.60;",
      "alpha=1;",
      "lambda=5"
    ),
    
    paste(
      "2 EN-selected transformed predictors;",
      "full-TRAIN centering/scaling"
    ),
    
    paste(
      "2 EN-selected transformed predictors;",
      "additive degree-2 raw polynomial terms;",
      "full-TRAIN centering/scaling"
    ),
    
    paste(
      "caret svmLinear;",
      "C=0.0625;",
      "full-TRAIN center/scale"
    ),
    
    paste(
      "caret svmPoly;",
      "degree=4;",
      "scale=0.1;",
      "C=0.25;",
      "full-TRAIN center/scale"
    ),
    
    paste(
      "caret svmRadial;",
      "sigma=0.015625;",
      "C=256;",
      "full-TRAIN center/scale"
    ),
    
    paste(
      "21 original predictors;",
      "mtry=4;",
      "min.node.size=15;",
      "sample.fraction=0.80;",
      "trees=500"
    ),
    
    paste(
      "21 original predictors;",
      "nrounds=600;",
      "depth=1;",
      "eta=0.01;",
      "gamma=0.5;",
      "colsample=0.60;",
      "min.child.weight=3;",
      "subsample=1.00;",
      "alpha=1;",
      "lambda=0.5"
    )
  ),
  
  stringsAsFactors =
    FALSE
)


workflow_freeze_time <-
  Sys.time()


readr::write_csv(
  
  workflow_specifications,
  
  file.path(
    output_dir,
    "00_final_workflow_specifications.csv"
  )
)


readr::write_csv(
  
  train_metrics,
  
  file.path(
    output_dir,
    "04_final_train_metrics.csv"
  )
)


readr::write_csv(
  
  train_reference_check,
  
  file.path(
    output_dir,
    "00B_frozen_TRAIN_metric_check.csv"
  )
)


# ============================================================
# 34. ========================================================
# FINAL HELD-OUT TEST EVALUATION
# ============================================================
#
# All model specifications above are fixed.
#
# From this point onward:
#
# - No hyperparameter changes.
# - No variable-selection changes.
# - No transformation changes.
# - No preprocessing decisions.
# - No model-selection changes.
#
# This script may be rerun for reproducibility, but its
# outputs must never be used to modify the frozen workflows.
#
# ============================================================

cat("\n\n")
cat("############################################################\n")
cat("# FINAL HELD-OUT TEST EVALUATION BEGINS\n")
cat("############################################################\n")


women_test_file <-
  "02_women_TEST.csv"


men_test_file <-
  "04_men_TEST.csv"


if (
  !file.exists(
    women_test_file
  )
) {
  
  stop(
    "02_women_TEST.csv not found."
  )
}


if (
  !file.exists(
    men_test_file
  )
) {
  
  stop(
    "04_men_TEST.csv not found."
  )
}


test_read_time <-
  Sys.time()


women_test <- readr::read_csv(
  women_test_file,
  show_col_types = FALSE
)


men_test <- readr::read_csv(
  men_test_file,
  show_col_types = FALSE
)


women_test <-
  standardize_variable_names(
    women_test
  )


men_test <-
  standardize_variable_names(
    men_test
  )


cat(
  "TEST files read for final evaluation.\n"
)


# ============================================================
# 35. TEST SAMPLE-SIZE AND VARIABLE CHECKS
# ============================================================

if (
  nrow(
    women_test
  ) != 50
) {
  
  stop(
    paste(
      "Women TEST should contain 50 observations.",
      "Observed:",
      nrow(
        women_test
      )
    )
  )
}


if (
  nrow(
    men_test
  ) != 25
) {
  
  stop(
    paste(
      "Men TEST should contain 25 observations.",
      "Observed:",
      nrow(
        men_test
      )
    )
  )
}


validate_raw_dataset(
  women_test,
  "Women TEST"
)


validate_raw_dataset(
  men_test,
  "Men TEST"
)


# ============================================================
# 36. TRAIN-TEST INDEPENDENCE CHECK
# ============================================================

women_id_overlap <- intersect(
  
  as.character(
    women_raw_train$ID
  ),
  
  as.character(
    women_test$ID
  )
)


men_id_overlap <- intersect(
  
  as.character(
    men_raw_train$ID
  ),
  
  as.character(
    men_test$ID
  )
)


overall_train_ids <- unique(
  
  c(
    as.character(
      women_raw_train$ID
    ),
    as.character(
      men_raw_train$ID
    )
  )
)


overall_test_ids <- unique(
  
  c(
    as.character(
      women_test$ID
    ),
    as.character(
      men_test$ID
    )
  )
)


overall_id_overlap <- intersect(
  overall_train_ids,
  overall_test_ids
)


if (
  length(
    women_id_overlap
  ) > 0
) {
  
  stop(
    "Women TRAIN/TEST ID overlap detected."
  )
}


if (
  length(
    men_id_overlap
  ) > 0
) {
  
  stop(
    "Men TRAIN/TEST ID overlap detected."
  )
}


if (
  length(
    overall_id_overlap
  ) > 0
) {
  
  stop(
    "Overall TRAIN/TEST ID overlap detected."
  )
}


women_test_duplicate_ids <- sum(
  duplicated(
    as.character(
      women_test$ID
    )
  )
)


men_test_duplicate_ids <- sum(
  duplicated(
    as.character(
      men_test$ID
    )
  )
)


cat("\n========================================\n")
cat("TEST INTEGRITY CHECK\n")
cat("========================================\n")

cat(
  "Women TEST:",
  nrow(
    women_test
  ),
  "\n"
)

cat(
  "Men TEST:",
  nrow(
    men_test
  ),
  "\n"
)

cat(
  "Women TRAIN/TEST ID overlap:",
  length(
    women_id_overlap
  ),
  "\n"
)

cat(
  "Men TRAIN/TEST ID overlap:",
  length(
    men_id_overlap
  ),
  "\n"
)

cat(
  "Overall TRAIN/TEST ID overlap:",
  length(
    overall_id_overlap
  ),
  "\n"
)

cat(
  "Women duplicated TEST IDs:",
  women_test_duplicate_ids,
  "\n"
)

cat(
  "Men duplicated TEST IDs:",
  men_test_duplicate_ids,
  "\n"
)


# ============================================================
# 37. APPLY PREDEFINED TRANSFORMATIONS TO TEST
#
# Transformation forms were determined from TRAIN.
# No transformation is chosen based on TEST.
# ============================================================

if (
  any(
    women_test$LDL < 0
  )
) {
  
  stop(
    "Negative LDL detected in Women TEST."
  )
}


if (
  any(
    men_test$HDL <= 0
  )
) {
  
  stop(
    "Non-positive HDL detected in Men TEST."
  )
}


women_test_transformed <- data.frame(
  
  ID =
    women_test$ID,
  
  HCY =
    women_test$HCY,
  
  BMI =
    women_test$BMI,
  
  VisceralFat =
    women_test$VisceralFat,
  
  MCHC =
    women_test$MCHC,
  
  GLU =
    women_test$GLU,
  
  LDL =
    sqrt(
      women_test$LDL
    )
)


men_test_transformed <- data.frame(
  
  ID =
    men_test$ID,
  
  HCY =
    men_test$HCY,
  
  BMI =
    men_test$BMI,
  
  HDL =
    log(
      men_test$HDL
    )
)


# ============================================================
# 38. TEST MATRICES FOR MLR / PR
# ============================================================

women_test_mlr_x <- as.matrix(
  
  women_test_transformed[
    ,
    women_selected_predictors,
    drop = FALSE
  ]
)


men_test_mlr_x <- as.matrix(
  
  men_test_transformed[
    ,
    men_selected_predictors,
    drop = FALSE
  ]
)


women_test_pr_x <- create_polynomial_matrix(
  
  women_test_mlr_x,
  
  degree =
    2
)


men_test_pr_x <- create_polynomial_matrix(
  
  men_test_mlr_x,
  
  degree =
    2
)


# ============================================================
# 39. FINAL WOMEN TEST PREDICTIONS
# ============================================================

women_test_predictions <- data.frame(
  
  ID =
    women_test$ID,
  
  Observed_HCY =
    women_test$HCY,
  
  MLR =
    predict_scaled_linear_model(
      women_mlr_model,
      women_test_mlr_x
    ),
  
  PR =
    predict_scaled_linear_model(
      women_pr_model,
      women_test_pr_x
    ),
  
  `SVR-Linear` =
    as.numeric(
      
      stats::predict(
        
        women_svr_linear,
        
        newdata =
          women_test_transformed
      )
    ),
  
  `SVR-Polynomial` =
    as.numeric(
      
      stats::predict(
        
        women_svr_polynomial,
        
        newdata =
          women_test_transformed
      )
    ),
  
  `SVR-RBF` =
    as.numeric(
      
      stats::predict(
        
        women_svr_rbf,
        
        newdata =
          women_test_transformed
      )
    ),
  
  RF =
    as.numeric(
      
      stats::predict(
        
        women_rf_model,
        
        data =
          as.data.frame(
            
            women_test[
              ,
              all_predictors,
              drop = FALSE
            ]
          )
      )$predictions
    ),
  
  XGB =
    as.numeric(
      
      stats::predict(
        
        women_xgb_model,
        
        newdata =
          as.matrix(
            
            women_test[
              ,
              all_predictors,
              drop = FALSE
            ]
          )
      )
    ),
  
  check.names =
    FALSE
)


# ============================================================
# 40. FINAL MEN TEST PREDICTIONS
# ============================================================

men_test_predictions <- data.frame(
  
  ID =
    men_test$ID,
  
  Observed_HCY =
    men_test$HCY,
  
  MLR =
    predict_scaled_linear_model(
      men_mlr_model,
      men_test_mlr_x
    ),
  
  PR =
    predict_scaled_linear_model(
      men_pr_model,
      men_test_pr_x
    ),
  
  `SVR-Linear` =
    as.numeric(
      
      stats::predict(
        
        men_svr_linear,
        
        newdata =
          men_test_transformed
      )
    ),
  
  `SVR-Polynomial` =
    as.numeric(
      
      stats::predict(
        
        men_svr_polynomial,
        
        newdata =
          men_test_transformed
      )
    ),
  
  `SVR-RBF` =
    as.numeric(
      
      stats::predict(
        
        men_svr_rbf,
        
        newdata =
          men_test_transformed
      )
    ),
  
  RF =
    as.numeric(
      
      stats::predict(
        
        men_rf_model,
        
        data =
          as.data.frame(
            
            men_test[
              ,
              all_predictors,
              drop = FALSE
            ]
          )
      )$predictions
    ),
  
  XGB =
    as.numeric(
      
      stats::predict(
        
        men_xgb_model,
        
        newdata =
          as.matrix(
            
            men_test[
              ,
              all_predictors,
              drop = FALSE
            ]
          )
      )
    ),
  
  check.names =
    FALSE
)


# ============================================================
# 41. VALIDATE FINAL TEST PREDICTIONS
# ============================================================

validate_prediction_table(
  women_test_predictions,
  50,
  "Women TEST"
)


validate_prediction_table(
  men_test_predictions,
  25,
  "Men TEST"
)


# ============================================================
# 42. FINAL TEST METRICS
# ============================================================

test_metrics <- dplyr::bind_rows(
  
  calculate_metric_table(
    women_test_predictions,
    "Women",
    "TEST"
  ),
  
  calculate_metric_table(
    men_test_predictions,
    "Men",
    "TEST"
  )
)


if (
  nrow(
    test_metrics
  ) != 14
) {
  
  stop(
    "Final TEST metric table must contain exactly 14 rows."
  )
}


if (
  any(
    !is.finite(
      test_metrics$RMSE
    )
  ) ||
  any(
    !is.finite(
      test_metrics$MAE
    )
  )
) {
  
  stop(
    "Non-finite TEST error metric detected."
  )
}


# ============================================================
# 43. TRAIN-TEST GAP
# ============================================================

train_test_gap <- train_metrics %>%
  
  dplyr::select(
    
    Sex,
    
    Model,
    
    Train_RMSE =
      RMSE,
    
    Train_MAE =
      MAE,
    
    Train_Predictive_R2 =
      Predictive_R2
  ) %>%
  
  dplyr::left_join(
    
    test_metrics %>%
      
      dplyr::select(
        
        Sex,
        
        Model,
        
        Test_RMSE =
          RMSE,
        
        Test_MAE =
          MAE,
        
        Test_Predictive_R2 =
          Predictive_R2
      ),
    
    by =
      c(
        "Sex",
        "Model"
      )
  ) %>%
  
  dplyr::mutate(
    
    RMSE_Gap =
      Test_RMSE -
      Train_RMSE,
    
    MAE_Gap =
      Test_MAE -
      Train_MAE,
    
    R2_Gap =
      Train_Predictive_R2 -
      Test_Predictive_R2
  )


if (
  nrow(
    train_test_gap
  ) != 14
) {
  
  stop(
    "TRAIN-TEST gap table must contain exactly 14 rows."
  )
}


# ============================================================
# 44. TEST INTEGRITY TABLE
# ============================================================

test_integrity <- data.frame(
  
  Check = c(
    
    "Women TRAIN N",
    
    "Women TEST N",
    
    "Men TRAIN N",
    
    "Men TEST N",
    
    "Women TRAIN-TEST ID overlap",
    
    "Men TRAIN-TEST ID overlap",
    
    "Overall TRAIN-TEST ID overlap",
    
    "Women duplicated TEST IDs",
    
    "Men duplicated TEST IDs",
    
    "Eligible predictors",
    
    "Models evaluated",
    
    "HCY transformed",
    
    "TEST used for tuning",
    
    "TEST used for variable selection",
    
    "TEST used for transformation decisions",
    
    "TEST used for scaling estimation"
  ),
  
  Value = c(
    
    nrow(
      women_raw_train
    ),
    
    nrow(
      women_test
    ),
    
    nrow(
      men_raw_train
    ),
    
    nrow(
      men_test
    ),
    
    length(
      women_id_overlap
    ),
    
    length(
      men_id_overlap
    ),
    
    length(
      overall_id_overlap
    ),
    
    women_test_duplicate_ids,
    
    men_test_duplicate_ids,
    
    length(
      all_predictors
    ),
    
    length(
      model_names
    ),
    
    "No",
    
    "No",
    
    "No",
    
    "No",
    
    "No"
  ),
  
  stringsAsFactors =
    FALSE
)


# ============================================================
# 45. SAVE FINAL OUTPUTS
# ============================================================

readr::write_csv(
  
  women_test_predictions,
  
  file.path(
    output_dir,
    "01_women_final_TEST_predictions.csv"
  )
)


readr::write_csv(
  
  men_test_predictions,
  
  file.path(
    output_dir,
    "02_men_final_TEST_predictions.csv"
  )
)


readr::write_csv(
  
  test_metrics,
  
  file.path(
    output_dir,
    "03_final_TEST_metrics.csv"
  )
)


# IMPORTANT:
# This filename is intentionally lowercase "train" everywhere.

readr::write_csv(
  
  train_metrics,
  
  file.path(
    output_dir,
    "04_final_train_metrics.csv"
  )
)


readr::write_csv(
  
  train_test_gap,
  
  file.path(
    output_dir,
    "05_TRAIN_TEST_gap_summary.csv"
  )
)


readr::write_csv(
  
  test_integrity,
  
  file.path(
    output_dir,
    "06_TEST_integrity_checks.csv"
  )
)


# ============================================================
# 46. TEST EVALUATION LOG
# ============================================================

evaluation_log <- c(
  
  paste(
    "Analysis version:",
    analysis_version
  ),
  
  paste(
    "Workflow freeze timestamp for this execution:",
    format(
      workflow_freeze_time,
      "%Y-%m-%d %H:%M:%S %Z"
    )
  ),
  
  paste(
    "TEST read timestamp for this execution:",
    format(
      test_read_time,
      "%Y-%m-%d %H:%M:%S %Z"
    )
  ),
  
  paste(
    "System timezone:",
    Sys.timezone()
  ),
  
  paste(
    "Women TRAIN file:",
    women_raw_train_file
  ),
  
  paste(
    "Men TRAIN file:",
    men_raw_train_file
  ),
  
  paste(
    "Women TEST file:",
    women_test_file
  ),
  
  paste(
    "Men TEST file:",
    men_test_file
  ),
  
  paste(
    "Women TRAIN N:",
    nrow(
      women_raw_train
    )
  ),
  
  paste(
    "Women TEST N:",
    nrow(
      women_test
    )
  ),
  
  paste(
    "Men TRAIN N:",
    nrow(
      men_raw_train
    )
  ),
  
  paste(
    "Men TEST N:",
    nrow(
      men_test
    )
  ),
  
  paste(
    "Overall TRAIN-TEST ID overlap:",
    length(
      overall_id_overlap
    )
  ),
  
  "All workflow specifications were fixed before TEST evaluation.",
  
  "This script may have been rerun after initial TEST access solely for reproducibility.",
  
  "No model specification was changed as a consequence of TEST results.",
  
  "No hyperparameter tuning was performed using TEST.",
  
  "No variable selection was performed using TEST.",
  
  "No transformation decision was made using TEST.",
  
  "No scaling parameter was estimated from TEST.",
  
  "HCY remained on its original scale.",
  
  "Predictive R2 was calculated as 1 - SSE/SST.",
  
  "Bootstrap confidence intervals are deferred to Script 14."
)


writeLines(
  
  evaluation_log,
  
  con =
    file.path(
      output_dir,
      "07_TEST_evaluation_log.txt"
    )
)


# ============================================================
# 47. SAVE FINAL MODEL BUNDLES
# ============================================================

women_model_bundle <- list(
  
  Analysis_Version =
    analysis_version,
  
  Sex =
    "Women",
  
  Selected_Predictors =
    women_selected_predictors,
  
  Transformations =
    c(
      "BMI=original",
      "VisceralFat=original",
      "MCHC=original",
      "GLU=original",
      "LDL=sqrt"
    ),
  
  MLR =
    women_mlr_model,
  
  PR =
    women_pr_model,
  
  SVR_Linear =
    women_svr_linear,
  
  SVR_Polynomial =
    women_svr_polynomial,
  
  SVR_RBF =
    women_svr_rbf,
  
  RF =
    women_rf_model,
  
  XGB_Model_File =
    women_xgb_model_file,
  
  XGB_Specification =
    xgb_specifications %>%
    
    dplyr::filter(
      Sex ==
        "Women"
    )
)


men_model_bundle <- list(
  
  Analysis_Version =
    analysis_version,
  
  Sex =
    "Men",
  
  Selected_Predictors =
    men_selected_predictors,
  
  Transformations =
    c(
      "BMI=original",
      "HDL=log"
    ),
  
  MLR =
    men_mlr_model,
  
  PR =
    men_pr_model,
  
  SVR_Linear =
    men_svr_linear,
  
  SVR_Polynomial =
    men_svr_polynomial,
  
  SVR_RBF =
    men_svr_rbf,
  
  RF =
    men_rf_model,
  
  XGB_Model_File =
    men_xgb_model_file,
  
  XGB_Specification =
    xgb_specifications %>%
    
    dplyr::filter(
      Sex ==
        "Men"
    )
)


saveRDS(
  
  women_model_bundle,
  
  file.path(
    output_dir,
    "08_women_final_model_bundle.rds"
  )
)


saveRDS(
  
  men_model_bundle,
  
  file.path(
    output_dir,
    "09_men_final_model_bundle.rds"
  )
)


# ============================================================
# 48. SESSION INFORMATION
# ============================================================

writeLines(
  
  capture.output(
    sessionInfo()
  ),
  
  con =
    file.path(
      output_dir,
      "10_session_info.txt"
    )
)


# ============================================================
# 49. VERIFY FINAL OUTPUT FILES
#
# Verification is case-insensitive to avoid Windows filename
# capitalization issues.
# ============================================================

expected_files <- c(
  
  "00_final_workflow_specifications.csv",
  
  "00B_frozen_TRAIN_metric_check.csv",
  
  "01_women_final_TEST_predictions.csv",
  
  "02_men_final_TEST_predictions.csv",
  
  "03_final_TEST_metrics.csv",
  
  "04_final_train_metrics.csv",
  
  "05_TRAIN_TEST_gap_summary.csv",
  
  "06_TEST_integrity_checks.csv",
  
  "07_TEST_evaluation_log.txt",
  
  "08_women_final_model_bundle.rds",
  
  "09_men_final_model_bundle.rds",
  
  "10_session_info.txt"
)


created_files <- list.files(
  output_dir
)


missing_output_files <- expected_files[
  !(tolower(expected_files) %in%
      tolower(created_files))
]


if (
  length(
    missing_output_files
  ) > 0
) {
  
  stop(
    paste(
      "Missing output files:",
      paste(
        missing_output_files,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 50. FINAL NUMERICAL VALIDATION
# ============================================================

if (
  nrow(
    women_test_predictions
  ) != 50
) {
  
  stop(
    "Women final prediction file must contain 50 rows."
  )
}


if (
  nrow(
    men_test_predictions
  ) != 25
) {
  
  stop(
    "Men final prediction file must contain 25 rows."
  )
}


if (
  nrow(
    test_metrics
  ) != 14
) {
  
  stop(
    "Final TEST metric table must contain 14 rows."
  )
}


if (
  nrow(
    train_metrics
  ) != 14
) {
  
  stop(
    "Final TRAIN metric table must contain 14 rows."
  )
}


if (
  nrow(
    train_test_gap
  ) != 14
) {
  
  stop(
    "Final TRAIN-TEST gap table must contain 14 rows."
  )
}


# ============================================================
# 51. DISPLAY FINAL WOMEN TEST RESULTS
# ============================================================

cat("\n\n")
cat("############################################################\n")
cat("# FINAL HELD-OUT TEST RESULTS\n")
cat("############################################################\n")


cat("\n========================================\n")
cat("WOMEN TEST\n")
cat("========================================\n")


women_test_results_display <- test_metrics %>%
  
  dplyr::filter(
    Sex ==
      "Women"
  ) %>%
  
  dplyr::select(
    Model,
    N,
    RMSE,
    MAE,
    Predictive_R2
  ) %>%
  
  dplyr::arrange(
    RMSE
  )


print(
  women_test_results_display
)


# ============================================================
# 52. DISPLAY FINAL MEN TEST RESULTS
# ============================================================

cat("\n========================================\n")
cat("MEN TEST\n")
cat("========================================\n")


men_test_results_display <- test_metrics %>%
  
  dplyr::filter(
    Sex ==
      "Men"
  ) %>%
  
  dplyr::select(
    Model,
    N,
    RMSE,
    MAE,
    Predictive_R2
  ) %>%
  
  dplyr::arrange(
    RMSE
  )


print(
  men_test_results_display
)


# ============================================================
# 53. DISPLAY TRAIN-TEST GAPS
# ============================================================

cat("\n========================================\n")
cat("TRAIN-TEST GAPS\n")
cat("========================================\n")


print(
  train_test_gap
)


# ============================================================
# 54. DISPLAY TEST INTEGRITY
# ============================================================

cat("\n========================================\n")
cat("TEST INTEGRITY CHECKS\n")
cat("========================================\n")


print(
  test_integrity
)


# ============================================================
# 55. FINAL MESSAGE
# ============================================================

cat("\n\n")
cat("############################################################\n")
cat("# SCRIPT 13 COMPLETED SUCCESSFULLY\n")
cat("############################################################\n")


cat(
  "Analysis version:",
  analysis_version,
  "\n"
)


cat(
  "Women TRAIN:",
  nrow(
    women_raw_train
  ),
  "\n"
)


cat(
  "Women TEST:",
  nrow(
    women_test
  ),
  "\n"
)


cat(
  "Men TRAIN:",
  nrow(
    men_raw_train
  ),
  "\n"
)


cat(
  "Men TEST:",
  nrow(
    men_test
  ),
  "\n"
)


cat(
  "Models evaluated:",
  length(
    model_names
  ),
  "\n"
)


cat(
  "Final TEST metric rows:",
  nrow(
    test_metrics
  ),
  "\n"
)


cat(
  "Final TRAIN metric rows:",
  nrow(
    train_metrics
  ),
  "\n"
)


cat(
  "TRAIN-TEST gap rows:",
  nrow(
    train_test_gap
  ),
  "\n"
)


cat(
  "Women TRAIN/TEST ID overlap:",
  length(
    women_id_overlap
  ),
  "\n"
)


cat(
  "Men TRAIN/TEST ID overlap:",
  length(
    men_id_overlap
  ),
  "\n"
)


cat(
  "Overall TRAIN/TEST ID overlap:",
  length(
    overall_id_overlap
  ),
  "\n"
)


cat(
  "HCY scale: ORIGINAL\n"
)


cat(
  "Predictive R2: 1 - SSE/SST\n"
)


cat(
  "TEST used for tuning: NO\n"
)


cat(
  "TEST used for variable selection: NO\n"
)


cat(
  "TEST used for transformation decisions: NO\n"
)


cat(
  "TEST used for scaling estimation: NO\n"
)


cat(
  "Frozen TRAIN workflow check: PASSED\n"
)


cat(
  "Next step: paired bootstrap with 2000 replicates.\n"
)


cat(
  "############################################################\n"
)


cat("\nFINAL OUTPUT FILES:\n")


print(
  list.files(
    output_dir
  )
)