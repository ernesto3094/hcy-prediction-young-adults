# ============================================================
# SCRIPT 02 - PRELIMINARY MLR DIAGNOSTICS
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
#   3. Calculate descriptive statistics using TRAIN data only.
#   4. Generate horizontal raw-data boxplots.
#   5. Fit one preliminary Multiple Linear Regression (MLR)
#      model for women and one for men.
#   6. Evaluate preliminary regression assumptions.
#   7. Save all generated files automatically in one folder.
#
# IMPORTANT:
#
#   - TRAIN data only.
#   - TEST data are NOT accessed.
#   - No transformations are applied.
#   - No centering or scaling is applied.
#   - No Elastic Net is performed.
#   - No observations are removed.
#   - No participant IDs are displayed in figures.
#   - HCY remains on its original measurement scale.
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
      "\nThe following packages are missing:\n\n",
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
# 2. CREATE OUTPUT DIRECTORY
# ------------------------------------------------------------
#
# A dedicated folder is automatically created inside the
# current R working directory.
#
# No computer-specific path is required.
#
# ------------------------------------------------------------

output_dir <- "02_preliminary_MLR_results"


dir.create(
  output_dir,
  showWarnings = FALSE,
  recursive = TRUE
)


# Helper function used throughout the script

output_file <- function(filename) {
  
  file.path(
    output_dir,
    filename
  )
}


cat(
  "\n=============================================\n",
  "OUTPUT DIRECTORY\n",
  "=============================================\n"
)

cat(
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
#
# 01_women_TRAIN.csv
#
# ------------------------------------------------------------

cat(
  "\nSelect file: 01_women_TRAIN.csv\n"
)


women_train <- read_csv(
  file.choose(),
  show_col_types = FALSE
)


# ------------------------------------------------------------
# 4. LOAD MEN TRAIN DATA
# ------------------------------------------------------------
#
# Select:
#
# 03_men_TRAIN.csv
#
# ------------------------------------------------------------

cat(
  "\nSelect file: 03_men_TRAIN.csv\n"
)


men_train <- read_csv(
  file.choose(),
  show_col_types = FALSE
)


# ------------------------------------------------------------
# 5. RENAME VARIABLES TO ENGLISH
# ------------------------------------------------------------
#
# The original dataset contains several Spanish variable names.
# From this point onward, all analytical variables are handled
# using English names consistent with the manuscript.
#
# The function only renames variables that are actually present.
#
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


women_train <- rename_to_english(
  women_train
)


men_train <- rename_to_english(
  men_train
)


# ------------------------------------------------------------
# 6. DEFINE THE 21 CANDIDATE PREDICTORS
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


# HCY is included in the descriptive analysis and boxplots,
# but it is the response variable and NOT a predictor.

descriptive_variables <- c(
  "HCY",
  predictors
)


# ------------------------------------------------------------
# 7. VERIFY REQUIRED ANALYTICAL VARIABLES
# ------------------------------------------------------------

required_analytical_variables <- c(
  "HCY",
  predictors
)


missing_women <- setdiff(
  required_analytical_variables,
  names(women_train)
)


missing_men <- setdiff(
  required_analytical_variables,
  names(men_train)
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
# 8. VERIFY SAMPLE SIZES
# ------------------------------------------------------------

cat(
  "\nWomen TRAIN sample size:",
  nrow(women_train),
  "\n"
)


cat(
  "Men TRAIN sample size:",
  nrow(men_train),
  "\n"
)


# ------------------------------------------------------------
# 9. VERIFY NUMERIC VARIABLES
# ------------------------------------------------------------

non_numeric_women <- descriptive_variables[
  !sapply(
    women_train[descriptive_variables],
    is.numeric
  )
]


non_numeric_men <- descriptive_variables[
  !sapply(
    men_train[descriptive_variables],
    is.numeric
  )
]


if (length(non_numeric_women) > 0) {
  
  stop(
    paste(
      "Non-numeric variables in WOMEN TRAIN:",
      paste(
        non_numeric_women,
        collapse = ", "
      )
    )
  )
}


if (length(non_numeric_men) > 0) {
  
  stop(
    paste(
      "Non-numeric variables in MEN TRAIN:",
      paste(
        non_numeric_men,
        collapse = ", "
      )
    )
  )
}


# ------------------------------------------------------------
# 10. CHECK MISSING VALUES
# ------------------------------------------------------------

women_missing <- sum(
  is.na(
    women_train[
      descriptive_variables
    ]
  )
)


men_missing <- sum(
  is.na(
    men_train[
      descriptive_variables
    ]
  )
)


cat(
  "\nMissing analytical values - Women:",
  women_missing,
  "\n"
)


cat(
  "Missing analytical values - Men:",
  men_missing,
  "\n"
)


if (
  women_missing > 0 ||
  men_missing > 0
) {
  
  stop(
    "Missing values were detected in the analytical variables."
  )
}


# ------------------------------------------------------------
# 11. SKEWNESS FUNCTION
# ------------------------------------------------------------
#
# The same general formulation previously used in the
# Supplementary Material is retained.
#
# A value near zero indicates approximate symmetry.
#
# ------------------------------------------------------------

skewness_function <- function(x) {
  
  x <- x[
    !is.na(x)
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
# 12. KURTOSIS FUNCTION
# ------------------------------------------------------------
#
# Pearson kurtosis is used:
#
# Normal distribution approximately = 3
#
# ------------------------------------------------------------

kurtosis_function <- function(x) {
  
  x <- x[
    !is.na(x)
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
# 13. DESCRIPTIVE STATISTICS FUNCTION
# ------------------------------------------------------------

calculate_descriptive_statistics <- function(
    data,
    sex_name
) {
  
  descriptive_table <- data %>%
    
    select(
      all_of(
        descriptive_variables
      )
    ) %>%
    
    pivot_longer(
      cols = all_of(
        descriptive_variables
      ),
      names_to = "Variable",
      values_to = "Value"
    ) %>%
    
    group_by(
      Variable
    ) %>%
    
    summarise(
      
      N = sum(
        !is.na(Value)
      ),
      
      Mean = mean(
        Value,
        na.rm = TRUE
      ),
      
      Median = median(
        Value,
        na.rm = TRUE
      ),
      
      SD = sd(
        Value,
        na.rm = TRUE
      ),
      
      Q1 = quantile(
        Value,
        0.25,
        na.rm = TRUE
      ),
      
      Q3 = quantile(
        Value,
        0.75,
        na.rm = TRUE
      ),
      
      IQR = IQR(
        Value,
        na.rm = TRUE
      ),
      
      Minimum = min(
        Value,
        na.rm = TRUE
      ),
      
      Maximum = max(
        Value,
        na.rm = TRUE
      ),
      
      Skewness =
        skewness_function(
          Value
        ),
      
      Kurtosis =
        kurtosis_function(
          Value
        ),
      
      .groups = "drop"
    ) %>%
    
    mutate(
      
      Sex = sex_name,
      
      Mean = round(
        Mean,
        3
      ),
      
      Median = round(
        Median,
        3
      ),
      
      SD = round(
        SD,
        3
      ),
      
      Q1 = round(
        Q1,
        3
      ),
      
      Q3 = round(
        Q3,
        3
      ),
      
      IQR = round(
        IQR,
        3
      ),
      
      Minimum = round(
        Minimum,
        3
      ),
      
      Maximum = round(
        Maximum,
        3
      ),
      
      Skewness = round(
        Skewness,
        3
      ),
      
      Kurtosis = round(
        Kurtosis,
        3
      )
    ) %>%
    
    select(
      Sex,
      Variable,
      N,
      Mean,
      Median,
      SD,
      Q1,
      Q3,
      IQR,
      Minimum,
      Maximum,
      Skewness,
      Kurtosis
    )
  
  
  return(
    descriptive_table
  )
}


# ------------------------------------------------------------
# 14. CALCULATE DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

women_descriptive <-
  calculate_descriptive_statistics(
    data = women_train,
    sex_name = "Women"
  )


men_descriptive <-
  calculate_descriptive_statistics(
    data = men_train,
    sex_name = "Men"
  )


# ------------------------------------------------------------
# 15. SAVE DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

write_csv(
  women_descriptive,
  output_file(
    "05_women_descriptive_statistics.csv"
  )
)


write_csv(
  men_descriptive,
  output_file(
    "06_men_descriptive_statistics.csv"
  )
)


# ------------------------------------------------------------
# 16. HORIZONTAL BOXPLOT FUNCTION
# ------------------------------------------------------------
#
# This reproduces the general format previously used in the
# Supplementary Material:
#
#   - One horizontal figure per sex.
#   - All analytical variables in the same figure.
#   - Variables displayed on the vertical axis.
#   - Raw values displayed on the horizontal axis.
#   - Potential outliers highlighted in red.
#   - No participant IDs or observation numbers displayed.
#
# Potential outliers are shown for descriptive purposes only.
# No observations are removed.
#
# ------------------------------------------------------------

create_horizontal_boxplot <- function(
    data,
    sex_name,
    output_filename
) {
  
  boxplot_data <- data %>%
    
    select(
      all_of(
        descriptive_variables
      )
    ) %>%
    
    pivot_longer(
      cols = all_of(
        descriptive_variables
      ),
      names_to = "Variable",
      values_to = "Value"
    ) %>%
    
    filter(
      !is.na(Value)
    )
  
  
  # Preserve the predefined variable order
  
  boxplot_data$Variable <- factor(
    boxplot_data$Variable,
    levels = rev(
      descriptive_variables
    )
  )
  
  
  boxplot_figure <- ggplot(
    boxplot_data,
    aes(
      x = Value,
      y = Variable
    )
  ) +
    
    geom_boxplot(
      
      outlier.colour = "red",
      
      outlier.shape = 8,
      
      outlier.size = 1.6,
      
      outlier.alpha = 0.85,
      
      width = 0.55
    ) +
    
    labs(
      
      title = paste(
        "Boxplots of continuous analytical variables using raw TRAIN data -",
        sex_name
      ),
      
      x = "Raw value",
      
      y = "Variable"
    ) +
    
    theme_minimal(
      base_size = 10
    ) +
    
    theme(
      
      plot.title = element_text(
        hjust = 0.5,
        face = "bold"
      ),
      
      axis.text.y = element_text(
        size = 7
      ),
      
      axis.text.x = element_text(
        size = 8
      ),
      
      panel.grid.major.y =
        element_line(),
      
      panel.grid.minor.y =
        element_blank()
    )
  
  
  print(
    boxplot_figure
  )
  
  
  ggsave(
    
    filename = output_file(
      output_filename
    ),
    
    plot = boxplot_figure,
    
    width = 11,
    
    height = 8.5,
    
    units = "in",
    
    dpi = 300
  )
}


# ------------------------------------------------------------
# 17. GENERATE BOXPLOTS
# ------------------------------------------------------------

create_horizontal_boxplot(
  
  data = women_train,
  
  sex_name = "Women",
  
  output_filename =
    "boxplot_women.png"
)


create_horizontal_boxplot(
  
  data = men_train,
  
  sex_name = "Men",
  
  output_filename =
    "boxplot_men.png"
)


# ------------------------------------------------------------
# 18. BUILD PRELIMINARY MLR FORMULA
# ------------------------------------------------------------
#
# One preliminary MLR is fitted per sex using:
#
# HCY ~ all 21 original candidate predictors
#
# These models are used ONLY for diagnostic purposes.
#
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


cat(
  "\nPreliminary MLR formula:\n"
)


print(
  mlr_formula
)


# ------------------------------------------------------------
# 19. FIT PRELIMINARY MLR - WOMEN
# ------------------------------------------------------------

women_mlr <- lm(
  
  formula = mlr_formula,
  
  data = women_train
)


# ------------------------------------------------------------
# 20. FIT PRELIMINARY MLR - MEN
# ------------------------------------------------------------

men_mlr <- lm(
  
  formula = mlr_formula,
  
  data = men_train
)


# ------------------------------------------------------------
# 21. RUNS TEST FUNCTION
# ------------------------------------------------------------
#
# Wald-Wolfowitz runs test based on the sequence of positive
# and negative residuals.
#
# H0:
# The sequence of residual signs is random.
#
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
  
  
  n_positive <- sum(
    signs == 1
  )
  
  
  n_negative <- sum(
    signs == 0
  )
  
  
  if (
    n_positive == 0 ||
    n_negative == 0
  ) {
    
    return(
      list(
        
        runs = NA_real_,
        
        expected_runs = NA_real_,
        
        z = NA_real_,
        
        p_value = NA_real_
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
      
      runs = number_of_runs,
      
      expected_runs =
        expected_runs,
      
      z = z_value,
      
      p_value = p_value
    )
  )
}


# ------------------------------------------------------------
# 22. PRELIMINARY DIAGNOSTIC FUNCTION
# ------------------------------------------------------------

run_preliminary_diagnostics <- function(
    model,
    sex_name
) {
  
  residual_values <- residuals(
    model
  )
  
  
  n <- length(
    residual_values
  )
  
  
  # ----------------------------------------------------------
  # Residual mean
  # ----------------------------------------------------------
  
  residual_mean <- mean(
    residual_values
  )
  
  
  residual_sd <- sd(
    residual_values
  )
  
  
  # ----------------------------------------------------------
  # Lilliefors normality test
  # ----------------------------------------------------------
  
  lillie_result <-
    nortest::lillie.test(
      residual_values
    )
  
  
  # ----------------------------------------------------------
  # Breusch-Pagan test
  # ----------------------------------------------------------
  
  bp_result <-
    lmtest::bptest(
      model
    )
  
  
  # ----------------------------------------------------------
  # Non-Constant Variance test
  # ----------------------------------------------------------
  
  ncv_result <- tryCatch(
    
    car::ncvTest(
      model
    ),
    
    error = function(e) {
      
      NULL
    }
  )
  
  
  # ----------------------------------------------------------
  # Durbin-Watson test
  # ----------------------------------------------------------
  
  dw_result <- tryCatch(
    
    lmtest::dwtest(
      model
    ),
    
    error = function(e) {
      
      NULL
    }
  )
  
  
  # ----------------------------------------------------------
  # Runs test
  # ----------------------------------------------------------
  
  runs_result <-
    runs_test_residuals(
      residual_values
    )
  
  
  # ----------------------------------------------------------
  # Residual autocorrelation
  # ----------------------------------------------------------
  
  acf_result <- acf(
    
    residual_values,
    
    plot = FALSE,
    
    lag.max = min(
      20,
      n - 1
    )
  )
  
  
  # Remove lag 0 because it is always equal to 1
  
  acf_values <- as.numeric(
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
      na.rm = TRUE
    )
  
  
  number_acf_outside_reference <-
    
    sum(
      abs(
        acf_values
      ) >
        acf_reference_limit,
      na.rm = TRUE
    )
  
  
  # ----------------------------------------------------------
  # Prepare NCV values
  # ----------------------------------------------------------
  
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
    
    ncv_chisquare <- NA_real_
    
    ncv_df <- NA_real_
    
    ncv_p <- NA_real_
  }
  
  
  # ----------------------------------------------------------
  # Prepare Durbin-Watson values
  # ----------------------------------------------------------
  
  if (!is.null(dw_result)) {
    
    dw_statistic <-
      unname(
        dw_result$statistic
      )
    
    
    dw_p <-
      dw_result$p.value
    
  } else {
    
    dw_statistic <- NA_real_
    
    dw_p <- NA_real_
  }
  
  
  # ----------------------------------------------------------
  # Create diagnostic summary
  # ----------------------------------------------------------
  
  diagnostic_summary <- data.frame(
    
    Sex = sex_name,
    
    N = n,
    
    Number_of_predictors =
      length(predictors),
    
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
  
  
  return(
    diagnostic_summary
  )
}


# ------------------------------------------------------------
# 23. RUN PRELIMINARY DIAGNOSTICS
# ------------------------------------------------------------

women_diagnostics <-
  run_preliminary_diagnostics(
    
    model = women_mlr,
    
    sex_name = "Women"
  )


men_diagnostics <-
  run_preliminary_diagnostics(
    
    model = men_mlr,
    
    sex_name = "Men"
  )


# ------------------------------------------------------------
# 24. SAVE PRELIMINARY DIAGNOSTIC RESULTS
# ------------------------------------------------------------

write_csv(
  
  women_diagnostics,
  
  output_file(
    "07_women_preliminary_MLR_diagnostics.csv"
  )
)


write_csv(
  
  men_diagnostics,
  
  output_file(
    "08_men_preliminary_MLR_diagnostics.csv"
  )
)


# ------------------------------------------------------------
# 25. GLOBAL VALIDATION OF LINEAR MODEL ASSUMPTIONS - WOMEN
# ------------------------------------------------------------

women_gvlma <- tryCatch(
  
  gvlma::gvlma(
    women_mlr
  ),
  
  error = function(e) {
    
    e
  }
)


capture.output(
  
  {
    
    cat(
      "WOMEN - PRELIMINARY GVLMA\n"
    )
    
    
    cat(
      "Original TRAIN predictors - before transformation\n\n"
    )
    
    
    if (
      inherits(
        women_gvlma,
        "error"
      )
    ) {
      
      cat(
        "GVLMA could not be calculated.\n\n"
      )
      
      
      cat(
        women_gvlma$message
      )
      
    } else {
      
      print(
        summary(
          women_gvlma
        )
      )
    }
  },
  
  file = output_file(
    "09_women_preliminary_GVLMA.txt"
  )
)


# ------------------------------------------------------------
# 26. GLOBAL VALIDATION OF LINEAR MODEL ASSUMPTIONS - MEN
# ------------------------------------------------------------

men_gvlma <- tryCatch(
  
  gvlma::gvlma(
    men_mlr
  ),
  
  error = function(e) {
    
    e
  }
)


capture.output(
  
  {
    
    cat(
      "MEN - PRELIMINARY GVLMA\n"
    )
    
    
    cat(
      "Original TRAIN predictors - before transformation\n\n"
    )
    
    
    if (
      inherits(
        men_gvlma,
        "error"
      )
    ) {
      
      cat(
        "GVLMA could not be calculated.\n\n"
      )
      
      
      cat(
        men_gvlma$message
      )
      
    } else {
      
      print(
        summary(
          men_gvlma
        )
      )
    }
  },
  
  file = output_file(
    "10_men_preliminary_GVLMA.txt"
  )
)


# ------------------------------------------------------------
# 27. SAVE COMPLETE PRELIMINARY MLR SUMMARIES
# ------------------------------------------------------------
#
# These summaries are saved for reproducibility.
#
# Training R-squared values shown here must NOT be interpreted
# as final predictive performance.
#
# ------------------------------------------------------------

capture.output(
  
  summary(
    women_mlr
  ),
  
  file = output_file(
    "11_women_preliminary_MLR_full_summary.txt"
  )
)


capture.output(
  
  summary(
    men_mlr
  ),
  
  file = output_file(
    "12_men_preliminary_MLR_full_summary.txt"
  )
)


# ------------------------------------------------------------
# 28. RESIDUAL HISTOGRAM FUNCTION
# ------------------------------------------------------------

save_residual_histogram <- function(
    model,
    sex_name,
    filename
) {
  
  residual_data <- data.frame(
    
    Residual = residuals(
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
      
      title = paste(
        "Residual Histogram -",
        sex_name
      ),
      
      x = "Residual",
      
      y = "Frequency"
    ) +
    
    theme_minimal(
      base_size = 11
    ) +
    
    theme(
      
      plot.title = element_text(
        hjust = 0.5,
        face = "bold"
      )
    )
  
  
  print(
    figure
  )
  
  
  ggsave(
    
    filename = output_file(
      filename
    ),
    
    plot = figure,
    
    width = 7,
    
    height = 5,
    
    units = "in",
    
    dpi = 300
  )
}


# ------------------------------------------------------------
# 29. NORMAL Q-Q PLOT FUNCTION
# ------------------------------------------------------------

save_qqplot <- function(
    model,
    sex_name,
    filename
) {
  
  residual_data <- data.frame(
    
    Residual = residuals(
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
      
      title = paste(
        "Normal Q-Q Plot -",
        sex_name
      ),
      
      x = "Theoretical Quantiles",
      
      y = "Sample Quantiles"
    ) +
    
    theme_minimal(
      base_size = 11
    ) +
    
    theme(
      
      plot.title = element_text(
        hjust = 0.5,
        face = "bold"
      )
    )
  
  
  print(
    figure
  )
  
  
  ggsave(
    
    filename = output_file(
      filename
    ),
    
    plot = figure,
    
    width = 7,
    
    height = 5,
    
    units = "in",
    
    dpi = 300
  )
}


# ------------------------------------------------------------
# 30. SCALE-LOCATION PLOT FUNCTION
# ------------------------------------------------------------

save_scale_location <- function(
    model,
    sex_name,
    filename
) {
  
  scale_location_data <- data.frame(
    
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
    
    scale_location_data,
    
    aes(
      
      x = Fitted,
      
      y =
        Sqrt_Abs_Standardized_Residual
    )
    
  ) +
    
    geom_point(
      size = 1.5
    ) +
    
    geom_smooth(
      
      method = "loess",
      
      formula = y ~ x,
      
      se = FALSE
    ) +
    
    labs(
      
      title = paste(
        "Scale-Location Plot -",
        sex_name
      ),
      
      x = "Fitted Values",
      
      y = "Square root of absolute standardized residuals"
    ) +
    
    theme_minimal(
      base_size = 11
    ) +
    
    theme(
      
      plot.title = element_text(
        hjust = 0.5,
        face = "bold"
      )
    )
  
  
  print(
    figure
  )
  
  
  ggsave(
    
    filename = output_file(
      filename
    ),
    
    plot = figure,
    
    width = 7,
    
    height = 5,
    
    units = "in",
    
    dpi = 300
  )
}


# ------------------------------------------------------------
# 31. RESIDUAL ACF PLOT FUNCTION
# ------------------------------------------------------------

save_residual_acf <- function(
    model,
    sex_name,
    filename
) {
  
  png(
    
    filename = output_file(
      filename
    ),
    
    width = 7,
    
    height = 5,
    
    units = "in",
    
    res = 300
  )
  
  
  acf(
    
    residuals(
      model
    ),
    
    main = paste(
      "Residual ACF -",
      sex_name
    )
  )
  
  
  dev.off()
}


# ------------------------------------------------------------
# 32. GENERATE WOMEN DIAGNOSTIC FIGURES
# ------------------------------------------------------------

save_residual_histogram(
  
  model = women_mlr,
  
  sex_name = "Women",
  
  filename =
    "residuals_histogram_women.png"
)


save_qqplot(
  
  model = women_mlr,
  
  sex_name = "Women",
  
  filename =
    "qqplot_women.png"
)


save_scale_location(
  
  model = women_mlr,
  
  sex_name = "Women",
  
  filename =
    "residuals_homoscedasticity_women.png"
)


save_residual_acf(
  
  model = women_mlr,
  
  sex_name = "Women",
  
  filename =
    "residuals_acf_plot_women.png"
)


# ------------------------------------------------------------
# 33. GENERATE MEN DIAGNOSTIC FIGURES
# ------------------------------------------------------------

save_residual_histogram(
  
  model = men_mlr,
  
  sex_name = "Men",
  
  filename =
    "residuals_histogram_men.png"
)


save_qqplot(
  
  model = men_mlr,
  
  sex_name = "Men",
  
  filename =
    "qqplot_men.png"
)


save_scale_location(
  
  model = men_mlr,
  
  sex_name = "Men",
  
  filename =
    "residuals_homoscedasticity_men.png"
)


save_residual_acf(
  
  model = men_mlr,
  
  sex_name = "Men",
  
  filename =
    "residuals_acf_plot_men.png"
)


# ------------------------------------------------------------
# 34. DISPLAY MAIN RESULTS IN THE R CONSOLE
# ------------------------------------------------------------

cat(
  "\n=============================================\n",
  "WOMEN DESCRIPTIVE STATISTICS\n",
  "=============================================\n"
)


print(
  women_descriptive,
  n = Inf
)


cat(
  "\n=============================================\n",
  "MEN DESCRIPTIVE STATISTICS\n",
  "=============================================\n"
)


print(
  men_descriptive,
  n = Inf
)


cat(
  "\n=============================================\n",
  "WOMEN PRELIMINARY MLR DIAGNOSTICS\n",
  "=============================================\n"
)


print(
  women_diagnostics
)


cat(
  "\n=============================================\n",
  "MEN PRELIMINARY MLR DIAGNOSTICS\n",
  "=============================================\n"
)


print(
  men_diagnostics
)


# ------------------------------------------------------------
# 35. FINAL MESSAGE
# ------------------------------------------------------------

cat(
  "\n\n=============================================\n",
  "SCRIPT 02 COMPLETED SUCCESSFULLY\n",
  "=============================================\n",
  "\nWomen TRAIN analyzed.",
  "\nMen TRAIN analyzed.",
  "\nVariables handled in English.",
  "\nDescriptive statistics generated.",
  "\nHorizontal boxplots generated.",
  "\nPotential outliers displayed but NOT removed.",
  "\nPreliminary MLR models fitted.",
  "\nLilliefors tests completed.",
  "\nBreusch-Pagan tests completed.",
  "\nNCV tests completed.",
  "\nDurbin-Watson tests completed.",
  "\nRuns tests completed.",
  "\nResidual ACF diagnostics completed.",
  "\nGVLMA diagnostics completed.",
  "\nDiagnostic figures saved as 300-dpi PNG files.",
  "\nNo transformations were applied.",
  "\nNo scaling was applied.",
  "\nNo Elastic Net was performed.",
  "\nTEST data were NOT accessed.",
  "\n\nAll results were saved in:\n",
  normalizePath(
    output_dir,
    winslash = "/",
    mustWork = FALSE
  ),
  "\n=============================================\n"
)
