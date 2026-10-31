# ============================================================
# SCRIPT 03 - PREDICTOR TRANSFORMATION ASSESSMENT
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
#   3. Evaluate selected candidate predictors for possible
#      transformation.
#   4. Compare the following representations:
#
#         - Original
#         - Natural logarithm
#         - Square root
#         - Cube root
#         - Yeo-Johnson
#
#   5. Calculate descriptive distributional statistics for
#      every candidate representation.
#   6. Generate graphical comparisons for each predictor.
#   7. Save women and men results in separate folders.
#
# IMPORTANT:
#
#   - TRAIN data only.
#   - TEST data are NOT accessed.
#   - HCY is NOT transformed.
#   - Age is NOT included in the transformation assessment.
#   - No transformation is selected automatically.
#   - No final transformed dataset is created in this script.
#   - No centering or scaling is performed.
#   - No Elastic Net is performed.
#   - No observations are removed.
#
# The purpose of this script is DIAGNOSTIC ASSESSMENT ONLY.
#
# ============================================================


# ------------------------------------------------------------
# 1. REQUIRED PACKAGES
# ------------------------------------------------------------

required_packages <- c(
  "dplyr",
  "tidyr",
  "readr",
  "ggplot2"
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


# ------------------------------------------------------------
# 2. CREATE OUTPUT FOLDERS
# ------------------------------------------------------------
#
# The script automatically creates one main results folder
# and separate subfolders for women and men.
#
# ------------------------------------------------------------

output_dir <- "03_transformation_assessment_results"

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

women_train <- read_csv(
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

men_train <- read_csv(
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


women_train <- rename_to_english(
  women_train
)

men_train <- rename_to_english(
  men_train
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


# ------------------------------------------------------------
# 7. DEFINE CANDIDATE PREDICTORS FOR TRANSFORMATION REVIEW
# ------------------------------------------------------------
#
# These lists define predictors requiring closer examination.
#
# Inclusion in these lists DOES NOT mean that a transformation
# will necessarily be applied.
#
# Age is deliberately excluded because its strong skewness is
# primarily associated with the narrow age structure of the
# study population.
#
# HCY is not included because it is the response variable and
# will remain on its original scale.
#
# ------------------------------------------------------------


women_candidates <- c(
  
  "Weight",
  "BMI",
  "MuscleMass",
  
  "MCV",
  "MCH",
  "RBC",
  "PLT",
  "WBC",
  
  "GLU",
  "TRG",
  "HDL",
  "LDL",
  
  "RDW_CV"
)


men_candidates <- c(
  
  "Height",
  "Weight",
  "BMI",
  "TotalFat",
  "VisceralFat",
  "MuscleMass",
  
  "Hgb",
  "Hct",
  "RBC",
  "WBC",
  
  "GLU",
  "TRG",
  "HDL"
)


# ------------------------------------------------------------
# 8. VERIFY CANDIDATE VARIABLES
# ------------------------------------------------------------

missing_women <- setdiff(
  women_candidates,
  names(women_train)
)

missing_men <- setdiff(
  men_candidates,
  names(men_train)
)


if (length(missing_women) > 0) {
  
  stop(
    paste(
      "Missing WOMEN candidate variables:",
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
      "Missing MEN candidate variables:",
      paste(
        missing_men,
        collapse = ", "
      )
    )
  )
}


# ------------------------------------------------------------
# 9. SKEWNESS FUNCTION
# ------------------------------------------------------------
#
# This is the same general skewness formulation used in
# Script 02 to maintain consistency across analyses.
#
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
# 10. KURTOSIS FUNCTION
# ------------------------------------------------------------
#
# Pearson kurtosis is used:
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
# 11. YEO-JOHNSON TRANSFORMATION
# ------------------------------------------------------------
#
# Yeo-Johnson transformation:
#
# For x >= 0:
#
#   lambda != 0:
#   ((x + 1)^lambda - 1) / lambda
#
#   lambda = 0:
#   log(x + 1)
#
#
# For x < 0:
#
#   lambda != 2:
#   -[((1 - x)^(2 - lambda) - 1) / (2 - lambda)]
#
#   lambda = 2:
#   -log(1 - x)
#
# ------------------------------------------------------------

yeo_johnson_transform <- function(
    x,
    lambda
) {
  
  result <- numeric(
    length(x)
  )
  
  
  positive <- x >= 0
  negative <- x < 0
  
  
  # Non-negative values
  
  if (
    abs(lambda) < 1e-8
  ) {
    
    result[positive] <-
      log1p(
        x[positive]
      )
    
  } else {
    
    result[positive] <-
      
      (
        (x[positive] + 1)^lambda -
          1
      ) /
      lambda
  }
  
  
  # Negative values
  
  if (
    abs(lambda - 2) < 1e-8
  ) {
    
    result[negative] <-
      
      -log1p(
        -x[negative]
      )
    
  } else {
    
    result[negative] <-
      
      -(
        (
          (1 - x[negative])^(2 - lambda) -
            1
        ) /
          (2 - lambda)
      )
  }
  
  
  return(
    result
  )
}


# ------------------------------------------------------------
# 12. ESTIMATE YEO-JOHNSON LAMBDA
# ------------------------------------------------------------
#
# Lambda is estimated within TRAIN only using a profile
# likelihood approach.
#
# The search is restricted to the interval [-2, 2].
#
# ------------------------------------------------------------

estimate_yj_lambda <- function(x) {
  
  x <- x[
    is.finite(x)
  ]
  
  
  objective_function <- function(lambda) {
    
    transformed <-
      yeo_johnson_transform(
        x,
        lambda
      )
    
    
    transformed_variance <-
      
      mean(
        (
          transformed -
            mean(transformed)
        )^2
      )
    
    
    if (
      !is.finite(transformed_variance) ||
      transformed_variance <= 0
    ) {
      
      return(
        Inf
      )
    }
    
    
    log_jacobian <- ifelse(
      
      x >= 0,
      
      (lambda - 1) *
        log1p(x),
      
      (1 - lambda) *
        log1p(-x)
    )
    
    
    log_likelihood <-
      
      -length(x) / 2 *
      log(
        transformed_variance
      ) +
      
      sum(
        log_jacobian
      )
    
    
    return(
      -log_likelihood
    )
  }
  
  
  optimization <- optimize(
    
    f = objective_function,
    
    interval = c(
      -2,
      2
    )
  )
  
  
  return(
    optimization$minimum
  )
}


# ------------------------------------------------------------
# 13. APPLY ONE TRANSFORMATION
# ------------------------------------------------------------

apply_transformation <- function(
    x,
    transformation,
    yj_lambda = NA_real_
) {
  
  if (
    transformation == "Original"
  ) {
    
    return(
      x
    )
  }
  
  
  if (
    transformation == "Log"
  ) {
    
    if (
      any(
        x <= 0,
        na.rm = TRUE
      )
    ) {
      
      return(
        rep(
          NA_real_,
          length(x)
        )
      )
    }
    
    return(
      log(x)
    )
  }
  
  
  if (
    transformation == "SquareRoot"
  ) {
    
    if (
      any(
        x < 0,
        na.rm = TRUE
      )
    ) {
      
      return(
        rep(
          NA_real_,
          length(x)
        )
      )
    }
    
    return(
      sqrt(x)
    )
  }
  
  
  if (
    transformation == "CubeRoot"
  ) {
    
    return(
      
      sign(x) *
        abs(x)^(1 / 3)
    )
  }
  
  
  if (
    transformation == "YeoJohnson"
  ) {
    
    return(
      yeo_johnson_transform(
        x,
        yj_lambda
      )
    )
  }
  
  
  stop(
    paste(
      "Unknown transformation:",
      transformation
    )
  )
}


# ------------------------------------------------------------
# 14. TRANSFORMATIONS TO BE COMPARED
# ------------------------------------------------------------

transformations <- c(
  
  "Original",
  "Log",
  "SquareRoot",
  "CubeRoot",
  "YeoJohnson"
)


# ------------------------------------------------------------
# 15. ASSESS ONE PREDICTOR
# ------------------------------------------------------------

assess_predictor <- function(
    data,
    variable,
    sex_name
) {
  
  original_values <-
    data[[variable]]
  
  
  yj_lambda <-
    estimate_yj_lambda(
      original_values
    )
  
  
  result_list <- list()
  
  
  for (
    transformation in transformations
  ) {
    
    transformed_values <-
      apply_transformation(
        
        x = original_values,
        
        transformation =
          transformation,
        
        yj_lambda =
          yj_lambda
      )
    
    
    valid_values <-
      transformed_values[
        is.finite(
          transformed_values
        )
      ]
    
    
    # Skip invalid transformations
    
    if (
      length(valid_values) == 0
    ) {
      
      result_list[[
        transformation
      ]] <- data.frame(
        
        Sex = sex_name,
        
        Variable = variable,
        
        Transformation =
          transformation,
        
        YeoJohnson_Lambda =
          ifelse(
            transformation ==
              "YeoJohnson",
            yj_lambda,
            NA_real_
          ),
        
        N = 0,
        
        Mean = NA_real_,
        
        SD = NA_real_,
        
        Minimum = NA_real_,
        
        Maximum = NA_real_,
        
        Skewness = NA_real_,
        
        Absolute_Skewness =
          NA_real_,
        
        Kurtosis = NA_real_,
        
        Kurtosis_Deviation_From_3 =
          NA_real_,
        
        Spearman_With_Original =
          NA_real_
      )
      
      
      next
    }
    
    
    correlation <-
      
      suppressWarnings(
        cor(
          
          original_values,
          
          transformed_values,
          
          method = "spearman",
          
          use = "complete.obs"
        )
      )
    
    
    skew_value <-
      skewness_function(
        valid_values
      )
    
    
    kurtosis_value <-
      kurtosis_function(
        valid_values
      )
    
    
    result_list[[
      transformation
    ]] <- data.frame(
      
      Sex = sex_name,
      
      Variable = variable,
      
      Transformation =
        transformation,
      
      YeoJohnson_Lambda =
        ifelse(
          
          transformation ==
            "YeoJohnson",
          
          yj_lambda,
          
          NA_real_
        ),
      
      N =
        length(
          valid_values
        ),
      
      Mean =
        mean(
          valid_values
        ),
      
      SD =
        sd(
          valid_values
        ),
      
      Minimum =
        min(
          valid_values
        ),
      
      Maximum =
        max(
          valid_values
        ),
      
      Skewness =
        skew_value,
      
      Absolute_Skewness =
        abs(
          skew_value
        ),
      
      Kurtosis =
        kurtosis_value,
      
      Kurtosis_Deviation_From_3 =
        abs(
          kurtosis_value -
            3
        ),
      
      Spearman_With_Original =
        correlation
    )
  }
  
  
  bind_rows(
    result_list
  )
}


# ------------------------------------------------------------
# 16. CREATE TRANSFORMATION COMPARISON FIGURE
# ------------------------------------------------------------
#
# Each candidate predictor receives one PNG file showing the
# distribution under all five representations.
#
# Free x-axis scales are used because each transformation
# changes the numerical measurement scale.
#
# The figures are intended for visual assessment and not for
# automatic transformation selection.
#
# ------------------------------------------------------------

create_transformation_figure <- function(
    data,
    variable,
    sex_name,
    figure_directory
) {
  
  original_values <-
    data[[variable]]
  
  
  yj_lambda <-
    estimate_yj_lambda(
      original_values
    )
  
  
  plot_data_list <- list()
  
  
  for (
    transformation in transformations
  ) {
    
    transformed_values <-
      apply_transformation(
        
        x = original_values,
        
        transformation =
          transformation,
        
        yj_lambda =
          yj_lambda
      )
    
    
    plot_data_list[[
      transformation
    ]] <- data.frame(
      
      Transformation =
        transformation,
      
      Value =
        transformed_values
    )
  }
  
  
  plot_data <-
    bind_rows(
      plot_data_list
    ) %>%
    
    filter(
      is.finite(Value)
    )
  
  
  plot_data$Transformation <-
    factor(
      
      plot_data$Transformation,
      
      levels =
        transformations
    )
  
  
  figure <- ggplot(
    
    plot_data,
    
    aes(
      x = Value
    )
    
  ) +
    
    geom_histogram(
      bins = 20
    ) +
    
    facet_wrap(
      
      ~ Transformation,
      
      scales = "free_x",
      
      ncol = 2
    ) +
    
    labs(
      
      title = paste(
        variable,
        "- Transformation Assessment -",
        sex_name
      ),
      
      subtitle = paste0(
        "Yeo-Johnson lambda = ",
        round(
          yj_lambda,
          3
        )
      ),
      
      x =
        "Value",
      
      y =
        "Frequency"
    ) +
    
    theme_minimal(
      base_size = 11
    ) +
    
    theme(
      
      plot.title =
        element_text(
          hjust = 0.5,
          face = "bold"
        ),
      
      plot.subtitle =
        element_text(
          hjust = 0.5
        ),
      
      strip.text =
        element_text(
          face = "bold"
        )
    )
  
  
  safe_variable_name <-
    
    gsub(
      
      "[^A-Za-z0-9_]+",
      
      "_",
      
      variable
    )
  
  
  figure_file <-
    
    file.path(
      
      figure_directory,
      
      paste0(
        safe_variable_name,
        "_transformation_comparison.png"
      )
    )
  
  
  ggsave(
    
    filename =
      figure_file,
    
    plot =
      figure,
    
    width =
      10,
    
    height =
      8,
    
    units =
      "in",
    
    dpi =
      300
  )
}


# ------------------------------------------------------------
# 17. ASSESS ALL WOMEN CANDIDATE PREDICTORS
# ------------------------------------------------------------

women_results_list <- list()


for (
  variable in women_candidates
) {
  
  cat(
    "\nWomen - assessing:",
    variable,
    "\n"
  )
  
  
  women_results_list[[
    variable
  ]] <-
    
    assess_predictor(
      
      data =
        women_train,
      
      variable =
        variable,
      
      sex_name =
        "Women"
    )
  
  
  create_transformation_figure(
    
    data =
      women_train,
    
    variable =
      variable,
    
    sex_name =
      "Women",
    
    figure_directory =
      women_figures_dir
  )
}


women_transformation_comparison <-
  
  bind_rows(
    women_results_list
  )


# ------------------------------------------------------------
# 18. ASSESS ALL MEN CANDIDATE PREDICTORS
# ------------------------------------------------------------

men_results_list <- list()


for (
  variable in men_candidates
) {
  
  cat(
    "\nMen - assessing:",
    variable,
    "\n"
  )
  
  
  men_results_list[[
    variable
  ]] <-
    
    assess_predictor(
      
      data =
        men_train,
      
      variable =
        variable,
      
      sex_name =
        "Men"
    )
  
  
  create_transformation_figure(
    
    data =
      men_train,
    
    variable =
      variable,
    
    sex_name =
      "Men",
    
    figure_directory =
      men_figures_dir
  )
}


men_transformation_comparison <-
  
  bind_rows(
    men_results_list
  )


# ------------------------------------------------------------
# 19. ROUND EXPORTED NUMERIC RESULTS
# ------------------------------------------------------------

round_comparison_table <- function(data) {
  
  data %>%
    
    mutate(
      
      YeoJohnson_Lambda =
        round(
          YeoJohnson_Lambda,
          4
        ),
      
      Mean =
        round(
          Mean,
          4
        ),
      
      SD =
        round(
          SD,
          4
        ),
      
      Minimum =
        round(
          Minimum,
          4
        ),
      
      Maximum =
        round(
          Maximum,
          4
        ),
      
      Skewness =
        round(
          Skewness,
          4
        ),
      
      Absolute_Skewness =
        round(
          Absolute_Skewness,
          4
        ),
      
      Kurtosis =
        round(
          Kurtosis,
          4
        ),
      
      Kurtosis_Deviation_From_3 =
        round(
          Kurtosis_Deviation_From_3,
          4
        ),
      
      Spearman_With_Original =
        round(
          Spearman_With_Original,
          4
        )
    )
}


women_transformation_comparison <-
  round_comparison_table(
    women_transformation_comparison
  )


men_transformation_comparison <-
  round_comparison_table(
    men_transformation_comparison
  )


# ------------------------------------------------------------
# 20. SAVE WOMEN COMPARISON TABLE
# ------------------------------------------------------------

write_csv(
  
  women_transformation_comparison,
  
  file.path(
    women_dir,
    "01_women_transformation_comparison.csv"
  )
)


# ------------------------------------------------------------
# 21. SAVE MEN COMPARISON TABLE
# ------------------------------------------------------------

write_csv(
  
  men_transformation_comparison,
  
  file.path(
    men_dir,
    "01_men_transformation_comparison.csv"
  )
)


# ------------------------------------------------------------
# 22. SAVE CANDIDATE LISTS
# ------------------------------------------------------------

write_csv(
  
  data.frame(
    Candidate_Predictor =
      women_candidates
  ),
  
  file.path(
    women_dir,
    "02_women_candidate_predictors.csv"
  )
)


write_csv(
  
  data.frame(
    Candidate_Predictor =
      men_candidates
  ),
  
  file.path(
    men_dir,
    "02_men_candidate_predictors.csv"
  )
)


# ------------------------------------------------------------
# 23. SAVE ORIGINAL DISTRIBUTIONAL SUMMARY
# ------------------------------------------------------------
#
# This table facilitates comparison of the raw predictor
# distributions before reviewing alternative transformations.
#
# ------------------------------------------------------------

create_original_summary <- function(
    data,
    candidates,
    sex_name
) {
  
  result_list <- list()
  
  
  for (
    variable in candidates
  ) {
    
    x <-
      data[[variable]]
    
    
    result_list[[
      variable
    ]] <- data.frame(
      
      Sex =
        sex_name,
      
      Variable =
        variable,
      
      N =
        sum(
          is.finite(x)
        ),
      
      Skewness =
        skewness_function(x),
      
      Kurtosis =
        kurtosis_function(x)
    )
  }
  
  
  bind_rows(
    result_list
  ) %>%
    
    mutate(
      
      Skewness =
        round(
          Skewness,
          4
        ),
      
      Kurtosis =
        round(
          Kurtosis,
          4
        )
    )
}


women_original_summary <-
  
  create_original_summary(
    
    data =
      women_train,
    
    candidates =
      women_candidates,
    
    sex_name =
      "Women"
  )


men_original_summary <-
  
  create_original_summary(
    
    data =
      men_train,
    
    candidates =
      men_candidates,
    
    sex_name =
      "Men"
  )


write_csv(
  
  women_original_summary,
  
  file.path(
    women_dir,
    "03_women_original_distribution_summary.csv"
  )
)


write_csv(
  
  men_original_summary,
  
  file.path(
    men_dir,
    "03_men_original_distribution_summary.csv"
  )
)


# ------------------------------------------------------------
# 24. CREATE ANALYSIS NOTE
# ------------------------------------------------------------

analysis_note <- c(
  
  "SCRIPT 03 - TRANSFORMATION ASSESSMENT",
  
  "",
  
  "This analysis was performed using TRAIN data only.",
  
  "",
  
  "The following representations were compared:",
  
  "Original",
  "Natural logarithm",
  "Square root",
  "Cube root",
  "Yeo-Johnson",
  
  "",
  
  "IMPORTANT:",
  
  "No transformation was automatically selected.",
  
  "The comparison tables and figures are diagnostic outputs only.",
  
  "Final transformation decisions must be made after joint review of:",
  "- original skewness and kurtosis",
  "- transformation behavior",
  "- graphical distributional changes",
  "- biological plausibility",
  "- preservation of useful information",
  
  "",
  
  "HCY was not transformed.",
  
  "Age was not included in the transformation assessment.",
  
  "TEST data were not accessed.",
  
  "No transformed final dataset was created.",
  
  "No centering or scaling was performed.",
  
  "No Elastic Net analysis was performed."
)


writeLines(
  
  analysis_note,
  
  con = file.path(
    output_dir,
    "00_READ_ME_FIRST.txt"
  )
)


# ------------------------------------------------------------
# 25. DISPLAY RESULTS IN THE CONSOLE
# ------------------------------------------------------------

cat(
  "\n\n=============================================\n",
  "WOMEN TRANSFORMATION COMPARISON\n",
  "=============================================\n"
)


print(
  as_tibble(
    women_transformation_comparison
  ),
  n = Inf
)


cat(
  "\n\n=============================================\n",
  "MEN TRANSFORMATION COMPARISON\n",
  "=============================================\n"
)


print(
  as_tibble(
    men_transformation_comparison
  ),
  n = Inf
)


# ------------------------------------------------------------
# 26. FINAL MESSAGE
# ------------------------------------------------------------

cat(
  "\n\n=============================================\n",
  "SCRIPT 03 COMPLETED SUCCESSFULLY\n",
  "=============================================\n",
  "\nWomen candidate predictors assessed:",
  length(women_candidates),
  "\nMen candidate predictors assessed:",
  length(men_candidates),
  "\n",
  "\nOriginal, Log, SquareRoot, CubeRoot, and",
  "\nYeoJohnson representations were compared.",
  "\n",
  "\nNo transformation was selected automatically.",
  "\nHCY was NOT transformed.",
  "\nAge was NOT transformed.",
  "\nTEST data were NOT accessed.",
  "\nNo final transformed dataset was created.",
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
