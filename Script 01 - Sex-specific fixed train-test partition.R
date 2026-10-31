# ============================================================
# Script 01 - Sex-specific fixed train-test partition
# Study: Sex-stratified modeling of homocysteine (HCY)
#
# Objective:
#   1. Load the final corrected dataset.
#   2. Retain the 21 predefined candidate predictors.
#   3. Split the dataset by sex.
#   4. Create HCY-based quintile strata.
#   5. Generate a single reproducible 80/20 train-test split
#      using random seed 123.
#   6. Keep duplicated participant IDs in the same partition
#      to prevent potential information leakage.
#   7. Export the final TRAIN and TEST datasets.
#
# IMPORTANT:
#   The TEST sets generated in this script must remain untouched
#   until the final evaluation of the fully specified models.
# ============================================================


# ------------------------------------------------------------
# 1. Load required packages
# ------------------------------------------------------------

library(dplyr)
library(readr)


# ------------------------------------------------------------
# 2. Load the final corrected dataset
# ------------------------------------------------------------
# Select:
# Alumnos_2019_Ambos_generos_Rev_HCY_MCH_3_correcciones.csv

data <- read_csv(
  file.choose(),
  show_col_types = FALSE
)


# ------------------------------------------------------------
# 3. Define the 21 candidate predictors
# ------------------------------------------------------------

predictors <- c(
  "EDAD",
  "TALLA",
  "PESO",
  "IMC",
  "GRASAT",
  "GRASAVIS",
  "MUSCULO",
  "HGB",
  "HTO",
  "VCM",
  "MCH",
  "MCHC",
  "RDWCV",
  "RBC",
  "PLT",
  "WBC",
  "GLU",
  "COL",
  "TRG",
  "HDL",
  "LDL"
)

# Confirm that exactly 21 predictors were defined
stopifnot(length(predictors) == 21)


# ------------------------------------------------------------
# 4. Check that all required variables are available
# ------------------------------------------------------------

required_variables <- c(
  "ID",
  "GENERO",
  "HCY",
  predictors
)

missing_variables <- setdiff(
  required_variables,
  names(data)
)

if (length(missing_variables) > 0) {
  
  stop(
    paste(
      "The following required variables are missing:",
      paste(missing_variables, collapse = ", ")
    )
  )
}


# ------------------------------------------------------------
# 5. Create the analytical dataset
# ------------------------------------------------------------
# ID is retained only for participant tracking and partition
# control. It will NOT be used as a predictor.
#
# GENERO is used only to create the sex-specific subsets.
#
# HCY is the continuous response variable.

model_data <- data %>%
  select(
    ID,
    GENERO,
    HCY,
    all_of(predictors)
  )


# ------------------------------------------------------------
# 6. Basic integrity checks
# ------------------------------------------------------------

cat("\nNumber of observations:", nrow(model_data), "\n")

cat("\nObservations by sex:\n")
print(
  model_data %>%
    count(GENERO)
)

cat("\nMissing HCY values:\n")
print(sum(is.na(model_data$HCY)))

cat("\nMissing values among the 21 candidate predictors:\n")
print(
  model_data %>%
    summarise(
      across(
        all_of(predictors),
        ~ sum(is.na(.x))
      )
    )
)


# ------------------------------------------------------------
# 7. Function for sex-specific train-test partitioning
# ------------------------------------------------------------
# HCY is divided into five approximately equal-frequency
# strata (quintiles).
#
# Participant ID is used as the grouping unit so that records
# sharing the same ID cannot appear in both TRAIN and TEST.
#
# For an ID appearing more than once, the median HCY value is
# used only to assign that ID to an HCY stratum.
#
# The HCY stratum is used exclusively for partitioning and is
# not included as a model predictor.

create_split <- function(df, seed = 123) {
  
  id_level_data <- df %>%
    group_by(ID) %>%
    summarise(
      HCY_group = median(HCY, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      HCY_stratum = ntile(HCY_group, 5)
    )
  
  
  # Fixed random seed for reproducibility
  set.seed(seed)
  
  
  # Select approximately 80% of participant IDs from each
  # HCY quintile for the training set
  train_ids <- id_level_data %>%
    group_by(HCY_stratum) %>%
    slice_sample(prop = 0.80) %>%
    ungroup() %>%
    pull(ID)
  
  
  # Assign every observation to TRAIN or TEST
  df %>%
    mutate(
      PARTITION = if_else(
        ID %in% train_ids,
        "TRAIN",
        "TEST"
      )
    )
}


# ------------------------------------------------------------
# 8. Create female and male analytical subsets
# ------------------------------------------------------------
# Coding used in the original dataset:
#   1 = women
#   2 = men

women <- model_data %>%
  filter(GENERO == 1)

men <- model_data %>%
  filter(GENERO == 2)


# ------------------------------------------------------------
# 9. Generate the fixed 80/20 split using seed 123
# ------------------------------------------------------------

women_split <- create_split(
  women,
  seed = 123
)

men_split <- create_split(
  men,
  seed = 123
)


# ------------------------------------------------------------
# 10. Combine partitions for verification
# ------------------------------------------------------------

final_partition <- bind_rows(
  women_split,
  men_split
)


# ------------------------------------------------------------
# 11. Display final sample sizes
# ------------------------------------------------------------

cat("\nFinal sample sizes by sex and partition:\n")

partition_counts <- final_partition %>%
  count(
    GENERO,
    PARTITION
  )

print(partition_counts)


# ------------------------------------------------------------
# 12. Verify that no participant ID appears in both
#     TRAIN and TEST
# ------------------------------------------------------------

id_overlap <- final_partition %>%
  distinct(
    ID,
    PARTITION
  ) %>%
  count(ID) %>%
  filter(n > 1)

cat("\nParticipant IDs appearing in both TRAIN and TEST:\n")
print(id_overlap)

if (nrow(id_overlap) == 0) {
  cat("\nPASS: No participant ID is shared between TRAIN and TEST.\n")
} else {
  warning(
    "At least one participant ID appears in both TRAIN and TEST."
  )
}


# ------------------------------------------------------------
# 13. Create the four final datasets
# ------------------------------------------------------------

women_train <- women_split %>%
  filter(PARTITION == "TRAIN") %>%
  select(-PARTITION)

women_test <- women_split %>%
  filter(PARTITION == "TEST") %>%
  select(-PARTITION)

men_train <- men_split %>%
  filter(PARTITION == "TRAIN") %>%
  select(-PARTITION)

men_test <- men_split %>%
  filter(PARTITION == "TEST") %>%
  select(-PARTITION)


# ------------------------------------------------------------
# 14. Final verification
# ------------------------------------------------------------

cat("\nWomen TRAIN:", nrow(women_train), "\n")
cat("Women TEST :", nrow(women_test), "\n")
cat("Men TRAIN  :", nrow(men_train), "\n")
cat("Men TEST   :", nrow(men_test), "\n")


# ------------------------------------------------------------
# 15. Export the datasets
# ------------------------------------------------------------

write_csv(
  women_train,
  "01_women_TRAIN.csv"
)

write_csv(
  women_test,
  "02_women_TEST.csv"
)

write_csv(
  men_train,
  "03_men_TRAIN.csv"
)

write_csv(
  men_test,
  "04_men_TEST.csv"
)

write_csv(
  final_partition,
  "00_final_partition_seed123.csv"
)


# ------------------------------------------------------------
# 16. End of Script 01
# ------------------------------------------------------------

cat(
  "\nScript 01 completed successfully.\n",
  "The TEST datasets must remain untouched until final model evaluation.\n"
)
