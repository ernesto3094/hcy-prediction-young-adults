SCRIPT 04 - SELECTED PREDICTOR TRANSFORMATIONS

TRAIN data only.

FINAL WOMEN TRANSFORMATIONS:
HDL -> Natural logarithm
PLT -> Natural logarithm
TRG -> Natural logarithm
WBC -> Natural logarithm
LDL -> Square root

All other women predictors remain on their original scale.

FINAL MEN TRANSFORMATIONS:
HDL -> Natural logarithm
TRG -> Natural logarithm
VisceralFat -> Natural logarithm
WBC -> Natural logarithm

All other men predictors remain on their original scale.

HCY remains on its original measurement scale in both sexes.

Transformation decisions were based exclusively on TRAIN data.

The transformation assessment considered:
- descriptive skewness and kurtosis
- raw-data boxplots
- preliminary MLR diagnostics
- alternative transformation statistics
- graphical comparison of transformed distributions
- biological plausibility
- preference for simple monotonic transformations

No transformation was selected solely to force predictor normality.

Yeo-Johnson was evaluated during the assessment stage but was not
retained in the final transformation strategy.

TEST data were not accessed in this script.
No centering or scaling was performed.
No Elastic Net predictor selection was performed.
No final predictive performance was estimated.
