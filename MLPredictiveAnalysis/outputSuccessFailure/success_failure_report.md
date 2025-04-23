# Startup Success/Failure Prediction Report

*Generated on: 2025-04-22 22:12:56*

## Overview

This report presents the results of a machine learning model trained to predict startup success or failure. The model analyzes various startup characteristics and predicts the likelihood of success.

## Dataset Summary

- **Total companies analyzed**: 641
- **Training set size**: 480
- **Test set size**: 161
- **Failure cases**: 0
- **Success cases**: 0

## Model Performance

### Key Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.988 |
| Precision | 0.889 |
| Recall | 1.000 |
| F1 Score | 0.941 |
| AUC-ROC | 1.000 |
| Brier Score | 0.009 |

## Feature Importance

### Top Predictive Factors

| Feature | Importance |
|---------|------------|
| stage_funding_ratio | 0.6915 |
| funding_adequacy_score | 0.2426 |
| industry_momentum | 0.0466 |
| industry_funding_ratio | 0.0104 |
| anomaly_severity | 0.0089 |
| stage_transition_prob | 0.0000 |
| survival_prob_18m | 0.0000 |
| survival_risk | 0.0000 |
| is_emerging_industry | 0.0000 |
| is_saturated_industry | 0.0000 |

### Feature Interpretation

The most important features for predicting startup success/failure include:

- **stage_funding_ratio**: A measure of the development phase of the startup
- **funding_adequacy_score**: A measure of the company's ability to secure financial resources
- **industry_momentum**: A measure of the sustained progress over time
- **industry_funding_ratio**: A measure of the company's ability to secure financial resources
- **anomaly_severity**: A measure of an important startup characteristic

## Visualizations

- [ROC Curve](roc_curve.png)
- [Feature Importance](feature_importance.png)
- [Calibration Curve](calibration_curve.png)
- [Confusion Matrix](confusion_matrix.png)

## Recommendations


## Limitations

This predictive model has the following limitations:

1. Past performance does not guarantee future results
2. The model cannot account for unpredictable market disruptions
3. Success and failure definitions are based on available historical data
4. Industry-specific factors may not be fully captured

## Conclusion

The success/failure prediction model shows reasonably good predictive performance. It can be used as one of several tools to assess startup viability and identify areas for improvement.
