# Funding Amount Forecast Analysis Report
Generated on: 2025-04-22 16:51:05

## Dataset Overview
- Total records: 1727
- Companies analyzed: 623
- Date range: 2025-02-01 to 2025-03-01
- Industries: 46
- Funding stages: nan, Series B, Seed, Series A, Series C, Venture - Series Unknown, Series E, Pre-Seed, Private Equity, Series G, Initial Coin Offering

## Funding Amount Statistics
- Average funding amount: $222,646,733.22
- Median funding amount: $17,300,000.00
- Maximum funding amount: $50,000,000,000.00
- Minimum funding amount: $0.00
- Standard deviation: $2,058,447,517.95

## Training and Testing Summary
- Training set size: 1395
- Testing set size: 332
- Training/Testing split ratio: 1395:332
- Actual next round data: 1104
- Estimated next round data: 623

## Model Performance Metrics

### Quantile Regression Model
- Model Type: quantile_regression_forest
- RMSE: $112,048,232.46
- MAE: $38,490,322.86
- R2: -0.0076
- Log RMSE: 5.5873
- Log MAE: 3.0589
- Log R2: -0.1653
- Within 10%: 4.0%
- Within 20%: 7.7%
- Within 50%: 21.8%
- Within Same Order: 77.5%
- Interval Coverage 80%: 66.3%

### Ensemble Model
- Model Type: ensemble
- RMSE: $284,090,903.09
- MAE: $76,145,887.66
- R2: -5.4771
- Log RMSE: 5.9961
- Log MAE: 3.3471
- Log R2: -0.3421
- Within 10%: 4.4%
- Within 20%: 8.7%
- Within 50%: 20.1%
- Within Same Order: 71.5%

#### Individual Model Performance
**Random Forest**
- RMSE: $269,763,588.02
- MAE: $62,034,274.86
- R2: -4.8402
- Within 20%: 10.1%

**Gradient Boosting**
- RMSE: $690,233,981.52
- MAE: $109,173,260.59
- R2: -37.2346
- Within 20%: 9.1%

**Extra Trees**
- RMSE: $138,310,342.82
- MAE: $62,284,567.86
- R2: -0.5352
- Within 20%: 6.7%

## Model Calibration Analysis
The model calibration analysis measures how well the predicted values match the actual values. A well-calibrated model will have residuals (actual minus predicted) centered around zero with no clear patterns.

- Mean residual (QRF): $13,108,851.47
- Mean absolute residual (QRF): $38,490,322.86
- Residual standard deviation (QRF): $111,278,768.91
- Calibration slope (QRF): -0.3657

- Mean residual (Ensemble): $954,016.06
- Mean absolute residual (Ensemble): $76,145,887.66
- Residual standard deviation (Ensemble): $284,089,301.23
- Calibration slope (Ensemble): -0.9951

## Key Insights

### Top Industries by Average Funding Amount
- Government: $190,000,000.00
- Community: $104,000,000.00
- Artificial Intelligence: $100,870,476.19
- FinTech: $98,937,500.00
- Telecommunications: $82,000,000.00

### Average Funding by Stage
- Series E: $3,500,000,000.00
- Series C: $132,600,000.00
- Initial Coin Offering: $75,000,000.00
- Series G: $50,000,000.00
- Private Equity: $47,500,000.00
- Series B: $42,160,000.00
- Series A: $22,805,263.16
- Venture - Series Unknown: $22,079,708.43
- Seed: $7,797,968.75
- Pre-Seed: $591,666.67

## Accuracy Analysis

### Prediction Accuracy by Percentile
- Predictions within 10% of actual: 
  - QRF: 4.0%
  - Ensemble: 4.4%
- Predictions within 20% of actual: 
  - QRF: 7.7%
  - Ensemble: 8.7%
- Predictions within 50% of actual: 
  - QRF: 21.8%
  - Ensemble: 20.1%
- Predictions within same order of magnitude: 
  - QRF: 77.5%
  - Ensemble: 71.5%

### Prediction Interval Coverage
The 80% prediction interval coverage of 66.3% indicates how well the model's uncertainty estimates match the actual data.
An ideal coverage would be exactly 80%.

## Recommendations
1. **Industry Focus**: The highest funding amounts are observed in [list top industries], suggesting these may be high-potential sectors for investment.
2. **Stage Optimization**: [highest funding stage] shows the largest average funding, while [lowest funding stage] shows the lowest. Consider this when planning fundraising strategies.
3. **Prediction Reliability**: The model shows [strong/moderate/weak] performance with R² of [metrics value] and [x]% of predictions within 20% of actual values.
4. **Uncertainty Handling**: Use the quantile regression model for risk assessment, as it provides both median predictions and uncertainty intervals.
