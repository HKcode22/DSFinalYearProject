# Funding Amount Forecast Analysis Report
Generated on: 2025-04-22 23:58:52

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
- RMSE: $112,302,500.05
- MAE: $38,379,605.71
- R2: -0.0121
- Log RMSE: 5.5996
- Log MAE: 3.0468
- Log R2: -0.1705
- Within 10%: 3.7%
- Within 20%: 8.7%
- Within 50%: 21.8%
- Within Same Order: 76.2%
- Interval Coverage 80%: 66.6%

### Ensemble Model
- Model Type: ensemble
- RMSE: $272,093,678.31
- MAE: $74,422,490.85
- R2: -4.9416
- Log RMSE: 5.9947
- Log MAE: 3.3478
- Log R2: -0.3415
- Within 10%: 4.7%
- Within 20%: 9.4%
- Within 50%: 19.5%
- Within Same Order: 70.8%

#### Individual Model Performance
**Random Forest**
- RMSE: $269,893,696.29
- MAE: $62,018,285.31
- R2: -4.8459
- Within 20%: 10.4%

**Gradient Boosting**
- RMSE: $642,472,777.80
- MAE: $104,201,196.69
- R2: -32.1264
- Within 20%: 8.4%

**Extra Trees**
- RMSE: $137,076,074.22
- MAE: $62,157,504.15
- R2: -0.5080
- Within 20%: 6.7%

## Model Calibration Analysis
The model calibration analysis measures how well the predicted values match the actual values. A well-calibrated model will have residuals (actual minus predicted) centered around zero with no clear patterns.

- Mean residual (QRF): $13,276,441.21
- Mean absolute residual (QRF): $38,379,605.71
- Residual standard deviation (QRF): $111,514,965.93
- Calibration slope (QRF): -0.4538

- Mean residual (Ensemble): $-991,950.77
- Mean absolute residual (Ensemble): $74,422,490.85
- Residual standard deviation (Ensemble): $272,091,870.16
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
  - QRF: 3.7%
  - Ensemble: 4.7%
- Predictions within 20% of actual: 
  - QRF: 8.7%
  - Ensemble: 9.4%
- Predictions within 50% of actual: 
  - QRF: 21.8%
  - Ensemble: 19.5%
- Predictions within same order of magnitude: 
  - QRF: 76.2%
  - Ensemble: 70.8%

### Prediction Interval Coverage
The 80% prediction interval coverage of 66.6% indicates how well the model's uncertainty estimates match the actual data.
An ideal coverage would be exactly 80%.

## Recommendations
1. **Industry Focus**: The highest funding amounts are observed in Government, Community, Artificial Intelligence, suggesting these may be high-potential sectors for investment.
2. **Stage Optimization**: Series E shows the largest average funding ($3,500,000,000.00), while Pre-Seed shows the lowest ($591,666.67). Consider this when planning fundraising strategies.
3. **Prediction Reliability**: The model shows weak performance with R² of -0.0121 and 8.7% of predictions within 20% of actual values.
4. **Uncertainty Handling**: Use the quantile regression model for risk assessment, as it provides both median predictions and uncertainty intervals.
