# Funding Amount Forecast Analysis Report
Generated on: 2025-04-22 16:59:49

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
- RMSE: $112,411,323.31
- MAE: $38,484,567.17
- R2: -0.0141
- Log RMSE: 5.5960
- Log MAE: 3.0471
- Log R2: -0.1690
- Within 10%: 5.4%
- Within 20%: 9.1%
- Within 50%: 22.5%
- Within Same Order: 76.2%
- Interval Coverage 80%: 65.4%

### Ensemble Model
- Model Type: ensemble
- RMSE: $274,894,941.15
- MAE: $74,885,053.65
- R2: -5.0645
- Log RMSE: 5.9933
- Log MAE: 3.3409
- Log R2: -0.3409
- Within 10%: 4.7%
- Within 20%: 8.1%
- Within 50%: 21.5%
- Within Same Order: 71.8%

#### Individual Model Performance
**Random Forest**
- RMSE: $267,970,810.07
- MAE: $61,957,396.00
- R2: -4.7629
- Within 20%: 10.1%

**Gradient Boosting**
- RMSE: $676,897,816.57
- MAE: $104,886,562.02
- R2: -35.7714
- Within 20%: 8.1%

**Extra Trees**
- RMSE: $147,541,950.45
- MAE: $63,177,869.88
- R2: -0.7470
- Within 20%: 6.7%

## Model Calibration Analysis
The model calibration analysis measures how well the predicted values match the actual values. A well-calibrated model will have residuals (actual minus predicted) centered around zero with no clear patterns.

- Mean residual (QRF): $12,946,715.84
- Mean absolute residual (QRF): $38,484,567.17
- Residual standard deviation (QRF): $111,663,280.26
- Calibration slope (QRF): -0.5147

- Mean residual (Ensemble): $1,988,473.38
- Mean absolute residual (Ensemble): $74,885,053.65
- Residual standard deviation (Ensemble): $274,887,749.17
- Calibration slope (Ensemble): -0.9953

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
  - QRF: 5.4%
  - Ensemble: 4.7%
- Predictions within 20% of actual: 
  - QRF: 9.1%
  - Ensemble: 8.1%
- Predictions within 50% of actual: 
  - QRF: 22.5%
  - Ensemble: 21.5%
- Predictions within same order of magnitude: 
  - QRF: 76.2%
  - Ensemble: 71.8%

### Prediction Interval Coverage
The 80% prediction interval coverage of 65.4% indicates how well the model's uncertainty estimates match the actual data.
An ideal coverage would be exactly 80%.

## Recommendations
1. **Industry Focus**: The highest funding amounts are observed in Government, Community, Artificial Intelligence, suggesting these may be high-potential sectors for investment.
2. **Stage Optimization**: Series E shows the largest average funding ($3,500,000,000.00), while Pre-Seed shows the lowest ($591,666.67). Consider this when planning fundraising strategies.
3. **Prediction Reliability**: The model shows weak performance with R² of -0.0141 and 9.1% of predictions within 20% of actual values.
4. **Uncertainty Handling**: Use the quantile regression model for risk assessment, as it provides both median predictions and uncertainty intervals.
