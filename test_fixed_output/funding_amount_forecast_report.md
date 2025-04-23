# Funding Amount Forecast Analysis Report
Generated on: 2025-04-22 16:55:36

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
- RMSE: $112,161,026.47
- MAE: $38,302,705.87
- R2: -0.0096
- Log RMSE: 5.5761
- Log MAE: 3.0420
- Log R2: -0.1607
- Within 10%: 4.0%
- Within 20%: 8.4%
- Within 50%: 22.1%
- Within Same Order: 75.8%
- Interval Coverage 80%: 66.3%

### Ensemble Model
- Model Type: ensemble
- RMSE: $288,009,226.79
- MAE: $76,717,529.11
- R2: -5.6570
- Log RMSE: 6.0687
- Log MAE: 3.3953
- Log R2: -0.3748
- Within 10%: 4.7%
- Within 20%: 8.7%
- Within 50%: 19.5%
- Within Same Order: 71.5%

#### Individual Model Performance
**Random Forest**
- RMSE: $263,950,280.64
- MAE: $61,611,283.96
- R2: -4.5912
- Within 20%: 10.1%

**Gradient Boosting**
- RMSE: $704,257,000.20
- MAE: $110,851,010.06
- R2: -38.8040
- Within 20%: 7.7%

**Extra Trees**
- RMSE: $145,023,511.63
- MAE: $62,879,490.96
- R2: -0.6879
- Within 20%: 6.7%

## Model Calibration Analysis
The model calibration analysis measures how well the predicted values match the actual values. A well-calibrated model will have residuals (actual minus predicted) centered around zero with no clear patterns.

- Mean residual (QRF): $13,451,922.45
- Mean absolute residual (QRF): $38,302,705.87
- Residual standard deviation (QRF): $111,351,433.04
- Calibration slope (QRF): -0.3886

- Mean residual (Ensemble): $1,015,356.68
- Mean absolute residual (Ensemble): $76,717,529.11
- Residual standard deviation (Ensemble): $288,007,437.00
- Calibration slope (Ensemble): -0.9957

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
  - Ensemble: 4.7%
- Predictions within 20% of actual: 
  - QRF: 8.4%
  - Ensemble: 8.7%
- Predictions within 50% of actual: 
  - QRF: 22.1%
  - Ensemble: 19.5%
- Predictions within same order of magnitude: 
  - QRF: 75.8%
  - Ensemble: 71.5%

### Prediction Interval Coverage
The 80% prediction interval coverage of 66.3% indicates how well the model's uncertainty estimates match the actual data.
An ideal coverage would be exactly 80%.

## Recommendations
1. **Industry Focus**: The highest funding amounts are observed in Government, Community, Artificial Intelligence, suggesting these may be high-potential sectors for investment.
2. **Stage Optimization**: Series E shows the largest average funding ($3,500,000,000.00), while Pre-Seed shows the lowest ($591,666.67). Consider this when planning fundraising strategies.
3. **Prediction Reliability**: The model shows weak performance with R² of -0.0096 and 8.4% of predictions within 20% of actual values.
4. **Uncertainty Handling**: Use the quantile regression model for risk assessment, as it provides both median predictions and uncertainty intervals.
