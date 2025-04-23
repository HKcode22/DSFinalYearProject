# Funding Amount Forecast Analysis Report
Generated on: 2025-04-22 16:51:56

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
- RMSE: $112,298,857.97
- MAE: $38,361,825.57
- R2: -0.0121
- Log RMSE: 5.6567
- Log MAE: 3.0881
- Log R2: -0.1945
- Within 10%: 4.0%
- Within 20%: 9.4%
- Within 50%: 22.8%
- Within Same Order: 75.8%
- Interval Coverage 80%: 66.3%

### Ensemble Model
- Model Type: ensemble
- RMSE: $261,660,686.59
- MAE: $73,848,843.61
- R2: -4.4947
- Log RMSE: 5.9937
- Log MAE: 3.3468
- Log R2: -0.3410
- Within 10%: 5.4%
- Within 20%: 8.7%
- Within 50%: 19.5%
- Within Same Order: 71.1%

#### Individual Model Performance
**Random Forest**
- RMSE: $270,011,482.02
- MAE: $62,080,033.74
- R2: -4.8510
- Within 20%: 10.4%

**Gradient Boosting**
- RMSE: $595,501,628.13
- MAE: $102,017,694.88
- R2: -27.4597
- Within 20%: 9.1%

**Extra Trees**
- RMSE: $140,994,414.90
- MAE: $62,526,778.11
- R2: -0.5954
- Within 20%: 6.7%

## Model Calibration Analysis
The model calibration analysis measures how well the predicted values match the actual values. A well-calibrated model will have residuals (actual minus predicted) centered around zero with no clear patterns.

- Mean residual (QRF): $13,254,240.26
- Mean absolute residual (QRF): $38,361,825.57
- Residual standard deviation (QRF): $111,513,939.11
- Calibration slope (QRF): -0.4543

- Mean residual (Ensemble): $-2,155,161.53
- Mean absolute residual (Ensemble): $73,848,843.61
- Residual standard deviation (Ensemble): $261,651,810.98
- Calibration slope (Ensemble): -0.9944

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
  - Ensemble: 5.4%
- Predictions within 20% of actual: 
  - QRF: 9.4%
  - Ensemble: 8.7%
- Predictions within 50% of actual: 
  - QRF: 22.8%
  - Ensemble: 19.5%
- Predictions within same order of magnitude: 
  - QRF: 75.8%
  - Ensemble: 71.1%

### Prediction Interval Coverage
The 80% prediction interval coverage of 66.3% indicates how well the model's uncertainty estimates match the actual data.
An ideal coverage would be exactly 80%.

## Recommendations
1. **Industry Focus**: The highest funding amounts are observed in [list top industries], suggesting these may be high-potential sectors for investment.
2. **Stage Optimization**: [highest funding stage] shows the largest average funding, while [lowest funding stage] shows the lowest. Consider this when planning fundraising strategies.
3. **Prediction Reliability**: The model shows [strong/moderate/weak] performance with R² of [metrics value] and [x]% of predictions within 20% of actual values.
4. **Uncertainty Handling**: Use the quantile regression model for risk assessment, as it provides both median predictions and uncertainty intervals.
