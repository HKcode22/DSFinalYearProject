# Funding Amount Forecast Analysis Report
Generated on: 2025-04-23 00:09:55

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
- RMSE: $112,193,501.94
- MAE: $38,288,360.55
- R2: -0.0102
- Log RMSE: 5.5859
- Log MAE: 3.0416
- Log R2: -0.1648
- Within 10%: 4.0%
- Within 20%: 8.7%
- Within 50%: 21.8%
- Within Same Order: 76.5%
- Interval Coverage 80%: 66.0%

### Ensemble Model
- Model Type: ensemble
- RMSE: $260,669,536.56
- MAE: $73,525,372.37
- R2: -4.4531
- Log RMSE: 5.9857
- Log MAE: 3.3412
- Log R2: -0.3374
- Within 10%: 6.7%
- Within 20%: 9.4%
- Within 50%: 19.8%
- Within Same Order: 71.5%

#### Individual Model Performance
**Random Forest**
- RMSE: $264,003,066.29
- MAE: $61,655,527.23
- R2: -4.5935
- Within 20%: 10.1%

**Gradient Boosting**
- RMSE: $596,723,757.12
- MAE: $101,940,180.52
- R2: -27.5766
- Within 20%: 7.4%

**Extra Trees**
- RMSE: $140,655,496.73
- MAE: $62,112,015.69
- R2: -0.5877
- Within 20%: 6.0%

## Model Calibration Analysis
The model calibration analysis measures how well the predicted values match the actual values. A well-calibrated model will have residuals (actual minus predicted) centered around zero with no clear patterns.

- Mean residual (QRF): $13,273,782.19
- Mean absolute residual (QRF): $38,288,360.55
- Residual standard deviation (QRF): $111,405,514.15
- Calibration slope (QRF): -0.4082

- Mean residual (Ensemble): $-1,497,035.66
- Mean absolute residual (Ensemble): $73,525,372.37
- Residual standard deviation (Ensemble): $260,665,237.76
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
  - Ensemble: 6.7%
- Predictions within 20% of actual: 
  - QRF: 8.7%
  - Ensemble: 9.4%
- Predictions within 50% of actual: 
  - QRF: 21.8%
  - Ensemble: 19.8%
- Predictions within same order of magnitude: 
  - QRF: 76.5%
  - Ensemble: 71.5%

### Prediction Interval Coverage
The 80% prediction interval coverage of 66.0% indicates how well the model's uncertainty estimates match the actual data.
An ideal coverage would be exactly 80%.

## Recommendations
1. **Industry Focus**: The highest funding amounts are observed in Government, Community, Artificial Intelligence, suggesting these may be high-potential sectors for investment.
2. **Stage Optimization**: Series E shows the largest average funding ($3,500,000,000.00), while Pre-Seed shows the lowest ($591,666.67). Consider this when planning fundraising strategies.
3. **Prediction Reliability**: The model shows weak performance with R² of -0.0102 and 8.7% of predictions within 20% of actual values.
4. **Uncertainty Handling**: Use the quantile regression model for risk assessment, as it provides both median predictions and uncertainty intervals.
