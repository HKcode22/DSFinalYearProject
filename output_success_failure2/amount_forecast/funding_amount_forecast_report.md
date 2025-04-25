# Funding Amount Forecast Analysis Report
Generated on: 2025-04-23 02:22:47

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
- RMSE: $112,212,273.43
- MAE: $38,205,684.84
- R2: -0.0105
- Log RMSE: 5.5802
- Log MAE: 3.0250
- Log R2: -0.1624
- Within 10%: 4.4%
- Within 20%: 8.7%
- Within 50%: 23.5%
- Within Same Order: 78.5%
- Interval Coverage 80%: 65.7%

### Ensemble Model
- Model Type: ensemble
- RMSE: $285,399,187.99
- MAE: $76,402,278.49
- R2: -5.5369
- Log RMSE: 5.9954
- Log MAE: 3.3496
- Log R2: -0.3418
- Within 10%: 4.4%
- Within 20%: 8.7%
- Within 50%: 19.8%
- Within Same Order: 70.8%

#### Individual Model Performance
**Random Forest**
- RMSE: $269,925,057.16
- MAE: $61,995,547.81
- R2: -4.8472
- Within 20%: 10.4%

**Gradient Boosting**
- RMSE: $695,685,827.42
- MAE: $109,786,196.90
- R2: -37.8410
- Within 20%: 9.4%

**Extra Trees**
- RMSE: $140,567,883.89
- MAE: $62,539,432.91
- R2: -0.5858
- Within 20%: 6.7%

## Model Calibration Analysis
The model calibration analysis measures how well the predicted values match the actual values. A well-calibrated model will have residuals (actual minus predicted) centered around zero with no clear patterns.

- Mean residual (QRF): $13,258,060.11
- Mean absolute residual (QRF): $38,205,684.84
- Residual standard deviation (QRF): $111,426,290.22
- Calibration slope (QRF): -0.4176

- Mean residual (Ensemble): $1,043,476.92
- Mean absolute residual (Ensemble): $76,402,278.49
- Residual standard deviation (Ensemble): $285,397,280.40
- Calibration slope (Ensemble): -0.9956

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
  - QRF: 4.4%
  - Ensemble: 4.4%
- Predictions within 20% of actual: 
  - QRF: 8.7%
  - Ensemble: 8.7%
- Predictions within 50% of actual: 
  - QRF: 23.5%
  - Ensemble: 19.8%
- Predictions within same order of magnitude: 
  - QRF: 78.5%
  - Ensemble: 70.8%

### Prediction Interval Coverage
The 80% prediction interval coverage of 65.7% indicates how well the model's uncertainty estimates match the actual data.
An ideal coverage would be exactly 80%.

## Recommendations
1. **Industry Focus**: The highest funding amounts are observed in Government, Community, Artificial Intelligence, suggesting these may be high-potential sectors for investment.
2. **Stage Optimization**: Series E shows the largest average funding ($3,500,000,000.00), while Pre-Seed shows the lowest ($591,666.67). Consider this when planning fundraising strategies.
3. **Prediction Reliability**: The model shows weak performance with R² of -0.0105 and 8.7% of predictions within 20% of actual values.
4. **Uncertainty Handling**: Use the quantile regression model for risk assessment, as it provides both median predictions and uncertainty intervals.
