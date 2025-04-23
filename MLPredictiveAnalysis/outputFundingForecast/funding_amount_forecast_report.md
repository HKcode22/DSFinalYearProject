# Funding Amount Forecast Analysis Report
Generated on: 2025-04-22 09:58:13

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
- RMSE: $112,245,680.41
- MAE: $38,531,849.66
- R2: -0.0111
- Log RMSE: 5.5927
- Log MAE: 3.0559
- Log R2: -0.1676
- Within 10%: 3.4%
- Within 20%: 6.7%
- Within 50%: 21.8%
- Within Same Order: 76.5%
- Interval Coverage 80%: 66.6%

### Ensemble Model
- Model Type: ensemble
- RMSE: $261,780,097.50
- MAE: $73,659,178.02
- R2: -4.4997
- Log RMSE: 5.9935
- Log MAE: 3.3451
- Log R2: -0.3410
- Within 10%: 4.4%
- Within 20%: 9.4%
- Within 50%: 19.8%
- Within Same Order: 70.8%

#### Individual Model Performance
**Random Forest**
- RMSE: $269,998,444.13
- MAE: $62,123,836.04
- R2: -4.8504
- Within 20%: 10.4%

**Gradient Boosting**
- RMSE: $601,075,083.82
- MAE: $102,227,973.00
- R2: -27.9949
- Within 20%: 7.7%

**Extra Trees**
- RMSE: $136,038,355.40
- MAE: $61,826,038.30
- R2: -0.4852
- Within 20%: 6.7%

## Model Calibration Analysis
The model calibration analysis measures how well the predicted values match the actual values. A well-calibrated model will have residuals (actual minus predicted) centered around zero with no clear patterns.

- Mean residual (QRF): $13,147,756.24
- Mean absolute residual (QRF): $38,531,849.66
- Residual standard deviation (QRF): $111,472,997.97
- Calibration slope (QRF): -0.4396

- Mean residual (Ensemble): $-1,670,468.40
- Mean absolute residual (Ensemble): $73,659,178.02
- Residual standard deviation (Ensemble): $261,774,767.65
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
  - QRF: 3.4%
  - Ensemble: 4.4%
- Predictions within 20% of actual: 
  - QRF: 6.7%
  - Ensemble: 9.4%
- Predictions within 50% of actual: 
  - QRF: 21.8%
  - Ensemble: 19.8%
- Predictions within same order of magnitude: 
  - QRF: 76.5%
  - Ensemble: 70.8%

### Prediction Interval Coverage
The 80% prediction interval coverage of 66.6% indicates how well the model's uncertainty estimates match the actual data.
An ideal coverage would be exactly 80%.

## Recommendations
- Focus on companies in high-funding industries for better investment returns
- Consider portfolio diversification across different funding stages
- Monitor market trends and adjust investment strategy accordingly
- Use prediction intervals to set reasonable expectations for funding outcomes
- For early-stage companies, consider wider prediction intervals to account for higher uncertainty
- For more established companies, the model's predictions are typically more accurate

## Visualizations
The following visualizations are available in the output directory:
- Funding by Stage: Breakdown of funding amounts across different stages
- Funding by Industry: Analysis of funding distribution across industries
- Prediction Accuracy: Comparison of predicted vs actual funding amounts
- Feature Importance: Key factors influencing funding amount predictions
- Model Calibration: Assessment of model prediction reliability
- Error Distribution: Analysis of prediction errors
- Prediction Intervals: Uncertainty in funding amount predictions
