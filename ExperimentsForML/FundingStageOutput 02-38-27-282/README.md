# Funding Stage Prediction Project Output

This directory contains the organized outputs from the Funding Stage Prediction project. The implementation processes 1,749 startup records from multiple data sources to perform classification of funding stages and time series forecasting of funding trends.

## Directory Structure

- **data/** - Contains merged and processed data files in CSV format
  - merged_data_*.csv - Raw merged data from all sources
  - processed_data_*.csv - Data after feature engineering

- **models/** - Trained machine learning models and their performance metrics
  - prediction_audit.csv - Audit trail of model predictions

- **dashboards/** - Visualizations of model performance and predictions
  - **classification_dashboards/** - Visualizations for classification models
    - **model_comparison/** - Comparative analysis of different ML models
    - **calibration_curves/** - Model calibration analysis
    - **feature_importance/** - Feature importance visualizations
    - **confusion_matrices/** - Confusion matrices for classification models
    - **model_metrics/** - Performance metrics (accuracy, precision, recall, F1)
  
  - **timeseries_dashboards/** - Visualizations for time series forecasts
    - **forecast_trends/** - Overall funding trend forecasts
    - **industry_breakdown/** - Industry-specific forecasts
    - **stage_evolution/** - Evolution of funding stages over time
    - **seasonality_analysis/** - Seasonal patterns in funding
    - **historical_vs_predicted/** - Comparison between historical data and predictions

- **time_series_forecasts/** - Forecasted data for different dimensions
  - Funding amounts forecasts (CSV and component visualizations)
  - Industry-specific forecasts
  - Funding stage forecasts
  - Funding range forecasts

## Improvements Made

1. **Data Handling**:
   - Now properly processes all 1,749 startup records
   - Fixed JSON structure handling to accommodate different formats
   - Properly handles rare classes for more robust model training

2. **Visualization Enhancements**:
   - Added calibration curves for classification models
   - Created historical vs. predicted comparisons for time series data
   - Added detailed growth rate analysis in time series comparisons

3. **Output Organization**:
   - Structured directory hierarchy for easy navigation
   - Consistent file naming conventions with timestamps
   - Separate subdirectories for different visualization types

4. **Classification Models**:
   - Random Forest
   - XGBoost
   - (Optional) LightGBM when available

5. **Time Series Forecasts**:
   - Overall funding amount trends
   - Industry-specific trends
   - Funding stage evolution
   - Seasonality analysis

## Usage

The CSV files can be used for further analysis, and the visualizations provide insights into funding patterns and model performance. All forecast data includes prediction intervals to assess uncertainty.

For custom queries or further analysis, use the processed data files in the `data/` directory. 