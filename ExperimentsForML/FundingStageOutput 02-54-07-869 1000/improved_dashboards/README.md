# Improved Dashboards for Funding Stage Prediction

This folder contains improved visualizations for both classification and time series forecasting models.

## Classification Dashboard

The classification dashboard includes:

### 1. Model Performance Overview
- Comparison of accuracy metrics across multiple models (Random Forest, XGBoost, Ensemble)
- Clear visualization of precision, recall, F1 score, and ROC AUC
- Visual indication of the best performing model

### 2. Multiclass Calibration Curves
- Improved calibration curves for each model with smoother plotting
- Clear class differentiation with consistent color scheme
- Proper explanation of over/under-confidence
- Better handling of classes with few examples
- Removal of erratic zigzag patterns in the original plots

## Time Series Dashboard

The time series dashboard includes:

### 1. Funding Trends Forecast
- Clear visualization of historical vs. predicted funding
- Visual distinction between historical data (solid line with markers) and forecast (dashed line)
- Confidence intervals that widen as predictions extend further into the future
- Vertical line marking the boundary between historical and forecast data
- Markers for significant market events
- Note about forecast horizon options (3, 6, 12, 24 months)

## Improvements Made

1. **Better Data Handling**
   - Applied proper filtering and minimal thresholds for calibration curves
   - Ensured proper binning strategy for smoother curves
   - Used consistent color schemes for better visual interpretation

2. **Enhanced Visual Design**
   - Applied professional formatting and styling
   - Added proper annotations and explanations
   - Improved legends and axes labels
   - Used appropriate markers and line styles

3. **Structural Improvements**
   - Organized visualizations in a logical folder structure
   - Created standalone scripts for generating visualizations
   - Ensured compatibility with existing data structures

## Usage

These dashboards provide a comprehensive view of model performance and predictions. They can be used for:

1. Evaluating model accuracy and calibration
2. Comparing different models
3. Visualizing funding trends and forecasts
4. Making informed decisions about startup funding stages 