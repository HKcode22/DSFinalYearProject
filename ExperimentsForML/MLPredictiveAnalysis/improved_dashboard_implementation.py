#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Dashboard Implementation for Funding Stage Prediction
Fixes issues with the original implementation and better organizes output
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

# Import classes from the existing funding_stage_prediction.py
from MLPredictiveAnalysis.funding_stage_prediction import (
    DataLoader, 
    FeatureEngineering,
    EnhancedModelTrainer,
    ModelTrainer,
    DashboardGenerator,
    AdvancedDashboardGenerator,
    EnhancedPipeline,
    Visualizer
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"improved_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedDashboardImplementation:
    """
    Implementation of dashboards for funding stage prediction visualization
    with better organization and handling all records.
    """
    
    def __init__(self, base_dir="./", output_dir="./FundingStageOutput"):
        """
        Initialize improved dashboard implementation.
        
        Args:
            base_dir (str): Base directory for data
            output_dir (str): Main output directory for all outputs
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create main output directory structure
        self.models_dir = os.path.join(output_dir, "models")
        self.data_dir = os.path.join(output_dir, "data")
        self.visualization_dir = os.path.join(output_dir, "visualizations")
        self.dashboard_dir = os.path.join(output_dir, "dashboards")
        self.timeseries_dir = os.path.join(output_dir, "time_series_forecasts")
        
        # Create comparison directories
        self.historical_comparison_dir = os.path.join(self.dashboard_dir, "historical_comparison")
        
        # Create all directories
        for directory in [self.models_dir, self.data_dir, self.visualization_dir, 
                         self.dashboard_dir, self.timeseries_dir, 
                         self.historical_comparison_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create subdirectories for dashboards
        self.classification_dir = os.path.join(self.dashboard_dir, "classification_dashboards")
        self.timeseries_dashboard_dir = os.path.join(self.dashboard_dir, "timeseries_dashboards")
        
        os.makedirs(os.path.join(self.classification_dir, "model_comparison"), exist_ok=True)
        os.makedirs(os.path.join(self.classification_dir, "calibration_curves"), exist_ok=True)
        os.makedirs(os.path.join(self.classification_dir, "feature_importance"), exist_ok=True)
        os.makedirs(os.path.join(self.classification_dir, "confusion_matrices"), exist_ok=True)
        os.makedirs(os.path.join(self.classification_dir, "model_metrics"), exist_ok=True)
        
        os.makedirs(os.path.join(self.timeseries_dashboard_dir, "forecast_trends"), exist_ok=True)
        os.makedirs(os.path.join(self.timeseries_dashboard_dir, "industry_breakdown"), exist_ok=True)
        os.makedirs(os.path.join(self.timeseries_dashboard_dir, "stage_evolution"), exist_ok=True)
        os.makedirs(os.path.join(self.timeseries_dashboard_dir, "seasonality_analysis"), exist_ok=True)
        os.makedirs(os.path.join(self.timeseries_dashboard_dir, "historical_vs_predicted"), exist_ok=True)
        
        # Initialize components
        self.data_loader = self._create_patched_data_loader(base_dir)
        self.feature_engineer = FeatureEngineering()
        
        # Initialize visualizers with improved organization
        self.visualizer = Visualizer(self.visualization_dir)
        
        # Initialize dashboard generators
        self.dashboard_generator = DashboardGenerator(self.dashboard_dir)
        self.advanced_dashboard_generator = AdvancedDashboardGenerator(
            os.path.join(output_dir, "advanced_dashboards")
        )
        
        # Initialize model trainer
        self.model_trainer = EnhancedModelTrainer(self.models_dir)
        
        logger.info(f"Improved dashboard implementation initialized with output to: {output_dir}")
        logger.info(f"Created organized directory structure for all outputs")

    def _create_patched_data_loader(self, base_dir):
        """
        Create a patched version of the DataLoader that can handle list-format JSON data.
        
        Args:
            base_dir (str): Base directory for data
            
        Returns:
            DataLoader: Patched DataLoader instance
        """
        
        data_loader = DataLoader(base_dir)
        
        # Patch the load_fundraiser_data method to handle list format JSON
        original_load_fundraiser = data_loader.load_fundraiser_data
        
        def patched_load_fundraiser():
            try:
                with open(data_loader.fundraiser_path, 'r') as file:
                    data = json.load(file)

                # Handle both list and dictionary formats
                if isinstance(data, list):
                    # Direct list of companies
                    companies = data
                else:
                    # Extract companies from the JSON structure
                    companies = data.get('companies', [])
                    
                df = pd.DataFrame(companies)

                # Add timestamp for versioning
                df['data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Convert numeric fields
                if 'Funding_Amount_USD' in df.columns:
                    df['Funding_Amount_USD'] = pd.to_numeric(
                        df['Funding_Amount_USD'], errors='coerce')

                if 'Total_Employees' in df.columns:
                    df['Total_Employees'] = pd.to_numeric(
                        df['Total_Employees'], errors='coerce')
                
                # Standardize Funding_Type if present
                if 'Funding_Type' in df.columns:
                    df['Funding_Type'] = df['Funding_Type'].apply(data_loader._standardize_funding_type)
                    
                    # Log unique funding types found
                    unique_funding_types = df['Funding_Type'].dropna().unique()
                    logger.info(f"Found funding types in fundraiser data: {unique_funding_types}")

                logger.info(f"Loaded {len(df)} records from fundraiser data")
                return df
            except Exception as e:
                logger.error(f"Error loading fundraiser data: {e}")
                return pd.DataFrame()
        
        # Replace the method
        data_loader.load_fundraiser_data = patched_load_fundraiser
        
        return data_loader

    def load_and_process_data(self):
        """
        Load and process data using existing functionality, ensuring all records are used.
        
        Returns:
            tuple: Processed data and features
        """
        logger.info("Loading and processing data...")
        
        # Use existing DataLoader to load and merge datasets
        merged_data = self.data_loader.merge_datasets()
        
        if merged_data.empty:
            logger.error("No data available after merging. Aborting.")
            return None, None
        
        # Save merged data for reference
        merged_data_path = os.path.join(self.data_dir, f"merged_data_{self.timestamp}.csv")
        merged_data.to_csv(merged_data_path, index=False)
        logger.info(f"Saved merged data with {len(merged_data)} records to {merged_data_path}")
        
        # Use existing FeatureEngineering to extract features
        processed_data = self.feature_engineer.extract_features(merged_data)
        
        # Save processed data for reference
        processed_data_path = os.path.join(self.data_dir, f"processed_data_{self.timestamp}.csv")
        processed_data.to_csv(processed_data_path, index=False)
        logger.info(f"Saved processed data with {len(processed_data)} records to {processed_data_path}")
        
        # Prepare model data
        X, y = self.feature_engineer.prepare_model_data(processed_data)
        
        return processed_data, (X, y)

    def run_classification_models(self, X, y):
        """
        Run classification models directly without using the pipeline.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            dict: Classification results
        """
        logger.info("Training classification models...")
        
        # Remove extremely rare classes (with only 1 instance)
        class_counts = pd.Series(y).value_counts()
        rare_classes = class_counts[class_counts < 2].index
        
        if len(rare_classes) > 0:
            logger.info(f"Removing {len(rare_classes)} extremely rare classes with fewer than 2 samples")
            mask = ~pd.Series(y).isin(rare_classes)
            X = X[mask] if isinstance(X, np.ndarray) else X.loc[mask]
            y = y[mask]
            logger.info(f"After removing rare classes: X shape={X.shape}, y shape={y.shape}")
        
        # Split data into train and test sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model_results = {}
        
        # Train Random Forest
        logger.info("Training Random Forest model...")
        rf_model, rf_results = self.model_trainer.train_random_forest(X_train, y_train)
        if rf_model is not None:
            rf_metrics = rf_results
            rf_metrics['y_test'] = y_test
            rf_metrics['y_pred'] = rf_model.predict(X_test)
            if hasattr(rf_model, 'predict_proba'):
                rf_metrics['y_proba'] = rf_model.predict_proba(X_test)
            model_results['Random Forest'] = rf_metrics
        
        # Train XGBoost
        logger.info("Training XGBoost model...")
        xgb_model, xgb_results = self.model_trainer.train_xgboost(X_train, y_train)
        if xgb_model is not None:
            xgb_metrics = xgb_results
            xgb_metrics['y_test'] = y_test
            xgb_metrics['y_pred'] = xgb_model.predict(X_test)
            if hasattr(xgb_model, 'predict_proba'):
                xgb_metrics['y_proba'] = xgb_model.predict_proba(X_test)
            model_results['XGBoost'] = xgb_metrics
        
        # Try LightGBM if available
        try:
            logger.info("Training LightGBM model...")
            lgbm_model, lgbm_results = self.model_trainer.train_lightgbm(X_train, y_train)
            if lgbm_model is not None:
                lgbm_metrics = self.model_trainer.evaluate_model(lgbm_model, X_test, y_test, "LightGBM")
                lgbm_metrics['y_test'] = y_test
                lgbm_metrics['y_pred'] = lgbm_model.predict(X_test)
                if hasattr(lgbm_model, 'predict_proba'):
                    lgbm_metrics['y_proba'] = lgbm_model.predict_proba(X_test)
                model_results['LightGBM'] = lgbm_metrics
        except Exception as e:
            logger.warning(f"Error training LightGBM model: {str(e)}")
        
        # Save model results
        for model_name, results in model_results.items():
            result_path = os.path.join(self.models_dir, f"{model_name.lower().replace(' ', '_')}_results_{self.timestamp}.json")
            try:
                # Save only serializable parts
                serializable_results = {
                    'accuracy': results.get('accuracy', None),
                    'f1_scores': results.get('f1_scores', None),
                    'precision': results.get('precision', None),
                    'recall': results.get('recall', None),
                }
                with open(result_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save model results to {result_path}: {str(e)}")
        
        return model_results

    def run_time_series_prediction(self):
        """
        Run time series prediction with historical vs. predicted analysis.
        
        Returns:
            dict: Time series results
        """
        logger.info("Running time series pipeline...")
        
        # Create a patched EnhancedPipeline with the fixed output directories
        pipeline = EnhancedPipeline(
            self.base_dir,
            output_dir=self.output_dir,
            archive=False
        )
        
        # Run time series prediction
        timeseries_results = pipeline.time_series_prediction()
        
        # Create historical vs. predicted comparison
        self._create_historical_vs_predicted_comparison(timeseries_results)
        
        return timeseries_results
        
    def _create_historical_vs_predicted_comparison(self, timeseries_results):
        """
        Create visualizations comparing historical data vs. predictions.
        
        Args:
            timeseries_results: Results from time series prediction
        """
        logger.info("Creating historical vs. predicted comparison visualizations...")
        
        try:
            # Only proceed if we have forecast data
            if not timeseries_results or 'forecasts' not in timeseries_results:
                logger.warning("No forecast data available for historical comparison")
                return
            
            forecasts = timeseries_results['forecasts']
            
            # Create combined plot with all forecast types
            plt.figure(figsize=(16, 10))
            plt.suptitle('Historical vs. Predicted Analysis - All Forecasts', fontsize=16)
            
            plot_count = len(forecasts)
            rows = (plot_count + 1) // 2  # Ceiling division
            cols = min(2, plot_count)
            
            # Process each forecast type
            for i, (forecast_type, forecast_data) in enumerate(forecasts.items()):
                if not isinstance(forecast_data, pd.DataFrame):
                    continue
                
                # Separate historical and forecast data
                historical = forecast_data[forecast_data['ds'] <= datetime.now()]
                future = forecast_data[forecast_data['ds'] > datetime.now()]
                
                if len(historical) == 0 or len(future) == 0:
                    continue
                
                # Create subplot
                plt.subplot(rows, cols, i+1)
                
                # Plot historical data
                plt.plot(historical['ds'], historical['y'], 'b-', 
                         label='Historical Data', linewidth=2)
                
                # Plot prediction
                plt.plot(future['ds'], future['yhat'], 'r-', 
                         label='Predicted', linewidth=2)
                
                # Plot prediction intervals
                if 'yhat_lower' in future.columns and 'yhat_upper' in future.columns:
                    plt.fill_between(future['ds'], future['yhat_lower'], future['yhat_upper'], 
                                   color='r', alpha=0.2, label='Prediction Interval')
                
                # Add labels and title
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.title(f'{forecast_type.replace("_", " ").title()}')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Format dates on x-axis
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
                plt.xticks(rotation=45)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
            
            # Save the combined figure
            output_path = os.path.join(
                self.timeseries_dashboard_dir,
                "historical_vs_predicted",
                f"combined_historical_vs_predicted_{self.timestamp}.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created combined historical vs. predicted visualization at {output_path}")
                
            # Process each forecast type individually
            for forecast_type, forecast_data in forecasts.items():
                if not isinstance(forecast_data, pd.DataFrame):
                    continue
                
                # Separate historical and forecast data
                historical = forecast_data[forecast_data['ds'] <= datetime.now()]
                future = forecast_data[forecast_data['ds'] > datetime.now()]
                
                if len(historical) == 0 or len(future) == 0:
                    continue
                
                # Create detailed comparison plot (with both line and trend analysis)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
                
                # Plot 1: Historical and predicted data with intervals
                ax1.plot(historical['ds'], historical['y'], 'b-', 
                        label='Historical Data', linewidth=2)
                ax1.plot(future['ds'], future['yhat'], 'r-', 
                        label='Predicted', linewidth=2)
                
                if 'yhat_lower' in future.columns and 'yhat_upper' in future.columns:
                    ax1.fill_between(future['ds'], future['yhat_lower'], future['yhat_upper'], 
                                   color='r', alpha=0.2, label='Prediction Interval')
                
                # Add trend line for historical data
                if len(historical) > 1:
                    from scipy import stats
                    # Convert dates to numerical values for regression
                    x_dates = historical['ds'].map(datetime.toordinal)
                    y_values = historical['y']
                    # Calculate trend line
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_dates, y_values)
                    trend_line = intercept + slope * x_dates
                    # Plot trend
                    ax1.plot(historical['ds'], trend_line, 'g--', linewidth=1.5, 
                            label=f'Historical Trend (r²={r_value**2:.2f})')
                
                ax1.set_title(f'Historical vs. Predicted: {forecast_type.replace("_", " ").title()}')
                ax1.set_ylabel('Value')
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # Plot 2: Growth rate or percent change
                if len(historical) > 1:
                    # Calculate percent change for historical data
                    hist_pct_change = historical['y'].pct_change() * 100
                    ax2.bar(historical['ds'][1:], hist_pct_change[1:], 
                           color='blue', alpha=0.6, label='Historical % Change')
                    
                    # Calculate average historical growth
                    avg_hist_change = hist_pct_change.mean()
                    ax2.axhline(y=avg_hist_change, color='blue', linestyle='--', 
                               label=f'Avg Historical: {avg_hist_change:.1f}%')
                
                if len(future) > 1:
                    # Calculate percent change for predicted data
                    pred_pct_change = future['yhat'].pct_change() * 100
                    ax2.bar(future['ds'][1:], pred_pct_change[1:], 
                           color='red', alpha=0.6, label='Predicted % Change')
                    
                    # Calculate average predicted growth
                    avg_pred_change = pred_pct_change.mean()
                    ax2.axhline(y=avg_pred_change, color='red', linestyle='--', 
                               label=f'Avg Predicted: {avg_pred_change:.1f}%')
                
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Percent Change (%)')
                ax2.set_title('Growth Rate Analysis')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.7)
                
                # Format dates on both x-axes
                for ax in [ax1, ax2]:
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
                plt.tight_layout()
                
                # Save the detailed figure
                output_path = os.path.join(
                    self.timeseries_dashboard_dir,
                    "historical_vs_predicted",
                    f"{forecast_type}_detailed_comparison_{self.timestamp}.png"
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Created detailed historical vs. predicted comparison for {forecast_type} at {output_path}")
                
        except Exception as e:
            logger.error(f"Error creating historical vs. predicted comparison: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def generate_dashboards(self):
        """
        Generate all dashboards with better organization.
        
        Returns:
            dict: Dashboard paths
        """
        logger.info("Generating organized dashboards...")
        
        # Load and process data
        processed_data, (X, y) = self.load_and_process_data()
        
        if processed_data is None:
            logger.error("Failed to load and process data.")
            return None
        
        # Get results from classification models
        classification_results = self.run_classification_models(X, y)
        
        # Get results from time series prediction
        timeseries_results = self.run_time_series_prediction()
        
        dashboard_paths = {}
        
        # Generate standard dashboards
        if classification_results or timeseries_results:
            logger.info("Generating standard dashboards...")
            standard_paths = self.dashboard_generator.generate_all_dashboards(
                classification_results, timeseries_results
            )
            dashboard_paths['standard'] = standard_paths
            
            # Generate advanced dashboards
            logger.info("Generating advanced dashboards...")
            advanced_paths = self.advanced_dashboard_generator.generate_advanced_dashboards(
                classification_results, timeseries_results
            )
            dashboard_paths['advanced'] = advanced_paths
        else:
            logger.error("No model results available to generate dashboards")
        
        return dashboard_paths

    def create_calibration_plots(self, classification_results):
        """
        Create calibration plot dashboards for classification models.
        
        Args:
            classification_results: Classification model results
        """
        logger.info("Creating calibration plots...")
        
        try:
            # Only proceed if we have classification results
            if not classification_results:
                logger.warning("No classification results available for calibration plots")
                return
            
            from sklearn.calibration import calibration_curve
            
            # Create directory for calibration plots
            calibration_dir = os.path.join(self.classification_dir, "calibration_curves")
            os.makedirs(calibration_dir, exist_ok=True)
            
            # Process each model
            for model_name, results in classification_results.items():
                # Skip if no probabilities available
                if 'y_proba' not in results or 'y_test' not in results:
                    continue
                
                y_test = results['y_test']
                y_proba = results['y_proba']
                
                # For binary classification
                if y_proba.shape[1] == 2:
                    plt.figure(figsize=(10, 8))
                    
                    # Calculate calibration curve
                    prob_true, prob_pred = calibration_curve(y_test, y_proba[:, 1], n_bins=10)
                    
                    # Plot calibration curve
                    plt.plot(prob_pred, prob_true, 's-', label=model_name)
                    
                    # Plot reference line (perfect calibration)
                    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
                    
                    # Add labels and title
                    plt.xlabel('Mean Predicted Probability')
                    plt.ylabel('Fraction of Positives')
                    plt.title(f'Calibration Curve - {model_name}')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Save the figure
                    output_path = os.path.join(
                        calibration_dir,
                        f"{model_name.lower().replace(' ', '_')}_calibration_{self.timestamp}.png"
                    )
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Created calibration plot for {model_name} at {output_path}")
                
                # For multiclass classification
                else:
                    # Create combined plot for all classes
                    plt.figure(figsize=(12, 10))
                    
                    # For each class
                    n_classes = y_proba.shape[1]
                    for i in range(n_classes):
                        # Create binary labels for this class
                        y_test_binary = (y_test == i).astype(int)
                        
                        # Calculate calibration curve for this class
                        prob_true, prob_pred = calibration_curve(
                            y_test_binary, y_proba[:, i], n_bins=10)
                        
                        # Plot calibration curve for this class
                        plt.plot(prob_pred, prob_true, 's-', 
                                label=f'Class {i}')
                    
                    # Plot reference line (perfect calibration)
                    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
                    
                    # Add labels and title
                    plt.xlabel('Mean Predicted Probability')
                    plt.ylabel('Fraction of Positives')
                    plt.title(f'Multiclass Calibration Curves - {model_name}')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Save the figure
                    output_path = os.path.join(
                        calibration_dir,
                        f"{model_name.lower().replace(' ', '_')}_multiclass_calibration_{self.timestamp}.png"
                    )
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Created multiclass calibration plot for {model_name} at {output_path}")
                
        except Exception as e:
            logger.error(f"Error creating calibration plots: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

def main():
    """Main function to run improved dashboard implementation."""
    logger.info("Starting improved dashboard implementation...")
    
    # Get base directory as current working directory
    base_dir = os.getcwd()
    
    # Create output directory with proper organization
    output_dir = os.path.join(base_dir, "FundingStageOutput")
    
    # Initialize improved dashboard implementation
    dashboard_impl = ImprovedDashboardImplementation(base_dir, output_dir)
    
    # Generate dashboards
    dashboard_paths = dashboard_impl.generate_dashboards()
    
    # Create additional calibration plots
    processed_data, (X, y) = dashboard_impl.load_and_process_data()
    if processed_data is not None:
        classification_results = dashboard_impl.run_classification_models(X, y)
        dashboard_impl.create_calibration_plots(classification_results)
    
    if dashboard_paths:
        logger.info("Dashboard generation complete.")
        logger.info(f"All outputs saved to: {output_dir}")
        
        # Print paths to dashboards
        for dash_type, paths in dashboard_paths.items():
            logger.info(f"{dash_type.capitalize()} Dashboards:")
            for name, path in paths.items():
                if isinstance(path, dict):
                    logger.info(f"  {name}:")
                    for subname, subpath in path.items():
                        logger.info(f"    {subname}: {subpath}")
                else:
                    logger.info(f"  {name}: {path}")
    else:
        logger.error("Failed to generate dashboards.")

if __name__ == "__main__":
    main() 