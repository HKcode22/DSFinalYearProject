#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dashboard Implementation for Funding Stage Prediction
This script uses only the existing data and functionality from funding_stage_prediction.py
to create dashboards for visualizing funding stage prediction results.
"""

import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Import classes from the existing funding_stage_prediction.py
from MLPredictiveAnalysis.funding_stage_prediction9 import (
    DataLoader, 
    FeatureEngineering,
    EnhancedModelTrainer,
    DashboardGenerator,
    AdvancedDashboardGenerator,
    EnhancedPipeline
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardImplementation:
    """
    Implementation of dashboards for funding stage prediction visualization
    using existing data and functionality.
    """
    
    def __init__(self, base_dir="./", output_dir="./dashboard_output"):
        """
        Initialize dashboard implementation.
        
        Args:
            base_dir (str): Base directory for data
            output_dir (str): Output directory for dashboards
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(base_dir)
        self.feature_engineer = FeatureEngineering()
        self.dashboard_generator = DashboardGenerator(
            os.path.join(output_dir, "standard_dashboards")
        )
        self.advanced_dashboard_generator = AdvancedDashboardGenerator(
            os.path.join(output_dir, "advanced_dashboards")
        )
        
        # Initialize pipeline
        self.pipeline = EnhancedPipeline(base_dir, output_dir=output_dir)
        
        logger.info(f"Dashboard implementation initialized with output to: {output_dir}")

    def load_and_process_data(self):
        """
        Load and process data using existing functionality.
        
        Returns:
            tuple: Processed data and features
        """
        logger.info("Loading and processing data...")
        
        # Use existing DataLoader to load and merge datasets
        merged_data = self.data_loader.merge_datasets()
        
        if merged_data.empty:
            logger.error("No data available after merging. Aborting.")
            return None, None
        
        # Use existing FeatureEngineering to extract features
        processed_data = self.feature_engineer.extract_features(merged_data)
        
        # Prepare model data
        X, y = self.feature_engineer.prepare_model_data(processed_data)
        
        logger.info(f"Data loaded and processed: {len(processed_data)} records")
        
        return processed_data, (X, y)

    def run_classification_pipeline(self):
        """
        Run classification models using the pipeline.
        
        Returns:
            dict: Classification results
        """
        logger.info("Running classification pipeline...")
        
        # Load data
        processed_data, (X, y) = self.load_and_process_data()
        
        if processed_data is None:
            return None
        
        # Train models using the existing pipeline functionality
        self.pipeline.run()
        
        # Get results from the pipeline
        # Instead of creating new models, access the results from the pipeline run
        model_results = {}
        
        # Check if pipeline has model results
        try:
            # Access model results if available from pipeline
            if hasattr(self.pipeline, 'model_results') and self.pipeline.model_results:
                model_results = self.pipeline.model_results
            else:
                # Otherwise create a minimal result set from the pipeline's best_model
                if hasattr(self.pipeline, 'best_model') and self.pipeline.best_model:
                    logger.info("Creating model results from pipeline's best model")
                    model_results[self.pipeline.best_model_name] = {
                        'model': self.pipeline.best_model,
                        'accuracy': self.pipeline.best_accuracy,
                        'predictions': self.pipeline.y_pred if hasattr(self.pipeline, 'y_pred') else None,
                        'probabilities': self.pipeline.y_proba if hasattr(self.pipeline, 'y_proba') else None,
                        'feature_importance': None
                    }
        except Exception as e:
            logger.error(f"Error accessing pipeline results: {str(e)}")
        
        return model_results

    def run_time_series_pipeline(self):
        """
        Run time series prediction using the pipeline.
        
        Returns:
            dict: Time series results
        """
        logger.info("Running time series pipeline...")
        
        # Use the existing time_series_prediction method from the pipeline
        timeseries_results = self.pipeline.time_series_prediction()
        
        return timeseries_results

    def generate_dashboards(self):
        """
        Generate all dashboards using existing functionality.
        
        Returns:
            dict: Dashboard paths
        """
        logger.info("Generating dashboards...")
        
        # Get classification and time series results
        classification_results = self.run_classification_pipeline()
        timeseries_results = self.run_time_series_pipeline()
        
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

def main():
    """Main function to run dashboard implementation."""
    logger.info("Starting dashboard implementation...")
    
    # Get current directory as base dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "dashboard_output")
    
    # Initialize dashboard implementation
    dashboard_impl = DashboardImplementation(base_dir, output_dir)
    
    # Generate dashboards
    dashboard_paths = dashboard_impl.generate_dashboards()
    
    if dashboard_paths:
        logger.info("Dashboard generation complete.")
        logger.info(f"Dashboards saved to: {output_dir}")
        
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