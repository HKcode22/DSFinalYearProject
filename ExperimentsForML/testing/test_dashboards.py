#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Advanced Dashboard Generator.
This script creates mock data and uses it to test the advanced dashboards.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the dashboard generator
from funding_stage_prediction import AdvancedDashboardGenerator

def run_dashboard_test():
    """Run a test of the Advanced Dashboard Generator with mock data"""
    try:
        logger.info("Testing Advanced Dashboard Generator with mock data")
        
        # Create output directory
        output_dir = './advanced_dashboards_test'
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the dashboard generator
        advanced_generator = AdvancedDashboardGenerator(output_dir)
        
        # Create mock classification results
        mock_classification_results = {
            'Random Forest': (None, {
                'accuracy': 0.85,
                'precision': {'macro': 0.83},
                'recall': {'macro': 0.82},
                'f1_scores': {'macro': 0.825},
                'confusion_matrix': np.array([
                    [45, 5, 0, 0],
                    [7, 38, 5, 0],
                    [0, 4, 36, 10],
                    [0, 0, 8, 42]
                ]),
                'classes': ['Seed', 'Series A', 'Series B', 'Series C']
            }, None, {
                'features': ['funding_amount_log', 'employees', 'employee_efficiency', 'funding_velocity'],
                'importance': [0.45, 0.25, 0.15, 0.15]
            }),
            
            'XGBoost': (None, {
                'accuracy': 0.87,
                'precision': {'macro': 0.86},
                'recall': {'macro': 0.84},
                'f1_scores': {'macro': 0.85},
                'confusion_matrix': np.array([
                    [47, 3, 0, 0],
                    [5, 41, 4, 0],
                    [0, 2, 39, 9],
                    [0, 0, 5, 45]
                ]),
                'classes': ['Seed', 'Series A', 'Series B', 'Series C']
            }, None, {
                'features': ['funding_amount_log', 'employees', 'employee_efficiency', 'funding_velocity'],
                'importance': [0.40, 0.30, 0.20, 0.10]
            })
        }
        
        # Create mock time series results
        # Generate dates from 2018 to 2025
        dates = pd.date_range(start='2018-01-01', end='2025-12-31', freq='M')
        n_dates = len(dates)
        
        # Split into historical and future
        historical_cutoff = '2023-01-01'
        historical_mask = dates < historical_cutoff
        
        # Create forecast dataframe with Prophet-like structure
        mock_forecast_df = pd.DataFrame({
            'ds': dates,
            'trend': np.linspace(5, 15, n_dates) + np.random.normal(0, 0.5, n_dates),
            'yhat': np.linspace(5, 15, n_dates) * (1 + 0.1 * np.sin(np.linspace(0, 8 * np.pi, n_dates))) + np.random.normal(0, 1, n_dates),
            'yhat_lower': np.linspace(4, 14, n_dates) * (1 + 0.1 * np.sin(np.linspace(0, 8 * np.pi, n_dates))) - 2 + np.random.normal(0, 0.5, n_dates),
            'yhat_upper': np.linspace(6, 16, n_dates) * (1 + 0.1 * np.sin(np.linspace(0, 8 * np.pi, n_dates))) + 2 + np.random.normal(0, 0.5, n_dates),
            'yearly': 1.5 * np.sin(np.linspace(0, 8 * np.pi, n_dates)),
            'monthly': 0.5 * np.sin(np.linspace(0, 50 * np.pi, n_dates)),
        })
        
        # Add historical flag and actual values for historical portion
        mock_forecast_df['historical'] = historical_mask
        mock_forecast_df['y'] = np.nan
        mock_forecast_df.loc[historical_mask, 'y'] = mock_forecast_df.loc[historical_mask, 'yhat'] + np.random.normal(0, 0.8, historical_mask.sum())
        
        # Create industry-specific forecasts with variations
        mock_timeseries_results = {
            'overall_forecast': mock_forecast_df,
            'software_forecast': mock_forecast_df.copy(),
            'fintech_forecast': mock_forecast_df.copy(),
            'healthcare_forecast': mock_forecast_df.copy()
        }
        
        # Add some industry-specific variations
        mock_timeseries_results['software_forecast']['yhat'] *= 1.2
        mock_timeseries_results['software_forecast']['yhat_lower'] *= 1.2
        mock_timeseries_results['software_forecast']['yhat_upper'] *= 1.2
        
        mock_timeseries_results['fintech_forecast']['yhat'] *= 0.9
        mock_timeseries_results['fintech_forecast']['yhat_lower'] *= 0.9
        mock_timeseries_results['fintech_forecast']['yhat_upper'] *= 0.9
        
        mock_timeseries_results['healthcare_forecast']['yhat'] *= 1.1
        mock_timeseries_results['healthcare_forecast']['yhat_lower'] *= 1.1
        mock_timeseries_results['healthcare_forecast']['yhat_upper'] *= 1.1
        
        # Generate the advanced dashboards
        logger.info("Generating advanced dashboards...")
        advanced_paths = advanced_generator.generate_advanced_dashboards(
            mock_classification_results, mock_timeseries_results)
        
        logger.info(f"Generated advanced dashboards at: {output_dir}")
        for dashboard_type, paths in advanced_paths.items():
            if paths:
                logger.info(f"- Advanced {dashboard_type} dashboards:")
                for subtype, path in paths.items():
                    if path:
                        logger.info(f"  - {subtype}: {path}")
        
        logger.info("Dashboard generation test completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error in dashboard test: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    run_dashboard_test() 