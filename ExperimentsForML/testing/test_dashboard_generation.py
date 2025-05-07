#!/usr/bin/env python3
"""
Test script for dashboard generation
This script creates dummy model results and tests the dashboard generation functionality
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the current directory to the path to find the MLPredictiveAnalysis module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import DashboardGenerator
from MLPredictiveAnalysis.funding_stage_prediction import DashboardGenerator

# Set up output directory
OUTPUT_DIR = "./test_dashboard_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

def create_dummy_data():
    """Create dummy classification data"""
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_classes=3, 
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    return X_train, X_test, y_train, y_test, feature_names

def train_models(X_train, X_test, y_train, y_test, feature_names):
    """Train dummy models and create metrics"""
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    
    # Train Logistic Regression (use as "LightGBM" for dashboard)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)
    
    # Create dummy results dictionary
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    results = {
        "Random Forest": {
            "accuracy": accuracy_score(y_test, rf_pred),
            "precision": precision_score(y_test, rf_pred, average='macro'),
            "recall": recall_score(y_test, rf_pred, average='macro'),
            "f1_scores": f1_score(y_test, rf_pred, average='macro'),
            "confusion_matrix": confusion_matrix(y_test, rf_pred),
            "y_test": y_test,
            "y_pred": rf_pred,
            "y_proba": rf_proba,
            "feature_names": feature_names,
            "feature_importance": rf.feature_importances_
        },
        "LightGBM": {  # Actually LogisticRegression, but named LightGBM for dashboard
            "accuracy": accuracy_score(y_test, lr_pred),
            "precision": precision_score(y_test, lr_pred, average='macro'),
            "recall": recall_score(y_test, lr_pred, average='macro'),
            "f1_scores": f1_score(y_test, lr_pred, average='macro'),
            "confusion_matrix": confusion_matrix(y_test, lr_pred),
            "y_test": y_test,
            "y_pred": lr_pred,
            "y_proba": lr_proba,
            "class_labels": ["Class 0", "Class 1", "Class 2"]
        },
        "Voting Ensemble": {  # Dummy ensemble (just copy RF results)
            "accuracy": accuracy_score(y_test, rf_pred),
            "precision": precision_score(y_test, rf_pred, average='macro'),
            "recall": recall_score(y_test, rf_pred, average='macro'),
            "f1_scores": f1_score(y_test, rf_pred, average='macro'),
            "confusion_matrix": confusion_matrix(y_test, rf_pred),
            "y_test": y_test,
            "y_pred": rf_pred,
            "y_proba": rf_proba
        }
    }
    
    return results

def create_dummy_timeseries_results():
    """Create dummy time series forecast results"""
    # Create dates from 2023-01-01 to 2023-12-31
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create dummy forecast data
    np.random.seed(42)
    values = np.cumsum(np.random.normal(0, 1, size=len(dates))) + 100
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'ds': dates,
        'yhat': values,
        'yhat_lower': values - 10,
        'yhat_upper': values + 10
    })
    
    # Create dummy historical data
    historical = pd.DataFrame({
        'ds': pd.date_range(start='2022-01-01', end='2022-12-31', freq='D'),
        'y': np.cumsum(np.random.normal(0, 1, size=365)) + 50
    })
    
    # Create dummy industry data
    industries = ['Tech', 'Finance', 'Healthcare', 'Retail', 'Manufacturing']
    industry_data = []
    
    for industry in industries:
        # Create random trend for each industry
        values = np.cumsum(np.random.normal(0, 0.5, size=12)) + 100 * (industries.index(industry) + 1)
        
        for i, month in enumerate(range(1, 13)):
            industry_data.append({
                'industry': industry,
                'month': f'2023-{month:02d}',
                'funding_amount': values[i],
                'deal_count': int(np.random.randint(5, 20))
            })
    
    industry_df = pd.DataFrame(industry_data)
    
    # Create dummy stage evolution data
    stages = ['Seed', 'Series A', 'Series B', 'Series C', 'Late Stage']
    stage_data = []
    
    for stage in stages:
        # Create random trend for each stage
        values = np.cumsum(np.random.normal(0, 0.5, size=12)) + 100 * (stages.index(stage) + 1)
        
        for i, month in enumerate(range(1, 13)):
            stage_data.append({
                'funding_stage': stage,
                'month': f'2023-{month:02d}',
                'funding_amount': values[i],
                'deal_count': int(np.random.randint(5, 20))
            })
    
    stage_df = pd.DataFrame(stage_data)
    
    # Bundle all results
    timeseries_results = {
        'forecast': forecast_df,
        'historical': historical,
        'industry_data': industry_df,
        'stage_evolution': stage_df,
        'components': {
            'trend': forecast_df['yhat'],
            'seasonal': np.sin(np.linspace(0, 12*np.pi, len(dates)))
        }
    }
    
    return timeseries_results

def test_dashboard_generation():
    """Test the dashboard generation functionality"""
    print("Creating dummy data...")
    X_train, X_test, y_train, y_test, feature_names = create_dummy_data()
    
    print("Training models...")
    classification_results = train_models(X_train, X_test, y_train, y_test, feature_names)
    
    print("Creating time series results...")
    timeseries_results = create_dummy_timeseries_results()
    
    print("Initializing dashboard generator...")
    dashboard_generator = DashboardGenerator(output_dir=OUTPUT_DIR)
    
    print("Generating dashboards...")
    dashboard_paths = dashboard_generator.generate_all_dashboards(
        classification_results, 
        timeseries_results
    )
    
    print("Dashboard generation complete!")
    print("Dashboard paths:")
    for dashboard_type, paths in dashboard_paths.items():
        print(f"- {dashboard_type}:")
        if isinstance(paths, dict):
            for name, path in paths.items():
                print(f"  - {name}: {path}")
        else:
            print(f"  - {paths}")

if __name__ == "__main__":
    try:
        test_dashboard_generation()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error in test: {str(e)}")
        import traceback
        traceback.print_exc() 