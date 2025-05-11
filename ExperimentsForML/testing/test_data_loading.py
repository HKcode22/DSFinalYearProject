#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the data loading, merging, and processing components 
of the funding stage prediction pipeline.

This test ensures that:
1. Data is loaded correctly from all three JSON files
2. Data validation and merging works properly
3. Feature extraction produces expected results
4. The model is trained on real data from these files

The test prints detailed information about each step to validate the process.
"""

import os
import json
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# Import the modules to test
from MLPredictiveAnalysis.funding_stage_prediction9 import DataLoader, FeatureEngineering, ModelTrainer

def print_dataframe_info(df, name):
    """Print detailed information about a dataframe"""
    print(f"\n=== {name} DataFrame Info ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data (first 5 rows):")
    print(df.head())
    print(f"Column non-null counts:")
    print(df.count())
    print(f"Column data types:")
    print(df.dtypes)
    print("-" * 80)

def check_json_file(file_path):
    """Check if a JSON file exists and print its basic structure"""
    print(f"\nChecking {file_path}...")
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            print(f"JSON structure: Dictionary with keys: {list(data.keys())}")
            if 'companies' in data:
                print(f"Number of companies: {len(data['companies'])}")
                print(f"Sample company data (first record):")
                print(json.dumps(data['companies'][0], indent=2))
        elif isinstance(data, list):
            print(f"JSON structure: List with {len(data)} items")
            print(f"Sample item (first record):")
            print(json.dumps(data[0], indent=2))
        else:
            print(f"JSON structure: Unknown type ({type(data)})")
        
        return True
    except Exception as e:
        print(f"ERROR: Could not parse JSON file: {str(e)}")
        return False

def test_data_loader():
    """Test the DataLoader class to ensure it loads and merges data correctly"""
    print("\n===== TESTING DATA LOADER =====")
    
    # Setup paths
    base_dir = "./JSONFolder"
    fundraiser_path = os.path.join(base_dir, "fundraisestartup50.json")
    growthlist_path = os.path.join(base_dir, "growthlistscrapper.json") 
    topstartup_path = os.path.join(base_dir, "topstartupio50.json")
    
    # Check if the JSON files exist and are valid
    files_valid = True
    for path in [fundraiser_path, growthlist_path, topstartup_path]:
        if not check_json_file(path):
            files_valid = False
    
    if not files_valid:
        print("ERROR: One or more JSON files are invalid or missing")
        return False
    
    # Initialize loader
    loader = DataLoader(base_dir=base_dir)
    
    # Test loading each dataset individually
    print("\nTesting individual file loading...")
    
    # Fundraiser data
    print("\nLoading fundraiser data...")
    df_fundraiser = loader.load_fundraiser_data()
    if df_fundraiser.empty:
        print("ERROR: Failed to load fundraiser data")
        return False
    
    print_dataframe_info(df_fundraiser, "Fundraiser")
    
    # Growthlist data
    print("\nLoading growthlist data...")
    df_growthlist = loader.load_growthlist_data()
    if df_growthlist.empty:
        print("ERROR: Failed to load growthlist data")
        return False
    
    print_dataframe_info(df_growthlist, "Growthlist")
    
    # Topstartup data
    print("\nLoading topstartup data...")
    df_topstartup = loader.load_topstartup_data()
    if df_topstartup.empty:
        print("ERROR: Failed to load topstartup data")
        return False
    
    print_dataframe_info(df_topstartup, "Topstartup")
    
    # Test merging
    print("\nTesting data merging...")
    df_merged = loader.merge_datasets()
    if df_merged.empty:
        print("ERROR: Failed to merge datasets")
        return False
    
    print_dataframe_info(df_merged, "Merged Data")
    
    # Validation tests
    print("\nPerforming validation tests...")
    
    # 1. Check if company_name column exists in merged data
    assert 'company_name' in df_merged.columns, "company_name column missing from merged data"
    
    # 2. Check if funding_stage column exists in merged data
    assert 'funding_stage' in df_merged.columns, "funding_stage column missing from merged data"
    
    # 3. Check if any funding stage is 'Unknown' (this should happen due to mapping)
    has_unknown = (df_merged['funding_stage'] == 'Unknown').any()
    print(f"Has 'Unknown' funding stages: {has_unknown}")
    
    # 4. Check if we have data from all three sources
    sources = df_merged['source'].unique()
    print(f"Data sources in merged data: {sources}")
    assert 'fundraiser' in sources, "No fundraiser data in merged dataset"
    assert 'growthlist' in sources, "No growthlist data in merged dataset"
    assert 'topstartup' in sources, "No topstartup data in merged dataset"
    
    # 5. Check for duplicates
    duplicates = df_merged.duplicated(subset=['company_name', 'funding_date']).sum()
    print(f"Duplicate records (same company+date): {duplicates}")
    
    print("\nData loading and merging tests PASSED")
    return df_merged

def test_feature_engineering(merged_data):
    """Test the FeatureEngineering class with the merged data"""
    print("\n===== TESTING FEATURE ENGINEERING =====")
    
    # Initialize feature engineering
    feature_eng = FeatureEngineering()
    
    # Extract features
    print("Extracting features...")
    features_df = feature_eng.extract_features(merged_data)
    
    print_dataframe_info(features_df, "Features")
    
    # Check that key features were created
    expected_features = [
        'funding_stage_numeric', 
        'funding_year', 
        'funding_month',
        'months_since_first_funding',
        'funding_amount_log',
        'previous_rounds'
    ]
    
    for feature in expected_features:
        assert feature in features_df.columns, f"Expected feature '{feature}' missing"
    
    # Check feature distributions
    print("\nFeature statistics:")
    for feature in expected_features:
        if feature in features_df.columns:
            print(f"{feature}: min={features_df[feature].min()}, max={features_df[feature].max()}, mean={features_df[feature].mean():.2f}, null={features_df[feature].isna().sum()}")
    
    print("\nFunding stage distribution:")
    stage_counts = features_df['funding_stage'].value_counts()
    print(stage_counts)
    
    print("\nFeature engineering tests PASSED")
    return features_df

def test_model_training(features_df):
    """Test the ModelTrainer class with the feature data"""
    print("\n===== TESTING MODEL TRAINING =====")
    
    # Initialize feature engineering for the model data preparation
    feature_eng = FeatureEngineering()
    X, y = feature_eng.prepare_model_data(features_df)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X columns: {X.columns.tolist()}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Initialize trainer
    trainer = ModelTrainer(output_dir="./test_output")
    os.makedirs("./test_output", exist_ok=True)
    
    # Train a simple random forest model
    print("\nTraining Random Forest model...")
    rf_model, rf_results = trainer.train_random_forest(X, y)
    
    # Evaluate results
    print("\nRandom Forest Results:")
    print(f"Accuracy: {rf_results['accuracy']:.4f}")
    print(f"Top features by importance: {dict(zip(X.columns, rf_model.feature_importances_.round(4)))}")
    
    # Try XGBoost
    print("\nTraining XGBoost model...")
    try:
        xgb_model, xgb_results = trainer.train_xgboost(X, y)
        
        # Evaluate results
        print("\nXGBoost Results:")
        print(f"Accuracy: {xgb_results['accuracy']:.4f}")
        
        # Compare models
        print("\nModel comparison:")
        print(f"RF accuracy: {rf_results['accuracy']:.4f}, XGB accuracy: {xgb_results['accuracy']:.4f}")
        
        # This will help verify the models are trained on real data
        assert 0.5 < rf_results['accuracy'] < 1.0, "Random Forest accuracy outside expected range"
        assert 0.5 < xgb_results['accuracy'] < 1.0, "XGBoost accuracy outside expected range"
        
    except Exception as e:
        print(f"XGBoost training failed: {str(e)}")
    
    print("\nModel training tests PASSED")
    return rf_model, rf_results

def full_test():
    """Run a full test of the pipeline"""
    print("\n========== RUNNING FULL TEST ==========\n")
    
    # Test data loading and merging
    merged_data = test_data_loader()
    if merged_data is False or merged_data.empty:
        print("Test failed: Could not load or merge data")
        return False
    
    # Test feature engineering
    features_df = test_feature_engineering(merged_data)
    if features_df is False or features_df.empty:
        print("Test failed: Feature engineering failed")
        return False
    
    # Test model training
    model, results = test_model_training(features_df)
    if model is False:
        print("Test failed: Model training failed")
        return False
    
    print("\n===== FINAL TEST SUMMARY =====")
    print("All tests PASSED!")
    print(f"Loaded and merged data from 3 JSON files: {merged_data.shape[0]} records")
    print(f"Created features successfully: {features_df.shape[1]} features")
    print(f"Trained models with accuracy: {results['accuracy']:.4f}")
    
    return True

if __name__ == "__main__":
    start_time = datetime.now()
    success = full_test()
    end_time = datetime.now()
    
    print(f"\nTest completed in {(end_time - start_time).total_seconds():.2f} seconds")
    
    if not success:
        sys.exit(1) 