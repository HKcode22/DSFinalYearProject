#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from MLPredictiveAnalysis.funding_amount_forecast import FundingAmountForecast

def copy_json_files_to_test_dir(test_data_dir):
    """Copy the JSON files to the test directory."""
    # Create JSONFolder in test directory
    test_json_folder = os.path.join(test_data_dir, "JSONFolder")
    os.makedirs(test_json_folder, exist_ok=True)
    
    # Source files
    source_dir = "JSONFolder"
    required_files = [
        "fundraisestartup50.json",
        "growthlistscrapper.json",
        "topstartupio50.json"
    ]
    
    # Copy files
    for file in required_files:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(test_json_folder, file)
        
        if os.path.exists(source_path):
            print(f"Copying {file} to test directory...")
            shutil.copy2(source_path, dest_path)
        else:
            print(f"Warning: {file} not found in {source_dir}")
    
    return [os.path.exists(os.path.join(test_json_folder, file)) for file in required_files]

def setup_test_environment():
    """Set up a test environment with the actual JSON data files."""
    # Create test directories
    test_data_dir = "test_data"
    test_output_dir = "test_output"
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Copy JSON files to test directory
    files_exist = copy_json_files_to_test_dir(test_data_dir)
    
    if not all(files_exist):
        print("Warning: Not all required JSON files were found or copied successfully.")
    
    return test_data_dir, test_output_dir

def test_funding_forecast_pipeline():
    """Test the full funding forecast pipeline with actual data."""
    print("Setting up test environment...")
    test_data_dir, test_output_dir = setup_test_environment()
    
    # Initialize forecaster
    print("Initializing funding amount forecaster...")
    forecaster = FundingAmountForecast(
        data_dir=test_data_dir,
        output_dir=test_output_dir
    )
    
    # Test loading data
    print("Testing data loading...")
    data = forecaster.load_data_from_json_files()
    
    if data is None or data.empty:
        print("❌ Failed to load data from JSON files")
        return False
        
    print(f"✅ Successfully loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Columns: {data.columns.tolist()}")
    
    # Check required columns
    required_cols = ['company_name', 'funding_stage', 'funding_amount_numeric', 'funding_date']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
        
    print("✅ All required columns present")
    
    # Check data quality
    null_counts = data[required_cols].isnull().sum()
    print(f"Null value counts in required columns:")
    for col, count in null_counts.items():
        print(f"  - {col}: {count} null values ({count/len(data)*100:.2f}%)")
    
    # Run the full analysis
    print("\nRunning full analysis pipeline...")
    
    try:
        forecaster.run_analysis()
        print("\n✅ Full pipeline test completed successfully!")
        
        # Check if output files were generated
        output_files = os.listdir(test_output_dir)
        if output_files:
            print(f"Generated output files: {', '.join(output_files)}")
        else:
            print("No output files were generated.")
            
        return True
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_create_forecast_features():
    """Test the feature creation functionality."""
    print("\nTesting feature creation...")
    
    # Initialize forecaster
    forecaster = FundingAmountForecast(
        data_dir="test_data",
        output_dir="test_output"
    )
    
    # Load data
    data = forecaster.load_data_from_json_files()
    
    if data is None or data.empty:
        print("❌ Cannot test feature creation - no data available")
        return False
    
    # Create features
    try:
        features_df = forecaster.create_forecast_features(data)
        
        # Check if feature creation was successful
        if features_df is None or features_df.empty:
            print("❌ Feature creation failed - empty result")
            return False
            
        # Check for expected feature columns
        expected_features = [
            'industry_avg_funding', 'expected_jump_factor', 'market_growth_rate'
        ]
        
        missing_features = [f for f in expected_features if f not in features_df.columns]
        
        if missing_features:
            print(f"❌ Missing expected features: {missing_features}")
            return False
            
        print(f"✅ Successfully created features: {features_df.shape[0]} rows, {features_df.shape[1]} columns")
        print(f"Feature columns: {[col for col in features_df.columns if col not in data.columns]}")
        
        return True
    except Exception as e:
        print(f"❌ Feature creation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_prepare_model_data():
    """Test the model data preparation functionality."""
    print("\nTesting model data preparation...")
    
    # Initialize forecaster
    forecaster = FundingAmountForecast(
        data_dir="test_data",
        output_dir="test_output"
    )
    
    # Load data
    data = forecaster.load_data_from_json_files()
    
    if data is None or data.empty:
        print("❌ Cannot test model data preparation - no data available")
        return False
    
    # Create features
    features_df = forecaster.create_forecast_features(data)
    
    if features_df is None or features_df.empty:
        print("❌ Cannot test model data preparation - feature creation failed")
        return False
    
    # Prepare model data
    try:
        X, y = forecaster.prepare_model_data(features_df)
        
        if X is None or y is None:
            print("❌ Model data preparation failed - empty result")
            return False
            
        print(f"✅ Successfully prepared model data:")
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        print(f"  - X columns: {X.columns.tolist()}")
        
        return True
    except Exception as e:
        print(f"❌ Model data preparation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_for_new_company():
    """Test making predictions for new companies."""
    print("\nTesting prediction for new company...")
    
    # Initialize forecaster using existing test data
    forecaster = FundingAmountForecast(
        data_dir="test_data",
        output_dir="test_output"
    )
    
    # Load model if available, otherwise train one
    try:
        forecaster.load_model("test_output/funding_model.pkl")
        print("Loaded existing model")
    except:
        print("No existing model found, attempting to train a new one...")
        # We need to run the pipeline first
        data = forecaster.load_data_from_json_files()
        if data is None or data.empty:
            print("❌ Cannot test prediction - no data available")
            return False
            
        features_df = forecaster.create_forecast_features(data)
        X, y = forecaster.prepare_model_data(features_df)
        
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            print("❌ Cannot test prediction - model preparation failed (no data points)")
            return False
            
        # Train a simple model
        try:
            model, _ = forecaster.train_quantile_regression_forest(X, y)
            forecaster.model = model
            forecaster.save_model("test_output/funding_model.pkl")
        except Exception as e:
            print(f"❌ Cannot test prediction - model training failed: {str(e)}")
            return False
    
    # Create test data for a new company with correct datetime format
    new_company_data = pd.DataFrame({
        'company_name': ['NewAIStartup'],
        'funding_stage_standard': ['Seed'],
        'funding_amount': ['$1,500,000'],
        'funding_date': pd.to_datetime(['2024-01-01']),  # Convert to datetime object
        'industry': ['Artificial Intelligence'],
        'country': ['United States'],
        'funding_amount_numeric': [1500000],
        'funding_stage_level': [1]  # Seed = 1
    })
    
    # Add needed features
    try:
        features_df = forecaster.create_forecast_features(new_company_data)
        
        # Prepare for prediction (only features needed for the model)
        required_features = [
            'funding_stage_level', 'expected_jump_factor',
            'industry_avg_funding', 'region_avg_funding', 'market_growth_rate'
        ]
        
        # Only use features that exist in the data
        pred_features = [f for f in required_features if f in features_df.columns]
        
        if not pred_features:
            print("❌ Prediction failed - no required features available")
            print(f"Available features: {features_df.columns.tolist()}")
            return False
            
        X_pred = features_df[pred_features]
        
        # Make prediction
        predicted_amount = forecaster.predict_funding_amount(X_pred)
        
        # Convert log-transformed prediction back to original scale
        predicted_amount_original = np.expm1(predicted_amount[0])
        
        print(f"✅ Successfully made prediction:")
        print(f"For a Seed stage AI company with previous funding of $1.5M:")
        print(f"Predicted next funding amount: ${predicted_amount_original:,.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Prediction failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Funding Amount Forecast Pipeline ===")
    
    # Test each component
    data_loading_success = test_funding_forecast_pipeline()
    
    if data_loading_success:
        feature_creation_success = test_create_forecast_features()
        
        if feature_creation_success:
            model_prep_success = test_prepare_model_data()
            
            # Only test prediction if previous steps succeeded
            if model_prep_success:
                test_prediction_for_new_company()
    
    print("\nTest complete!") 