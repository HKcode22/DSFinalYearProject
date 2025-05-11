import funding_stage_prediction as fsp
import pandas as pd
import numpy as np
import os
import logging
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Add the directory to the path so we can import the funding_stage_prediction module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module
from MLPredictiveAnalysis.funding_stage_prediction9 import DashboardGenerator

def generate_mock_data():
    """Generate mock data for testing the dashboard"""
    # Create features with realistic names
    n_samples = 100
    data = {
        'funding_amount': np.random.lognormal(12, 1, n_samples),  # Realistic funding amounts
        'employees': np.random.randint(5, 500, n_samples),
        'industry': np.random.choice(['Tech', 'Biotech', 'Fintech', 'E-commerce', 'Healthcare'], n_samples),
        'total_funding': np.random.lognormal(12.5, 1.5, n_samples),
        'founding_year': np.random.randint(2010, 2023, n_samples),
        'founder_experience': np.random.randint(0, 20, n_samples),
        'burn_rate': np.random.lognormal(10, 1, n_samples),
        'growth_rate': np.random.normal(0.2, 0.1, n_samples),
        'revenue': np.random.lognormal(11, 2, n_samples),
        'user_count': np.random.randint(1000, 1000000, n_samples),
        'is_tech_company': np.random.choice([0, 1], n_samples),
        'funding_velocity': np.random.normal(0.5, 0.3, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['industry'], drop_first=False)
    
    # Create target variable - funding stages
    stages = ['Seed', 'Series A', 'Series B', 'Series C', 'Series D']
    y = pd.Series(np.random.choice(stages, n_samples))
    
    return df_encoded, y, df_encoded.columns.tolist()

def generate_time_series_data():
    """Generate mock time series data for testing the dashboard"""
    # Create 3 years of monthly data
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='M')
    
    # Create funding amounts with seasonality and trend
    n = len(dates)
    trend = np.linspace(500000, 5000000, n)  # Increasing trend
    seasonality = 500000 * np.sin(np.linspace(0, 6*np.pi, n))  # Seasonal component
    noise = np.random.normal(0, 200000, n)  # Random noise
    
    # Combine components
    funding_amounts = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': dates,
        'y': funding_amounts
    })
    
    return df

def test_timeseries_dashboard():
    """Test the generation of the time series dashboard"""
    logger.info("Testing time series dashboard generation")
    
    # Generate time series data
    historical_data = generate_time_series_data()
    
    # Create dashboard generator
    output_dir = os.path.join(os.getcwd(), "output", "dashboards")
    generator = DashboardGenerator(output_dir=output_dir)
    
    # Generate time series dashboard
    generator.generate_timeseries_dashboard(historical_data, None)
    
    print(f"Time series dashboard generated at {os.path.join(output_dir, 'timeseries')}")
    return True

def test_classification_dashboard():
    """Test the generation of the classification dashboard"""
    logger.info("Testing classification dashboard generation")
    
    # Create mock data
    X, y, features = generate_mock_data()
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions for test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Create mock model results
    model_results = {
        'Random Forest': {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.82,
            'f1': 0.82,
            'confusion_matrix': np.array([[10, 2, 0, 0, 0], 
                                         [1, 8, 1, 0, 0],
                                         [0, 2, 7, 1, 0],
                                         [0, 0, 1, 5, 1],
                                         [0, 0, 0, 2, 4]]),
            'classes': ['Seed', 'Series A', 'Series B', 'Series C', 'Series D']
        },
        'XGBoost': {
            'accuracy': 0.88,
            'precision': 0.87,
            'recall': 0.86,
            'f1': 0.86,
            'confusion_matrix': np.array([[11, 1, 0, 0, 0], 
                                         [2, 7, 1, 0, 0],
                                         [0, 1, 8, 1, 0],
                                         [0, 0, 0, 6, 1],
                                         [0, 0, 0, 1, 5]]),
            'classes': ['Seed', 'Series A', 'Series B', 'Series C', 'Series D']
        },
        'y_true': y_test,
        'y_proba': y_proba
    }
    
    # Create dashboard generator
    output_dir = os.path.join(os.getcwd(), "output", "dashboards")
    generator = DashboardGenerator(output_dir=output_dir)
    
    # Create models dictionary
    models = {
        'Random Forest': model,
        'best_model': model
    }
    
    # Generate classification dashboard
    generator.generate_classification_dashboard(
        model_results=model_results,
        feature_names=features,
        X_sample=X_test.iloc[:5],
        y_sample=y_test.iloc[:5],
        models=models
    )
    
    print(f"Classification dashboard generated at {os.path.join(output_dir, 'classification')}")
    return True

if __name__ == "__main__":
    print("Testing dashboard generation...")
    
    # Test classification dashboard
    print("\nTesting classification dashboard...")
    if test_classification_dashboard():
        print("Classification dashboard test successful!")
    else:
        print("Classification dashboard test failed!")
    
    # Test time series dashboard
    print("\nTesting time series dashboard...")
    if test_timeseries_dashboard():
        print("Time series dashboard test successful!")
    else:
        print("Time series dashboard test failed!")
    
    logger.info("All dashboard tests completed successfully") 