import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from funding_amount_forecast import FundingAmountForecast
import os
from sklearn.ensemble import RandomForestRegressor

def test_prediction_intervals():
    # Create test data
    X_test = np.random.rand(20, 5)
    y_test = np.random.rand(20) * 10
    
    # Create and train models for different quantiles
    models = {
        0.1: RandomForestRegressor(),
        0.5: RandomForestRegressor(),
        0.9: RandomForestRegressor()
    }
    
    for q in models:
        # Adjust predictions based on quantile
        multiplier = 0.8 if q == 0.1 else 1.0 if q == 0.5 else 1.2
        models[q].fit(X_test, y_test * multiplier)
    
    # Create output directory
    os.makedirs('test_output', exist_ok=True)
    
    # Initialize FundingAmountForecast
    faf = FundingAmountForecast()
    
    # Test the visualize_prediction_intervals method
    try:
        output_path = faf.visualize_prediction_intervals(X_test, y_test, models, output_dir='test_output')
        print(f'Test completed successfully. Output saved to {output_path}')
        return True
    except Exception as e:
        print(f'Test failed with error: {e}')
        return False

if __name__ == "__main__":
    test_prediction_intervals() 