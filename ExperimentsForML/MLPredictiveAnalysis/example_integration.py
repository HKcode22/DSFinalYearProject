import os
import pandas as pd
import numpy as np
import json
from funding_anomaly_detection import FundingAnomalyDetection, integrate_with_meta_model

def create_mock_stage_prediction_data():
    """Create mock data for funding stage prediction"""
    # Create a simple DataFrame with stage predictions
    data = {
        'company_name': [
            'Anthropic', 'Mercury', 'Peregrine Technologies', 'Sayso',
            'Safely You', 'Augment', 'Bitwise', 'CompScience', 'DeepNight'
        ],
        'predicted_stage': [
            'Series E', 'Series C', 'Series C', 'Seed',
            'Series C', 'Seed', 'Series B', 'Series B', 'Seed'
        ],
        'stage_probability': [
            0.95, 0.88, 0.92, 0.75,
            0.80, 0.78, 0.65, 0.82, 0.90
        ]
    }
    
    return pd.DataFrame(data)

def create_mock_continuation_data():
    """Create mock data for funding continuation analysis"""
    # Create a simple DataFrame with continuation predictions
    data = {
        'company_name': [
            'Anthropic', 'Mercury', 'Peregrine Technologies', 'Sayso',
            'Safely You', 'Augment', 'Bitwise', 'CompScience', 'DeepNight'
        ],
        'survival_probability': [
            0.85, 0.78, 0.82, 0.65,
            0.60, 0.70, 0.75, 0.80, 0.72
        ],
        'expected_duration': [
            450, 380, 320, 270,
            310, 290, 340, 365, 280
        ]
    }
    
    return pd.DataFrame(data)

def main():
    """Run example integration"""
    print("Running Example Integration of Anomaly Detection with Other Components")
    print("=" * 70)
    
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), 'JSONFolder')
    output_dir = os.path.join(base_dir, 'output', 'integrated_analysis')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run anomaly detection
    print("\n1. Running Anomaly Detection...")
    anomaly_detector = FundingAnomalyDetection(data_dir=data_dir, output_dir=output_dir)
    anomaly_results = anomaly_detector.run_analysis()
    
    # Create mock data for other components
    print("\n2. Loading funding stage prediction results...")
    stage_results = create_mock_stage_prediction_data()
    
    print("\n3. Loading funding continuation analysis results...")
    continuation_results = create_mock_continuation_data()
    
    # Integrate results
    print("\n4. Integrating results for meta-model...")
    integrated_results = integrate_with_meta_model(
        anomaly_results,
        stage_results,
        continuation_results
    )
    
    # Save integrated results
    integrated_results.to_csv(os.path.join(output_dir, 'integrated_results.csv'), index=False)
    
    # Display example results
    print("\n5. Example Integrated Results:")
    print("-" * 70)
    print(integrated_results.head(5).to_string())
    print("-" * 70)
    
    # Calculate success rates
    success_count = integrated_results['predicted_success'].sum()
    total_count = len(integrated_results)
    success_rate = success_count / total_count * 100
    
    print(f"\nOverall Success Prediction Rate: {success_rate:.2f}%")
    print(f"Total Companies: {total_count}")
    print(f"Predicted Successful: {success_count}")
    print(f"Predicted Unsuccessful: {total_count - success_count}")
    
    # Example of using the predict_for_new_company method
    print("\n6. Example Prediction for a New Company:")
    new_company = {
        'name': 'AI Startup X',
        'industry': 'Artificial Intelligence, Cloud Computing',
        'funding_amount': '$45,000,000',
        'funding_type': 'Series A',
        'last_funding_date': 'Apr 2025'
    }
    
    prediction = anomaly_detector.predict_for_new_company(new_company)
    print(f"Company: {new_company['name']}")
    print(f"Is Anomaly: {prediction['is_anomaly']}")
    print(f"Anomaly Severity: {prediction['anomaly_severity']:.4f}")
    if 'anomaly_type' in prediction and prediction['anomaly_type']:
        print(f"Anomaly Type: {prediction['anomaly_type']}")
    print(f"Explanation: {prediction['explanation']}")
    
    print("\nIntegrated analysis complete. Results saved to:", output_dir)

if __name__ == "__main__":
    main() 