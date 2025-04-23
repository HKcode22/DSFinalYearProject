import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import tempfile
import os
from MLPredictiveAnalysis.data_loader import FundingDataLoader

class TestFundingDataLoader(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample test data
        self.test_data = [
            {
                "company_name": "TestCo1",
                "funding_date": "2022-01-01",
                "funding_amount": "1M",
                "funding_stage": "Series A",
                "industry": "Tech"
            },
            {
                "company_name": "TestCo2",
                "funding_date": "2022-02-01",
                "funding_amount": "2.5M",
                "funding_stage": "Seed",
                "industry": "Tech"
            }
        ]
        
        # Write test data to file
        with open(os.path.join(self.temp_dir, "test_data.json"), "w") as f:
            json.dump(self.test_data, f)
            
        self.data_loader = FundingDataLoader(self.temp_dir)
    
    def tearDown(self):
        # Clean up temporary directory
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_load_json_file(self):
        df = self.data_loader.load_json_file("test_data.json")
        self.assertEqual(len(df), 2)
        self.assertIn("company_name", df.columns)
        self.assertIn("funding_amount", df.columns)
    
    def test_standardize_funding_amount(self):
        test_cases = [
            ("1M", 1.0),
            ("2.5B", 2500.0),
            ("500K", 0.5),
            ("invalid", np.nan)
        ]
        
        for input_str, expected in test_cases:
            result = self.data_loader.standardize_funding_amount(input_str)
            if np.isnan(expected):
                self.assertTrue(np.isnan(result))
            else:
                self.assertEqual(result, expected)
    
    def test_engineer_features(self):
        # Create sample DataFrame
        data = {
            'company_name': ['TestCo1', 'TestCo1', 'TestCo2'],
            'funding_date': pd.to_datetime(['2022-01-01', '2022-06-01', '2022-01-01']),
            'funding_amount': [1.0, 2.0, 1.5],
            'funding_stage': ['Seed', 'Series A', 'Seed'],
            'industry': ['Tech', 'Tech', 'Biotech']
        }
        df = pd.DataFrame(data)
        
        # Engineer features
        result_df = self.data_loader.engineer_features(df)
        
        # Check if new features are present
        expected_features = [
            'funding_year', 'funding_quarter', 'days_since_last_funding',
            'funding_stage_number', 'is_early_stage', 'funding_amount_log',
            'cumulative_funding', 'funding_sequence', 'funding_count',
            'funding_growth', 'time_between_rounds', 'industry_funding_mean'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result_df.columns)
        
        # Check specific feature values
        self.assertEqual(result_df['funding_year'].iloc[0], 2022)
        self.assertEqual(result_df['funding_count'].iloc[0], 2)  # TestCo1 has 2 rounds
    
    def test_merge_dataframes(self):
        # Create two sample DataFrames
        df1 = pd.DataFrame({
            'company_name': ['TestCo1'],
            'funding_date': pd.to_datetime(['2022-01-01']),
            'funding_amount': [1.0]
        })
        
        df2 = pd.DataFrame({
            'company_name': ['TestCo1', 'TestCo2'],
            'funding_date': pd.to_datetime(['2022-01-01', '2022-02-01']),
            'funding_amount': [1.0, 2.0]
        })
        
        # Merge DataFrames
        result_df = self.data_loader.merge_dataframes([df1, df2])
        
        # Check results
        self.assertEqual(len(result_df), 2)  # Should have 2 unique records
        self.assertTrue(all(result_df['company_name'].isin(['TestCo1', 'TestCo2'])))

if __name__ == '__main__':
    unittest.main() 