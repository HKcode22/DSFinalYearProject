#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify prediction functionality and model validity.
This will confirm that the model predictions are reasonable and not fabricated.
"""

import os
import sys
import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime, timedelta

# Add the project directory to the path to import the module
sys.path.append("/Users/hk/Downloads/DSFinalYearProject")
from MLPredictiveAnalysis.funding_stage_prediction9 import EnhancedPipeline, DataLoader, ModelManager, FeatureEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("test_predictions")

def generate_test_company(name_prefix="Test Company", variation=None):
    """Generate a test company with controlled parameters for prediction testing."""
    companies = [
        {
            "company_name": f"{name_prefix} Early Seed",
            "industry": "Artificial Intelligence",
            "funding_amount": "$500,000",
            "funding_stage": "Pre-Seed",
            "employees": 5,
            "funding_date": (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        },
        {
            "company_name": f"{name_prefix} Mid Seed",
            "industry": "Software",
            "funding_amount": "$1,500,000",
            "funding_stage": "Seed",
            "employees": 12,
            "funding_date": (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
        },
        {
            "company_name": f"{name_prefix} Series A Ready",
            "industry": "Financial Technology",
            "funding_amount": "$4,000,000",
            "funding_stage": "Seed",
            "employees": 25,
            "funding_date": (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        },
        {
            "company_name": f"{name_prefix} Mid Growth",
            "industry": "Healthcare",
            "funding_amount": "$25,000,000",
            "funding_stage": "Series B",
            "employees": 110,
            "funding_date": (datetime.now() - timedelta(days=360)).strftime("%Y-%m-%d")
        },
        {
            "company_name": f"{name_prefix} Late Stage",
            "industry": "Logistics",
            "funding_amount": "$150,000,000",
            "funding_stage": "Series D",
            "employees": 550,
            "funding_date": (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        }
    ]
    
    if variation is not None and 0 <= variation < len(companies):
        return companies[variation]
    else:
        return random.choice(companies)

def test_pipeline_predictions():
    """Test if the pipeline makes reasonable predictions for test companies."""
    logger.info("=" * 80)
    logger.info("TESTING MODEL PREDICTIONS")
    logger.info("=" * 80)
    
    try:
        # Initialize the pipeline
        pipeline = EnhancedPipeline(output_dir="./test_output")
        logger.info("Pipeline initialized successfully")
        
        # Run the pipeline to ensure models are trained
        logger.info("Running pipeline to train models (this may take a moment)...")
        pipeline.run()
        
        # Test predictions with generated companies
        logger.info("\nTesting predictions for sample companies:")
        for i in range(5):
            company = generate_test_company(variation=i)
            logger.info(f"\nTest Company {i+1}: {company['company_name']}")
            logger.info(f"  Industry: {company['industry']}")
            logger.info(f"  Current Stage: {company['funding_stage']}")
            logger.info(f"  Funding Amount: {company['funding_amount']}")
            logger.info(f"  Employees: {company['employees']}")
            logger.info(f"  Last Funding Date: {company['funding_date']}")
            
            # Make prediction
            prediction = pipeline.make_prediction(company)
            
            # Display prediction
            logger.info("  Prediction Results:")
            logger.info(f"    Predicted Stage: {prediction['predicted_stage']}")
            logger.info(f"    Confidence: {prediction['confidence']:.4f}")
            
            # Check if prediction makes sense
            current_stage = company['funding_stage']
            predicted_stage = prediction['predicted_stage']
            
            if current_stage == "Pre-Seed" and predicted_stage not in ["Seed", "Series A"]:
                logger.warning(f"    ⚠️ Unusual progression: {current_stage} -> {predicted_stage}")
            elif current_stage == "Seed" and predicted_stage not in ["Series A", "Series B"]:
                logger.warning(f"    ⚠️ Unusual progression: {current_stage} -> {predicted_stage}")
            elif current_stage == "Series A" and predicted_stage not in ["Series B", "Series C"]:
                logger.warning(f"    ⚠️ Unusual progression: {current_stage} -> {predicted_stage}")
            elif current_stage == "Series B" and predicted_stage not in ["Series C", "Series D"]:
                logger.warning(f"    ⚠️ Unusual progression: {current_stage} -> {predicted_stage}")
            else:
                logger.info(f"    ✅ Reasonable progression: {current_stage} -> {predicted_stage}")
    
    except Exception as e:
        logger.error(f"Error testing predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION TESTING COMPLETED")
    logger.info("=" * 80)
    return True

def test_model_consistency():
    """Test model consistency by making predictions with slightly varied inputs."""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING MODEL CONSISTENCY")
    logger.info("=" * 80)
    
    try:
        # Initialize model manager
        model_manager = ModelManager(model_dir='models/')
        logger.info("Model manager initialized")
        
        # Create a base company
        base_company = generate_test_company(name_prefix="Consistency Test")
        logger.info(f"Base Company: {base_company['company_name']}")
        logger.info(f"  Industry: {base_company['industry']}")
        logger.info(f"  Current Stage: {base_company['funding_stage']}")
        logger.info(f"  Funding Amount: {base_company['funding_amount']}")
        logger.info(f"  Employees: {base_company['employees']}")
        
        # Make base prediction
        base_prediction = model_manager.predict(base_company)
        
        logger.info(f"Base Prediction: {base_prediction['predicted_stage']} (Confidence: {base_prediction['confidence']:.4f})")
        
        # Test with varying funding amounts
        logger.info("\nTesting consistency with varied funding amounts:")
        funding_variations = [
            "$" + str(int(float(base_company['funding_amount'].replace('$', '').replace(',', '')) * 0.8)),
            "$" + str(int(float(base_company['funding_amount'].replace('$', '').replace(',', '')) * 1.2))
        ]
        
        for var in funding_variations:
            modified_company = base_company.copy()
            modified_company['funding_amount'] = var
            prediction = model_manager.predict(modified_company)
            logger.info(f"  Funding: {var} → Prediction: {prediction['predicted_stage']} (Confidence: {prediction['confidence']:.4f})")
        
        # Test with varying employee counts
        logger.info("\nTesting consistency with varied employee counts:")
        employee_variations = [
            max(1, int(base_company['employees'] * 0.7)),
            int(base_company['employees'] * 1.3)
        ]
        
        for var in employee_variations:
            modified_company = base_company.copy()
            modified_company['employees'] = var
            prediction = model_manager.predict(modified_company)
            logger.info(f"  Employees: {var} → Prediction: {prediction['predicted_stage']} (Confidence: {prediction['confidence']:.4f})")
        
        # Test with different industries
        logger.info("\nTesting consistency with different industries:")
        industry_variations = [
            "Artificial Intelligence",
            "Financial Technology",
            "Biotechnology",
            "E-commerce"
        ]
        
        for var in industry_variations:
            if var != base_company['industry']:
                modified_company = base_company.copy()
                modified_company['industry'] = var
                prediction = model_manager.predict(modified_company)
                logger.info(f"  Industry: {var} → Prediction: {prediction['predicted_stage']} (Confidence: {prediction['confidence']:.4f})")
        
    except Exception as e:
        logger.error(f"Error testing model consistency: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("CONSISTENCY TESTING COMPLETED")
    logger.info("=" * 80)
    return True

def test_input_manipulation():
    """Test if the model can be manipulated with extreme inputs."""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING RESISTANCE TO MANIPULATION")
    logger.info("=" * 80)
    
    try:
        # Initialize model manager
        model_manager = ModelManager(model_dir='models/')
        logger.info("Model manager initialized")
        
        # Test with extreme values
        logger.info("\nTesting with extreme input values:")
        
        # Extreme funding amounts
        extreme_companies = [
            {
                "company_name": "Extreme Funding Test",
                "industry": "Software",
                "funding_amount": "$10,000,000,000",  # $10 billion
                "funding_stage": "Seed",
                "employees": 5,
                "funding_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "company_name": "Extreme Employee Test",
                "industry": "Software",
                "funding_amount": "$2,000,000",
                "funding_stage": "Seed",
                "employees": 50000,  # 50,000 employees
                "funding_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "company_name": "Invalid Stage Test",
                "industry": "Software",
                "funding_amount": "$2,000,000",
                "funding_stage": "Made Up Stage",  # Invalid stage
                "employees": 20,
                "funding_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "company_name": "Garbage Industry Test",
                "industry": "XYZZYX123456789",  # Random garbage
                "funding_amount": "$2,000,000",
                "funding_stage": "Seed",
                "employees": 20,
                "funding_date": datetime.now().strftime("%Y-%m-%d")
            }
        ]
        
        for i, company in enumerate(extreme_companies):
            logger.info(f"\nExtreme Test {i+1}: {company['company_name']}")
            for key, value in company.items():
                if key != 'company_name':
                    logger.info(f"  {key}: {value}")
            
            try:
                prediction = model_manager.predict(company)
                logger.info(f"  Prediction: {prediction['predicted_stage']} (Confidence: {prediction['confidence']:.4f})")
                
                # Check if the model handles the extreme inputs gracefully
                if 'confidence' in prediction and prediction['confidence'] < 0.5:
                    logger.info("  ✅ Model shows low confidence for extreme inputs (good)")
                else:
                    logger.warning("  ⚠️ Model shows high confidence for extreme inputs (concerning)")
                    
            except Exception as e:
                logger.warning(f"  ⚠️ Model crashed on extreme input: {str(e)}")
                
        # Test with missing values
        logger.info("\nTesting with missing inputs:")
        missing_companies = [
            {
                "company_name": "Missing Funding Test",
                "industry": "Software",
                "funding_stage": "Seed",
                "employees": 20,
                "funding_date": datetime.now().strftime("%Y-%m-%d")
                # Missing funding_amount
            },
            {
                "company_name": "Missing Employees Test",
                "industry": "Software",
                "funding_amount": "$2,000,000",
                "funding_stage": "Seed",
                "funding_date": datetime.now().strftime("%Y-%m-%d")
                # Missing employees
            },
            {
                "company_name": "Missing Industry Test",
                "funding_amount": "$2,000,000",
                "funding_stage": "Seed",
                "employees": 20,
                "funding_date": datetime.now().strftime("%Y-%m-%d")
                # Missing industry
            }
        ]
        
        for i, company in enumerate(missing_companies):
            logger.info(f"\nMissing Value Test {i+1}: {company['company_name']}")
            for key, value in company.items():
                if key != 'company_name':
                    logger.info(f"  {key}: {value}")
            
            try:
                prediction = model_manager.predict(company)
                logger.info(f"  Prediction: {prediction['predicted_stage']} (Confidence: {prediction['confidence']:.4f})")
                logger.info("  ✅ Model handles missing values")
            except Exception as e:
                logger.warning(f"  ⚠️ Model crashed on missing input: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error testing manipulation resistance: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("MANIPULATION TESTING COMPLETED")
    logger.info("=" * 80)
    return True

if __name__ == "__main__":
    # Run all tests
    os.makedirs("./test_output", exist_ok=True)
    
    # Only run one test at a time to keep output manageable
    test_pipeline_predictions()
    # test_model_consistency()
    # test_input_manipulation() 