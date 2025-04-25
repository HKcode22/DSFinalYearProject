#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Success/Failure Prediction Module

This module implements a comprehensive success/failure prediction system by integrating:
1. Funding Stage Prediction
2. Funding Continuation Analysis  
3. Funding Anomaly Detection
4. Funding Amount Forecast
5. Industry Trend Analysis

Together, these models form a meta-model for startup success/failure prediction.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import seaborn as sns
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Optional, Tuple, Union
import traceback
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import re

# Import individual prediction modules
from funding_stage_prediction import EnhancedPipeline as FundingStagePipeline
from funding_continuation import FundingContinuationAnalysis
from funding_amount_forecast import FundingAmountForecast
from funding_anomaly_detection import FundingAnomalyDetection
from industry_trend_analysis import IndustryTrendAnalyzer

class SuccessFailurePrediction:
    """
    Core Architecture: Meta-model aggregating insights from all components
    
    Input Features:
    - Stage transition probability (30%) - XGBoost Classifier
    - 18-month survival probability (25%) - Cox Model
    - Funding adequacy score (20%) - QRF Forecast
    - Industry growth momentum (15%) - STL Decomposition
    - Anomaly severity (10%) - Isolation Forest
    
    Success Definition:
    - Received Series B+ funding within 3 years
    - Not marked as outlier in 2 consecutive quarters
    - Employee growth > industry 75th percentile
    
    Failure Signals:
    - Survival probability <40% for 6 months
    - Burn rate > industry 90th percentile
    - Anomaly score persists >3 months
    """
    
    def __init__(self, data_dir=None, output_dir=None):
        """Initialize the Success/Failure prediction system."""
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'JSONFolder')
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'outputSuccessFailure')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize component models
        self.stage_predictor = None
        self.continuation_analyzer = None
        self.amount_forecaster = None
        self.anomaly_detector = None
        self.trend_analyzer = None
        
        # Initialize merged data
        self.merged_data = None
        
        # Meta-model weights
        self.model_weights = {
            'stage_transition_probability': 0.30,
            'survival_probability': 0.25,
            'funding_adequacy': 0.20,
            'industry_momentum': 0.15,
            'anomaly_severity': 0.10
        }
        
        # Set up matplotlib parameters
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        self.logger.info("Success/Failure Prediction system initialized")
        
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create a logger
        self.logger = logging.getLogger('success_failure_prediction')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        f_handler = logging.FileHandler(os.path.join(self.output_dir, 'success_failure.log'))
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
        
    def initialize_components(self):
        """Initialize all component prediction systems."""
        self.logger.info("Initializing component prediction systems")
        
        try:
            # Create subdirectories for components
            os.makedirs(os.path.join(self.output_dir, 'stage_prediction'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'continuation'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'amount_forecast'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'anomaly_detection'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'industry_trends'), exist_ok=True)
            
            # Initialize funding stage prediction
            self.stage_predictor = FundingStagePipeline(
                base_dir=self.data_dir,
                output_dir=os.path.join(self.output_dir, 'stage_prediction')
            )
            
            # Initialize funding continuation analysis
            self.continuation_analyzer = FundingContinuationAnalysis(
                data_dir=self.data_dir,
                output_dir=os.path.join(self.output_dir, 'continuation')
            )
            
            # Initialize funding amount forecast
            self.amount_forecaster = FundingAmountForecast(
                data_dir=self.data_dir,
                output_dir=os.path.join(self.output_dir, 'amount_forecast')
            )
            
            # Initialize funding anomaly detection
            self.anomaly_detector = FundingAnomalyDetection(
                data_dir=self.data_dir,
                output_dir=os.path.join(self.output_dir, 'anomaly_detection'),
                contamination=0.01
            )
            
            # Initialize industry trend analysis
            self.trend_analyzer = IndustryTrendAnalyzer(
                data_dir=self.data_dir,
                output_dir=os.path.join(self.output_dir, 'industry_trends')
            )
            
            self.logger.info("All component systems initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
            
    def run_components(self):
        """Run all component analyses to generate input data for meta-model."""
        self.logger.info("Running component analyses")
        
        results = {}
        
        try:
            # Ensure components are initialized
            if not all([self.stage_predictor, self.continuation_analyzer, 
                        self.amount_forecaster, self.anomaly_detector, 
                        self.trend_analyzer]):
                self.initialize_components()
                
            # Run funding stage prediction
            self.logger.info("Running funding stage prediction")
            stage_results = self.stage_predictor.run()
            results['stage_prediction'] = stage_results
            
            # Run funding continuation analysis
            self.logger.info("Running funding continuation analysis")
            continuation_results = self.continuation_analyzer.run_analysis()
            results['continuation'] = continuation_results
            
            # Run funding amount forecast
            self.logger.info("Running funding amount forecast")
            amount_results = self.amount_forecaster.run_analysis()
            results['amount_forecast'] = amount_results
            
            # Run funding anomaly detection
            self.logger.info("Running funding anomaly detection")
            anomaly_results = self.anomaly_detector.run_analysis()
            results['anomaly_detection'] = anomaly_results
            
            # Run industry trend analysis
            self.logger.info("Running industry trend analysis")
            trend_results = self.trend_analyzer.generate_report()
            results['industry_trends'] = trend_results
            
            self.logger.info("All component analyses completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running component analyses: {str(e)}")
            self.logger.error(traceback.format_exc())
            return results 

    def integrate_data(self, component_results=None):
        """
        Integrate data from all component analyses into a unified dataset.
        
        Args:
            component_results (dict): Results from component analyses, if already run
            
        Returns:
            pandas.DataFrame: Integrated dataset
        """
        self.logger.info("Integrating data from all components")
        
        try:
            # Run components if not already run, but don't let failures stop us
            if component_results is None:
                try:
                    component_results = self.run_components()
                except Exception as e:
                    self.logger.error(f"Error running components: {str(e)}")
                    component_results = {}
            
            # Start with loading data directly from JSON files
            self.logger.info("Loading data directly from JSON files")
            
            # Create empty dataframe
            integrated_data = pd.DataFrame()
            
            # Load data from JSON files
            json_files = [
                os.path.join(self.data_dir, 'fundraisestartup50.json'),
                os.path.join(self.data_dir, 'growthlistscrapper.json'),
                os.path.join(self.data_dir, 'topstartupio50.json')
            ]
            
            all_data = []
            
            # Print the paths to debug
            for file_path in json_files:
                self.logger.info(f"Checking file existence: {file_path}, exists: {os.path.exists(file_path)}")
            
            # Fundraiser data (has a "companies" key)
            fundraiser_path = os.path.join(self.data_dir, 'fundraisestartup50.json')
            if os.path.exists(fundraiser_path):
                try:
                    with open(fundraiser_path, 'r') as f:
                        fundraiser_data = json.load(f)
                        if isinstance(fundraiser_data, dict) and 'companies' in fundraiser_data:
                            fundraiser_df = pd.DataFrame(fundraiser_data['companies'])
                            self.logger.info(f"Loaded {len(fundraiser_df)} records from fundraisestartup50.json")
                            
                            # Standardize column names for fundraiser data
                            column_renames_fundraiser = {
                                'Company': 'company_name',
                                'Funding_Type': 'funding_stage',
                                'Funding_Amount_USD': 'funding_amount_usd',
                                'Funding_Date': 'funding_date',
                                'Industry': 'industry',
                                'Total_Employees': 'employees'
                            }
                            
                            for old_col, new_col in column_renames_fundraiser.items():
                                if old_col in fundraiser_df.columns:
                                    fundraiser_df[new_col] = fundraiser_df[old_col]
                            
                            all_data.append(fundraiser_df)
                        else:
                            self.logger.error("Invalid structure in fundraisestartup50.json")
                except Exception as e:
                    self.logger.error(f"Error loading fundraisestartup50.json: {str(e)}")
            
            # Growthlist data (direct list of companies)
            growthlist_path = os.path.join(self.data_dir, 'growthlistscrapper.json')
            if os.path.exists(growthlist_path):
                try:
                    with open(growthlist_path, 'r') as f:
                        growthlist_data = json.load(f)
                        if isinstance(growthlist_data, list):
                            growthlist_df = pd.DataFrame(growthlist_data)
                            self.logger.info(f"Loaded {len(growthlist_df)} records from growthlistscrapper.json")
                            
                            # Standardize column names for growthlist data
                            column_renames_growthlist = {
                                'name': 'company_name',
                                'funding_type': 'funding_stage',
                                'funding_amount': 'funding_amount_usd',
                                'last_funding_date': 'funding_date',
                                'industry': 'industry'
                            }
                            
                            for old_col, new_col in column_renames_growthlist.items():
                                if old_col in growthlist_df.columns:
                                    growthlist_df[new_col] = growthlist_df[old_col]
                            
                            all_data.append(growthlist_df)
                        else:
                            self.logger.error("Invalid structure in growthlistscrapper.json")
                except Exception as e:
                    self.logger.error(f"Error loading growthlistscrapper.json: {str(e)}")
            
            # Topstartup data
            topstartup_path = os.path.join(self.data_dir, 'topstartupio50.json')
            if os.path.exists(topstartup_path):
                try:
                    with open(topstartup_path, 'r') as f:
                        topstartup_data = json.load(f)
                        if isinstance(topstartup_data, list):
                            topstartup_df = pd.DataFrame(topstartup_data)
                            self.logger.info(f"Loaded {len(topstartup_df)} records from topstartupio50.json")
                            
                            # Parse funding information
                            if 'funding' in topstartup_df.columns:
                                def parse_funding_info(funding_str):
                                    if not isinstance(funding_str, str):
                                        return pd.Series({'amount': None, 'stage': None, 'date': None})
                                        
                                    # Example: "Bessemer Sequoia $11M Series A in 2024"
                                    amount = None
                                    stage = None
                                    date = None
                                    
                                    # Extract amount
                                    amount_match = re.search(r'\$(\d+(?:\.\d+)?[KMB]?)', funding_str)
                                    if amount_match:
                                        amount = amount_match.group(1)
                                    
                                    # Extract stage
                                    stage_match = re.search(r'(Seed|Series [A-Z]|Angel)', funding_str)
                                    if stage_match:
                                        stage = stage_match.group(1)
                                    
                                    # Extract date
                                    date_match = re.search(r'in (\d{4})', funding_str)
                                    if date_match:
                                        date = f"{date_match.group(1)}-01-01"  # Default to January 1st
                                        
                                    return pd.Series({'amount': amount, 'stage': stage, 'date': date})
                                
                                funding_info = topstartup_df['funding'].apply(parse_funding_info)
                                topstartup_df['funding_amount_usd'] = funding_info['amount']
                                topstartup_df['funding_stage'] = funding_info['stage']
                                topstartup_df['funding_date'] = funding_info['date']
                            
                            # Standardize column names for topstartup data
                            column_renames_topstartup = {
                                'name': 'company_name',
                                'category': 'industry',
                                'employees': 'employees'
                            }
                            
                            for old_col, new_col in column_renames_topstartup.items():
                                if old_col in topstartup_df.columns:
                                    topstartup_df[new_col] = topstartup_df[old_col]
                            
                            all_data.append(topstartup_df)
                        else:
                            self.logger.error("Invalid structure in topstartupio50.json")
                except Exception as e:
                    self.logger.error(f"Error loading topstartupio50.json: {str(e)}")
            
            # Check if we have any data to work with
            if not all_data:
                self.logger.error("Failed to load any data from JSON files")
                return pd.DataFrame()
            
            # Merge all dataframes
            integrated_data = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Combined {len(integrated_data)} records from all JSON files")
            
            # Parse funding amount to ensure all values are numeric
            if 'funding_amount_usd' in integrated_data.columns:
                def parse_funding_amount(amount_str):
                    if pd.isna(amount_str) or amount_str == '':
                        return np.nan
                    
                    try:
                        # Remove currency symbols and commas
                        amount_str = str(amount_str).replace('$', '').replace(',', '')
                        
                        # Check for million/billion indicators
                        if 'B' in amount_str or 'b' in amount_str:
                            return float(amount_str.replace('B', '').replace('b', '')) * 1_000_000_000
                        elif 'M' in amount_str or 'm' in amount_str:
                            return float(amount_str.replace('M', '').replace('m', '')) * 1_000_000
                        elif 'K' in amount_str or 'k' in amount_str:
                            return float(amount_str.replace('K', '').replace('k', '')) * 1_000
                        else:
                            return float(amount_str)
                    except:
                        return np.nan
                
                integrated_data['funding_amount_usd'] = integrated_data['funding_amount_usd'].apply(parse_funding_amount)
            
            # Convert funding dates to datetime format
            if 'funding_date' in integrated_data.columns:
                integrated_data['funding_date'] = pd.to_datetime(integrated_data['funding_date'], errors='coerce')
            
            # Remove duplicate companies
            integrated_data = integrated_data.drop_duplicates(subset=['company_name', 'funding_date']).reset_index(drop=True)
            
            # Add component analysis results if available
            if component_results:
                # Add funding stage prediction metrics if available
                if ('stage_prediction' in component_results and 
                    component_results['stage_prediction'] is not None and 
                    isinstance(component_results['stage_prediction'], dict) and
                    'predictions' in component_results['stage_prediction']):
                    
                    stage_predictions = component_results['stage_prediction']['predictions']
                    if isinstance(stage_predictions, pd.DataFrame) and not stage_predictions.empty:
                        key_cols = ['company_name', 'predicted_stage', 'stage_probability']
                        avail_cols = [col for col in key_cols if col in stage_predictions.columns]
                        
                        if len(avail_cols) > 1:
                            self.logger.info("Adding stage prediction metrics")
                            integrated_data = pd.merge(
                                integrated_data, 
                                stage_predictions[avail_cols],
                                on='company_name', 
                                how='left'
                            )
                
                # Add continuation analysis results if available
                if ('continuation' in component_results and 
                    component_results['continuation'] is not None and
                    isinstance(component_results['continuation'], dict) and
                    'survival_probabilities' in component_results['continuation']):
                    
                    survival_probs = component_results['continuation']['survival_probabilities']
                    if isinstance(survival_probs, pd.DataFrame) and not survival_probs.empty:
                        self.logger.info("Adding survival probability metrics")
                        integrated_data = pd.merge(
                            integrated_data,
                            survival_probs,
                            on='company_name',
                            how='left'
                        )
                
                # Add anomaly detection results if available
                if ('anomaly_detection' in component_results and 
                    component_results['anomaly_detection'] is not None and
                    isinstance(component_results['anomaly_detection'], dict) and
                    'anomaly_scores' in component_results['anomaly_detection']):
                    
                    anomaly_scores = component_results['anomaly_detection']['anomaly_scores']
                    if isinstance(anomaly_scores, pd.DataFrame) and not anomaly_scores.empty:
                        self.logger.info("Adding anomaly detection metrics")
                        integrated_data = pd.merge(
                            integrated_data,
                            anomaly_scores,
                            on='company_name',
                            how='left'
                        )
            
            # Store the integrated data
            self.merged_data = integrated_data
            
            self.logger.info(f"Successfully integrated data: {len(integrated_data)} records with {len(integrated_data.columns)} features")
            return integrated_data
            
        except Exception as e:
            self.logger.error(f"Error integrating data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
        
    def calculate_success_scores(self, integrated_data=None):
        """
        Calculate success scores for each company based on the integrated metrics.
        
        Args:
            integrated_data (pandas.DataFrame): Integrated data from all components
            
        Returns:
            pandas.DataFrame: Data with calculated success scores
        """
        self.logger.info("Calculating success scores")
        
        try:
            # Use stored merged data if none provided
            if integrated_data is None:
                if self.merged_data is None:
                    self.logger.warning("No integrated data available, running integration")
                    integrated_data = self.integrate_data()
                else:
                    integrated_data = self.merged_data
            
            # Create a copy to avoid modifying the original
            data = integrated_data.copy()
            
            # Calculate component scores based on available metrics
            # 1. Stage transition score (higher stages = higher score)
            if 'next_stage_probability' in data.columns:
                data['stage_transition_score'] = data['next_stage_probability'] * 100
            elif 'stage_probability' in data.columns:
                data['stage_transition_score'] = data['stage_probability'] * 100
            elif 'funding_stage' in data.columns:
                # IMPROVED: Better mapping for funding stages with more granular weights
                stage_mapping = {
                    'Pre-Seed': 1.5,
                    'Seed': 2.5,
                    'Series A': 4.0,
                    'Series B': 5.5,
                    'Series C': 7.0,
                    'Series D': 8.0,
                    'Series E': 8.5,
                    'Series F': 9.0,
                    'Series G': 9.5,
                    'IPO': 10.0,
                    'Venture - Series Unknown': 4.0,  # More optimistic approximation
                    'Private Equity': 8.0,  # Added new stage
                    'Debt Financing': 6.0,  # Added new stage
                    'Angel': 2.0,  # Added new stage
                    'Grant': 1.5,  # Added new stage
                    'Undisclosed': 3.0,  # Added new stage
                }
                
                # Enhanced stage extraction to handle more formats
                def extract_base_stage(stage_str):
                    if pd.isna(stage_str):
                        return np.nan
                    
                    stage_str = str(stage_str).strip().title()
                    
                    # Direct matches first
                    if stage_str in stage_mapping:
                        return stage_str
                    
                    # Pattern matching
                    if 'Pre-Seed' in stage_str or 'Preseed' in stage_str:
                        return 'Pre-Seed'
                    elif 'Seed' in stage_str and 'Pre' not in stage_str:
                        return 'Seed'
                    elif 'Angel' in stage_str:
                        return 'Angel'
                    elif 'Grant' in stage_str:
                        return 'Grant'
                    elif 'IPO' in stage_str:
                        return 'IPO'
                    elif 'Debt' in stage_str:
                        return 'Debt Financing'
                    elif 'Private Equity' in stage_str:
                        return 'Private Equity'
                    elif 'Undisclosed' in stage_str:
                        return 'Undisclosed'
                    elif 'Series' in stage_str:
                        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                            if f'Series {letter}' in stage_str:
                                return f'Series {letter}'
                        return 'Venture - Series Unknown'
                    else:
                        return 'Venture - Series Unknown'
                
                # Apply the extraction
                data['base_stage'] = data['funding_stage'].apply(extract_base_stage)
                
                # Map to numeric and scale to 0-100
                data['stage_transition_score'] = data['base_stage'].map(
                    lambda x: stage_mapping.get(x, 3.0) * 10  # Scale to 0-100
                )
            else:
                # Default score if no stage information available
                self.logger.warning("No stage information available, using default score")
                data['stage_transition_score'] = 50  # Neutral score
            
            # IMPROVED: Add new feature - funding recency
            if 'funding_date' in data.columns:
                # Calculate months since funding
                current_date = pd.Timestamp.now()
                
                # Ensure funding_date is datetime
                if not pd.api.types.is_datetime64_any_dtype(data['funding_date']):
                    data['funding_date'] = pd.to_datetime(data['funding_date'], errors='coerce')
                
                # Calculate recency score (more recent = higher score)
                data['months_since_funding'] = data['funding_date'].apply(
                    lambda x: (current_date - x).days / 30 if pd.notnull(x) else np.nan
                )
                
                # Convert to score (exponential decay - more recent is better)
                data['funding_recency_score'] = data['months_since_funding'].apply(
                    lambda x: 100 * np.exp(-0.1 * x) if pd.notnull(x) else 50
                )
                
                # Cap the score
                data['funding_recency_score'] = np.clip(data['funding_recency_score'], 0, 100)
            else:
                data['funding_recency_score'] = 50  # Neutral score
            
            # 2. Survival score (18-month survival probability)
            if '18_month_survival' in data.columns:
                data['survival_score'] = data['18_month_survival'] * 100
            elif 'survival_probability' in data.columns:
                data['survival_score'] = data['survival_probability'] * 100
            else:
                # IMPROVED: Better default based on stage and funding amount
                self.logger.warning("No survival information available, estimating from stage and funding")
                if 'stage_transition_score' in data.columns and 'funding_amount_usd' in data.columns:
                    # Higher funded companies at higher stages have better survival odds
                    data['has_funding'] = data['funding_amount_usd'].notnull() & (data['funding_amount_usd'] > 0)
                    
                    # Base survival on stage with bonus for having funding
                    data['survival_score'] = data['stage_transition_score'] * 0.7
                    data.loc[data['has_funding'], 'survival_score'] += 15  # Bonus for having funding
                    
                    # Cap at 100
                    data['survival_score'] = np.clip(data['survival_score'], 0, 100)
                else:
                    data['survival_score'] = 50  # Neutral score
            
            # 3. Funding adequacy score - IMPROVED
            if 'funding_adequacy' in data.columns:
                data['funding_adequacy_score'] = data['funding_adequacy'] * 100
            elif 'predicted_amount' in data.columns and 'upper_bound' in data.columns:
                # Calculate ratio of predicted amount to upper bound
                data['funding_adequacy_score'] = np.minimum(
                    data['predicted_amount'] / data['upper_bound'], 
                    1.5  # Cap at 150%
                ) * 100
            elif 'funding_amount_usd' in data.columns:
                # IMPROVED: Better normalization within industry and stage
                if 'industry' in data.columns and 'base_stage' in data.columns:
                    # Group by both industry and stage for more accurate comparison
                    data['industry_stage_group'] = data.apply(
                        lambda x: str(x['industry']) + '_' + str(x['base_stage']) 
                        if pd.notnull(x['industry']) and pd.notnull(x['base_stage']) 
                        else None, 
                        axis=1
                    )
                    
                    # Calculate group statistics
                    group_mean = data.groupby('industry_stage_group')['funding_amount_usd'].transform('mean')
                    group_std = data.groupby('industry_stage_group')['funding_amount_usd'].transform('std')
                    
                    # If not enough samples in group, use just industry
                    industry_mean = data.groupby('industry')['funding_amount_usd'].transform('mean')
                    industry_std = data.groupby('industry')['funding_amount_usd'].transform('std')
                    
                    # Replace NaN group stats with industry stats
                    group_mean = group_mean.fillna(industry_mean)
                    group_std = group_std.fillna(industry_std)
                    
                    # Calculate z-score
                    data['funding_z_score'] = (data['funding_amount_usd'] - group_mean) / group_std.replace(0, 1)
                    
                    # Convert to percentile-like score (sigmoid transformation)
                    data['funding_adequacy_score'] = 100 / (1 + np.exp(-0.5 * data['funding_z_score'])) 
                else:
                    # Normalize within the entire dataset
                    overall_mean = data['funding_amount_usd'].mean()
                    overall_std = data['funding_amount_usd'].std()
                    
                    data['funding_z_score'] = (data['funding_amount_usd'] - overall_mean) / overall_std
                    data['funding_adequacy_score'] = 100 / (1 + np.exp(-0.5 * data['funding_z_score']))
                
                # Cap between 0-100
                data['funding_adequacy_score'] = np.clip(data['funding_adequacy_score'], 0, 100)
            else:
                # No adequacy metrics available
                self.logger.warning("No funding adequacy information available, using neutral score")
                data['funding_adequacy_score'] = 50  # Neutral score
            
            # 4. Industry momentum score
            if 'industry_momentum' in data.columns:
                data['industry_momentum_score'] = np.minimum(data['industry_momentum'] * 200, 100)  # Scale and cap
            elif 'industry_growth' in data.columns:
                data['industry_momentum_score'] = np.minimum(data['industry_growth'] * 200, 100)
            elif 'industry' in data.columns:
                # Calculate momentum based on industry funding patterns in the dataset
                industry_counts = data['industry'].value_counts()
                industry_momentum = industry_counts / industry_counts.sum()
                
                # Map momentum to companies
                data['industry_momentum_score'] = data['industry'].map(
                    lambda x: min(industry_momentum.get(x, 0) * 500, 100)  # Scale and cap
                )
            else:
                # No industry momentum metrics available
                self.logger.warning("No industry momentum information available, using neutral score")
                data['industry_momentum_score'] = 50  # Neutral score
            
            # 5. Anomaly score (lower anomaly scores = higher normalized scores)
            if 'anomaly_score' in data.columns:
                # Invert anomaly score (anomaly scores are negative, with -1 being most anomalous)
                data['anomaly_normalized_score'] = (data['anomaly_score'] + 1) * 50  # Convert -1..0 to 0..50
            elif 'is_anomaly' in data.columns:
                # Binary anomaly flag
                data['anomaly_normalized_score'] = data['is_anomaly'].map({False: 100, True: 0})
            else:
                # No anomaly metrics available
                self.logger.warning("No anomaly detection information available, using optimistic score")
                data['anomaly_normalized_score'] = 75  # Somewhat optimistic default
            
            # Apply weights to calculate the final success score with new funding_recency component
            data['success_score'] = (
                data['stage_transition_score'] * 0.25 +  # Changed from 0.30
                data['survival_score'] * 0.20 +          # Changed from 0.25
                data['funding_adequacy_score'] * 0.20 +  # Same
                data['industry_momentum_score'] * 0.15 + # Same
                data['anomaly_normalized_score'] * 0.10 + # Same
                data['funding_recency_score'] * 0.10     # New component
            )
            
            # Clip to ensure scores are in 0-100 range
            data['success_score'] = np.clip(data['success_score'], 0, 100)
            
            self.logger.info("Success scores calculated successfully")
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating success scores: {str(e)}")
            self.logger.error(traceback.format_exc())
            return integrated_data 

    def classify_startups(self, scored_data=None):
        """
        Classify companies into success/failure categories based on scores.
        
        Args:
            scored_data (pandas.DataFrame): Data with success scores
            
        Returns:
            pandas.DataFrame: Data with success/failure classifications
        """
        self.logger.info("Classifying companies by success potential")
        
        try:
            # Use calculated scores if none provided
            if scored_data is None:
                self.logger.warning("No scored data provided, calculating scores")
                scored_data = self.calculate_success_scores()
            
            # Create a copy to avoid modifying the original
            data = scored_data.copy()
            
            # ENHANCED: Implement more optimistic classification with multiple models
            # First, let's create additional features for classification
            
            # 1. Create company maturity score based on how long they've been operating
            if 'funding_date' in data.columns:
                # Ensure funding_date is datetime
                if not pd.api.types.is_datetime64_any_dtype(data['funding_date']):
                    data['funding_date'] = pd.to_datetime(data['funding_date'], errors='coerce')
                
                # Get current year
                current_year = pd.Timestamp.now().year
                
                # Extract year from funding date
                data['funding_year'] = data['funding_date'].dt.year
                
                # Calculate years since funding (use most recent if multiple)
                data['years_since_funding'] = data['funding_year'].apply(
                    lambda x: current_year - x if pd.notnull(x) else 3  # Default to 3 years if unknown
                )
                
                # Company maturity score - higher for companies operating longer
                data['maturity_score'] = data['years_since_funding'].apply(
                    lambda x: min(100, x * 10) if pd.notnull(x) else 30  # 10 points per year, max 100
                )
            else:
                data['maturity_score'] = 30  # Default if no funding date
            
            # 2. Calculate funding momentum (higher stages get bonus)
            if 'funding_stage' in data.columns:
                # Define momentum based on stage
                stage_momentum = {
                    'Pre-Seed': 10,
                    'Seed': 20,
                    'Series A': 40,
                    'Series B': 60,
                    'Series C': 75,
                    'Series D': 85,
                    'Series E': 90,
                    'Series F': 95,
                    'Series G': 97,
                    'Series H': 98,
                    'IPO': 100,
                    'Private Equity': 80,
                    'Venture - Series Unknown': 50,
                    'Debt Financing': 40,
                    'Grant': 30,
                    'Undisclosed': 25
                }
                
                # Map stage to momentum score
                data['funding_momentum'] = data['funding_stage'].map(
                    lambda x: stage_momentum.get(x, 30) if pd.notnull(x) else 30
                )
            else:
                data['funding_momentum'] = 30  # Default
            
            # 3. Create industry success likelihood based on historical performance
            if 'industry' in data.columns:
                # Calculate industry success rates (can expand with external data)
                industry_success_rates = {
                    'Artificial Intelligence': 65,
                    'information technology & services': 60,
                    'Software': 58,
                    'Biotechnology': 55,
                    'FinTech': 53,
                    'Healthcare': 52,
                    'Robotics': 50,
                    'Data': 48,
                    'Energy': 45,
                    'Transportation': 42
                }
                
                # Apply industry rates with a default of 45%
                data['industry_success_likelihood'] = data['industry'].map(
                    lambda x: industry_success_rates.get(x, 45) if pd.notnull(x) else 45
                )
            else:
                data['industry_success_likelihood'] = 45  # Default
            
            # 4. Create funding amount tier (higher amounts = higher tier)
            if 'funding_amount_usd' in data.columns:
                # Create funding tiers
                def get_funding_tier(amount):
                    if pd.isnull(amount) or amount == 0:
                        return 20
                    elif amount < 1_000_000:  # < $1M
                        return 30
                    elif amount < 5_000_000:  # $1M-$5M
                        return 40
                    elif amount < 20_000_000:  # $5M-$20M
                        return 60
                    elif amount < 50_000_000:  # $20M-$50M
                        return 70
                    elif amount < 100_000_000:  # $50M-$100M
                        return 80
                    else:  # $100M+
                        return 90
                
                data['funding_tier'] = data['funding_amount_usd'].apply(get_funding_tier)
            else:
                data['funding_tier'] = 30  # Default
            
            # 5. Calculate optimistic success score - weighted combination of multiple factors
            data['optimistic_score'] = (
                data['success_score'] * 0.35 +             # Original success score
                data['funding_momentum'] * 0.25 +          # Funding stage momentum
                data['industry_success_likelihood'] * 0.15 + # Industry success rate
                data['funding_tier'] * 0.15 +              # Funding amount tier
                data['maturity_score'] * 0.10              # Company maturity
            )
            
            # 6. Create ensemble classification using multiple approaches
            
            # 6.1 Dynamic thresholds based on data distribution
            score_mean = data['optimistic_score'].mean()
            score_std = data['optimistic_score'].std()
            
            # More optimistic thresholds
            LOW_SUCCESS = max(35, score_mean - 0.9 * score_std)    # ~20th percentile
            MEDIUM_SUCCESS = max(45, score_mean - 0.2 * score_std)  # ~45th percentile  
            HIGH_SUCCESS = min(75, score_mean + 0.7 * score_std)   # ~75th percentile
            
            self.logger.info(f"Optimistic thresholds - Low: {LOW_SUCCESS:.1f}, Medium: {MEDIUM_SUCCESS:.1f}, High: {HIGH_SUCCESS:.1f}")
            
            # Classify based on optimistic score
            data['optimistic_classification'] = pd.cut(
                data['optimistic_score'],
                bins=[0, LOW_SUCCESS, MEDIUM_SUCCESS, HIGH_SUCCESS, 100],
                labels=['High Failure Risk', 'Moderate Failure Risk', 'Moderate Success Potential', 'High Success Potential']
            )
            
            # Binary classification
            data['is_likely_success_optimistic'] = data['optimistic_score'] >= MEDIUM_SUCCESS
            
            # 6.2 Also keep the original classification for comparison
            data['original_classification'] = pd.cut(
                data['success_score'],
                bins=[0, LOW_SUCCESS, MEDIUM_SUCCESS, HIGH_SUCCESS, 100],
                labels=['High Failure Risk', 'Moderate Failure Risk', 'Moderate Success Potential', 'High Success Potential']
            )
            
            # Use the optimistic classification as the primary one
            data['success_classification'] = data['optimistic_classification']
            data['is_likely_success'] = data['is_likely_success_optimistic']
            
            # Count by category
            classification_counts = data['success_classification'].value_counts()
            self.logger.info(f"Classification results: {dict(classification_counts)}")
            
            # Save original success scores
            data['original_success_score'] = data['success_score']
            
            # Use optimistic score as the main success score
            data['success_score'] = data['optimistic_score']
            
            # Enhanced: Time horizon predictions
            # Short-term success (1 year)
            data['short_term_success_score'] = (
                data['funding_momentum'] * 0.30 +
                data['funding_tier'] * 0.25 +
                data['original_success_score'] * 0.20 +
                data['industry_success_likelihood'] * 0.15 +
                data['maturity_score'] * 0.10
            )
            
            # Long-term success (5+ years)
            data['long_term_success_score'] = (
                data['industry_success_likelihood'] * 0.30 +
                data['original_success_score'] * 0.25 +
                data['funding_momentum'] * 0.20 +
                data['maturity_score'] * 0.15 +
                data['funding_tier'] * 0.10
            )
            
            # Classifications for different time horizons
            data['short_term_outlook'] = pd.cut(
                data['short_term_success_score'],
                bins=[0, LOW_SUCCESS, MEDIUM_SUCCESS, HIGH_SUCCESS, 100],
                labels=['Poor', 'Fair', 'Good', 'Excellent']
            )
            
            data['long_term_outlook'] = pd.cut(
                data['long_term_success_score'],
                bins=[0, LOW_SUCCESS, MEDIUM_SUCCESS, HIGH_SUCCESS, 100],
                labels=['Poor', 'Fair', 'Good', 'Excellent']
            )
            
            # Calculate risk levels
            def calculate_risk_level(row):
                # Higher score = lower risk
                score = row['optimistic_score']
                if score >= 65:
                    return 'Minimal Risk'
                elif score >= 50:
                    return 'Low Risk'  
                elif score >= 35:
                    return 'Medium Risk'
                else:
                    return 'High Risk'
            
            data['risk_level'] = data.apply(calculate_risk_level, axis=1)
            
            # Log additional analysis
            risk_counts = data['risk_level'].value_counts()
            risk_percentages = (risk_counts / len(data) * 100).round(1)
            risk_stats = {k: f"{v} ({risk_percentages[k]}%)" for k, v in risk_counts.items()}
            
            self.logger.info(f"Risk level distribution: {risk_stats}")
            
            # Add meta-information about thresholds used
            data.attrs['thresholds'] = {
                'low_success': LOW_SUCCESS,
                'medium_success': MEDIUM_SUCCESS,
                'high_success': HIGH_SUCCESS
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error classifying startups: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return empty DataFrame in case of error
            return pd.DataFrame()

    def visualize_results(self, classified_data=None, output_dir=None):
        """
        Generate visualizations for the success/failure predictions.
        
        Args:
            classified_data (pandas.DataFrame): Data with classifications
            output_dir (str): Directory to save visualizations
            
        Returns:
            bool: Whether visualization was successful
        """
        self.logger.info("Generating visualizations")
        
        try:
            # Use classified data if none provided
            if classified_data is None:
                self.logger.warning("No classified data provided, running classification")
                classified_data = self.classify_startups()
            
            # If the dataframe is empty, we can't create visualizations
            if classified_data.empty:
                self.logger.error("No data available for visualization")
                return False
            
            # Set output directory
            if output_dir is None:
                output_dir = os.path.join(self.output_dir, 'visualizations')
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Success score distribution
            plt.figure(figsize=(12, 7))
            sns.histplot(classified_data['success_score'], bins=20, kde=True)
            plt.title('Distribution of Success Scores')
            plt.xlabel('Success Score')
            plt.ylabel('Count')
            
            # Add vertical lines for thresholds
            plt.axvline(x=40, color='r', linestyle='--', alpha=0.7, label='High Failure Risk Threshold')
            plt.axvline(x=60, color='y', linestyle='--', alpha=0.7, label='Success Threshold')
            plt.axvline(x=75, color='g', linestyle='--', alpha=0.7, label='High Success Threshold')
            
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'success_score_distribution.png'), bbox_inches='tight')
            plt.close()
            
            # 2. Classification breakdown
            plt.figure(figsize=(12, 7))
            ax = sns.countplot(y='success_classification', data=classified_data, 
                              order=classified_data['success_classification'].value_counts().index)
            
            # Add count labels
            for p in ax.patches:
                width = p.get_width()
                plt.text(width + 1, p.get_y() + p.get_height()/2, f'{int(width)}', 
                        ha='left', va='center')
            
            plt.title('Startup Success/Failure Classification')
            plt.xlabel('Count')
            plt.ylabel('Classification')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'classification_breakdown.png'), bbox_inches='tight')
            plt.close()
            
            # Skip additional visualizations if necessary columns are missing
            
            # 3. Success score by funding stage
            if 'funding_stage' in classified_data.columns:
                plt.figure(figsize=(14, 8))
                
                # Get top 7 most common stages for readability
                top_stages = classified_data['funding_stage'].value_counts().head(7).index
                stage_data = classified_data[classified_data['funding_stage'].isin(top_stages)]
                
                if not stage_data.empty:
                    ax = sns.boxplot(x='funding_stage', y='success_score', data=stage_data)
                    plt.title('Success Score by Funding Stage')
                    plt.xlabel('Funding Stage')
                    plt.ylabel('Success Score')
                    plt.xticks(rotation=45)
                    
                    # Add median labels
                    medians = stage_data.groupby(['funding_stage'])['success_score'].median().values
                    pos = range(len(medians))
                    for tick, label in zip(pos, medians):
                        plt.text(tick, label + 2, f'{label:.1f}', horizontalalignment='center', 
                                size='small', color='black', weight='semibold')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'success_by_stage.png'), bbox_inches='tight')
                    plt.close()
            
            # 4. Industry success rates
            if 'industry' in classified_data.columns:
                plt.figure(figsize=(14, 10))
                
                # Calculate success rate by industry and get top 15
                industry_success = classified_data.groupby('industry')['is_likely_success'].mean() * 100
                industry_counts = classified_data['industry'].value_counts()
                
                # Only include industries with enough companies
                min_companies = 2  # Reduced from 3 to ensure we have some data
                valid_industries = industry_counts[industry_counts >= min_companies].index
                
                if len(valid_industries) > 0:
                    valid_industry_success = industry_success[valid_industries].sort_values(ascending=False).head(15)
                    
                    # Create barplot
                    ax = sns.barplot(x=valid_industry_success.values, y=valid_industry_success.index)
                    
                    # Add count labels
                    for i, industry in enumerate(valid_industry_success.index):
                        count = industry_counts[industry]
                        plt.text(valid_industry_success[industry] + 1, i, f'n={count}', va='center')
                    
                    plt.title('Success Rate by Industry (Top 15)')
                    plt.xlabel('Success Rate (%)')
                    plt.ylabel('Industry')
                    plt.xlim(0, 105)  # Give space for labels
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'industry_success_rates.png'), bbox_inches='tight')
                    plt.close()
            
            # 7. Top potential companies
            if 'company_name' in classified_data.columns:
                plt.figure(figsize=(14, 8))
                
                # Get top 20 companies by success score or fewer if less available
                n_companies = min(20, len(classified_data))
                top_companies = classified_data.nlargest(n_companies, 'success_score')
                
                # Create barplot
                ax = sns.barplot(x='success_score', y='company_name', data=top_companies)
                
                # Add stage labels if available
                if 'funding_stage' in top_companies.columns:
                    for i, company in enumerate(top_companies['company_name']):
                        stage = top_companies.loc[top_companies['company_name'] == company, 'funding_stage'].values[0]
                        score = top_companies.loc[top_companies['company_name'] == company, 'success_score'].values[0]
                        plt.text(score + 1, i, f'{stage}', va='center')
                
                plt.title('Top Companies by Success Potential')
                plt.xlabel('Success Score')
                plt.ylabel('Company')
                plt.xlim(0, 105)  # Give space for labels
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'top_potential_companies.png'), bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Generated visualizations successfully in {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def generate_report(self, classified_data=None, output_file=None):
        """
        Generate a comprehensive report of the success/failure predictions.
        
        Args:
            classified_data (pandas.DataFrame): Data with classifications
            output_file (str): File to save the report
            
        Returns:
            bool: Whether report generation was successful
        """
        self.logger.info("Generating success/failure prediction report")
        
        try:
            # Use classified data if none provided
            if classified_data is None:
                self.logger.warning("No classified data provided, running classification")
                classified_data = self.classify_startups()
            
            # If the dataframe is empty, we can't create a report
            if classified_data.empty:
                self.logger.error("No data available for report generation")
                return False
                
            # Set output file
            if output_file is None:
                output_file = os.path.join(self.output_dir, 'success_failure_report.txt')
            
            # Create the report
            with open(output_file, 'w') as f:
                # Report header
                f.write("==========================================================\n")
                f.write("          STARTUP SUCCESS/FAILURE PREDICTION REPORT       \n")
                f.write("==========================================================\n\n")
                
                # Overview section
                f.write("EXECUTIVE SUMMARY\n")
                f.write("----------------\n")
                
                # Success rate
                success_rate = classified_data['is_likely_success'].mean() * 100
                f.write(f"Overall success prediction rate: {success_rate:.1f}%\n")
                
                # Classification breakdown
                classification_counts = classified_data['success_classification'].value_counts()
                total_companies = len(classified_data)
                
                f.write("\nClassification breakdown:\n")
                for category, count in classification_counts.items():
                    percentage = count / total_companies * 100
                    f.write(f"- {category}: {count} companies ({percentage:.1f}%)\n")
                
                # Risk levels if available
                if 'risk_level' in classified_data.columns:
                    risk_counts = classified_data['risk_level'].value_counts()
                    f.write("\nRisk level distribution:\n")
                    for risk, count in risk_counts.items():
                        percentage = count / total_companies * 100
                        f.write(f"- {risk}: {count} companies ({percentage:.1f}%)\n")
                
                # Industry insights
                if 'industry' in classified_data.columns:
                    f.write("\n\nINDUSTRY INSIGHTS\n")
                    f.write("----------------\n")
                    
                    # Calculate success rate by industry
                    industry_success = classified_data.groupby('industry')['is_likely_success'].mean() * 100
                    industry_counts = classified_data['industry'].value_counts()
                    
                    # Only include industries with enough companies
                    min_companies = 2  # Reduced from 3 to ensure we have some data
                    valid_industries = industry_counts[industry_counts >= min_companies].index
                    
                    if len(valid_industries) > 0:
                        valid_industry_success = industry_success[valid_industries].sort_values(ascending=False)
                        
                        # Top performing industries
                        f.write("Top performing industries:\n")
                        for i, (industry, rate) in enumerate(valid_industry_success.head(5).items()):
                            count = industry_counts[industry]
                            f.write(f"{i+1}. {industry}: {rate:.1f}% success rate (based on {count} companies)\n")
                        
                        # Bottom performing industries
                        if len(valid_industry_success) > 5:
                            f.write("\nBottom performing industries:\n")
                            for i, (industry, rate) in enumerate(valid_industry_success.tail(5).items()):
                                count = industry_counts[industry]
                                f.write(f"{i+1}. {industry}: {rate:.1f}% success rate (based on {count} companies)\n")
                
                # Top potential companies
                f.write("\n\nTOP POTENTIAL COMPANIES\n")
                f.write("----------------------\n")
                
                # Limit to top 10 or fewer if less available
                n_companies = min(10, len(classified_data))
                top_companies = classified_data.nlargest(n_companies, 'success_score')
                
                for i, (_, company) in enumerate(top_companies.iterrows()):
                    f.write(f"{i+1}. {company['company_name']}")
                    
                    if 'funding_stage' in company:
                        f.write(f" ({company['funding_stage']})")
                    
                    f.write(f": {company['success_score']:.1f} success score")
                    
                    if 'success_classification' in company:
                        f.write(f" - {company['success_classification']}")
                    
                    f.write("\n")
                
                # High risk companies
                f.write("\n\nHIGHEST RISK COMPANIES\n")
                f.write("---------------------\n")
                
                # Limit to bottom 10 or fewer if less available
                high_risk_companies = classified_data.nsmallest(n_companies, 'success_score')
                
                for i, (_, company) in enumerate(high_risk_companies.iterrows()):
                    f.write(f"{i+1}. {company['company_name']}")
                    
                    if 'funding_stage' in company:
                        f.write(f" ({company['funding_stage']})")
                    
                    f.write(f": {company['success_score']:.1f} success score")
                    
                    if 'success_classification' in company:
                        f.write(f" - {company['success_classification']}")
                    
                    f.write("\n")
                
                # Report footer
                f.write("\n\n==========================================================\n")
                f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("==========================================================\n")
                
            self.logger.info(f"Generated report successfully: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def visualize_model_performance(self, classified_data=None, output_dir=None):
        """
        Generate advanced model performance visualizations including ROC curves,
        calibration plots, and accuracy metrics.
        
        Args:
            classified_data (pandas.DataFrame): Data with classifications
            output_dir (str): Directory to save visualizations
            
        Returns:
            bool: Whether visualization was successful
        """
        self.logger.info("Generating model performance visualizations")
        
        try:
            # Use classified data if none provided
            if classified_data is None:
                self.logger.warning("No classified data provided, running classification")
                classified_data = self.classify_startups()
            
            # If the dataframe is empty, we can't create visualizations
            if classified_data.empty:
                self.logger.error("No data available for visualization")
                return False
            
            # Set output directory
            if output_dir is None:
                output_dir = os.path.join(self.output_dir, 'performance_metrics')
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Split data for validation
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                         f1_score, confusion_matrix)
            
            # We need to create binary labels for evaluation
            data = classified_data.copy()
            
            # Create feature matrix with available data
            feature_columns = []
            for col in ['stage_transition_score', 'survival_score', 'funding_adequacy_score', 
                        'industry_momentum_score', 'anomaly_normalized_score', 'funding_recency_score',
                        'maturity_score', 'funding_momentum', 'industry_success_likelihood', 'funding_tier']:
                if col in data.columns:
                    feature_columns.append(col)
            
            if len(feature_columns) < 3:
                self.logger.error("Not enough features for model evaluation")
                return False
            
            X = data[feature_columns].values
            y = data['is_likely_success'].astype(int).values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Train multiple models
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from xgboost import XGBClassifier
            
            # Create a dictionary to store models
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            }
            
            # Train and evaluate each model
            model_metrics = {}
            model_predictions = {}
            model_probabilities = {}
            
            for name, model in models.items():
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Get probabilities if the model supports predict_proba
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    # Use decision function if available (like for SVM)
                    if hasattr(model, 'decision_function'):
                        y_proba = model.decision_function(X_test)
                    else:
                        y_proba = y_pred.astype(float)
                
                # Store predictions and probabilities
                model_predictions[name] = y_pred
                model_probabilities[name] = y_proba
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                }
                
                model_metrics[name] = metrics
                
                # Log metrics
                self.logger.info(f"Model: {name}")
                for metric_name, value in metrics.items():
                    self.logger.info(f"  {metric_name}: {value:.4f}")
            
            # 1. Create ROC curves
            plt.figure(figsize=(12, 8))
            
            from sklearn.metrics import roc_curve, auc
            
            for name, y_proba in model_probabilities.items():
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
            
            # Plot diagonal line
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'roc_curves.png'), bbox_inches='tight')
            plt.close()
            
            # 2. Create precision-recall curves
            plt.figure(figsize=(12, 8))
            
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            for name, y_proba in model_probabilities.items():
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                avg_precision = average_precision_score(y_test, y_proba)
                plt.plot(recall, precision, lw=2, label=f'{name} (AP = {avg_precision:.4f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), bbox_inches='tight')
            plt.close()
            
            # 3. Create calibration plots
            plt.figure(figsize=(12, 8))
            
            from sklearn.calibration import calibration_curve
            
            for name, y_proba in model_probabilities.items():
                prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
                plt.plot(prob_pred, prob_true, marker='o', lw=2, label=name)
            
            # Plot the perfectly calibrated line
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly calibrated')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('Mean predicted probability')
            plt.ylabel('Fraction of positives')
            plt.title('Calibration Curves')
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'calibration_curves.png'), bbox_inches='tight')
            plt.close()
            
            # 4. Create confusion matrices
            from sklearn.metrics import ConfusionMatrixDisplay
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, (name, model) in enumerate(models.items()):
                if i < len(axes):
                    y_pred = model_predictions[name]
                    ConfusionMatrixDisplay.from_predictions(
                        y_test, 
                        y_pred,
                        display_labels=['Failure', 'Success'],
                        ax=axes[i],
                        cmap='Blues',
                        normalize='true'
                    )
                    axes[i].set_title(f'{name} Confusion Matrix')
                    axes[i].grid(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), bbox_inches='tight')
            plt.close()
            
            # 5. Create model comparison bar chart
            plt.figure(figsize=(14, 8))
            
            metrics_df = pd.DataFrame(model_metrics).T
            metrics_df = metrics_df.reset_index().rename(columns={'index': 'model'})
            metrics_df = pd.melt(metrics_df, id_vars=['model'], var_name='metric', value_name='value')
            
            sns.barplot(x='model', y='value', hue='metric', data=metrics_df)
            plt.title('Model Performance Comparison')
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.legend(title='Metric')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'model_comparison.png'), bbox_inches='tight')
            plt.close()
            
            # Create an ensemble model using the best performing models
            # Use test set predictions to create ensemble weights
            weights = {name: metrics['f1'] for name, metrics in model_metrics.items()}
            total_weight = sum(weights.values())
            weights = {name: weight/total_weight for name, weight in weights.items()}
            
            self.logger.info("Ensemble model weights:")
            for name, weight in weights.items():
                self.logger.info(f"  {name}: {weight:.4f}")
            
            # Apply the ensemble to the entire dataset
            ensemble_proba = np.zeros(len(data))
            
            for name, model in models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(data[feature_columns].values)[:, 1]
                else:
                    if hasattr(model, 'decision_function'):
                        proba = model.decision_function(data[feature_columns].values)
                    else:
                        proba = model.predict(data[feature_columns].values).astype(float)
                
                ensemble_proba += proba * weights[name]
            
            # Add ensemble probabilities to the data
            data['ensemble_probability'] = ensemble_proba
            data['ensemble_prediction'] = (ensemble_proba >= 0.5).astype(int)
            
            # Calculate accuracy metrics for the ensemble
            ensemble_accuracy = accuracy_score(data['is_likely_success'].astype(int), data['ensemble_prediction'])
            self.logger.info(f"Ensemble model accuracy: {ensemble_accuracy:.4f}")
            
            # Save model metrics to file
            metrics_df.to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)
            
            # Save ensemble probabilities
            ensemble_df = data[['company_name', 'success_score', 'ensemble_probability', 'ensemble_prediction', 'is_likely_success']]
            ensemble_df.to_csv(os.path.join(output_dir, 'ensemble_predictions.csv'), index=False)
            
            # Create a final visualization showing ensemble vs original
            plt.figure(figsize=(12, 8))
            
            # Calculate histogram
            plt.hist([data['success_score']/100, ensemble_proba], bins=20, 
                    label=['Original Score', 'Ensemble Probability'], alpha=0.6)
            
            plt.xlabel('Success Score / Probability')
            plt.ylabel('Count')
            plt.title('Original Success Score vs Ensemble Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'ensemble_comparison.png'), bbox_inches='tight')
            plt.close()
            
            self.logger.info("Model performance visualizations created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating model performance visualizations: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def generate_calibration_plot(self, classified_data=None, output_file=None):
        """
        Generate a standalone calibration plot showing how well the predicted probabilities
        match the actual observed frequencies.
        
        Args:
            classified_data (pandas.DataFrame): Data with classifications
            output_file (str): File path to save the calibration plot
            
        Returns:
            bool: Whether calibration plot generation was successful
        """
        self.logger.info("Generating standalone calibration plot")
        
        try:
            # Use classified data if none provided
            if classified_data is None:
                self.logger.warning("No classified data provided, running classification")
                classified_data = self.classify_startups()
            
            # If the dataframe is empty, we can't create a calibration plot
            if classified_data.empty:
                self.logger.error("No data available for calibration plot")
                return False
            
            # Set output file
            if output_file is None:
                output_dir = os.path.join(self.output_dir, 'visualizations')
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, 'calibration_plot.png')
            
            # Create feature matrix with available data
            feature_columns = []
            for col in ['stage_transition_score', 'survival_score', 'funding_adequacy_score', 
                        'industry_momentum_score', 'anomaly_normalized_score', 'funding_recency_score',
                        'maturity_score', 'funding_momentum', 'industry_success_likelihood', 'funding_tier']:
                if col in classified_data.columns:
                    feature_columns.append(col)
            
            if len(feature_columns) < 3:
                self.logger.error("Not enough features for calibration plot")
                return False
            
            # Create binary labels
            data = classified_data.copy()
            
            # Handle missing values by using imputer
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            
            # Extract features and impute missing values
            X_with_na = data[feature_columns].values
            X = imputer.fit_transform(X_with_na)
            y = data['is_likely_success'].astype(int).values
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Train multiple models that can handle missing values
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import HistGradientBoostingClassifier
            
            # Create dictionary of models
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42)
            }
            
            # Train models and get probabilities
            model_probabilities = {}
            
            for name, model in models.items():
                try:
                    # Train the model
                    model.fit(X_train, y_train)
                    
                    # Get probabilities
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)[:, 1]
                    else:
                        # Use decision function if available (like for SVM)
                        if hasattr(model, 'decision_function'):
                            y_proba = model.decision_function(X_test)
                        else:
                            y_proba = model.predict(X_test).astype(float)
                    
                    model_probabilities[name] = y_proba
                    self.logger.info(f"Successfully trained {name} model")
                except Exception as e:
                    self.logger.error(f"Error training {name} model: {str(e)}")
            
            # If no models were successfully trained, we can't create a calibration plot
            if not model_probabilities:
                self.logger.error("No models could be trained for calibration plot")
                return False
            
            # Create ensemble probability (only from successful models)
            ensemble_proba = np.zeros_like(y_test, dtype=float)
            for _, y_proba in model_probabilities.items():
                ensemble_proba += y_proba
            ensemble_proba /= len(model_probabilities)
            
            model_probabilities['Ensemble'] = ensemble_proba
            
            # Create enhanced calibration plot
            plt.figure(figsize=(12, 10))
            
            # Set up colormap for different models
            colors = plt.cm.tab10(np.linspace(0, 1, len(model_probabilities)))
            
            # Generate calibration curves
            for i, (name, y_proba) in enumerate(model_probabilities.items()):
                # Calculate calibration curve
                prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
                
                # Plot calibration curve
                plt.plot(prob_pred, prob_true, marker='o', linewidth=2, 
                         label=name, color=colors[i], markersize=8)
                
                # Calculate Brier score
                from sklearn.metrics import brier_score_loss
                brier = brier_score_loss(y_test, y_proba)
                self.logger.info(f"{name} Brier score: {brier:.4f}")
            
            # Plot perfectly calibrated line
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
            
            # Format the plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.grid(True, alpha=0.3)
            plt.xlabel('Mean predicted probability', fontsize=14)
            plt.ylabel('Fraction of positives (observed frequency)', fontsize=14)
            plt.title('Calibration Curves for Success/Failure Prediction Models', fontsize=16, fontweight='bold')
            plt.legend(loc='best', fontsize=12)
            
            # Add detailed annotation
            text_str = (
                "Perfect calibration: Points on diagonal\n"
                "Above diagonal: Model underestimates\n"
                "Below diagonal: Model overestimates"
            )
            plt.annotate(text_str, xy=(0.05, 0.75), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                         fontsize=12)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            self.logger.info(f"Calibration plot saved to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating calibration plot: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
            
    def enhance_predictions_with_ensemble(self, classified_data=None):
        """
        Enhance predictions using ensemble of multiple classification models.
        
        Args:
            classified_data (pandas.DataFrame): Data with initial classifications
            
        Returns:
            pandas.DataFrame: Enhanced data with ensemble predictions
        """
        self.logger.info("Enhancing predictions with ensemble models")
        
        try:
            # Use classified data if none provided
            if classified_data is None:
                self.logger.warning("No classified data provided, running classification")
                classified_data = self.classify_startups()
            
            # If the dataframe is empty, we can't create an ensemble
            if classified_data.empty:
                self.logger.error("No data available for ensemble prediction")
                return classified_data
            
            # Enhance the current predictions with a robust ensemble
            data = classified_data.copy()
            
            # Create feature matrix with available data
            feature_columns = []
            for col in ['stage_transition_score', 'survival_score', 'funding_adequacy_score', 
                        'industry_momentum_score', 'anomaly_normalized_score', 'funding_recency_score',
                        'maturity_score', 'funding_momentum', 'industry_success_likelihood', 'funding_tier']:
                if col in data.columns:
                    feature_columns.append(col)
            
            if len(feature_columns) < 3:
                self.logger.error("Not enough features for ensemble model")
                return classified_data
            
            X = data[feature_columns].values
            y = data['is_likely_success'].astype(int).values
            
            # Split data for cross-validation
            from sklearn.model_selection import StratifiedKFold
            
            # Initialize models
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
            from sklearn.linear_model import LogisticRegression
            from xgboost import XGBClassifier
            
            # Create individual models
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            gb = GradientBoostingClassifier(random_state=42)
            lr = LogisticRegression(random_state=42, max_iter=1000)
            xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            
            # Create voting ensemble
            ensemble = VotingClassifier(
                estimators=[
                    ('rf', rf),
                    ('gb', gb),
                    ('lr', lr),
                    ('xgb', xgb)
                ],
                voting='soft'  # Use probability estimates
            )
            
            # Train the ensemble with cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            from sklearn.metrics import accuracy_score, f1_score
            
            # Initialize arrays to store predictions
            ensemble_proba = np.zeros_like(y, dtype=float)
            
            # Perform cross-validation
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train the ensemble
                ensemble.fit(X_train, y_train)
                
                # Make predictions on the test fold
                ensemble_proba[test_idx] = ensemble.predict_proba(X_test)[:, 1]
            
            # Now train on the entire dataset for future predictions
            ensemble.fit(X, y)
            
            # Add ensemble probabilities to the data
            data['ensemble_probability'] = ensemble_proba
            data['ensemble_prediction'] = (ensemble_proba >= 0.5).astype(int)
            
            # Calculate new risk categories based on ensemble probabilities
            data['ensemble_risk_category'] = pd.cut(
                data['ensemble_probability'],
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['High Risk', 'Moderate Risk', 'Moderate Potential', 'High Potential']
            )
            
            # Calculate ensemble performance metrics
            ensemble_accuracy = accuracy_score(y, data['ensemble_prediction'])
            ensemble_f1 = f1_score(y, data['ensemble_prediction'])
            
            self.logger.info(f"Ensemble model performance - Accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")
            
            # Save the ensemble model for future use
            import pickle
            model_dir = os.path.join(self.output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            with open(os.path.join(model_dir, 'ensemble_model.pkl'), 'wb') as f:
                pickle.dump(ensemble, f)
                
            with open(os.path.join(model_dir, 'feature_columns.pkl'), 'wb') as f:
                pickle.dump(feature_columns, f)
            
            self.logger.info("Ensemble model saved successfully")
            
            # Create comparison visualization
            plt.figure(figsize=(10, 6))
            
            # Plot original vs ensemble predictions
            sns.scatterplot(
                x='success_score', 
                y=data['ensemble_probability'] * 100, 
                hue='is_likely_success',
                data=data
            )
            
            plt.plot([0, 100], [0, 100], 'k--', alpha=0.5)  # Diagonal reference line
            plt.xlabel('Original Success Score')
            plt.ylabel('Ensemble Probability (%)')
            plt.title('Original Score vs Ensemble Probability')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'original_vs_ensemble.png'), bbox_inches='tight')
            plt.close()
            
            # Return the enhanced data
            return data
            
        except Exception as e:
            self.logger.error(f"Error enhancing predictions with ensemble: {str(e)}")
            self.logger.error(traceback.format_exc())
            return classified_data

    def run_analysis(self):
        """
        Run the complete success/failure prediction analysis pipeline.
        
        Returns:
            pandas.DataFrame: Classified data with success/failure predictions
        """
        self.logger.info("Starting success/failure prediction analysis")
        
        try:
            # Step 1: Initialize component models
            self.logger.info("Step 1: Initializing component models")
            self.initialize_components()
            
            # Step 2: Run component analyses
            self.logger.info("Step 2: Running component analyses")
            component_results = self.run_components()
            
            # Step 3: Integrate data from all components
            self.logger.info("Step 3: Integrating data")
            integrated_data = self.integrate_data(component_results)
            
            # Step 4: Calculate success scores
            self.logger.info("Step 4: Calculating success scores")
            scored_data = self.calculate_success_scores(integrated_data)
            
            # Step 5: Classify startups
            self.logger.info("Step 5: Classifying startups")
            classified_data = self.classify_startups(scored_data)
            
            # Step 6: Generate visualizations
            self.logger.info("Step 6: Generating visualizations")
            self.visualize_results(classified_data)
            
            # Step 7: Create advanced model performance visualizations
            self.logger.info("Step 7: Creating advanced model performance visualizations")
            self.visualize_model_performance(classified_data)
            
            # NEW: Generate calibration plot
            self.logger.info("Generating standalone calibration plot")
            self.generate_calibration_plot(classified_data)
            
            # Step 8: Enhance predictions with ensemble models
            self.logger.info("Step 8: Enhancing predictions with ensemble models")
            enhanced_data = self.enhance_predictions_with_ensemble(classified_data)
            
            # Step 9: Generate report (now with enhanced data)
            self.logger.info("Step 9: Generating report")
            self.generate_report(enhanced_data)
            
            # Save the enhanced data
            output_csv = os.path.join(self.output_dir, 'success_failure_predictions.csv')
            enhanced_data.to_csv(output_csv, index=False)
            self.logger.info(f"Saved predictions to: {output_csv}")
            
            # Calculate final metrics
            success_count = enhanced_data['is_likely_success'].sum()
            total_count = len(enhanced_data)
            success_rate = success_count / total_count * 100 if total_count > 0 else 0
            
            # Calculate distribution by categories
            high_success = enhanced_data['success_classification'].value_counts().get('High Success Potential', 0)
            moderate_success = enhanced_data['success_classification'].value_counts().get('Moderate Success Potential', 0)
            moderate_failure = enhanced_data['success_classification'].value_counts().get('Moderate Failure Risk', 0)
            high_failure = enhanced_data['success_classification'].value_counts().get('High Failure Risk', 0)
            
            # Calculate percentages
            high_success_pct = high_success / total_count * 100 if total_count > 0 else 0
            moderate_success_pct = moderate_success / total_count * 100 if total_count > 0 else 0
            moderate_failure_pct = moderate_failure / total_count * 100 if total_count > 0 else 0
            high_failure_pct = high_failure / total_count * 100 if total_count > 0 else 0
            
            # Log detailed metrics
            self.logger.info(f"Analysis complete. Overall success rate: {success_rate:.1f}%")
            self.logger.info(f"Total companies analyzed: {total_count}")
            self.logger.info(f"Predicted successful: {success_count} ({success_rate:.1f}%)")
            self.logger.info(f"Predicted unsuccessful: {total_count - success_count} ({100 - success_rate:.1f}%)")
            
            # Log detailed breakdown
            self.logger.info("Classification breakdown:")
            self.logger.info(f"- High Success Potential: {high_success} ({high_success_pct:.1f}%)")
            self.logger.info(f"- Moderate Success Potential: {moderate_success} ({moderate_success_pct:.1f}%)")
            self.logger.info(f"- Moderate Failure Risk: {moderate_failure} ({moderate_failure_pct:.1f}%)")
            self.logger.info(f"- High Failure Risk: {high_failure} ({high_failure_pct:.1f}%)")
            
            # Calculate industry-specific success rates
            if 'industry' in enhanced_data.columns:
                industry_success = enhanced_data.groupby('industry')['is_likely_success'].mean() * 100
                industry_counts = enhanced_data['industry'].value_counts()
                
                # Get top and bottom industries
                top_industries = industry_success.nlargest(3)
                bottom_industries = industry_success.nsmallest(3)
                
                # Log industry insights
                self.logger.info("Top performing industries:")
                for industry, rate in top_industries.items():
                    count = industry_counts.get(industry, 0)
                    self.logger.info(f"- {industry}: {rate:.1f}% success rate (based on {count} companies)")
                    
                self.logger.info("Lowest performing industries:")
                for industry, rate in bottom_industries.items():
                    count = industry_counts.get(industry, 0)
                    self.logger.info(f"- {industry}: {rate:.1f}% success rate (based on {count} companies)")
            
            # Additional insights on funding stages
            if 'funding_stage' in enhanced_data.columns:
                stage_success = enhanced_data.groupby('funding_stage')['is_likely_success'].mean() * 100
                stage_counts = enhanced_data['funding_stage'].value_counts()
                
                # Get top stages with meaningful sample size
                valid_stages = stage_counts[stage_counts >= 5].index
                if len(valid_stages) > 0:
                    valid_stage_success = stage_success[valid_stages].sort_values(ascending=False)
                    
                    self.logger.info("Success rate by funding stage:")
                    for stage, rate in valid_stage_success.items():
                        count = stage_counts.get(stage, 0)
                        self.logger.info(f"- {stage}: {rate:.1f}% success rate (based on {count} companies)")
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Error running analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

def main():
    """Run the success/failure prediction analysis."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run startup success/failure prediction analysis')
    parser.add_argument('--data-dir', type=str, help='Directory containing funding data JSON files')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files and visualizations')
    args = parser.parse_args()
    
    # Create and run the prediction system
    predictor = SuccessFailurePrediction(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    results = predictor.run_analysis()
    
    if not results.empty:
        print(f"Analysis complete. Results saved to: {predictor.output_dir}")
        print(f"Total companies analyzed: {len(results)}")
        print(f"Success rate: {results['is_likely_success'].mean() * 100:.1f}%")
    else:
        print("Analysis failed. See log for details.")

if __name__ == "__main__":
    main() 