import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import joblib
import traceback
import warnings
import re
import scipy.stats as stats
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
from MLPredictiveAnalysis.data_loader import FundingDataLoader
from typing import Dict, List, Optional, Tuple, Union
import sys
import pickle
import time
import threading
import argparse

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("funding_continuation.log"),
                             logging.StreamHandler()])
logger = logging.getLogger(__name__)

class FundingContinuationAnalysis:
    """
    Analyze and predict startup funding continuation using survival analysis.
    Implements section 2.2 of the funding stage prediction system.
    
    Objective: Predict likelihood of securing next funding round within 18 months
    Approach:
    - Features: Funding velocity, burn rate, market comparables
    - Model: Cox Proportional Hazards with time-varying covariates
    - Output: Survival curves with confidence intervals
    
    h(t∣X)=h0(t)exp⁡(β1X1+β2X2) - Baseline hazard function with covariate effects
    """
    def __init__(self, data_dir=None, output_dir=None):
        """
        Initialize the FundingContinuationAnalysis class.
        
        Args:
            data_dir (str): Directory containing the funding data files
            output_dir (str): Directory to save output files and visualizations
        """
        self.data_dir = data_dir or "JSONFolder"
        self.output_dir = output_dir or "./outputContinuation"
        
        # Data attributes
        self.data = None
        self.survival_data = None
        
        # Model attributes
        self.kmf = None
        self.cox_model = None
        self.cph = None
        self.is_fitted = False
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        self.kmf = KaplanMeierFitter()
        self.fit_timestamp = None
        self.feature_names = None
        self.duration_col = None
        self.event_col = None
        
        # Set up matplotlib style
        plt.style.use('default')  # Use default style instead of seaborn
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    def _setup_logging(self):
        """Set up logging configuration for the analysis."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'funding_continuation.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data_from_json_files(self, base_dir="."):
        """Load data from the three JSON files and merge them"""
        self.logger.info("Loading data from JSON files")
        
        # Define paths to source files in JSONFolder
        json_folder = os.path.join(base_dir, "JSONFolder")
        fundraiser_path = os.path.join(json_folder, "fundraisestartup50.json")
        growthlist_path = os.path.join(json_folder, "growthlistscrapper.json")
        topstartup_path = os.path.join(json_folder, "topstartupio50.json")
        
        # Verify files exist
        for path in [fundraiser_path, growthlist_path, topstartup_path]:
            if not os.path.exists(path):
                self.logger.error(f"Required file not found: {path}")
                return pd.DataFrame()
        
        try:
            # Load data from JSON files
            # Fundraiser data
            with open(fundraiser_path, 'r') as file:
                fundraiser_data = json.load(file)
            fundraiser_df = pd.DataFrame(fundraiser_data.get('companies', []))
            self.logger.info(f"Loaded {len(fundraiser_df)} records from fundraiser data")
            self.logger.debug(f"Fundraiser columns: {fundraiser_df.columns.tolist()}")
            
            # Growthlist data
            with open(growthlist_path, 'r') as file:
                growthlist_data = json.load(file)
            growthlist_df = pd.DataFrame(growthlist_data)
            self.logger.info(f"Loaded {len(growthlist_df)} records from growthlist data")
            self.logger.debug(f"Growthlist columns: {growthlist_df.columns.tolist()}")
            
            # Topstartup data - parse funding information
            with open(topstartup_path, 'r') as file:
                topstartup_data = json.load(file)
            topstartup_df = pd.DataFrame(topstartup_data)
            self.logger.info(f"Loaded {len(topstartup_df)} records from topstartup data")
            self.logger.debug(f"Topstartup columns: {topstartup_df.columns.tolist()}")
            
            # Parse funding information from topstartup data
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
                    # Convert K, M, B to actual numbers
                    if amount.endswith('K'):
                        amount = float(amount[:-1]) * 1000
                    elif amount.endswith('M'):
                        amount = float(amount[:-1]) * 1000000
                    elif amount.endswith('B'):
                        amount = float(amount[:-1]) * 1000000000
                    else:
                        amount = float(amount)
                
                # Extract stage
                stage_match = re.search(r'(Seed|Series [A-Z]|Angel)', funding_str)
                if stage_match:
                    stage = stage_match.group(1)
                
                # Extract date
                date_match = re.search(r'in (\d{4})', funding_str)
                if date_match:
                    date = f"{date_match.group(1)}-01-01"  # Default to January 1st
                    
                return pd.Series({'amount': amount, 'stage': stage, 'date': date})
            
            # Apply parsing to funding column
            if 'funding' in topstartup_df.columns:
                funding_info = topstartup_df['funding'].apply(parse_funding_info)
                topstartup_df['funding_amount'] = funding_info['amount']
                topstartup_df['funding_stage'] = funding_info['stage']
                topstartup_df['funding_date'] = funding_info['date']
                self.logger.debug("Successfully parsed funding information from topstartup data")
            
            # Standardize column names
            column_mapping = {
                # Fundraiser columns
                'Company': 'company_name',
                'Funding_Type': 'funding_stage',
                'Funding_Amount_USD': 'funding_amount',
                'Funding_Date': 'funding_date',
                'Industry': 'industry',
                'Total_Employees': 'employees',
                
                # Growthlist columns
                'name': 'company_name',
                'funding_type': 'funding_stage',
                'last_funding_date': 'funding_date',
                'funding_usd': 'funding_amount',
                'industry': 'industry',
                
                # Topstartup columns
                'category': 'industry',
                'name': 'company_name',
                'employees': 'employees'
            }
            
            # Apply mapping to each dataframe
            for df in [fundraiser_df, growthlist_df, topstartup_df]:
                existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
                df.rename(columns=existing_columns, inplace=True)
                self.logger.debug(f"After mapping, columns: {df.columns.tolist()}")
            
            # Convert funding amounts to numeric
            def clean_funding_amount(amount_series):
                if amount_series is None:
                    return None
                if isinstance(amount_series, pd.DataFrame):
                    # If we got a DataFrame, extract the first column
                    amount_series = amount_series.iloc[:, 0]
                return pd.to_numeric(
                    amount_series.astype(str)
                    .str.replace('$', '', regex=False)
                    .str.replace(',', '', regex=False)
                    .str.replace('K', '000', regex=False)
                    .str.replace('M', '000000', regex=False)
                    .str.replace('B', '000000000', regex=False),
                    errors='coerce'
                )

            for df in [fundraiser_df, growthlist_df, topstartup_df]:
                # Convert funding amounts to numeric
                if 'funding_amount' in df.columns:
                    df['funding_amount'] = clean_funding_amount(df[['funding_amount']])
                if 'Funding_Amount_USD' in df.columns:
                    df['funding_amount'] = clean_funding_amount(df[['Funding_Amount_USD']])
                    df = df.drop('Funding_Amount_USD', axis=1)
                if 'funding_usd' in df.columns:
                    df['funding_amount'] = clean_funding_amount(df[['funding_usd']])
                    df = df.drop('funding_usd', axis=1)
            
            # Convert dates to datetime
            for df in [fundraiser_df, growthlist_df, topstartup_df]:
                if 'funding_date' in df.columns:
                    df['funding_date'] = pd.to_datetime(df['funding_date'], errors='coerce')
            
            # Combine all records
            all_records = []
            
            # Process each dataframe
            required_columns = ['company_name', 'funding_stage', 'funding_amount', 'funding_date', 'industry', 'employees']
            
            for df in [fundraiser_df, growthlist_df, topstartup_df]:
                self.logger.debug(f"Processing dataframe with columns: {df.columns.tolist()}")
                for _, row in df.iterrows():
                    if pd.notna(row.get('company_name')):
                        record = {}
                        for col in required_columns:
                            record[col] = row.get(col, None)  # Use None as default
                        all_records.append(record)
            
            # Create merged dataframe
            merged_data = pd.DataFrame(all_records)
            self.logger.debug(f"Created merged dataframe with columns: {merged_data.columns.tolist()}")
            
            # Remove duplicates based on company and funding date
            merged_data = merged_data.drop_duplicates(
                subset=['company_name', 'funding_date']
            ).reset_index(drop=True)
            
            # Sort by company and funding date
            merged_data = merged_data.sort_values(
                ['company_name', 'funding_date']
            ).reset_index(drop=True)
            
            # Fill missing values
            merged_data = merged_data.fillna({
                'industry': 'Unknown',
                'employees': 0,
                'funding_stage': 'Unknown',
                'funding_amount': 0
            })
            
            # Convert employees to numeric
            merged_data['employees'] = pd.to_numeric(
                merged_data['employees'].astype(str)
                .str.extract(r'(\d+)', expand=False),
                errors='coerce'
            ).fillna(0)
            
            # Clean up funding stages
            stage_mapping = {
                'SEED': 'Seed',
                'seed': 'Seed',
                'Seed Round': 'Seed',
                'SERIES A': 'Series A',
                'series a': 'Series A',
                'Series-A': 'Series A',
                'SERIES B': 'Series B',
                'series b': 'Series B',
                'Series-B': 'Series B',
                'SERIES C': 'Series C',
                'series c': 'Series C',
                'Series-C': 'Series C',
                'ANGEL': 'Angel',
                'angel': 'Angel',
                'Angel Round': 'Angel'
            }
            merged_data['funding_stage'] = merged_data['funding_stage'].map(stage_mapping).fillna(merged_data['funding_stage'])
            
            # Clean up industry categories
            industry_mapping = {
                'AI': 'Artificial Intelligence',
                'ML': 'Artificial Intelligence',
                'SAAS': 'Software',
                'SaaS': 'Software',
                'SOFTWARE': 'Software',
                'Tech': 'Technology',
                'TECH': 'Technology',
                'FinTech': 'Financial Technology',
                'FINTECH': 'Financial Technology',
                'Health': 'Healthcare',
                'HEALTH': 'Healthcare',
                'E-commerce': 'eCommerce',
                'Ecommerce': 'eCommerce',
                'ECOMMERCE': 'eCommerce'
            }
            merged_data['industry'] = merged_data['industry'].map(industry_mapping).fillna(merged_data['industry'])
            
            # Remove records with invalid dates
            merged_data = merged_data.dropna(subset=['funding_date']).reset_index(drop=True)
            
            # Add funding sequence number for each company
            merged_data['funding_sequence'] = merged_data.groupby('company_name').cumcount() + 1
            
            # Calculate time since first funding for each company
            merged_data['first_funding_date'] = merged_data.groupby('company_name')['funding_date'].transform('min')
            merged_data['days_since_first_funding'] = (merged_data['funding_date'] - merged_data['first_funding_date']).dt.days
            
            # Calculate funding growth rate
            merged_data = merged_data.sort_values(['company_name', 'funding_date']).reset_index(drop=True)
            merged_data['prev_funding_amount'] = merged_data.groupby('company_name')['funding_amount'].shift(1)
            merged_data['funding_growth_rate'] = merged_data['funding_amount'] / merged_data['prev_funding_amount']
            
            # Handle infinite and missing values
            merged_data.loc[merged_data['funding_growth_rate'].isin([np.inf, -np.inf]), 'funding_growth_rate'] = 1
            merged_data['funding_growth_rate'] = merged_data['funding_growth_rate'].fillna(1)
            
            # Calculate time between funding rounds
            merged_data['prev_funding_date'] = merged_data.groupby('company_name')['funding_date'].shift(1)
            merged_data['days_between_rounds'] = (merged_data['funding_date'] - merged_data['prev_funding_date']).dt.days
            merged_data['days_between_rounds'] = merged_data['days_between_rounds'].fillna(0)
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['funding_amount', 'funding_growth_rate', 'days_between_rounds']
            for col in numeric_columns:
                merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')
            
            # Calculate industry averages
            industry_stats = merged_data.groupby('industry').agg({
                'funding_amount': ['mean', 'median'],
                'funding_growth_rate': ['mean', 'median'],
                'days_between_rounds': ['mean', 'median']
            }).round(2)
            
            # Flatten column names
            industry_stats.columns = [f"{col[0]}_{col[1]}" for col in industry_stats.columns]
            
            # Add industry stats back to main dataframe
            for stat in industry_stats.columns:
                merged_data[f'industry_{stat}'] = merged_data['industry'].map(industry_stats[stat])
            
            # Calculate relative metrics
            merged_data['amount_vs_industry'] = merged_data['funding_amount'] / merged_data['industry_funding_amount_mean']
            merged_data['velocity_vs_industry'] = merged_data['industry_days_between_rounds_mean'] / merged_data['days_between_rounds']
            
            # Fill missing relative metrics with 1 (indicating average performance)
            merged_data['amount_vs_industry'] = merged_data['amount_vs_industry'].fillna(1)
            merged_data['velocity_vs_industry'] = merged_data['velocity_vs_industry'].fillna(1)
            
            # Create funding stage numeric mapping
            funding_stages = {
                'pre-seed': 0,
                'seed': 1,
                'angel': 2,
                'series a': 3,
                'a': 3,
                'series b': 4,
                'b': 4,
                'series c': 5,
                'c': 5,
                'series d': 6,
                'd': 6,
                'series e': 7,
                'e': 7,
                'series f': 8,
                'f': 8,
                'series g': 9,
                'g': 9,
                'series h': 10,
                'h': 10,
                'private equity': 11,
                'ipo': 12
            }
            
            # Apply stage mapping with case insensitivity
            merged_data['funding_stage_numeric'] = merged_data['funding_stage'].str.lower().map(funding_stages).fillna(-1)
            
            # Fill missing stage values
            merged_data.loc[merged_data['funding_stage_numeric'] == -1, 'funding_stage_numeric'] = 1  # Default to Seed
            
            # Log data quality metrics
            self.logger.info(f"Data quality metrics:")
            self.logger.info(f"Total records: {len(merged_data)}")
            self.logger.info(f"Unique companies: {merged_data['company_name'].nunique()}")
            self.logger.info(f"Missing values:\n{merged_data.isnull().sum()}")
            self.logger.info(f"Funding stages distribution:\n{merged_data['funding_stage'].value_counts()}")
            self.logger.info(f"Industry distribution:\n{merged_data['industry'].value_counts().head()}")
            
            # Assign the merged data to self.data
            self.data = merged_data
            
            return merged_data
            
        except Exception as e:
            self.logger.error(f"Error loading or merging data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _parse_funding_amount(self, amount_str):
        """Convert funding amount strings (e.g., "$27.6M") to numeric values"""
        if not amount_str or pd.isna(amount_str) or amount_str == "":
            return np.nan
        
        try:
            # Remove currency symbol and commas
            amount_str = str(amount_str).replace('$', '').replace(',', '').strip()
            
            # Convert based on unit (M=million, B=billion, K=thousand)
            if 'B' in amount_str or 'b' in amount_str:
                return float(amount_str.replace('B', '').replace('b', '')) * 1e9
            elif 'M' in amount_str or 'm' in amount_str:
                return float(amount_str.replace('M', '').replace('m', '')) * 1e6
            elif 'K' in amount_str or 'k' in amount_str:
                return float(amount_str.replace('K', '').replace('k', '')) * 1e3
            else:
                return float(amount_str)
        except Exception:
            return np.nan

    def create_enhanced_features(self, data):
        """Create enhanced features for funding continuation analysis"""
        self.logger.info("Creating enhanced features")
        
        # Basic features
        features = data.copy()
        
        # Convert funding date to datetime
        features['funding_date'] = pd.to_datetime(features['funding_date'], errors='coerce')
        
        # Time-based features
        features['days_since_last_funding'] = (pd.Timestamp.now() - features['funding_date']).dt.days
        
        # Funding velocity features
        features['funding_velocity'] = features.groupby('company_name')['funding_amount'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # Industry-specific features
        industry_stats = features.groupby('industry').agg({
            'funding_amount': ['mean', 'std', 'count'],
            'days_since_last_funding': 'mean'
        }).reset_index()
        
        industry_stats.columns = ['industry', 'industry_avg_funding', 'industry_funding_std', 
                                'industry_company_count', 'industry_avg_funding_interval']
        
        features = features.merge(industry_stats, on='industry', how='left')
        
        # Market position features
        features['funding_rank'] = features.groupby('funding_date')['funding_amount'].rank(pct=True)
        features['industry_funding_rank'] = features.groupby(['industry', 'funding_date'])['funding_amount'].rank(pct=True)
        
        # Growth metrics
        features['funding_growth_rate'] = features.groupby('company_name')['funding_amount'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # Competition level
        features['industry_competition'] = features['industry_company_count'] / features['industry_company_count'].max()
        
        # Funding stage progression
        stage_order = {'Seed': 1, 'Series A': 2, 'Series B': 3, 'Series C': 4, 'Series D': 5}
        features['stage_level'] = features['funding_stage'].map(stage_order).fillna(0)
        
        # Interaction terms
        features['funding_velocity_stage'] = features['funding_velocity'] * features['stage_level']
        features['market_position'] = features['funding_rank'] * features['industry_competition']
        
        return features

    def prepare_survival_data(self, df):
        """
        Prepare survival data for analysis.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw features
            
        Returns:
            pd.DataFrame: DataFrame ready for survival analysis
        """
        try:
            # Create a copy to avoid modifying the original
            df_survival = df.copy()
            
            # Calculate current amount (latest funding amount for each company)
            df_survival['current_amount'] = df_survival.groupby('company_name')['funding_amount'].transform('last')
            
            # Calculate total funding for each company
            df_survival['total_funding'] = df_survival.groupby('company_name')['funding_amount'].transform('sum')
            
            # Calculate number of rounds for each company
            df_survival['num_rounds'] = df_survival.groupby('company_name')['funding_amount'].transform('count')
            
            # Calculate funding velocity (amount per day since first funding)
            df_survival['first_funding_date'] = df_survival.groupby('company_name')['funding_date'].transform('min')
            df_survival['days_since_first'] = (df_survival['funding_date'] - df_survival['first_funding_date']).dt.days
            # More efficient calculation without apply
            df_survival['funding_velocity'] = np.where(
                df_survival['days_since_first'] > 0,
                df_survival['total_funding'] / df_survival['days_since_first'],
                df_survival['total_funding']  # For cases where days_since_first is 0
            )
            
            # Calculate time to next funding or censoring
            df_survival = df_survival.sort_values(['company_name', 'funding_date'])
            df_survival['next_funding_date'] = df_survival.groupby('company_name')['funding_date'].shift(-1)
            df_survival['duration'] = (df_survival['next_funding_date'] - df_survival['funding_date']).dt.days
            
            # Handle censoring (companies without next funding)
            df_survival['event'] = ~df_survival['duration'].isna()
            df_survival['duration'] = df_survival['duration'].fillna(
                (pd.Timestamp.now() - df_survival['funding_date']).dt.days
            )
            
            # Ensure all numeric columns are properly formatted
            numeric_columns = ['current_amount', 'funding_velocity', 'total_funding', 'num_rounds', 'duration']
            for col in numeric_columns:
                df_survival[col] = pd.to_numeric(df_survival[col], errors='coerce')
            
            # Drop any remaining rows with NaN values
            df_survival = df_survival.dropna(subset=numeric_columns)
            
            # Log the number of companies after preparation
            self.logger.info(f"Prepared survival data for {len(df_survival)} companies")
            
            return df_survival
            
        except Exception as e:
            self.logger.error(f"Error preparing survival data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def fit_cox_ph(self, survival_data: pd.DataFrame) -> None:
        """
        Fit a Cox Proportional Hazards model to the survival data.
        
        Args:
            survival_data (pd.DataFrame): DataFrame containing survival data with columns:
                - duration: time to event
                - event: binary indicator (1 if event occurred, 0 if censored)
                - feature columns: various predictors
                - company_name: identifier for each company
                - funding_date: dates of funding rounds
        
        Raises:
            ValueError: If survival_data is None or empty
            Exception: For any other fitting errors
        """
        try:
            if survival_data is None or survival_data.empty:
                raise ValueError("Survival data is None or empty")
            
            self.logger.info("Starting Cox PH model fitting...")
            
            # Create a copy of the data for feature selection
            fit_data = survival_data.copy()
            
            # Get feature columns (exclude special columns, datetime columns, and non-numeric columns)
            excluded_cols = [
                'duration', 'event', 'company_name', 'funding_date', 
                'first_funding_date', 'prev_funding_date', 'next_funding_date'
            ]
            
            # Identify numeric columns
            numeric_cols = []
            for col in fit_data.columns:
                if col not in excluded_cols:
                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(fit_data[col]):
                        numeric_cols.append(col)
                    else:
                        self.logger.warning(f"Excluding non-numeric column from Cox PH model: {col}")
            
            # Prioritize a subset of the most important features to reduce collinearity
            key_features = [
                'funding_amount', 'funding_velocity', 'days_since_first_funding',
                'total_funding', 'num_rounds', 'funding_stage_numeric'
            ]
            
            # Filter to include only key features that exist in the data
            key_features = [col for col in key_features if col in numeric_cols]
            
            self.logger.info(f"Using key features: {key_features}")
            
            # Prepare data for fitting: only include duration, event, and numeric features
            fit_data = fit_data[['duration', 'event'] + key_features].copy()
            
            # Handle NaN values
            if fit_data.isnull().sum().sum() > 0:
                self.logger.warning(f"NaN values found in data. Filling with zeros.")
                fit_data = fit_data.fillna(0)
            
            # Handle infinity values
            inf_counts = np.isinf(fit_data).sum().sum()
            if inf_counts > 0:
                self.logger.warning(f"Found {inf_counts} infinity values. Replacing with large finite values.")
                # Replace +inf with large number and -inf with large negative number
                fit_data = fit_data.replace([np.inf, -np.inf], [1e10, -1e10])
            
            # Check for extreme values
            for col in key_features:
                if fit_data[col].abs().max() > 1e10:
                    self.logger.warning(f"Column {col} has extreme values. Capping at 1e10.")
                    fit_data[col] = fit_data[col].clip(-1e10, 1e10)
            
            # Standardize features to reduce numerical issues
            feature_scaler = StandardScaler()
            fit_data[key_features] = feature_scaler.fit_transform(fit_data[key_features])
            
            # Check for and handle collinearity
            corr_matrix = fit_data[key_features].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
            
            if to_drop:
                self.logger.warning(f"Dropping highly correlated features: {to_drop}")
                key_features = [f for f in key_features if f not in to_drop]
                fit_data = fit_data[['duration', 'event'] + key_features]
            
            # Initialize the model with regularization
            self.cph = CoxPHFitter(penalizer=0.1)
            
            # Fit the model
            self.cph.fit(
                df=fit_data,
                duration_col='duration',
                event_col='event',
                show_progress=True
            )
            
            # Log model statistics
            try:
                concordance = concordance_index(
                    fit_data['duration'],
                    -self.cph.predict_partial_hazard(fit_data),
                    fit_data['event']
                )
                
                self.logger.info(f"Model fitting completed successfully")
                self.logger.info(f"Concordance Index: {concordance:.3f}")
                self.logger.info(f"Log Likelihood: {self.cph.log_likelihood_:.3f}")
                self.logger.info(f"AIC: {self.cph.AIC_}")
                self.logger.info("\nModel Parameters:")
                self.logger.info(self.cph.print_summary())
            except Exception as stats_e:
                self.logger.warning(f"Could not calculate model statistics: {str(stats_e)}")
            
        except Exception as e:
            self.logger.error(f"Error fitting Cox PH model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def fit_kaplan_meier(self, survival_data):
        """
        Fit a Kaplan-Meier model to the survival data.
        
        Args:
            survival_data (pd.DataFrame): DataFrame prepared by prepare_survival_data
            
        Returns:
            lifelines.KaplanMeierFitter: Fitted Kaplan-Meier model
        """
        try:
            if survival_data is None or len(survival_data) == 0:
                self.logger.error("No valid survival data provided for Kaplan-Meier fitting")
                return None
            
            # Create and fit the model
            kmf = KaplanMeierFitter()
            kmf.fit(
                survival_data['duration'],
                survival_data['event']
            )
            
            # Log model summary
            self.logger.info("Kaplan-Meier Model Summary:")
            self.logger.info(f"Number of observations: {len(survival_data)}")
            
            # Check if median_survival_time exists
            if hasattr(kmf, 'median_survival_time_'):
                self.logger.info(f"Median survival time: {kmf.median_survival_time_:.2f} days")
            else:
                # Calculate approximate median survival time
                median_idx = np.argmin(abs(kmf.survival_function_.values - 0.5))
                median_time = kmf.survival_function_.index[median_idx]
                self.logger.info(f"Approximate median survival time: {median_time:.2f} days")
            
            return kmf
            
        except Exception as e:
            self.logger.error(f"Error fitting Kaplan-Meier model: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def predict_survival_probabilities(self, df, times=None):
        """
        Predict survival probabilities for new data using the fitted Cox PH model.
        
        Args:
            df (pd.DataFrame): DataFrame containing features for prediction
            times (list, optional): List of time points to predict probabilities for.
                                   Defaults to [30, 90, 180, 365] days.
        
        Returns:
            pd.DataFrame: DataFrame containing predicted survival probabilities
        """
        try:
            if self.cph is None:
                self.logger.error("Cox PH model not fitted. Please fit the model first.")
                return None
            
            if times is None:
                times = [30, 90, 180, 365]  # Default time points in days
            
            # Ensure all required features are present
            required_features = [
                'current_amount', 'funding_velocity', 
                'total_funding', 'num_rounds', 'duration'
            ]
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                self.logger.error(f"Missing required features: {missing_features}")
                return None
            
            # Make predictions
            predictions = self.cph.predict_survival_function(
                df[required_features],
                times=times
            )
            
            # Add company identifiers if available
            if 'company_name' in df.columns:
                predictions['company_name'] = df['company_name']
            
            self.logger.info(f"Generated survival predictions for {len(df)} companies")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting survival probabilities: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
            
    def predict_median_time_to_funding(self, data):
        """
        Predict the median time to next funding round for each company
        """
        if not self.is_fitted:
            self.logger.error("Model not fitted! Please fit the model before predicting.")
            return None
            
        self.logger.info(f"Predicting median time to funding for {len(data)} startups")
        try:
            predictions = self.cph.predict_median(data)
            predictions_df = pd.DataFrame({
                'median_days': predictions,
                'median_months': predictions / 30.44
            })
            self.logger.info("Median time-to-funding predictions generated")
            return predictions_df
        except Exception as e:
            self.logger.error(f"Error predicting median time to funding: {e}")
            self.logger.error(traceback.format_exc())
            return None
            
    def visualize_survival_curves(self, kmf, output_dir=None):
        """
        Visualize survival curves with confidence intervals.
        
        Args:
            kmf (lifelines.KaplanMeierFitter): Fitted Kaplan-Meier model
            output_dir (str, optional): Directory to save the plot. If None, displays the plot.
        
        Returns:
            bool: True if visualization was successful, False otherwise
        """
        try:
            if kmf is None:
                self.logger.error("No fitted Kaplan-Meier model provided")
                return False
            
            plt.figure(figsize=(10, 6))
            kmf.plot_survival_function()
            plt.title('Survival Curve with 95% Confidence Intervals')
            plt.xlabel('Time (days)')
            plt.ylabel('Survival Probability')
            plt.grid(True, alpha=0.3)
            
            # Add median survival time annotation
            median_time = kmf.median_survival_time_
            plt.axvline(x=median_time, color='r', linestyle='--', alpha=0.5)
            plt.text(median_time, 0.5, f'Median: {median_time:.0f} days',
                    rotation=90, va='center')
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'survival_curves.png'))
                self.logger.info(f"Saved survival curves plot to {output_dir}")
            else:
                plt.show()
            
            plt.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error visualizing survival curves: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def visualize_feature_importance(self, output_dir=None):
        """
        Visualize feature importance from the Cox PH model.
        
        Args:
            output_dir (str, optional): Directory to save the plot. If None, displays the plot.
        
        Returns:
            bool: True if visualization was successful, False otherwise
        """
        try:
            if self.cph is None:
                self.logger.error("No fitted Cox PH model provided")
                return False
            
            # Get feature coefficients and standard errors
            coef = self.cph.params_
            se = self.cph.standard_errors_
            
            # Calculate z-scores and p-values
            z_scores = coef / se
            p_values = 2 * (1 - stats.norm.cdf(abs(z_scores)))
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': coef.index,
                'Coefficient': coef.values,
                'Standard Error': se,
                'Z-score': z_scores,
                'P-value': p_values
            })
            
            # Sort by absolute coefficient value
            importance_df['Abs_Coefficient'] = abs(importance_df['Coefficient'])
            importance_df = importance_df.sort_values('Abs_Coefficient', ascending=True)
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            bars = plt.barh(importance_df['Feature'], importance_df['Coefficient'])
            
            # Add error bars
            plt.errorbar(importance_df['Coefficient'], importance_df['Feature'],
                        xerr=importance_df['Standard Error'], fmt='none', color='black')
            
            # Add significance stars
            for i, p_val in enumerate(importance_df['P-value']):
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**'
                elif p_val < 0.05:
                    stars = '*'
                else:
                    stars = ''
                plt.text(importance_df['Coefficient'].iloc[i], i, stars,
                        va='center', ha='left')
            
            plt.title('Feature Importance in Cox PH Model')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Feature')
            plt.grid(True, alpha=0.3)
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
                self.logger.info(f"Saved feature importance plot to {output_dir}")
            else:
                plt.show()
            
            plt.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error visualizing feature importance: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def calculate_funding_risk_scores(self, df, time_horizons=None):
        """
        Calculate funding risk scores for companies at specified time horizons.
        
        Args:
            df (pd.DataFrame): DataFrame containing company features
            time_horizons (list, optional): List of time horizons (in days) at which to calculate risk.
                If None, uses [180, 360, 540] (6 months, 12 months, 18 months)
        
        Returns:
            pd.DataFrame: DataFrame with risk scores for each company at each time horizon
        """
        try:
            if self.cph is None:
                self.logger.error("Cox PH model not fitted. Please fit the model first.")
                return None
                
            if time_horizons is None:
                time_horizons = [180, 360, 540]  # 6 months, 12 months, 18 months
            
            # Get the feature names used by the model
            model_features = self.cph.params_.index.tolist()
            self.logger.info(f"Model uses features: {model_features}")
            
            # Check which features we have in the data
            missing_features = [f for f in model_features if f not in df.columns]
            if missing_features:
                self.logger.warning(f"Missing features in prediction data: {missing_features}")
                
                # Try to compute missing features if possible
                if 'funding_velocity' in missing_features and 'funding_amount' in df.columns:
                    self.logger.info("Computing funding_velocity from available data")
                    df['funding_velocity'] = df.apply(
                        lambda row: row.get('total_funding', row.get('funding_amount', 0)) / 
                                  (row.get('days_since_first', 1) or 1),
                        axis=1
                    )
                
                # Check again which features are still missing
                missing_features = [f for f in model_features if f not in df.columns]
                if missing_features:
                    self.logger.error(f"Still missing required features: {missing_features}")
                    raise ValueError("Failed to calculate survival probabilities")
            
            # Prepare data with only the model features
            prediction_data = df[model_features].copy()
            
            # Handle NaN values
            if prediction_data.isnull().sum().sum() > 0:
                self.logger.warning(f"NaN values found in prediction data. Filling with zeros.")
                prediction_data = prediction_data.fillna(0)
            
            # Handle infinity values
            inf_counts = np.isinf(prediction_data).sum().sum()
            if inf_counts > 0:
                self.logger.warning(f"Found {inf_counts} infinity values in prediction data. Replacing with large finite values.")
                prediction_data = prediction_data.replace([np.inf, -np.inf], [1e10, -1e10])
            
            # Check for extreme values
            for col in model_features:
                if prediction_data[col].abs().max() > 1e10:
                    self.logger.warning(f"Column {col} has extreme values in prediction data. Capping at 1e10.")
                    prediction_data[col] = prediction_data[col].clip(-1e10, 1e10)
            
            # Get survival probabilities
            self.logger.info(f"Calculating survival probabilities for {len(prediction_data)} companies at {len(time_horizons)} time horizons")
            surv_probs = self.cph.predict_survival_function(prediction_data, times=time_horizons)
            if surv_probs is None or surv_probs.empty:
                raise ValueError("Failed to calculate survival probabilities")
            
            # Calculate risk scores (1 - survival probability)
            risk_scores = 1 - surv_probs
            
            # Convert to more usable format
            # Create a DataFrame with companies as rows and time horizons as columns
            risk_df = pd.DataFrame(index=df.index)
            
            # Add risk scores for each time horizon
            for t in time_horizons:
                if t in risk_scores.index:
                    risk_df[f'risk_score_{t}d'] = risk_scores.loc[t].values
            
            # Add company identifiers if available
            if 'company_name' in df.columns:
                risk_df['company_name'] = df['company_name'].values
            
            self.logger.info(f"Calculated risk scores for {len(risk_df)} companies at {len(time_horizons)} time horizons")
            return risk_df
            
        except Exception as e:
            self.logger.error(f"Error calculating funding risk scores: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def validate_assumptions(self, data):
        """
        Validate Cox PH assumptions (proportional hazards)
        """
        if not self.is_fitted:
            self.logger.error("Model not fitted! Please fit the model before validating assumptions.")
            return None
            
        self.logger.info("Validating Cox PH assumptions")
        try:
            assumptions = self.cph.check_assumptions(data, show_plots=False)
            p_values = assumptions.summary['p'].values
            features = assumptions.summary.index.values
            
            self.logger.info("Proportional hazards test results:")
            for feature, p in zip(features, p_values):
                status = "PASSED" if p >= 0.05 else "FAILED"
                self.logger.info(f"  {feature}: p={p:.4f} ({status})")
                
            # Identify features that violate the proportional hazards assumption
            failed_features = assumptions.summary[assumptions.summary['p'] < 0.05].index
            if len(failed_features) > 0:
                self.logger.warning(f"Proportional hazards assumption violated for: {', '.join(failed_features)}")
                
                # Generate Schoenfeld residual plots for failed features
                fig_path = os.path.join(self.output_dir, "schoenfeld_residuals.png")
                assumptions.plot()
                plt.tight_layout()
                plt.savefig(fig_path)
                self.logger.info(f"Saved Schoenfeld residual plots to {fig_path}")
                plt.close()
                
            return {
                'p_values': dict(zip(features, p_values)),
                'passed': all(p >= 0.05 for p in p_values),
                'failed_features': list(failed_features)
            }
        except Exception as e:
            self.logger.error(f"Error validating assumptions: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def analyze_funding_intervals(self, data, id_col='company_name', time_col='funding_date'):
        """
        Analyze intervals between funding rounds
        """
        self.logger.info("Analyzing funding intervals")
        df = data.copy()
        
        # Convert date column to datetime if needed
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Sort by company and funding date
        df = df.sort_values([id_col, time_col])
        
        # Calculate intervals between funding rounds
        df['prev_funding_date'] = df.groupby(id_col)[time_col].shift(1)
        df['days_since_prev'] = (df[time_col] - df['prev_funding_date']).dt.days
        
        # Filter out null intervals (first rounds) and negative intervals
        intervals = df[df['days_since_prev'] > 0].copy()
        
        # Calculate statistics by funding stage if available
        if 'funding_stage' in intervals.columns:
            intervals['prev_stage'] = intervals.groupby(id_col)['funding_stage'].shift(1)
            intervals['stage_transition'] = intervals['prev_stage'] + ' → ' + intervals['funding_stage']
            
            # Get statistics by stage
            stage_stats = intervals.groupby('funding_stage')['days_since_prev'].agg(
                ['count', 'mean', 'std', 'min', 'median', 'max']
            )
            self.logger.info(f"Funding interval statistics by stage:\n{stage_stats}")
            
        # Calculate overall statistics
        overall_stats = intervals['days_since_prev'].agg(
            ['count', 'mean', 'std', 'min', 'median', 'max']
        )
        self.logger.info(f"Overall funding interval statistics:\n{overall_stats}")
        
        # Visualize intervals
        plt.figure(figsize=(12, 10))
        
        # Distribution of intervals
        plt.subplot(2, 1, 1)
        sns.histplot(intervals['days_since_prev'], bins=30, kde=True)
        plt.axvline(x=intervals['days_since_prev'].median(), color='red', linestyle='--', 
                    label=f"Median: {intervals['days_since_prev'].median():.0f} days")
        plt.title("Distribution of Intervals Between Funding Rounds", fontsize=14)
        plt.xlabel("Days Between Rounds", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Intervals by funding stage if available
        if 'funding_stage' in intervals.columns:
            plt.subplot(2, 1, 2)
            sns.boxplot(x='funding_stage', y='days_since_prev', data=intervals)
            plt.title("Funding Intervals by Stage", fontsize=14)
            plt.xlabel("Funding Stage", fontsize=12)
            plt.ylabel("Days Between Rounds", fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
        plt.tight_layout()
        
        # Prepare return values
        if 'funding_stage' in intervals.columns:
            stats = {'overall': overall_stats, 'by_stage': stage_stats}
        else:
            stats = {'overall': overall_stats}
            
        return stats, plt.gcf()
    
    def save_model(self, filepath=None):
        """
        Save the fitted Cox model for later use
        """
        if not self.is_fitted:
            self.logger.error("Model not fitted! Please fit the model before saving.")
            return None
            
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"continuation_model_{timestamp}.joblib")
            
        try:
            model_data = {
                'kmf': self.kmf,
                'cph': self.cph,
                'feature_names': self.feature_names,
                'duration_col': self.duration_col,
                'event_col': self.event_col,
                'fit_timestamp': self.fit_timestamp,
                'model_version': '1.0'
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Saved funding continuation model to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def load_model(self, filepath):
        """
        Load a previously saved model
        """
        try:
            model_data = joblib.load(filepath)
            self.kmf = model_data['kmf']
            self.cph = model_data['cph']
            self.feature_names = model_data['feature_names']
            self.duration_col = model_data['duration_col']
            self.event_col = model_data['event_col']
            self.fit_timestamp = model_data['fit_timestamp']
            self.is_fitted = True
            
            logger.info(f"Loaded funding continuation model from {filepath}")
            logger.info(f"Model was fitted on: {self.fit_timestamp}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def visualize_funding_velocity(self, output_dir=None):
        """
        Visualize funding velocity patterns and trends.
        
        Args:
            output_dir (str, optional): Directory to save the visualization
        """
        try:
            if output_dir is None:
                output_dir = self.output_dir
            
            if len(self.data) == 0:
                logging.warning("No data available for funding velocity visualization")
                return
            
            plt.figure(figsize=(10, 6))
            
            # Calculate funding velocity (amount per day since first funding)
            data = self.data.copy()
            data['days_since_first'] = (data['funding_date'] - data.groupby('company_name')['funding_date'].transform('min')).dt.days
            # Handle edge cases in velocity calculation
            data['funding_velocity'] = np.where(
                data['days_since_first'] == 0,
                data['funding_amount'],
                data['funding_amount'] / data['days_since_first']
            )
            
            # Get total funding by company for filtering
            total_funding = data.groupby('company_name')['funding_amount'].sum().sort_values(ascending=False)
            
            # Filter to top companies by total funding (max 15 for readability)
            MAX_COMPANIES = 15
            top_companies = total_funding.head(MAX_COMPANIES).index
            filtered_data = data[data['company_name'].isin(top_companies)]
            
            # Create simpler scatter plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_companies)))
            color_dict = dict(zip(top_companies, colors))
            
            # Plot only the most recent point for each company to reduce clutter
            latest_points = filtered_data.loc[filtered_data.groupby('company_name')['funding_date'].idxmax()]
            
            for i, (company, company_data) in enumerate(filtered_data.groupby('company_name')):
                plt.scatter(company_data['days_since_first'], 
                          company_data['funding_velocity'],
                          color=[color_dict[company]],
                          label=company if i < 10 else None,  # Only include first 10 in legend
                          alpha=0.6,
                          s=30)  # Smaller point size
            
            plt.xlabel('Days Since First Funding')
            plt.ylabel('Funding Velocity ($/day)')
            plt.title('Funding Velocity for Top Companies')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Only show legend for a limited number of companies
            if len(top_companies) <= 10:
                plt.legend(fontsize='small', loc='best')
            
            # Add trend line using the filtered data
            z = np.polyfit(filtered_data['days_since_first'], np.log10(filtered_data['funding_velocity'].clip(1)), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(filtered_data['days_since_first'].min(), 
                                filtered_data['days_since_first'].max(), 100)
            plt.plot(x_trend, 10**p(x_trend), 'r--', alpha=0.5, 
                    label=f'Trend (slope: {z[0]:.2e})')
            
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, 'funding_velocity.png'), 
                       bbox_inches='tight',
                       dpi=300)
            plt.close()
            
            logging.info("Funding velocity visualization saved successfully")
            
        except Exception as e:
            logging.error(f"Error in funding velocity visualization: {str(e)}")
            logging.error(traceback.format_exc())

    def visualize_funding_patterns(self, output_dir=None):
        """
        Visualize funding patterns including amount distributions and stage transitions.
        
        Args:
            output_dir (str, optional): Directory to save the visualization
        """
        try:
            if output_dir is None:
                output_dir = self.output_dir
            
            if len(self.data) == 0:
                logging.warning("No data available for funding patterns visualization")
                return
            
            # Sample data if too large (for performance)
            data = self.data
            if len(data) > 1000:
                logging.warning(f"Large dataset detected ({len(data)} records). Sampling 1000 records.")
                data = data.sample(1000, random_state=42)
            
            # Create a figure with multiple subplots - 2x2 grid
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            
            # 1. Funding amount distribution by stage (top left)
            # Group stages with few samples into "Other" category
            stage_counts = data['funding_stage'].value_counts()
            rare_stages = stage_counts[stage_counts < 10].index
            data_plot = data.copy()
            if len(rare_stages) > 0:
                data_plot.loc[data_plot['funding_stage'].isin(rare_stages), 'funding_stage'] = 'Other'
            
            # Exclude outliers for better visualization
            # Use 99th percentile as upper limit
            upper_limit = data_plot['funding_amount'].quantile(0.99)
            plot_data = data_plot[data_plot['funding_amount'] <= upper_limit]
            
            sns.boxplot(data=plot_data, x='funding_stage', y='funding_amount', ax=axs[0, 0])
            axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45, ha='right')
            axs[0, 0].set_yscale('log')
            axs[0, 0].set_title('Funding Amount Distribution by Stage')
            axs[0, 0].set_xlabel('Funding Stage')
            axs[0, 0].set_ylabel('Funding Amount (USD)')
            
            # 2. Stage transitions heatmap (top right)
            # Create transition matrix
            df_sorted = data.sort_values(['company_name', 'funding_date'])
            df_sorted['next_stage'] = df_sorted.groupby('company_name')['funding_stage'].shift(-1)
            
            # Remove rows where next_stage is NaN (last funding round for each company)
            df_transitions = df_sorted.dropna(subset=['next_stage'])
            
            # Simplify rare categories for better visualization
            if len(rare_stages) > 0:
                df_transitions.loc[df_transitions['funding_stage'].isin(rare_stages), 'funding_stage'] = 'Other'
                df_transitions.loc[df_transitions['next_stage'].isin(rare_stages), 'next_stage'] = 'Other'
            
            # Calculate transitions
            transitions = pd.crosstab(
                df_transitions['funding_stage'],
                df_transitions['next_stage'],
                normalize='index'
            )
            
            # Filter to include only the top 8 most common stages for readability
            top_stages = stage_counts.head(8).index
            if 'Other' in transitions.index:
                top_stages = list(top_stages) + ['Other']
            
            transitions_filtered = transitions.loc[
                transitions.index.isin(top_stages),
                transitions.columns.isin(top_stages)
            ]
            
            sns.heatmap(transitions_filtered, annot=True, fmt='.2f', cmap='YlOrRd', ax=axs[0, 1])
            axs[0, 1].set_title('Funding Stage Transitions (Probability)')
            
            # 3. Time between funding rounds distribution (bottom left)
            # Calculate time differences between consecutive rounds
            time_between_rounds = data.groupby('company_name')['funding_date'].diff().dt.days
            time_between_rounds = time_between_rounds[time_between_rounds > 0]  # Remove negative values
            
            # Exclude extreme outliers (> 95th percentile)
            time_between_rounds = time_between_rounds[time_between_rounds <= time_between_rounds.quantile(0.95)]
            
            sns.histplot(data=time_between_rounds, bins=20, ax=axs[1, 0])
            axs[1, 0].set_title('Distribution of Time Between Funding Rounds')
            axs[1, 0].set_xlabel('Days Between Rounds')
            axs[1, 0].set_ylabel('Count')
            
            # Add mean and median lines
            mean_time = time_between_rounds.mean()
            median_time = time_between_rounds.median()
            axs[1, 0].axvline(mean_time, color='r', linestyle='--', 
                       label=f'Mean: {mean_time:.0f} days')
            axs[1, 0].axvline(median_time, color='g', linestyle='--', 
                       label=f'Median: {median_time:.0f} days')
            axs[1, 0].legend()
            
            # 4. Average funding amount by round sequence (bottom right)
            # Calculate average funding amount by sequence
            # Limit to first 10 rounds for readability
            MAX_ROUNDS = 10
            avg_by_sequence = data.groupby('funding_sequence')['funding_amount'].mean()
            avg_by_sequence = avg_by_sequence[avg_by_sequence.index <= MAX_ROUNDS]
            
            # Convert to millions for better readability
            avg_by_sequence_millions = avg_by_sequence / 1_000_000
            
            # Plot bar chart
            sns.barplot(x=avg_by_sequence.index, y=avg_by_sequence_millions.values, ax=axs[1, 1])
            axs[1, 1].set_title('Average Funding Amount by Round Sequence')
            axs[1, 1].set_xlabel('Round Number')
            axs[1, 1].set_ylabel('Average Amount ($ millions)')
            
            # Add value labels on top of bars
            for i, v in enumerate(avg_by_sequence_millions.values):
                axs[1, 1].text(i, v, f'${v:.1f}M', 
                        ha='center', va='bottom', fontsize=9)
            
            # Apply tight layout
            fig.tight_layout(pad=3.0)
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, 'funding_patterns.png'),
                       bbox_inches='tight',
                       dpi=300)
            plt.close(fig)
            
            logging.info("Funding patterns visualization saved successfully")
            
        except Exception as e:
            logging.error(f"Error in funding patterns visualization: {str(e)}")
            logging.error(traceback.format_exc())

    def generate_funding_continuation_report(self, output_dir=None):
        """
        Generate a comprehensive report of the funding continuation analysis.
        
        Args:
            output_dir (str): Directory to save the report
        """
        try:
            if self.data is None or len(self.data) == 0:
                self.logger.error("No data available for report generation")
                return
            
            if output_dir is None:
                output_dir = self.output_dir
            
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, 'funding_continuation_report.md')
            
            with open(report_path, 'w') as f:
                f.write("# Funding Continuation Analysis Report\n\n")
                
                # Data Overview
                f.write("## Data Overview\n\n")
                f.write(f"Total companies analyzed: {len(self.data['company_name'].unique())}\n")
                f.write(f"Total funding rounds: {len(self.data)}\n")
                f.write(f"Date range: {self.data['funding_date'].min().strftime('%Y-%m-%d')} to {self.data['funding_date'].max().strftime('%Y-%m-%d')}\n\n")
                
                # Funding Stage Distribution
                f.write("## Funding Stage Distribution\n\n")
                stage_dist = self.data['funding_stage'].value_counts()
                f.write("| Stage | Count | Percentage |\n")
                f.write("|-------|-------|------------|\n")
                for stage, count in stage_dist.items():
                    percentage = (count / len(self.data)) * 100
                    f.write(f"| {stage} | {count} | {percentage:.1f}% |\n")
                f.write("\n")
                
                # Funding Amount Statistics
                f.write("## Funding Amount Statistics\n\n")
                # Convert funding amount to millions if not already
                if 'amount_in_millions' not in self.data.columns:
                    self.data['amount_in_millions'] = self.data['funding_amount'] / 1_000_000
                amount_stats = self.data['amount_in_millions'].describe()
                f.write("| Statistic | Value (in millions) |\n")
                f.write("|-----------|--------------------|\n")
                for stat, value in amount_stats.items():
                    f.write(f"| {stat} | {value:.2f} |\n")
                f.write("\n")
                
                # Time Between Rounds
                if self.survival_data is not None:
                    f.write("## Time Between Rounds\n\n")
                    time_stats = self.survival_data['duration'].describe()
                    f.write("| Statistic | Days |\n")
                    f.write("|-----------|------|\n")
                    for stat, value in time_stats.items():
                        f.write(f"| {stat} | {value:.1f} |\n")
                    f.write("\n")
                    
                    # Success Rates
                    f.write("## Success Rates\n\n")
                    success_rate = (self.survival_data['event'].sum() / len(self.survival_data)) * 100
                    f.write(f"Overall success rate: {success_rate:.1f}%\n")
                    f.write(f"Number of successful follow-on rounds: {self.survival_data['event'].sum()}\n")
                    f.write(f"Number of right-censored observations: {len(self.survival_data) - self.survival_data['event'].sum()}\n\n")
                
                # Model Performance
                if self.cph is not None:
                    f.write("## Model Performance\n\n")
                    f.write(f"Concordance Index: {self.cph.concordance_index_:.3f}\n\n")
                
            self.logger.info(f"Report generated successfully at {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            self.logger.error(traceback.format_exc())

    def visualize_risk_distribution(self, output_dir=None):
        """
        Visualize the distribution of risk scores across different time horizons.
        
        Args:
            output_dir (str, optional): Directory to save the visualization
        """
        try:
            if output_dir is None:
                output_dir = self.output_dir
            
            if not hasattr(self, 'risk_scores') or self.risk_scores is None or self.risk_scores.empty:
                logging.warning("No risk scores available for visualization")
                return
            
            # Get risk score columns (those starting with 'risk_score_')
            risk_columns = [col for col in self.risk_scores.columns if col.startswith('risk_score_')]
            if not risk_columns:
                logging.warning("No risk score columns found in risk_scores DataFrame")
                return
                
            # Create a long-format DataFrame for easier plotting
            risk_data = pd.melt(
                self.risk_scores, 
                id_vars=['company_name'] if 'company_name' in self.risk_scores.columns else [],
                value_vars=risk_columns,
                var_name='time_horizon',
                value_name='risk_score'
            )
            
            # Extract time horizon values for labels
            risk_data['time_horizon_days'] = risk_data['time_horizon'].str.extract(r'risk_score_(\d+)d').astype(int)
            risk_data['time_horizon_label'] = risk_data['time_horizon_days'].apply(
                lambda x: f"{x} days ({x/30:.1f} months)"
            )
            
            # Sample data if too large
            if len(risk_data) > 3000:
                logging.warning(f"Large risk dataset detected ({len(risk_data)} records). Sampling for visualization.")
                risk_data = risk_data.sample(3000, random_state=42)
            
            plt.figure(figsize=(10, 6))
            
            # Create violin plot of risk scores by time horizon
            ax = sns.violinplot(data=risk_data, x='time_horizon_label', y='risk_score')
            plt.title('Distribution of Risk Scores by Time Horizon', fontsize=14)
            plt.xlabel('Time Horizon', fontsize=12)
            plt.ylabel('Risk Score (higher = higher risk)', fontsize=12)
            plt.xticks(rotation=45)
            
            # Add mean and median lines for each time horizon
            for i, th in enumerate(sorted(risk_data['time_horizon_label'].unique())):
                th_data = risk_data[risk_data['time_horizon_label'] == th]['risk_score']
                mean_risk = th_data.mean()
                median_risk = th_data.median()
                
                plt.plot([i-0.3, i+0.3], [mean_risk, mean_risk], 'b-', linewidth=2)
                plt.plot([i-0.3, i+0.3], [median_risk, median_risk], 'g-', linewidth=2)
                
                # Add text labels
                plt.text(i, mean_risk + 0.02, f"Mean: {mean_risk:.2f}", 
                        ha='center', va='bottom', color='blue', fontsize=9)
                plt.text(i, median_risk - 0.04, f"Median: {median_risk:.2f}", 
                        ha='center', va='top', color='green', fontsize=9)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, 'risk_distribution.png'),
                       bbox_inches='tight',
                       dpi=300)
            plt.close()
            
            logging.info("Risk distribution visualization saved successfully")
            
            # Also create a summary table by time horizon using describe() instead of agg()
            try:
                summary_stats = risk_data.groupby('time_horizon_label')['risk_score'].describe().round(3)
                summary_path = os.path.join(output_dir, 'risk_score_summary.csv')
                summary_stats.to_csv(summary_path)
                logging.info(f"Risk score summary saved to {summary_path}")
            except Exception as e:
                logging.warning(f"Error creating risk score summary: {str(e)}")
            
        except Exception as e:
            logging.error(f"Error in calibration plot visualization: {str(e)}")
            logging.error(traceback.format_exc())

    def visualize_calibration(self, output_dir=None):
        """
        Create calibration plots for model predictions.
        
        Args:
            output_dir (str, optional): Directory to save the visualization
        """
        try:
            if output_dir is None:
                output_dir = self.output_dir
            
            if not hasattr(self, 'predictions') or self.predictions is None or self.predictions.empty:
                logging.warning("No predictions available for calibration plot")
                return
            
            plt.figure(figsize=(8, 6))
            
            # Calculate calibration curve with fewer bins for more stability
            prob_true, prob_pred = calibration_curve(
                self.predictions['actual'],
                self.predictions['predicted'],
                n_bins=5  # Fewer bins for stability
            )
            
            # Plot calibration curve
            plt.plot(prob_pred, prob_true, 's-', label='Model')
            plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
            
            # Add confidence intervals
            conf_intervals = np.sqrt(prob_true * (1 - prob_true) / len(self.predictions))
            plt.fill_between(prob_pred, 
                           prob_true - conf_intervals,
                           prob_true + conf_intervals,
                           alpha=0.2)
            
            plt.xlabel('Predicted Probability')
            plt.ylabel('Actual Probability')
            plt.title('Calibration Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add reliability score
            reliability_score = np.mean(np.abs(prob_true - prob_pred))
            plt.text(0.05, 0.95, f'Reliability Score: {reliability_score:.3f}',
                   transform=plt.gca().transAxes,
                   bbox=dict(facecolor='white', alpha=0.8))
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, 'calibration_plot.png'),
                       bbox_inches='tight',
                       dpi=300)
            plt.close()
            
            logging.info("Calibration plot saved successfully")
            
        except Exception as e:
            logging.error(f"Error in calibration plot visualization: {str(e)}")
            logging.error(traceback.format_exc())

    def run_analysis(self):
        """
        Run the complete funding continuation analysis pipeline.
        This method sequences all the analysis steps and visualizations.
        """
        try:
            self.logger.info("Starting funding continuation analysis pipeline")
            
            # Check if data is loaded
            if self.data is None or len(self.data) == 0:
                self.logger.error("No data available for analysis")
                return False
            
            # 1. Prepare survival data
            self.logger.info("Preparing survival data...")
            self.survival_data = self.prepare_survival_data(self.data)
            if self.survival_data is None:
                self.logger.error("Failed to prepare survival data")
                return False
            
            # 2. Fit Kaplan-Meier estimator
            self.logger.info("Fitting Kaplan-Meier estimator...")
            self.kmf = self.fit_kaplan_meier(self.survival_data)
            if self.kmf is None:
                self.logger.error("Failed to fit Kaplan-Meier estimator")
                return False
            
            # 3. Fit Cox Proportional Hazards model
            self.logger.info("Fitting Cox PH model...")
            try:
                self.fit_cox_ph(self.survival_data)
                if self.cph is None:
                    self.logger.error("Failed to fit Cox PH model")
                    return False
                self.is_fitted = True
            except Exception as e:
                self.logger.error(f"Error fitting Cox PH model: {e}")
                return False
            
            # 4. Calculate risk scores and predictions
            self.logger.info("Calculating risk scores...")
            try:
                self.risk_scores = self.calculate_funding_risk_scores(self.survival_data)
                
                # Calculate predictions for calibration plot
                if self.cph is not None:
                    try:
                        model_features = self.cph.params_.index.tolist()
                        predictions = self.cph.predict_survival_function(self.survival_data[model_features])
                        self.predictions = pd.DataFrame({
                            'actual': self.survival_data['event'],
                            'predicted': 1 - predictions.T.iloc[:, 0]
                        })
                    except Exception as e:
                        self.logger.warning(f"Could not calculate predictions: {str(e)}")
                        self.predictions = None
            except Exception as e:
                self.logger.warning(f"Error calculating risk scores: {str(e)}")
                self.risk_scores = None
                self.predictions = None
            
            # 5. Generate visualizations
            self.logger.info("Generating visualizations...")
            
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Generate visualizations based on what we have
            if self.kmf is not None:
                self.visualize_survival_curves(self.kmf, self.output_dir)
            
            if self.cph is not None:
                self.visualize_feature_importance(self.output_dir)
            
            # These visualizations don't depend on model fitting
            self.visualize_funding_patterns(self.output_dir)
            self.visualize_funding_velocity(self.output_dir)
            
            # These visualizations require risk scores and predictions
            if self.risk_scores is not None:
                self.visualize_risk_distribution(self.output_dir)
            
            if self.predictions is not None:
                self.visualize_calibration(self.output_dir)
            
            # 6. Generate comprehensive report
            self.logger.info("Generating analysis report...")
            self.generate_funding_continuation_report(self.output_dir)
            
            self.logger.info("Funding continuation analysis completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in run_analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

def main():
    """Main function to run the funding continuation analysis pipeline."""
    # Get the base directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), 'JSONFolder')
    output_dir = os.path.join(base_dir, 'output', 'funding_continuation')
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize funding continuation analysis
    analysis = FundingContinuationAnalysis(data_dir=data_dir, output_dir=output_dir)
    
    # Run analysis now and then every 24 hours
    interval_hours = 24
    print(f"\n===== FUNDING CONTINUATION ANALYSIS =====")
    print(f"Starting analysis with automatic 24-hour scheduling")
    print(f"First run starting now, will repeat every 24 hours")
    
    # Track the last run time to schedule next runs properly
    last_run_time = None
    
    # Run in a loop that repeats every 24 hours
    while True:
        try:
            current_time = datetime.now()
            
            # Only run if this is the first time or if interval_hours have passed
            if last_run_time is None or (current_time - last_run_time).total_seconds() >= interval_hours * 3600:
                # Run the analysis
                print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Running funding continuation analysis...")
                
                # Load the data from JSON files first
                print("Loading data from JSON files...")
                analysis.data = analysis.load_data_from_json_files(base_dir=os.path.dirname(base_dir))
                
                if analysis.data is None or len(analysis.data) == 0:
                    print("ERROR: No data was loaded. Check the JSON files in the JSONFolder directory.")
                    print("Will retry in 1 hour.")
                    time.sleep(3600)  # Wait 1 hour before retrying
                    continue
                    
                print(f"Loaded {len(analysis.data)} companies' data successfully.")
                
                # Now run the analysis with the loaded data
                success = analysis.run_analysis()
                
                # Update last run time
                last_run_time = current_time
                
                # Schedule next run
                next_run = current_time + timedelta(hours=interval_hours)
                print(f"Analysis complete.")
                print(f"Next analysis scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')} (in {interval_hours} hours)")
                print("The program will continue running in the background.")
                print("Close this window or press Ctrl+C to stop the automatic scheduling.")
            
            # Sleep for a shorter interval and check if it's time to run again
            # This approach is more responsive to keyboard interrupts
            time.sleep(60)  # Check every minute instead of sleeping for the full interval
            
        except KeyboardInterrupt:
            print("\nAutomatic scheduling stopped by user.")
            break
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            print(traceback.format_exc())
            print("Will retry in 1 hour.")
            time.sleep(3600)  # Wait 1 hour before retrying


if __name__ == "__main__":
    sys.exit(main())
            

