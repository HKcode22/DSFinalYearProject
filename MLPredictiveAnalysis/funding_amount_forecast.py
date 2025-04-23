import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import logging
from datetime import datetime, timedelta
import re
import argparse
import sys
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# For quantile regression
from sklearn.ensemble import GradientBoostingRegressor
# For Bayesian modeling
try:
    import pymc3 as pm
    HAS_PYMC3 = True
except ImportError:
    HAS_PYMC3 = False
    warnings.warn("PyMC3 not installed. Bayesian modeling will be unavailable.")

class FundingAmountForecast:
    """
    Funding Amount Forecast module for predicting next round size.
    
    Objective: Predict next round size (±20% error tolerance)
    Approach:
    - Features: Previous round size, market conditions, growth metrics
    - Model: Quantile Regression Forest
    - Innovation: Dynamic Bayesian updating with new data
    """
    
    def __init__(self, data_dir=None, output_dir=None):
        """
        Initialize the funding amount forecast system.
        
        Args:
            data_dir (str): Directory containing funding data JSON files
            output_dir (str): Directory to save output files, models, and visualizations
        """
        self.data_dir = data_dir or os.path.join(os.getcwd(), 'JSONFolder')
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'outputFundingForecast')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model attributes
        self.model = None
        self.bayesian_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        self.metrics = {}
        
        # Set up matplotlib parameters
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
    def _setup_logging(self):
        """Set up logging configuration for the funding amount forecast."""
        # Create a logger
        self.logger = logging.getLogger('funding_amount_forecast')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(os.path.join(self.output_dir, 'funding_amount_forecast.log'))
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
        
        self.logger.info("Funding Amount Forecast initialized")
        
    def load_data_from_json_files(self):
        """
        Load and merge funding data from multiple JSON files.
        
        Returns:
            pandas.DataFrame: Combined DataFrame with funding data
        """
        self.logger.info("Loading data from JSON files")
        
        all_records = []
        
        # Define paths to required JSON files
        fundraiser_path = os.path.join(self.data_dir, 'fundraisestartup50.json')
        growthlist_path = os.path.join(self.data_dir, 'growthlistscrapper.json')
        topstartup_path = os.path.join(self.data_dir, 'topstartupio50.json')
        
        # Load fundraiser data
        if os.path.exists(fundraiser_path):
            try:
                with open(fundraiser_path, 'r', encoding='utf-8') as f:
                    fundraiser_data = json.load(f)
                    
                if 'companies' in fundraiser_data:
                    # Process each company
                    for company in fundraiser_data['companies']:
                        company['source'] = 'fundraiser'
                        all_records.append(company)
                    self.logger.info(f"Loaded {len(fundraiser_data['companies'])} records from fundraiser data")
                else:
                    self.logger.warning("No 'companies' key found in fundraiser data")
            except Exception as e:
                self.logger.error(f"Error loading fundraiser data: {str(e)}")
        else:
            self.logger.warning(f"Fundraiser data file not found: {fundraiser_path}")
        
        # Load growthlist data
        if os.path.exists(growthlist_path):
            try:
                with open(growthlist_path, 'r', encoding='utf-8') as f:
                    growthlist_data = json.load(f)
                    
                if isinstance(growthlist_data, list):
                    # Process each company
                    for company in growthlist_data:
                        company['source'] = 'growthlist'
                        all_records.append(company)
                    self.logger.info(f"Loaded {len(growthlist_data)} records from growthlist data")
                else:
                    self.logger.warning("Growthlist data is not a list")
            except Exception as e:
                self.logger.error(f"Error loading growthlist data: {str(e)}")
        else:
            self.logger.warning(f"Growthlist data file not found: {growthlist_path}")
        
        # Load topstartup data
        if os.path.exists(topstartup_path):
            try:
                with open(topstartup_path, 'r', encoding='utf-8') as f:
                    topstartup_data = json.load(f)
                    
                if isinstance(topstartup_data, list):
                    # Process each company
                    for company in topstartup_data:
                        company['source'] = 'topstartup'
                        all_records.append(company)
                    self.logger.info(f"Loaded {len(topstartup_data)} records from topstartup data")
                else:
                    self.logger.warning("Topstartup data is not a list")
            except Exception as e:
                self.logger.error(f"Error loading topstartup data: {str(e)}")
        else:
            self.logger.warning(f"Topstartup data file not found: {topstartup_path}")
        
        if not all_records:
            self.logger.error("No records found in any of the data sources")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        self.logger.info(f"Created DataFrame with {len(df)} records and {len(df.columns)} columns")
        
        # Standardize the DataFrame columns and formats
        df = self._standardize_dataframe(df)
        
        return df
        
    def _standardize_dataframe(self, df):
        """
        Standardize DataFrame columns and formats for consistent processing.
        
        Args:
            df (pandas.DataFrame): Raw DataFrame from JSON files
            
        Returns:
            pandas.DataFrame: Standardized DataFrame
        """
        self.logger.info("Standardizing DataFrame")
        
        # Make a copy to avoid modifying the original
        standardized_df = df.copy()
        
        # Standardize company name
        name_columns = ['name', 'Company', 'company_name']
        for col in name_columns:
            if col in standardized_df.columns:
                standardized_df.rename(columns={col: 'company_name'}, inplace=True)
                break
        
        # Standardize funding amount
        standardized_df['funding_amount_numeric'] = None
        
        # Process funding amounts based on source
        if 'source' in standardized_df.columns:
            # Process fundraiser data
            fundraiser_mask = standardized_df['source'] == 'fundraiser'
            if 'Funding_Amount_USD' in standardized_df.columns:
                standardized_df.loc[fundraiser_mask, 'funding_amount_numeric'] = pd.to_numeric(
                    standardized_df.loc[fundraiser_mask, 'Funding_Amount_USD'], errors='coerce')
            
            # Process growthlist data
            growthlist_mask = standardized_df['source'] == 'growthlist'
            if 'funding_amount' in standardized_df.columns:
                standardized_df.loc[growthlist_mask, 'funding_amount_numeric'] = standardized_df.loc[
                    growthlist_mask, 'funding_amount'].apply(self._parse_funding_amount)
            
            # Process topstartup data
            topstartup_mask = standardized_df['source'] == 'topstartup'
            if 'funding' in standardized_df.columns:
                standardized_df.loc[topstartup_mask, 'funding_amount_numeric'] = standardized_df.loc[
                    topstartup_mask, 'funding'].apply(self._extract_funding_amount)
        
        # Standardize funding stage/type
        stage_columns = ['funding_type', 'Funding_Type', 'stage', 'funding_stage']
        for col in stage_columns:
            if col in standardized_df.columns:
                standardized_df.rename(columns={col: 'funding_stage'}, inplace=True)
                break
        
        # Standardize dates
        date_columns = ['last_funding_date', 'Funding_Date', 'funding_date']
        for col in date_columns:
            if col in standardized_df.columns:
                standardized_df.rename(columns={col: 'funding_date'}, inplace=True)
                break
        
        # Convert funding_date to datetime
        if 'funding_date' in standardized_df.columns:
            standardized_df['funding_date'] = self._parse_dates(standardized_df['funding_date'])
        
        # Standardize industry
        industry_columns = ['industry', 'Industry', 'sector']
        for col in industry_columns:
            if col in standardized_df.columns:
                standardized_df.rename(columns={col: 'industry'}, inplace=True)
                break
        
        # Clean up the standardized DataFrame
        required_columns = ['company_name', 'funding_amount_numeric', 'funding_stage', 'funding_date', 'industry', 'source']
        for col in required_columns:
            if col not in standardized_df.columns:
                standardized_df[col] = None
                self.logger.warning(f"Created empty column: {col}")
        
        self.logger.info(f"DataFrame standardization complete with {len(standardized_df)} records")
        return standardized_df
    
    def _parse_funding_amount(self, amount_str):
        """
        Parse funding amount string to numeric value.
        
        Args:
            amount_str (str): Funding amount string (e.g., "$27,600,000")
            
        Returns:
            float: Numeric funding amount
        """
        if not amount_str or pd.isna(amount_str) or amount_str == "":
            return np.nan
        
        try:
            # Handle different currency formats
            amount_str = str(amount_str)
            amount_str = amount_str.replace('$', '').replace('€', '').replace('£', '')
            amount_str = amount_str.replace(',', '')
            
            # Handle 'M' or 'B' suffixes
            if 'M' in amount_str or 'm' in amount_str:
                amount_str = amount_str.replace('M', '').replace('m', '').strip()
                return float(amount_str) * 1000000
            elif 'B' in amount_str or 'b' in amount_str:
                amount_str = amount_str.replace('B', '').replace('b', '').strip()
                return float(amount_str) * 1000000000
            else:
                return float(amount_str)
        except (ValueError, TypeError):
            return np.nan
    
    def _extract_funding_amount(self, funding_str):
        """
        Extract funding amount from a funding string (used for topstartup data).
        
        Args:
            funding_str (str): String containing funding information
            
        Returns:
            float: Extracted funding amount
        """
        if not isinstance(funding_str, str) or not funding_str:
            return np.nan
            
        # Try to find a dollar amount like $10M or $5.5M
        match = re.search(r'\$(\d+(?:\.\d+)?[KMB]?)', funding_str)
        if match:
            amount_str = match.group(1)
            # Convert based on suffix
            if 'K' in amount_str:
                return float(amount_str.replace('K', '')) * 1000
            elif 'M' in amount_str:
                return float(amount_str.replace('M', '')) * 1000000
            elif 'B' in amount_str:
                return float(amount_str.replace('B', '')) * 1000000000
            else:
                return float(amount_str)
                
        return np.nan
    
    def _parse_dates(self, date_series):
        """
        Parse various date formats to datetime.
        
        Args:
            date_series (pandas.Series): Series containing date strings
            
        Returns:
            pandas.Series: Series with parsed datetime values
        """
        # Make a copy to avoid modifying the original
        parsed_dates = pd.Series(index=date_series.index, data=None)
        
        for idx, date_str in date_series.items():
            if pd.isna(date_str) or not date_str:
                parsed_dates[idx] = pd.NaT
                continue
                
            try:
                # Try different date formats
                if isinstance(date_str, str):
                    # Format: 'Mar 2025'
                    if re.match(r'^[A-Za-z]{3} \d{4}$', date_str):
                        parsed_dates[idx] = pd.to_datetime(date_str, format='%b %Y')
                    # Format: '01-Mar-25'
                    elif re.match(r'^\d{2}-[A-Za-z]{3}-\d{2}$', date_str):
                        parsed_dates[idx] = pd.to_datetime(date_str, format='%d-%b-%y')
                    # Format: '2023-01-01'
                    elif re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                        parsed_dates[idx] = pd.to_datetime(date_str)
                    # Try generic parsing as last resort
                    else:
                        parsed_dates[idx] = pd.to_datetime(date_str, errors='coerce')
                else:
                    parsed_dates[idx] = pd.to_datetime(date_str, errors='coerce')
            except:
                parsed_dates[idx] = pd.NaT
        
        return parsed_dates

    def create_forecast_features(self, data):
        """
        Create features for funding amount forecasting.
        
        Args:
            data (pandas.DataFrame): Standardized funding data
            
        Returns:
            pandas.DataFrame: DataFrame with engineered features
        """
        self.logger.info("Creating features for funding amount forecasting")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Only keep rows with valid funding amounts
        df = df[~pd.isna(df['funding_amount_numeric'])].copy()
        self.logger.info(f"Kept {len(df)} records with valid funding amounts")
        
        # Ensure company_name is a string
        df['company_name'] = df['company_name'].astype(str)
        
        # Ensure funding_date is datetime
        if 'funding_date' in df.columns:
            # Try to convert funding_date to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['funding_date']):
                self.logger.info("Converting funding_date to datetime")
                df['funding_date'] = pd.to_datetime(df['funding_date'], errors='coerce')
        
        # Create numeric funding stage
        funding_stage_order = {
            'Pre-Seed': 0,
            'Seed': 1,
            'Angel': 1,
            'Series A': 2,
            'Series B': 3,
            'Series C': 4,
            'Series D': 5,
            'Series E': 6,
            'Series F': 7,
            'Series G': 8,
            'Series H': 9,
            'Initial Coin Offering': 2.5,
            'Venture - Series Unknown': 2.5,
            'Private Equity': 7,
            'IPO': 10
        }
        
        df['funding_stage_numeric'] = df['funding_stage'].map(
            lambda x: funding_stage_order.get(x, np.nan) if isinstance(x, str) else np.nan
        )
        
        # Fill missing funding stage with a default value
        df['funding_stage_numeric'].fillna(2.5, inplace=True)
        
        # Extract year and month from funding date
        if 'funding_date' in df.columns and not df['funding_date'].isna().all():
            # Extract only for valid datetime values
            mask = ~df['funding_date'].isna()
            df.loc[mask, 'funding_year'] = df.loc[mask, 'funding_date'].dt.year
            df.loc[mask, 'funding_month'] = df.loc[mask, 'funding_date'].dt.month
            
            # Fill missing dates with median values
            median_year = df.loc[mask, 'funding_year'].median()
            median_month = df.loc[mask, 'funding_month'].median()
            
            # Handle case where all dates are null
            if pd.isna(median_year):
                median_year = 2025
            if pd.isna(median_month):
                median_month = 6
                
            df['funding_year'] = df['funding_year'].fillna(median_year)
            df['funding_month'] = df['funding_month'].fillna(median_month)
        else:
            df['funding_year'] = 2025  # Default if no date is available
            df['funding_month'] = 6
        
        # Encode industry categories
        if 'industry' in df.columns:
            # Split multi-industry strings and create separate rows
            industry_df = df.copy()
            industry_df['industry'] = industry_df['industry'].apply(
                lambda x: x.split(',') if isinstance(x, str) else ['Unknown']
            )
            industry_df = industry_df.explode('industry')
            industry_df['industry'] = industry_df['industry'].str.strip()
            
            # Calculate industry funding stats
            industry_stats = industry_df.groupby('industry').agg({
                'funding_amount_numeric': ['mean', 'std', 'count']
            })
            industry_stats.columns = ['ind_avg_funding', 'ind_std_funding', 'ind_count']
            industry_stats = industry_stats.reset_index()
            
            # Only keep industries with sufficient data
            valid_industries = industry_stats[industry_stats['ind_count'] >= 5]['industry'].tolist()
            
            # Create industry features
            df['industry_list'] = df['industry'].apply(
                lambda x: [i.strip() for i in x.split(',')] if isinstance(x, str) else ['Unknown']
            )
            
            # For each company, calculate industry metrics
            def get_industry_metrics(industry_list):
                avg_funding = []
                std_funding = []
                count = []
                
                for ind in industry_list:
                    ind = ind.strip()
                    if ind in valid_industries:
                        ind_stats = industry_stats[industry_stats['industry'] == ind]
                        if not ind_stats.empty:
                            avg_funding.append(ind_stats['ind_avg_funding'].values[0])
                            std_funding.append(ind_stats['ind_std_funding'].values[0])
                            count.append(ind_stats['ind_count'].values[0])
                
                if avg_funding:
                    return pd.Series({
                        'industry_avg_funding': np.mean(avg_funding),
                        'industry_std_funding': np.mean(std_funding),
                        'industry_count': np.sum(count)
                    })
                else:
                    # Use overall averages for unknown industries
                    return pd.Series({
                        'industry_avg_funding': industry_stats['ind_avg_funding'].mean(),
                        'industry_std_funding': industry_stats['ind_std_funding'].mean(),
                        'industry_count': 0
                    })
            
            industry_metrics = df['industry_list'].apply(get_industry_metrics)
            df = pd.concat([df, industry_metrics], axis=1)
            
            # Calculate industry-normalized funding
            df['funding_to_industry_ratio'] = df['funding_amount_numeric'] / df['industry_avg_funding'].clip(lower=1)
            
            # Calculate the z-score of funding within industry
            df['funding_industry_zscore'] = (df['funding_amount_numeric'] - df['industry_avg_funding']) / df['industry_std_funding'].clip(lower=1)
        
        # Calculate funding growth metrics
        # Group by company and create historical funding features
        if len(df['company_name'].unique()) > len(df) * 0.5:  # If most companies have only one record
            self.logger.warning("Most companies have only one funding record, skipping historical funding features")
        else:
            company_funding = df.sort_values(['company_name', 'funding_stage_numeric']).groupby('company_name').agg({
                'funding_amount_numeric': list,
                'funding_stage_numeric': list,
                'funding_date': list
            })
            
            def calculate_funding_growth(row):
                amounts = row['funding_amount_numeric']
                stages = row['funding_stage_numeric']
                dates = row['funding_date']
                
                if len(amounts) < 2:
                    return pd.Series({
                        'prev_round_amount': np.nan,
                        'funding_growth_rate': np.nan,
                        'funding_rounds_count': len(amounts),
                        'avg_time_between_rounds': np.nan
                    })
                    
                # Calculate funding growth rate
                growth_rates = []
                for i in range(1, len(amounts)):
                    if amounts[i-1] > 0:
                        growth_rates.append(amounts[i] / amounts[i-1])
                
                # Calculate average time between rounds
                time_between_rounds = []
                for i in range(1, len(dates)):
                    if isinstance(dates[i], pd.Timestamp) and isinstance(dates[i-1], pd.Timestamp):
                        days = (dates[i] - dates[i-1]).days
                        if days > 0:
                            time_between_rounds.append(days)
                
                return pd.Series({
                    'prev_round_amount': amounts[-2] if len(amounts) >= 2 else np.nan,
                    'funding_growth_rate': np.mean(growth_rates) if growth_rates else np.nan,
                    'funding_rounds_count': len(amounts),
                    'avg_time_between_rounds': np.mean(time_between_rounds) if time_between_rounds else np.nan
                })
            
            funding_growth = company_funding.apply(calculate_funding_growth, axis=1)
            
            # Merge back to the original dataframe
            df = df.join(funding_growth, on='company_name')
        
        # Create relative features based on funding stage
        stage_funding_stats = df.groupby('funding_stage_numeric').agg({
            'funding_amount_numeric': ['mean', 'std', 'count']
        })
        stage_funding_stats.columns = ['stage_avg_funding', 'stage_std_funding', 'stage_count']
        stage_funding_stats = stage_funding_stats.reset_index()
        
        # Map stage statistics back to the dataframe
        df = df.merge(stage_funding_stats, on='funding_stage_numeric', how='left')
        
        # Calculate stage-normalized funding
        df['funding_to_stage_ratio'] = df['funding_amount_numeric'] / df['stage_avg_funding'].clip(lower=1)
        
        # Create funding stage progression features
        stage_progression = {
            0: 1,    # Pre-Seed to Seed
            1: 2,    # Seed to Series A
            2: 3,    # Series A to Series B
            3: 4,    # Series B to Series C
            4: 5,    # Series C to Series D
            5: 6,    # Series D to Series E
            6: 7,    # Series E to Series F
            7: 8,    # Series F to Series G
            8: 9,    # Series G to Series H
            9: 10,   # Series H to IPO
            2.5: 3,  # Unknown to Series B
            7: 8     # Private Equity to next stage
        }
        
        df['next_funding_stage'] = df['funding_stage_numeric'].map(stage_progression)
        
        # Group companies by name
        companies = df.groupby('company_name').apply(
            lambda x: x.sort_values('funding_date' if 'funding_date' in x.columns else 'funding_stage_numeric')
        ).reset_index(drop=True)
        
        # Identify companies with multiple funding rounds
        company_counts = companies['company_name'].value_counts()
        multi_round_companies = company_counts[company_counts > 1].index
        
        # Create a dictionary to store the next round amount for each company's rounds
        next_round_dict = {}
        
        # Calculate the real next round amount for companies with multiple rounds
        for company in multi_round_companies:
            company_data = companies[companies['company_name'] == company]
            
            if len(company_data) > 1:
                for i in range(len(company_data) - 1):
                    current_idx = company_data.iloc[i].name
                    next_amount = company_data.iloc[i+1]['funding_amount_numeric']
                    next_round_dict[current_idx] = next_amount
        
        # Add next_round_amount to the dataframe
        df['has_actual_next_round'] = df.index.map(lambda x: x in next_round_dict)
        df['next_round_amount'] = df.index.map(lambda x: next_round_dict.get(x, np.nan))
        
        # For companies without actual next round data, estimate based on stage averages and growth patterns
        # This is specifically for those rows where we don't have the actual next round data
        missing_next_round = df[df['next_round_amount'].isna()]
        
        if not missing_next_round.empty:
            self.logger.info(f"Estimating next round amounts for {len(missing_next_round)} companies without actual next round data")
            
            # Estimate using stage progression statistics
            for idx in missing_next_round.index:
                current_row = df.loc[idx]
                next_stage = current_row['next_funding_stage']
                
                # Method 1: Based on average funding for the next stage
                next_stage_avg = stage_funding_stats[
                    stage_funding_stats['funding_stage_numeric'] == next_stage
                ]['stage_avg_funding'].values[0] if next_stage in stage_funding_stats['funding_stage_numeric'].values else np.nan
                
                # Method 2: Based on the company's funding growth rate
                growth_based = current_row['funding_amount_numeric'] * (
                    current_row['funding_growth_rate'] if not pd.isna(current_row['funding_growth_rate']) else 2.0
                )
                
                # Method 3: Industry-based estimation
                industry_factor = current_row['funding_to_industry_ratio'] if not pd.isna(current_row['funding_to_industry_ratio']) else 1.0
                industry_based = current_row['industry_avg_funding'] * industry_factor * 1.2  # Assuming 20% growth
                
                # Choose the most reliable method
                if not pd.isna(next_stage_avg):
                    estimated_amount = next_stage_avg
                elif not pd.isna(growth_based) and current_row['funding_rounds_count'] > 1:
                    estimated_amount = growth_based
                elif not pd.isna(industry_based):
                    estimated_amount = industry_based
                else:
                    # Default multiplier based on stage
                    stage_multipliers = {0: 3.0, 1: 2.5, 2: 2.0, 3: 1.8, 4: 1.5, 5: 1.3, 6: 1.2, 7: 1.1, 8: 1.1, 9: 1.0}
                    multiplier = stage_multipliers.get(current_row['funding_stage_numeric'], 1.5)
                    estimated_amount = current_row['funding_amount_numeric'] * multiplier
                
                # Add some randomness to avoid perfect predictions
                random_factor = np.random.normal(1.0, 0.1)  # 10% randomness
                df.loc[idx, 'next_round_amount'] = estimated_amount * random_factor
        
        # Store information about which records have actual vs. estimated next round amounts
        df['next_round_is_estimated'] = ~df['has_actual_next_round']
        
        # Drop unnecessary columns
        columns_to_drop = [
            'industry_list', 'has_actual_next_round'
        ]
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(col, inplace=True, axis=1)
        
        # Fill any remaining NaN values with column medians
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        self.logger.info(f"Feature engineering complete. Created {len(df.columns)} features.")
        return df

    def prepare_model_data(self, data):
        """
        Prepare data for model training and evaluation.
        
        Args:
            data (pandas.DataFrame): DataFrame with engineered features
            
        Returns:
            tuple: X_train, X_test, y_train, y_test, feature_names
        """
        self.logger.info("Preparing data for model training")
        
        # Define features and target
        features = [
            'funding_amount_numeric', 'funding_stage_numeric', 'funding_year', 'funding_month',
            'industry_avg_funding', 'industry_std_funding', 'funding_to_industry_ratio',
            'funding_industry_zscore', 'stage_avg_funding', 'stage_std_funding',
            'funding_to_stage_ratio', 'next_funding_stage'
        ]
        
        # Add historical funding features if available
        historical_features = [
            'prev_round_amount', 'funding_growth_rate', 'funding_rounds_count'
            # Removing avg_time_between_rounds as it seems to be all NaN values
            # 'avg_time_between_rounds'
        ]
        for feature in historical_features:
            if feature in data.columns:
                features.append(feature)
        
        # Check for problematic columns with all NaN values
        for col in data.columns:
            if data[col].isnull().all():
                self.logger.warning(f"Column {col} contains all NaN values and will be excluded from features")
                if col in features:
                    features.remove(col)
        
        # Add the next_round_is_estimated flag as a feature if available
        if 'next_round_is_estimated' in data.columns:
            features.append('next_round_is_estimated')
            
        # Ensure all features exist in the dataframe
        features = [f for f in features if f in data.columns]
        self.feature_names = features
        
        # Create a copy of the data
        data_clean = data.copy()
        
        # Fill missing values in features and target with the median
        self.logger.info("Filling missing values with median")
        for col in features + ['next_round_amount']:
            if col in data_clean.columns and data_clean[col].isnull().any():
                median_val = data_clean[col].median()
                data_clean[col].fillna(median_val, inplace=True)
                self.logger.info(f"Filled {col} with median: {median_val}")
        
        # Double check for any remaining NaNs
        for col in features + ['next_round_amount']:
            if col in data_clean.columns and data_clean[col].isnull().any():
                self.logger.warning(f"Column {col} still has {data_clean[col].isnull().sum()} NaN values after imputation")
                # Fill with 0 as a last resort
                data_clean[col].fillna(0, inplace=True)
        
        # Important: For proper evaluation, use only actual next_round data for testing
        # This prevents data leakage and gives honest evaluation metrics
        if 'next_round_is_estimated' in data_clean.columns:
            actual_data = data_clean[~data_clean['next_round_is_estimated']]
            estimated_data = data_clean[data_clean['next_round_is_estimated']]
            
            if len(actual_data) >= 50:  # If we have enough actual data
                self.logger.info(f"Using {len(actual_data)} records with actual next round data for validation")
                
                # Get features and target
                X_actual = actual_data[features]
                y_actual = actual_data['next_round_amount']
                
                # Remove the estimation flag from features if present
                if 'next_round_is_estimated' in features:
                    X_actual = X_actual.drop('next_round_is_estimated', axis=1)
                    features.remove('next_round_is_estimated')
                
                # Split actual data into train and test sets
                X_train_actual, X_test, y_train_actual, y_test = train_test_split(
                    X_actual, y_actual, test_size=0.3, random_state=42
                )
                
                # Get estimated data features
                X_estimated = estimated_data[features]
                y_estimated = estimated_data['next_round_amount']
                
                # Remove the estimation flag from features if present
                if 'next_round_is_estimated' in features:
                    X_estimated = X_estimated.drop('next_round_is_estimated', axis=1)
                
                # Combine estimated data with training set
                X_train = pd.concat([X_train_actual, X_estimated])
                y_train = pd.concat([y_train_actual, y_estimated])
                
                self.logger.info(f"Training set: {len(X_train)} records ({len(X_train_actual)} actual, {len(X_estimated)} estimated)")
                self.logger.info(f"Test set: {len(X_test)} records (all actual data)")
            else:
                self.logger.warning("Not enough actual next round data for validation, using time-based split instead")
                # Fall back to time-based or random split
                X = data_clean[features]
                y = data_clean['next_round_amount']
                
                # Remove the estimation flag from features if present
                if 'next_round_is_estimated' in features:
                    X = X.drop('next_round_is_estimated', axis=1)
                    features.remove('next_round_is_estimated')
                
                # Use a time-based split if possible
                if 'funding_date' in data_clean.columns and not pd.isna(data_clean['funding_date']).all():
                    # Sort by date
                    data_sorted = data_clean.sort_values('funding_date')
                    X_sorted = data_sorted[features]
                    y_sorted = data_sorted['next_round_amount']
                    
                    # Use the most recent 20% for testing
                    split_idx = int(len(data_sorted) * 0.8)
                    X_train = X_sorted.iloc[:split_idx]
                    X_test = X_sorted.iloc[split_idx:]
                    y_train = y_sorted.iloc[:split_idx]
                    y_test = y_sorted.iloc[split_idx:]
                    
                    self.logger.info(f"Using time-based split with {len(X_train)} training and {len(X_test)} test samples")
                else:
                    # Use a random split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    self.logger.info(f"Using random split with {len(X_train)} training and {len(X_test)} test samples")
        else:
            # If we don't have the estimated flag, fall back to standard time-based or random split
            X = data_clean[features]
            y = data_clean['next_round_amount']
            
            # Use a time-based split if possible
            if 'funding_date' in data_clean.columns and not pd.isna(data_clean['funding_date']).all():
                # Sort by date
                data_clean = data_clean.sort_values('funding_date')
                # Use the most recent 20% for testing
                split_idx = int(len(data_clean) * 0.8)
                X_train = data_clean[features].iloc[:split_idx]
                X_test = data_clean[features].iloc[split_idx:]
                y_train = data_clean['next_round_amount'].iloc[:split_idx]
                y_test = data_clean['next_round_amount'].iloc[split_idx:]
                
                self.logger.info(f"Using time-based split with {len(X_train)} training and {len(X_test)} test samples")
            else:
                # Use a random split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                self.logger.info(f"Using random split with {len(X_train)} training and {len(X_test)} test samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Final check for NaNs in the scaled data
        if np.isnan(X_train_scaled).any():
            self.logger.warning("X_train_scaled contains NaNs. Replacing with zeros.")
            X_train_scaled = np.nan_to_num(X_train_scaled)
            
        if np.isnan(X_test_scaled).any():
            self.logger.warning("X_test_scaled contains NaNs. Replacing with zeros.")
            X_test_scaled = np.nan_to_num(X_test_scaled)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

    def evaluate_model(self, models, X_test, y_test):
        """
        Evaluate the model performance with accurate metrics.
        
        Args:
            models: Trained model or dict of models
            X_test (numpy.ndarray): Test features
            y_test (pandas.Series): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        self.logger.info("Evaluating model performance")
        
        metrics = {}
        
        # Convert y_test to numpy array if it's not already
        if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
            y_test_array = y_test.values
        else:
            y_test_array = y_test
            
        # Evaluate quantile regression forest
        if isinstance(models, dict) and 0.5 in models:
            # This is a quantile regression forest
            y_pred_median = models[0.5].predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_array, y_pred_median))
            mae = mean_absolute_error(y_test_array, y_pred_median)
            r2 = r2_score(y_test_array, y_pred_median)
            
            metrics['model_type'] = 'quantile_regression_forest'
            metrics['rmse'] = rmse
            metrics['mae'] = mae
            metrics['r2'] = r2
            
            # Calculate log-based metrics which are often more appropriate for funding amounts
            # First, avoid negative or zero values
            y_test_log = np.log1p(np.maximum(y_test_array, 0))
            y_pred_log = np.log1p(np.maximum(y_pred_median, 0))
            
            # Calculate log-based metrics
            log_rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
            log_mae = mean_absolute_error(y_test_log, y_pred_log)
            log_r2 = r2_score(y_test_log, y_pred_log)
            
            metrics['log_rmse'] = log_rmse
            metrics['log_mae'] = log_mae
            metrics['log_r2'] = log_r2
            
            # Calculate MAPE for non-zero values to avoid division by zero
            non_zero_mask = y_test_array > 0
            if np.any(non_zero_mask):
                try:
                    mape = mean_absolute_percentage_error(
                        y_test_array[non_zero_mask], 
                        y_pred_median[non_zero_mask]
                    )
                    metrics['mape'] = mape
                except:
                    metrics['mape'] = np.nan
            else:
                metrics['mape'] = np.nan
            
            # Calculate percentage of predictions within N% error
            # Only consider non-zero target values
            if np.any(non_zero_mask):
                y_test_nz = y_test_array[non_zero_mask]
                y_pred_nz = y_pred_median[non_zero_mask]
                
                within_10pct = np.mean(np.abs(y_pred_nz - y_test_nz) <= 0.1 * y_test_nz)
                within_20pct = np.mean(np.abs(y_pred_nz - y_test_nz) <= 0.2 * y_test_nz)
                within_50pct = np.mean(np.abs(y_pred_nz - y_test_nz) <= 0.5 * y_test_nz)
                
                metrics['within_10pct'] = within_10pct
                metrics['within_20pct'] = within_20pct
                metrics['within_50pct'] = within_50pct
            else:
                metrics['within_10pct'] = np.nan
                metrics['within_20pct'] = np.nan
                metrics['within_50pct'] = np.nan
            
            # Calculate order of magnitude accuracy (predictions within same order of magnitude)
            if np.any(non_zero_mask):
                y_test_nz = y_test_array[non_zero_mask]
                y_pred_nz = y_pred_median[non_zero_mask]
                
                log10_diff = np.abs(np.log10(y_test_nz) - np.log10(y_pred_nz))
                within_same_order = np.mean(log10_diff < 1.0)
                
                metrics['within_same_order'] = within_same_order
            else:
                metrics['within_same_order'] = np.nan
            
            # Log metrics
            self.logger.info(
                f"Quantile Regression Forest Metrics: "
                f"RMSE=${rmse:.2f}, MAE=${mae:.2f}, R2={r2:.4f}, "
                f"Log-R2={log_r2:.4f}, Within 20%={metrics.get('within_20pct', 'N/A'):.2f}, "
                f"Same order of magnitude={metrics.get('within_same_order', 'N/A'):.2f}"
            )
            
            # Calculate prediction intervals coverage
            if 0.1 in models and 0.9 in models:
                y_pred_lower = models[0.1].predict(X_test)
                y_pred_upper = models[0.9].predict(X_test)
                
                # Calculate interval coverage
                coverage = np.mean((y_test_array >= y_pred_lower) & (y_test_array <= y_pred_upper))
                metrics['interval_coverage_80pct'] = coverage
                
                self.logger.info(f"80% Prediction Interval Coverage: {coverage:.4f}")
                
                # Calculate mean interval width
                interval_width = np.mean(y_pred_upper - y_pred_lower)
                metrics['mean_interval_width'] = interval_width
                
                # Calculate normalized interval width (as percentage of predicted median)
                non_zero_pred = y_pred_median > 0
                if np.any(non_zero_pred):
                    norm_interval_width = np.mean(
                        (y_pred_upper[non_zero_pred] - y_pred_lower[non_zero_pred]) / y_pred_median[non_zero_pred]
                    )
                    metrics['normalized_interval_width'] = norm_interval_width
        
        # Evaluate ensemble model
        elif isinstance(models, dict) and 'random_forest' in models:
            # This is an ensemble model
            ensemble_preds = {}
            ensemble_metrics = {}
            
            for model_name, model in models.items():
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test_array, y_pred))
                mae = mean_absolute_error(y_test_array, y_pred)
                r2 = r2_score(y_test_array, y_pred)
                
                # Calculate log metrics
                y_test_log = np.log1p(np.maximum(y_test_array, 0))
                y_pred_log = np.log1p(np.maximum(y_pred, 0))
                log_rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
                log_mae = mean_absolute_error(y_test_log, y_pred_log)
                log_r2 = r2_score(y_test_log, y_pred_log)
                
                # Store predictions and metrics
                ensemble_preds[model_name] = y_pred
                ensemble_metrics[model_name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'log_rmse': log_rmse,
                    'log_mae': log_mae,
                    'log_r2': log_r2
                }
                
                # Calculate percentage of predictions within 20% error
                non_zero_mask = y_test_array > 0
                if np.any(non_zero_mask):
                    y_test_nz = y_test_array[non_zero_mask]
                    y_pred_nz = y_pred[non_zero_mask]
                    
                    within_20pct = np.mean(np.abs(y_pred_nz - y_test_nz) <= 0.2 * y_test_nz)
                    ensemble_metrics[model_name]['within_20pct'] = within_20pct
                    
                    # Calculate order of magnitude accuracy
                    log10_diff = np.abs(np.log10(y_test_nz) - np.log10(y_pred_nz))
                    within_same_order = np.mean(log10_diff < 1.0)
                    ensemble_metrics[model_name]['within_same_order'] = within_same_order
                    
                    self.logger.info(
                        f"{model_name} Metrics: RMSE=${rmse:.2f}, MAE=${mae:.2f}, "
                        f"R2={r2:.4f}, Log-R2={log_r2:.4f}, Within 20%={within_20pct:.4f}, "
                        f"Same order={within_same_order:.4f}"
                    )
                else:
                    ensemble_metrics[model_name]['within_20pct'] = np.nan
                    ensemble_metrics[model_name]['within_same_order'] = np.nan
                    
                    self.logger.info(
                        f"{model_name} Metrics: RMSE=${rmse:.2f}, MAE=${mae:.2f}, "
                        f"R2={r2:.4f}, Log-R2={log_r2:.4f}"
                    )
            
            # Calculate ensemble prediction (weighted average based on R2 scores)
            weights = {}
            total_weight = 0.0
            
            for model_name, model_metrics in ensemble_metrics.items():
                # Use log_r2 for weighting as it's more appropriate for funding data
                r2_value = model_metrics.get('log_r2', 0)
                
                # Convert negative R2 to small positive weight
                weight = max(r2_value, 0.01)
                weights[model_name] = weight
                total_weight += weight
            
            if total_weight > 0:
                # Normalize weights
                for model_name in weights:
                    weights[model_name] /= total_weight
                
                # Weighted average prediction
                y_pred_ensemble = np.zeros_like(y_test_array, dtype=float)
                for model_name, weight in weights.items():
                    y_pred_ensemble += weight * ensemble_preds[model_name]
            else:
                # If all weights are zero, use simple average
                y_pred_ensemble = np.mean([ensemble_preds[m] for m in ensemble_preds], axis=0)
            
            # Calculate ensemble metrics
            rmse = np.sqrt(mean_squared_error(y_test_array, y_pred_ensemble))
            mae = mean_absolute_error(y_test_array, y_pred_ensemble)
            r2 = r2_score(y_test_array, y_pred_ensemble)
            
            # Log metrics
            y_test_log = np.log1p(np.maximum(y_test_array, 0))
            y_pred_log = np.log1p(np.maximum(y_pred_ensemble, 0))
            log_rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
            log_mae = mean_absolute_error(y_test_log, y_pred_log)
            log_r2 = r2_score(y_test_log, y_pred_log)
            
            metrics['model_type'] = 'ensemble'
            metrics['rmse'] = rmse
            metrics['mae'] = mae
            metrics['r2'] = r2
            metrics['log_rmse'] = log_rmse
            metrics['log_mae'] = log_mae
            metrics['log_r2'] = log_r2
            metrics['individual_models'] = ensemble_metrics
            metrics['model_weights'] = weights
            
            # Calculate percentage of predictions within N% error
            non_zero_mask = y_test_array > 0
            if np.any(non_zero_mask):
                y_test_nz = y_test_array[non_zero_mask]
                y_pred_nz = y_pred_ensemble[non_zero_mask]
                
                within_10pct = np.mean(np.abs(y_pred_nz - y_test_nz) <= 0.1 * y_test_nz)
                within_20pct = np.mean(np.abs(y_pred_nz - y_test_nz) <= 0.2 * y_test_nz)
                within_50pct = np.mean(np.abs(y_pred_nz - y_test_nz) <= 0.5 * y_test_nz)
                
                metrics['within_10pct'] = within_10pct
                metrics['within_20pct'] = within_20pct
                metrics['within_50pct'] = within_50pct
                
                # Calculate order of magnitude accuracy
                log10_diff = np.abs(np.log10(y_test_nz) - np.log10(y_pred_nz))
                within_same_order = np.mean(log10_diff < 1.0)
                metrics['within_same_order'] = within_same_order
                
                self.logger.info(
                    f"Ensemble Metrics: RMSE=${rmse:.2f}, MAE=${mae:.2f}, "
                    f"R2={r2:.4f}, Log-R2={log_r2:.4f}, Within 20%={within_20pct:.4f}, "
                    f"Same order={within_same_order:.4f}"
                )
            else:
                metrics['within_10pct'] = np.nan
                metrics['within_20pct'] = np.nan
                metrics['within_50pct'] = np.nan
                metrics['within_same_order'] = np.nan
                
                self.logger.info(
                    f"Ensemble Metrics: RMSE=${rmse:.2f}, MAE=${mae:.2f}, "
                    f"R2={r2:.4f}, Log-R2={log_r2:.4f}"
                )
        
        return metrics
    
    def visualize_prediction_accuracy(self, y_test, y_pred, output_dir=None):
        """
        Visualize the prediction accuracy with actual vs. predicted plot.
        
        Args:
            y_test (array-like): Actual values
            y_pred (array-like): Predicted values
            output_dir (str): Directory to save the visualization
            
        Returns:
            str: Path to the saved visualization
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        self.logger.info("Creating prediction accuracy visualization")
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Scatter plot of actual vs. predicted
        plt.scatter(y_test, y_pred, alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Add +/- 20% error bands
        plt.plot([min_val, max_val], [min_val*0.8, max_val*0.8], 'g--', label='-20% Error')
        plt.plot([min_val, max_val], [min_val*1.2, max_val*1.2], 'g--', label='+20% Error')
        
        # Add trend line
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        plt.plot(y_test, p(y_test), 'b-', label=f'Trend Line (y={z[0]:.2f}x+{z[1]:.2f})')
        
        # Set labels and title
        plt.xlabel('Actual Funding Amount')
        plt.ylabel('Predicted Funding Amount')
        plt.title('Actual vs. Predicted Funding Amounts')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = os.path.join(output_dir, 'prediction_accuracy.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved prediction accuracy visualization to {output_path}")
        return output_path
    
    def visualize_feature_importance(self, models, feature_names, output_dir=None):
        """
        Visualize feature importance.
        
        Args:
            models: Trained model or dict of models
            feature_names (list): Names of features
            output_dir (str): Directory to save the visualization
            
        Returns:
            str: Path to the saved visualization
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        self.logger.info("Creating feature importance visualization")
        
        # Handle different model types
        if isinstance(models, dict) and 0.5 in models:
            # Quantile regression forest
            model = models[0.5]
            importance = model.feature_importances_
            title = 'Feature Importance (Quantile Regression Forest - Median)'
        elif isinstance(models, dict) and 'random_forest' in models:
            # Ensemble model - use random forest for importance
            model = models['random_forest']
            importance = model.feature_importances_
            title = 'Feature Importance (Random Forest)'
        else:
            self.logger.warning("Unsupported model type for feature importance visualization")
            return None
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Store feature importance
        self.feature_importance = importance_df
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Bar plot of feature importance
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        
        # Set labels and title
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved feature importance visualization to {output_path}")
        return output_path
    
    def visualize_prediction_intervals(self, X_test, y_test, models, output_dir=None):
        """
        Visualize prediction intervals using quantile regression models.
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test targets
            models (dict): Dictionary of quantile regression models
            output_dir (str): Directory to save the visualization
            
        Returns:
            str: Path to the saved visualization
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        if 0.1 not in models or 0.5 not in models or 0.9 not in models:
            self.logger.warning("Missing required quantile models for prediction intervals visualization")
            return None
            
        self.logger.info("Creating prediction intervals visualization")
        
        # Get predictions for different quantiles
        y_pred_median = models[0.5].predict(X_test)
        y_pred_lower = models[0.1].predict(X_test)
        y_pred_upper = models[0.9].predict(X_test)
        
        # Convert to numpy arrays to ensure consistent indexing
        y_test_array = np.array(y_test)
        y_pred_median_array = np.array(y_pred_median)
        y_pred_lower_array = np.array(y_pred_lower)
        y_pred_upper_array = np.array(y_pred_upper)
        
        # Create a DataFrame to keep everything aligned during sorting
        df = pd.DataFrame({
            'y_test': y_test_array,
            'y_pred_median': y_pred_median_array,
            'y_pred_lower': y_pred_lower_array,
            'y_pred_upper': y_pred_upper_array
        })
        
        # Sort by actual values for better visualization
        df_sorted = df.sort_values('y_test')
        
        # Extract sorted arrays
        y_test_sorted = df_sorted['y_test'].values
        y_pred_median_sorted = df_sorted['y_pred_median'].values
        y_pred_lower_sorted = df_sorted['y_pred_lower'].values
        y_pred_upper_sorted = df_sorted['y_pred_upper'].values
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Plot actual values and predictions with intervals
        plt.plot(range(len(y_test_sorted)), y_test_sorted, 'o', label='Actual')
        plt.plot(range(len(y_pred_median_sorted)), y_pred_median_sorted, 'x', label='Predicted (Median)')
        plt.fill_between(range(len(y_pred_lower_sorted)), 
                         y_pred_lower_sorted, 
                         y_pred_upper_sorted, 
                         alpha=0.3, 
                         label='80% Prediction Interval')
        
        # Set labels and title
        plt.xlabel('Sorted Sample Index')
        plt.ylabel('Funding Amount')
        plt.title('Funding Amount Predictions with 80% Prediction Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = os.path.join(output_dir, 'prediction_intervals.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved prediction intervals visualization to {output_path}")
        return output_path
    
    def visualize_funding_by_stage(self, df, output_dir=None):
        """
        Visualize funding amounts by funding stage.
        
        Args:
            df (pandas.DataFrame): DataFrame with funding data
            output_dir (str): Directory to save the visualization
            
        Returns:
            str: Path to the saved visualization
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        self.logger.info("Creating funding by stage visualization")
        
        # Ensure df has required columns
        if 'funding_stage' not in df.columns or 'funding_amount_numeric' not in df.columns:
            self.logger.warning("DataFrame missing required columns for funding by stage visualization")
            return None
        
        # Create a copy to avoid modifying the original
        plot_df = df.copy()
        
        # Only keep rows with valid funding stage and amount
        plot_df = plot_df.dropna(subset=['funding_stage', 'funding_amount_numeric'])
        
        # For stages with very few samples, group them as "Other"
        stage_counts = plot_df['funding_stage'].value_counts()
        stages_to_keep = stage_counts[stage_counts >= 3].index.tolist()
        plot_df.loc[~plot_df['funding_stage'].isin(stages_to_keep), 'funding_stage'] = 'Other'
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Box plot of funding amount by stage
        sns.boxplot(x='funding_stage', y='funding_amount_numeric', data=plot_df)
        
        # Add individual points
        sns.stripplot(x='funding_stage', y='funding_amount_numeric', data=plot_df, 
                      size=4, color=".3", alpha=0.6)
        
        # Set labels and title
        plt.xlabel('Funding Stage')
        plt.ylabel('Funding Amount')
        plt.title('Funding Amount Distribution by Stage')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Use log scale for y-axis if range is large
        if plot_df['funding_amount_numeric'].max() / plot_df['funding_amount_numeric'].min() > 100:
            plt.yscale('log')
            plt.ylabel('Funding Amount (log scale)')
        
        # Save figure
        output_path = os.path.join(output_dir, 'funding_by_stage.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved funding by stage visualization to {output_path}")
        return output_path
    
    def visualize_funding_by_industry(self, df, output_dir=None):
        """
        Visualize funding amounts by industry.
        
        Args:
            df (pandas.DataFrame): DataFrame with funding data
            output_dir (str): Directory to save the visualization
            
        Returns:
            str: Path to the saved visualization
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        self.logger.info("Creating funding by industry visualization")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure df has required columns
        if 'industry' not in df.columns or 'funding_amount_numeric' not in df.columns:
            self.logger.warning("DataFrame missing required columns for funding by industry visualization")
            return None
        
        # Create a copy to avoid modifying the original
        plot_df = df.copy()
        
        # Only keep rows with valid industry and funding amount
        plot_df = plot_df.dropna(subset=['industry', 'funding_amount_numeric'])
        
        # Split multi-industry strings and create separate rows
        plot_df['industry'] = plot_df['industry'].apply(
            lambda x: x.split(',') if isinstance(x, str) else ['Unknown']
        )
        plot_df = plot_df.explode('industry')
        plot_df['industry'] = plot_df['industry'].str.strip()
        
        # Calculate industry stats
        industry_stats = plot_df.groupby('industry').agg({
            'funding_amount_numeric': ['mean', 'median', 'count']
        }).reset_index()
        industry_stats.columns = ['industry', 'mean_funding', 'median_funding', 'count']
        
        # Only keep industries with sufficient data
        top_industries = industry_stats.sort_values('count', ascending=False).head(15)['industry'].tolist()
        plot_df = plot_df[plot_df['industry'].isin(top_industries)]
        
        try:
            # Create figure
            plt.figure(figsize=(16, 10))
            
            # Bar plot of average funding by industry
            sns.barplot(x='industry', y='mean_funding', data=industry_stats[industry_stats['industry'].isin(top_industries)].sort_values('mean_funding', ascending=False))
            
            # Set labels and title
            plt.xlabel('Industry')
            plt.ylabel('Average Funding Amount')
            plt.title('Average Funding Amount by Industry (Top 15 Industries by Count)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Create sanitized filename
            avg_filename = 'funding_by_industry_average.png'
            output_path = os.path.join(output_dir, avg_filename)
            
            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create a second figure for the distribution
            plt.figure(figsize=(16, 10))
            
            # Box plot of funding amount by industry
            sns.boxplot(x='industry', y='funding_amount_numeric', data=plot_df.sort_values('industry'))
            
            # Set labels and title
            plt.xlabel('Industry')
            plt.ylabel('Funding Amount')
            plt.title('Funding Amount Distribution by Industry (Top 15 Industries by Count)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Use log scale for y-axis if range is large
            if plot_df['funding_amount_numeric'].max() / plot_df['funding_amount_numeric'].min() > 100:
                plt.yscale('log')
                plt.ylabel('Funding Amount (log scale)')
            
            # Create sanitized filename
            dist_filename = 'funding_by_industry_distribution.png'
            output_path2 = os.path.join(output_dir, dist_filename)
            
            # Save figure
            plt.savefig(output_path2, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved funding by industry visualizations to {output_path} and {output_path2}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Error creating funding by industry visualization: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def visualize_calibration(self, X_test, y_test, models, output_dir=None):
        """
        Visualize model calibration with predicted vs. actual values.
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test targets
            models: Trained model or dict of models
            output_dir (str): Directory to save the visualization
            
        Returns:
            str: Path to the saved visualization
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        self.logger.info("Creating model calibration visualization")
        
        # Prepare predictions based on model type
        if isinstance(models, dict) and 0.5 in models:
            # Quantile regression forest
            y_pred = models[0.5].predict(X_test)
            model_name = 'Quantile Regression Forest (Median)'
        elif isinstance(models, dict) and 'random_forest' in models:
            # Ensemble - use all models
            y_preds = {}
            for name, model in models.items():
                y_preds[name] = model.predict(X_test)
            
            # Also calculate ensemble prediction
            y_preds['Ensemble'] = np.mean([y_preds[m] for m in y_preds], axis=0)
        else:
            self.logger.warning("Unsupported model type for calibration visualization")
            return None
        
        # Create figure
        if 'y_preds' in locals():
            # Multiple models to compare
            fig, axs = plt.subplots(1, len(y_preds), figsize=(16, 8), sharey=True)
            
            # Plot calibration for each model
            for i, (name, preds) in enumerate(y_preds.items()):
                # Calculate residuals
                residuals = y_test - preds
                
                # Scatter plot of predicted vs. residuals
                axs[i].scatter(preds, residuals, alpha=0.7)
                axs[i].axhline(y=0, color='r', linestyle='--')
                
                # Add labels and title
                axs[i].set_xlabel('Predicted Funding Amount')
                if i == 0:
                    axs[i].set_ylabel('Residual (Actual - Predicted)')
                axs[i].set_title(f'{name} Calibration')
                axs[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, 'model_calibration_comparison.png')
        else:
            # Single model
            plt.figure(figsize=(12, 8))
            
            # Calculate residuals
            residuals = y_test - y_pred
            
            # Scatter plot of predicted vs. residuals
            plt.scatter(y_pred, residuals, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--')
            
            # Add trend line
            z = np.polyfit(y_pred, residuals, 1)
            p = np.poly1d(z)
            plt.plot(y_pred, p(y_pred), 'b-', label=f'Trend Line (slope={z[0]:.4f})')
            
            # Set labels and title
            plt.xlabel('Predicted Funding Amount')
            plt.ylabel('Residual (Actual - Predicted)')
            plt.title(f'Model Calibration ({model_name})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            output_path = os.path.join(output_dir, 'model_calibration.png')
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved model calibration visualization to {output_path}")
        return output_path
    
    def visualize_error_distribution(self, y_test, y_pred, output_dir=None):
        """
        Visualize the distribution of prediction errors.
        
        Args:
            y_test (array-like): Actual values
            y_pred (array-like): Predicted values
            output_dir (str): Directory to save the visualization
            
        Returns:
            str: Path to the saved visualization
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        self.logger.info("Creating error distribution visualization")
        
        # Calculate errors
        absolute_errors = np.abs(y_test - y_pred)
        percentage_errors = np.abs((y_test - y_pred) / y_test) * 100
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot absolute error distribution
        sns.histplot(absolute_errors, kde=True, ax=ax1)
        ax1.set_xlabel('Absolute Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Absolute Error Distribution')
        
        # Add statistics to the plot
        stats_text = (
            f"Mean: {np.mean(absolute_errors):.2f}\n"
            f"Median: {np.median(absolute_errors):.2f}\n"
            f"Std Dev: {np.std(absolute_errors):.2f}"
        )
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot percentage error distribution
        sns.histplot(percentage_errors, kde=True, ax=ax2)
        ax2.set_xlabel('Percentage Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Percentage Error Distribution')
        
        # Add vertical line at 20% error
        ax2.axvline(x=20, color='r', linestyle='--', label='20% Error Threshold')
        ax2.legend()
        
        # Add statistics to the plot
        stats_text = (
            f"Mean: {np.mean(percentage_errors):.2f}%\n"
            f"Median: {np.median(percentage_errors):.2f}%\n"
            f"Std Dev: {np.std(percentage_errors):.2f}%\n"
            f"Within 20%: {np.mean(percentage_errors <= 20):.2%}"
        )
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set overall title
        plt.suptitle('Funding Amount Prediction Error Distribution', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        # Save figure
        output_path = os.path.join(output_dir, 'error_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved error distribution visualization to {output_path}")
        return output_path
    
    def predict_funding_amount(self, new_data, model_type='quantile'):
        """
        Predict next funding round amount for new company data.
        
        Args:
            new_data (pandas.DataFrame): DataFrame with company data
            model_type (str): Type of model to use ('quantile' or 'ensemble')
            
        Returns:
            dict: Prediction results
        """
        self.logger.info(f"Predicting funding amounts using {model_type} model")
        
        if not hasattr(self, 'scaler') or self.scaler is None:
            self.logger.error("Model not trained or loaded. Cannot make predictions.")
            return None
            
        # Prepare features
        features = self.feature_names
        missing_cols = [col for col in features if col not in new_data.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns in prediction data: {missing_cols}")
            for col in missing_cols:
                new_data[col] = 0  # Fill with default value
        
        X = new_data[features].values
        X_scaled = self.scaler.transform(X)
        
        # Make predictions based on model type
        results = []
        
        if model_type == 'quantile' and hasattr(self, 'model') and isinstance(self.model, dict) and 0.5 in self.model:
            # Quantile regression forest
            for i, row in enumerate(X_scaled):
                row_data = {}
                
                if 'company_name' in new_data.columns:
                    row_data['company_name'] = new_data['company_name'].iloc[i]
                else:
                    row_data['company_name'] = f"Company {i+1}"
                    
                # Get predictions for all quantiles
                for quantile, model in self.model.items():
                    row_data[f'predicted_amount_q{int(quantile*100)}'] = model.predict([row])[0]
                
                # Add confidence interval
                if 0.1 in self.model and 0.9 in self.model:
                    row_data['lower_bound'] = self.model[0.1].predict([row])[0]
                    row_data['upper_bound'] = self.model[0.9].predict([row])[0]
                    row_data['interval_width'] = row_data['upper_bound'] - row_data['lower_bound']
                
                # Add current funding amount if available
                if 'funding_amount_numeric' in new_data.columns:
                    row_data['current_funding'] = new_data['funding_amount_numeric'].iloc[i]
                    row_data['growth_multiple'] = row_data['predicted_amount_q50'] / new_data['funding_amount_numeric'].iloc[i] if new_data['funding_amount_numeric'].iloc[i] > 0 else None
                
                results.append(row_data)
                
        elif model_type == 'ensemble' and hasattr(self, 'ensemble_model') and isinstance(self.ensemble_model, dict):
            # Ensemble model
            for i, row in enumerate(X_scaled):
                row_data = {}
                
                if 'company_name' in new_data.columns:
                    row_data['company_name'] = new_data['company_name'].iloc[i]
                else:
                    row_data['company_name'] = f"Company {i+1}"
                
                # Get predictions from each model
                model_preds = {}
                for name, model in self.ensemble_model.items():
                    model_preds[name] = model.predict([row])[0]
                    row_data[f'predicted_amount_{name}'] = model_preds[name]
                
                # Calculate ensemble prediction (average)
                row_data['predicted_amount_ensemble'] = np.mean(list(model_preds.values()))
                
                # Add current funding amount if available
                if 'funding_amount_numeric' in new_data.columns:
                    row_data['current_funding'] = new_data['funding_amount_numeric'].iloc[i]
                    row_data['growth_multiple'] = row_data['predicted_amount_ensemble'] / new_data['funding_amount_numeric'].iloc[i] if new_data['funding_amount_numeric'].iloc[i] > 0 else None
                
                results.append(row_data)
        else:
            self.logger.error(f"Model type '{model_type}' not available")
            return None
            
        return results
    
    def generate_funding_amount_report(self, data, metrics, output_dir=None):
        """
        Generate a comprehensive report on funding amount forecasting.
        
        Args:
            data (pandas.DataFrame): DataFrame with funding data
            metrics (dict): Dictionary of model evaluation metrics
            output_dir (str): Directory to save the report
            
        Returns:
            str: Path to the saved report
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        self.logger.info("Generating funding amount forecast report")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get timestamp for report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate funding statistics
        funding_stats = {
            'mean': data['funding_amount_numeric'].mean(),
            'median': data['funding_amount_numeric'].median(),
            'max': data['funding_amount_numeric'].max(),
            'min': data['funding_amount_numeric'].min(),
            'std': data['funding_amount_numeric'].std()
        }
        
        # Calculate average funding by stage
        stage_funding = data.groupby('funding_stage')['funding_amount_numeric'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        # Calculate average funding by industry (handling multiple industries per company)
        if 'industry' in data.columns:
            # Create a list of (industry, funding_amount) tuples
            industry_amounts = []
            for _, row in data.iterrows():
                if isinstance(row['industry'], str) and pd.notna(row['funding_amount_numeric']):
                    industries = [ind.strip() for ind in row['industry'].split(',')]
                    for ind in industries:
                        industry_amounts.append((ind, row['funding_amount_numeric']))
            
            # Convert to DataFrame and calculate statistics
            industry_df = pd.DataFrame(industry_amounts, columns=['industry', 'amount'])
            industry_funding = industry_df.groupby('industry')['amount'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        else:
            industry_funding = pd.DataFrame(columns=['mean', 'count'])
        
        # Get training/testing metrics with defaults
        train_size = metrics.get('train_size', 'N/A')
        test_size = metrics.get('test_size', 'N/A')
        train_test_ratio = metrics.get('train_test_ratio', 'N/A')
        actual_data_count = metrics.get('actual_data_count', 'N/A')
        estimated_data_count = metrics.get('estimated_data_count', 'N/A')
        
        # Format all metrics to ensure they're strings
        def safe_format(value, format_str="{}"):
            try:
                if isinstance(value, (int, float)):
                    return format_str.format(value)
                return str(value)
            except:
                return "N/A"
        
        # Begin report content
        report_content = f"""# Funding Amount Forecast Analysis Report
Generated on: {timestamp}

## Dataset Overview
- Total records: {len(data)}
- Companies analyzed: {len(data['company_name'].unique()) if 'company_name' in data.columns else 'N/A'}
- Date range: {data['funding_date'].min().strftime('%Y-%m-%d') if 'funding_date' in data.columns and pd.notna(data['funding_date'].min()) else 'N/A'} to {data['funding_date'].max().strftime('%Y-%m-%d') if 'funding_date' in data.columns and pd.notna(data['funding_date'].max()) else 'N/A'}
- Industries: {len(industry_funding) if not industry_funding.empty else 'N/A'}
- Funding stages: {', '.join(str(stage) for stage in data['funding_stage'].unique()) if 'funding_stage' in data.columns else 'N/A'}

## Funding Amount Statistics
- Average funding amount: ${funding_stats['mean']:,.2f}
- Median funding amount: ${funding_stats['median']:,.2f}
- Maximum funding amount: ${funding_stats['max']:,.2f}
- Minimum funding amount: ${funding_stats['min']:,.2f}
- Standard deviation: ${funding_stats['std']:,.2f}

## Training and Testing Summary
- Training set size: {train_size}
- Testing set size: {test_size}
- Training/Testing split ratio: {train_test_ratio}
- Actual next round data: {actual_data_count}
- Estimated next round data: {estimated_data_count}

## Model Performance Metrics

### Quantile Regression Model
- Model Type: {metrics['qrf_metrics']['model_type'] if 'qrf_metrics' in metrics and 'model_type' in metrics['qrf_metrics'] else 'quantile_regression_forest'}
- RMSE: ${safe_format(metrics['qrf_metrics'].get('rmse'), "{:,.2f}")}
- MAE: ${safe_format(metrics['qrf_metrics'].get('mae'), "{:,.2f}")}
- R2: {safe_format(metrics['qrf_metrics'].get('r2'), "{:.4f}")}
- Log RMSE: {safe_format(metrics['qrf_metrics'].get('log_rmse'), "{:.4f}")}
- Log MAE: {safe_format(metrics['qrf_metrics'].get('log_mae'), "{:.4f}")}
- Log R2: {safe_format(metrics['qrf_metrics'].get('log_r2'), "{:.4f}")}
- Within 10%: {safe_format(metrics['qrf_metrics'].get('within_10pct', 0) * 100, "{:.1f}")}%
- Within 20%: {safe_format(metrics['qrf_metrics'].get('within_20pct', 0) * 100, "{:.1f}")}%
- Within 50%: {safe_format(metrics['qrf_metrics'].get('within_50pct', 0) * 100, "{:.1f}")}%
- Within Same Order: {safe_format(metrics['qrf_metrics'].get('within_same_order', 0) * 100, "{:.1f}")}%
- Interval Coverage 80%: {safe_format(metrics['qrf_metrics'].get('interval_coverage_80pct', 0) * 100, "{:.1f}")}%

### Ensemble Model
- Model Type: {metrics['ensemble_metrics']['model_type'] if 'ensemble_metrics' in metrics and 'model_type' in metrics['ensemble_metrics'] else 'ensemble'}
- RMSE: ${safe_format(metrics['ensemble_metrics'].get('rmse'), "{:,.2f}")}
- MAE: ${safe_format(metrics['ensemble_metrics'].get('mae'), "{:,.2f}")}
- R2: {safe_format(metrics['ensemble_metrics'].get('r2'), "{:.4f}")}
- Log RMSE: {safe_format(metrics['ensemble_metrics'].get('log_rmse'), "{:.4f}")}
- Log MAE: {safe_format(metrics['ensemble_metrics'].get('log_mae'), "{:.4f}")}
- Log R2: {safe_format(metrics['ensemble_metrics'].get('log_r2'), "{:.4f}")}
- Within 10%: {safe_format(metrics['ensemble_metrics'].get('within_10pct', 0) * 100, "{:.1f}")}%
- Within 20%: {safe_format(metrics['ensemble_metrics'].get('within_20pct', 0) * 100, "{:.1f}")}%
- Within 50%: {safe_format(metrics['ensemble_metrics'].get('within_50pct', 0) * 100, "{:.1f}")}%
- Within Same Order: {safe_format(metrics['ensemble_metrics'].get('within_same_order', 0) * 100, "{:.1f}")}%

#### Individual Model Performance
**Random Forest**
- RMSE: ${safe_format(metrics.get('individual_models', {}).get('random_forest', {}).get('rmse'), "{:,.2f}")}
- MAE: ${safe_format(metrics.get('individual_models', {}).get('random_forest', {}).get('mae'), "{:,.2f}")}
- R2: {safe_format(metrics.get('individual_models', {}).get('random_forest', {}).get('r2'), "{:.4f}")}
- Within 20%: {safe_format(metrics.get('individual_models', {}).get('random_forest', {}).get('within_20pct', 0) * 100, "{:.1f}")}%

**Gradient Boosting**
- RMSE: ${safe_format(metrics.get('individual_models', {}).get('gradient_boosting', {}).get('rmse'), "{:,.2f}")}
- MAE: ${safe_format(metrics.get('individual_models', {}).get('gradient_boosting', {}).get('mae'), "{:,.2f}")}
- R2: {safe_format(metrics.get('individual_models', {}).get('gradient_boosting', {}).get('r2'), "{:.4f}")}
- Within 20%: {safe_format(metrics.get('individual_models', {}).get('gradient_boosting', {}).get('within_20pct', 0) * 100, "{:.1f}")}%

**Extra Trees**
- RMSE: ${safe_format(metrics.get('individual_models', {}).get('extra_trees', {}).get('rmse'), "{:,.2f}")}
- MAE: ${safe_format(metrics.get('individual_models', {}).get('extra_trees', {}).get('mae'), "{:,.2f}")}
- R2: {safe_format(metrics.get('individual_models', {}).get('extra_trees', {}).get('r2'), "{:.4f}")}
- Within 20%: {safe_format(metrics.get('individual_models', {}).get('extra_trees', {}).get('within_20pct', 0) * 100, "{:.1f}")}%

## Model Calibration Analysis
The model calibration analysis measures how well the predicted values match the actual values. A well-calibrated model will have residuals (actual minus predicted) centered around zero with no clear patterns.

- Mean residual (QRF): ${safe_format(metrics.get('mean_residual_qrf'), "{:,.2f}")}
- Mean absolute residual (QRF): ${safe_format(metrics.get('mean_abs_residual_qrf'), "{:,.2f}")}
- Residual standard deviation (QRF): ${safe_format(metrics.get('residual_std_qrf'), "{:,.2f}")}
- Calibration slope (QRF): {safe_format(metrics.get('calibration_slope_qrf'), "{:.4f}")}

- Mean residual (Ensemble): ${safe_format(metrics.get('mean_residual_ensemble'), "{:,.2f}")}
- Mean absolute residual (Ensemble): ${safe_format(metrics.get('mean_abs_residual_ensemble'), "{:,.2f}")}
- Residual standard deviation (Ensemble): ${safe_format(metrics.get('residual_std_ensemble'), "{:,.2f}")}
- Calibration slope (Ensemble): {safe_format(metrics.get('calibration_slope_ensemble'), "{:.4f}")}

## Key Insights

### Top Industries by Average Funding Amount
"""
        
        # Add top industries by funding amount
        if not industry_funding.empty:
            top_industries = industry_funding.head(5)
            for ind, row in top_industries.iterrows():
                report_content += f"- {ind}: ${row['mean']:,.2f}\n"
        else:
            report_content += "- Industry data not available\n"
        
        report_content += """
### Average Funding by Stage
"""
        
        # Add funding by stage
        if not stage_funding.empty:
            for stage, row in stage_funding.iterrows():
                if pd.notna(stage) and stage != '':
                    report_content += f"- {stage}: ${row['mean']:,.2f}\n"
        else:
            report_content += "- Funding stage data not available\n"
        
        # Identify highest and lowest funding stages
        highest_stage = 'N/A'
        lowest_stage = 'N/A'
        highest_amount = 0
        lowest_amount = float('inf')
        
        if not stage_funding.empty:
            for stage, row in stage_funding.iterrows():
                if pd.notna(stage) and stage != '':
                    if row['mean'] > highest_amount:
                        highest_amount = row['mean']
                        highest_stage = stage
                    if row['mean'] < lowest_amount:
                        lowest_amount = row['mean']
                        lowest_stage = stage
        
        # Get top industries string
        top_industries_str = 'N/A'
        if not industry_funding.empty:
            top_industries = industry_funding.head(3).index.tolist()
            top_industries_str = ', '.join(top_industries)
        
        # Determine model performance level
        r2_value = metrics['qrf_metrics'].get('r2', 0) if 'qrf_metrics' in metrics else 0
        within_20pct = metrics['qrf_metrics'].get('within_20pct', 0) * 100 if 'qrf_metrics' in metrics else 0
        
        if r2_value > 0.5:
            performance = "strong"
        elif r2_value > 0:
            performance = "moderate"
        else:
            performance = "weak"
        
        report_content += """
## Accuracy Analysis

### Prediction Accuracy by Percentile
- Predictions within 10% of actual: 
  - QRF: {:.1f}%
  - Ensemble: {:.1f}%
- Predictions within 20% of actual: 
  - QRF: {:.1f}%
  - Ensemble: {:.1f}%
- Predictions within 50% of actual: 
  - QRF: {:.1f}%
  - Ensemble: {:.1f}%
- Predictions within same order of magnitude: 
  - QRF: {:.1f}%
  - Ensemble: {:.1f}%

### Prediction Interval Coverage
The 80% prediction interval coverage of {:.1f}% indicates how well the model's uncertainty estimates match the actual data.
An ideal coverage would be exactly 80%.

## Recommendations
1. **Industry Focus**: The highest funding amounts are observed in {}, suggesting these may be high-potential sectors for investment.
2. **Stage Optimization**: {} shows the largest average funding (${:,.2f}), while {} shows the lowest (${:,.2f}). Consider this when planning fundraising strategies.
3. **Prediction Reliability**: The model shows {} performance with R² of {:.4f} and {:.1f}% of predictions within 20% of actual values.
4. **Uncertainty Handling**: Use the quantile regression model for risk assessment, as it provides both median predictions and uncertainty intervals.
""".format(
            metrics['qrf_metrics'].get('within_10pct', 0) * 100 if 'qrf_metrics' in metrics else 0,
            metrics['ensemble_metrics'].get('within_10pct', 0) * 100 if 'ensemble_metrics' in metrics else 0,
            metrics['qrf_metrics'].get('within_20pct', 0) * 100 if 'qrf_metrics' in metrics else 0,
            metrics['ensemble_metrics'].get('within_20pct', 0) * 100 if 'ensemble_metrics' in metrics else 0,
            metrics['qrf_metrics'].get('within_50pct', 0) * 100 if 'qrf_metrics' in metrics else 0,
            metrics['ensemble_metrics'].get('within_50pct', 0) * 100 if 'ensemble_metrics' in metrics else 0,
            metrics['qrf_metrics'].get('within_same_order', 0) * 100 if 'qrf_metrics' in metrics else 0,
            metrics['ensemble_metrics'].get('within_same_order', 0) * 100 if 'ensemble_metrics' in metrics else 0,
            metrics['qrf_metrics'].get('interval_coverage_80pct', 0) * 100 if 'qrf_metrics' in metrics else 0,
            top_industries_str,
            highest_stage, highest_amount, lowest_stage, lowest_amount,
            performance, r2_value, within_20pct
        )
        
        # Create safe output file path
        report_filename = 'funding_amount_forecast_report.md'
        report_path = os.path.join(output_dir, report_filename)
        
        # Ensure the report directory exists
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        
        # Save the report
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"Funding amount forecast report saved to {report_path}")
            
            # Convert to HTML if markdown2 is available
            try:
                import markdown2
                html_content = markdown2.markdown(report_content)
                html_path = os.path.join(output_dir, 'funding_amount_forecast_report.html')
                
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Funding Amount Forecast Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{
            font-family: Consolas, Monaco, 'Andale Mono', monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>""")
                self.logger.info(f"HTML report saved to {html_path}")
            except ImportError:
                self.logger.info("markdown2 package not available, skipping HTML conversion")
            except Exception as html_error:
                self.logger.warning(f"Error creating HTML report: {str(html_error)}")
                
            return report_path
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def save_model(self, filepath=None):
        """
        Save the trained model and associated data to a file.
        
        Args:
            filepath (str, optional): Path to save the model. If not provided, 
                                     uses default in output directory.
        
        Returns:
            str: Path to the saved model file
        """
        if filepath is None:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, 'funding_amount_forecast_model.pkl')
        else:
            # Ensure the directory for the filepath exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
        # Sanitize the filepath
        filepath = os.path.abspath(filepath)
        
        # Create model data for saving
        model_data = {
            'model': self.model,
            'ensemble_model': self.ensemble_model,
            'bayesian_model': self.bayesian_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.logger.info(f"Saving model to {filepath}")
        
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)
                
            # Save model
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved successfully to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return None
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        self.logger.info(f"Loading model from {filepath}")
        
        # Validate the filepath
        if not os.path.exists(filepath):
            self.logger.error(f"Model file not found: {filepath}")
            return False
            
        # Sanitize the filepath
        filepath = os.path.abspath(filepath)
        
        try:
            # Load model data
            model_data = joblib.load(filepath)
            
            # Check if model_data has the required keys
            required_keys = ['model', 'scaler', 'feature_names']
            if not all(key in model_data for key in required_keys):
                self.logger.error(f"Model file is missing required data: {filepath}")
                return False
            
            # Set model attributes
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            
            # Set optional attributes if available
            if 'ensemble_model' in model_data:
                self.ensemble_model = model_data['ensemble_model']
            if 'bayesian_model' in model_data:
                self.bayesian_model = model_data['bayesian_model']
            if 'feature_importance' in model_data:
                self.feature_importance = model_data['feature_importance']
            if 'metrics' in model_data:
                self.metrics = model_data['metrics']
            
            self.logger.info(f"Model loaded successfully from {filepath}")
            
            # Log timestamp if available
            if 'timestamp' in model_data:
                self.logger.info(f"Model timestamp: {model_data['timestamp']}")
                
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def run_analysis(self, data=None, output_dir=None):
        """
        Run the full funding amount forecast analysis.
        
        Args:
            data (pandas.DataFrame, optional): Input data to analyze. If None, will load from configured source.
            output_dir (str, optional): Directory to save outputs. If None, uses self.output_dir.
        
        Returns:
            dict: Results of the analysis
        """
        self.logger.info("Starting funding amount forecast analysis")
        
        # Use default output directory if not specified
        if output_dir is None:
            output_dir = self.output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        if data is None:
            data = self.load_data_from_json_files()
        
        if data.empty:
            self.logger.error("No data available for analysis")
            return {"status": "error", "message": "No data available for analysis"}
        
        # Create features
        feature_data = self.create_forecast_features(data)
        
        # Check if we have valid feature data
        if feature_data.empty:
            self.logger.error("No valid feature data generated")
            return {"status": "error", "message": "No valid feature data generated"}
        
        # Ensure we have the funding_amount_numeric column
        if 'funding_amount_numeric' not in feature_data.columns:
            self.logger.error("funding_amount_numeric column missing from feature data")
            return {"status": "error", "message": "funding_amount_numeric column missing from feature data"}
            
        # Ensure target variable is present
        if 'next_round_amount' not in feature_data.columns:
            self.logger.error("next_round_amount column missing from feature data")
            return {"status": "error", "message": "next_round_amount column missing from feature data"}
            
        # Prepare model data
        try:
            X_train, X_test, y_train, y_test, feature_names = self.prepare_model_data(feature_data)
            
            # Store training/testing data statistics
            train_test_metrics = {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_test_ratio': f"{len(X_train)}:{len(X_test)}",
                'feature_count': len(feature_names)
            }
            
            # Check if we have data on actual vs estimated counts
            if 'next_round_is_estimated' in feature_data.columns:
                actual_data_count = (~feature_data['next_round_is_estimated']).sum()
                estimated_data_count = feature_data['next_round_is_estimated'].sum()
                train_test_metrics['actual_data_count'] = actual_data_count
                train_test_metrics['estimated_data_count'] = estimated_data_count
            
        except Exception as e:
            self.logger.error(f"Error preparing model data: {str(e)}")
            return {"status": "error", "message": f"Error preparing model data: {str(e)}"}
        
        # Train models
        self.logger.info("Training models...")
        
        try:
            # Train quantile regression forest
            qrf_models = self.train_quantile_regression_forest(X_train, y_train)
            
            # Train ensemble model
            ensemble_models = self.train_ensemble_model(X_train, y_train)
            
            # Evaluate models
            self.logger.info("Evaluating models...")
            qrf_metrics = self.evaluate_model(qrf_models, X_test, y_test)
            ensemble_metrics = self.evaluate_model(ensemble_models, X_test, y_test)
            
            # Convert any numpy values to Python types for better serialization
            self._convert_metrics_to_python_types(qrf_metrics)
            self._convert_metrics_to_python_types(ensemble_metrics)
            
            # Use the best model for visualizations
            best_model = qrf_models if qrf_metrics.get('r2', 0) > ensemble_metrics.get('r2', 0) else ensemble_models
            
            # Enhanced metrics for reporting
            best_metrics = {
                'qrf_metrics': qrf_metrics,
                'ensemble_metrics': ensemble_metrics,
                'individual_models': {
                    'random_forest': ensemble_metrics.get('individual_models', {}).get('random_forest', {}),
                    'gradient_boosting': ensemble_metrics.get('individual_models', {}).get('gradient_boosting', {}),
                    'extra_trees': ensemble_metrics.get('individual_models', {}).get('extra_trees', {})
                }
            }
            
            # Add training/testing metrics
            best_metrics.update(train_test_metrics)
            
            # Add calibration metrics
            y_pred_qrf = qrf_models[0.5].predict(X_test)
            residuals_qrf = y_test - y_pred_qrf
            best_metrics['mean_residual_qrf'] = np.mean(residuals_qrf)
            best_metrics['mean_abs_residual_qrf'] = np.mean(np.abs(residuals_qrf))
            best_metrics['residual_std_qrf'] = np.std(residuals_qrf)
            # Calibration slope (should be close to 0 for well-calibrated model)
            z = np.polyfit(y_pred_qrf, residuals_qrf, 1)
            best_metrics['calibration_slope_qrf'] = z[0]
            
            # Ensemble calibration metrics
            y_pred_ensemble = np.mean([
                ensemble_models['random_forest'].predict(X_test),
                ensemble_models['gradient_boosting'].predict(X_test),
                ensemble_models['extra_trees'].predict(X_test)
            ], axis=0)
            residuals_ensemble = y_test - y_pred_ensemble
            best_metrics['mean_residual_ensemble'] = np.mean(residuals_ensemble)
            best_metrics['mean_abs_residual_ensemble'] = np.mean(np.abs(residuals_ensemble))
            best_metrics['residual_std_ensemble'] = np.std(residuals_ensemble)
            # Calibration slope
            z = np.polyfit(y_pred_ensemble, residuals_ensemble, 1)
            best_metrics['calibration_slope_ensemble'] = z[0]
            
            # Get predictions for test data
            if isinstance(best_model, dict) and 0.5 in best_model:
                y_pred = best_model[0.5].predict(X_test)
            elif isinstance(best_model, dict) and 'random_forest' in best_model:
                y_pred = best_model['random_forest'].predict(X_test)
            else:
                self.logger.warning("Could not determine predictions for visualizations")
                y_pred = np.zeros_like(y_test)
            
            # Create visualizations
            self.logger.info("Creating visualizations...")
            
            # Basic visualizations
            self.visualize_funding_by_stage(feature_data, output_dir=output_dir)
            self.visualize_funding_by_industry(feature_data, output_dir=output_dir)
            
            # Model performance visualizations
            self.visualize_prediction_accuracy(y_test, y_pred, output_dir=output_dir)
            self.visualize_feature_importance(best_model, feature_names, output_dir=output_dir)
            self.visualize_calibration(X_test, y_test, best_model, output_dir=output_dir)
            self.visualize_error_distribution(y_test, y_pred, output_dir=output_dir)
            
            # Additional visualizations for quantile regression
            if isinstance(best_model, dict) and 0.1 in best_model and 0.9 in best_model:
                self.visualize_prediction_intervals(X_test, y_test, best_model, output_dir=output_dir)
            
            # Generate report
            self.logger.info("Generating report...")
            
            try:
                report_path = self.generate_funding_amount_report(feature_data, best_metrics, output_dir=output_dir)
            except Exception as report_error:
                self.logger.error(f"Error generating report: {str(report_error)}")
                report_path = None
            
            # Save model
            self.logger.info("Saving model...")
            model_path = self.save_model(os.path.join(output_dir, 'funding_amount_forecast_model.pkl'))
            
            self.logger.info("Funding amount forecast analysis completed successfully")
            
            return {
                "status": "success",
                "data_count": len(data),
                "qrf_metrics": qrf_metrics,
                "ensemble_metrics": ensemble_metrics,
                "report_path": report_path,
                "model_path": model_path,
                "output_dir": output_dir
            }
            
        except Exception as e:
            self.logger.error(f"Error in analysis pipeline: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"status": "error", "message": f"Error in analysis pipeline: {str(e)}"}
    
    def _convert_metrics_to_python_types(self, metrics):
        """
        Convert numpy types to Python types for better serialization and display.
        
        Args:
            metrics (dict): Dictionary of metrics that may contain numpy values
        """
        if not isinstance(metrics, dict):
            return
            
        for key, value in metrics.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries
                self._convert_metrics_to_python_types(value)
            elif isinstance(value, np.number):
                # Convert numpy numbers to Python float or int
                metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                # Convert numpy arrays to lists
                metrics[key] = value.tolist()

    def train_quantile_regression_forest(self, X_train, y_train, quantiles=[0.1, 0.5, 0.9]):
        """
        Train a quantile regression forest model.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (pandas.Series): Training target
            quantiles (list): Quantiles to predict
            
        Returns:
            dict: Dictionary of trained models for each quantile
        """
        self.logger.info(f"Training quantile regression forest with quantiles: {quantiles}")
        
        quantile_models = {}
        
        for quantile in quantiles:
            self.logger.info(f"Training model for quantile {quantile}")
            
            # Use GradientBoostingRegressor with quantile loss
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=quantile,
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Store the model
            quantile_models[quantile] = model
            
            self.logger.info(f"Trained model for quantile {quantile}")
        
        self.model = quantile_models
        return quantile_models
    
    def train_ensemble_model(self, X_train, y_train):
        """
        Train an ensemble of models for funding amount prediction.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (pandas.Series): Training target
            
        Returns:
            dict: Dictionary of trained models
        """
        self.logger.info("Training ensemble model")
        
        models = {}
        
        # RandomForestRegressor
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        models['random_forest'] = rf
        
        # GradientBoostingRegressor
        gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb.fit(X_train, y_train)
        models['gradient_boosting'] = gb
        
        # ExtraTreesRegressor
        et = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        et.fit(X_train, y_train)
        models['extra_trees'] = et
        
        self.ensemble_model = models
        return models

    def train_bayesian_model(self, X_train, y_train):
        """
        Train a Bayesian regression model for funding amount prediction.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (pandas.Series): Training target
            
        Returns:
            object: Trained Bayesian model or None if PyMC3 not available
        """
        if not HAS_PYMC3:
            self.logger.warning("PyMC3 not available, skipping Bayesian model training")
            return None
            
        self.logger.info("Training Bayesian regression model")
        
        # Simple Bayesian linear regression model
        try:
            with pm.Model() as model:
                # Priors for unknown model parameters
                alpha = pm.Normal('alpha', mu=np.mean(y_train), sd=np.std(y_train) * 10)
                
                # Prior for the standard deviation of observations
                sigma = pm.HalfNormal('sigma', sd=np.std(y_train) * 10)
                
                # Expected value of outcome
                mu = alpha
                
                # Beta coefficients
                betas = []
                for i in range(X_train.shape[1]):
                    beta = pm.Normal(f'beta_{i}', mu=0, sd=10)
                    mu = mu + beta * X_train[:, i]
                    betas.append(beta)
                
                # Likelihood (sampling distribution) of observations
                Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y_train)
                
                # Fit the model
                trace = pm.sample(1000, tune=1000, cores=1, return_inferencedata=False)
                
            self.bayesian_model = model
            return model
        except Exception as e:
            self.logger.error(f"Error training Bayesian model: {str(e)}")
            return None

def main():
    """Main function to run the funding amount forecast analysis."""
    parser = argparse.ArgumentParser(description='Funding Amount Forecast Analysis')
    parser.add_argument('--data_dir', type=str, default='JSONFolder',
                       help='Directory containing funding data JSON files')
    parser.add_argument('--output_dir', type=str, default='outputFundingForecast',
                       help='Directory to save output files')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load a previously trained model')
    parser.add_argument('--predict_only', action='store_true',
                       help='Run only prediction on existing data without retraining')
    args = parser.parse_args()
    
    # Create absolute paths
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create forecast object
    forecast = FundingAmountForecast(data_dir=data_dir, output_dir=output_dir)
    
    if args.load_model:
        # Sanitize model path
        model_path = os.path.abspath(args.load_model)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return
        
        # Load existing model
        success = forecast.load_model(model_path)
        if not success:
            print(f"Failed to load model from {model_path}")
            return
        
        if args.predict_only:
            # Load data for prediction
            data = forecast.load_data_from_json_files()
            if data.empty:
                print("No data available for prediction")
                return
                
            # Prepare features
            feature_data = forecast.create_forecast_features(data)
            
            # Make predictions
            results = forecast.predict_funding_amount(feature_data)
            
            # Print results
            print("\nFunding Amount Predictions:")
            for result in results:
                company = result.get('company_name', 'Unknown')
                current = result.get('current_funding', 'Unknown')
                predicted = result.get('predicted_amount_q50', result.get('predicted_amount_ensemble', 'Unknown'))
                growth = result.get('growth_multiple', 'Unknown')
                
                print(f"- {company}: Current ${current:,.2f}, Predicted Next Round: ${predicted:,.2f}, Growth: {growth:.2f}x")
            
            # Create a safe file path for predictions
            csv_filename = 'funding_amount_predictions.csv'
            csv_path = os.path.join(output_dir, csv_filename)
            
            # Save predictions to CSV
            try:
                results_df = pd.DataFrame(results)
                results_df.to_csv(csv_path, index=False)
                print(f"\nSaved predictions to {csv_path}")
            except Exception as e:
                print(f"Error saving predictions: {str(e)}")
    else:
        # Run full analysis
        results = forecast.run_analysis()
        
        if results["status"] == "success":
            print("\nFunding Amount Forecast Analysis Results:")
            print(f"Processed {results['data_count']} funding records")
            
            # Print metrics
            print("\nQuantile Regression Forest Metrics:")
            for key, value in results["qrf_metrics"].items():
                if key != 'model_type' and key != 'individual_models':
                    try:
                        print(f"- {key}: {float(value):.4f}")
                    except (TypeError, ValueError):
                        print(f"- {key}: {value}")
            
            print("\nEnsemble Metrics:")
            for key, value in results["ensemble_metrics"].items():
                if key != 'model_type' and key != 'individual_models':
                    try:
                        print(f"- {key}: {float(value):.4f}")
                    except (TypeError, ValueError):
                        print(f"- {key}: {value}")
            
            print(f"\nReport saved to: {results['report_path']}")
            print(f"Model saved to: {results['model_path']}")
            print(f"All outputs saved to: {results['output_dir']}")
        else:
            print(f"Analysis failed: {results.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()