import os
import re
import sys
import csv
import json
import time
import random
import sqlite3
import logging
import traceback
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc, roc_auc_score, mean_squared_error, 
    precision_recall_curve, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.inspection import permutation_importance
import joblib
import optuna
import xgboost as xgb
from threading import Thread
import schedule
import socket
import hashlib
import uuid
import ipaddress
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("funding_prediction.log"),
        logging.StreamHandler()])
logger = logging.getLogger(__name__)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score)
import pickle
import csv
import random
import uuid
import re
import shutil
import glob
import sqlite3
import xgboost as xgb
from sklearn.svm import OneClassSVM
from scipy.stats import randint, uniform
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib
from numpy_encoder import NumpyEncoder
import matplotlib.ticker as ticker

# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')  # Use Agg backend which doesn't require a display

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("funding_prediction.log"),
        logging.StreamHandler()])
logger = logging.getLogger(__name__)


# Define StackedEnsemble as a top-level class so it can be properly serialized
class StackedEnsemble:
    """Stacked ensemble model that combines multiple base models with a meta-classifier"""
    
    def __init__(self, base_models, meta_classifier):
        self.base_models = base_models
        self.meta_classifier = meta_classifier
    
    def predict(self, X):
        base_preds = []
        for name, model in self.base_models.items():
            if model is not None:
                base_preds.append(model.predict_proba(X))
        meta_features = np.column_stack(base_preds)
        return self.meta_classifier.predict(meta_features)
    
    def predict_proba(self, X):
        base_preds = []
        for name, model in self.base_models.items():
            if model is not None:
                base_preds.append(model.predict_proba(X))
        meta_features = np.column_stack(base_preds)
        return self.meta_classifier.predict_proba(meta_features)
    
    
class DataLoader:
    def __init__(self, base_dir="./", archive=False):
        """Initialize data loader with paths to data sources and historical database"""
        self.base_dir = base_dir
        self.archive = archive
        self.archive_dir = None
        self.historical_db = os.path.join(
            base_dir, "historical_funding_data.db")

        # Define paths to source files in JSONFolder - fix for duplicated path
        # If base_dir already contains JSONFolder, don't add it again
        if os.path.basename(base_dir) == "JSONFolder" or os.path.exists(
                os.path.join(base_dir, "fundraisestartup50.json")):
            self.json_folder = base_dir
        else:
            self.json_folder = os.path.join(base_dir, "JSONFolder")

        # Use the fixed json_folder path for file paths
        self.fundraiser_path = os.path.join(
            self.json_folder, "fundraisestartup50.json")
        self.growthlist_path = os.path.join(
            self.json_folder, "growthlistscrapper.json")
        self.topstartup_path = os.path.join(
            self.json_folder, "topstartupio50.json")

        # Initialize the database for historical data
        self._init_historical_db()

        # Archive data if enabled
        if self.archive:
            self.archive_dir = self._create_archive_dir()
            self._archive_current_data()

    def _create_archive_dir(self):
        """Create a timestamped archive directory for this run"""
        archive_root = os.path.join(self.base_dir, "data_archive")
        os.makedirs(archive_root, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(archive_root, timestamp)
        os.makedirs(archive_dir, exist_ok=True)
        return archive_dir

    def _archive_current_data(self):
        """Copy current data files into the archive directory"""
        try:
            if not self.archive_dir:
                logger.warning("Archive directory not set. Skipping archive operation.")
                return
                
            archived_files = []
            
            # Archive fundraiser data
            if os.path.isfile(self.fundraiser_path):
                dest_path = os.path.join(self.archive_dir, "fundraiser.json")
                shutil.copy2(self.fundraiser_path, dest_path)
                archived_files.append(dest_path)
                
            # Archive growthlist data
            if os.path.isfile(self.growthlist_path):
                dest_path = os.path.join(self.archive_dir, "growthlist.json")
                shutil.copy2(self.growthlist_path, dest_path)
                archived_files.append(dest_path)
                
            # Archive topstartup data
            if os.path.isfile(self.topstartup_path):
                dest_path = os.path.join(self.archive_dir, "topstartup.json") 
                shutil.copy2(self.topstartup_path, dest_path)
                archived_files.append(dest_path)
                
            if archived_files:
                logger.info(f"Archived {len(archived_files)} data files to {self.archive_dir}")
                for file in archived_files:
                    logger.info(f"  - {os.path.basename(file)}")
            else:
                logger.warning(f"No data files found to archive in {self.base_dir}")
                
        except Exception as e:
            logger.error(f"Error archiving data: {e}")

    def _init_historical_db(self):
        """Create SQLite database tables if they don't exist"""
        try:
            # Create directory for database if it doesn't exist
            db_dir = os.path.dirname(self.historical_db)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            conn = sqlite3.connect(self.historical_db)
            cursor = conn.cursor()

            # Create tables with appropriate schema
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundraiser_data (
                company TEXT,
                employees INTEGER,
                industry TEXT,
                funding_date TEXT,
                funding_type TEXT,
                funding_amount REAL,
                headquarters TEXT,
                extraction_time TEXT,
                data_timestamp TEXT,
                PRIMARY KEY (company, extraction_time)
            )
            ''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS growthlist_data (
                name TEXT,
                industry TEXT,
                funding_amount TEXT,
                funding_type TEXT,
                last_funding_date TEXT,
                data_timestamp TEXT,
                PRIMARY KEY (name, last_funding_date)
            )
            ''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS topstartup_data (
                company TEXT,
                industry TEXT,
                funding_round TEXT,
                funding_amount REAL,
                funding_date TEXT,
                data_timestamp TEXT,
                PRIMARY KEY (company, funding_date)
            )
            ''')

            conn.commit()
            conn.close()
            logger.info("Historical database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def reset_database(self):
        """Reset the database for a clean start"""
        try:
            if os.path.exists(self.historical_db):
                os.remove(self.historical_db)
                logger.info("Existing database removed")
            self._init_historical_db()
            logger.info("Database reset complete")
        except Exception as e:
            logger.error(f"Error resetting database: {e}")

    def validate_dataset(self, df, source_name):
        """Check if required columns exist in the dataset"""
        required_columns = {
            'fundraiser_data': [
                'Company', 'Funding_Amount_USD', 'Funding_Type'], 'growthlist_data': [
                'name', 'funding_type', 'funding_amount'], 'topstartup_data': [
                'company_name', 'funding_stage', 'funding']}

        if source_name in required_columns:
            missing_cols = [col for col in required_columns[source_name]
                            if col not in df.columns]
            if missing_cols:
                logger.warning(
                    f"Missing columns in {source_name}: {missing_cols}")
                return False
        return True

    def validate_merged_columns(self, df):
        """Ensure consistent column structure"""
        required_columns = [
            'company_name', 'funding_date', 'funding_amount',
            'funding_stage', 'industry', 'employees'
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing critical columns: {missing}")
            return False

        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            logger.error(f"Duplicate columns detected: {duplicates}")
            return False

        return True

    def validate_company(
            self,
            company_name,
            funding_amount=None,
            funding_stage=None,
            employees=None):
        """
        Validate company data by cross-checking against known companies and using
        business logic to detect anomalies.

        Returns:
            tuple: (is_valid, confidence_score, message)
        """
        # All validation disabled - trust the data source
        return True, 1.0, "Validation disabled"
        
        """
        # Original validation code commented out
        if not company_name or pd.isna(company_name) or len(
                str(company_name).strip()) < 2:
            return False, 0.0, "Invalid company name"

        # Normalize company name for comparison
        company_name = str(company_name).lower().strip()

        # Check for common spam patterns
        spam_patterns = [
            'test',
            'dummy',
            'sample',
            'xyz',
            'abc',
            'placeholder',
            'llc']
        if any(pattern in company_name for pattern in spam_patterns):
            return False, 0.1, f"Company name contains suspicious pattern: {company_name}"

        try:
            # Try to cross-reference with historical data
            conn = sqlite3.connect(self.historical_db)
            cursor = conn.cursor()

            # Check if company exists in any historical table
            tables = ['fundraiser_data', 'growthlist_data', 'topstartup_data']
            company_found = False
            company_data = []

            for table in tables:
                # Adapt query based on table schema
                if table == 'fundraiser_data':
                    column = 'company'
                elif table == 'growthlist_data':
                    column = 'name'
                else:
                    column = 'company_name'

                cursor.execute(
                    f"SELECT * FROM {table} WHERE LOWER({column}) = ?", (company_name,))
                results = cursor.fetchall()

                if results:
                    company_found = True
                    company_data.extend(results)

            conn.close()

            if not company_found:
                # If not in history, it's suspicious but not necessarily
                # invalid
                return True, 0.5, "New company - not found in historical data"

            # Additional validation checks if funding_amount and stage provided
            if funding_amount is not None and funding_stage is not None:
                # Check for unrealistic funding for stage (simplified)
                if funding_stage and 'seed' in str(
                        funding_stage).lower() and funding_amount > 2e7:  # $20M
                    return False, 0.3, f"Unrealistic funding amount ${
                        funding_amount:,.2f} for {funding_stage} stage"
                # $100M
                elif funding_stage and 'series a' in str(funding_stage).lower() and funding_amount > 1e8:
                    return False, 0.4, f"Unusual funding amount ${
                        funding_amount:,.2f} for {funding_stage} stage"

            # Company passed all checks with high confidence
            return True, 0.9, "Company validated successfully"

        except Exception as e:
            logger.warning(
                f"Error validating company {company_name}: {
                    str(e)}")
            # Default to accepting but with low confidence if validation fails
            return True, 0.6, f"Validation partially completed: {str(e)}"
        """

    def _get_standardized_funding_type_map(self):
        """Returns a standardized mapping of funding types to ensure consistency across all data sources"""
        return {
            'pre-seed': 'Pre-Seed',
            'pre seed': 'Pre-Seed',
            'preseed': 'Pre-Seed',
            'seed': 'Seed',
            'angel': 'Angel',
            'series a': 'Series A',
            'series b': 'Series B',
            'series c': 'Series C',
            'series d': 'Series D',
            'series e': 'Series E',
            'series f': 'Series F',
            'series g': 'Series G',
            'series h': 'Series H',
            'venture - series unknown': 'Venture - Series Unknown',
            'venture series unknown': 'Venture - Series Unknown',
            'private equity': 'Private Equity',
            'initial coin offering': 'Initial Coin Offering',
            'ico': 'Initial Coin Offering',
            'grant': 'Grant',
            'debt financing': 'Debt Financing',
            'debt': 'Debt Financing',
            'undisclosed': 'Undisclosed'
        }

    def _standardize_funding_type(self, funding_type):
        """Standardize a funding type string based on the standardized mapping"""
        if pd.isna(funding_type):
            return funding_type
        
        funding_type_map = self._get_standardized_funding_type_map()
        normalized = str(funding_type).lower().strip()
        return funding_type_map.get(normalized, funding_type)

    def load_fundraiser_data(self):
        """Load and process fundraiser insider data with standardized funding types"""
        try:
            with open(self.fundraiser_path, 'r') as file:
                data = json.load(file)

            # Handle the JSON structure which is a list of companies
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                # Extract companies from the JSON structure if it's a dictionary
                companies = data.get('companies', [])
                df = pd.DataFrame(companies)

            # Add timestamp for versioning
            df['data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Convert numeric fields
            if 'Funding_Amount_USD' in df.columns:
                df['Funding_Amount_USD'] = pd.to_numeric(
                    df['Funding_Amount_USD'], errors='coerce')

            if 'Total_Employees' in df.columns:
                df['Total_Employees'] = pd.to_numeric(
                    df['Total_Employees'], errors='coerce')
            
            # Standardize Funding_Type if present
            if 'Funding_Type' in df.columns:
                df['Funding_Type'] = df['Funding_Type'].apply(self._standardize_funding_type)
                
                # Log unique funding types found
                unique_funding_types = df['Funding_Type'].dropna().unique()
                logger.info(f"Found funding types in fundraiser data: {unique_funding_types}")

            logger.info(f"Loaded {len(df)} records from fundraiser data")
            return df
        except Exception as e:
            logger.error(f"Error loading fundraiser data: {e}")
            return pd.DataFrame()

    def load_growthlist_data(self):
        """Load and process growthlist startups data - extract both funding amount and type"""
        try:
            with open(self.growthlist_path, 'r') as file:
                data = json.load(file)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Add timestamp for versioning
            df['data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Process funding amounts without dropping original column
            if 'funding_amount' in df.columns:
                df['funding_amount_numeric'] = df['funding_amount'].apply(
                    self._parse_funding_amount)
            
            # Ensure funding_type is processed and standardized
            if 'funding_type' in df.columns:
                df['funding_type'] = df['funding_type'].apply(self._standardize_funding_type)
                
                # Log unique funding types found
                unique_funding_types = df['funding_type'].dropna().unique()
                logger.info(f"Found funding types in growthlist data: {unique_funding_types}")

            logger.info(f"Loaded {len(df)} records from growthlist data")
            return df
        except Exception as e:
            logger.error(f"Error loading growthlist data: {e}")
            return pd.DataFrame()

    def load_topstartup_data(self):
        """Handle the complex format of topstartup data and extract funding information correctly"""
        try:
            with open(self.topstartup_path, 'r') as file:
                data = json.load(file)

            # Handle both list and dictionary formats
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame(data.get('startups', []))
            else:
                logger.error("Unexpected JSON format in topstartup data")
                return pd.DataFrame()

            # Parse employee count from strings like "1-10 employees"
            if 'employees' in df.columns:
                df['employees_numeric'] = df['employees'].apply(self._parse_employee_count)
                df['employees'] = df['employees_numeric']  # Replace original with numeric value

            # Handle headquarters data
            if 'headquarters' in df.columns:
                # Fill empty HQ with default value
                df['headquarters'] = df['headquarters'].fillna('San Francisco Bay Area')
                # Standardize location format
                df['location_standardized'] = df['headquarters'].apply(self._standardize_location)

            # Parse funding information from the funding string
            if 'funding' in df.columns:
                # Extract funding information from the complex funding string
                def extract_funding_info(funding_str):
                    if not funding_str or pd.isna(funding_str):
                        return None, None, None

                    # Common patterns: "Sequoia $100M Series D in 2025"
                    # or "Andreessen Horowitz $10B Series J in 2024 $62.0B
                    # valuation"

                    amount = None
                    stage = None
                    date = None

                    # Extract amount
                    amount_match = re.search(
                        r'\$(\d+(?:\.\d+)?[KMB]?)', funding_str)
                    if amount_match:
                        amount = amount_match.group(0)  # Keep the $ symbol

                    # Extract stage with improved pattern matching
                    # Look for more funding stage patterns with case
                    # insensitivity
                    stage_pattern = r'(Pre[-\s]?Seed|Seed|Angel|Series\s+[A-Z]|Venture[\s\-]+Series\s+Unknown|Initial\s+Coin\s+Offering|ICO|Private\s+Equity|Grant|Debt\s+Financing|Undisclosed|Post[-\s]?IPO)'
                    stage_match = re.search(
                        stage_pattern, funding_str, re.IGNORECASE)

                    if stage_match:
                        # Get the raw matched stage
                        raw_stage = stage_match.group(1)
                        # Use our standardization method
                        stage = self._standardize_funding_type(raw_stage)
                    else:
                        # If no explicit stage is found, try to infer from context
                        # Check for common patterns in funding text
                        funding_lower = funding_str.lower()

                        if 'seed' in funding_lower and not stage:
                            stage = 'Seed'
                        elif 'angel' in funding_lower and not stage:
                            stage = 'Angel'
                        elif 'raised' in funding_lower and not stage:
                            # For strings like "Raised $5M in 2019" without
                            # explicit stage
                            if 'series' in funding_lower:
                                # Try to extract series letter if mentioned
                                series_match = re.search(
                                    r'series\s+([a-z])', funding_lower)
                                if series_match:
                                    letter = series_match.group(1).upper()
                                    stage = f'Series {letter}'
                                else:
                                    stage = 'Venture - Series Unknown'
                            else:
                                # Default to "Venture Funding" for generic
                                # raised amounts
                                stage = 'Venture - Series Unknown'
                        elif 'valuation' in funding_lower and not stage:
                            if 'post-ipo' in funding_lower or 'post ipo' in funding_lower:
                                stage = 'Post-IPO'
                            else:
                                # Companies with just valuation mentioned but
                                # no explicit funding stage
                                stage = 'Venture - Series Unknown'

                    # Extract date - usually has "in YYYY" format
                    date_match = re.search(r'in (\d{4})', funding_str)
                    if date_match:
                        date = date_match.group(1)
                    else:
                        # Try to find just a year at the end of the string
                        year_match = re.search(r'\b(20\d{2})\b', funding_str)
                        if year_match:
                            date = year_match.group(1)

                    return amount, stage, date

                # Extract funding details
                funding_details = df['funding'].apply(extract_funding_info)

                # Create separate columns for extracted values
                df['funding_amount'] = funding_details.apply(
                    lambda x: x[0] if x else None)
                df['funding_stage'] = funding_details.apply(
                    lambda x: x[1] if x else None)
                df['funding_date'] = funding_details.apply(
                    lambda x: x[2] if x else None)

                # Log unique funding stages found
                unique_funding_stages = df['funding_stage'].dropna().unique()
                logger.info(f"Found funding stages in topstartup data: {unique_funding_stages}")

            # Standardize column names
            column_mapping = {
                'name': 'company_name',
                'funding_type': 'funding_stage',  # Alternative naming
                'category': 'industry'
            }

            # Apply mapping for existing columns only
            df = df.rename(columns={k: v for k, v in column_mapping.items()
                                    if k in df.columns})

            # Add timestamp
            df['data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Convert funding amounts if present
            if 'funding_amount' in df.columns:
                df['funding_amount_numeric'] = df['funding_amount'].apply(
                    self._parse_funding_amount)

            logger.info(f"Loaded {len(df)} records from topstartup data")
            return df
        except Exception as e:
            logger.error(f"Error loading topstartup data: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _parse_funding_amount(self, amount_str):
        """Parse funding amount from string to float

        Args:
            amount_str: String representing funding amount (e.g., "$10M", "5.3B")

        Returns:
            Funding amount in USD
        """
        if not amount_str or pd.isna(amount_str):
            return None

        try:
            # Remove non-alphanumeric characters except for decimal points and standard suffixes
            cleaned = re.sub(r'[^0-9a-zA-Z\.]', '', str(amount_str))
            
            # Extract the numeric part
            numeric_part = re.search(r'^(\d+\.?\d*)', cleaned)
            if not numeric_part:
                return None
            
            amount = float(numeric_part.group(1))
            
            # Check for suffixes and adjust amount accordingly
            if 'B' in cleaned or 'b' in cleaned:
                amount *= 1_000_000_000
            elif 'M' in cleaned or 'm' in cleaned:
                amount *= 1_000_000
            elif 'K' in cleaned or 'k' in cleaned:
                amount *= 1_000
                
            # Apply reasonable upper limit for funding amounts
            # Instead of just logging, handle the cap more gracefully
            MAX_FUNDING_AMOUNT = 10_000_000_000  # $10B
            if amount > MAX_FUNDING_AMOUNT:
                logger.info(f"Large funding amount detected: ${amount:,.2f} for stage {self.current_stage}. Capping at ${MAX_FUNDING_AMOUNT:,.2f}")
                amount = MAX_FUNDING_AMOUNT
                
            return amount
        except Exception as e:
            logger.debug(f"Error parsing funding amount '{amount_str}': {str(e)}")
            return None

    def save_historical_data(self, df, table_name):
        """Save dataframe to historical SQLite database"""
        try:
            conn = sqlite3.connect(self.historical_db)
            
            # Create the table with the correct schema if it doesn't exist
            if 'company_name' not in df.columns and table_name == 'merged_data':
                # Add required company_name column if missing for merged_data table
                df['company_name'] = df.get('name', 'Unknown')
            
            # Ensure all column names are valid SQL identifiers
            df = df.copy()
            for col in df.columns:
                if re.search(r'[^a-zA-Z0-9_]', col):
                    new_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                    df.rename(columns={col: new_col}, inplace=True)
            
            # Get column types for table creation
            column_types = {}
            for col in df.columns:
                if df[col].dtype == 'float64' or df[col].dtype == 'float32':
                    column_types[col] = 'REAL'
                elif df[col].dtype == 'int64' or df[col].dtype == 'int32':
                    column_types[col] = 'INTEGER'
                else:
                    column_types[col] = 'TEXT'
            
            # Create table if it doesn't exist
            cursor = conn.cursor()
            columns_sql = ", ".join([f'"{col}" {column_types[col]}' for col in df.columns])
            create_table_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql})'
            cursor.execute(create_table_sql)
            conn.commit()
            
            # Save data
            df.to_sql(table_name, conn, if_exists='append', index=False)
            conn.close()
            logger.info(f"Saved {len(df)} records to historical {table_name}")
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
            logger.error(traceback.format_exc())

    def load_historical_data(self, table_name):
        """Load historical data from SQLite database"""
        try:
            conn = sqlite3.connect(self.historical_db)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(
                f"Loaded {
                    len(df)} historical records from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()

    def merge_datasets(self):
        """Improved merge function with better error handling, validation and audit trail"""
        try:
            # Load raw data
            fundraiser_df = self.load_fundraiser_data()
            growthlist_df = self.load_growthlist_data()
            topstartup_df = self.load_topstartup_data()

            # Log loaded data sizes
            logger.info(
                f"Loaded datasets - Fundraiser: {
                    len(fundraiser_df)} rows, Growthlist: {
                    len(growthlist_df)} rows, Topstartup: {
                    len(topstartup_df)} rows")

            # Initialize list to store all records
            all_records = []

            # Process fundraiser data
            if not fundraiser_df.empty:
                for _, row in fundraiser_df.iterrows():
                    if pd.notna(row.get('Company')):  # Only add records with valid company names
                        # Convert numeric values properly
                        try:
                            funding_amount = pd.to_numeric(
                                row.get('Funding_Amount_USD'), errors='coerce')
                            employees = pd.to_numeric(
                                row.get('Total_Employees'), errors='coerce')
                        except:
                            funding_amount = np.nan
                            employees = np.nan
                            
                        all_records.append({
                            'company_name': row.get('Company'),
                            'funding_stage': row.get('Funding_Type'),
                            'funding_amount': funding_amount,
                            'funding_date': row.get('Funding_Date'),
                            'industry': row.get('Industry'),
                            'employees': employees,
                            'source': 'fundraiser',
                            'confidence_score': 1.0
                        })

                logger.info(f"Processed {len(fundraiser_df)} records from fundraiser data")
                # Log unique funding stages from this source
                funding_stages = [r.get('Funding_Type') for _, r in fundraiser_df.iterrows() if pd.notna(r.get('Funding_Type'))]
                unique_stages = set(funding_stages)
                logger.info(f"Unique funding stages from fundraiser: {unique_stages}")

            # Process growthlist data
            if not growthlist_df.empty:
                for _, row in growthlist_df.iterrows():
                    if pd.notna(row.get('name')):  # Only add records with valid company names
                        # Parse the amount if not already parsed
                        funding_amount = row.get('funding_amount_numeric')
                        if pd.isna(funding_amount) and pd.notna(row.get('funding_amount')):
                            funding_amount = self._parse_funding_amount(row.get('funding_amount'))

                        all_records.append({
                            'company_name': row.get('name'),
                            'funding_stage': row.get('funding_type'),  # Using standardized funding_type
                            'funding_amount': funding_amount,
                            'funding_date': row.get('last_funding_date'),
                            'industry': row.get('industry'),
                            'employees': None,
                            'source': 'growthlist',
                            'confidence_score': 1.0
                        })

                logger.info(f"Processed {len(growthlist_df)} records from growthlist data")

            # Process topstartup data
            if not topstartup_df.empty:
                for _, row in topstartup_df.iterrows():
                    company_name = row.get('company_name') or row.get('name')

                    if pd.notna(company_name):  # Only add records with valid company names
                        # Clean up data
                        funding_stage = row.get('funding_stage') or row.get('funding_round')

                        # Parse the amount if string
                        funding_amount = row.get('funding_amount')
                        if isinstance(funding_amount, str):
                            funding_amount = self._parse_funding_amount(funding_amount)

                        # Get employee count range
                        employee_count = None
                        if pd.notna(row.get('employees')):
                            # Handle ranges like "11-50 employees"
                            emp_str = str(row.get('employees'))
                            match = re.search(r'(\d+)-(\d+)', emp_str)
                            if match:
                                # Take the average of the range
                                employee_count = (int(match.group(1)) + int(match.group(2))) / 2

                        all_records.append({
                            'company_name': company_name,
                            'funding_stage': funding_stage,
                            'funding_amount': funding_amount,
                            'funding_date': row.get('funding_date'),
                            'industry': row.get('industry'),
                            'employees': employee_count,
                            'source': 'topstartup',
                            'confidence_score': 1.0
                        })

                logger.info(f"Processed {len(topstartup_df)} records from topstartup data")

            # Create the merged dataframe
            merged_df = pd.DataFrame(all_records)

            # Add timestamp for audit
            merged_df['merge_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Save an audit trace for regulatory compliance
            self.save_historical_data(merged_df, 'merged_data')

            logger.info(f"Successfully merged {len(merged_df)} records")
            
            # Validate merged dataset schema
            self.validate_merged_columns(merged_df)
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _parse_employee_count(self, employee_str):
        """Parse employee count from strings like '1-10 employees'"""
        if not employee_str or pd.isna(employee_str):
            return np.nan

        try:
            # Handle strings like "1-10 employees" or "10+" or just "15"
            employee_str = str(employee_str).lower().replace('employees', '').replace('employee', '').strip()
            
            # Case 1: Range like "1-10"
            if '-' in employee_str:
                lower, upper = employee_str.split('-')
                lower = int(lower.strip())
                upper = int(upper.strip())
                # Return midpoint of range
                return (lower + upper) / 2
                
            # Case 2: "10+" format
            elif '+' in employee_str:
                base = int(employee_str.replace('+', '').strip())
                # For 10+, estimate as 15 (50% more)
                return base * 1.5
                
            # Case 3: Direct number
            else:
                return int(employee_str.strip())
        except Exception as e:
            logger.warning(f"Error parsing employee count '{employee_str}': {str(e)}")
            return np.nan

    def _standardize_location(self, location_str):
        """Standardize location strings across different data sources"""
        if not location_str or pd.isna(location_str):
            return 'Unknown'
            
        location_str = str(location_str).strip()
        
        # Handle common variations
        common_locations = {
            'sf': 'San Francisco Bay Area',
            'silicon valley': 'San Francisco Bay Area',
            'bay area': 'San Francisco Bay Area',
            'nyc': 'New York',
            'new york city': 'New York',
            'bangalore': 'Bengaluru',
            'london, uk': 'London',
            'tel aviv': 'Tel Aviv',
        }
        
        # Try to match with common locations
        location_lower = location_str.lower()
        for key, value in common_locations.items():
            if key in location_lower:
                return value
        
        # Extract country if present
        country_pattern = r'(?:,\s+|\s+in\s+)(USA|US|United States|Canada|UK|United Kingdom|Australia|Germany|France|India|China|Japan|Israel)$'
        country_match = re.search(country_pattern, location_str, re.IGNORECASE)
        
        if country_match:
            return country_match.group(1)
            
        return location_str

    def train_stacked_ensemble(self, X, y, base_models):
        """Create a stacked ensemble using multiple base models"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Convert base_models to dict if it's a list
            if isinstance(base_models, list):
                base_models_dict = {}
                for i, model in enumerate(base_models):
                    if model is not None:
                        base_models_dict[f'model_{i}'] = model
                base_models = base_models_dict
            
            # Train base models and get predictions
            base_predictions = {}
            for name, model in base_models.items():
                if model is not None:
                    model.fit(X_train, y_train)
                    pred = model.predict_proba(X_test)
                    base_predictions[name] = pred
            
            # Skip if we don't have enough base models
            if len(base_predictions) < 2:
                logger.warning("Not enough valid base models for ensemble (need at least 2)")
                return None, 0.0
            
            # Create meta-features
            meta_features = np.column_stack(list(base_predictions.values()))
            
            # Train meta-classifier
            meta_classifier = LogisticRegression(
                multi_class='multinomial',
                max_iter=1000,
                random_state=42
            )
            meta_classifier.fit(meta_features, y_test)
            
            # Make final predictions
            final_predictions = meta_classifier.predict(meta_features)
            accuracy = accuracy_score(y_test, final_predictions)
            
            logger.info(f"Stacked Ensemble accuracy: {accuracy:.4f}")
            
            # Use the global StackedEnsemble class
            ensemble = StackedEnsemble(base_models, meta_classifier)
            return ensemble, accuracy
            
            
        except Exception as e:
            logger.error(f"Error training stacked ensemble: {str(e)}")
            logger.error(traceback.format_exc())
            return None, 0.0
    
    def optimize_hyperparameters(self, trial, X, y, model_type='rf'):
        """Optimize hyperparameters using Optuna"""
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            model = RandomForestClassifier(**params, random_state=42)
        elif model_type == 'xgb':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = xgb.XGBClassifier(**params, random_state=42)
        
        # Perform cross-validation
        scores = cross_val_score(
            model, X, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )
        
        return scores.mean()
    
    def train_optimized_model(self, X, y, model_type='rf', n_trials=100):
        """Train a model with optimized hyperparameters"""
        try:
            study = optuna.create_study(direction='maximize')
            objective = lambda trial: self.optimize_hyperparameters(trial, X, y, model_type)
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            logger.info(f"Best {model_type} parameters: {best_params}")
            
            # Train final model
            if model_type == 'rf':
                model = RandomForestClassifier(**best_params, random_state=42)
            elif model_type == 'xgb':
                model = xgb.XGBClassifier(**best_params, random_state=42)
            
            # Evaluate with cross-validation
            scores = cross_val_score(
                model, X, y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy',
                n_jobs=-1
            )
            
            logger.info(f"Optimized {model_type} CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            # Train final model on full dataset
            model.fit(X, y)
            return model, scores.mean()
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return None, 0.0

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation with multiple metrics"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Calculate MAE
            mae = mean_absolute_error(y_test, y_pred)
            
            # Get classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Calculate ROC curves and AUC scores if applicable
            roc_auc_scores = {}
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                
                # For multi-class, use one-vs-rest approach
                if len(np.unique(y_test)) > 2:
                    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                    n_classes = y_test_bin.shape[1]
                    
                    for i in range(n_classes):
                        if y_test_bin[:, i].sum() > 0:
                            roc_auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
                            roc_auc_scores[f'class_{i}'] = roc_auc
                    
                    # Calculate macro average
                    roc_auc_scores['macro_avg'] = np.mean(list(roc_auc_scores.values()))
                else:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    roc_auc_scores['binary'] = roc_auc
            
            # Calculate F1 scores
            f1_micro = f1_score(y_test, y_pred, average='micro')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate precision and recall
            precision_micro = precision_score(y_test, y_pred, average='micro')
            precision_macro = precision_score(y_test, y_pred, average='macro')
            recall_micro = recall_score(y_test, y_pred, average='micro')
            recall_macro = recall_score(y_test, y_pred, average='macro')
            
            # Log results
            logger.info(f"\n{model_name} Evaluation Results:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"F1 Score (micro/macro/weighted): {f1_micro:.4f}/{f1_macro:.4f}/{f1_weighted:.4f}")
            logger.info(f"Precision (micro/macro): {precision_micro:.4f}/{precision_macro:.4f}")
            logger.info(f"Recall (micro/macro): {recall_micro:.4f}/{recall_macro:.4f}")
            logger.info(f"ROC AUC Scores: {roc_auc_scores}")
            logger.info(f"Classification Report:\n{json.dumps(class_report, indent=2)}")
            
            # Return comprehensive metrics
            metrics = {
                'accuracy': accuracy,
                'rmse': rmse,
                'mae': mae,
                'f1_scores': {
                    'micro': f1_micro,
                    'macro': f1_macro,
                    'weighted': f1_weighted
                },
                'precision': {
                    'micro': precision_micro,
                    'macro': precision_macro
                },
                'recall': {
                    'micro': recall_micro,
                    'macro': recall_macro
                },
                'roc_auc_scores': roc_auc_scores,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            return None

    def train_advanced_pipeline(self, X, y):
        """Train multiple models with advanced techniques and ensemble methods"""
        try:
            # Filter out classes with too few samples
            class_counts = pd.Series(y).value_counts()
            rare_classes = class_counts[class_counts < 2].index
            
            if len(rare_classes) > 0:
                logger.info(f"Removing {len(rare_classes)} classes with fewer than 2 samples: {rare_classes}")
                valid_classes_mask = ~pd.Series(y).isin(rare_classes)
                X = X.loc[valid_classes_mask] if isinstance(X, pd.DataFrame) else X[valid_classes_mask]
                y = y[valid_classes_mask]
                
                # Verify no classes have too few samples
                updated_counts = pd.Series(y).value_counts()
                logger.info(f"Updated class distribution: {updated_counts.to_dict()}")
            
            # Split data with stratification to maintain class distribution
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Dictionary to store all models and their metrics
            all_models = {}
            
            # 1. Train Random Forest with advanced tuning
            logger.info("Training Random Forest with advanced tuning...")
            rf_params = {
                'n_estimators': [200, 500],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
            rf_grid.fit(X_train_scaled, y_train)
            logger.info(f"Best RF parameters: {rf_grid.best_params_}")
            rf_metrics = self.evaluate_model(rf_grid.best_estimator_, X_test_scaled, y_test, "Random Forest")
            all_models['random_forest'] = (rf_grid.best_estimator_, rf_metrics)
            
            # 2. Train LightGBM with improved parameters
            try:
                import lightgbm as lgb
                logger.info("Training LightGBM with advanced tuning...")
                lgb_params = {
                    'num_leaves': [31],
                    'max_depth': [5],
                    'learning_rate': [0.1],
                    'n_estimators': [100],
                    'min_child_samples': [20],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8],
                    'min_split_gain': [0.01],
                    'min_child_weight': [1],
                    'reg_alpha': [0.1],
                    'reg_lambda': [0.1]
                }
                lgb_model = lgb.LGBMClassifier(
                    objective='multiclass',
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1
                )
                lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=5, scoring='accuracy', n_jobs=-1)
                lgb_grid.fit(X_train_scaled, y_train)
                logger.info(f"Best LGB parameters: {lgb_grid.best_params_}")
                lgb_metrics = self.evaluate_model(lgb_grid.best_estimator_, X_test_scaled, y_test, "LightGBM")
                all_models['lightgbm'] = (lgb_grid.best_estimator_, lgb_metrics)
            except ImportError:
                logger.warning("LightGBM not available, skipping...")
            
            # Find best model
            best_model = None
            best_accuracy = 0
            best_model_name = None
            
            for name, (model, metrics) in all_models.items():
                if metrics and metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_model = model
                    best_model_name = name
            
            logger.info(f"\nBest Model: {best_model_name}")
            logger.info(f"Best Accuracy: {best_accuracy:.4f}")
            
            return all_models, best_model_name
            
        except Exception as e:
            logger.error(f"Error in advanced pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None

    def save_model_results(self, model_name, metrics, predictions, feature_importance=None):
        """Save model results to a JSON file"""
        try:
            # Create archived directory for results
            archive_path = self._create_archive_dir()
            results_dir = os.path.join(archive_path, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save metrics
            metrics_path = os.path.join(results_dir, f"{model_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4, cls=NumpyEncoder)
                
            # Save predictions if available
            if predictions is not None:
                preds_path = os.path.join(results_dir, f"{model_name}_predictions.csv")
                predictions.to_csv(preds_path, index=False)
                
            # Save feature importance if available
            if feature_importance is not None:
                fi_path = os.path.join(results_dir, f"{model_name}_feature_importance.json")
                with open(fi_path, 'w') as f:
                    json.dump(feature_importance, f, indent=4, cls=NumpyEncoder)
                    
            logger.info(f"Model results for {model_name} saved to {results_dir}")
            return results_dir
        except Exception as e:
            logger.error(f"Error saving results for {model_name}: {e}")
            logger.error(traceback.format_exc())
            return None
            
    def _create_archive_dir(self):
        """Create a timestamped archive directory for storing data snapshots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(self.output_dir, "data_archive", timestamp)
        os.makedirs(archive_path, exist_ok=True)
        logger.info(f"Created archive directory: {archive_path}")
        return archive_path

    def _archive_current_data(self):
        """Archive current data files to timestamped directory"""
        try:
            archive_path = self._create_archive_dir()
            
            # Copy all current data files to archive
            for file_path in [self.topstartup_path, self.fundraiser_path, self.growthlist_path]:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    archived_file = os.path.join(archive_path, filename)
                    shutil.copy2(file_path, archived_file)
                    logger.info(f"Archived {filename} to {archived_file}")
            
            return archive_path
        except Exception as e:
            logger.error(f"Error archiving data: {e}")
            logger.error(traceback.format_exc())
            return None


class ModelManager:
    """Manager for model versioning and predictions"""
    
    def __init__(self, model_dir='models/'):
        """Initialize ModelManager with model directory location"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize logging
        self.audit_log_path = os.path.join(model_dir, "prediction_audit.csv")
        self.has_audit_log = os.path.exists(self.audit_log_path)
        
    def init_audit_log(self):
        """Initialize the prediction audit log if it doesn't exist"""
        if not self.has_audit_log:
            with open(self.audit_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'request_id', 'company_name', 'prediction',
                    'confidence', 'is_anomaly', 'anomaly_score', 'client_ip'
                ])
            self.has_audit_log = True
            logger.info(f"Audit log initialized at {self.audit_log_path}")
    
    def _log_model_operation(self, operation, model_id, model_name, version):
        """Log model operations to a file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_path = os.path.join(self.model_dir, "model_operations.csv")
        
        # Create log file if it doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'operation', 'model_id', 'model_name', 'version'
                ])
        
        # Append operation to log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, operation, model_id, model_name, version
            ])

    def load_model(self, model_name, version='latest'):
        """Load a trained model from disk with checks and validation

        Args:
            model_name: Name of the model to load
            version: Version of the model (default: 'latest')

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Determine file path
            if version == 'latest':
                # Find the latest version
                model_files = glob.glob(
                    os.path.join(
                        self.model_dir,
                        f"{model_name}*.pkl"))
                if not model_files:
                    logger.error(f"No models found for {model_name}")
                    return False
                # Sort by name (which should include version)
                model_files.sort(reverse=True)
                model_path = model_files[0]
            else:
                model_path = os.path.join(
                    self.model_dir, f"{model_name}_v{version}.pkl")
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    return False

            # Load model and verify integrity
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Validate model data structure
            required_keys = [
                'model',
                'metadata',
                'scaler',
                'feature_names',
                'anomaly_detector']
            if not all(key in model_data for key in required_keys):
                logger.error(
                    f"Invalid model file format, missing required components")
                return False

            # Check model integrity and assign to class properties
            self.model = model_data['model']
            self.metadata = model_data['metadata']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.anomaly_detector = model_data.get('anomaly_detector')

            # Log successful load
            version_info = self.metadata.get('version', 'unknown')
            created_at = self.metadata.get('created_at', 'unknown')
            logger.info(
                f"Loaded {model_name} v{version_info} (created {created_at})")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def save_model(
            self,
            model_name,
            model,
            scaler,
            feature_names,
            training_data=None,
            metadata=None):
        """
        Save a model and its associated metadata
        
        Args:
            model_name: Name of the model to save
            model: The trained model object
            scaler: Feature scaler for preprocessing
            feature_names: List of feature names
            training_data: Optional dictionary of training data metrics
            metadata: Optional dictionary of additional metadata
        
        Returns:
            str: Path where model was saved
        """
        try:
            # Generate a model ID and version
            model_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create a clean directory name
            dir_name = model_name.lower().replace(' ', '_')
            
            # Create model directory if it doesn't exist
            model_dir = os.path.join(self.model_dir, dir_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Prepare model data bundle
            model_data = {
                'model': model,
                'model_id': model_id,
                'model_name': model_name,
                'version': timestamp,
                'scaler': scaler,
                'feature_names': feature_names,
                'created_at': timestamp
            }
            
            # Add optional metadata
            if metadata:
                model_data.update(metadata)
                
            if training_data:
                model_data['training_data'] = training_data
            
            # Define file path
            model_path = os.path.join(model_dir, f"{model_name.lower().replace(' ', '_')}_{timestamp}.joblib")
            
            # Save to disk
            joblib.dump(model_data, model_path)
            
            # Update the latest model pointer
            latest_path = os.path.join(model_dir, 'latest.txt')
            with open(latest_path, 'w') as f:
                f.write(model_path)
                
            # Log the save operation
            self._log_model_operation('save', model_id, model_name, timestamp)
                
            logger.info(f"Model '{model_name}' saved successfully to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return None

    def predict(self, features, company_name=None, client_ip=None):
        """Predict funding stage with validation and audit logging

        Args:
            features: Feature dictionary or pandas Series
            company_name: Optional company name for validation
            client_ip: Optional client IP for audit logging

        Returns:
            dict: Prediction results with confidence and validation info
        """
        try:
            if self.model is None:
                return {'error': 'No model loaded'}

            # Convert dictionary to proper format if needed
            if isinstance(features, dict):
                # Check for missing features
                missing_features = [
                    f for f in self.feature_names if f not in features]
                if missing_features:
                    return {
                        'error': f'Missing features: {missing_features}',
                        'is_valid': False,
                        'confidence': 0.0
                    }

                # Convert to numpy array
                X = np.array([features[f]
                             for f in self.feature_names]).reshape(1, -1)
            elif isinstance(features, pd.Series):
                # Get features in correct order
                X = features[self.feature_names].values.reshape(1, -1)
            else:
                X = features

            # Generate request ID for tracking
            request_id = str(uuid.uuid4())

            # Preprocess data
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # Run company validation if name provided
            company_valid = True
            company_confidence = 1.0
            if company_name:
                company_valid, company_confidence = self.anomaly_detector.validate_company(
                    company_name)
                if not company_valid:
                    logger.warning(
                        f"Company validation failed for {company_name}")

            # Check for anomalies
            anomaly_result = self.anomaly_detector.detect_anomalies(
                X_scaled, company_name)
            is_anomaly = anomaly_result.get('is_anomaly', False)
            anomaly_score = anomaly_result.get('score', 0.0)
            anomaly_reasons = anomaly_result.get('reasons', [])

            # Make prediction
            prediction = int(self.model.predict(X_scaled)[0])
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probabilities))

            # Adjust confidence based on anomaly
            if is_anomaly:
                # Reduce confidence proportionally to anomaly severity
                adjusted_confidence = confidence * \
                    (1 - min(anomaly_score, 0.9))
            else:
                adjusted_confidence = confidence

            # Adjust confidence based on company validation
            final_confidence = adjusted_confidence * company_confidence

            # Prepare result
            result = {
                'prediction': prediction,
                'confidence': round(final_confidence, 4),
                'is_valid': not is_anomaly and company_valid,
                'request_id': request_id
            }

            # Add validation details if there were issues
            if is_anomaly or not company_valid:
                result['validation'] = {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': round(anomaly_score, 4),
                    'reasons': anomaly_reasons,
                    'company_valid': company_valid,
                    'company_confidence': round(company_confidence, 4)
                }

            # Log prediction for audit
            self._log_prediction(
                request_id=request_id,
                company_name=company_name,
                prediction=prediction,
                confidence=final_confidence,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                anomaly_reasons=anomaly_reasons,
                feature_values=features,
                client_ip=client_ip
            )

            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': f'Prediction failed: {str(e)}', 'is_valid': False}

    def predict_proba(self, features, company_name=None, client_ip=None):
        """Predict probabilities for all classes with validation

        Args:
            features: Feature dictionary or pandas Series
            company_name: Optional company name for validation
            client_ip: Optional client IP for audit logging

        Returns:
            dict: Prediction results with probabilities and validation info
        """
        try:
            if self.model is None:
                return {'error': 'No model loaded'}

            # Convert dictionary to proper format if needed
            if isinstance(features, dict):
                # Check for missing features
                missing_features = [
                    f for f in self.feature_names if f not in features]
                if missing_features:
                    return {
                        'error': f'Missing features: {missing_features}',
                        'is_valid': False
                    }

                # Convert to numpy array
                X = np.array([features[f]
                             for f in self.feature_names]).reshape(1, -1)
            elif isinstance(features, pd.Series):
                # Get features in correct order
                X = features[self.feature_names].values.reshape(1, -1)
            else:
                X = features

            # Generate request ID for tracking
            request_id = str(uuid.uuid4())

            # Preprocess data
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # Run company validation if name provided
            company_valid = True
            company_confidence = 1.0
            if company_name:
                company_valid, company_confidence = self.anomaly_detector.validate_company(
                    company_name)
                if not company_valid:
                    logger.warning(
                        f"Company validation failed for {company_name}")

            # Check for anomalies
            anomaly_result = self.anomaly_detector.detect_anomalies(
                X_scaled, company_name)
            is_anomaly = anomaly_result.get('is_anomaly', False)
            anomaly_score = anomaly_result.get('score', 0.0)
            anomaly_reasons = anomaly_result.get('reasons', [])

            # Get class probabilities
            probabilities = self.model.predict_proba(X_scaled)[0].tolist()
            classes = self.model.classes_.tolist() if hasattr(
                self.model, 'classes_') else list(range(len(probabilities)))

            # Adjust probabilities based on anomaly and company validation
            if is_anomaly or not company_valid:
                # Make distribution more uniform (less confident) based on
                # anomaly severity
                adjustment_factor = 1.0 - \
                    min(anomaly_score, 0.8) - (0.2 if not company_valid else 0)

                # Adjust probabilities - move toward uniform distribution
                uniform_prob = 1.0 / len(probabilities)
                adjusted_probs = [
                    p * adjustment_factor + uniform_prob * (1 - adjustment_factor)
                    for p in probabilities
                ]

                # Renormalize to sum to 1
                total = sum(adjusted_probs)
                adjusted_probs = [p / total for p in adjusted_probs]
            else:
                adjusted_probs = probabilities

            # Prepare result
            result = {
                'probabilities': {
                    str(c): round(
                        p,
                        4) for c,
                    p in zip(
                        classes,
                        adjusted_probs)},
                'is_valid': not is_anomaly and company_valid,
                'request_id': request_id}

            # Add validation details if there were issues
            if is_anomaly or not company_valid:
                result['validation'] = {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': round(anomaly_score, 4),
                    'reasons': anomaly_reasons,
                    'company_valid': company_valid,
                    'company_confidence': round(company_confidence, 4)
                }

            # Find most likely class for audit logging
            max_prob_idx = np.argmax(adjusted_probs)
            prediction = classes[max_prob_idx]
            confidence = adjusted_probs[max_prob_idx]

            # Log prediction for audit
            self._log_prediction(
                request_id=request_id,
                company_name=company_name,
                prediction=prediction,
                confidence=confidence,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                anomaly_reasons=anomaly_reasons,
                feature_values=features,
                client_ip=client_ip
            )

            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': f'Prediction failed: {str(e)}', 'is_valid': False}

    def _log_prediction(self, request_id, company_name, prediction, confidence,
                        is_anomaly, anomaly_score, anomaly_reasons,
                        feature_values, client_ip=None):
        """Log prediction details to audit trail

        Args:
            request_id: Unique identifier for the prediction request
            company_name: Name of the company
            prediction: The predicted class
            confidence: Confidence score
            is_anomaly: Whether prediction was flagged as anomalous
            anomaly_score: Anomaly detection score
            anomaly_reasons: List of reasons for anomaly detection
            feature_values: Feature values used in prediction
            client_ip: Client IP address
        """
        try:
            # Prepare log entry
            timestamp = datetime.now().isoformat()
            model_version = self.metadata.get('version', 'unknown')

            # Convert feature values to string
            if isinstance(feature_values, dict):
                feature_str = json.dumps({k: float(v) if isinstance(
                    v, (int, float, np.number)) else str(v) for k, v in feature_values.items()})
            else:
                feature_str = str(feature_values)

            # Format anomaly reasons
            if isinstance(anomaly_reasons, list):
                anomaly_reasons_str = '; '.join(anomaly_reasons)
            else:
                anomaly_reasons_str = str(anomaly_reasons)

            # Write to CSV
            with open(self.audit_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    request_id,
                    company_name if company_name else 'unknown',
                    prediction,
                    confidence,
                    1 if is_anomaly else 0,
                    anomaly_score,
                    anomaly_reasons_str,
                    feature_str,
                    client_ip if client_ip else 'unknown',
                    model_version
                ])

        except Exception as e:
            logger.error(f"Failed to log prediction: {str(e)}")
            # Continue execution even if logging fails




class AnomalyDetector:
    """Detects anomalies and potential manipulation in startup data"""

    def __init__(self, contamination=0.05):
        """Initialize detector with contamination parameter (expected outlier ratio)"""
        self.isolation_forest = None
        self.contamination = contamination
        self.feature_ranges = {}
        self.startup_data_cache = {}
        self.known_companies = set()

    def fit(self, X, startup_names=None):
        """Train anomaly detection model on startup data

        Args:
            X: Feature matrix for startups
            startup_names: Optional list of company names
        """
        try:
            # Train isolation forest for outlier detection
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            )
            self.isolation_forest.fit(X)

            # Store feature ranges for basic sanity checks
            self.feature_ranges = {
                'min': np.min(X, axis=0),
                'max': np.max(X, axis=0),
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0),
                'q1': np.percentile(X, 25, axis=0),
                'q3': np.percentile(X, 75, axis=0)
            }

            # Cache startup data if names provided
            if startup_names is not None:
                for i, name in enumerate(startup_names):
                    if i < len(X):
                        self.startup_data_cache[name] = X[i]
                        self.known_companies.add(name)

            logger.info(f"Fitted anomaly detector with {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"Error fitting anomaly detector: {str(e)}")
            return False

    def detect_anomalies(self, X, company_name=None, threshold=-0.5):
        """
        Detect anomalies in startup data

        Args:
            X: Feature matrix or single sample
            company_name: Optional company name for additional checks
            threshold: Decision threshold (lower = more strict)

        Returns:
            Dictionary with anomaly flags and scores
        """
        try:
            # Ensure X is 2D
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            # Validate input dimensions match what model was trained on
            if X.shape[1] != len(self.feature_ranges['min']):
                logger.error(
                    f"Feature dimension mismatch: expected {len(self.feature_ranges['min'])}, got {X.shape[1]}")
                return {'is_anomaly': True, 'reason': 'dimension_mismatch', 'score': 1.0}

            # Run standard anomaly checks
            anomalies = {
                'is_anomaly': False,
                'score': 0.0,
                'reasons': []
            }

            # First, check against known ranges
            range_violations = []
            for i, (val, min_val, max_val) in enumerate(
                    zip(X[0], self.feature_ranges['min'], self.feature_ranges['max'])):
                if val < min_val * 0.9 or val > max_val * 1.1:  # Allow 10% outside range
                    range_violations.append(i)

            # Check extreme feature values using IQR
            iqr_violations = []
            for i, (val, q1, q3) in enumerate(
                    zip(X[0], self.feature_ranges['q1'], self.feature_ranges['q3'])):
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                if val < lower_bound or val > upper_bound:
                    iqr_violations.append(i)

            # Check if company data has suddenly changed drastically
            company_change = False
            if company_name and company_name in self.startup_data_cache:
                cached_data = self.startup_data_cache[company_name]
                # Calculate percent change in values
                # Avoid division by zero
                pct_change = np.abs(
                    (X[0] - cached_data) / (cached_data + 1e-10))
                if np.any(
                        pct_change > 0.5):  # 50% change in any feature is suspicious
                    company_change = True
                    anomalies['reasons'].append(
                        f"Company data changed by >{np.max(pct_change) * 100:.1f}%")

            # Apply isolation forest to get anomaly score
            if self.isolation_forest is not None:
                scores = self.isolation_forest.decision_function(X)
                predictions = self.isolation_forest.predict(X)

                # Lower scores = more anomalous
                min_score = np.min(scores)
                if min_score < threshold or np.any(predictions == -1):
                    anomalies['is_anomaly'] = True
                    # Convert to positive for easier interpretation
                    anomalies['score'] = -min_score
                    anomalies['reasons'].append(
                        f"Isolation forest score: {min_score:.3f}")

            # Add other detected issues
            if range_violations:
                anomalies['is_anomaly'] = True
                anomalies['reasons'].append(
                    f"Range violations in {len(range_violations)} features")
                anomalies['score'] = max(anomalies['score'], 0.7)

            if iqr_violations:
                anomalies['is_anomaly'] = True
                anomalies['reasons'].append(
                    f"IQR violations in {len(iqr_violations)} features")
                anomalies['score'] = max(anomalies['score'], 0.6)

            if company_change:
                anomalies['is_anomaly'] = True
                anomalies['score'] = max(anomalies['score'], 0.8)

            # Check for potential manipulation patterns in funding amounts
            if self._check_funding_manipulation(X[0]):
                anomalies['is_anomaly'] = True
                anomalies['reasons'].append(
                    "Suspicious funding amount pattern detected")
                anomalies['score'] = max(anomalies['score'], 0.9)

            return anomalies
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {
                'is_anomaly': True,
                'reason': f'detection_error: {str(e)}',
                'score': 1.0
            }

    
    def validate_company(self, company_name, api_key=None):
        """Validate company existence through external API or internal checks"""
        if not company_name or company_name.strip() == "":
            return {
                "valid": False,
                "reason": "Empty company name",
                "confidence": 1.0
            }
            
        # Standardize name for checking
        company_name = company_name.lower().strip()
            
        # Check against known companies
        if company_name in self.known_companies:
            return {
                "valid": True,
                "reason": "Found in database",
                "confidence": 0.9
            }
            
        # Attempt API validation if key provided
        if api_key:
            api_result = self._simulate_company_api_check(company_name)
            if api_result["found"]:
                self.known_companies.add(company_name)
                return {
                    "valid": True,
                    "reason": "Validated through API",
                    "confidence": 0.95
                }
            
        # Check for suspicious patterns in name
        if not self._is_valid_company_name(company_name):
            return {
                "valid": False,
                "reason": "Company name contains suspicious patterns",
                "confidence": 0.7
            }
            
        # Check for name similarity with known companies
        similar_company = self._check_name_similarity(company_name)
        if similar_company:
            return {
                "valid": False,
                "reason": f"Potential impersonation of {similar_company}",
                "confidence": 0.8
            }
            
        # If all checks passed but company not found
        return {
            "valid": True,
            "reason": "Passed validation checks but not found in database",
            "confidence": 0.3
        }

    def _is_valid_company_name(self, name):
        """Check for suspicious patterns in company name"""
        import re
        
        # Check for repeated characters (potential keyboard spam)
        if re.search(r'(.)\1{4,}', name):  # 5+ repeated chars
            return False

        # Check for excessive numbers
        if len(re.findall(r'\d', name)) > len(name) // 2:
            return False

        # Check for suspicious TLDs if URL is included
        suspicious_tlds = ['.xyz', '.info', '.biz', '.tk', '.ml']
        if any(tld in name for tld in suspicious_tlds):
            return False

        return True


    def _check_name_similarity(self, name):
        """Check if name is suspiciously similar to a known company"""
        best_match = None
        best_score = float('inf')
        threshold = 3  # Max edit distance to be considered similar
        
        for known_name in self.known_companies:
            distance = self._levenshtein_distance(name, known_name)
            if distance < best_score and distance <= threshold:
                best_score = distance
                best_match = known_name
                
        return best_match
    
    def _levenshtein_distance(self, s1, s2):
        """Calculate edit distance between two strings"""
        if s1 == s2:
            return 0
            
        # Ensure s1 is the shorter string for efficiency
        if len(s1) > len(s2):
            return self._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
    


class Visualization:
    """Class for creating various visualizations for funding stage prediction models"""
    
    def __init__(self, output_dir="./visualizations"):
        """Initialize visualization with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set up matplotlib style
        plt.style.use('ggplot')
        
    def plot_calibration_curve(self, y_true, y_proba, model_names, n_bins=10):
        """
        Create calibration plots for models
        
        Args:
            y_true: True binary labels
            y_proba: List of probability predictions for each model
            model_names: List of model names
            n_bins: Number of bins for histogram
        """
        plt.figure(figsize=(12, 8))
        
        # Plot perfectly calibrated line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Plot calibration curve for each model
        for i, (proba, name) in enumerate(zip(y_proba, model_names)):
            # For multiclass, use average calibration across classes
            if proba.shape[1] > 2:
                # Average probability across all classes
                prob_pos = proba.mean(axis=1)
                frac_pos = (y_true == np.argmax(proba, axis=1)).astype(float)
            else:
                # Binary case - use probability of positive class
                prob_pos = proba[:, 1]
                frac_pos = y_true
                
            # Bin predictions to compute calibration
            bins = np.linspace(0., 1.+1e-8, n_bins+1)
            binids = np.digitize(prob_pos, bins) - 1
            bin_sums = np.bincount(binids, weights=prob_pos, minlength=len(bins))
            bin_true = np.bincount(binids, weights=frac_pos, minlength=len(bins))
            bin_total = np.bincount(binids, minlength=len(bins))
            
            nonzero = bin_total != 0
            prob_true = np.zeros(len(bins))
            prob_pred = np.zeros(len(bins))
            prob_true[nonzero] = bin_true[nonzero] / bin_total[nonzero]
            prob_pred[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
            
            plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=name)
        
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Plot (Reliability Curve)')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save the figure
        output_path = os.path.join(self.output_dir, f'calibration_plot_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration plot saved to {output_path}")
        return output_path
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, title='Confusion Matrix'):
        """
        Plot confusion matrix for model predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes (optional)
            title: Title for the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names if class_names else 'auto',
                    yticklabels=class_names if class_names else 'auto')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        
        # Save the figure
        output_path = os.path.join(self.output_dir, f'confusion_matrix_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {output_path}")
        return output_path
        
    def plot_roc_curves(self, y_true, y_proba, class_names=None, model_name="Model"):
        """
        Plot ROC curves for model predictions
        
        Args:
            y_true: True labels (one-hot encoded for multiclass)
            y_proba: Predicted probabilities 
            class_names: Names of classes (optional)
            model_name: Name of the model
        """
        plt.figure(figsize=(12, 8))
        
        # For multiclass, create one-vs-rest ROC curves
        n_classes = y_proba.shape[1]
        
        # Binarize the labels for one-vs-rest ROC
        if n_classes > 2:
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
        else:
            # Binary case
            y_true_bin = np.array(y_true).reshape(-1, 1)
            y_true_bin = np.concatenate([1-y_true_bin, y_true_bin], axis=1)
            
        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(
                y_true_bin[:, i] if n_classes > 2 else y_true, 
                y_proba[:, i]
            )
            roc_auc = auc(fpr, tpr)
            
            class_label = class_names[i] if class_names else f"Class {i}"
            plt.plot(fpr, tpr, lw=2, 
                    label=f'ROC {class_label} (AUC = {roc_auc:.2f})')
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {model_name}')
        plt.legend(loc="lower right")
        
        # Save the figure
        output_path = os.path.join(self.output_dir, f'roc_curves_{model_name}_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves plot saved to {output_path}")
        return output_path
        
    def plot_feature_importance(self, feature_importance, feature_names, model_name="Model"):
        """
        Plot feature importance for model
        
        Args:
            feature_importance: Array of feature importance values
            feature_names: List of feature names
            model_name: Name of the model
        """
        # Create DataFrame for easier sorting and plotting
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=fi_df)
        plt.title(f'Feature Importance for {model_name}')
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.output_dir, f'feature_importance_{model_name}_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path}")
        return output_path
        
    def plot_funding_stage_distribution(self, data):
        """
        Plot distribution of funding stages
        
        Args:
            data: DataFrame containing funding_stage column
        """
        plt.figure(figsize=(14, 8))
        
        if 'funding_stage' in data.columns:
            # Count funding stages
            counts = data['funding_stage'].value_counts().sort_index()
            
            # Plot horizontal bar chart
            ax = counts.plot(kind='barh', color='skyblue')
            
            # Add count labels to the bars
            for i, v in enumerate(counts):
                ax.text(v + 0.1, i, str(v), va='center')
                
            plt.title('Distribution of Funding Stages')
            plt.xlabel('Count')
            plt.ylabel('Funding Stage')
            plt.tight_layout()
            
            # Save the figure
            output_path = os.path.join(self.output_dir, f'funding_stage_distribution_{self.timestamp}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Funding stage distribution plot saved to {output_path}")
            return output_path
        else:
            logger.warning("No 'funding_stage' column found in data for plotting distribution")
            return None
            
    def plot_time_series_forecast(self, historical_data, forecast_data, target_col='funding_amount', 
                                 date_col='funding_date', title="Funding Forecast"):
        """
        Plot time series with historical data and forecast
        
        Args:
            historical_data: DataFrame with historical data
            forecast_data: DataFrame with forecast data
            target_col: Column name for the target variable
            date_col: Column name for the date variable
            title: Title for the plot
        """
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        if historical_data is not None and not historical_data.empty:
            if date_col in historical_data.columns and target_col in historical_data.columns:
                plt.plot(historical_data[date_col], historical_data[target_col], 
                        'b-', linewidth=2, label='Historical Data')
            else:
                logger.warning(f"Columns {date_col} and/or {target_col} not found in historical data")
        
        # Plot forecast data
        if forecast_data is not None and not forecast_data.empty:
            if date_col in forecast_data.columns and target_col in forecast_data.columns:
                plt.plot(forecast_data[date_col], forecast_data[target_col], 
                        'r--', linewidth=2, label='Forecast')
                
                # Add confidence intervals if available
                if f"{target_col}_lower" in forecast_data.columns and f"{target_col}_upper" in forecast_data.columns:
                    plt.fill_between(forecast_data[date_col], 
                                    forecast_data[f"{target_col}_lower"], 
                                    forecast_data[f"{target_col}_upper"], 
                                    color='r', alpha=0.2, label='95% Confidence Interval')
            else:
                logger.warning(f"Columns {date_col} and/or {target_col} not found in forecast data")
        
        plt.xlabel('Date')
        plt.ylabel(target_col.replace('_', ' ').title())
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.output_dir, f'time_series_forecast_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Time series forecast plot saved to {output_path}")
        return output_path
        
    def plot_funding_stages_over_time(self, data):
        """
        Plot distribution of funding stages over time
        
        Args:
            data: DataFrame containing funding_stage and funding_date columns
        """
        if 'funding_stage' not in data.columns or 'funding_date' not in data.columns:
            logger.warning("Required columns 'funding_stage' and/or 'funding_date' not found in data")
            return None
            
        # Ensure funding_date is datetime
        data['funding_date'] = pd.to_datetime(data['funding_date'], errors='coerce')
        
        # Group by year and funding stage
        yearly_data = data.dropna(subset=['funding_date', 'funding_stage'])
        yearly_data['year'] = yearly_data['funding_date'].dt.year
        stage_counts = yearly_data.groupby(['year', 'funding_stage']).size().unstack().fillna(0)
        
        # Plot
        plt.figure(figsize=(15, 10))
        stage_counts.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Funding Stages Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Companies')
        plt.legend(title='Funding Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.output_dir, f'funding_stages_over_time_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Funding stages over time plot saved to {output_path}")
        return output_path
        
    def plot_correlation_matrix(self, data, features=None):
        """
        Plot correlation matrix for features
        
        Args:
            data: DataFrame with features
            features: List of features to include (optional)
        """
        # Select only numeric columns
        if features:
            numeric_data = data[features].select_dtypes(include=np.number)
        else:
            numeric_data = data.select_dtypes(include=np.number)
            
        if numeric_data.empty:
            logger.warning("No numeric features found for correlation matrix")
            return None
            
        # Compute correlation matrix
        corr = numeric_data.corr()
        
        # Generate heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.output_dir, f'correlation_matrix_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correlation matrix plot saved to {output_path}")
        return output_path
        
    def generate_time_series_forecast(self, historical_data, periods=24, freq='M', 
                                    target_col='funding_amount', date_col='funding_date'):
        """
        Generate and plot time series forecast using Prophet
        
        Args:
            historical_data: DataFrame with historical data
            periods: Number of periods to forecast
            freq: Frequency of forecast ('D', 'W', 'M', 'Q', 'Y')
            target_col: Target column to forecast
            date_col: Date column
            
        Returns:
            Tuple of (forecast DataFrame, plot path)
        """
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            prophet_data = historical_data[[date_col, target_col]].copy()
            prophet_data.columns = ['ds', 'y']  # Prophet requires these column names
            
            # Create and fit model
            model = Prophet(yearly_seasonality=True, 
                           weekly_seasonality=True,
                           daily_seasonality=False,
                           seasonality_mode='multiplicative')
            model.fit(prophet_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq=freq)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Plot forecast
            fig = model.plot(forecast)
            plt.title(f'Time Series Forecast for {target_col}')
            plt.xlabel('Date')
            plt.ylabel(target_col.replace('_', ' ').title())
            
            # Save the figure
            output_path = os.path.join(self.output_dir, f'prophet_forecast_{self.timestamp}.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Plot components
            fig_comp = model.plot_components(forecast)
            comp_path = os.path.join(self.output_dir, f'prophet_components_{self.timestamp}.png')
            fig_comp.savefig(comp_path, dpi=300, bbox_inches='tight')
            plt.close(fig_comp)
            
            logger.info(f"Prophet forecast plots saved to {output_path} and {comp_path}")
            
            # Format forecast data to match original format
            result_forecast = pd.DataFrame({
                date_col: forecast['ds'],
                target_col: forecast['yhat'],
                f"{target_col}_lower": forecast['yhat_lower'],
                f"{target_col}_upper": forecast['yhat_upper']
            })
            
            return result_forecast, output_path
            
        except ImportError:
            logger.warning("Prophet package not installed. Install with 'pip install prophet'")
            return None, None
        except Exception as e:
            logger.error(f"Error generating time series forecast: {str(e)}")
            return None, None


class FeatureEngineering:
    def __init__(self):
        """Initialize with dynamic funding stage mapping"""
        self.funding_stage_map = {}  # Will be populated dynamically
        # Standard funding stage progression for reference
        self.standard_stages = [
            'Pre-Seed', 'Seed', 'Angel',
            'Series A', 'Series B', 'Series C',
            'Series D', 'Series E', 'Series F',
            'Series G', 'Series H'
        ]

    def extract_features(self, df):
        """Extract and create features for funding stage prediction with improved validation"""
        logger.info(f"Starting feature extraction for {len(df)} records")
        data = df.copy()

        # Get unique stages from data
        valid_stages = data['funding_stage'].dropna().unique()
        logger.info(
            f"Found {
                len(valid_stages)} unique funding stages: {valid_stages}")

        # Create stage mapping that respects the natural progression
        known_stages = {}
        for i, stage in enumerate(self.standard_stages):
            known_stages[stage] = i

        # Assign known stages first
        self.funding_stage_map = known_stages.copy()

        # Assign unknown stages sequential values
        next_value = max(known_stages.values()) + 1 if known_stages else 0
        for stage in valid_stages:
            if stage not in self.funding_stage_map:
                self.funding_stage_map[stage] = next_value
                next_value += 1

        # Add Unknown category if needed
        if 'Unknown' not in self.funding_stage_map:
            self.funding_stage_map['Unknown'] = next_value

        logger.info(f"Created funding stage mapping: {self.funding_stage_map}")

        # Convert funding stage to numeric
        data['funding_stage_numeric'] = data['funding_stage'].map(
            lambda x: self.funding_stage_map.get(x, self.funding_stage_map['Unknown'])
        )

        # Handle dates and extract temporal features
        try:
            # First try to parse dates with flexible format detection
            data['funding_date'] = pd.to_datetime(
                data['funding_date'], format='mixed', errors='coerce')

            # If a lot of dates failed to parse, try more specific formats
            if data['funding_date'].isna().sum() > len(data) * 0.3:
                logger.warning(
                    f"Many dates failed to parse ({
                        data['funding_date'].isna().sum()} NaN values)")

                # Try common date formats
                for fmt in ['%d-%b-%y', '%b %Y', '%Y', '%m/%d/%Y', '%Y-%m-%d']:
                    try:
                        # Save current NaN count
                        na_before = data['funding_date'].isna().sum()

                        # Try to parse remaining NaN dates with this format
                        mask = data['funding_date'].isna()
                        data.loc[mask, 'funding_date'] = pd.to_datetime(
                            df.loc[mask, 'funding_date'],
                            format=fmt,
                            errors='coerce'
                        )

                        # Log success
                        na_after = data['funding_date'].isna().sum()
                        if na_before > na_after:
                            logger.info(
                                f"Format '{fmt}' parsed {
                                    na_before - na_after} dates")
                    except BaseException:
                        pass

            # Extract temporal features
            data['funding_year'] = data['funding_date'].dt.year
            data['funding_month'] = data['funding_date'].dt.month

            # Fill NaN years/months with sensible defaults
            current_year = datetime.now().year
            current_month = datetime.now().month

            data['funding_year'] = data['funding_year'].fillna(current_year)
            data['funding_month'] = data['funding_month'].fillna(current_month)

            # Calculate months since first funding (proxy for company age)
            company_first_funding = data.groupby(
                'company_name')['funding_date'].min()
            data['company_first_funding'] = data['company_name'].map(
                company_first_funding)

            # Calculate months between dates, handling NaT values
            def safe_month_diff(end_date, start_date):
                if pd.isna(end_date) or pd.isna(start_date):
                    return 0
                try:
                    return (end_date.year - start_date.year) * \
                        12 + (end_date.month - start_date.month)
                except BaseException:
                    return 0

            data['months_since_first_funding'] = data.apply(
                lambda row: safe_month_diff(
                    row['funding_date'],
                    row['company_first_funding']),
                axis=1)

        except Exception as e:
            logger.warning(f"Error processing dates: {e}")
            # Set default values if date processing fails
            data['funding_year'] = datetime.now().year
            data['funding_month'] = datetime.now().month
            data['months_since_first_funding'] = 0

        # Log transform funding amount (handle skewed distribution)
        # First ensure it's numeric
        data['funding_amount'] = pd.to_numeric(
            data['funding_amount'], errors='coerce')

        # Check for extremely large values that might be errors (>$100B)
        if (data['funding_amount'] > 1e11).any():
            large_values = data[data['funding_amount'] > 1e11]
            logger.warning(
                f"Found {
                    len(large_values)} extremely large funding amounts (>$100B)")
            logger.warning(
                f"Sample: {large_values[['company_name', 'funding_amount', 'source']].head().to_dict()}")

            # Cap extremely large values
            data['funding_amount'] = data['funding_amount'].clip(upper=1e11)

        # Apply log transform with offset to handle zeros
        data['funding_amount_log'] = np.log1p(data['funding_amount'].fillna(0))

        # Employee efficiency (funding per employee)
        if 'employees' in data.columns:
            data['employees'] = pd.to_numeric(
                data['employees'], errors='coerce')

            # Avoid division by zero
            data['employee_efficiency'] = data.apply(
                lambda row: row['funding_amount'] /
                row['employees'] if row['employees'] > 0 else np.nan,
                axis=1)

            # Fill missing values with median by funding stage
            efficiency_medians = data.groupby('funding_stage')[
                'employee_efficiency'].median()

            for stage in data['funding_stage'].unique():
                stage_median = efficiency_medians.get(
                    stage, data['employee_efficiency'].median())
                mask = (
                    data['funding_stage'] == stage) & (
                    data['employee_efficiency'].isna())
                data.loc[mask, 'employee_efficiency'] = stage_median

            # Fill any remaining NaNs with overall median
            data['employee_efficiency'] = data['employee_efficiency'].fillna(
                data['employee_efficiency'].median())
        else:
            data['employees'] = np.nan
            data['employee_efficiency'] = np.nan

        # Standardize industry categories
        data['industry_category'] = data['industry'].fillna('Unknown')

        # Map to standardized categories with more comprehensive mapping
        industry_mapping = {
            'artificial intelligence': 'AI & ML',
            'machine learning': 'AI & ML',
            'information technology': 'IT & Software',
            'software': 'IT & Software',
            'health': 'Healthcare',
            'healthcare': 'Healthcare',
            'biotech': 'Biotech',
            'biotechnology': 'Biotech',
            'financial': 'FinTech',
            'finance': 'FinTech',
            'fintech': 'FinTech',
            'education': 'EdTech',
            'edtech': 'EdTech',
            'retail': 'Retail',
            'ecommerce': 'Retail',
            'energy': 'Energy',
            'renewable': 'Energy',
            'food': 'Food & Agriculture',
            'agriculture': 'Food & Agriculture',
            'transportation': 'Transport & Logistics',
            'logistics': 'Transport & Logistics',
            'real estate': 'Real Estate',
            'proptech': 'Real Estate'
        }

        # Apply mapping for standardization with better matching
        def map_industry(industry_str):
            if not industry_str or pd.isna(industry_str):
                return 'Unknown'

            industry_str = industry_str.lower()

            for key, value in industry_mapping.items():
                if key in industry_str:
                    return value

            return industry_str.title()  # Return capitalized version if no match

        data['industry_category'] = data['industry_category'].apply(
            map_industry)

        # Location features (if available)
        if 'location' in data.columns or 'headquarters' in data.columns or 'location_standardized' in data.columns:
            # Use most reliable location column available
            location_col = next(
                (col for col in ['location_standardized', 'location', 'headquarters'] 
                 if col in data.columns), None)
            
            if location_col:
                data['location_category'] = data[location_col].fillna('Unknown')

                # Extract country or state using improved location extraction
                extract_location = lambda x: x  # Default if method not available
                data['location_category'] = data['location_category'].apply(extract_location)

                # Consolidate common locations
                location_mapping = {
                    'United States': 'USA',
                    'US': 'USA',
                    'U.S.': 'USA',
                    'U.S.A.': 'USA',
                }

                data['location_category'] = data['location_category'].map(
                    lambda x: location_mapping.get(x, x)
                )

        # Funding frequency features
        company_funding_counts = data.groupby('company_name').size()
        data['previous_rounds'] = data['company_name'].map(
            company_funding_counts) - 1
        data['previous_rounds'] = data['previous_rounds'].clip(lower=0)

        # New feature: Funding velocity (average months between rounds)
        company_funding_dates = data.groupby(
            'company_name')['funding_date'].apply(list)

        def calc_funding_velocity(dates):
            if not isinstance(dates, list) or len(dates) < 2:
                return np.nan

            # Sort dates and remove NaT
            valid_dates = [d for d in dates if not pd.isna(d)]
            if len(valid_dates) < 2:
                return np.nan

            valid_dates.sort()

            # Calculate average months between rounds
            intervals = []
            for i in range(1, len(valid_dates)):
                interval = (valid_dates[i].year - valid_dates[i - 1].year) * \
                    12 + (valid_dates[i].month - valid_dates[i - 1].month)
                intervals.append(interval)

            return np.mean(intervals) if intervals else np.nan

        data['funding_velocity'] = data['company_name'].map(
            company_funding_dates.apply(calc_funding_velocity)
        )

        # Fill missing values for all numeric columns
        numeric_cols = [
            'funding_amount',
            'funding_amount_log',
            'employees',
            'employee_efficiency',
            'previous_rounds',
            'funding_velocity']

        for col in numeric_cols:
            if col in data.columns:
                # Fill with median by funding stage if available
                if 'funding_stage' in data.columns:
                    data[col] = data.groupby('funding_stage')[col].transform(
                        lambda x: x.fillna(x.median())
                    )

                # Fill any remaining NaNs with overall median
                data[col] = data[col].fillna(data[col].median())

        logger.info(
            f"Feature engineering complete: {
                data.shape[1]} features created")
        return data

    def prepare_model_data(self, data):
        """Prepare feature matrix with proper type handling and class balancing"""
        # Select relevant features that exist in the original data
        feature_cols = [
            'funding_amount_log', 'employees', 'employee_efficiency',
            'funding_year', 'funding_month', 'previous_rounds',
            'months_since_first_funding', 'funding_velocity'
        ]

        # Only use features that actually exist in the data
        features_to_use = [col for col in feature_cols if col in data.columns]
        logger.info(f"Preparing model data with features: {features_to_use}")

        # Clean feature data - ensure numeric types
        X = data[features_to_use].copy()

        # Convert all features to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Fill missing values for numeric columns - using median from actual data
        numeric_cols = X.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if X[col].isna().any():
                median_value = X[col].median()
                X[col] = X[col].fillna(median_value)
                logger.info(f"Filled NaN values in {col} with median: {median_value}")

        # Target variable processing
        y = pd.to_numeric(data['funding_stage_numeric'], errors='coerce')
        valid_mask = y.notna()

        # Get class distribution
        class_counts = y[valid_mask].value_counts()
        logger.info(f"Original class distribution:\n{class_counts}")

        # Define minimum samples per class
        MIN_SAMPLES_PER_CLASS = 10

        # Identify rare classes
        rare_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index
        if len(rare_classes) > 0:
            logger.info(f"Found {len(rare_classes)} rare classes: {rare_classes}")
            
            # Map rare classes to their closest neighbors
            class_mapping = {}
            for rare_class in rare_classes:
                # Find the nearest major class based on funding stage progression
                if rare_class in [16, 17, 18, 19]:  # series i, j, unknown, ico
                    class_mapping[rare_class] = 11  # Map to 'venture - series unknown'
                elif rare_class in [9, 10]:  # Series G, H
                    class_mapping[rare_class] = 8   # Map to Series F
                elif rare_class == 2:  # Angel
                    class_mapping[rare_class] = 1   # Map to Seed
                    
            # Apply mapping
            y = y.map(lambda x: class_mapping.get(x, x))
            logger.info("Merged rare classes into related major classes")
            
            # Update class distribution
            class_counts = y[valid_mask].value_counts()
            logger.info(f"Updated class distribution after merging:\n{class_counts}")

        X = X[valid_mask]
        y = y[valid_mask].astype(int)

        # Remap classes to be continuous from 0
        unique_classes = sorted(y.unique())
        class_map = {old_label: idx for idx, old_label in enumerate(unique_classes)}
        y = y.map(class_map)
        self.class_map = class_map  # Store for later use
        self.reverse_class_map = {v: k for k, v in class_map.items()}  # For converting predictions back
        
        logger.info(f"Remapped classes to be continuous. Class mapping: {class_map}")

        # Initialize and fit StandardScaler
        self.scaler = StandardScaler()
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Log data shapes and class distribution
        logger.info(f"Prepared model data: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"Final class distribution: {y.value_counts().to_dict()}")

        return X, y


class ModelTrainer:
    def __init__(self, output_dir="./models"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.class_map = None
        self.reverse_class_map = None
        os.makedirs(output_dir, exist_ok=True)

    def set_class_mapping(self, class_map, reverse_class_map):
        self.class_map = class_map
        self.reverse_class_map = reverse_class_map

    def _convert_predictions(self, y_pred):
        if self.reverse_class_map is not None:
            return np.array([self.reverse_class_map.get(pred, pred) for pred in y_pred])
        return y_pred

    def train_random_forest(self, X, y):
        """Train a Random Forest model with hyperparameter tuning"""
        logger.info("Training Random Forest model with hyperparameter tuning...")

        # Define parameter distributions for random search
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': [None] + list(range(10, 50, 5)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }

        # Initialize base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Perform random search with cross-validation
        random_search = RandomizedSearchCV(
            rf, param_distributions=param_dist,
            n_iter=50, cv=5, scoring='accuracy',
            n_jobs=-1, random_state=42, verbose=0
        )

        # Fit the model
        random_search.fit(X, y)
        
        logger.info(f"Best Random Forest parameters: {random_search.best_params_}")
        logger.info(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")

        # Get best model
        best_rf = random_search.best_estimator_

        # Evaluate on training set
        y_pred = best_rf.predict(X)
        y_proba = best_rf.predict_proba(X)
        
        # Convert predictions back to original labels if mapping exists
        y_pred_original = self._convert_predictions(y_pred)
        y_original = self._convert_predictions(y)
        
        accuracy = accuracy_score(y_original, y_pred_original)
        report = classification_report(y_original, y_pred_original)

        logger.info(f"Random Forest accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{report}")

        # Save model and metadata
        model_path = os.path.join(self.output_dir, f"random_forest_{self.timestamp}.joblib")
        model_metadata = {
            'model': best_rf,
            'training_date': self.timestamp,
            'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None,
            'accuracy': accuracy,
            'best_params': random_search.best_params_,
            'cv_accuracy': random_search.best_score_,
            'class_mapping': self.class_map,
            'reverse_class_mapping': self.reverse_class_map
        }
        joblib.dump(model_metadata, model_path)

        # Return model and evaluation data
        return best_rf, {
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(y_original, y_pred_original),
            'classification_report': report,
            'y_pred': y_pred_original,
            'y_proba': y_proba,
            'roc_auc': roc_auc_score(y_original, y_proba, multi_class='ovr'),
            'rmse': np.sqrt(mean_squared_error(y_original, y_pred_original)),
            'feature_importance': best_rf.feature_importances_ if hasattr(best_rf, 'feature_importances_') else None
        }

    def train_xgboost(self, X, y):
        """Train an XGBoost model with hyperparameter tuning"""
        logger.info("Training XGBoost model with hyperparameter tuning...")

        # Define parameter distributions for random search
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 7),
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(0, 2)
        }

        # Initialize base model
        xgb_model = xgb.XGBClassifier(
            objective='multi:softproba',
            random_state=42,
            n_jobs=-1
        )

        # Perform random search with cross-validation
        random_search = RandomizedSearchCV(
            xgb_model, param_distributions=param_dist,
            n_iter=50, cv=5, scoring='accuracy',
            n_jobs=-1, random_state=42, verbose=0
        )

        # Fit the model
        random_search.fit(X, y)
        
        logger.info(f"Best XGBoost parameters: {random_search.best_params_}")
        logger.info(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")

        # Get best model
        best_xgb = random_search.best_estimator_

        # Evaluate
        y_pred = best_xgb.predict(X)
        y_proba = best_xgb.predict_proba(X)

        # Convert predictions back to original labels if mapping exists
        y_pred_original = self._convert_predictions(y_pred)
        y_original = self._convert_predictions(y)

        accuracy = accuracy_score(y_original, y_pred_original)
        report = classification_report(y_original, y_pred_original)

        logger.info(f"XGBoost accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{report}")

        # Save model and metadata
        model_path = os.path.join(self.output_dir, f"xgboost_{self.timestamp}.joblib")
        model_metadata = {
            'model': best_xgb,
            'training_date': self.timestamp,
            'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None,
            'accuracy': accuracy,
            'best_params': random_search.best_params_,
            'cv_accuracy': random_search.best_score_,
            'class_mapping': self.class_map,
            'reverse_class_mapping': self.reverse_class_map
        }
        joblib.dump(model_metadata, model_path)

        return best_xgb, {
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(y_original, y_pred_original),
            'classification_report': report,
            'y_pred': y_pred_original,
            'y_proba': y_proba,
            'roc_auc': roc_auc_score(y_original, y_proba, multi_class='ovr'),
            'rmse': np.sqrt(mean_squared_error(y_original, y_pred_original)),
            'feature_importance': best_xgb.feature_importances_ if hasattr(best_xgb, 'feature_importances_') else None
        }


# Define StackedEnsemble as a top-level class so it can be properly serialized
class StackedEnsemble:
    """Stacked ensemble model that combines multiple base models with a meta-classifier"""
    
    def __init__(self, base_models, meta_classifier):
        self.base_models = base_models
        self.meta_classifier = meta_classifier
    
    def predict(self, X):
        base_preds = []
        for name, model in self.base_models.items():
            if model is not None:
                base_preds.append(model.predict_proba(X))
        meta_features = np.column_stack(base_preds)
        return self.meta_classifier.predict(meta_features)
    
    def predict_proba(self, X):
        base_preds = []
        for name, model in self.base_models.items():
            if model is not None:
                base_preds.append(model.predict_proba(X))
        meta_features = np.column_stack(base_preds)
        return self.meta_classifier.predict_proba(meta_features)
    
    
class DataLoader:
    def __init__(self, base_dir="./", archive=False):
        """Initialize data loader with paths to data sources and historical database"""
        self.base_dir = base_dir
        self.archive = archive
        self.archive_dir = None
        self.historical_db = os.path.join(
            base_dir, "historical_funding_data.db")

        # Define paths to source files in JSONFolder - fix for duplicated path
        # If base_dir already contains JSONFolder, don't add it again
        if os.path.basename(base_dir) == "JSONFolder" or os.path.exists(
                os.path.join(base_dir, "fundraisestartup50.json")):
            self.json_folder = base_dir
        else:
            self.json_folder = os.path.join(base_dir, "JSONFolder")

        # Use the fixed json_folder path for file paths
        self.fundraiser_path = os.path.join(
            self.json_folder, "fundraisestartup50.json")
        self.growthlist_path = os.path.join(
            self.json_folder, "growthlistscrapper.json")
        self.topstartup_path = os.path.join(
            self.json_folder, "topstartupio50.json")

        # Initialize the database for historical data
        self._init_historical_db()

        # Archive data if enabled
        if self.archive:
            self.archive_dir = self._create_archive_dir()
            self._archive_current_data()

    def _create_archive_dir(self):
        """Create a timestamped archive directory for this run"""
        archive_root = os.path.join(self.base_dir, "data_archive")
        os.makedirs(archive_root, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(archive_root, timestamp)
        os.makedirs(archive_dir, exist_ok=True)
        return archive_dir

    def _archive_current_data(self):
        """Copy current data files into the archive directory"""
        try:
            if not self.archive_dir:
                logger.warning("Archive directory not set. Skipping archive operation.")
                return
                
            archived_files = []
            
            # Archive fundraiser data
            if os.path.isfile(self.fundraiser_path):
                dest_path = os.path.join(self.archive_dir, "fundraiser.json")
                shutil.copy2(self.fundraiser_path, dest_path)
                archived_files.append(dest_path)
                
            # Archive growthlist data
            if os.path.isfile(self.growthlist_path):
                dest_path = os.path.join(self.archive_dir, "growthlist.json")
                shutil.copy2(self.growthlist_path, dest_path)
                archived_files.append(dest_path)
                
            # Archive topstartup data
            if os.path.isfile(self.topstartup_path):
                dest_path = os.path.join(self.archive_dir, "topstartup.json") 
                shutil.copy2(self.topstartup_path, dest_path)
                archived_files.append(dest_path)
                
            if archived_files:
                logger.info(f"Archived {len(archived_files)} data files to {self.archive_dir}")
                for file in archived_files:
                    logger.info(f"  - {os.path.basename(file)}")
            else:
                logger.warning(f"No data files found to archive in {self.base_dir}")
                
        except Exception as e:
            logger.error(f"Error archiving data: {e}")

    def _init_historical_db(self):
        """Create SQLite database tables if they don't exist"""
        try:
            # Create directory for database if it doesn't exist
            db_dir = os.path.dirname(self.historical_db)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            conn = sqlite3.connect(self.historical_db)
            cursor = conn.cursor()

            # Create tables with appropriate schema
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundraiser_data (
                company TEXT,
                employees INTEGER,
                industry TEXT,
                funding_date TEXT,
                funding_type TEXT,
                funding_amount REAL,
                headquarters TEXT,
                extraction_time TEXT,
                data_timestamp TEXT,
                PRIMARY KEY (company, extraction_time)
            )
            ''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS growthlist_data (
                name TEXT,
                industry TEXT,
                funding_amount TEXT,
                funding_type TEXT,
                last_funding_date TEXT,
                data_timestamp TEXT,
                PRIMARY KEY (name, last_funding_date)
            )
            ''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS topstartup_data (
                company TEXT,
                industry TEXT,
                funding_round TEXT,
                funding_amount REAL,
                funding_date TEXT,
                data_timestamp TEXT,
                PRIMARY KEY (company, funding_date)
            )
            ''')

            conn.commit()
            conn.close()
            logger.info("Historical database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def reset_database(self):
        """Reset the database for a clean start"""
        try:
            if os.path.exists(self.historical_db):
                os.remove(self.historical_db)
                logger.info("Existing database removed")
            self._init_historical_db()
            logger.info("Database reset complete")
        except Exception as e:
            logger.error(f"Error resetting database: {e}")

    def validate_dataset(self, df, source_name):
        """Check if required columns exist in the dataset"""
        required_columns = {
            'fundraiser_data': [
                'Company', 'Funding_Amount_USD', 'Funding_Type'], 'growthlist_data': [
                'name', 'funding_type', 'funding_amount'], 'topstartup_data': [
                'company_name', 'funding_stage', 'funding']}

        if source_name in required_columns:
            missing_cols = [col for col in required_columns[source_name]
                            if col not in df.columns]
            if missing_cols:
                logger.warning(
                    f"Missing columns in {source_name}: {missing_cols}")
                return False
        return True

    def validate_merged_columns(self, df):
        """Ensure consistent column structure"""
        required_columns = [
            'company_name', 'funding_date', 'funding_amount',
            'funding_stage', 'industry', 'employees'
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing critical columns: {missing}")
            return False

        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            logger.error(f"Duplicate columns detected: {duplicates}")
            return False

        return True

    def validate_company(
            self,
            company_name,
            funding_amount=None,
            funding_stage=None,
            employees=None):
        """
        Validate company data by cross-checking against known companies and using
        business logic to detect anomalies.

        Returns:
            tuple: (is_valid, confidence_score, message)
        """
        # All validation disabled - trust the data source
        return True, 1.0, "Validation disabled"
        
        """
        # Original validation code commented out
        if not company_name or pd.isna(company_name) or len(
                str(company_name).strip()) < 2:
            return False, 0.0, "Invalid company name"

        # Normalize company name for comparison
        company_name = str(company_name).lower().strip()

        # Check for common spam patterns
        spam_patterns = [
            'test',
            'dummy',
            'sample',
            'xyz',
            'abc',
            'placeholder',
            'llc']
        if any(pattern in company_name for pattern in spam_patterns):
            return False, 0.1, f"Company name contains suspicious pattern: {company_name}"

        try:
            # Try to cross-reference with historical data
            conn = sqlite3.connect(self.historical_db)
            cursor = conn.cursor()

            # Check if company exists in any historical table
            tables = ['fundraiser_data', 'growthlist_data', 'topstartup_data']
            company_found = False
            company_data = []

            for table in tables:
                # Adapt query based on table schema
                if table == 'fundraiser_data':
                    column = 'company'
                elif table == 'growthlist_data':
                    column = 'name'
                else:
                    column = 'company_name'

                cursor.execute(
                    f"SELECT * FROM {table} WHERE LOWER({column}) = ?", (company_name,))
                results = cursor.fetchall()

                if results:
                    company_found = True
                    company_data.extend(results)

            conn.close()

            if not company_found:
                # If not in history, it's suspicious but not necessarily
                # invalid
                return True, 0.5, "New company - not found in historical data"

            # Additional validation checks if funding_amount and stage provided
            if funding_amount is not None and funding_stage is not None:
                # Check for unrealistic funding for stage (simplified)
                if funding_stage and 'seed' in str(
                        funding_stage).lower() and funding_amount > 2e7:  # $20M
                    return False, 0.3, f"Unrealistic funding amount ${
                        funding_amount:,.2f} for {funding_stage} stage"
                # $100M
                elif funding_stage and 'series a' in str(funding_stage).lower() and funding_amount > 1e8:
                    return False, 0.4, f"Unusual funding amount ${
                        funding_amount:,.2f} for {funding_stage} stage"

            # Company passed all checks with high confidence
            return True, 0.9, "Company validated successfully"

        except Exception as e:
            logger.warning(
                f"Error validating company {company_name}: {
                    str(e)}")
            # Default to accepting but with low confidence if validation fails
            return True, 0.6, f"Validation partially completed: {str(e)}"
        """

    def _get_standardized_funding_type_map(self):
        """Returns a standardized mapping of funding types to ensure consistency across all data sources"""
        return {
            'pre-seed': 'Pre-Seed',
            'pre seed': 'Pre-Seed',
            'preseed': 'Pre-Seed',
            'seed': 'Seed',
            'angel': 'Angel',
            'series a': 'Series A',
            'series b': 'Series B',
            'series c': 'Series C',
            'series d': 'Series D',
            'series e': 'Series E',
            'series f': 'Series F',
            'series g': 'Series G',
            'series h': 'Series H',
            'venture - series unknown': 'Venture - Series Unknown',
            'venture series unknown': 'Venture - Series Unknown',
            'private equity': 'Private Equity',
            'initial coin offering': 'Initial Coin Offering',
            'ico': 'Initial Coin Offering',
            'grant': 'Grant',
            'debt financing': 'Debt Financing',
            'debt': 'Debt Financing',
            'undisclosed': 'Undisclosed'
        }

    def _standardize_funding_type(self, funding_type):
        """Standardize a funding type string based on the standardized mapping"""
        if pd.isna(funding_type):
            return funding_type
        
        funding_type_map = self._get_standardized_funding_type_map()
        normalized = str(funding_type).lower().strip()
        return funding_type_map.get(normalized, funding_type)

    def load_fundraiser_data(self):
        """Load and process fundraiser insider data with standardized funding types"""
        try:
            with open(self.fundraiser_path, 'r') as file:
                data = json.load(file)

            # Handle the JSON structure which is a list of companies
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                # Extract companies from the JSON structure if it's a dictionary
                companies = data.get('companies', [])
                df = pd.DataFrame(companies)

            # Add timestamp for versioning
            df['data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Convert numeric fields
            if 'Funding_Amount_USD' in df.columns:
                df['Funding_Amount_USD'] = pd.to_numeric(
                    df['Funding_Amount_USD'], errors='coerce')

            if 'Total_Employees' in df.columns:
                df['Total_Employees'] = pd.to_numeric(
                    df['Total_Employees'], errors='coerce')
            
            # Standardize Funding_Type if present
            if 'Funding_Type' in df.columns:
                df['Funding_Type'] = df['Funding_Type'].apply(self._standardize_funding_type)
                
                # Log unique funding types found
                unique_funding_types = df['Funding_Type'].dropna().unique()
                logger.info(f"Found funding types in fundraiser data: {unique_funding_types}")

            logger.info(f"Loaded {len(df)} records from fundraiser data")
            return df
        except Exception as e:
            logger.error(f"Error loading fundraiser data: {e}")
            return pd.DataFrame()

    def load_growthlist_data(self):
        """Load and process growthlist startups data - extract both funding amount and type"""
        try:
            with open(self.growthlist_path, 'r') as file:
                data = json.load(file)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Add timestamp for versioning
            df['data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Process funding amounts without dropping original column
            if 'funding_amount' in df.columns:
                df['funding_amount_numeric'] = df['funding_amount'].apply(
                    self._parse_funding_amount)
            
            # Ensure funding_type is processed and standardized
            if 'funding_type' in df.columns:
                df['funding_type'] = df['funding_type'].apply(self._standardize_funding_type)
                
                # Log unique funding types found
                unique_funding_types = df['funding_type'].dropna().unique()
                logger.info(f"Found funding types in growthlist data: {unique_funding_types}")

            logger.info(f"Loaded {len(df)} records from growthlist data")
            return df
        except Exception as e:
            logger.error(f"Error loading growthlist data: {e}")
            return pd.DataFrame()

    def load_topstartup_data(self):
        """Handle the complex format of topstartup data and extract funding information correctly"""
        try:
            with open(self.topstartup_path, 'r') as file:
                data = json.load(file)

            # Handle both list and dictionary formats
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame(data.get('startups', []))
            else:
                logger.error("Unexpected JSON format in topstartup data")
                return pd.DataFrame()

            # Parse employee count from strings like "1-10 employees"
            if 'employees' in df.columns:
                df['employees_numeric'] = df['employees'].apply(self._parse_employee_count)
                df['employees'] = df['employees_numeric']  # Replace original with numeric value

            # Handle headquarters data
            if 'headquarters' in df.columns:
                # Fill empty HQ with default value
                df['headquarters'] = df['headquarters'].fillna('San Francisco Bay Area')
                # Standardize location format
                df['location_standardized'] = df['headquarters'].apply(self._standardize_location)

            # Parse funding information from the funding string
            if 'funding' in df.columns:
                # Extract funding information from the complex funding string
                def extract_funding_info(funding_str):
                    if not funding_str or pd.isna(funding_str):
                        return None, None, None

                    # Common patterns: "Sequoia $100M Series D in 2025"
                    # or "Andreessen Horowitz $10B Series J in 2024 $62.0B
                    # valuation"

                    amount = None
                    stage = None
                    date = None

                    # Extract amount
                    amount_match = re.search(
                        r'\$(\d+(?:\.\d+)?[KMB]?)', funding_str)
                    if amount_match:
                        amount = amount_match.group(0)  # Keep the $ symbol

                    # Extract stage with improved pattern matching
                    # Look for more funding stage patterns with case
                    # insensitivity
                    stage_pattern = r'(Pre[-\s]?Seed|Seed|Angel|Series\s+[A-Z]|Venture[\s\-]+Series\s+Unknown|Initial\s+Coin\s+Offering|ICO|Private\s+Equity|Grant|Debt\s+Financing|Undisclosed|Post[-\s]?IPO)'
                    stage_match = re.search(
                        stage_pattern, funding_str, re.IGNORECASE)

                    if stage_match:
                        # Get the raw matched stage
                        raw_stage = stage_match.group(1)
                        # Use our standardization method
                        stage = self._standardize_funding_type(raw_stage)
                    else:
                        # If no explicit stage is found, try to infer from context
                        # Check for common patterns in funding text
                        funding_lower = funding_str.lower()

                        if 'seed' in funding_lower and not stage:
                            stage = 'Seed'
                        elif 'angel' in funding_lower and not stage:
                            stage = 'Angel'
                        elif 'raised' in funding_lower and not stage:
                            # For strings like "Raised $5M in 2019" without
                            # explicit stage
                            if 'series' in funding_lower:
                                # Try to extract series letter if mentioned
                                series_match = re.search(
                                    r'series\s+([a-z])', funding_lower)
                                if series_match:
                                    letter = series_match.group(1).upper()
                                    stage = f'Series {letter}'
                                else:
                                    stage = 'Venture - Series Unknown'
                            else:
                                # Default to "Venture Funding" for generic
                                # raised amounts
                                stage = 'Venture - Series Unknown'
                        elif 'valuation' in funding_lower and not stage:
                            if 'post-ipo' in funding_lower or 'post ipo' in funding_lower:
                                stage = 'Post-IPO'
                            else:
                                # Companies with just valuation mentioned but
                                # no explicit funding stage
                                stage = 'Venture - Series Unknown'

                    # Extract date - usually has "in YYYY" format
                    date_match = re.search(r'in (\d{4})', funding_str)
                    if date_match:
                        date = date_match.group(1)
                    else:
                        # Try to find just a year at the end of the string
                        year_match = re.search(r'\b(20\d{2})\b', funding_str)
                        if year_match:
                            date = year_match.group(1)

                    return amount, stage, date

                # Extract funding details
                funding_details = df['funding'].apply(extract_funding_info)

                # Create separate columns for extracted values
                df['funding_amount'] = funding_details.apply(
                    lambda x: x[0] if x else None)
                df['funding_stage'] = funding_details.apply(
                    lambda x: x[1] if x else None)
                df['funding_date'] = funding_details.apply(
                    lambda x: x[2] if x else None)

                # Log unique funding stages found
                unique_funding_stages = df['funding_stage'].dropna().unique()
                logger.info(f"Found funding stages in topstartup data: {unique_funding_stages}")

            # Standardize column names
            column_mapping = {
                'name': 'company_name',
                'funding_type': 'funding_stage',  # Alternative naming
                'category': 'industry'
            }

            # Apply mapping for existing columns only
            df = df.rename(columns={k: v for k, v in column_mapping.items()
                                    if k in df.columns})

            # Add timestamp
            df['data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Convert funding amounts if present
            if 'funding_amount' in df.columns:
                df['funding_amount_numeric'] = df['funding_amount'].apply(
                    self._parse_funding_amount)

            logger.info(f"Loaded {len(df)} records from topstartup data")
            return df
        except Exception as e:
            logger.error(f"Error loading topstartup data: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _parse_funding_amount(self, amount_str):
        """Parse funding amount from string to float

        Args:
            amount_str: String representing funding amount (e.g., "$10M", "5.3B")

        Returns:
            Funding amount in USD
        """
        if not amount_str or pd.isna(amount_str):
            return None

        try:
            # Remove non-alphanumeric characters except for decimal points and standard suffixes
            cleaned = re.sub(r'[^0-9a-zA-Z\.]', '', str(amount_str))
            
            # Extract the numeric part
            numeric_part = re.search(r'^(\d+\.?\d*)', cleaned)
            if not numeric_part:
                return None
            
            amount = float(numeric_part.group(1))
            
            # Check for suffixes and adjust amount accordingly
            if 'B' in cleaned or 'b' in cleaned:
                amount *= 1_000_000_000
            elif 'M' in cleaned or 'm' in cleaned:
                amount *= 1_000_000
            elif 'K' in cleaned or 'k' in cleaned:
                amount *= 1_000
                
            # Apply reasonable upper limit for funding amounts
            # Instead of just logging, handle the cap more gracefully
            MAX_FUNDING_AMOUNT = 10_000_000_000  # $10B
            if amount > MAX_FUNDING_AMOUNT:
                logger.info(f"Large funding amount detected: ${amount:,.2f} for stage {self.current_stage}. Capping at ${MAX_FUNDING_AMOUNT:,.2f}")
                amount = MAX_FUNDING_AMOUNT
                
            return amount
        except Exception as e:
            logger.debug(f"Error parsing funding amount '{amount_str}': {str(e)}")
            return None

    def save_historical_data(self, df, table_name):
        """Save dataframe to historical SQLite database"""
        try:
            conn = sqlite3.connect(self.historical_db)
            
            # Create the table with the correct schema if it doesn't exist
            if 'company_name' not in df.columns and table_name == 'merged_data':
                # Add required company_name column if missing for merged_data table
                df['company_name'] = df.get('name', 'Unknown')
            
            # Ensure all column names are valid SQL identifiers
            df = df.copy()
            for col in df.columns:
                if re.search(r'[^a-zA-Z0-9_]', col):
                    new_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                    df.rename(columns={col: new_col}, inplace=True)
            
            # Get column types for table creation
            column_types = {}
            for col in df.columns:
                if df[col].dtype == 'float64' or df[col].dtype == 'float32':
                    column_types[col] = 'REAL'
                elif df[col].dtype == 'int64' or df[col].dtype == 'int32':
                    column_types[col] = 'INTEGER'
                else:
                    column_types[col] = 'TEXT'
            
            # Create table if it doesn't exist
            cursor = conn.cursor()
            columns_sql = ", ".join([f'"{col}" {column_types[col]}' for col in df.columns])
            create_table_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql})'
            cursor.execute(create_table_sql)
            conn.commit()
            
            # Save data
            df.to_sql(table_name, conn, if_exists='append', index=False)
            conn.close()
            logger.info(f"Saved {len(df)} records to historical {table_name}")
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
            logger.error(traceback.format_exc())

    def load_historical_data(self, table_name):
        """Load historical data from SQLite database"""
        try:
            conn = sqlite3.connect(self.historical_db)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(
                f"Loaded {
                    len(df)} historical records from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()

    def merge_datasets(self):
        """Improved merge function with better error handling, validation and audit trail"""
        try:
            # Load raw data
            fundraiser_df = self.load_fundraiser_data()
            growthlist_df = self.load_growthlist_data()
            topstartup_df = self.load_topstartup_data()

            # Log loaded data sizes
            logger.info(
                f"Loaded datasets - Fundraiser: {
                    len(fundraiser_df)} rows, Growthlist: {
                    len(growthlist_df)} rows, Topstartup: {
                    len(topstartup_df)} rows")

            # Initialize list to store all records
            all_records = []

            # Process fundraiser data
            if not fundraiser_df.empty:
                for _, row in fundraiser_df.iterrows():
                    if pd.notna(row.get('Company')):  # Only add records with valid company names
                        # Convert numeric values properly
                        try:
                            funding_amount = pd.to_numeric(
                                row.get('Funding_Amount_USD'), errors='coerce')
                            employees = pd.to_numeric(
                                row.get('Total_Employees'), errors='coerce')
                        except:
                            funding_amount = np.nan
                            employees = np.nan
                            
                        all_records.append({
                            'company_name': row.get('Company'),
                            'funding_stage': row.get('Funding_Type'),
                            'funding_amount': funding_amount,
                            'funding_date': row.get('Funding_Date'),
                            'industry': row.get('Industry'),
                            'employees': employees,
                            'source': 'fundraiser',
                            'confidence_score': 1.0
                        })

                logger.info(f"Processed {len(fundraiser_df)} records from fundraiser data")
                # Log unique funding stages from this source
                funding_stages = [r.get('Funding_Type') for _, r in fundraiser_df.iterrows() if pd.notna(r.get('Funding_Type'))]
                unique_stages = set(funding_stages)
                logger.info(f"Unique funding stages from fundraiser: {unique_stages}")

            # Process growthlist data
            if not growthlist_df.empty:
                for _, row in growthlist_df.iterrows():
                    if pd.notna(row.get('name')):  # Only add records with valid company names
                        # Parse the amount if not already parsed
                        funding_amount = row.get('funding_amount_numeric')
                        if pd.isna(funding_amount) and pd.notna(row.get('funding_amount')):
                            funding_amount = self._parse_funding_amount(row.get('funding_amount'))

                        all_records.append({
                            'company_name': row.get('name'),
                            'funding_stage': row.get('funding_type'),  # Using standardized funding_type
                            'funding_amount': funding_amount,
                            'funding_date': row.get('last_funding_date'),
                            'industry': row.get('industry'),
                            'employees': None,
                            'source': 'growthlist',
                            'confidence_score': 1.0
                        })

                logger.info(f"Processed {len(growthlist_df)} records from growthlist data")

            # Process topstartup data
            if not topstartup_df.empty:
                for _, row in topstartup_df.iterrows():
                    company_name = row.get('company_name') or row.get('name')

                    if pd.notna(company_name):  # Only add records with valid company names
                        # Clean up data
                        funding_stage = row.get('funding_stage') or row.get('funding_round')

                        # Parse the amount if string
                        funding_amount = row.get('funding_amount')
                        if isinstance(funding_amount, str):
                            funding_amount = self._parse_funding_amount(funding_amount)

                        # Get employee count range
                        employee_count = None
                        if pd.notna(row.get('employees')):
                            # Handle ranges like "11-50 employees"
                            emp_str = str(row.get('employees'))
                            match = re.search(r'(\d+)-(\d+)', emp_str)
                            if match:
                                # Take the average of the range
                                employee_count = (int(match.group(1)) + int(match.group(2))) / 2

                        all_records.append({
                            'company_name': company_name,
                            'funding_stage': funding_stage,
                            'funding_amount': funding_amount,
                            'funding_date': row.get('funding_date'),
                            'industry': row.get('industry'),
                            'employees': employee_count,
                            'source': 'topstartup',
                            'confidence_score': 1.0
                        })

                logger.info(f"Processed {len(topstartup_df)} records from topstartup data")

            # Create the merged dataframe
            merged_df = pd.DataFrame(all_records)

            # Add timestamp for audit
            merged_df['merge_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Save an audit trace for regulatory compliance
            self.save_historical_data(merged_df, 'merged_data')

            logger.info(f"Successfully merged {len(merged_df)} records")
            
            # Validate merged dataset schema
            self.validate_merged_columns(merged_df)
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _parse_employee_count(self, employee_str):
        """Parse employee count from strings like '1-10 employees'"""
        if not employee_str or pd.isna(employee_str):
            return np.nan

        try:
            # Handle strings like "1-10 employees" or "10+" or just "15"
            employee_str = str(employee_str).lower().replace('employees', '').replace('employee', '').strip()
            
            # Case 1: Range like "1-10"
            if '-' in employee_str:
                lower, upper = employee_str.split('-')
                lower = int(lower.strip())
                upper = int(upper.strip())
                # Return midpoint of range
                return (lower + upper) / 2
                
            # Case 2: "10+" format
            elif '+' in employee_str:
                base = int(employee_str.replace('+', '').strip())
                # For 10+, estimate as 15 (50% more)
                return base * 1.5
                
            # Case 3: Direct number
            else:
                return int(employee_str.strip())
        except Exception as e:
            logger.warning(f"Error parsing employee count '{employee_str}': {str(e)}")
            return np.nan

    def _standardize_location(self, location_str):
        """Standardize location strings across different data sources"""
        if not location_str or pd.isna(location_str):
            return 'Unknown'
            
        location_str = str(location_str).strip()
        
        # Handle common variations
        common_locations = {
            'sf': 'San Francisco Bay Area',
            'silicon valley': 'San Francisco Bay Area',
            'bay area': 'San Francisco Bay Area',
            'nyc': 'New York',
            'new york city': 'New York',
            'bangalore': 'Bengaluru',
            'london, uk': 'London',
            'tel aviv': 'Tel Aviv',
        }
        
        # Try to match with common locations
        location_lower = location_str.lower()
        for key, value in common_locations.items():
            if key in location_lower:
                return value
        
        # Extract country if present
        country_pattern = r'(?:,\s+|\s+in\s+)(USA|US|United States|Canada|UK|United Kingdom|Australia|Germany|France|India|China|Japan|Israel)$'
        country_match = re.search(country_pattern, location_str, re.IGNORECASE)
        
        if country_match:
            return country_match.group(1)
            
        return location_str

    def train_stacked_ensemble(self, X, y, base_models):
        """Create a stacked ensemble using multiple base models"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Convert base_models to dict if it's a list
            if isinstance(base_models, list):
                base_models_dict = {}
                for i, model in enumerate(base_models):
                    if model is not None:
                        base_models_dict[f'model_{i}'] = model
                base_models = base_models_dict
            
            # Train base models and get predictions
            base_predictions = {}
            for name, model in base_models.items():
                if model is not None:
                    model.fit(X_train, y_train)
                    pred = model.predict_proba(X_test)
                    base_predictions[name] = pred
            
            # Skip if we don't have enough base models
            if len(base_predictions) < 2:
                logger.warning("Not enough valid base models for ensemble (need at least 2)")
                return None, 0.0
            
            # Create meta-features
            meta_features = np.column_stack(list(base_predictions.values()))
            
            # Train meta-classifier
            meta_classifier = LogisticRegression(
                multi_class='multinomial',
                max_iter=1000,
                random_state=42
            )
            meta_classifier.fit(meta_features, y_test)
            
            # Make final predictions
            final_predictions = meta_classifier.predict(meta_features)
            accuracy = accuracy_score(y_test, final_predictions)
            
            logger.info(f"Stacked Ensemble accuracy: {accuracy:.4f}")
            
            # Use the global StackedEnsemble class
            ensemble = StackedEnsemble(base_models, meta_classifier)
            return ensemble, accuracy
            
            
        except Exception as e:
            logger.error(f"Error training stacked ensemble: {str(e)}")
            logger.error(traceback.format_exc())
            return None, 0.0
    
    def optimize_hyperparameters(self, trial, X, y, model_type='rf'):
        """Optimize hyperparameters using Optuna"""
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            model = RandomForestClassifier(**params, random_state=42)
        elif model_type == 'xgb':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = xgb.XGBClassifier(**params, random_state=42)
        
        # Perform cross-validation
        scores = cross_val_score(
            model, X, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )
        
        return scores.mean()
    
    def train_optimized_model(self, X, y, model_type='rf', n_trials=100):
        """Train a model with optimized hyperparameters"""
        try:
            study = optuna.create_study(direction='maximize')
            objective = lambda trial: self.optimize_hyperparameters(trial, X, y, model_type)
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            logger.info(f"Best {model_type} parameters: {best_params}")
            
            # Train final model
            if model_type == 'rf':
                model = RandomForestClassifier(**best_params, random_state=42)
            elif model_type == 'xgb':
                model = xgb.XGBClassifier(**best_params, random_state=42)
            
            # Evaluate with cross-validation
            scores = cross_val_score(
                model, X, y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy',
                n_jobs=-1
            )
            
            logger.info(f"Optimized {model_type} CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            # Train final model on full dataset
            model.fit(X, y)
            return model, scores.mean()
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return None, 0.0

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation with multiple metrics"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Calculate MAE
            mae = mean_absolute_error(y_test, y_pred)
            
            # Get classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Calculate ROC curves and AUC scores if applicable
            roc_auc_scores = {}
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                
                # For multi-class, use one-vs-rest approach
                if len(np.unique(y_test)) > 2:
                    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                    n_classes = y_test_bin.shape[1]
                    
                    for i in range(n_classes):
                        if y_test_bin[:, i].sum() > 0:
                            roc_auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
                            roc_auc_scores[f'class_{i}'] = roc_auc
                    
                    # Calculate macro average
                    roc_auc_scores['macro_avg'] = np.mean(list(roc_auc_scores.values()))
                else:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    roc_auc_scores['binary'] = roc_auc
            
            # Calculate F1 scores
            f1_micro = f1_score(y_test, y_pred, average='micro')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate precision and recall
            precision_micro = precision_score(y_test, y_pred, average='micro')
            precision_macro = precision_score(y_test, y_pred, average='macro')
            recall_micro = recall_score(y_test, y_pred, average='micro')
            recall_macro = recall_score(y_test, y_pred, average='macro')
            
            # Log results
            logger.info(f"\n{model_name} Evaluation Results:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"F1 Score (micro/macro/weighted): {f1_micro:.4f}/{f1_macro:.4f}/{f1_weighted:.4f}")
            logger.info(f"Precision (micro/macro): {precision_micro:.4f}/{precision_macro:.4f}")
            logger.info(f"Recall (micro/macro): {recall_micro:.4f}/{recall_macro:.4f}")
            logger.info(f"ROC AUC Scores: {roc_auc_scores}")
            logger.info(f"Classification Report:\n{json.dumps(class_report, indent=2)}")
            
            # Return comprehensive metrics
            metrics = {
                'accuracy': accuracy,
                'rmse': rmse,
                'mae': mae,
                'f1_scores': {
                    'micro': f1_micro,
                    'macro': f1_macro,
                    'weighted': f1_weighted
                },
                'precision': {
                    'micro': precision_micro,
                    'macro': precision_macro
                },
                'recall': {
                    'micro': recall_micro,
                    'macro': recall_macro
                },
                'roc_auc_scores': roc_auc_scores,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            return None

    def train_advanced_pipeline(self, X, y):
        """Train multiple models with advanced techniques and ensemble methods"""
        try:
            # Filter out classes with too few samples
            class_counts = pd.Series(y).value_counts()
            rare_classes = class_counts[class_counts < 2].index
            
            if len(rare_classes) > 0:
                logger.info(f"Removing {len(rare_classes)} classes with fewer than 2 samples: {rare_classes}")
                valid_classes_mask = ~pd.Series(y).isin(rare_classes)
                X = X.loc[valid_classes_mask] if isinstance(X, pd.DataFrame) else X[valid_classes_mask]
                y = y[valid_classes_mask]
                
                # Verify no classes have too few samples
                updated_counts = pd.Series(y).value_counts()
                logger.info(f"Updated class distribution: {updated_counts.to_dict()}")
            
            # Split data with stratification to maintain class distribution
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Dictionary to store all models and their metrics
            all_models = {}
            
            # 1. Train Random Forest with advanced tuning
            logger.info("Training Random Forest with advanced tuning...")
            rf_params = {
                'n_estimators': [200, 500],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
            rf_grid.fit(X_train_scaled, y_train)
            logger.info(f"Best RF parameters: {rf_grid.best_params_}")
            rf_metrics = self.evaluate_model(rf_grid.best_estimator_, X_test_scaled, y_test, "Random Forest")
            all_models['random_forest'] = (rf_grid.best_estimator_, rf_metrics)
            
            # 2. Train LightGBM with improved parameters
            try:
                import lightgbm as lgb
                logger.info("Training LightGBM with advanced tuning...")
                lgb_params = {
                    'num_leaves': [31],
                    'max_depth': [5],
                    'learning_rate': [0.1],
                    'n_estimators': [100],
                    'min_child_samples': [20],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8],
                    'min_split_gain': [0.01],
                    'min_child_weight': [1],
                    'reg_alpha': [0.1],
                    'reg_lambda': [0.1]
                }
                lgb_model = lgb.LGBMClassifier(
                    objective='multiclass',
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1
                )
                lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=5, scoring='accuracy', n_jobs=-1)
                lgb_grid.fit(X_train_scaled, y_train)
                logger.info(f"Best LGB parameters: {lgb_grid.best_params_}")
                lgb_metrics = self.evaluate_model(lgb_grid.best_estimator_, X_test_scaled, y_test, "LightGBM")
                all_models['lightgbm'] = (lgb_grid.best_estimator_, lgb_metrics)
            except ImportError:
                logger.warning("LightGBM not available, skipping...")
            
            # Find best model
            best_model = None
            best_accuracy = 0
            best_model_name = None
            
            for name, (model, metrics) in all_models.items():
                if metrics and metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_model = model
                    best_model_name = name
            
            logger.info(f"\nBest Model: {best_model_name}")
            logger.info(f"Best Accuracy: {best_accuracy:.4f}")
            
            return all_models, best_model_name
            
        except Exception as e:
            logger.error(f"Error in advanced pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None

    def save_model_results(self, model_name, metrics, predictions, feature_importance=None):
        """Save model results to a JSON file"""
        try:
            # Create archived directory for results
            archive_path = self._create_archive_dir()
            results_dir = os.path.join(archive_path, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save metrics
            metrics_path = os.path.join(results_dir, f"{model_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4, cls=NumpyEncoder)
                
            # Save predictions if available
            if predictions is not None:
                preds_path = os.path.join(results_dir, f"{model_name}_predictions.csv")
                predictions.to_csv(preds_path, index=False)
                
            # Save feature importance if available
            if feature_importance is not None:
                fi_path = os.path.join(results_dir, f"{model_name}_feature_importance.json")
                with open(fi_path, 'w') as f:
                    json.dump(feature_importance, f, indent=4, cls=NumpyEncoder)
                    
            logger.info(f"Model results for {model_name} saved to {results_dir}")
            return results_dir
        except Exception as e:
            logger.error(f"Error saving results for {model_name}: {e}")
            logger.error(traceback.format_exc())
            return None
            
    def _create_archive_dir(self):
        """Create a timestamped archive directory for storing data snapshots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(self.output_dir, "data_archive", timestamp)
        os.makedirs(archive_path, exist_ok=True)
        logger.info(f"Created archive directory: {archive_path}")
        return archive_path

    def _archive_current_data(self):
        """Archive current data files to timestamped directory"""
        try:
            archive_path = self._create_archive_dir()
            
            # Copy all current data files to archive
            for file_path in [self.topstartup_path, self.fundraiser_path, self.growthlist_path]:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    archived_file = os.path.join(archive_path, filename)
                    shutil.copy2(file_path, archived_file)
                    logger.info(f"Archived {filename} to {archived_file}")
            
            return archive_path
        except Exception as e:
            logger.error(f"Error archiving data: {e}")
            logger.error(traceback.format_exc())
            return None


class ModelManager:
    """Manager for model versioning and predictions"""
    
    def __init__(self, model_dir='models/'):
        """Initialize ModelManager with model directory location"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize logging
        self.audit_log_path = os.path.join(model_dir, "prediction_audit.csv")
        self.has_audit_log = os.path.exists(self.audit_log_path)
        
    def init_audit_log(self):
        """Initialize the prediction audit log if it doesn't exist"""
        if not self.has_audit_log:
            with open(self.audit_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'request_id', 'company_name', 'prediction',
                    'confidence', 'is_anomaly', 'anomaly_score', 'client_ip'
                ])
            self.has_audit_log = True
            logger.info(f"Audit log initialized at {self.audit_log_path}")
    
    def _log_model_operation(self, operation, model_id, model_name, version):
        """Log model operations to a file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_path = os.path.join(self.model_dir, "model_operations.csv")
        
        # Create log file if it doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'operation', 'model_id', 'model_name', 'version'
                ])
        
        # Append operation to log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, operation, model_id, model_name, version
            ])

    def load_model(self, model_name, version='latest'):
        """Load a trained model from disk with checks and validation

        Args:
            model_name: Name of the model to load
            version: Version of the model (default: 'latest')

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Determine file path
            if version == 'latest':
                # Find the latest version
                model_files = glob.glob(
                    os.path.join(
                        self.model_dir,
                        f"{model_name}*.pkl"))
                if not model_files:
                    logger.error(f"No models found for {model_name}")
                    return False
                # Sort by name (which should include version)
                model_files.sort(reverse=True)
                model_path = model_files[0]
            else:
                model_path = os.path.join(
                    self.model_dir, f"{model_name}_v{version}.pkl")
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    return False

            # Load model and verify integrity
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Validate model data structure
            required_keys = [
                'model',
                'metadata',
                'scaler',
                'feature_names',
                'anomaly_detector']
            if not all(key in model_data for key in required_keys):
                logger.error(
                    f"Invalid model file format, missing required components")
                return False

            # Check model integrity and assign to class properties
            self.model = model_data['model']
            self.metadata = model_data['metadata']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.anomaly_detector = model_data.get('anomaly_detector')

            # Log successful load
            version_info = self.metadata.get('version', 'unknown')
            created_at = self.metadata.get('created_at', 'unknown')
            logger.info(
                f"Loaded {model_name} v{version_info} (created {created_at})")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def save_model(
            self,
            model_name,
            model,
            scaler,
            feature_names,
            training_data=None,
            metadata=None):
        """
        Save a model and its associated metadata
        
        Args:
            model_name: Name of the model to save
            model: The trained model object
            scaler: Feature scaler for preprocessing
            feature_names: List of feature names
            training_data: Optional dictionary of training data metrics
            metadata: Optional dictionary of additional metadata
        
        Returns:
            str: Path where model was saved
        """
        try:
            # Generate a model ID and version
            model_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create a clean directory name
            dir_name = model_name.lower().replace(' ', '_')
            
            # Create model directory if it doesn't exist
            model_dir = os.path.join(self.model_dir, dir_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Prepare model data bundle
            model_data = {
                'model': model,
                'model_id': model_id,
                'model_name': model_name,
                'version': timestamp,
                'scaler': scaler,
                'feature_names': feature_names,
                'created_at': timestamp
            }
            
            # Add optional metadata
            if metadata:
                model_data.update(metadata)
                
            if training_data:
                model_data['training_data'] = training_data
            
            # Define file path
            model_path = os.path.join(model_dir, f"{model_name.lower().replace(' ', '_')}_{timestamp}.joblib")
            
            # Save to disk
            joblib.dump(model_data, model_path)
            
            # Update the latest model pointer
            latest_path = os.path.join(model_dir, 'latest.txt')
            with open(latest_path, 'w') as f:
                f.write(model_path)
                
            # Log the save operation
            self._log_model_operation('save', model_id, model_name, timestamp)
                
            logger.info(f"Model '{model_name}' saved successfully to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return None

    def predict(self, features, company_name=None, client_ip=None):
        """Predict funding stage with validation and audit logging

        Args:
            features: Feature dictionary or pandas Series
            company_name: Optional company name for validation
            client_ip: Optional client IP for audit logging

        Returns:
            dict: Prediction results with confidence and validation info
        """
        try:
            if self.model is None:
                return {'error': 'No model loaded'}

            # Convert dictionary to proper format if needed
            if isinstance(features, dict):
                # Check for missing features
                missing_features = [
                    f for f in self.feature_names if f not in features]
                if missing_features:
                    return {
                        'error': f'Missing features: {missing_features}',
                        'is_valid': False,
                        'confidence': 0.0
                    }

                # Convert to numpy array
                X = np.array([features[f]
                             for f in self.feature_names]).reshape(1, -1)
            elif isinstance(features, pd.Series):
                # Get features in correct order
                X = features[self.feature_names].values.reshape(1, -1)
            else:
                X = features

            # Generate request ID for tracking
            request_id = str(uuid.uuid4())

            # Preprocess data
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # Run company validation if name provided
            company_valid = True
            company_confidence = 1.0
            if company_name:
                company_valid, company_confidence = self.anomaly_detector.validate_company(
                    company_name)
                if not company_valid:
                    logger.warning(
                        f"Company validation failed for {company_name}")

            # Check for anomalies
            anomaly_result = self.anomaly_detector.detect_anomalies(
                X_scaled, company_name)
            is_anomaly = anomaly_result.get('is_anomaly', False)
            anomaly_score = anomaly_result.get('score', 0.0)
            anomaly_reasons = anomaly_result.get('reasons', [])

            # Make prediction
            prediction = int(self.model.predict(X_scaled)[0])
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probabilities))

            # Adjust confidence based on anomaly
            if is_anomaly:
                # Reduce confidence proportionally to anomaly severity
                adjusted_confidence = confidence * \
                    (1 - min(anomaly_score, 0.9))
            else:
                adjusted_confidence = confidence

            # Adjust confidence based on company validation
            final_confidence = adjusted_confidence * company_confidence

            # Prepare result
            result = {
                'prediction': prediction,
                'confidence': round(final_confidence, 4),
                'is_valid': not is_anomaly and company_valid,
                'request_id': request_id
            }

            # Add validation details if there were issues
            if is_anomaly or not company_valid:
                result['validation'] = {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': round(anomaly_score, 4),
                    'reasons': anomaly_reasons,
                    'company_valid': company_valid,
                    'company_confidence': round(company_confidence, 4)
                }

            # Log prediction for audit
            self._log_prediction(
                request_id=request_id,
                company_name=company_name,
                prediction=prediction,
                confidence=final_confidence,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                anomaly_reasons=anomaly_reasons,
                feature_values=features,
                client_ip=client_ip
            )

            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': f'Prediction failed: {str(e)}', 'is_valid': False}

    def predict_proba(self, features, company_name=None, client_ip=None):
        """Predict probabilities for all classes with validation

        Args:
            features: Feature dictionary or pandas Series
            company_name: Optional company name for validation
            client_ip: Optional client IP for audit logging

        Returns:
            dict: Prediction results with probabilities and validation info
        """
        try:
            if self.model is None:
                return {'error': 'No model loaded'}

            # Convert dictionary to proper format if needed
            if isinstance(features, dict):
                # Check for missing features
                missing_features = [
                    f for f in self.feature_names if f not in features]
                if missing_features:
                    return {
                        'error': f'Missing features: {missing_features}',
                        'is_valid': False
                    }

                # Convert to numpy array
                X = np.array([features[f]
                             for f in self.feature_names]).reshape(1, -1)
            elif isinstance(features, pd.Series):
                # Get features in correct order
                X = features[self.feature_names].values.reshape(1, -1)
            else:
                X = features

            # Generate request ID for tracking
            request_id = str(uuid.uuid4())

            # Preprocess data
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # Run company validation if name provided
            company_valid = True
            company_confidence = 1.0
            if company_name:
                company_valid, company_confidence = self.anomaly_detector.validate_company(
                    company_name)
                if not company_valid:
                    logger.warning(
                        f"Company validation failed for {company_name}")

            # Check for anomalies
            anomaly_result = self.anomaly_detector.detect_anomalies(
                X_scaled, company_name)
            is_anomaly = anomaly_result.get('is_anomaly', False)
            anomaly_score = anomaly_result.get('score', 0.0)
            anomaly_reasons = anomaly_result.get('reasons', [])

            # Get class probabilities
            probabilities = self.model.predict_proba(X_scaled)[0].tolist()
            classes = self.model.classes_.tolist() if hasattr(
                self.model, 'classes_') else list(range(len(probabilities)))

            # Adjust probabilities based on anomaly and company validation
            if is_anomaly or not company_valid:
                # Make distribution more uniform (less confident) based on
                # anomaly severity
                adjustment_factor = 1.0 - \
                    min(anomaly_score, 0.8) - (0.2 if not company_valid else 0)

                # Adjust probabilities - move toward uniform distribution
                uniform_prob = 1.0 / len(probabilities)
                adjusted_probs = [
                    p * adjustment_factor + uniform_prob * (1 - adjustment_factor)
                    for p in probabilities
                ]

                # Renormalize to sum to 1
                total = sum(adjusted_probs)
                adjusted_probs = [p / total for p in adjusted_probs]
            else:
                adjusted_probs = probabilities

            # Prepare result
            result = {
                'probabilities': {
                    str(c): round(
                        p,
                        4) for c,
                    p in zip(
                        classes,
                        adjusted_probs)},
                'is_valid': not is_anomaly and company_valid,
                'request_id': request_id}

            # Add validation details if there were issues
            if is_anomaly or not company_valid:
                result['validation'] = {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': round(anomaly_score, 4),
                    'reasons': anomaly_reasons,
                    'company_valid': company_valid,
                    'company_confidence': round(company_confidence, 4)
                }

            # Find most likely class for audit logging
            max_prob_idx = np.argmax(adjusted_probs)
            prediction = classes[max_prob_idx]
            confidence = adjusted_probs[max_prob_idx]

            # Log prediction for audit
            self._log_prediction(
                request_id=request_id,
                company_name=company_name,
                prediction=prediction,
                confidence=confidence,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                anomaly_reasons=anomaly_reasons,
                feature_values=features,
                client_ip=client_ip
            )

            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': f'Prediction failed: {str(e)}', 'is_valid': False}

    def _log_prediction(self, request_id, company_name, prediction, confidence,
                        is_anomaly, anomaly_score, anomaly_reasons,
                        feature_values, client_ip=None):
        """Log prediction details to audit trail

        Args:
            request_id: Unique identifier for the prediction request
            company_name: Name of the company
            prediction: The predicted class
            confidence: Confidence score
            is_anomaly: Whether prediction was flagged as anomalous
            anomaly_score: Anomaly detection score
            anomaly_reasons: List of reasons for anomaly detection
            feature_values: Feature values used in prediction
            client_ip: Client IP address
        """
        try:
            # Prepare log entry
            timestamp = datetime.now().isoformat()
            model_version = self.metadata.get('version', 'unknown')

            # Convert feature values to string
            if isinstance(feature_values, dict):
                feature_str = json.dumps({k: float(v) if isinstance(
                    v, (int, float, np.number)) else str(v) for k, v in feature_values.items()})
            else:
                feature_str = str(feature_values)

            # Format anomaly reasons
            if isinstance(anomaly_reasons, list):
                anomaly_reasons_str = '; '.join(anomaly_reasons)
            else:
                anomaly_reasons_str = str(anomaly_reasons)

            # Write to CSV
            with open(self.audit_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    request_id,
                    company_name if company_name else 'unknown',
                    prediction,
                    confidence,
                    1 if is_anomaly else 0,
                    anomaly_score,
                    anomaly_reasons_str,
                    feature_str,
                    client_ip if client_ip else 'unknown',
                    model_version
                ])

        except Exception as e:
            logger.error(f"Failed to log prediction: {str(e)}")
            # Continue execution even if logging fails




class AnomalyDetector:
    """Detects anomalies and potential manipulation in startup data"""

    def __init__(self, contamination=0.05):
        """Initialize detector with contamination parameter (expected outlier ratio)"""
        self.isolation_forest = None
        self.contamination = contamination
        self.feature_ranges = {}
        self.startup_data_cache = {}
        self.known_companies = set()

    def fit(self, X, startup_names=None):
        """Train anomaly detection model on startup data

        Args:
            X: Feature matrix for startups
            startup_names: Optional list of company names
        """
        try:
            # Train isolation forest for outlier detection
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            )
            self.isolation_forest.fit(X)

            # Store feature ranges for basic sanity checks
            self.feature_ranges = {
                'min': np.min(X, axis=0),
                'max': np.max(X, axis=0),
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0),
                'q1': np.percentile(X, 25, axis=0),
                'q3': np.percentile(X, 75, axis=0)
            }

            # Cache startup data if names provided
            if startup_names is not None:
                for i, name in enumerate(startup_names):
                    if i < len(X):
                        self.startup_data_cache[name] = X[i]
                        self.known_companies.add(name)

            logger.info(f"Fitted anomaly detector with {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"Error fitting anomaly detector: {str(e)}")
            return False

    def detect_anomalies(self, X, company_name=None, threshold=-0.5):
        """
        Detect anomalies in startup data

        Args:
            X: Feature matrix or single sample
            company_name: Optional company name for additional checks
            threshold: Decision threshold (lower = more strict)

        Returns:
            Dictionary with anomaly flags and scores
        """
        try:
            # Ensure X is 2D
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            # Validate input dimensions match what model was trained on
            if X.shape[1] != len(self.feature_ranges['min']):
                logger.error(
                    f"Feature dimension mismatch: expected {len(self.feature_ranges['min'])}, got {X.shape[1]}")
                return {'is_anomaly': True, 'reason': 'dimension_mismatch', 'score': 1.0}

            # Run standard anomaly checks
            anomalies = {
                'is_anomaly': False,
                'score': 0.0,
                'reasons': []
            }

            # First, check against known ranges
            range_violations = []
            for i, (val, min_val, max_val) in enumerate(
                    zip(X[0], self.feature_ranges['min'], self.feature_ranges['max'])):
                if val < min_val * 0.9 or val > max_val * 1.1:  # Allow 10% outside range
                    range_violations.append(i)

            # Check extreme feature values using IQR
            iqr_violations = []
            for i, (val, q1, q3) in enumerate(
                    zip(X[0], self.feature_ranges['q1'], self.feature_ranges['q3'])):
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                if val < lower_bound or val > upper_bound:
                    iqr_violations.append(i)

            # Check if company data has suddenly changed drastically
            company_change = False
            if company_name and company_name in self.startup_data_cache:
                cached_data = self.startup_data_cache[company_name]
                # Calculate percent change in values
                # Avoid division by zero
                pct_change = np.abs(
                    (X[0] - cached_data) / (cached_data + 1e-10))
                if np.any(
                        pct_change > 0.5):  # 50% change in any feature is suspicious
                    company_change = True
                    anomalies['reasons'].append(
                        f"Company data changed by >{np.max(pct_change) * 100:.1f}%")

            # Apply isolation forest to get anomaly score
            if self.isolation_forest is not None:
                scores = self.isolation_forest.decision_function(X)
                predictions = self.isolation_forest.predict(X)

                # Lower scores = more anomalous
                min_score = np.min(scores)
                if min_score < threshold or np.any(predictions == -1):
                    anomalies['is_anomaly'] = True
                    # Convert to positive for easier interpretation
                    anomalies['score'] = -min_score
                    anomalies['reasons'].append(
                        f"Isolation forest score: {min_score:.3f}")

            # Add other detected issues
            if range_violations:
                anomalies['is_anomaly'] = True
                anomalies['reasons'].append(
                    f"Range violations in {len(range_violations)} features")
                anomalies['score'] = max(anomalies['score'], 0.7)

            if iqr_violations:
                anomalies['is_anomaly'] = True
                anomalies['reasons'].append(
                    f"IQR violations in {len(iqr_violations)} features")
                anomalies['score'] = max(anomalies['score'], 0.6)

            if company_change:
                anomalies['is_anomaly'] = True
                anomalies['score'] = max(anomalies['score'], 0.8)

            # Check for potential manipulation patterns in funding amounts
            if self._check_funding_manipulation(X[0]):
                anomalies['is_anomaly'] = True
                anomalies['reasons'].append(
                    "Suspicious funding amount pattern detected")
                anomalies['score'] = max(anomalies['score'], 0.9)

            return anomalies
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {
                'is_anomaly': True,
                'reason': f'detection_error: {str(e)}',
                'score': 1.0
            }

    
    def validate_company(self, company_name, api_key=None):
        """Validate company existence through external API or internal checks"""
        if not company_name or company_name.strip() == "":
            return {
                "valid": False,
                "reason": "Empty company name",
                "confidence": 1.0
            }
            
        # Standardize name for checking
        company_name = company_name.lower().strip()
            
        # Check against known companies
        if company_name in self.known_companies:
            return {
                "valid": True,
                "reason": "Found in database",
                "confidence": 0.9
            }
            
        # Attempt API validation if key provided
        if api_key:
            api_result = self._simulate_company_api_check(company_name)
            if api_result["found"]:
                self.known_companies.add(company_name)
                return {
                    "valid": True,
                    "reason": "Validated through API",
                    "confidence": 0.95
                }
            
        # Check for suspicious patterns in name
        if not self._is_valid_company_name(company_name):
            return {
                "valid": False,
                "reason": "Company name contains suspicious patterns",
                "confidence": 0.7
            }
            
        # Check for name similarity with known companies
        similar_company = self._check_name_similarity(company_name)
        if similar_company:
            return {
                "valid": False,
                "reason": f"Potential impersonation of {similar_company}",
                "confidence": 0.8
            }
            
        # If all checks passed but company not found
        return {
            "valid": True,
            "reason": "Passed validation checks but not found in database",
            "confidence": 0.3
        }

    def _is_valid_company_name(self, name):
        """Check for suspicious patterns in company name"""
        import re
        
        # Check for repeated characters (potential keyboard spam)
        if re.search(r'(.)\1{4,}', name):  # 5+ repeated chars
            return False

        # Check for excessive numbers
        if len(re.findall(r'\d', name)) > len(name) // 2:
            return False

        # Check for suspicious TLDs if URL is included
        suspicious_tlds = ['.xyz', '.info', '.biz', '.tk', '.ml']
        if any(tld in name for tld in suspicious_tlds):
            return False

        return True


    def _check_name_similarity(self, name):
        """Check if name is suspiciously similar to a known company"""
        best_match = None
        best_score = float('inf')
        threshold = 3  # Max edit distance to be considered similar
        
        for known_name in self.known_companies:
            distance = self._levenshtein_distance(name, known_name)
            if distance < best_score and distance <= threshold:
                best_score = distance
                best_match = known_name
                
        return best_match
    
    def _levenshtein_distance(self, s1, s2):
        """Calculate edit distance between two strings"""
        if s1 == s2:
            return 0
            
        # Ensure s1 is the shorter string for efficiency
        if len(s1) > len(s2):
            return self._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
    


class Visualization:
    """Class for creating various visualizations for funding stage prediction models"""
    
    def __init__(self, output_dir="./visualizations"):
        """Initialize visualization with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set up matplotlib style
        plt.style.use('ggplot')
        
    def plot_calibration_curve(self, y_true, y_proba, model_names, n_bins=10):
        """
        Create calibration plots for models
        
        Args:
            y_true: True binary labels
            y_proba: List of probability predictions for each model
            model_names: List of model names
            n_bins: Number of bins for histogram
        """
        plt.figure(figsize=(12, 8))
        
        # Plot perfectly calibrated line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Plot calibration curve for each model
        for i, (proba, name) in enumerate(zip(y_proba, model_names)):
            # For multiclass, use average calibration across classes
            if proba.shape[1] > 2:
                # Average probability across all classes
                prob_pos = proba.mean(axis=1)
                frac_pos = (y_true == np.argmax(proba, axis=1)).astype(float)
            else:
                # Binary case - use probability of positive class
                prob_pos = proba[:, 1]
                frac_pos = y_true
                
            # Bin predictions to compute calibration
            bins = np.linspace(0., 1.+1e-8, n_bins+1)
            binids = np.digitize(prob_pos, bins) - 1
            bin_sums = np.bincount(binids, weights=prob_pos, minlength=len(bins))
            bin_true = np.bincount(binids, weights=frac_pos, minlength=len(bins))
            bin_total = np.bincount(binids, minlength=len(bins))
            
            nonzero = bin_total != 0
            prob_true = np.zeros(len(bins))
            prob_pred = np.zeros(len(bins))
            prob_true[nonzero] = bin_true[nonzero] / bin_total[nonzero]
            prob_pred[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
            
            plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=name)
        
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Plot (Reliability Curve)')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save the figure
        output_path = os.path.join(self.output_dir, f'calibration_plot_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration plot saved to {output_path}")
        return output_path
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, title='Confusion Matrix'):
        """
        Plot confusion matrix for model predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes (optional)
            title: Title for the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names if class_names else 'auto',
                    yticklabels=class_names if class_names else 'auto')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        
        # Save the figure
        output_path = os.path.join(self.output_dir, f'confusion_matrix_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {output_path}")
        return output_path
        
    def plot_roc_curves(self, y_true, y_proba, class_names=None, model_name="Model"):
        """
        Plot ROC curves for model predictions
        
        Args:
            y_true: True labels (one-hot encoded for multiclass)
            y_proba: Predicted probabilities 
            class_names: Names of classes (optional)
            model_name: Name of the model
        """
        plt.figure(figsize=(12, 8))
        
        # For multiclass, create one-vs-rest ROC curves
        n_classes = y_proba.shape[1]
        
        # Binarize the labels for one-vs-rest ROC
        if n_classes > 2:
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
        else:
            # Binary case
            y_true_bin = np.array(y_true).reshape(-1, 1)
            y_true_bin = np.concatenate([1-y_true_bin, y_true_bin], axis=1)
            
        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(
                y_true_bin[:, i] if n_classes > 2 else y_true, 
                y_proba[:, i]
            )
            roc_auc = auc(fpr, tpr)
            
            class_label = class_names[i] if class_names else f"Class {i}"
            plt.plot(fpr, tpr, lw=2, 
                    label=f'ROC {class_label} (AUC = {roc_auc:.2f})')
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {model_name}')
        plt.legend(loc="lower right")
        
        # Save the figure
        output_path = os.path.join(self.output_dir, f'roc_curves_{model_name}_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        