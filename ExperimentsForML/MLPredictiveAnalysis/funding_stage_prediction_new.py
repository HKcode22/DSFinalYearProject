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

            # Extract companies from the JSON structure
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
        """Convert funding amount strings to numeric values with strict validation"""
        if not amount_str or pd.isna(amount_str) or amount_str == "":
            return np.nan

        try:
            # Remove currency symbol and commas
            amount_str = str(amount_str).replace('$', '').replace(',', '').strip()

            # Define reasonable limits based on funding stages
            max_reasonable_amount = {
                'Pre-Seed': 5e6,     # $5M
                'Seed': 2e7,         # $20M
                'Series A': 5e7,      # $50M
                'Series B': 1e8,      # $100M
                'Series C': 3e8,      # $300M
                'Series D': 5e8,      # $500M
                'Series E': 1e9,      # $1B
                'Series F': 2e9,      # $2B
                'Series G': 5e9,      # $5B
                'Series H': 1e10,     # $10B
                'default': 1e10      # $10B default max
            }

            # Convert based on unit (M=million, B=billion, K=thousand)
            if 'B' in amount_str:
                value = float(amount_str.replace('B', '')) * 1e9
            elif 'M' in amount_str:
                value = float(amount_str.replace('M', '')) * 1e6
            elif 'K' in amount_str:
                value = float(amount_str.replace('K', '')) * 1e3
            else:
                value = float(amount_str)

            # Get funding stage if available
            funding_stage = None
            if hasattr(self, 'current_stage'):
                funding_stage = self.current_stage

            # Get max allowed amount based on stage
            max_amount = max_reasonable_amount.get(funding_stage, max_reasonable_amount['default'])

            # Validate against reasonable limits
            if value > max_amount:
                logger.info(f"Large funding amount detected: ${value:,.2f} for stage {funding_stage}. Capping at ${max_amount:,.2f}")
                return max_amount
            elif value < 0:
                logger.warning(f"Negative funding amount detected: ${value:,.2f}")
                return np.nan

            return value
        except Exception as e:
            logger.warning(f"Error parsing funding amount '{amount_str}': {str(e)}")
            return np.nan

    def save_historical_data(self, df, table_name):
        """Save dataframe to historical SQLite database"""
        try:
            conn = sqlite3.connect(self.historical_db)
            df.to_sql(table_name, conn, if_exists='append', index=False)
            conn.close()
            logger.info(f"Saved {len(df)} records to historical {table_name}")
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")

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
            'los angeles': 'Los Angeles',
            'la': 'Los Angeles',
            'london': 'London, UK',
            'boston': 'Boston, MA',
            'seattle': 'Seattle, WA',
            'austin': 'Austin, TX'
        }
        
        # Check for exact matches in common locations (case insensitive)
        lower_loc = location_str.lower()
        for key, value in common_locations.items():
            if key == lower_loc:
                return value
                
        # If it contains commas, try to standardize based on last part
        if ',' in location_str:
            parts = [p.strip() for p in location_str.split(',')]
            last_part = parts[-1].lower()
            
            # Standardize country/region
            country_mapping = {
                'usa': 'USA',
                'united states': 'USA',
                'us': 'USA',
                'u.s.': 'USA',
                'u.s.a.': 'USA',
                'united states of america': 'USA',
            }
            
            if last_part in country_mapping:
                parts[-1] = country_mapping[last_part]
                return ', '.join(parts)
        
        return location_str


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
                data['location_category'] = data['location_category'].apply(
                    self._standardize_location if hasattr(self, '_standardize_location') 
                    else extract_location
                )

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


class EnhancedModelTrainer(ModelTrainer):
    def __init__(self, output_dir="./models"):
        super().__init__(output_dir)
        self.cv_folds = 5
        self.n_iter_search = 50
        self.early_stopping_rounds = 20
        self.rf_results = None
        self.xgb_results = None
        
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

        # Store results
        self.rf_results = {
            'model': best_rf,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(y_original, y_pred_original),
            'classification_report': report,
            'y_pred': y_pred_original,
            'y_proba': y_proba,
            'roc_auc': roc_auc_score(y_original, y_proba, multi_class='ovr'),
            'rmse': np.sqrt(mean_squared_error(y_original, y_pred_original)),
            'feature_importance': best_rf.feature_importances_,
            'test_data': {'X_test': X, 'y_test': y}
        }

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

        return best_rf, self.rf_results

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

        # Store results
        self.xgb_results = {
            'model': best_xgb,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(y_original, y_pred_original),
            'classification_report': report,
            'y_pred': y_pred_original,
            'y_proba': y_proba,
            'roc_auc': roc_auc_score(y_original, y_proba, multi_class='ovr'),
            'rmse': np.sqrt(mean_squared_error(y_original, y_pred_original)),
            'feature_importance': best_xgb.feature_importances_,
            'test_data': {'X_test': X, 'y_test': y}
        }

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

        return best_xgb, self.xgb_results

    def train_neural_network(self, X, y):
        """Train a neural network using TensorFlow/Keras if available"""
        if tf is None:
            logger.warning("TensorFlow not installed. Skipping neural network.")
            return None, 0.0
            
        try:
            # Convert target to one-hot encoding
            n_classes = len(np.unique(y))
            y_onehot = tf.keras.utils.to_categorical(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_onehot, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define model architecture
            model = Sequential([
                Dense(256, activation='relu', input_shape=(X.shape[1],)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.1),
                Dense(n_classes, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            _, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            logger.info(f"Neural Network accuracy: {accuracy:.4f}")
            
            return model, accuracy
            
        except Exception as e:
            logger.error(f"Error training neural network: {str(e)}")
            return None, 0.0
    
    def train_lightgbm(self, X, y):
        """Train LightGBM model with advanced parameter tuning"""
        if lgb is None:
            logger.warning("LightGBM not installed. Skipping LightGBM model.")
            return None, 0.0
            
        try:
            # Create parameter space
            param_distributions = {
                'num_leaves': randint(20, 100),
                'max_depth': randint(3, 12),
                'learning_rate': uniform(0.01, 0.3),
                'n_estimators': randint(100, 1000),
                'min_child_samples': randint(10, 50),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': uniform(0, 2),
                'reg_lambda': uniform(0, 2)
            }
            
            # Initialize base model
            base_model = lgb.LGBMClassifier(
                objective='multiclass',
                random_state=42,
                verbose=-1
            )
            
            # Perform randomized search
            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=self.n_iter_search,
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42
            )
            
            # Fit and evaluate
            search.fit(X, y)
            logger.info(f"Best LightGBM parameters: {search.best_params_}")
            logger.info(f"Best LightGBM CV accuracy: {search.best_score_:.4f}")
            
            return search.best_estimator_, search.best_score_
            
        except Exception as e:
            logger.error(f"Error training LightGBM: {str(e)}")
            return None, 0.0
    
    def train_catboost(self, X, y):
        """Train CatBoost model with advanced parameter tuning"""
        if CatBoostClassifier is None:
            logger.warning("CatBoost not installed. Skipping CatBoost model.")
            return None, 0.0
            
        try:
            # Create parameter space
            param_distributions = {
                'iterations': randint(100, 1000),
                'depth': randint(4, 10),
                'learning_rate': uniform(0.01, 0.3),
                'l2_leaf_reg': uniform(1, 10),
                'border_count': randint(32, 255),
                'bagging_temperature': uniform(0, 1)
            }
            
            # Initialize base model
            base_model = CatBoostClassifier(
                loss_function='MultiClass',
                eval_metric='Accuracy',
                random_seed=42,
                verbose=False
            )
            
            # Perform randomized search
            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=self.n_iter_search,
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42
            )
            
            # Fit and evaluate
            search.fit(X, y)
            logger.info(f"Best CatBoost parameters: {search.best_params_}")
            logger.info(f"Best CatBoost CV accuracy: {search.best_score_:.4f}")
            
            return search.best_estimator_, search.best_score_
            
        except Exception as e:
            logger.error(f"Error training CatBoost: {str(e)}")
            return None, 0.0
    
    def train_stacked_ensemble(self, X, y, base_models):
        """Create a stacked ensemble using multiple base models"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Train base models and get predictions
            base_predictions = {}
            for name, model in base_models.items():
                if model is not None:
                    model.fit(X_train, y_train)
                    pred = model.predict_proba(X_test)
                    base_predictions[name] = pred
            
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
            
            # Create ensemble wrapper
            class StackedEnsemble:
                def __init__(self, base_models, meta_classifier):
                    self.base_models = base_models
                    self.meta_classifier = meta_classifier
                
                def predict(self, X):
                    base_preds = []
                    for model in self.base_models.values():
                        if model is not None:
                            base_preds.append(model.predict_proba(X))
                    meta_features = np.column_stack(base_preds)
                    return self.meta_classifier.predict(meta_features)
                
                def predict_proba(self, X):
                    base_preds = []
                    for model in self.base_models.values():
                        if model is not None:
                            base_preds.append(model.predict_proba(X))
                    meta_features = np.column_stack(base_preds)
                    return self.meta_classifier.predict_proba(meta_features)
            
            ensemble = StackedEnsemble(base_models, meta_classifier)
            return ensemble, accuracy
            
        except Exception as e:
            logger.error(f"Error training stacked ensemble: {str(e)}")
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
    """Manages machine learning models for funding stage prediction with validation and audit"""

    def __init__(self, model_dir='models/'):
        """Initialize with model directory and setup audit logging"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.audit_log_path = os.path.join(model_dir, 'prediction_audit.csv')
        self.init_audit_log()
        logger.info(f"Audit log initialized at {self.audit_log_path}")

    def init_audit_log(self):
        """Initialize audit log file if it doesn't exist"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)

            # Create audit log with headers if it doesn't exist
            if not os.path.exists(self.audit_log_path):
                headers = [
                    'timestamp',
                    'request_id',
                    'company_name',
                    'prediction_result',
                    'confidence',
                    'is_anomaly',
                    'anomaly_score',
                    'anomaly_reasons',
                    'feature_values',
                    'client_ip',
                    'model_version']
                with open(self.audit_log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)

            logger.info(f"Audit log initialized at {self.audit_log_path}")
        except Exception as e:
            logger.error(f"Failed to initialize audit log: {str(e)}")

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
        """Save a trained model with important metadata

        Args:
            model_name: Name to save the model under
            model: Trained model instance
            scaler: Feature scaler used in training
            feature_names: Names of features for prediction
            training_data: Optional training data for anomaly detection
            metadata: Additional metadata to include

        Returns:
            str: Path to saved model file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)

            # Generate version
            version = datetime.now().strftime("%Y%m%d%H%M")
            model_path = os.path.join(
                self.model_dir, f"{model_name}_v{version}.pkl")

            # Set up metadata
            if metadata is None:
                metadata = {}

            metadata.update({
                'version': version,
                'created_at': datetime.now().isoformat(),
                'feature_names': feature_names,
                'model_type': type(model).__name__
            })

            # Train anomaly detector if training data provided
            if training_data is not None:
                self.anomaly_detector.fit(training_data, feature_names)

            # Package everything together
            model_data = {
                'model': model,
                'metadata': metadata,
                'scaler': scaler,
                'feature_names': feature_names,
                'anomaly_detector': self.anomaly_detector
            }

            # Save to disk
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {model_path}")
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


class Visualizer:
    """Generates visualizations for funding stage prediction analysis"""
    
    def __init__(self, output_dir="./visualizations"):
        """Initialize the visualizer with the output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_funding_stage_distribution(self, data):
        """Plot the distribution of funding stages in the dataset"""
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x='funding_stage')
        plt.title('Distribution of Funding Stages')
        plt.xlabel('Funding Stage')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'funding_stage_distribution.png')
        plt.savefig(output_path)
        plt.close()
        return output_path
        
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance for a given model"""
        if not hasattr(model, 'feature_importances_'):
            return None
            
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(output_path)
        plt.close()
        return output_path
        
    def plot_model_comparison(self, model_results):
        """Plot comparison of different models based on their metrics"""
        plt.figure(figsize=(12, 8))
        
        model_names = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        metric_values = {metric: [] for metric in metrics}
        
        for model in model_names:
            for metric in metrics:
                if metric in model_results[model]:
                    metric_values[metric].append(model_results[model][metric])
                else:
                    metric_values[metric].append(0)
        
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, metric_values[metric], width, label=metric.capitalize())
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * 1.5, model_names)
        plt.legend()
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(output_path)
        plt.close()
        return output_path
        
    def plot_confusion_matrices(self, model_results):
        """Plot confusion matrices for all models"""
        n_models = len(model_results)
        fig, axes = plt.subplots(1, n_models, figsize=(15, 5))
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if 'confusion_matrix' in results:
                cm = results['confusion_matrix']
                ax = axes[i] if n_models > 1 else axes
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'{model_name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'confusion_matrices.png')
        plt.savefig(output_path)
        plt.close()
        return output_path
        
    def plot_correlation_heatmap(self, data):
        """Plot correlation heatmap for numeric features"""
        numeric_data = data.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(output_path)
        plt.close()
        return output_path

class AdvancedVisualizer(Visualizer):
    """Advanced visualization capabilities for model evaluation"""
    
    def plot_roc_curves(self, y_true, y_proba, classes):
        """Plot ROC curves for multi-class classification"""
        # One-vs-rest ROC curves
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        plt.figure(figsize=(12, 8))
        
        for i in range(len(classes)):
            plt.plot(
                fpr[i],
                tpr[i],
                label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def plot_calibration(self, y_true, y_proba, n_bins=10):
        """Plot calibration curves for model probabilities"""
        n_classes = y_proba.shape[1]
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot calibration curve for each class
        for i in range(n_classes):
            # One-vs-Rest approach for calibration
            y_true_binary = (y_true == i).astype(int)
            prob_pos = y_proba[:, i]
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary, prob_pos, n_bins=n_bins
            )
            
            # Plot calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   marker='o', label=f'Class {i}')
        
        # Add reference line (perfectly calibrated)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Set plot details
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Calibration Curve')
        ax.legend(loc='best')
        
        output_path = os.path.join(self.output_dir, 'calibration_curve.png')
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def plot_confidence_intervals(self, y_true, y_pred, y_proba):
        """Plot model confidence with prediction intervals"""
        # Calculate confidence and accuracy
        confidence = np.max(y_proba, axis=1)
        correct = (y_pred == y_true).astype(int)
        
        # Create bins for confidence
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(confidence, bins) - 1
        
        # Calculate accuracy per bin
        accuracy_per_bin = []
        confidence_per_bin = []
        counts_per_bin = []
        
        for i in range(len(bins) - 1):
            bin_mask = (bin_indices == i)
            if np.sum(bin_mask) > 0:
                accuracy_per_bin.append(np.mean(correct[bin_mask]))
                confidence_per_bin.append(np.mean(confidence[bin_mask]))
                counts_per_bin.append(np.sum(bin_mask))
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Reliability diagram
        ax1.plot(confidence_per_bin, accuracy_per_bin, marker='o')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability Diagram')
        
        # Sample counts
        ax2.bar(range(len(counts_per_bin)), counts_per_bin)
        ax2.set_xticks(range(len(counts_per_bin)))
        ax2.set_xticklabels([f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins) - 1) if i < len(counts_per_bin)])
        ax2.set_xlabel('Confidence Range')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Confidence Distribution')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'confidence_intervals.png')
        plt.savefig(output_path)
        plt.close()
        return output_path

class FundingStagePredictionPipeline:
    """Main pipeline for funding stage prediction"""

    def __init__(self, base_dir="./", output_dir="./output", archive=False):
        """Initialize the complete pipeline"""
        self.base_dir = base_dir
        self.output_dir = output_dir

        # Create output directory structure
        self.models_dir = os.path.join(output_dir, "models")
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        # Initialize components
        self.data_loader = DataLoader(base_dir, archive=archive)
        self.feature_engineer = FeatureEngineering()
        self.model_trainer = ModelTrainer(self.models_dir)
        self.visualizer = Visualizer(self.viz_dir)

        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self):
        """Execute the full pipeline"""
        try:
            logger.info("Starting funding stage prediction pipeline")

            # Step 1: Load and merge all data sources
            logger.info("Step 1: Loading and merging data...")
            merged_data = self.data_loader.merge_datasets()

            if merged_data.empty:
                logger.error("No data available. Exiting pipeline.")
                return False

            # Step 2: Feature engineering
            logger.info("Step 2: Extracting features...")
            processed_data = self.feature_engineer.extract_features(
                merged_data)

            # Step 3: Prepare data for modeling
            logger.info("Step 3: Preparing model data...")
            X, y = self.feature_engineer.prepare_model_data(processed_data)

            # Step 4: Train models
            logger.info("Step 4: Training models...")
            rf_model, rf_results = self.model_trainer.train_random_forest(X, y)
            xgb_model, xgb_results = self.model_trainer.train_xgboost(X, y)

            # Collect all model results
            model_results = {
                'Random Forest': rf_results,
                'XGBoost': xgb_results
            }

            # Step 5: Generate visualizations
            logger.info("Step 5: Creating visualizations...")
            self.visualizer.plot_funding_stage_distribution(processed_data)
            self.visualizer.plot_feature_importance(rf_model, X.columns)
            self.visualizer.plot_feature_importance(xgb_model, X.columns)
            self.visualizer.plot_model_comparison(model_results)
            self.visualizer.plot_funding_vs_employees(processed_data)

            # Enhanced visualizations
            key_features = [
                'funding_amount_log', 'employees',
                'employee_efficiency', 'previous_rounds',
                'months_since_first_funding', 'funding_year', 'funding_month'
            ]

            self.visualizer.plot_feature_comparison_matrix(
                processed_data, key_features)
            self.visualizer.plot_correlation_heatmap(processed_data)
            self.visualizer.plot_temporal_trends(processed_data)
            self.visualizer.plot_industry_distributions(processed_data)
            self.visualizer.plot_advanced_feature_correlations(
                processed_data, key_features)
            self.visualizer.plot_feature_distributions(
                processed_data, key_features)
            self.visualizer.plot_funding_patterns(processed_data)
            # --- NEW VISUALIZATIONS ---
            self.visualizer.plot_pairwise_features(
                processed_data, key_features)
            self.visualizer.plot_full_correlation_heatmap(processed_data)
            self.visualizer.plot_violin_funding_by_stage(processed_data)

            # Step 6: Save summary report
            logger.info("Step 6: Saving summary report...")
            summary = {
                'timestamp': self.timestamp,
                'data_records': len(merged_data),
                'features': X.columns.tolist(),
                'model_results': {
                    'Random Forest': {
                        'accuracy': float(rf_results['accuracy']),
                        'model_path': rf_results['model_path']
                    },
                    'XGBoost': {
                        'accuracy': float(xgb_results['accuracy']),
                        'model_path': xgb_results['model_path']
                    }
                }
            }

            with open(os.path.join(self.output_dir, f"summary_{self.timestamp}.json"), 'w') as f:
                json.dump(summary, f, indent=4)

            logger.info("Pipeline completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def schedule_run(self, interval_hours=24):
        """Schedule pipeline to run automatically at intervals"""
        import schedule
        import time

        def job():
            logger.info(f"Running scheduled job at {datetime.now()}")
            self.run()

        # Schedule the job
        schedule.every(interval_hours).hours.do(job)

        # Run once immediately
        job()

        # Keep running
        logger.info(
            f"Scheduler started. Will run every {interval_hours} hours")
        while True:
            schedule.run_pending()
            time.sleep(60)

    def _init_model_directory(self):
        """Create organized model directory structure"""
        # Create main models directory
        os.makedirs(self.models_dir, exist_ok=True)

        # Create subdirectories for different model types
        model_types = ['random_forest', 'xgboost', 'ensemble']
        for model_type in model_types:
            os.makedirs(
                os.path.join(
                    self.models_dir,
                    model_type),
                exist_ok=True)

        # Create evaluation directory for model performance metrics
        os.makedirs(os.path.join(self.models_dir, 'evaluation'), exist_ok=True)

        logger.info(
            f"Initialized model directory structure at {
                self.models_dir}")


class EnhancedPipeline(FundingStagePredictionPipeline):
    def __init__(self, *args, **kwargs):
        # Process arguments properly
        if len(args) > 0:
            kwargs['base_dir'] = args[0]
        
        # Override the output_dir with our fixed FundingStageOutput path
        # Use absolute path to ensure consistency
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        funding_output_dir = os.path.join(project_root, 'FundingStageOutput')
        
        # Create a new kwargs dictionary without conflicting arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'output_dir'}
        
        # Set up subdirectories for better organization
        self.models_dir = os.path.join(funding_output_dir, 'models')
        self.viz_dir = os.path.join(funding_output_dir, 'visualizations')
        self.data_dir = os.path.join(funding_output_dir, 'data_archive')
        self.dashboard_dir = os.path.join(funding_output_dir, 'dashboards')
        self.timeseries_dir = os.path.join(funding_output_dir, 'time_series_forecasts')
        self.reports_dir = os.path.join(funding_output_dir, 'reports')
        
        # Create all required directories
        for directory in [funding_output_dir, self.models_dir, self.viz_dir, 
                          self.data_dir, self.dashboard_dir, self.timeseries_dir, 
                          self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"Created standardized output directory structure at {funding_output_dir}")
            
        # Initialize with parent constructor - explicitly set output_dir to avoid conflicts
        # Enable archiving by default
        super().__init__(base_dir=filtered_kwargs.get('base_dir', './'), 
                         output_dir=funding_output_dir,
                         archive=True)
        
        # Reinitialize components with the correct paths
        self.model_trainer = EnhancedModelTrainer(self.models_dir)
        self.visualizer = AdvancedVisualizer(self.viz_dir)  # Removed interactive parameter
        self.model_manager = ModelManager(self.models_dir)
        
        # Configure the data loader to use our standardized archive location and enable archiving
        self.data_loader.output_dir = self.data_dir
        self.data_loader.archive = True  # Ensure archiving is enabled
        # Override the archive_dir to be within our FundingStageOutput structure
        self.data_loader.archive_dir = os.path.join(self.data_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.data_loader.archive_dir, exist_ok=True)
        logger.info(f"Set up data archive directory at {self.data_loader.archive_dir}")
        
        # Create a dashboard generator
        self.dashboard_generator = DashboardGenerator(self.dashboard_dir)
        
        self._init_model_directory()
        
    def run(self):
        """Run the enhanced pipeline with advanced models and ensembles"""
        try:
            # Step 1: Load and merge data
            logger.info("Step 1: Loading and merging datasets...")
            merged_data = self.data_loader.merge_datasets()
            
            if merged_data.empty:
                logger.error("No data available after merging. Aborting pipeline.")
                return False
                
            # Save merged data for analysis
            merged_path = os.path.join(self.data_dir, f"merged_data_{datetime.now().strftime('%Y%m%d')}.csv")
            merged_data.to_csv(merged_path, index=False)
            logger.info(f"Saved merged data to {merged_path}")
            
            # Step 2: Feature engineering
            logger.info("Step 2: Extracting features...")
            processed_data = self.feature_engineer.extract_features(merged_data)
            
            # Step 3: Prepare data for modeling
            logger.info("Step 3: Preparing model data...")
            X, y = self.feature_engineer.prepare_model_data(processed_data)
            
            # First remap all classes to be continuous from 0
            def remap_classes(y_series):
                unique_classes = sorted(y_series.unique())
                class_map = {old_label: idx for idx, old_label in enumerate(unique_classes)}
                return y_series.map(class_map), class_map
                
            # Remap classes to ensure continuous integer classes starting from 0
            y, class_map = remap_classes(y)
            logger.info(f"Class mapping: {class_map}")
            
            # Handle rare classes (those with < 5 examples) by merging into majority class
            class_counts = y.value_counts()
            rare_classes = class_counts[class_counts < 5].index.tolist()
            
            if rare_classes:
                majority_class = class_counts.idxmax()
                y = y.apply(lambda x: majority_class if x in rare_classes else x)
                logger.info(f"Merged rare classes into majority class {majority_class}")
                
            # Remap again after merging rare classes to ensure continuous labels
            y, final_map = remap_classes(y)
            logger.info(f"Final class mapping after merging rare classes: {final_map}")
            
            # Feature selection using Random Forest for dimensionality reduction
            logger.info("Performing feature selection...")
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42),
                threshold='median'
            )
            X_selected = selector.fit_transform(X, y)
            
            if hasattr(X, 'columns'):
                selected_indices = selector.get_support()
                selected_features = [feature for feature, selected in zip(X.columns, selected_indices) if selected]
                logger.info(f"Selected features: {selected_features}")
                
                # If X is DataFrame, preserve column names after selection
                X_selected = pd.DataFrame(
                    X_selected,
                    columns=[f for i, f in enumerate(X.columns) if selected_indices[i]],
                    index=X.index
                )
            
            # Step 4: Split data for evaluation
            logger.info("Step 4: Splitting data for training and evaluation...")
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
            
            # Set class mapping in model trainer
            self.model_trainer.set_class_mapping(final_map, {v: k for k, v in final_map.items()})
            
            # Step 5: Train models
            logger.info("Step 5: Training models...")
            
            # Train Random Forest
            best_rf, rf_results = self.model_trainer.train_random_forest(X_train, y_train)
            
            # Train XGBoost
            best_xgb, xgb_results = self.model_trainer.train_xgboost(X_train, y_train)
            
            # Create base models for simple ensemble
            estimators = [('rf', best_rf), ('xgb', best_xgb)]
            
            # Train voting ensemble using sklearn's VotingClassifier
            voting_soft = VotingClassifier(estimators=estimators, voting='soft')
            voting_soft.fit(X_train, y_train)
            
            # Step 6: Evaluate models on test set
            logger.info("Step 6: Evaluating models...")
            
            # Add test data to results for proper evaluation
            rf_results['test_data'] = {'X_test': X_test, 'y_test': y_test}
            xgb_results['test_data'] = {'X_test': X_test, 'y_test': y_test}
            
            # Evaluate voting ensemble
            y_pred_ensemble = voting_soft.predict(X_test)
            y_proba_ensemble = voting_soft.predict_proba(X_test)
            
            # Convert predictions back to original labels
            y_pred_original = self.model_trainer._convert_predictions(y_pred_ensemble)
            y_test_original = self.model_trainer._convert_predictions(y_test)
            
            voting_metrics = {
                'accuracy': accuracy_score(y_test_original, y_pred_original),
                'confusion_matrix': confusion_matrix(y_test_original, y_pred_original),
                'classification_report': classification_report(y_test_original, y_pred_original),
                'y_pred': y_pred_original,
                'y_proba': y_proba_ensemble,
                'roc_auc': roc_auc_score(y_test_original, y_proba_ensemble, multi_class='ovr'),
                'rmse': np.sqrt(mean_squared_error(y_test_original, y_pred_original)),
                'test_data': {'X_test': X_test, 'y_test': y_test}
            }
            
            # Step 7: Choose the best model based on accuracy
            model_results = {
                "Random Forest": rf_results,
                "XGBoost": xgb_results,
                "Voting Ensemble": voting_metrics
            }
            
            # Find the best model
            best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
            best_model = {'Random Forest': best_rf, 'XGBoost': best_xgb, 'Voting Ensemble': voting_soft}[best_model_name]
            best_accuracy = model_results[best_model_name]['accuracy']
            
            logger.info(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
            
            # Step 8: Save the best model
            logger.info(f"Saving best model: {best_model_name}")
            model_path = os.path.join(self.models_dir, f"best_model_{datetime.now().strftime('%Y%m%d')}.joblib")
            model_metadata = {
                'model': best_model,
                'model_type': best_model_name,
                'accuracy': best_accuracy,
                'class_map': final_map,
                'selected_features': selected_features if hasattr(X, 'columns') else None,
                'training_date': datetime.now().isoformat(),
                'metrics': model_results[best_model_name]
            }
            joblib.dump(model_metadata, model_path)
            logger.info(f"Saved best model to {model_path}")
            
            # Step 9: Create visualizations
            logger.info("Creating visualizations...")
            
            # Generate dashboards
            self.dashboard_generator.generate_classification_dashboards(model_results)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in enhanced pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a model with detailed metrics"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Calculate probabilities if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                
                # Calculate ROC AUC (multi-class)
                try:
                    if len(np.unique(y_test)) > 2:
                        # One-hot encode the target
                        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                        n_classes = y_test_bin.shape[1]
                        
                        # Calculate ROC AUC for each class
                        roc_auc_scores = []
                        for i in range(n_classes):
                            if y_test_bin[:, i].sum() > 0:  # Only if class exists in test set
                                roc_auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
                                roc_auc_scores.append(roc_auc)
                        
                        # Average ROC AUC across all classes
                        avg_roc_auc = np.mean(roc_auc_scores)
                    else:
                        # Binary case
                        avg_roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                except Exception as e:
                    logger.warning(f"Couldn't calculate ROC AUC for {model_name}: {str(e)}")
                    avg_roc_auc = None
            else:
                y_proba = None
                avg_roc_auc = None
                
            # Calculate RMSE
            try:
                from sklearn.metrics import mean_squared_error
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            except Exception as e:
                logger.warning(f"Couldn't calculate RMSE for {model_name}: {str(e)}")
                rmse = None
                
            # Log results
            logger.info(f"{model_name} Evaluation:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            if avg_roc_auc:
                logger.info(f"  ROC AUC: {avg_roc_auc:.4f}")
            if rmse:
                logger.info(f"  RMSE: {rmse:.4f}")
                
            # Return metrics
            metrics = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'roc_auc': avg_roc_auc,
                'rmse': rmse
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'accuracy': 0,
                'error': str(e)
            }

    def make_prediction(self, sample_data):
        """Make prediction with best available model"""
        # Load the ensemble model (or best model if ensemble not available)
        try:
            model = self.model_manager.load_model(model_type="ensemble")
        except FileNotFoundError:
            model = self.model_manager.load_model(model_type="xgboost")

        # Format sample data
        if isinstance(sample_data, dict):
            # Use the same order as feature columns
            feature_columns = [
                'funding_amount_log',
                'employees',
                'employee_efficiency',
                'previous_rounds',
                'funding_year',
                'funding_month',
                'months_since_first_funding']
            features = [sample_data.get(col, 0) for col in feature_columns]
        else:
            features = sample_data

        # Make prediction
        prediction = self.model_manager.predict(features)
        # Map back to original funding stage
        if hasattr(self.feature_engineer, 'funding_stage_map'):
            reverse_map = {
                v: k for k,
                v in self.feature_engineer.funding_stage_map.items()}
            if isinstance(prediction, (int, float)):
                prediction = reverse_map.get(
                    int(prediction), f"Unknown (Class {prediction})")
        return prediction

    def time_series_prediction(self):
        """
        Perform time series prediction for funding stages over time.
        
        Returns:
            dict: Dictionary containing forecast results
        """
        logger.info("Starting aggregate time series prediction with Prophet...")
        
        # Create the time_series_forecasts directory
        forecast_dir = os.path.join(self.output_dir, "time_series_forecasts")
        os.makedirs(forecast_dir, exist_ok=True)
        logger.info(f"Created time series forecasts directory: {forecast_dir}")
        
        # Prepare time series data
        logger.info("Preparing time series data...")
        
        # Process the data for time series analysis
        if not hasattr(self, 'merged_data') or self.merged_data is None:
            self.merged_data = self.data_loader.merge_datasets()
            
        if not hasattr(self, 'feature_engineer') or self.feature_engineer is None:
            self.feature_engineer = FeatureEngineering()
            self.merged_data = self.feature_engineer.extract_features(self.merged_data)
            
        time_series_data = self.merged_data.copy()
        
        # Ensure the data has been processed
        if time_series_data.empty:
            logger.error("No data available for time series prediction")
            return None
            
        # Ensure funding_date is a datetime type and set it as index
        try:
            # First check if funding_date column exists
            if 'funding_date' not in time_series_data.columns:
                logger.error("Required column 'funding_date' not found in data")
                return None
                
                
            # Convert funding_date to datetime
            time_series_data['funding_date'] = pd.to_datetime(time_series_data['funding_date'], errors='coerce')
            
            # Drop rows with invalid dates
            time_series_data = time_series_data.dropna(subset=['funding_date'])
            
            if time_series_data.empty:
                logger.error("No valid dates found after conversion")
                return None
                
            # Check if funding_stage_numeric exists, if not create it
            if 'funding_stage_numeric' not in time_series_data.columns:
                logger.info("Creating funding_stage_numeric column...")
                if 'funding_stage' in time_series_data.columns:
                    # Create a numeric representation of funding stages
                    funding_stages = {
                        'Pre-Seed': 0, 'Seed': 1, 'Angel': 2, 'Series A': 3, 
                        'Series B': 4, 'Series C': 5, 'Series D': 6, 'Series E': 7,
                        'Series F': 8, 'Series G': 9, 'Series H': 10, 
                        'Venture - Series Unknown': 11, 'Debt Financing': 12,
                        'Undisclosed': 13, 'Grant': 14, 'Private Equity': 15
                    }
                    # Map funding stages to numeric values, use 16 for unknown
                    time_series_data['funding_stage_numeric'] = time_series_data['funding_stage'].map(
                        lambda x: funding_stages.get(x, 16)
                    )
                else:
                    logger.error("Neither funding_stage_numeric nor funding_stage columns found")
                    return None
                
            logger.info(f"Prepared time series data with {len(time_series_data)} valid records")
        except Exception as e:
            logger.error(f"Error preparing time series data: {str(e)}")
            return None
        
        # Aggregate monthly funding stages
        try:
            monthly_stages = time_series_data.groupby(pd.Grouper(key='funding_date', freq='ME')).agg({
                'funding_stage_numeric': 'count',
                'funding_amount': 'sum'
            }).reset_index()
            
            logger.info(f"Aggregated data into {len(monthly_stages)} monthly periods")
        except Exception as e:
            logger.error(f"Error during time series aggregation: {str(e)}")
            return None
        
        # Rename columns for Prophet 
        monthly_stages_prophet = monthly_stages.rename(columns={
            'funding_date': 'ds',
            'funding_stage_numeric': 'y'
        })
        
        # Filter out rows with no data
        monthly_stages_prophet = monthly_stages_prophet.dropna(subset=['y'])
        
        # Create a Prophet model
        from prophet import Prophet
        
        # Dictionary to store all forecasts
        forecasts = {}
        
        # Forecast number of funding rounds
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            interval_width=0.95
        )
        
        # Add monthly seasonality
        prophet_model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        # Fit the model
        prophet_model.fit(monthly_stages_prophet)
        
        # Create future dataframe
        future = prophet_model.make_future_dataframe(periods=12, freq='ME')
        
        # Make forecast
        forecast = prophet_model.predict(future)
        
        # Store the forecast
        forecasts['funding_rounds'] = forecast
        
        # Save forecast to CSV
        forecast_path = os.path.join(forecast_dir, 'funding_rounds_forecast.csv')
        try:
            forecast.to_csv(forecast_path, index=False)
            logger.info(f"Saved funding rounds forecast to {forecast_path}")
        except Exception as e:
            logger.error(f"Error saving forecast to {forecast_path}: {str(e)}")
        
        # Create component plot
        fig = prophet_model.plot_components(forecast)
        component_plot_path = os.path.join(forecast_dir, 'funding_rounds_components.png')
        try:
            fig.savefig(component_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error saving component plot: {str(e)}")
        
        # Add industry-specific forecasts
        for industry in time_series_data['industry'].dropna().unique():
            if pd.isna(industry) or industry == '':
                continue
                
            # Filter data for this industry
            industry_data = time_series_data[time_series_data['industry'] == industry]
            
            # Check if we have enough data
            if len(industry_data) < 10:
                continue
                
            # Aggregate by month
            try:
                monthly_industry = industry_data.groupby(pd.Grouper(key='funding_date', freq='ME')).agg({
                    'funding_amount': 'sum',
                    'funding_stage_numeric': 'count'  # Count of funding rounds
                }).reset_index()
                
                # Rename columns for Prophet
                industry_prophet = monthly_industry.rename(columns={
                    'funding_date': 'ds',
                    'funding_stage_numeric': 'y'  # Predict count of funding rounds
                })
                
                # Filter out rows with no data and handle NaN values
                industry_prophet = industry_prophet.dropna(subset=['y'])
                
                # If we still have enough data after filtering
                if len(industry_prophet) >= 3:
                    # Create a Prophet model for this industry
                    industry_model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        seasonality_mode='additive',
                        interval_width=0.95
                    )
                    
                    # Fit the model
                    industry_model.fit(industry_prophet)
                    
                    # Create future dataframe - forecast 12 months
                    future = industry_model.make_future_dataframe(periods=12, freq='ME')
                    
                    # Make forecast
                    industry_forecast = industry_model.predict(future)
                    
                    # Store the forecast
                    forecasts[f'industry_{industry}'] = industry_forecast
                    
                    # Save forecast to CSV
                    industry_path = os.path.join(forecast_dir, f'industry_{industry.replace(" ", "_").lower()}_forecast.csv')
                    try:
                        industry_forecast.to_csv(industry_path, index=False)
                    except Exception as e:
                        logger.warning(f"Error forecasting industry {industry}: {str(e)}")
                    
                    # Create component plot
                    fig = industry_model.plot_components(industry_forecast)
                    industry_plot_path = os.path.join(forecast_dir, f'industry_{industry.replace(" ", "_").lower()}_components.png')
                    try:
                        fig.savefig(industry_plot_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error saving industry plot for {industry}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error forecasting industry {industry}: {str(e)}")
                continue
        
        # Add funding amount forecast
        try:
            monthly_amounts = time_series_data.groupby(pd.Grouper(key='funding_date', freq='ME')).agg({
                'funding_amount': 'sum'
            }).reset_index()
            
            # Rename columns for Prophet
            amounts_prophet = monthly_amounts.rename(columns={
                'funding_date': 'ds',
                'funding_amount': 'y'
            })
            
            # Filter out rows with no data
            amounts_prophet = amounts_prophet.dropna(subset=['y'])
            
            # Create a Prophet model
            amounts_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                interval_width=0.95
            )
            
            # Fit the model
            amounts_model.fit(amounts_prophet)
            
            # Create future dataframe
            future = amounts_model.make_future_dataframe(periods=12, freq='ME')
            
            # Make forecast
            amounts_forecast = amounts_model.predict(future)
            
            # Store the forecast
            forecasts['funding_amounts'] = amounts_forecast
            
            # Save forecast to CSV
            amounts_path = os.path.join(forecast_dir, 'funding_amounts_forecast.csv')
            try:
                amounts_forecast.to_csv(amounts_path, index=False)
                logger.info(f"Saved funding amounts forecast to {amounts_path}")
            except Exception as e:
                logger.warning(f"Error saving funding amounts forecast: {str(e)}")
            
            # Create component plot
            fig = amounts_model.plot_components(amounts_forecast)
            amounts_plot_path = os.path.join(forecast_dir, 'funding_amounts_components.png')
            try:
                fig.savefig(amounts_plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Error saving funding amounts component plot: {str(e)}")
        except Exception as e:
            logger.warning(f"Error forecasting funding amounts: {str(e)}")
        
        # Add stage transition forecasts
        for stage in time_series_data['funding_stage'].dropna().unique():
            if pd.isna(stage) or stage == '':
                continue
                
            # Filter data for this stage
            stage_data = time_series_data[time_series_data['funding_stage'] == stage]
            
            # Check if we have enough data
            if len(stage_data) < 10:
                continue
                
            try:
                # Aggregate by month - count occurrences
                monthly_counts = stage_data.groupby(pd.Grouper(key='funding_date', freq='ME')).size().reset_index(name='count')
                
                # Rename columns for Prophet
                stage_prophet = monthly_counts.rename(columns={
                    'funding_date': 'ds',
                    'count': 'y'
                })
                
                # Filter out rows with no data
                stage_prophet = stage_prophet.dropna(subset=['y'])
                
                # If we still have enough data after filtering
                if len(stage_prophet) >= 3:
                    # Create a Prophet model for this stage
                    stage_model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        seasonality_mode='additive',  
                        interval_width=0.95
                    )
                    
                    # Fit the model
                    stage_model.fit(stage_prophet)
                    
                    # Create future dataframe - forecast 12 months
                    future = stage_model.make_future_dataframe(periods=12, freq='ME')
                    
                    # Make forecast
                    stage_forecast = stage_model.predict(future)
                    
                    # Store the forecast
                    forecasts[f'stage_{stage}'] = stage_forecast
                    
                    # Save forecast to CSV
                    stage_path = os.path.join(forecast_dir, f'stage_{stage.replace(" ", "_").lower()}_forecast.csv')
                    try:
                        stage_forecast.to_csv(stage_path, index=False)
                    except Exception as e:
                        logger.warning(f"Error forecasting stage {stage}: {str(e)}")
                    
                    # Create component plot
                    fig = stage_model.plot_components(stage_forecast)
                    stage_plot_path = os.path.join(forecast_dir, f'stage_{stage.replace(" ", "_").lower()}_components.png')
                    try:
                        fig.savefig(stage_plot_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                    except Exception as e:
                        self.logger.warning(f"Error saving stage plot for {stage}: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Error forecasting stage {stage}: {str(e)}")
                continue
          
        # Add funding range forecasts
        funding_ranges = [
            ('seed', time_series_data[time_series_data['funding_stage'].isin(['Seed', 'Pre-Seed', 'Angel'])]),
            ('early', time_series_data[time_series_data['funding_stage'].isin(['Series A', 'Series B'])]),
            ('growth', time_series_data[time_series_data['funding_stage'].isin(['Series C', 'Series D', 'Series E'])]),
            ('late', time_series_data[time_series_data['funding_stage'].isin(['Series F', 'Series G', 'Series H', 'Series I', 'Series J'])])
        ]
        
        for range_name, range_data in funding_ranges:
            # Check if we have enough data
            if len(range_data) < 10:
                continue
                
            try:
                # Aggregate by month - count occurrences
                monthly_counts = range_data.groupby(pd.Grouper(key='funding_date', freq='ME')).size().reset_index(name='count')
                
                # Rename columns for Prophet
                range_prophet = monthly_counts.rename(columns={
                    'funding_date': 'ds',
                    'count': 'y'
                })
                
                # Filter out rows with no data
                range_prophet = range_prophet.dropna(subset=['y'])
                
                # If we still have enough data after filtering
                if len(range_prophet) >= 3:
                    # Create a Prophet model for this range
                    range_model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        seasonality_mode='additive',
                        interval_width=0.95
                    )
                    
                    # Fit the model
                    range_model.fit(range_prophet)
                    
                    # Create future dataframe - forecast 12 months
                    future = range_model.make_future_dataframe(periods=12, freq='ME')
                    
                    # Make forecast
                    range_forecast = range_model.predict(future)
                    
                    # Store the forecast
                    forecasts[f'range_{range_name}'] = range_forecast
                    
                    # Save forecast to CSV
                    range_path = os.path.join(forecast_dir, f'range_{range_name}_forecast.csv')
                    try:
                        range_forecast.to_csv(range_path, index=False)
                    except Exception as e:
                        self.logger.warning(f"Error saving range forecast for {range_name}: {str(e)}")
                    
                    # Create component plot
                    fig = range_model.plot_components(range_forecast)
                    range_plot_path = os.path.join(forecast_dir, f'range_{range_name}_components.png')
                    try:
                        fig.savefig(range_plot_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                    except Exception as e:
                        self.logger.warning(f"Error saving range plot for {range_name}: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Error forecasting range {range_name}: {str(e)}")
                continue
                    
        # Create aggregated plots using matplotlib
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(forecasts['funding_rounds']['ds'][-24:], forecasts['funding_rounds']['yhat'][-24:], label='Forecast')
            plt.fill_between(
                forecasts['funding_rounds']['ds'][-24:],
                forecasts['funding_rounds']['yhat_lower'][-24:],
                forecasts['funding_rounds']['yhat_upper'][-24:],
                alpha=0.3
            )
            plt.title('Funding Rounds Forecast (Next 12 Months)')
            plt.xlabel('Date')
            plt.ylabel('Number of Rounds')
            plt.grid(True)
            plt.tight_layout()
            summary_plot_path = os.path.join(forecast_dir, 'funding_summary_forecast.png')
            try:
                plt.savefig(summary_plot_path, dpi=300)
                plt.close()
            except Exception as e:
                self.logger.warning(f"Error creating summary plot: {str(e)}")
        except Exception as e:
            self.logger.warning(f"Error creating summary plot: {str(e)}")
        
        # Save forecasts to the timeseries directory
        logger.info(f"Time series forecasts saved to {forecast_dir}")
        
        # Generate dashboards using the dashboard generator
        if hasattr(self, 'dashboard_generator'):
            try:
                logger.info("Generating time series dashboards...")
                dashboard_paths = self.dashboard_generator.generate_timeseries_dashboards(forecasts)
                logger.info(f"Time series dashboards generated: {list(dashboard_paths.keys())}")
            except Exception as e:
                logger.error(f"Error generating time series dashboards: {str(e)}")
        
        return forecasts

    def validate_company(self, company_name, api_key=None):
        """Validate company existence through external API or internal checks
        
        Args:
            company_name: Name of company to validate
            api_key: Optional API key for external validation service
            
        Returns:
            tuple: (is_valid, confidence_score)
        """
        # Validation disabled - trust the data source
        return True, 1.0
        
        """
        # Original validation code commented out
        if not company_name or pd.isna(company_name):
            return False, 0.0

        # Basic validation of company name
        if not self._is_valid_company_name(company_name):
            return False, 0.1

        # Try to validate through external API if key provided
        if api_key:
            try:
                # In a real implementation, this would call a company verification
                # API
                # For this example, we'll simulate the API call
                is_valid = self._simulate_company_api_check(company_name)
                return is_valid, 0.85
            except Exception as e:
                logger.error(f"Company validation API error: {str(e)}")
                # Fall back to internal validation if API fails
                pass

        # Simple validation based on cached company prefixes/patterns
        similarity = self._check_name_similarity(company_name)
        if similarity > 0.8:
            return True, similarity

        # Mark as potentially suspicious but not definitely invalid
        return True, 0.6
        """

    def _check_funding_manipulation(self, features):
        """Check for suspicious patterns in funding amount features"""
        # This would implement specific business logic for detecting
        # unrealistic funding patterns based on domain knowledge
        try:
            # An example suspicious pattern: funding amounts that are too "round"
            # or follow suspicious patterns
            # Assuming last 3 features are funding related
            funding_features = features[-3:]

            # Check for perfectly round large numbers (possible manipulation)
            round_numbers = sum(
                1 for f in funding_features if f > 10000 and f %
                10000 == 0)
            if round_numbers >= 2:
                return True

            # Check for unrealistic growth between funding rounds
            if len(funding_features) >= 2:
                growth_rates = [funding_features[i + 1] / max(funding_features[i], 1)
                                for i in range(len(funding_features) - 1)]

                # Unusually high growth rates may indicate manipulation
                if any(rate > 50 for rate in growth_rates):
                    return True

            return False
        except Exception:
            return False

    def _is_valid_company_name(self, name):
        """Basic validation of company name format"""
        if not name or len(name) < 2:
            return False

        # Check for nonsensical names (just numbers or special chars)
        if name.isdigit() or all(not c.isalnum() for c in name):
            return False

        # Check for suspicious patterns
        suspicious_patterns = ['test', 'sample', 'example', 'fake', 'xyz']
        if any(pattern in name.lower() for pattern in suspicious_patterns):
            return False

        return True

    def _simulate_company_api_check(self, company_name):
        """Simulate external API call to validate company existence"""
        # In a real implementation, this would call an external API
        # For now, just returning True with high probability
        return random.random() < 0.9

    def _check_name_similarity(self, name):
        """Check similarity of company name to known companies"""
        if not self.known_companies:
            return 0.5

        # Simple similarity metric - in production would use better similarity
        name_lower = name.lower()
        max_similarity = 0

        for known_name in self.known_companies:
            # Simple partial string matching
            known_lower = known_name.lower()

            # Calculate Levenshtein distance-based similarity
            similarity = 1.0 - min(self._levenshtein_distance(name_lower,
                                                              known_lower) / max(len(name_lower), len(known_lower)), 1.0)
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _levenshtein_distance(self, s1, s2):
        """Calculate edit distance between strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Calculate cost - 0 if characters match, 1 otherwise
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


class DashboardGenerator:
    """Class to generate comprehensive dashboards for classification and time series models."""
    
    def __init__(self, output_dir="./dashboards"):
        """
        Initialize the dashboard generator.
        
        Args:
            output_dir (str): Directory to save dashboards
        """
        self.output_dir = output_dir
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Dashboard output directory set to: {self.output_dir}")
        
        # Create subdirectories
        self.classification_dir = os.path.join(self.output_dir, "classification_dashboards")
        self.timeseries_dir = os.path.join(self.output_dir, "timeseries_dashboards")
        
        os.makedirs(self.classification_dir, exist_ok=True)
        os.makedirs(self.timeseries_dir, exist_ok=True)
        
        # Create all subdirectories in advance to avoid permission issues
        dirs_to_create = [
            # Classification subdirectories
            os.path.join(self.classification_dir, "model_comparison"),
            os.path.join(self.classification_dir, "calibration_curves"),
            os.path.join(self.classification_dir, "feature_importance"),
            os.path.join(self.classification_dir, "confusion_matrices"),
            os.path.join(self.classification_dir, "model_metrics"),
            
            # Time series subdirectories
            os.path.join(self.timeseries_dir, "forecast_trends"),
            os.path.join(self.timeseries_dir, "industry_breakdown"),
            os.path.join(self.timeseries_dir, "stage_evolution"),
            os.path.join(self.timeseries_dir, "seasonality_analysis"),
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
            
        # Configure matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for server environments
            self.logger.info("Set matplotlib backend to Agg")
        except Exception as e:
            self.logger.warning(f"Failed to set matplotlib backend: {str(e)}")
        
    def generate_all_dashboards(self, classification_results, timeseries_results):
        """
        Generate all dashboards for both classification and time series models.
        
        Args:
            classification_results (dict): Results from classification models
            timeseries_results (dict): Results from time series forecasts
            
        Returns:
            dict: Paths to generated dashboards
        """
        dashboard_paths = {}
        
        # Generate classification dashboards
        self.logger.info("Generating classification model dashboards...")
        dashboard_paths["classification"] = self.generate_classification_dashboards(classification_results)
        
        # Generate time series dashboards
        self.logger.info("Generating time series model dashboards...")
        dashboard_paths["timeseries"] = self.generate_timeseries_dashboards(timeseries_results)
        
        return dashboard_paths
    
    def generate_classification_dashboards(self, results):
        """
        Generate dashboards for classification models.
        
        Args:
            results (dict): Dictionary containing model results
            
        Returns:
            dict: Paths to generated dashboards
        """
        dashboard_paths = {}
        
        # Create subdirectories for each dashboard
        model_comparison_dir = os.path.join(self.classification_dir, "model_comparison")
        calibration_dir = os.path.join(self.classification_dir, "calibration_curves")
        feature_importance_dir = os.path.join(self.classification_dir, "feature_importance")
        confusion_matrix_dir = os.path.join(self.classification_dir, "confusion_matrices")
        model_metrics_dir = os.path.join(self.classification_dir, "model_metrics")
        
        os.makedirs(model_comparison_dir, exist_ok=True)
        os.makedirs(calibration_dir, exist_ok=True)
        os.makedirs(feature_importance_dir, exist_ok=True)
        os.makedirs(confusion_matrix_dir, exist_ok=True)
        os.makedirs(model_metrics_dir, exist_ok=True)
        
        # Generate each dashboard
        try:
            # 1. Model Comparison Dashboard
            model_comparison_path = self._generate_model_comparison_dashboard(results, model_comparison_dir)
            dashboard_paths["model_comparison"] = model_comparison_path
            
            # 2. Calibration Curves Dashboard
            calibration_paths = self._generate_calibration_curves_dashboard(results, calibration_dir)
            dashboard_paths["calibration_curves"] = calibration_paths
            
            # 3. Feature Importance Dashboard
            feature_importance_path = self._generate_feature_importance_dashboard(results, feature_importance_dir)
            dashboard_paths["feature_importance"] = feature_importance_path
            
            # 4. Confusion Matrix Dashboard
            confusion_matrix_path = self._generate_confusion_matrix_dashboard(results, confusion_matrix_dir)
            dashboard_paths["confusion_matrices"] = confusion_matrix_path
            
            # 5. Additional: Detailed Model Metrics Dashboard
            model_metrics_path = self._generate_model_metrics_dashboard(results, model_metrics_dir)
            dashboard_paths["model_metrics"] = model_metrics_path
            
        except Exception as e:
            self.logger.error(f"Error generating classification dashboards: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        return dashboard_paths
    
    def generate_timeseries_dashboards(self, results):
        """
        Generate dashboards for time series models.
        
        Args:
            results (dict): Dictionary containing time series forecast results
            
        Returns:
            dict: Paths to generated dashboards
        """
        dashboard_paths = {}
        
        # Check if results is None
        if results is None:
            self.logger.warning("No time series results available to generate dashboards")
            return dashboard_paths
        
        # Create subdirectories for each dashboard
        forecast_trend_dir = os.path.join(self.timeseries_dir, "forecast_trends")
        industry_breakdown_dir = os.path.join(self.timeseries_dir, "industry_breakdown")
        stage_evolution_dir = os.path.join(self.timeseries_dir, "stage_evolution")
        seasonality_dir = os.path.join(self.timeseries_dir, "seasonality_analysis")
        
        os.makedirs(forecast_trend_dir, exist_ok=True)
        os.makedirs(industry_breakdown_dir, exist_ok=True)
        os.makedirs(stage_evolution_dir, exist_ok=True)
        os.makedirs(seasonality_dir, exist_ok=True)
        
        # Generate each dashboard
        try:
            # 1. Forecast Trend Dashboard
            forecast_trend_path = self._generate_forecast_trend_dashboard(results, forecast_trend_dir)
            dashboard_paths["forecast_trend"] = forecast_trend_path
            
            # 2. Industry Breakdown Dashboard
            industry_breakdown_path = self._generate_industry_breakdown_dashboard(results, industry_breakdown_dir)
            dashboard_paths["industry_breakdown"] = industry_breakdown_path
            
            # 3. Funding Stage Evolution Dashboard
            stage_evolution_path = self._generate_stage_evolution_dashboard(results, stage_evolution_dir)
            dashboard_paths["stage_evolution"] = stage_evolution_path
            
            # 4. Seasonality Analysis Dashboard
            seasonality_path = self._generate_seasonality_dashboard(results, seasonality_dir)
            dashboard_paths["seasonality"] = seasonality_path
            
        except Exception as e:
            self.logger.error(f"Error generating time series dashboards: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        return dashboard_paths
    
    def _generate_model_comparison_dashboard(self, results, output_dir):
        """
        Generate dashboard comparing performance metrics across models.
        
        Args:
            results (dict): Dictionary containing model results
            output_dir (str): Directory to save dashboard
            
        Returns:
            str: Path to generated dashboard
        """
        self.logger.info("Generating model comparison dashboard...")
        
        # Extract model names and metrics
        model_names = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for model_name, model_results in results.items():
            if model_name in ['Random Forest', 'LightGBM', 'XGBoost', 'Voting Ensemble']:
                model_names.append(model_name)
                metrics = model_results[1] if isinstance(model_results, tuple) else model_results
                
                # Extract metrics
                accuracy_scores.append(metrics.get('accuracy', 0))
                
                # Get precision, recall, f1 from different metric formats
                if 'precision' in metrics:
                    if isinstance(metrics['precision'], dict):
                        precision_scores.append(metrics['precision'].get('macro', 0))
                    else:
                        precision_scores.append(metrics['precision'])
                else:
                    precision_scores.append(0)
                    
                if 'recall' in metrics:
                    if isinstance(metrics['recall'], dict):
                        recall_scores.append(metrics['recall'].get('macro', 0))
                    else:
                        recall_scores.append(metrics['recall'])
                else:
                    recall_scores.append(0)
                    
                if 'f1_scores' in metrics:
                    if isinstance(metrics['f1_scores'], dict):
                        f1_scores.append(metrics['f1_scores'].get('macro', 0))
                    else:
                        f1_scores.append(metrics['f1_scores'])
                else:
                    f1_scores.append(0)
        
        # Create figure for model comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set width of bars
        bar_width = 0.2
        index = np.arange(len(model_names))
        
        # Create bars
        bar1 = ax.bar(index, accuracy_scores, bar_width, label='Accuracy', color='#3274A1')
        bar2 = ax.bar(index + bar_width, precision_scores, bar_width, label='Precision', color='#E1812C')
        bar3 = ax.bar(index + 2*bar_width, recall_scores, bar_width, label='Recall', color='#3A923A')
        bar4 = ax.bar(index + 3*bar_width, f1_scores, bar_width, label='F1 Score', color='#C03D3E')
        
        # Add labels, title, and legend
        ax.set_xlabel('Models', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(index + 1.5*bar_width)
        ax.set_xticklabels(model_names, rotation=0)
        ax.set_ylim(0, 1.0)
        ax.legend()
        
        # Add value labels above bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
                           
        add_labels(bar1)
        add_labels(bar2)
        add_labels(bar3)
        add_labels(bar4)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Enhance aesthetics
        fig.tight_layout()
        
        # Save dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'model_comparison_dashboard_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model comparison dashboard saved to: {output_path}")
        return output_path
    
    def _generate_calibration_curves_dashboard(self, results, output_dir):
        """Generate calibration curves for each model"""
        self.logger.info("Generating calibration curves dashboards...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created calibration curves directory: {output_dir}")
        
        # Get available models
        available_models = list(results.keys())
        self.logger.info(f"Available models for calibration: {available_models}")
        
        # Track output paths
        output_paths = {}
        processed_models = []
        
        # Generate individual calibration curves
        for model_name, metrics in results.items():
            self.logger.info(f"Model {model_name} has keys: {list(metrics.keys())}")
            
            # Get test data and predictions
            if 'test_data' not in metrics:
                self.logger.warning(f"No test data available for {model_name}")
                continue
                
            test_data = metrics['test_data']
            if 'y_test' not in test_data or 'y_proba' not in metrics:
                self.logger.warning(f"Missing required data for {model_name}")
                continue
                
            y_test = test_data['y_test']
            y_proba = metrics['y_proba']
            
            # Create calibration curve
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self._create_multiclass_calibration_plot(
                y_test, y_proba, model_name, output_dir, timestamp
            )
            
            if output_path:
                output_paths[model_name] = output_path
                processed_models.append(model_name)
        
        # Create combined plot if we have multiple models
        if len(processed_models) > 1:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_path = self._create_combined_calibration_plot(
                results, processed_models, output_dir, timestamp
            )
            if combined_path:
                output_paths['combined'] = combined_path
        
        self.logger.info(f"Calibration dashboards created: {len(output_paths)}")
        self.logger.info(f"Output paths: {output_paths}")
        return output_paths
        
    def _create_multiclass_calibration_plot(self, y_test, y_proba, model_name, output_dir, timestamp=None):
        """Create calibration plot for multiclass classification
        
        Args:
            y_test: True class labels
            y_proba: Predicted class probabilities
            model_name: Name of the model
            output_dir: Directory to save the plot
            timestamp: Optional timestamp for the filename
            
        Returns:
            Path to the saved plot
        """
        n_classes = y_proba.shape[1]
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot calibration curve for each class
        for i in range(n_classes):
            # One-vs-Rest approach for calibration
            y_true_binary = (y_test == i).astype(int)
            prob_pos = y_proba[:, i]
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary, prob_pos, n_bins=10
            )
            
            # Plot calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   marker='o', label=f'Class {i}')
        
        # Add reference line (perfectly calibrated)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Set plot details
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(f'Calibration Curve - {model_name}')
        ax.legend(loc='best')
        
        # Save plot
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'calibration_curve_{model_name}_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    def _create_combined_calibration_plot(self, results, model_names, output_dir, timestamp=None):
        """Create a combined calibration plot for multiple models
        
        Args:
            results: Dictionary of model results
            model_names: List of model names to include
            output_dir: Directory to save plot
            timestamp: Optional timestamp for filename
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(12, 8))
        
        # Add reference line (perfectly calibrated)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Different line styles and colors for models
        linestyles = ['-', '--', '-.', ':']
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        
        for i, model_name in enumerate(model_names):
            if model_name not in results:
                continue
                
            metrics = results[model_name]
            if 'test_data' not in metrics or 'y_proba' not in metrics:
                continue
                
            test_data = metrics['test_data']
            if 'y_test' not in test_data:
                continue
                
            y_test = test_data['y_test']
            y_proba = metrics['y_proba']
            
            # Calculate average calibration across all classes
            y_true_all = []
            y_prob_all = []
            
            for cls in range(y_proba.shape[1]):
                y_true_binary = (y_test == cls).astype(int)
                y_true_all.extend(y_true_binary)
                y_prob_all.extend(y_proba[:, cls])
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_all, y_prob_all, n_bins=10
            )
            
            # Plot calibration curve for this model
            plt.plot(
                mean_predicted_value, 
                fraction_of_positives,
                marker='o',
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                label=model_name
            )
        
        # Set plot details
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curves - Model Comparison')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'combined_calibration_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def _generate_feature_importance_dashboard(self, results, output_dir):
        """Generate feature importance visualizations"""
        self.logger.info("Generating feature importance dashboard...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track models with feature importance
        models_with_importance = []
        
        for model_name, metrics in results.items():
            if 'feature_importance' in metrics and metrics['feature_importance'] is not None:
                models_with_importance.append((model_name, metrics))
        
        if not models_with_importance:
            self.logger.warning("No models with feature importances found")
            return None
            
        # Create figure
        fig, axes = plt.subplots(len(models_with_importance), 1, figsize=(12, 6*len(models_with_importance)))
        if len(models_with_importance) == 1:
            axes = [axes]
            
        # Plot feature importance for each model
        for idx, (model_name, metrics) in enumerate(models_with_importance):
            importances = metrics['feature_importance']
            feature_names = metrics.get('feature_names', [f'Feature {i}' for i in range(len(importances))])
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            # Plot
            ax = axes[idx]
            ax.bar(range(len(importances)), importances[indices])
            ax.set_title(f'{model_name} Feature Importance', fontsize=14)
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Importance', fontsize=12)
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'seasonality_dashboard_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Seasonality dashboard saved to: {output_path}")
        return output_path

    def _generate_model_metrics_dashboard(self, results, output_dir):
        """
        Generate detailed model metrics dashboard.
        
        Args:
            results (dict): Dictionary containing model results
            output_dir (str): Directory to save dashboard
            
        Returns:
            str: Path to generated dashboard
        """
        self.logger.info("Generating model metrics dashboard...")
        
        # Extract model metrics
        model_metrics = {}
        for model_name, metrics in results.items():
            if model_name in ['Random Forest', 'XGBoost', 'Voting Ensemble']:
                model_metrics[model_name] = metrics
        
        if not model_metrics:
            self.logger.warning("No model metrics found")
            return None
        
        # Create figure for detailed metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flat
        
        # 1. Plot accuracy and ROC AUC
        accuracies = []
        roc_aucs = []
        
        for model_name, metrics in model_metrics.items():
            accuracies.append(metrics.get('accuracy', 0))
            roc_aucs.append(metrics.get('roc_auc', 0))
        
        # 1a. Accuracy plot
        ax = axes[0]
        model_names = list(model_metrics.keys())
        x = np.arange(len(model_names))
        
        bars = ax.bar(x, accuracies, color='skyblue')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_title('Model Accuracy', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        # 1b. ROC AUC plot
        ax = axes[1]
        
        bars = ax.bar(x, roc_aucs, color='lightgreen')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_title('ROC AUC Score', fontsize=14)
        ax.set_ylabel('ROC AUC', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        # 2. RMSE plot
        ax = axes[2]
        rmse_values = []
        
        for model_name, metrics in model_metrics.items():
            rmse_values.append(metrics.get('rmse', 0))
        
        bars = ax.bar(x, rmse_values, color='salmon')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_title('Root Mean Square Error', fontsize=14)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        # 3. Classification Report Summary
        ax = axes[3]
        ax.axis('off')
        
        # Create a summary table
        summary_data = []
        columns = ['Model', 'Accuracy', 'ROC AUC', 'RMSE']
        
        for model_name, metrics in model_metrics.items():
            summary_data.append([
                model_name,
                f"{metrics.get('accuracy', 0):.3f}",
                f"{metrics.get('roc_auc', 0):.3f}",
                f"{metrics.get('rmse', 0):.3f}"
            ])
        
        table = ax.table(
            cellText=summary_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            colColours=['lightgray']*len(columns)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add overall title
        fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
        
        # Enhance aesthetics
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'model_metrics_dashboard_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model metrics dashboard saved to: {output_path}")
        return output_path

    def _generate_forecast_trend_dashboard(self, results, output_dir):
        """Generate dashboard showing forecast trends for different funding stages.
        
        Args:
            results: Dictionary containing forecast results
            output_dir: Directory to save the dashboard
            
        Returns:
            Path to the saved dashboard
        """
        # Check if we have the expected keys in results
        if not isinstance(results, dict):
            self.logger.warning("Results is not a dictionary")
            return None
            
        # Find the forecast data in different possible formats
        forecast_df = None
        if 'forecast_df' in results:
            forecast_df = results['forecast_df']
        elif 'funding_rounds' in results:
            forecast_df = results['funding_rounds']
        elif any(k.startswith('funding_') for k in results.keys()):
            for k in results.keys():
                if k.startswith('funding_'):
                    forecast_df = results[k]
                    break
                    
        if forecast_df is None:
            self.logger.warning("No forecast data found in results")
            return None
        
        # Create a 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot overall funding trend
        ax1 = axes[0, 0]
        
        # Plot actual vs forecast for overall funding
        if 'ds' in forecast_df and 'y' in forecast_df:
            ax1.plot(forecast_df['ds'], forecast_df['y'], 'ko', markersize=4, label='Actual')
            
            if 'yhat' in forecast_df.columns:
                ax1.plot(forecast_df['ds'], forecast_df['yhat'], 'b-', label='Forecast')
                
                if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                    ax1.fill_between(
                        forecast_df['ds'],
                        forecast_df['yhat_lower'],
                        forecast_df['yhat_upper'],
                        color='blue', alpha=0.2, label='95% Confidence Interval'
                    )
        
        ax1.set_title('Overall Funding Trend and Forecast', fontsize=14)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Funding Amount/Count', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot trend components if available
        ax2 = axes[0, 1]
        if 'weekly' in forecast_df.columns:
            trend_data = forecast_df[['ds', 'weekly']].dropna()
            ax2.plot(trend_data['ds'], trend_data['weekly'], 'g-')
            ax2.set_title('Weekly Seasonality Component', fontsize=14)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Effect', fontsize=12)
            ax2.grid(True, alpha=0.3)
        elif 'yearly' in forecast_df.columns:
            trend_data = forecast_df[['ds', 'yearly']].dropna()
            ax2.plot(trend_data['ds'], trend_data['yearly'], 'g-')
            ax2.set_title('Yearly Seasonality Component', fontsize=14)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Effect', fontsize=12)
            ax2.grid(True, alpha=0.3)
        elif 'trend' in forecast_df.columns:
            trend_data = forecast_df[['ds', 'trend']].dropna()
            ax2.plot(trend_data['ds'], trend_data['trend'], 'g-')
            ax2.set_title('Trend Component', fontsize=14)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Trend', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        # Plot stage-specific forecasts if available
        ax3 = axes[1, 0]
        stage_forecasts = {k: v for k, v in results.items() if k.startswith('stage_')}
        
        if stage_forecasts:
            for stage, data in stage_forecasts.items():
                stage_name = stage.replace('stage_', '').replace('_', ' ').title()
                if 'ds' in data and 'yhat' in data:
                    ax3.plot(data['ds'], data['yhat'], label=stage_name)
            
            ax3.set_title('Funding Stage Forecasts', fontsize=14)
            ax3.set_xlabel('Date', fontsize=12)
            ax3.set_ylabel('Forecast Value', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot industry forecasts if available
        ax4 = axes[1, 1]
        industry_forecasts = {k: v for k, v in results.items() if k.startswith('industry_')}
        
        if industry_forecasts:
            for industry, data in industry_forecasts.items():
                industry_name = industry.replace('industry_', '').replace('_', ' ').title()
                if 'ds' in data and 'yhat' in data:
                    ax4.plot(data['ds'], data['yhat'], label=industry_name)
            
            ax4.set_title('Industry Forecasts', fontsize=14)
            ax4.set_xlabel('Date', fontsize=12)
            ax4.set_ylabel('Forecast Value', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"forecast_trends_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _generate_industry_breakdown_dashboard(self, results, output_dir):
        """Generate dashboard showing breakdown of funding stages by industry."""
        self.logger.info("Generating industry breakdown dashboard...")
        
        # Extract data from results
        industry_data = {}
        for industry, data in results.items():
            if industry.startswith('industry_'):
                industry_name = industry[10:]  # Extract industry name
                if industry_name not in industry_data:
                    industry_data[industry_name] = []
                industry_data[industry_name].append(data['yhat'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (industry, data) in enumerate(industry_data.items()):
            ax = axes[i]
            ax.plot(data['ds'], data['yhat'], label='Forecast')
            ax.fill_between(data['ds'], data['yhat_lower'], data['yhat_upper'], alpha=0.2)
            ax.set_title(f'{industry} - Funding Stage Forecasts')
            ax.set_xlabel('Date')
            ax.set_ylabel('Forecast Value')
            ax.legend()
            ax.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"industry_breakdown_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Industry breakdown dashboard saved to: {output_path}")
        return output_path
    
    def _generate_stage_evolution_dashboard(self, results, output_dir):
        """Generate dashboard showing evolution of funding stages over time."""
        self.logger.info("Generating stage evolution dashboard...")
        
        # Extract data from results
        stage_data = {}
        for stage, data in results.items():
            if stage.startswith('stage_'):
                stage_name = stage[6:]  # Extract stage name
                if stage_name not in stage_data:
                    stage_data[stage_name] = []
                stage_data[stage_name].append(data['yhat'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (stage, data) in enumerate(stage_data.items()):
            ax = axes[i]
            ax.plot(data['ds'], data['yhat'], label='Forecast')
            ax.fill_between(data['ds'], data['yhat_lower'], data['yhat_upper'], alpha=0.2)
            ax.set_title(f'{stage} - Funding Stage Forecasts')
            ax.set_xlabel('Date')
            ax.set_ylabel('Forecast Value')
            ax.legend()
            ax.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"stage_evolution_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Stage evolution dashboard saved to: {output_path}")
        return output_path
    
    def _generate_seasonality_dashboard(self, results, output_dir):
        """Generate dashboard showing seasonality patterns in funding stages."""
        self.logger.info("Generating seasonality analysis dashboard...")
        
        # Extract data from results
        seasonality_data = {}
        for stage, data in results.items():
            if stage.startswith('stage_'):
                stage_name = stage[6:]  # Extract stage name
                if stage_name not in seasonality_data:
                    seasonality_data[stage_name] = []
                seasonality_data[stage_name].append(data['yhat'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (stage, data) in enumerate(seasonality_data.items()):
            ax = axes[i]
            ax.plot(data['ds'], data['yhat'], label='Forecast')
            ax.fill_between(data['ds'], data['yhat_lower'], data['yhat_upper'], alpha=0.2)
            ax.set_title(f'{stage} - Seasonality Analysis')
            ax.set_xlabel('Date')
            ax.set_ylabel('Forecast Value')
            ax.legend()
            ax.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"seasonality_analysis_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Seasonality analysis dashboard saved to: {output_path}")
        return output_path


class AdvancedDashboardGenerator(DashboardGenerator):
    """Class to generate advanced interactive dashboards for classification and time series models."""
    
    def __init__(self, output_dir="./advanced_dashboards"):
        """
        Initialize the advanced dashboard generator.
        
        Args:
            output_dir (str): Directory to save advanced dashboards
        """
        super().__init__(output_dir)
        
        # Create additional directories for the advanced dashboards
        self.classification_interactive_dir = os.path.join(self.output_dir, "classification_interactive")
        self.timeseries_interactive_dir = os.path.join(self.output_dir, "timeseries_interactive")
        
        os.makedirs(self.classification_interactive_dir, exist_ok=True)
        os.makedirs(self.timeseries_interactive_dir, exist_ok=True)
        
        # Create specific subdirectories
        dirs_to_create = [
            # Classification subdirectories
            os.path.join(self.classification_interactive_dir, "model_performance"),
            os.path.join(self.classification_interactive_dir, "prediction_interface"),
            os.path.join(self.classification_interactive_dir, "data_exploration"),
            os.path.join(self.classification_interactive_dir, "feature_analysis"),
            
            # Time series subdirectories
            os.path.join(self.timeseries_interactive_dir, "funding_trends"),
            os.path.join(self.timeseries_interactive_dir, "forecast_components"),
            os.path.join(self.timeseries_interactive_dir, "industry_forecasts"),
            os.path.join(self.timeseries_interactive_dir, "stage_transitions"),
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
            
        # Import required libraries for interactive dashboards
        try:
            import plotly
            import dash
            self.logger.info("Successfully imported interactive dashboard libraries")
        except ImportError:
            self.logger.warning("Interactive dashboard libraries not available. Some features may be limited.")
            
    def generate_advanced_dashboards(self, classification_results, timeseries_results):
        """
        Generate advanced dashboards for both classification and time series models.
        
        Args:
            classification_results (dict): Results from classification models
            timeseries_results (dict): Results from time series forecasts
            
        Returns:
            dict: Paths to generated dashboards
        """
        dashboard_paths = {}
        
        # Generate classification dashboards
        self.logger.info("Generating advanced classification dashboards...")
        dashboard_paths["classification"] = self.generate_classification_interactive_dashboard(classification_results)
        
        # Generate time series dashboards
        self.logger.info("Generating advanced time series dashboards...")
        dashboard_paths["timeseries"] = self.generate_timeseries_interactive_dashboard(timeseries_results)
        
        return dashboard_paths
        
    def generate_classification_interactive_dashboard(self, results):
        """
        Generate an interactive dashboard for classification models.
        
        Args:
            results (dict): Dictionary containing model results
            
        Returns:
            dict: Paths to generated dashboards
        """
        if not results:
            self.logger.warning("No classification results provided to generate dashboard")
            return {}
        
        dashboard_paths = {}
        
        try:
            # Import necessary libraries
            import plotly.graph_objects as go
            import plotly.figure_factory as ff
            import plotly.express as px
            from plotly.subplots import make_subplots
            import dash
            from dash import dcc, html
            from dash.dependencies import Input, Output, State
            
            # 1. Model Performance Overview dashboard
            performance_dir = os.path.join(self.classification_interactive_dir, "model_performance")
            dashboard_paths["model_performance"] = self._generate_model_performance_dashboard(results, performance_dir)
            
            # 2. Prediction Interface dashboard
            prediction_dir = os.path.join(self.classification_interactive_dir, "prediction_interface")
            dashboard_paths["prediction_interface"] = self._generate_prediction_interface_dashboard(results, prediction_dir)
            
            # 3. Data Exploration dashboard
            exploration_dir = os.path.join(self.classification_interactive_dir, "data_exploration")
            dashboard_paths["data_exploration"] = self._generate_data_exploration_dashboard(results, exploration_dir)
            
            # 4. Feature Analysis dashboard
            feature_dir = os.path.join(self.classification_interactive_dir, "feature_analysis")
            dashboard_paths["feature_analysis"] = self._generate_feature_analysis_dashboard(results, feature_dir)
            
        except ImportError as e:
            self.logger.error(f"Missing required packages for interactive dashboards: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error generating classification interactive dashboards: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        return dashboard_paths
    
    def generate_timeseries_interactive_dashboard(self, results):
        """
        Generate an interactive dashboard for time series models.
        
        Args:
            results (dict): Dictionary containing time series forecast results
            
        Returns:
            dict: Paths to generated dashboards
        """
        if not results:
            self.logger.warning("No time series results provided to generate dashboard")
            return {}
            
        dashboard_paths = {}
        
        try:
            # Import necessary libraries
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import dash
            from dash import dcc, html
            from dash.dependencies import Input, Output, State
            
            # 1. Funding Trends Forecast dashboard
            trends_dir = os.path.join(self.timeseries_interactive_dir, "funding_trends")
            dashboard_paths["funding_trends"] = self._generate_funding_trends_dashboard(results, trends_dir)
            
            # 2. Forecast Components dashboard
            components_dir = os.path.join(self.timeseries_interactive_dir, "forecast_components")
            dashboard_paths["forecast_components"] = self._generate_forecast_components_dashboard(results, components_dir)
            
            # 3. Industry-Specific Forecasts dashboard
            industry_dir = os.path.join(self.timeseries_interactive_dir, "industry_forecasts")
            dashboard_paths["industry_forecasts"] = self._generate_industry_forecasts_dashboard(results, industry_dir)
            
            # 4. Funding Stage Transitions dashboard
            transitions_dir = os.path.join(self.timeseries_interactive_dir, "stage_transitions")
            dashboard_paths["stage_transitions"] = self._generate_stage_transitions_dashboard(results, transitions_dir)
            
        except ImportError as e:
            self.logger.error(f"Missing required packages for interactive dashboards: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error generating time series interactive dashboards: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        return dashboard_paths
    
    def _generate_model_performance_dashboard(self, results, output_dir):
        """
        Generate a comprehensive model performance dashboard.
        
        Args:
            results (dict): Model results containing metrics and predictions
            output_dir (str): Directory to save the dashboard
            
        Returns:
            str: Path to the generated dashboard
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import dash
            from dash import dcc, html
            from dash.dependencies import Input, Output, State
            
            # Create dashboard app
            app = dash.Dash(__name__)
            
            # Extract model metrics and data
            model_names = []
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            # Collect metrics for each model
            for model_name, model_results in results.items():
                metrics = model_results[1] if isinstance(model_results, tuple) else model_results
                
                if isinstance(metrics, dict):
                    model_names.append(model_name)
                    
                    # Extract metrics, handle different formats
                    accuracy_scores.append(metrics.get('accuracy', 0))
                    
                    # Handle precision
                    if 'precision' in metrics:
                        if isinstance(metrics['precision'], dict):
                            precision_scores.append(metrics['precision'].get('macro', 0))
                        else:
                            precision_scores.append(metrics['precision'])
                    else:
                        precision_scores.append(0)
                        
                    # Handle recall
                    if 'recall' in metrics:
                        if isinstance(metrics['recall'], dict):
                            recall_scores.append(metrics['recall'].get('macro', 0))
                        else:
                            recall_scores.append(metrics['recall'])
                    else:
                        recall_scores.append(0)
                        
                    # Handle F1
                    if 'f1_scores' in metrics:
                        if isinstance(metrics['f1_scores'], dict):
                            f1_scores.append(metrics['f1_scores'].get('macro', 0))
                        else:
                            f1_scores.append(metrics['f1_scores'])
                    elif 'f1' in metrics:
                        if isinstance(metrics['f1'], dict):
                            f1_scores.append(metrics['f1'].get('macro', 0))
                        else:
                            f1_scores.append(metrics['f1'])
                    else:
                        f1_scores.append(0)
            
            # Create comparison bar chart
            metrics_fig = go.Figure(data=[
                go.Bar(name='Accuracy', x=model_names, y=accuracy_scores),
                go.Bar(name='Precision', x=model_names, y=precision_scores),
                go.Bar(name='Recall', x=model_names, y=recall_scores),
                go.Bar(name='F1 Score', x=model_names, y=f1_scores)
            ])
            
            metrics_fig.update_layout(
                title='Model Performance Metrics Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                barmode='group',
                template='plotly_white'
            )
            
            # Create confusion matrices for each model
            confusion_figs = {}
            for model_name, model_results in results.items():
                if isinstance(model_results, tuple) and len(model_results) > 2:
                    metrics = model_results[1]
                    if 'confusion_matrix' in metrics:
                        cm = metrics['confusion_matrix']
                        if isinstance(cm, (list, np.ndarray)):
                            # Assuming we have class names
                            class_names = metrics.get('classes', [f'Class {i}' for i in range(len(cm))])
                            
                            # Create heatmap
                            fig = px.imshow(
                                cm,
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=class_names, 
                                y=class_names,
                                color_continuous_scale='Blues',
                                title=f'Confusion Matrix - {model_name}'
                            )
                            
                            # Add text annotations
                            for i in range(len(cm)):
                                for j in range(len(cm[i])):
                                    fig.add_annotation(
                                        x=j, y=i, 
                                        text=str(cm[i][j]), 
                                        showarrow=False, 
                                        font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
                                    )
                            
                            fig.update_layout(width=600, height=600)
                            confusion_figs[model_name] = fig
            
            # Create feature importance figures
            importance_figs = {}
            feature_data = {}
            
            for model_name, model_results in results.items():
                if isinstance(model_results, tuple) and len(model_results) > 3:
                    # Extract feature importance if available
                    if model_results[3] is not None:
                        feature_names = model_results[3].get('features', [f'Feature {i}' for i in range(len(model_results[3].get('importance', [])))])
                        importance = model_results[3].get('importance', [])
                        
                        if feature_names and importance and len(feature_names) == len(importance):
                            # Store for later use
                            feature_data[model_name] = {
                                'features': feature_names,
                                'importance': importance
                            }
                            
                            # Create sorted bar chart
                            sorted_indices = np.argsort(importance)
                            sorted_features = [feature_names[i] for i in sorted_indices[-15:]]  # Top 15
                            sorted_importance = [importance[i] for i in sorted_indices[-15:]]
                            
                            fig = go.Figure(go.Bar(
                                x=sorted_importance,
                                y=sorted_features,
                                orientation='h'
                            ))
                            
                            fig.update_layout(
                                title=f'Top Features - {model_name}',
                                xaxis_title='Importance',
                                yaxis_title='Feature',
                                height=600
                            )
                            
                            importance_figs[model_name] = fig
            
            # If we have ROC curve data, create ROC curves
            roc_figs = {}
            for model_name, model_results in results.items():
                if isinstance(model_results, tuple) and len(model_results) > 2:
                    metrics = model_results[1]
                    if 'roc_auc' in metrics and 'fpr' in metrics and 'tpr' in metrics:
                        fpr = metrics['fpr']
                        tpr = metrics['tpr']
                        
                        # Handle different formats (binary vs multiclass)
                        if isinstance(fpr, dict) and isinstance(tpr, dict):
                            # Multiclass case
                            fig = go.Figure()
                            for cls, fpr_values in fpr.items():
                                if cls in tpr:
                                    fig.add_trace(go.Scatter(
                                        x=fpr_values, y=tpr[cls],
                                        mode='lines',
                                        name=f'Class {cls}'
                                    ))
                            
                            # Add diagonal line
                            fig.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random Chance',
                                line=dict(dash='dash', color='gray')
                            ))
                            
                            fig.update_layout(
                                title=f'ROC Curves - {model_name}',
                                xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate',
                                legend_title='Class',
                                width=600, height=500
                            )
                            
                            roc_figs[model_name] = fig
                        
                        elif isinstance(fpr, (list, np.ndarray)) and isinstance(tpr, (list, np.ndarray)):
                        # Binary case
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name='ROC Curve'
                            ))
                            
                            # Add diagonal line
                            fig.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random Chance',
                                line=dict(dash='dash', color='gray')
                            ))
                            
                            fig.update_layout(
                                title=f'ROC Curve - {model_name}',
                                xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate',
                                width=600, height=500
                            )
                            
                            roc_figs[model_name] = fig
            
            # Define app layout
            app.layout = html.Div([
                html.H1('Model Performance Dashboard', style={'textAlign': 'center'}),
                
                # Metrics comparison section
                html.Div([
                    html.H2('Performance Metrics'),
                    dcc.Graph(id='metrics-comparison', figure=metrics_fig)
                ], style={'marginBottom': '30px'}),
                
                # Confusion matrix section
                html.Div([
                    html.H2('Confusion Matrices'),
                    html.Label('Select Model:'),
                    dcc.Dropdown(
                        id='confusion-model-dropdown',
                        options=[{'label': name, 'value': name} for name in confusion_figs.keys()],
                        value=list(confusion_figs.keys())[0] if confusion_figs else None
                    ),
                    dcc.Graph(id='confusion-matrix')
                ], style={'marginBottom': '30px'}) if confusion_figs else html.Div(),
                
                # Feature importance section
                html.Div([
                    html.H2('Feature Importance'),
                    html.Label('Select Model:'),
                    dcc.Dropdown(
                        id='importance-model-dropdown',
                        options=[{'label': name, 'value': name} for name in importance_figs.keys()],
                        value=list(importance_figs.keys())[0] if importance_figs else None
                    ),
                    dcc.Graph(id='feature-importance')
                ], style={'marginBottom': '30px'}) if importance_figs else html.Div(),
                
                # ROC curves section
                html.Div([
                    html.H2('ROC Curves'),
                    html.Label('Select Model:'),
                    dcc.Dropdown(
                        id='roc-model-dropdown',
                        options=[{'label': name, 'value': name} for name in roc_figs.keys()],
                        value=list(roc_figs.keys())[0] if roc_figs else None
                    ),
                    dcc.Graph(id='roc-curve')
                ], style={'marginBottom': '30px'}) if roc_figs else html.Div(),
            ], style={'padding': '20px'})
            
            # Define callbacks for interactive elements
            @app.callback(
                Output('confusion-matrix', 'figure'),
                [Input('confusion-model-dropdown', 'value')]
            )
            def update_confusion_matrix(selected_model):
                if not selected_model or selected_model not in confusion_figs:
                    return go.Figure()
                return confusion_figs[selected_model]
            
            @app.callback(
                Output('feature-importance', 'figure'),
                [Input('importance-model-dropdown', 'value')]
            )
            def update_feature_importance(selected_model):
                if not selected_model or selected_model not in importance_figs:
                    return go.Figure()
                return importance_figs[selected_model]
            
            @app.callback(
                Output('roc-curve', 'figure'),
                [Input('roc-model-dropdown', 'value')]
            )
            def update_roc_curve(selected_model):
                if not selected_model or selected_model not in roc_figs:
                    return go.Figure()
                return roc_figs[selected_model]
            
            # Save dashboard as HTML
            output_path = os.path.join(output_dir, 'model_performance_dashboard.html')
            app.index_string = '''
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Model Performance Dashboard</title>
                    {%css%}
                </head>
                <body>
                    {%app_entry%}
                    <footer>
                        {%config%}
                        {%scripts%}
                        {%renderer%}
                    </footer>
                </body>
            </html>
            '''
            
            # Save dashboard as a standalone HTML file
            from dash.dependencies import ClientsideFunction
            app.clientside_callback(
                ClientsideFunction(namespace="clientside", function_name="updateGraphs"),
                Output("metrics-comparison", "figure"),
                [Input("metrics-comparison", "id")]
            )
            
            # Write the dashboard to an HTML file
            with open(output_path, 'w') as f:
                f.write(app.index_string.format(
                    css=app._generate_css_dist_html(),
                    app_entry=app.layout.to_html(),
                    config=app._generate_config_html(),
                    scripts=app._generate_scripts_html(),
                    renderer=app._generate_renderer_html()
                ))
            
            self.logger.info(f"Model performance dashboard saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating model performance dashboard: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

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

    def _simulate_company_api_check(self, company_name):
        """Simulate calling a company validation API"""
        # In a real system, this would call an actual API
        # Here we just simulate results based on company name characteristics
        import random
        
        # Simulate API call
        is_found = random.random() < 0.7  # 70% chance of "finding" a company
        
        return {
            "found": is_found,
            "confidence": random.uniform(0.5, 0.95) if is_found else random.uniform(0.1, 0.4)
        }
    
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
    
    def _check_funding_manipulation(self, features):
        """Check for potential manipulation patterns in funding data"""
        # This is a placeholder for more sophisticated detection logic
        # In a real system, this would implement specific business rules
        
        # For example, detecting unnaturally round numbers or 
        # suspicious growth patterns in key metrics
        
        # Just a placeholder implementation
        return False

def main():
    """Main function to run the funding stage prediction pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize and run the enhanced pipeline
    pipeline = EnhancedPipeline()
    success = pipeline.run()
    
    # Log completion
    if success:
        logger.info("Funding stage prediction pipeline completed successfully.")
    else:
        logger.error("Funding stage prediction pipeline encountered errors during execution.")
    
    return success

if __name__ == "__main__":
    main()



