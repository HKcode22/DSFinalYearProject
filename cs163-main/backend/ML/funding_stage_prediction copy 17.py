from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error)
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    RandomizedSearchCV,
    cross_val_score)
import pickle
import csv
import random  # Added import for random module
import uuid  # For generating unique identifiers for audit logs
import re
import shutil
import glob
import traceback
import sqlite3
import joblib
import xgboost as xgb
from sklearn.svm import OneClassSVM
from scipy.stats import randint, uniform
from sklearn.preprocessing import label_binarize, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# +++ Add LightGBM Import +++
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
    logger.warning("LightGBM not installed. Skipping LightGBM model. Install with: pip install lightgbm")
# +++ End LightGBM Import +++
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')  # Use Agg backend which doesn't require a display
# --- Add Prophet Import --- # Updated comment
try:
    from prophet import Prophet # Indent this line
except ImportError:
    Prophet = None # Indent this line
    logger.warning("Prophet library not found. Time series forecasting features will be disabled. Install with: pip install prophet") # Indent this line
# from prophet import Prophet # Removed
# from prophet.plot import plot_plotly # Removed # Using plotly for potentially interactive plots if needed later

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("funding_prediction.log"),
        logging.StreamHandler()])
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, base_dir="./", archive=False, output_dir_for_db="./MainOutput"):
        """Initialize data loader with paths to data sources and historical database"""
        # Resolve base_dir to an absolute path immediately
        # This base_dir is assumed to be the directory containing the JSON files.
        self.base_dir = os.path.abspath(base_dir)
        logger.info(f"DataLoader using absolute base_dir for JSON files: {self.base_dir}")

        self.archive = archive
        self.archive_dir = None
        # output_dir_for_db is expected to be an absolute path or relative to script execution
        # If pipeline passes an absolute path like /path/to/cs163-main/backend/MainOutput, this is fine.
        os.makedirs(output_dir_for_db, exist_ok=True) 
        self.historical_db = os.path.join(
            output_dir_for_db, "historical_funding_data.db")

        # self.base_dir IS the JSON folder
        self.json_folder = self.base_dir

        # Use the fixed json_folder path for file paths
        self.fundraiser_path = os.path.join(
            self.json_folder, "fundraisestartup50.json")
        self.growthlist_path = os.path.join(
            self.json_folder, "growthlistscrapper.json")
        self.topstartup_path = os.path.join(
            self.json_folder, "topstartupio50.json")
        
        logger.info(f"Fundraiser path: {self.fundraiser_path}")
        logger.info(f"Growthlist path: {self.growthlist_path}")
        logger.info(f"Topstartup path: {self.topstartup_path}")

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
                return
            # Archive fundraiser data
            if os.path.isfile(self.fundraiser_path):
                shutil.copy2(
                    self.fundraiser_path,
                    os.path.join(
                        self.archive_dir,
                        "fundraiser.json"))
            # Archive growthlist data
            if os.path.isfile(self.growthlist_path):
                shutil.copy2(
                    self.growthlist_path,
                    os.path.join(
                        self.archive_dir,
                        "growthlist.json"))
            # Archive topstartup data
            if os.path.isfile(self.topstartup_path):
                shutil.copy2(
                    self.topstartup_path,
                    os.path.join(
                        self.archive_dir,
                        "topstartup.json"))
            logger.info(f"Archived data files to {self.archive_dir}")
        except Exception as e:
            logger.error(f"Error archiving data: {e}")

    def _init_historical_db(self):
        """Create SQLite database tables if they don't exist"""
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

    def validate_company(self, company_name, funding_amount=None, 
                        funding_stage=None, employees=None):
        """
        Validate company data for consistency and quality
        
        Parameters:
        company_name (str): Name of the company
        funding_amount (float): Funding amount in USD
        funding_stage (str): Funding stage (e.g., Seed, Series A)
        employees (int): Number of employees
        
        Returns:
        tuple: (is_valid, confidence_score, message)
        """
        # Initialize validation results
        is_valid = True
        confidence_score = 1.0
        messages = []
        
        # Check company name validity
        if pd.isna(company_name) or not company_name or len(str(company_name).strip()) < 2:
            is_valid = False
            confidence_score = 0.0
            return is_valid, confidence_score, "Invalid company name"
        
        # Normalize funding stage if present
        valid_funding_stages = [
            'pre-seed', 'seed', 'angel', 'series a', 'series b', 'series c', 
            'series d', 'series e', 'series f', 'series g', 'series h', 
            'venture - series unknown', 'private equity', 'grant', 
            'debt financing', 'undisclosed', 'post-ipo'
        ]
        
        if funding_stage is not None and pd.notna(funding_stage):
            normalized_stage = str(funding_stage).lower().strip()
            if normalized_stage not in valid_funding_stages:
                # Check for partial matches
                stage_found = False
                for valid_stage in valid_funding_stages:
                    if valid_stage in normalized_stage or normalized_stage in valid_stage:
                        stage_found = True
                        break
                
                if not stage_found:
                    messages.append(f"Unknown funding stage: {funding_stage}")
                    confidence_score *= 0.7
        else:
            messages.append("Missing funding stage")
            confidence_score *= 0.8
        
        # Validate funding amount
        if funding_amount is not None:
            if pd.isna(funding_amount):
                messages.append("Missing funding amount")
                confidence_score *= 0.9
            elif funding_amount <= 0:
                messages.append("Invalid negative or zero funding amount")
                confidence_score *= 0.6
            elif funding_amount > 10e9:  # $10B is very high for most funding rounds
                messages.append(f"Suspicious high funding amount: ${funding_amount:,.2f}")
                confidence_score *= 0.5
        
        # Validate employee count if provided
        if employees is not None and pd.notna(employees):
            try:
                emp_count = float(employees)
                if emp_count <= 0:
                    messages.append("Invalid negative or zero employee count")
                    confidence_score *= 0.8
                elif emp_count > 1000000:  # Very large company
                    messages.append(f"Suspicious high employee count: {emp_count:,.0f}")
                    confidence_score *= 0.7
            except (ValueError, TypeError):
                messages.append(f"Invalid employee data format: {employees}")
                confidence_score *= 0.7
        
        # Consistency check between funding amount and stage
        if (funding_stage is not None and funding_amount is not None and 
            pd.notna(funding_stage) and pd.notna(funding_amount)):
            
            normalized_stage = str(funding_stage).lower().strip()
            
            # Check for inconsistencies like seed rounds with very high funding
            if ('seed' in normalized_stage or 'pre-seed' in normalized_stage) and funding_amount > 50e6:
                messages.append(f"Unusual high funding for {funding_stage}: ${funding_amount:,.2f}")
                confidence_score *= 0.6
            elif 'series a' in normalized_stage and funding_amount > 500e6:
                messages.append(f"Unusual high funding for {funding_stage}: ${funding_amount:,.2f}")
                confidence_score *= 0.7
        
        # Make final validation decision
        if confidence_score < 0.3:
            is_valid = False
        
        # Create final message
        if messages:
            final_message = "; ".join(messages)
        else:
            final_message = "Valid company data"
            
        return is_valid, confidence_score, final_message

    def load_fundraiser_data(self):
        """Load and process fundraiser insider data"""
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

            logger.info(f"Loaded {len(df)} records from fundraiser data")
            return df
        except Exception as e:
            logger.error(f"Error loading fundraiser data: {e}")
            return pd.DataFrame()

    def load_growthlist_data(self):
        """Load and process growthlist startups data"""
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
                        # Standardize the stage format
                        raw_stage = stage_match.group(1)

                        # Normalize stage name
                        if re.match(
                            r'pre[-\s]?seed',
                            raw_stage,
                                re.IGNORECASE):
                            stage = 'Pre-Seed'
                        elif re.match(r'seed', raw_stage, re.IGNORECASE):
                            stage = 'Seed'
                        elif re.match(r'angel', raw_stage, re.IGNORECASE):
                            stage = 'Angel'
                        elif re.match(r'series\s+([a-z])', raw_stage, re.IGNORECASE):
                            # Ensure proper capitalization for series (e.g.
                            # "Series A")
                            series_letter = re.match(
                                r'series\s+([a-z])', raw_stage, re.IGNORECASE).group(1).upper()
                            stage = f'Series {series_letter}'
                        elif re.match(r'venture[-\s]+series[-\s]+unknown', raw_stage, re.IGNORECASE):
                            stage = 'venture - series unknown'
                        elif re.match(r'initial\s+coin\s+offering|ico', raw_stage, re.IGNORECASE):
                            stage = 'initial coin offering'
                        elif re.match(r'private\s+equity', raw_stage, re.IGNORECASE):
                            stage = 'Private Equity'
                        elif re.match(r'grant', raw_stage, re.IGNORECASE):
                            stage = 'Grant'
                        elif re.match(r'debt\s+financing', raw_stage, re.IGNORECASE):
                            stage = 'debt financing'
                        elif re.match(r'undisclosed', raw_stage, re.IGNORECASE):
                            stage = 'undisclosed'
                        elif re.match(r'post[-\s]?ipo', raw_stage, re.IGNORECASE):
                            stage = 'Post-IPO'
                        else:
                            stage = raw_stage  # Use as-is if no specific match
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
                                    stage = f'Series {
                                        series_match.group(1).upper()}'
                                else:
                                    stage = 'venture - series unknown'
                            else:
                                # Default to "Venture Funding" for generic
                                # raised amounts
                                stage = 'venture - series unknown'
                        elif 'valuation' in funding_lower and not stage:
                            if 'post-ipo' in funding_lower or 'post ipo' in funding_lower:
                                stage = 'Post-IPO'
                            else:
                                # Companies with just valuation mentioned but
                                # no explicit funding stage
                                stage = 'venture - series unknown'

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

            logger.info(f"Loaded {len(df)} records from topstartup data")
            return df

        except Exception as e:
            logger.error(f"Error loading topstartup data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _parse_funding_amount(self, amount_str):
        """Improved function to convert funding amount strings to numeric values with strict validation"""
        if not amount_str or pd.isna(amount_str) or amount_str == "":
            return np.nan

        try:
            # Remove currency symbol and commas
            amount_str = str(amount_str).replace(
                '$', '').replace(',', '').strip()

            # Define reasonable limits based on funding stages
            max_reasonable_amount = 1e10  # $10B is a reasonable upper limit for most funding

            # Convert based on unit (M=million, B=billion, K=thousand)
            if 'B' in amount_str:
                value = float(amount_str.replace('B', '')) * 1e9
            elif 'M' in amount_str:
                value = float(amount_str.replace('M', '')) * 1e6
            elif 'K' in amount_str:
                value = float(amount_str.replace('K', '')) * 1e3
            else:
                value = float(amount_str)

            # Validate against reasonable limits
            if value > max_reasonable_amount:
                logger.warning(
                    f"Unreasonably large funding amount detected: ${
                        value:,.2f}")
                return np.nan
            elif value < 0:
                logger.warning(
                    f"Negative funding amount detected: ${
                        value:,.2f}")
                return np.nan

            return value
        except Exception as e:
            logger.warning(
                f"Error parsing funding amount '{amount_str}': {
                    str(e)}")
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

            # Initialize list to store all records and an audit log
            all_records = []
            audit_log = []
            rejected_records = []

            # Process fundraiser data
            if not fundraiser_df.empty:
                for _, row in fundraiser_df.iterrows():
                    if pd.notna(
                            row.get('Company')):  # Only add records with valid company names
                        # Convert numeric values properly
                        try:
                            funding_amount = pd.to_numeric(
                                row.get('Funding_Amount_USD'), errors='coerce')
                            employees = pd.to_numeric(
                                row.get('Total_Employees'), errors='coerce')
                        except BaseException:
                            funding_amount = np.nan
                            employees = np.nan

                        # Validate company and data
                        is_valid, confidence, message = self.validate_company(
                            row.get('Company'),
                            funding_amount=funding_amount,
                            funding_stage=row.get('Funding_Type'),
                            employees=employees
                        )

                        # Record the validation outcome
                        audit_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'company': row.get('Company'),
                            'source': 'fundraiser',
                            'validation_result': is_valid,
                            'confidence_score': confidence,
                            'message': message
                        }
                        audit_log.append(audit_entry)

                        if is_valid and confidence >= 0.3:  # Minimum confidence threshold
                            all_records.append({
                                'company_name': row.get('Company'),
                                'funding_stage': row.get('Funding_Type'),
                                'funding_amount': funding_amount,
                                'funding_date': row.get('Funding_Date'),
                                'industry': row.get('Industry'),
                                'employees': employees,
                                'source': 'fundraiser',
                                'confidence_score': confidence
                            })
                        else:
                            rejected_records.append({
                                'company': row.get('Company'),
                                'reason': message,
                                'source': 'fundraiser'
                            })

                logger.info(
                    f"Processed {
                        len(fundraiser_df)} records from fundraiser data, rejected {
                        len(rejected_records)} suspicious records")

            # Process growthlist data with similar validation
            if not growthlist_df.empty:
                growthlist_rejected = len(rejected_records)
                for _, row in growthlist_df.iterrows():
                    if pd.notna(
                            row.get('name')):  # Only add records with valid company names
                        # Parse the amount if not already parsed
                        funding_amount = row.get('funding_amount_numeric')
                        if pd.isna(funding_amount) and pd.notna(
                                row.get('funding_amount')):
                            funding_amount = self._parse_funding_amount(
                                row.get('funding_amount'))

                        # Validate company and data
                        is_valid, confidence, message = self.validate_company(
                            row.get('name'),
                            funding_amount=funding_amount,
                            funding_stage=row.get('funding_type')
                        )

                        # Record validation
                        audit_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'company': row.get('name'),
                            'source': 'growthlist',
                            'validation_result': is_valid,
                            'confidence_score': confidence,
                            'message': message
                        }
                        audit_log.append(audit_entry)

                        if is_valid and confidence >= 0.3:
                            all_records.append({
                                'company_name': row.get('name'),
                                'funding_stage': row.get('funding_type'),
                                'funding_amount': funding_amount,
                                'funding_date': row.get('last_funding_date'),
                                'industry': row.get('industry'),
                                'employees': None,
                                'source': 'growthlist',
                                'confidence_score': confidence
                            })
                        else:
                            rejected_records.append({
                                'company': row.get('name'),
                                'reason': message,
                                'source': 'growthlist'
                            })

                logger.info(
                    f"Processed {
                        len(growthlist_df)} records from growthlist data, rejected {
                        len(rejected_records) -
                        growthlist_rejected} suspicious records")

            # Process topstartup data with validation
            if not topstartup_df.empty:
                topstartup_rejected = len(rejected_records)
                for _, row in topstartup_df.iterrows():
                    company_name = row.get('company_name') or row.get('name')

                    if pd.notna(
                            company_name):  # Only add records with valid company names
                        # Clean up data
                        funding_stage = row.get(
                            'funding_stage') or row.get('funding_round')

                        # Parse the amount if string
                        funding_amount = row.get('funding_amount')
                        if isinstance(funding_amount, str):
                            funding_amount = self._parse_funding_amount(
                                funding_amount)

                        # Get employee count range
                        employee_count = None
                        if pd.notna(row.get('employees')):
                            # Handle ranges like "11-50 employees"
                            emp_str = str(row.get('employees'))
                            match = re.search(r'(\d+)-(\d+)', emp_str)
                            if match:
                                # Take the average of the range
                                employee_count = (
                                    int(match.group(1)) + int(match.group(2))) / 2

                        # Validate company data
                        is_valid, confidence, message = self.validate_company(
                            company_name,
                            funding_amount=funding_amount,
                            funding_stage=funding_stage,
                            employees=employee_count
                        )

                        # Record validation
                        audit_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'company': company_name,
                            'source': 'topstartup',
                            'validation_result': is_valid,
                            'confidence_score': confidence,
                            'message': message
                        }
                        audit_log.append(audit_entry)

                        if is_valid and confidence >= 0.3:
                            all_records.append({
                                'company_name': company_name,
                                'funding_stage': funding_stage,
                                'funding_amount': funding_amount,
                                'funding_date': row.get('funding_date'),
                                'industry': row.get('industry') or row.get('category'),
                                'employees': employee_count,
                                'source': 'topstartup',
                                'confidence_score': confidence
                            })
                        else:
                            rejected_records.append({
                                'company': company_name,
                                'reason': message,
                                'source': 'topstartup'
                            })

                logger.info(
                    f"Processed {
                        len(topstartup_df)} records from topstartup data, rejected {
                        len(rejected_records) -
                        topstartup_rejected} suspicious records")

            # Save audit log and rejected records
            audit_df = pd.DataFrame(audit_log)
            rejected_df = pd.DataFrame(rejected_records)

            if not audit_df.empty:
                audit_path = os.path.join(
                    self.base_dir, "data_validation_audit.csv")
                audit_df.to_csv(
                    audit_path,
                    mode='a',
                    header=not os.path.exists(audit_path),
                    index=False)
                logger.info(f"Saved data validation audit to {audit_path}")

            if not rejected_df.empty:
                rejected_path = os.path.join(
                    self.base_dir, "rejected_records.csv")
                rejected_df.to_csv(
                    rejected_path,
                    mode='a',
                    header=not os.path.exists(rejected_path),
                    index=False)
                logger.info(
                    f"Saved {
                        len(rejected_df)} rejected records to {rejected_path}")

            if all_records:
                # Create DataFrame from records
                merged_data = pd.DataFrame(all_records)

                # Log column counts to debug missing data
                logger.info(
                    f"Merged data columns: {
                        merged_data.columns.tolist()}")
                logger.info(
                    f"Non-null counts: {merged_data.count().to_dict()}")

                # Drop duplicates after creation
                pre_dedup_count = len(merged_data)
                merged_data = merged_data.drop_duplicates(
                    subset=['company_name', 'funding_date']
                ).reset_index(drop=True)
                logger.info(
                    f"Removed {
                        pre_dedup_count -
                        len(merged_data)} duplicate records")

                # Standardize funding stage names
                stage_mapping = {
                    'series a': 'Series A',
                    'series b': 'Series B',
                    'series c': 'Series C',
                    'series d': 'Series D',
                    'series e': 'Series E',
                    'series f': 'Series F',
                    'series g': 'Series G',
                    'series h': 'Series H',
                    'pre-seed': 'Pre-Seed',
                    'seed': 'Seed',
                    'angel': 'Angel',
                    'grant': 'Grant',
                    'debt': 'Debt Financing',
                    'ipo': 'IPO',
                    'private equity': 'Private Equity'
                }

                if 'funding_stage' in merged_data.columns:
                    merged_data['funding_stage'] = merged_data['funding_stage'].str.lower().map(
                        lambda x: stage_mapping.get(x, x) if pd.notna(x) else 'Unknown')

                # Fill missing values with meaningful defaults
                merged_data = merged_data.fillna({
                    'industry': 'Unknown',
                    'employees': 0,
                    'funding_stage': 'Unknown',
                    'confidence_score': 0.5  # Default confidence for missing scores
                })

                logger.info(
                    f"Successfully merged {
                        len(merged_data)} validated records")
                return merged_data

            logger.warning(
                "No valid records available after merging and validation")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error in merge_datasets: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()



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
        # +++ Add attributes to store binning info +++
        self.age_bin_edges = None
        self.age_bin_labels = None

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

            # +++ Add Cyclical Month Features +++
            data['month_sin'] = np.sin(2 * np.pi * data['funding_month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['funding_month'] / 12)
            # +++ End Cyclical Month Features +++

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

        # --- Consolidate Rare Industries --- #
        min_frequency = 10
        industry_counts = data['industry_category'].value_counts()
        rare_industries = industry_counts[industry_counts < min_frequency].index.tolist()
        # Ensure 'Other' itself isn't marked as rare if it meets threshold
        if 'Other' in data['industry_category'].unique() and 'Other' in rare_industries:
            if industry_counts.get('Other', 0) >= min_frequency:
                rare_industries.remove('Other')

        if rare_industries:
            logger.info(f"Mapping {len(rare_industries)} rare industries (count < {min_frequency}) to 'Other': {rare_industries[:10]}...")
            data['industry_category'] = data['industry_category'].replace(rare_industries, 'Other')
        # --- End Consolidate Rare Industries --- #

        # --- ADD Features based on Funding History --- #
        try:
            data = data.sort_values(by=['company_name', 'funding_date'])

            # Calculate time since last funding
            data['time_since_last_funding'] = data.groupby('company_name')['funding_date'].diff().dt.days / 30.44 # Approx months
            data['time_since_last_funding'] = data['time_since_last_funding'].fillna(0) # Fill first round NaNs

            # Calculate funding ratio vs previous round
            data['prev_funding_amount'] = data.groupby('company_name')['funding_amount'].shift(1)
            data['funding_amount_ratio_vs_prev'] = data.apply(
                 lambda row: (row['funding_amount'] / row['prev_funding_amount'])
                             if pd.notna(row['funding_amount']) and pd.notna(row['prev_funding_amount']) and row['prev_funding_amount'] > 0
                             else 1.0,
                 axis=1
            )

            # Calculate funding vs industry median
            industry_median_funding = data.groupby('industry_category')['funding_amount'].transform('median')
            data['funding_vs_industry_median'] = data.apply(
                lambda row: (row['funding_amount'] / industry_median_funding[row.name])
                            if pd.notna(row['funding_amount']) and pd.notna(industry_median_funding[row.name]) and industry_median_funding[row.name] > 0
                            else 1.0,
                axis=1
            )
            logger.info("Added features: time_since_last_funding, funding_amount_ratio_vs_prev, funding_vs_industry_median")
        except Exception as hist_err:
            logger.error(f"Error creating funding history features: {hist_err}")
            # Add default values if calculation fails
            data['time_since_last_funding'] = 0
            data['funding_amount_ratio_vs_prev'] = 1.0
            data['funding_vs_industry_median'] = 1.0
        # --- End Funding History Features --- #

        # Log transform funding amount (handle skewed distribution)
        # First ensure it's numeric
        data['funding_amount'] = pd.to_numeric(data['funding_amount'], errors='coerce')

        # --- Cap funding amount at $2B before log transform --- #
        funding_cap = 2e9
        data['funding_amount'] = data['funding_amount'].clip(upper=funding_cap)
        logger.info(f"Capped funding_amount at ${funding_cap:,.0f}")
        # --- End Capping --- #

        # Check for extremely large values that might be errors (>$100B)
        # Note: This check is now less relevant after capping, but kept for logging
        if (data['funding_amount'] > 1e11).any(): 
            large_values = data[data['funding_amount'] > 1e11]
            logger.warning(
                f"Found {len(large_values)} extremely large funding amounts (>$100B)")
            logger.warning(
                f"Sample: {large_values[['company_name', 'funding_amount', 'source']].head().to_dict()}")

        # Apply log transform with offset to handle zeros
        data['funding_amount_log'] = np.log1p(data['funding_amount'].fillna(0))

        # Employee efficiency (funding per employee)
        if 'employees' in data.columns:
            data['employees'] = pd.to_numeric(
                data['employees'], errors='coerce')
            data['employees'] = data['employees'].fillna(0) # Ensure employees is filled

            # Avoid division by zero
            data['employee_efficiency'] = data.apply(
                lambda row: row['funding_amount'] /
                row['employees'] if row['employees'] > 0 else np.nan,
                axis=1)

            # Fill missing values with median by funding stage
            efficiency_medians = data.groupby('funding_stage_numeric')[ # Use numeric stage for groupby
                'employee_efficiency'].median()

            for stage_numeric_val in data['funding_stage_numeric'].unique(): # Iterate over numeric stage
                stage_median = efficiency_medians.get(
                    stage_numeric_val, data['employee_efficiency'].median())
                mask = (
                    data['funding_stage_numeric'] == stage_numeric_val) & ( # Use numeric stage
                    data['employee_efficiency'].isna())
                data.loc[mask, 'employee_efficiency'] = stage_median

            # Fill any remaining NaNs with overall median
            data['employee_efficiency'] = data['employee_efficiency'].fillna(
                data['employee_efficiency'].median())
        else:
            data['employees'] = 0 # Default to 0 if column doesn't exist
            data['employee_efficiency'] = np.nan

        # --- Add Binning Feature --- # # Renamed from company_age_bin
        age_feature = 'months_since_first_funding'
        if age_feature in data.columns:
            try:
                # --- Use fixed bins for Age --- #
                age_bins = [-np.inf, 12, 24, 48, 72, np.inf] # Bins: <=12, 13-24, 25-48, 49-72, 73+
                age_labels = ['0-12m', '13-24m', '25-48m', '49-72m', '73m+']
                data['company_age_bin'] = pd.cut(
                    data[age_feature],
                    bins=age_bins,
                    labels=age_labels,
                    right=True # Include right edge (e.g., 12 months is in 0-12m)
                ).astype(str) # Ensure result is string

                # Store binning info for reference (though edges are fixed now)
                self.age_bin_edges = age_bins
                self.age_bin_labels = age_labels
                logger.info(f"Created 'company_age_bin' feature with fixed bins: {age_labels}")

                # Fill potential NaNs from binning (if few unique values) with a default bin label
                data['company_age_bin'] = data['company_age_bin'].fillna('Unknown_Age').astype(str) # Ensure final column is string, use specific unknown

                # Keep the numeric version if needed elsewhere, otherwise it can be removed
                # data['company_age_bin_numeric'] = data['company_age_bin_numeric'].fillna(-1).astype(int)
            except Exception as bin_err:
                 logger.warning(f"Could not create 'company_age_bin': {bin_err}")
                 data['company_age_bin'] = 'Unknown_Age' # Assign default string label if binning fails
                 # data['company_age_bin_numeric'] = -1 # Assign default if binning fails
                 self.age_bin_edges = None
                 self.age_bin_labels = None
        else:
            data['company_age_bin'] = 'Unknown_Age' # Assign default string label if source column missing
            # data['company_age_bin_numeric'] = -1
            self.age_bin_edges = None
            self.age_bin_labels = None
        # --- End Binning Feature --- #

        # --- Add Binning for Employees --- #
        emp_feature = 'employees'
        if emp_feature in data.columns:
            try:
                emp_bins = [-np.inf, 10, 50, 200, 1000, np.inf] # Bins: 0(or less)-10, 11-50, 51-200, 201-1000, 1001+
                emp_labels = ['1-10', '11-50', '51-200', '201-1000', '1001+']
                data['employees_bin'] = pd.cut(
                    data[emp_feature],
                    bins=emp_bins,
                    labels=emp_labels,
                    right=True
                ).astype(str)
                data['employees_bin'] = data['employees_bin'].fillna('Unknown_Emp').astype(str)
                logger.info(f"Created 'employees_bin' feature with fixed bins: {emp_labels}")
            except Exception as emp_bin_err:
                 logger.warning(f"Could not create 'employees_bin': {emp_bin_err}")
                 data['employees_bin'] = 'Unknown_Emp'
        else:
             data['employees_bin'] = 'Unknown_Emp'
        # --- End Employee Binning --- #

        # Funding frequency features
        company_funding_counts = data.groupby('company_name').size()
        data['previous_rounds'] = data['company_name'].map(
            company_funding_counts) - 1
        data['previous_rounds'] = data['previous_rounds'].clip(lower=0)
        data['previous_rounds'] = data['previous_rounds'].fillna(0) # Ensure previous_rounds is filled

        # New feature: Funding velocity (average months between rounds)
        company_funding_dates = data.groupby(
            'company_name')['funding_date'].apply(list)

        def calc_funding_velocity(dates):
            if not isinstance(dates, list) or len(dates) < 2:
                return np.nan

            valid_dates = [d for d in dates if not pd.isna(d)]
            if len(valid_dates) < 2:
                return np.nan

            valid_dates.sort()
            intervals = []
            for i in range(1, len(valid_dates)):
                interval = (valid_dates[i].year - valid_dates[i - 1].year) * \\
                    12 + (valid_dates[i].month - valid_dates[i - 1].month)
                intervals.append(interval)

            return np.mean(intervals) if intervals else np.nan

        data['funding_velocity'] = data['company_name'].map(
            company_funding_dates.apply(calc_funding_velocity)
        )
        funding_velocity_median = data['funding_velocity'].median()
        if pd.isna(funding_velocity_median):
            funding_velocity_median = 0 
        data['funding_velocity'] = data['funding_velocity'].fillna(funding_velocity_median) # Ensure funding_velocity is filled

        # --- Add Interaction Features +++
        # Ensure required columns exist and are numeric before creating interactions
        # ... existing code ...
        # Fill missing values for all numeric columns
        numeric_cols = [
            'funding_amount',
            'funding_amount_log',
            'employees',
            'employee_efficiency',
            'previous_rounds',
            'funding_velocity',
            'month_sin', 
            'month_cos', 
            'funding_amount_x_age', 
            'employees_x_rounds', 
            'velocity_x_rounds', 
            'age_x_employees',
            'time_since_last_funding',      # <<< ENSURED PRESENT
            'funding_amount_ratio_vs_prev', # <<< ENSURED PRESENT
            'funding_vs_industry_median'  # <<< ENSURED PRESENT
        ]

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
        """Prepare feature matrix with proper type handling"""
        # Select relevant features that exist in the original data
        numeric_feature_cols = [
            'funding_amount_log',
            'employee_efficiency',
            'funding_year',
            'funding_month',
            'previous_rounds',
            'funding_velocity',
            'month_sin',
            'month_cos',
            'funding_amount_x_age',
            'employees_x_rounds',
            'velocity_x_rounds',
            'age_x_employees',
            'time_since_last_funding',
            'funding_amount_ratio_vs_prev',
            'funding_vs_industry_median'
        ]
        categorical_feature_cols = [
            'industry_category', 
            'company_age_bin', 
            'employees_bin'
        ]

        features_to_use_num = [col for col in numeric_feature_cols if col in data.columns]
        features_to_use_cat = [col for col in categorical_feature_cols if col in data.columns]

        logger.info(f"Preparing model data with numeric features: {features_to_use_num}")
        logger.info(f"Preparing model data with categorical features: {features_to_use_cat}")

        X_num = data[features_to_use_num].copy()
        for col in X_num.columns:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce')

        numeric_cols_present = X_num.select_dtypes(include=np.number).columns
        for col in numeric_cols_present:
            if X_num[col].isna().any():
                median_value = X_num[col].median()
                X_num[col] = X_num[col].fillna(median_value)
                logger.info(f"Filled NaN values in {col} with median: {median_value}")

        X_cat_encoded = pd.DataFrame()
        if features_to_use_cat:
            X_cat = data[features_to_use_cat].copy()
            for col in X_cat.columns:
                 X_cat[col] = X_cat[col].astype('category')
            X_cat_encoded = pd.get_dummies(X_cat, prefix=features_to_use_cat, dummy_na=False)
            logger.info(f"Created {X_cat_encoded.shape[1]} features from one-hot encoding: {X_cat_encoded.columns.tolist()}")

        X = pd.concat([X_num, X_cat_encoded], axis=1)

        try:
            original_columns = X.columns.tolist()
            sanitized_columns = []
            for col in original_columns:
                sanitized_col = str(col).replace('[', '_').replace(']', '_').replace('<', '_') \\
                                         .replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '')
                sanitized_columns.append(sanitized_col)
            X.columns = sanitized_columns
            if original_columns: # Add check for empty list
                 logger.info(f"Sanitized {len(original_columns)} feature names. Example: '{original_columns[-1]}' -> '{X.columns[-1]}'")
        except Exception as sanitize_err:
            logger.error(f"Error sanitizing feature names: {sanitize_err}")
            pass

        y = pd.to_numeric(data['funding_stage_numeric'], errors='coerce')
        if y.notna().sum() < 10:
            logger.warning(f"Very few valid target values: {y.notna().sum()} out of {len(y)}")

        X = X[y.notna()]
        y = y[y.notna()].astype(int)

        logger.info(f"Prepared model data: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    def _train_final_model(self, model, X, y, name):
        """Helper method to train final models with detailed metrics including RMSE and Calibration""" # Updated docstring
        # Need to import clone
        from sklearn.base import clone
        # Ensure X is DataFrame for consistent feature name handling if possible
        if not isinstance(X, pd.DataFrame):
            logger.warning(f"Input X for {name} is not a DataFrame. Creating one with generic feature names.")
            feature_names_list = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names_list)
        else:
            X_df = X.copy() # Use copy to avoid modifying original
            feature_names_list = X_df.columns.tolist()

        # Ensure y is a Series for stratify and indexing
        if not isinstance(y, pd.Series):
             y = pd.Series(y)

        # Align indices before splitting
        X_df = X_df.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42, stratify=y)

        # Standardize features for better model performance
        scaler = StandardScaler() # Use StandardScaler
        # Fit scaler ONLY on training data
        X_train_scaled = scaler.fit_transform(X_train)
        # Transform test data using the SAME scaler
        X_test_scaled = scaler.transform(X_test)

        # Convert scaled arrays back to DataFrames for consistent feature names if needed by model
        # Some models might handle numpy arrays directly, others might benefit from DataFrame
        # For simplicity, we pass numpy arrays to fit/predict, but retain feature_names_list

        # Train the model with cross-validation on scaled training data
        cv_scores = np.array([np.nan]) # Initialize with NaN
        try:
            # Ensure model is a fresh instance if it was already fit during tuning search
            # Clone models with random_state or ensembles to ensure they are fresh for CV
            if hasattr(model, 'random_state') or isinstance(model, (VotingClassifier)):
                 model_clone = clone(model)
            else:
                 # For models without random_state assume it can be refit or doesn't need cloning.
                 # Handle specific cases if needed.
                 # If it's already trained (best_estimator_ from search), cloning is safest
                 try:
                     model_clone = clone(model)
                 except TypeError: # Cannot clone object - maybe simple function or non-sklearn?
                     logger.warning(f"Could not clone model {name} for CV. Using original instance.")
                     model_clone = model

            # Perform cross-validation on the training set
            # cv_scores = cross_val_score(model_clone, X_train_scaled, y_train, cv=5, scoring='accuracy', error_score='raise') # Disabled temporarily
            # logger.info(f"{name} cross-validation accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
            logger.info(f"Skipping cross-validation for {name} in this run.") # Placeholder log
            cv_scores = np.array([np.nan]) # Assign NaN score if CV fails or is skipped

        except Exception as cv_err:
            logger.error(f"Cross-validation failed for {name}: {cv_err}")
            cv_scores = np.array([np.nan]) # Assign NaN score if CV fails

        # Fit the final model on the full scaled training data
        # Ensure model is a fresh instance before final fit if needed (e.g., if CV modified it)
        try:
            model_final = clone(model) # Use a clone to ensure fresh state for final fit
        except TypeError:
             logger.warning(f"Could not clone model {name} for final fit. Using original instance.")
             model_final = model

        # Handle potential issues fitting final model
        fitted_model = None
        calibrated_model = None # <<< Initialize calibrated model
        try:
             fitted_model = model_final.fit(X_train_scaled, y_train)

             # --- Add Calibration Step --- #
             logger.info(f"Calibrating {name} model using Isotonic Regression...")
             # Wrap the fitted model. Use cv='prefit' as it's already trained.
             # Ensure fitted_model is not None before passing
             if fitted_model:
                 calibrated_model = CalibratedClassifierCV(fitted_model, method='isotonic', cv='prefit') # <<< FIXED INDENTATION
             # Fit the calibrator using the training data
             calibrated_model.fit(X_train_scaled, y_train)
             logger.info(f"Calibration complete for {name}.")
             # --- End Calibration Step --- #

             # --- Fit Anomaly Detector for this specific model/split --- #
             anomaly_detector_instance = None
             # if fitted_model: # Only fit if model training succeeded
             #     try:
             #         logger.info(f"Fitting anomaly detector for {name}...")
             if fitted_model: # Only fit if model training succeeded
                 try:
                     logger.info(f"Fitting anomaly detector for {name}...")
                     anomaly_detector_instance = AnomalyDetector(contamination=0.05) # Use class default or adjust
                     # Fit on the same scaled training data used for the model
                     anomaly_detector_instance.fit(X_train_scaled)
                     logger.info(f"Anomaly detector fitted for {name}.")
                 except Exception as ad_fit_err:
                     logger.error(f"Failed to fit anomaly detector for {name}: {ad_fit_err}")
                     anomaly_detector_instance = None # Ensure it's None on failure
             # --- End Anomaly Detector Fitting --- #

        except Exception as fit_err:
             logger.error(f"Fitting or Calibrating final model failed for {name}: {fit_err}")
             # Still return results dict, but with None for models and detector
             return None, { # Return None for model if fit failed
                 'model': None,
                 'calibrated_model': None, # <<< Added
                 'anomaly_detector': None, # <<< Added
                 'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan,
                 'rmse': np.nan, 'confusion_matrix': None, 'classification_report': None,
                 'cross_val_scores': cv_scores, 'X_test': X_test_scaled, 'y_test': y_test,
                 'y_pred': None, 'y_proba': None, 'feature_names': feature_names_list, 'scaler': scaler
             }


        # Make predictions on scaled test data USING CALIBRATED MODEL
        y_pred = calibrated_model.predict(X_test_scaled)
        y_proba = calibrated_model.predict_proba(X_test_scaled) # Probabilities from calibrated model

        # Calculate detailed metrics (using predictions from calibrated model)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Calculate class-specific metrics
        # Get unique labels present in y_test or y_pred for the report
        report_labels = np.unique(np.concatenate((y_test, y_pred)))
        class_report = classification_report(y_test, y_pred, labels=report_labels, output_dict=True, zero_division=0)

        # Print detailed metrics
        logger.info(f"--- {name} Model Performance ---")
        logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        logger.info(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")
        logger.info(f"RMSE: {rmse:.4f} (lower is better)")
        logger.info("-------------------------")

        # Also print to stdout for visibility
        print(f"\\n--- {name} Model Performance ---")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")
        print(f"RMSE: {rmse:.4f} (lower is better)")
        print("-------------------------")

        # Return the fitted final model and the scaler used
        return fitted_model, { # Return original fitted_model and calibrated_model
            'model': fitted_model, # Original uncalibrated model (e.g., for feature importance)
            'calibrated_model': calibrated_model, # <<< ADDED CALIBRATED MODEL
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'rmse': rmse,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'cross_val_scores': cv_scores,
            'X_test': X_test_scaled, # Return scaled test set
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'feature_names': feature_names_list, # Use list of names
            'scaler': scaler, # Return the scaler fitted on training data
            'anomaly_detector': anomaly_detector_instance # <<< ADDED ANOMALY DETECTOR
        }

    def _create_advanced_visualizations(
            self,
            rf_model,
            xgb_model,
            gb_model, # Added GB model
            rf_results,
            xgb_results,
            gb_results, # Added GB results
            dt_results, # Added DT results
            stacking_ensemble_results, # Use stacking ensemble results (will be None now)
            y, # Original y before splitting (needed for overall class info)
            processed_data): # Pass processed data for visualizations
        """Generate advanced model diagnostics and data visualizations""" # Updated docstring
        try:
            unique_classes = np.unique(y) # Use classes from the final y before split

            # Create a combined dictionary for plotting, ensure results are valid
            all_results_plot = {}
            if rf_results and rf_results.get('model'): all_results_plot['Random Forest'] = rf_results
            if xgb_results and xgb_results.get('model'): all_results_plot['XGBoost'] = xgb_results
            if gb_results and gb_results.get('model'): all_results_plot['Gradient Boosting'] = gb_results
            if dt_results and dt_results.get('model'): all_results_plot['Decision Tree'] = dt_results
            if stacking_ensemble_results and stacking_ensemble_results.get('model'): all_results_plot['Optimized Ensemble'] = stacking_ensemble_results

            if not all_results_plot:
                 logger.warning("No valid model results available for visualization.")
                 return # Exit early if no models to visualize

            # --- Time Series Forecasting Plot --- #
            # logger.info("Step 4.5: Performing Time Series Forecasting...") # Removed
            # try: # Removed
            #     # Use processed_data_viz as it's the data before splitting/scaling # Removed
            #     prophet_data = self.time_series_forecaster.prepare_prophet_data(processed_data_viz) # Removed
            #     if prophet_data is not None and not prophet_data.empty: # Removed
            #         # +++ Reduce forecast period to 12 months +++ # Removed
            #         prophet_model, prophet_forecast = self.time_series_forecaster.train_predict(prophet_data, periods=6) # Removed
            #         if prophet_model and prophet_forecast is not None: # Removed
            #             self.time_series_forecaster.plot_forecast(prophet_model, prophet_forecast) # Removed
            #         else: # Removed
            #             logger.warning("Prophet model training/prediction failed. Skipping forecast plot.") # Removed
            #     else: # Removed
            #         logger.warning("Data preparation for Prophet failed or yielded no data. Skipping forecast.") # Removed
            # except ImportError: # Removed
            #     logger.warning("Prophet library not found. Skipping time series forecasting. Please install it (`pip install prophet`).") # Removed
            # except Exception as ts_err: # Removed
            #      logger.error(f"Error during time series forecasting step: {ts_err}") # Removed
            #      logger.error(traceback.format_exc()) # Removed

            # --- Model Diagnostic Plots --- #
            logger.info("Creating advanced model diagnostic visualizations...") # Updated Log Message
            # Plot ROC/Calibration/Confidence only for models with probabilities
            for model_name, results in all_results_plot.items():
                 if results.get('y_proba') is not None and results.get('y_test') is not None and results.get('y_pred') is not None:
                     # Ensure y_test and y_proba have compatible shapes/classes
                     y_test_current = results['y_test']
                     y_proba_current = results['y_proba']
                     current_classes = np.unique(y_test_current) # Use classes present in y_test

                     if len(current_classes) > 1: # Need at least 2 classes for ROC/Calibration
                         try:
                             # Check if y_proba has probability for each class in unique_classes
                             # Use unique_classes from the overall y for consistency
                             if y_proba_current.shape[1] == len(unique_classes):
                                 self.visualizer.plot_roc_curves(
                                     y_test_current,
                                     y_proba_current,
                                     classes=unique_classes) # Plot against all potential classes
                             else:
                                 logger.warning(f"Mismatch between y_proba columns ({y_proba_current.shape[1]}) and unique classes ({len(unique_classes)}) for {model_name}. Skipping ROC.")
                         except Exception as roc_err:
                              logger.warning(f"Could not plot ROC for {model_name}: {roc_err}")

                         try:
                             self.visualizer.plot_calibration(
                                 y_test_current,
                                 y_proba_current)
                         except Exception as cal_err:
                              logger.warning(f"Could not plot Calibration for {model_name}: {cal_err}")

                         try:
                             self.visualizer.plot_confidence_intervals(
                                 y_test_current,
                                 results['y_pred'],
                                 y_proba_current)
                         except Exception as conf_err:
                             logger.warning(f"Could not plot Confidence Intervals for {model_name}: {conf_err}")
                 else:
                     logger.info(f"Skipping probability-based plots for {model_name} (probabilities, y_test or y_pred not available)")

            # Plot feature importance for each base model if available
            base_models = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'Gradient Boosting': gb_model, 'Decision Tree': dt_model}
            base_results = {'Random Forest': rf_results, 'XGBoost': xgb_results, 'Gradient Boosting': gb_results, 'Decision Tree': dt_results}

            for model_name, model_instance in base_models.items():
                 results_dict = base_results.get(model_name, {})
                 fitted_model = results_dict.get('model')
                 if fitted_model and hasattr(fitted_model, 'feature_importances_') and 'feature_names' in results_dict:
                     try:
                         if len(fitted_model.feature_importances_) == len(results_dict['feature_names']):
                             self.visualizer.plot_feature_importance(
                                 fitted_model, results_dict['feature_names'])
                         else:
                              logger.warning(f"Mismatch between feature importances ({len(fitted_model.feature_importances_)}) and feature names ({len(results_dict['feature_names'])}) for {model_name}. Skipping FI plot.")
                     except Exception as fi_err:
                         logger.warning(f"Could not plot Feature Importance for {model_name}: {fi_err}")
                 else:
                      logger.info(f"Skipping Feature Importance for {model_name} (not available or model not fitted)")

            # Compare model performances (including RMSE)
            if all_results_plot:
                self.visualizer.plot_model_comparison(all_results_plot)
                self.visualizer.plot_confusion_matrices(all_results_plot)


            # --- Data Exploration Plots ---
            # Use original processed_data (before scaling/splitting) passed as processed_data_viz
            if processed_data is not None and not processed_data.empty:
                key_features = [col for col in [
                    'funding_amount_log', 'employees',
                    'employee_efficiency', 'previous_rounds',
                    'months_since_first_funding', 'funding_year', 'funding_month', 'funding_velocity'
                ] if col in processed_data.columns]

                # Ensure we only plot if key_features are actually present
                valid_key_features = [kf for kf in key_features if kf in processed_data.columns]

                try:
                    # Plot stage distribution using the correct reverse mapping
                    if hasattr(self, 'reverse_final_class_mapping'):
                        self.visualizer.plot_funding_stage_distribution(
                            processed_data,
                            stage_mapping_rev=self.reverse_final_class_mapping)
                    else:
                         logger.warning("Reverse class mapping not found, cannot plot stage distribution with correct labels.")
                         # Optionally call with default mapping or skip
                         # self.visualizer.plot_funding_stage_distribution(processed_data_viz)

                except Exception as viz_err:
                    logger.warning(f"Error plotting funding stage distribution: {viz_err}")

                # Other data visualizations
                if valid_key_features:
                    try:
                        self.visualizer.plot_temporal_trends(processed_data)
                    except Exception as viz_err:
                         logger.warning(f"Error plotting temporal trends: {viz_err}")
                    try:
                        self.visualizer.plot_funding_vs_employees(processed_data)
                    except Exception as viz_err:
                         logger.warning(f"Error plotting funding vs employees: {viz_err}")
                    try:
                        # Plot correlations including the target variable
                        cols_for_corr = valid_key_features + ['funding_stage_numeric']
                        self.visualizer.plot_correlation_heatmap(processed_data[cols_for_corr])
                    except Exception as viz_err:
                         logger.warning(f"Error plotting correlation heatmap: {viz_err}")
                    # Add calls to other desired Visualizer plots here, e.g.:
                    # try:
                    #     self.visualizer.plot_industry_distributions(processed_data_viz)
                    # except Exception as viz_err:
                    #      logger.warning(f"Error plotting industry distributions: {viz_err}")
                    # try:
                    #     self.visualizer.plot_violin_funding_by_stage(processed_data_viz)
                    # except Exception as viz_err:
                    #      logger.warning(f"Error plotting violin funding: {viz_err}")

                else:
                    logger.warning("Skipping some data visualizations as key features are missing from processed_data.")
            else:
                 logger.warning("Skipping data exploration plots as processed_data_viz is empty or None.")


        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            logger.error(traceback.format_exc()) # Log full traceback

    def _save_summary(self, merged_data, X, model_results_summary, processed_data): # <<< Added processed_data
        # Save metrics as JSON
        summary = {
            'run_timestamp': self.timestamp,
            'data_shape': {
                'initial_records_merged': len(merged_data),
                'records_after_preprocessing': X.shape[0] if hasattr(X, 'shape') else 'N/A',
                'features_used': X.shape[1] if hasattr(X, 'shape') else 'N/A'
            },
            'feature_names': self.feature_names,
             'class_mapping': {str(k): str(v) for k, v in getattr(self, 'final_index_to_string_label_map', {}).items()},
            # +++ Add age bin info +++
            'age_bin_edges': getattr(self.feature_engineer, 'age_bin_edges', None),
            'age_bin_labels': getattr(self.feature_engineer, 'age_bin_labels', None),
            # +++ End age bin info +++
            'metrics': {}
        }

        # Add metrics for each model
        best_accuracy = -1.0
        best_model_name = None
        for model_name, results in model_results_summary.items():
            if results is None: # Skip if a model failed (e.g., LGBM not installed or failed)
                 summary['metrics'][model_name] = None
                 continue
            # Safely get metrics using .get()
            accuracy = float(results.get('accuracy', np.nan))
            summary['metrics'][model_name] = {
                'accuracy': accuracy, # Use the variable
                'precision_weighted': float(results.get('precision', np.nan)),
                'recall_weighted': float(results.get('recall', np.nan)),
                'f1_weighted': float(results.get('f1_score', np.nan)),
                'rmse': float(results.get('rmse', np.nan)),
                'cross_val_accuracy_mean': float(np.mean(results.get('cross_val_scores', [np.nan]))),
                'cross_val_accuracy_std': float(np.std(results.get('cross_val_scores', [np.nan]))),
                'classification_report': results.get('classification_report', None)
            }
            # Determine best model based on accuracy
            if not pd.isna(accuracy) and accuracy > best_accuracy:
                 best_accuracy = accuracy
                 best_model_name = model_name

        # Add the name of the best model found to the summary
        summary['best_model_by_accuracy'] = best_model_name

        # --- Calculate and add Benchmark Statistics ---
        try:
             benchmarks = {}
             # Ensure the necessary columns exist in processed_data
             # We need the original string stage labels for clear benchmark keys
             if 'funding_stage' in processed_data.columns and \
                'funding_stage_numeric' in processed_data.columns and \
                'funding_amount' in processed_data.columns and \
                'employees' in processed_data.columns:

                 # --- Use the *original* string funding_stage for grouping ---
                 # Create a mapping from the final numeric index back to the original string label if needed,
                 # OR group directly by the original 'funding_stage' string column.
                 # Let's group by the original string column for clarity.
                 grouped_by_string_label = processed_data.groupby('funding_stage')

                 for stage_label, group_data in grouped_by_string_label:
                     # Use the string label directly as the key
                     stage_key = str(stage_label) # Ensure it's a string for JSON
                     benchmarks[stage_key] = {
                         'funding_amount_median': float(group_data['funding_amount'].median()) if pd.notna(group_data['funding_amount'].median()) else None,
                         'funding_amount_q1': float(group_data['funding_amount'].quantile(0.25)) if pd.notna(group_data['funding_amount'].quantile(0.25)) else None,
                         'funding_amount_q3': float(group_data['funding_amount'].quantile(0.75)) if pd.notna(group_data['funding_amount'].quantile(0.75)) else None,
                         'employees_median': float(group_data['employees'].median()) if pd.notna(group_data['employees'].median()) else None,
                         'employees_q1': float(group_data['employees'].quantile(0.25)) if pd.notna(group_data['employees'].quantile(0.25)) else None,
                         'employees_q3': float(group_data['employees'].quantile(0.75)) if pd.notna(group_data['employees'].quantile(0.75)) else None,
                         'count': int(len(group_data)) # Add count for context
                     }
                 summary['benchmarks'] = benchmarks
                 logger.info("Successfully calculated and added benchmark statistics (keyed by string label).")
             else:
                 summary['benchmarks'] = None
                 logger.warning("Could not calculate benchmarks: Required columns missing in processed_data.")
        except Exception as bench_err:
             summary['benchmarks'] = None
             logger.error(f"Error calculating benchmark statistics: {bench_err}")
             logger.error(traceback.format_exc())
        # --- End Benchmark Statistics ---


        # Extract and sort feature importance from the best *base* model
        # (Ensemble doesn't have FI on original features)
        best_base_model_importance = None
        importance_source_model = None
        feature_names_imp = self.feature_names # Default

        # Determine best BASE model by accuracy (excluding ensemble)
        base_model_accuracies = {
             name: metrics['accuracy']
             for name, metrics in summary['metrics'].items()
             if metrics is not None and name != 'Stacking Ensemble (LR Meta)' and not np.isnan(metrics.get('accuracy', np.nan))
        }

        best_base_model_name = None
        if base_model_accuracies:
             # Find the base model with the highest accuracy
             best_base_model_name = max(base_model_accuracies.items(), key=lambda x: x[1])[0]
             logger.info(f"Attempting FI extraction from best base model: {best_base_model_name}")

             if best_base_model_name in model_results_summary:
                 results = model_results_summary[best_base_model_name]
                 uncalibrated_model = results.get('model')
                 if uncalibrated_model and hasattr(uncalibrated_model, 'feature_importances_') and results.get('feature_names'):
                      importance = uncalibrated_model.feature_importances_
                      temp_feature_names = results['feature_names']
                      if len(importance) == len(temp_feature_names):
                           importance_source_model = best_base_model_name
                           best_base_model_importance = importance
                           feature_names_imp = temp_feature_names
                           logger.info(f"Successfully extracted FI from {best_base_model_name}")
                      else:
                           logger.warning(f"FI length mismatch for {best_base_model_name}")
                 else:
                      logger.warning(f"Could not get model/FI/names for FI from {best_base_model_name}")
             else:
                  logger.warning(f"Results not found for determined best base model {best_base_model_name}")
        else:
             logger.warning("No valid base models found to determine best for FI.")

        # Save the extracted FI (if any)
        if importance_source_model and best_base_model_importance is not None:
            feature_importance_dict = {
                feature: float(imp) for feature, imp in zip(feature_names_imp, best_base_model_importance)
            }
            summary['feature_importance'] = {
                'source_model': importance_source_model,
                'importance': dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))
            }
        else:
             summary['feature_importance'] = None
             logger.warning("Could not extract valid feature importance from any base model for summary.")

        # Determine best OVERALL model to save (can be base or ensemble)
        best_overall_model_name = summary.get('best_model_by_accuracy')
        model_to_save = None
        scaler_to_save = None
        metadata_to_save = {}
        feature_names_to_save = self.feature_names # Use original features for saving interface

        if best_overall_model_name and best_overall_model_name in model_results_summary:
            results_to_use = model_results_summary[best_overall_model_name]
            if results_to_use:
                metadata_to_save = summary['metrics'].get(best_overall_model_name, {})

                # --- Updated Logic to Handle Ensemble Saving --- #
                is_ensemble = 'Stacking Ensemble' in best_overall_model_name

                if is_ensemble:
                    model_to_save = results_to_use.get('model') # Save the meta-model
                    feature_names_to_save = self.feature_names # Original features needed for base models
                    # Find scaler from a reliable base model
                    scaler_found = False
                    # Prioritize base models likely used in the ensemble
                    for base_key_lookup in [
                        'LightGBM', # Tuned LGBM if present
                        'XGBoost (Calibrated)',
                        'Gradient Boosting (Calibrated)',
                        'Random Forest (Calibrated)'
                        # Add others only if strictly necessary as fallback
                        # 'SVC (Linear, Calibrated)', 
                        # 'MLP Classifier (Calibrated)',
                        # 'K-Nearest Neighbors (Calibrated)',
                        # 'Logistic Regression (Calibrated)'
                    ]:
                        if base_key_lookup in model_results_summary and model_results_summary[base_key_lookup]:
                            scaler_candidate = model_results_summary[base_key_lookup].get('scaler')
                            if scaler_candidate is not None:
                                scaler_to_save = scaler_candidate
                                logger.info(f"Using scaler from base model '{base_key_lookup}' for saving with ensemble.")
                                scaler_found = True
                                break # Stop once a scaler is found
                    if not scaler_found:
                         logger.error("Scaler not found from any base model for ensemble saving.")
                         scaler_to_save = None # Ensure scaler is None if not found
                else: # Logic for saving a BASE model
                    model_to_save = results_to_use.get('calibrated_model') # Save calibrated base model
                    scaler_to_save = results_to_use.get('scaler')
                    feature_names_to_save = results_to_use.get('feature_names') # Use feature names from base model results
                # --- End Updated Logic --- #

                # Final check on components before saving
                if not model_to_save: logger.error(f"Model object missing for {best_overall_model_name}")
                if not scaler_to_save: logger.error(f"Scaler object missing for {best_overall_model_name}")
                if not feature_names_to_save: logger.error(f"Feature names missing for {best_overall_model_name}")

            else: # results_to_use is None
                 logger.error(f"Results dict missing for {best_overall_model_name}")
                 best_overall_model_name = None # Prevent saving
        else: # best_overall_model_name is None or invalid
            logger.error(f"Best overall model name '{best_overall_model_name}' invalid. Cannot save.")
            best_overall_model_name = None

        # --- Actual Saving Call --- #
        # 1. Save the overall best model (could be base or ensemble)
        saved_path = None # Initialize
        if best_overall_model_name and model_to_save and scaler_to_save and feature_names_to_save:
            # Sanitize name for filename
            safe_model_name = best_overall_model_name.replace(" ", "_").replace("(", "").replace(")", "")
            # Get the anomaly detector associated with the best model
            anomaly_detector_to_save = results_to_use.get('anomaly_detector')
            if not anomaly_detector_to_save:
                logger.warning(f"Anomaly detector instance not found in results for {best_overall_model_name}. Model will be saved without it.")

            saved_path = self.model_manager.save_model(
                model_name=safe_model_name,
                model=model_to_save,
                scaler=scaler_to_save,
                feature_names=feature_names_to_save,
                metadata={'training_metadata': metadata_to_save, # Wrap metrics dict
                          'class_mapping': getattr(self, 'final_index_to_string_label_map', {})},
                anomaly_detector=anomaly_detector_to_save # Pass the detector
            )
            if saved_path:
                 logger.info(f"Successfully saved best model ({best_overall_model_name}) and scaler to {saved_path}")
                 # +++ Add file existence check +++
                 if not os.path.exists(saved_path):
                     logger.error(f"!!! FILE NOT FOUND AFTER SAVING: {saved_path} !!!")
                 else:
                     logger.info(f"Confirmed file exists: {saved_path}")
                 # +++ End file existence check +++
            else:
                 logger.error(f"Failed to save the best model ({best_overall_model_name}).")
        else:
            logger.error(f"Could not save best model '{best_overall_model_name}' - model or scaler missing.")

        # --- 2. Explicitly save the best model with Dashboard_Model prefix --- #
        if best_overall_model_name and model_to_save and scaler_to_save and feature_names_to_save:
            # Determine the name based on whether it's an ensemble
            # For now, let's assume the best model is likely a base model for dashboard simplicity
            # If it IS an ensemble, we might need specific handling or save a simpler base model instead.
            safe_best_model_name_dash = best_overall_model_name.replace(" ", "_").replace("(", "").replace(")", "") # Use same sanitized name
            dashboard_model_name = f"Dashboard_Model_{safe_best_model_name_dash}" # Include original best model type in name
            logger.info(f"Attempting to save best model also as {dashboard_model_name}...")
            # Retrieve anomaly detector again for this save call
            anomaly_detector_dash_save = results_to_use.get('anomaly_detector')
            if not anomaly_detector_dash_save:
                 logger.warning(f"Anomaly detector not found for dashboard save of {best_overall_model_name}")

            dashboard_saved_path = self.model_manager.save_model(
                 model_name=dashboard_model_name, # Use the specific Dashboard_Model prefix
                 model=model_to_save,
                 scaler=scaler_to_save,
                 feature_names=feature_names_to_save,
                 metadata={'training_metadata': metadata_to_save, # Wrap metrics dict
                           'class_mapping': getattr(self, 'final_index_to_string_label_map', {})},
                 anomaly_detector=anomaly_detector_dash_save # Pass the same detector
            )
            if dashboard_saved_path:
                 logger.info(f"Successfully saved best model ({best_overall_model_name}) with dashboard prefix to {dashboard_saved_path}")
                 # Optional: Verify existence
                 if not os.path.exists(dashboard_saved_path):
                     logger.error(f"!!! FILE NOT FOUND AFTER SAVING DASHBOARD MODEL: {dashboard_saved_path} !!!")
                 else:
                     logger.info(f"Confirmed dashboard model file exists: {dashboard_saved_path}")
                     # Moved the success log inside the final else, ensuring file exists first
                     logger.info(f"Successfully saved best model ({best_overall_model_name}) with dashboard prefix to {dashboard_saved_path}")
            else:
                 logger.error(f"Failed to save the best model ({best_overall_model_name}) with the dashboard prefix.")
        else:
            logger.error("Could not save best model with dashboard prefix - components missing.")
        # --- End Saving Call --- #

        # --- Explicitly Save Key Base Models for Dashboard --- #
        logger.info("Attempting to explicitly save key base models for dashboard use...")
        base_models_to_save = {
            'XGBoost (Calibrated)': 'XGBoost_(Calibrated)', # Key in results dict: Sanitized name for saving
            'Random Forest (Calibrated)': 'Random_Forest_(Calibrated)',
            'Gradient Boosting (Calibrated)': 'Gradient_Boosting_(Calibrated)', # Added GB
            'Decision Tree (Calibrated)': 'Decision_Tree_(Calibrated)' # Added DT
        }

        for model_key, save_name_suffix in base_models_to_save.items():
            if model_key in model_results_summary and model_results_summary[model_key]:
                results_data = model_results_summary[model_key]
                base_model_to_save = results_data.get('calibrated_model')
                base_scaler_to_save = results_data.get('scaler')
                base_feature_names = results_data.get('feature_names')
                base_metrics = summary['metrics'].get(model_key, {}) # Get metrics
                base_anomaly_detector = results_data.get('anomaly_detector') # Get detector
                
                if base_model_to_save and base_scaler_to_save and base_feature_names:
                    # Save with the standard name
                    logger.info(f"Saving base model: {model_key} as {save_name_suffix}")
                    base_saved_path = self.model_manager.save_model(
                        model_name=save_name_suffix,
                        model=base_model_to_save,
                        scaler=base_scaler_to_save,
                        feature_names=base_feature_names,
                        metadata={'training_metadata': base_metrics, 
                                  'class_mapping': getattr(self, 'final_index_to_string_label_map', {})},
                        anomaly_detector=base_anomaly_detector 
                    )
                    if base_saved_path:
                        logger.info(f"Successfully saved {model_key} to {base_saved_path}")
                        if not os.path.exists(base_saved_path):
                             logger.error(f"!!! FILE NOT FOUND AFTER SAVING BASE MODEL: {base_saved_path} !!!")
                        else:
                             logger.info(f"Confirmed base model file exists: {base_saved_path}")
                        
                        # Also save with "Dashboard_Model_" prefix if it's one of the preferred ones
                        if model_key in ['XGBoost (Calibrated)', 'Random Forest (Calibrated)']:
                            dashboard_prefixed_name = f"Dashboard_Model_{save_name_suffix}"
                            logger.info(f"Also saving {model_key} as {dashboard_prefixed_name} for dashboard direct load.")
                            dashboard_specific_save_path = self.model_manager.save_model(
                                model_name=dashboard_prefixed_name,
                                model=base_model_to_save,
                                scaler=base_scaler_to_save,
                                feature_names=base_feature_names,
                                metadata={'training_metadata': base_metrics,
                                          'class_mapping': getattr(self, 'final_index_to_string_label_map', {})},
                                anomaly_detector=base_anomaly_detector
                            )
                            if dashboard_specific_save_path:
                                logger.info(f"Successfully saved {model_key} to {dashboard_specific_save_path} for dashboard.")
                            else:
                                logger.error(f"Failed to save {model_key} with dashboard prefix.")

                    else:
                        logger.error(f"Failed to save base model: {model_key}")
                else:
                     logger.warning(f"Skipping save for {model_key} - missing model, scaler, or feature names in results.")
            else:
                logger.warning(f"Results for base model {model_key} not found or invalid in summary. Cannot save explicitly.")
        # --- End Explicit Base Model Save --- #

        # Print comparison table to console
        print("\n===== MODEL PERFORMANCE COMPARISON =====")
        print(f"{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'RMSE':<12}") # Increased width for model name
        print("-" * 95) # Adjusted separator length

        # Sort models for printing by accuracy (descending), handle None metrics
        sorted_models = sorted(
            summary['metrics'].items(),
            key=lambda item: item[1]['accuracy'] if item[1] and not pd.isna(item[1]['accuracy']) else -1,
            reverse=True
        )

        for model_name, metrics in sorted_models: # Use sorted list
            if metrics is None:
                print(f"{model_name:<30} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue

            # Format metrics, handling potential NaN or None
            acc = f"{metrics.get('accuracy', np.nan) * 100:.2f} %" if not pd.isna(metrics.get('accuracy', np.nan)) else 'N/A'
            prec = f"{metrics.get('precision_weighted', np.nan) * 100:.2f} %" if not pd.isna(metrics.get('precision_weighted', np.nan)) else 'N/A'
            rec = f"{metrics.get('recall_weighted', np.nan) * 100:.2f} %" if not pd.isna(metrics.get('recall_weighted', np.nan)) else 'N/A'
            f1 = f"{metrics.get('f1_weighted', np.nan) * 100:.2f} %" if not pd.isna(metrics.get('f1_weighted', np.nan)) else 'N/A'
            rmse_val = metrics.get('rmse', np.nan)
            rmse = f"{rmse_val:.4f}" if not pd.isna(rmse_val) else 'N/A'

            print(f"{model_name:<30} {acc:<12} {prec:<12} {rec:<12} {f1:<12} {rmse:<12}")

        print("=" * 95)

        # --- Save Summary JSON --- #
        summary_path = os.path.join(self.output_dir, f"summary_{self.timestamp}.json")
        try:
            # +++ Add Logging before saving +++
            # Log the final mapping being saved
            final_map_to_save = summary.get('class_mapping', {})
            logger.info(f"Final Class mapping being saved to summary (Index -> Label): {final_map_to_save}")
            # +++ End Logging +++
            with open(summary_path, 'w') as f:
                # Use the NumpyEncoder to handle potential numpy types
                json.dump(summary, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Pipeline summary saved to {summary_path}")
        except Exception as json_err:
            logger.error(f"Failed to save summary JSON: {json_err}")
            logger.error(traceback.format_exc())
        # --- End Save Summary JSON --- #

        return summary # Make sure summary is returned

    def make_prediction(self, sample_data):
        """Make prediction with best available model"""
        # Identify the best model name from the latest summary file
        latest_summary_path = None
        try:
            summary_files = glob.glob(os.path.join(self.output_dir, "summary_*.json"))
            if not summary_files:
                logger.error("No summary files found in output directory. Cannot determine best model.")
                return {'error': 'No summary file found'}
            latest_summary_path = max(summary_files, key=os.path.getctime)
            with open(latest_summary_path, 'r') as f:
                summary_data = json.load(f)
            best_model_name = summary_data.get('best_model_by_accuracy')
            if not best_model_name:
                logger.error(f"Best model name not found in latest summary: {latest_summary_path}")
                return {'error': 'Best model name missing in summary'}

            # Sanitize the name to match potential saved filenames
            safe_model_name = best_model_name.replace(" ", "_").replace("(", "").replace(")", "")
            logger.info(f"Attempting to load best model: {best_model_name} (Filename pattern: {safe_model_name})")

        except Exception as e:
            logger.error(f"Error reading latest summary file: {e}")
            return {'error': 'Failed to read summary file'}


        # Load the determined best model using the sanitized name
        # Use the new load_model_joblib method
        loaded_ok = self.model_manager.load_model_joblib(model_name=safe_model_name)
        if not loaded_ok:
            # Fallback logic (try loading common names if best failed) - could be removed if strict adherence to summary is desired
             logger.warning(f"Failed to load '{safe_model_name}', trying fallbacks...")
             fallbacks = ["Stacking_Ensemble_GB_Meta", "XGBoost_Calibrated", "Random_Forest_Calibrated"]
             for fb_name in fallbacks:
                 logger.info(f"Trying fallback: {fb_name}")
                 loaded_ok = self.model_manager.load_model_joblib(model_name=fb_name)
                 if loaded_ok:
                     logger.info(f"Successfully loaded fallback model: {fb_name}")
                     break
             if not loaded_ok:
                 logger.error("Could not load the best model or any fallback models.")
                 return {'error': 'No suitable model found'}

        # Ensure scaler and feature names are loaded by load_model_joblib
        if self.model_manager.scaler is None or not self.model_manager.feature_names:
            logger.error("Scaler or feature names not loaded with the model.")
            return {'error': 'Model loaded without scaler/feature names'}

        # Format sample data
        try:
            if isinstance(sample_data, dict):
                # Use the feature names loaded with the model
                feature_columns = self.model_manager.feature_names
                # Create a pandas Series/DataFrame in the correct order, handling missing values
                ordered_data = pd.Series(index=feature_columns, dtype=float)
                missing_input_features = []
                for col in feature_columns:
                     if col in sample_data:
                         ordered_data[col] = sample_data[col]
                     else:
                         ordered_data[col] = 0 # Default missing features to 0 (consider median/mean?)
                         missing_input_features.append(col)
                if missing_input_features:
                     logger.warning(f"Input data missing keys: {missing_input_features}. Defaulted to 0.")

                features_df = pd.DataFrame([ordered_data])
            elif isinstance(sample_data, (list, np.ndarray)):
                 if len(sample_data) != len(self.model_manager.feature_names):
                      return {'error': f'Input data has {len(sample_data)} features, expected {len(self.model_manager.feature_names)}'}
                 features_df = pd.DataFrame([sample_data], columns=self.model_manager.feature_names)
            else:
                 return {'error': 'Invalid sample_data format. Provide dict, list, or numpy array.'}
        except Exception as format_err:
            logger.error(f"Error formatting prediction input: {format_err}")
            return {'error': 'Input formatting error'}

        # Make prediction using the loaded model and manager's predict method (which handles scaling)
        # The predict method needs the features in the correct format (e.g., dictionary or Series)
        # Pass the DataFrame row as a dictionary
        # Extract company name if provided in the sample data for the manager's predict method
        company_name_input = sample_data.get('company_name', None) if isinstance(sample_data, dict) else None
        prediction_result = self.model_manager.predict(features_df.iloc[0].to_dict(), company_name=company_name_input)

        # Map back to original funding stage using the pipeline's stored reverse mapping
        # Load the mapping from the summary file if not already loaded in the pipeline instance
        # (This assumes make_prediction might be called standalone)
        reverse_map = getattr(self, 'reverse_final_class_mapping', None)
        if not reverse_map and latest_summary_path:
             try:
                 with open(latest_summary_path, 'r') as f:
                     summary_data = json.load(f)
                 # Class mapping keys in JSON are strings, convert back to int
                 final_map_from_summary = {int(k): v for k, v in summary_data.get('class_mapping', {}).items()}
                 reverse_map = {v: k for k, v in final_map_from_summary.items()} # Create reverse map
             except Exception as map_load_err:
                 logger.error(f"Could not load class mapping from summary for label conversion: {map_load_err}")


        if 'prediction' in prediction_result and reverse_map:
             predicted_class_index = prediction_result['prediction']
             # Use the numeric stage index from the model's prediction
             predicted_stage_numeric = int(predicted_class_index)
             # Find the *original* stage label using the reverse mapping derived from the summary
             predicted_stage_label = reverse_map.get(predicted_stage_numeric, f"Unknown Stage (Index {predicted_stage_numeric})")

             prediction_result['predicted_stage_label'] = predicted_stage_label
             prediction_result['predicted_stage_numeric'] = predicted_stage_numeric # Also return the numeric value
        elif 'error' not in prediction_result:
             logger.warning("Could not map prediction index back to stage label (mapping missing?).")
             prediction_result['predicted_stage_label'] = f"Unknown (Class {prediction_result.get('prediction')})"


        return prediction_result


class AnomalyDetector:
    """Detects anomalies and potential manipulation in startup data"""

    def __init__(self, contamination=0.01): # <<< Lowered contamination
        """Initialize detector with contamination parameter (expected outlier ratio)"""
        # Need to import IsolationForest here if not globally imported
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("IsolationForest not installed. Skipping anomaly detection.")
            self.isolation_forest = None # Set to None if import fails
            self.contamination = contamination # Still store contamination
            self.feature_ranges = {}
            self.startup_data_cache = {}
            self.known_companies = set()
            return # Exit init if import failed

        # Initialize IsolationForest if import succeeded
        self.isolation_forest = IsolationForest(
            contamination=contamination, # Use the passed contamination value
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.contamination = contamination # Store contamination
        self.feature_ranges = {}
        self.startup_data_cache = {}
        self.known_companies = set()
        # --- END OF __init__ --- #

    # --- fit METHOD starts here --- #
    def fit(self, X, startup_names=None):
        """Train anomaly detection model on startup data

        Args:
            X: Feature matrix for startups
            startup_names: Optional list of company names
        """
        # Ensure IsolationForest was initialized
        if self.isolation_forest is None:
             logger.error("IsolationForest not initialized, cannot fit.")
             return False
        try:
            # Train isolation forest for outlier detection
            self.isolation_forest.fit(X) # Fit the initialized forest

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
                    f"Feature dimension mismatch: expected {
                        len(
                            self.feature_ranges['min'])}, got {
                        X.shape[1]}")
                return {
                    'is_anomaly': True,
                    'reason': 'dimension_mismatch',
                    'score': 1.0}

            # Run standard anomaly checks
            anomalies = {
                'is_anomaly': False,
                'score': 0.0,
                'reasons': []
            }

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

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {
                'is_anomaly': True,
                'reason': f'detection_error: {
                    str(e)}',
                'score': 1.0}


# +++  TimeSeriesForecaster Class +++
class TimeSeriesForecaster:
    def __init__(self, output_dir="./visualizations"):
        """Initialize time series forecaster"""
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_prophet_data(self, data):
        """Aggregate data monthly and prepare for Prophet"""
        logger.info("Preparing data for Prophet...")
        # Ensure 'funding_date' is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['funding_date']):
             data['funding_date'] = pd.to_datetime(data['funding_date'], errors='coerce')

        # Ensure regressors are numeric
        data['funding_amount_log'] = pd.to_numeric(data['funding_amount_log'], errors='coerce')

        # Drop rows where date conversion failed or target/regressors are missing
        data = data.dropna(subset=['funding_date', 'funding_stage_numeric', 'funding_amount_log'])
        if data.empty:
             logger.warning("No valid date data found for time series analysis.")
             return None

        # --- Corrected Aggregation --- #
        # Set date as index
        monthly_data = data.set_index('funding_date')

        # --- REMOVED: Filter Data from 2024 onwards --- #
        # monthly_data = monthly_data[monthly_data.index >= '2024-01-01']
        # if monthly_data.empty:
        #     logger.warning("No data found from 2024-01-01 onwards for time series analysis.")
        #     return None
        # logger.info(f"Using data from {monthly_data.index.min()} to {monthly_data.index.max()} for quarterly analysis.")

        # +++ Log Raw Quarterly Counts Before Aggregation (Full History) +++
        raw_quarterly_counts = monthly_data.resample('QS').size() # Resample by Quarter Start
        logger.info(f"Raw quarterly deal counts (Full History, before aggregation/filtering):\n{raw_quarterly_counts.to_string()}")
        # +++ End Logging +++ #

        # --- Implement Rolling Window Aggregation (4-Quarter window) --- #
        window_size = 4 # Quarters # <<< REVERTED WINDOW SIZE FROM 6 back to 4
        # Select only numeric columns needed for resampling median
        numeric_cols_for_resample = ['funding_stage_numeric', 'funding_amount_log']
        # Resample only the numeric columns needed by Quarter Start
        quarterly_resampled = monthly_data[numeric_cols_for_resample].resample('QS').median()

        # Calculate rolling statistics on the resampled numeric data
        quarterly_agg = pd.DataFrame(index=quarterly_resampled.index)
        # --- Use EWMA instead of Rolling Median for smoother historical y ---
        quarterly_agg['y'] = quarterly_resampled['funding_stage_numeric'].ewm(span=4, adjust=False).mean() # EWMA for target
        # quarterly_agg['median_funding_log'] = quarterly_resampled['funding_amount_log'].rolling(window=window_size, min_periods=1).median()
        # quarterly_agg['deal_count'] = quarterly_counts.rolling(window=window_size, min_periods=1).sum()

        # For deal count, get the size per quarter separately and apply rolling sum
        quarterly_counts = monthly_data.resample('QS').size()
        quarterly_agg['deal_count'] = quarterly_counts.rolling(window=window_size, min_periods=1).sum()

        # Calculate rolling median for funding amount log separately to handle potential NaNs after resampling
        # --- Use EWMA for regressor as well --- #
        quarterly_funding_log = monthly_data['funding_amount_log'].resample('QS').median()
        # quarterly_agg['median_funding_log'] = quarterly_funding_log.rolling(window=window_size, min_periods=1).median()
        quarterly_agg['median_funding_log'] = quarterly_funding_log.ewm(span=4, adjust=False).mean() # EWMA for regressor


        # Reset index to get 'ds' column
        prophet_df = quarterly_agg.reset_index()
        prophet_df = prophet_df.rename(columns={'funding_date': 'ds'})
        # --- End Rolling Window Aggregation --- #

        # Drop quarters with no data (NaN median/count after aggregation)
        prophet_df = prophet_df.dropna()

        if prophet_df.empty: # Add check after dropna
            logger.warning("No valid quarterly data points remaining after aggregation and dropna.")
            return None

        logger.info(f"Prepared {len(prophet_df)} quarterly data points with regressors for Prophet.")
        return prophet_df

    def train_predict(self, prophet_df, periods=6, freq='MS'):
        """Train Prophet model and make future predictions"""
        if prophet_df is None or prophet_df.empty:
             logger.error("Cannot train Prophet model: Input data is empty.")
             return None, None # Corrected indentation

        # Check if regressor columns exist
        required_regressors = ['median_funding_log', 'deal_count']
        missing_regressors = [reg for reg in required_regressors if reg not in prophet_df.columns]
        if missing_regressors:
             logger.error(f"Missing required regressor columns in Prophet data: {missing_regressors}")
             return None, None
 
        # Add check for minimum data length
        min_data_points = 4 # Require at least 4 Quarters of data
        if len(prophet_df) < min_data_points:
            logger.warning(f"Insufficient historical data ({len(prophet_df)} quarters) for reliable Prophet forecast. Minimum required: {min_data_points}. Skipping forecast.")
            return None, None

        # --- Adjust forecast period and frequency for Quarterly data ---
        forecast_periods = 2 # Forecast 2 quarters ahead
        forecast_freq = 'QS' # Quarter Start frequency

        logger.info(f"Training Prophet model to forecast {forecast_periods} quarters ({forecast_freq})...") # Updated log message
        try:
            # Initialize and fit model
            # Reduce n_changepoints if data is sparse
            n_points = len(prophet_df)
            n_changepoints = min(25, n_points - 1) if n_points > 1 else 0 # Default is 25

            model = Prophet(
                n_changepoints=n_changepoints, # Adjust changepoints based on data length
                # --- Increase Regularization ---
                changepoint_prior_scale=0.01, # Default 0.05 - Make trend less flexible
                seasonality_prior_scale=1.0,  # Default 10.0 / Prev 0.5 - REVERTED
                yearly_seasonality=True,      # Keep yearly for quarterly patterns
                # --- End Regularization ---
                weekly_seasonality=False, # Weekly doesn't make sense for monthly funding stages
                daily_seasonality=False,
                interval_width=0.95 # 95% confidence interval
             )

            # Add regressors before fitting
            model.add_regressor('median_funding_log')
            model.add_regressor('deal_count')

            model.fit(prophet_df)
 
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_periods, freq=forecast_freq) # Use adjusted period/freq
 
            # --- Prepare future regressors --- #
            # For the forecast plot, we need values for regressors in the future dataframe.
            # Simplest approach: Carry forward the last known value or use a simple trend.
            # Let's use a simple mean of the last few months for this example.
            last_months = 3
            future_median_funding_log = prophet_df['median_funding_log'].iloc[-last_months:].mean()
            future_deal_count = prophet_df['deal_count'].iloc[-last_months:].mean()

            # Fill future dataframe with these values
            future['median_funding_log'] = future_median_funding_log
            future['deal_count'] = future_deal_count

            # Overwrite historical part with actual values
            future.loc[future['ds'].isin(prophet_df['ds']), 'median_funding_log'] = prophet_df['median_funding_log']
            future.loc[future['ds'].isin(prophet_df['ds']), 'deal_count'] = prophet_df['deal_count']

            # --- Add Robust NaN handling for future regressors --- #
            future['median_funding_log'] = future['median_funding_log'].fillna(prophet_df['median_funding_log'].median()) # Fill with historical median if future calc is NaN
            future['deal_count'] = future['deal_count'].fillna(prophet_df['deal_count'].median()) # Fill with historical median if future calc is NaN

            # Make predictions
            forecast = model.predict(future)

            # --- Enforce non-negative floor on predictions ---
            forecast['yhat'] = forecast['yhat'].clip(lower=0.0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0.0)
            # --- End non-negative floor ---

            logger.info("Prophet model trained and forecast generated.")
            return model, forecast
        except Exception as e:
             logger.error(f"Error during Prophet model training or prediction: {e}")
             logger.error(traceback.format_exc())
             return None, None

    def plot_forecast(self, model, forecast):
        """Plot historical data and forecast"""
        if model is None or forecast is None:
             logger.error("Cannot plot forecast: Model or forecast data missing.")
             return

        logger.info("Generating Prophet forecast plot...")
        try:
            # +++ Increase figure size +++
            fig = plt.figure(figsize=(15, 6)) # Increased width from default
            ax = fig.add_subplot(111)
            model.plot(forecast, ax=ax) # Pass the axes to prophet plot

            plt.title('Funding Stage Trend Forecast (Median Stage) - 6 Months', fontsize=14) # Updated title for 6 months
            plt.xlabel('Date')
            plt.ylabel('Median Funding Stage (Numeric)')
            plt.grid(True, linestyle='--', alpha=0.7)

            # Add vertical line to distinguish history from forecast
            last_hist_date = model.history_dates.max()
            plt.axvline(last_hist_date, color='r', linestyle='--', lw=1, label='Forecast Start')
            plt.legend()

            plot_path = os.path.join(self.output_dir, f"prophet_forecast_{self.timestamp}.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            logger.info(f"Prophet forecast plot saved to {plot_path}")
            plt.close(fig) # Close the figure to free memory

            # Optionally plot components
            # fig_comp = model.plot_components(forecast)
            # comp_plot_path = os.path.join(self.output_dir, f"prophet_components_{self.timestamp}.png")
            # plt.savefig(comp_plot_path)
            # plt.close(fig_comp)
            # logger.info(f"Prophet forecast components plot saved to {comp_plot_path}")

        except Exception as e:
            logger.error(f"Error generating Prophet plot: {e}")
            logger.error(traceback.format_exc())
            plt.close() # Ensure plot is closed even if error occurs

    def plot_dashboard_prototype(self, forecast, history_df):
        """Plot historical actuals and forecast (if provided) for the dashboard prototype as an interactive HTML."""
        # Import plotly graph objects here
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error("Plotly not installed. Cannot create interactive plot. `pip install plotly`")
            return # Cannot proceed without plotly

        if history_df is None or history_df.empty:
             logger.error("Cannot plot dashboard prototype: Historical data missing for comparison.")
             return

        logger.info("Generating interactive dashboard prototype plot (HTML)...") # Updated log message
        prototype_output_dir = os.path.join(self.output_dir, "..", "prototype_dashboard") # Go up one level from viz dir
        os.makedirs(prototype_output_dir, exist_ok=True)

        try:
            fig = go.Figure()

            # Plot historical actuals (using y from history_df)
            fig.add_trace(go.Scatter(
                x=history_df['ds'],
                y=history_df['y'],
                mode='lines+markers',
                name='Historical EWMA Stage (4-Qtr Span)', # Updated label
                line=dict(color='black'),
                marker=dict(size=4)
            ))

            plot_title = 'Bay Area Startups: Historical EWMA Funding Stage (4-Quarter Span)' # Default title

            # --- Plot forecast and fit only if forecast is provided --- #
            if forecast is not None and not forecast.empty:
                # Merge forecast with history dates for easy plotting
                plot_df = history_df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')

                # Plot historical fit (in-sample yhat)
                fig.add_trace(go.Scatter(
                    x=plot_df['ds'],
                    y=plot_df['yhat'],
                    mode='lines',
                    name='Model Fit (Historical)',
                    line=dict(color='red', dash='dash')
                ))

                # Find the date where forecast starts
                last_hist_date = history_df['ds'].max()
                forecast_future = forecast[forecast['ds'] > last_hist_date]

                # Plot forecast (out-of-sample yhat)
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat'],
                    mode='lines+markers',
                    name='Forecast Median Stage (2 Qtrs)', # Updated label
                    line=dict(color='blue'),
                    marker=dict(size=4)
                ))

                # Plot confidence interval (only for forecast period) - Upper bound
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat_upper'],
                    mode='lines',
                    line=dict(width=0), # Don't draw the line itself
                    showlegend=False # Hide legend entry for this trace
                ))
                # Plot confidence interval - Lower bound, filling to the upper bound
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat_lower'],
                    mode='lines',
                    line=dict(width=0), # Don't draw the line itself
                    fillcolor='rgba(0, 0, 255, 0.2)', # Blue fill with transparency
                    fill='tonexty', # Fill area between this trace and the previous one (yhat_upper)
                    name='95% Confidence Interval'
                ))

                plot_title = 'Funding Stage Trend & Forecast' # Further shortened title
            # --- End Forecast Plotting --- #

            # Formatting
            fig.update_layout(
                title=plot_title,
                xaxis_title='Date (Quarter Start)',
                yaxis_title='Median Funding Stage (Numeric)',
                hovermode="x unified", # Show hover info for all traces at a given x
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(0,0,0,0)'), # Position legend & make background transparent
                margin=dict(l=40, r=40, t=80, b=40) # Increase top margin further to 80
            )
            # Optional: Add grid lines if desired
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')


            # Save the plot as HTML
            plot_path_html = os.path.join(prototype_output_dir, f"bay_area_funding_trend_interactive_{self.timestamp}.html") # New filename
            fig.write_html(plot_path_html)

            logger.info(f"Interactive dashboard prototype plot saved to {plot_path_html}")
            # No plt.close() needed for Plotly figures

        except Exception as e:
            logger.error(f"Error generating interactive dashboard prototype plot: {e}")
            logger.error(traceback.format_exc())
            # No plt.close() needed here either


def main():
    """Main entry point for the funding stage prediction pipeline"""
    import argparse
    import schedule
    import time
    from datetime import datetime, timedelta
    
    # Suppress all warnings
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, 
                          message='Mean of empty slice')
    warnings.filterwarnings('ignore', category=UserWarning, 
                          module='xgboost')
    warnings.filterwarnings('ignore', category=UserWarning, 
                          message='Tight layout not applied')
    warnings.filterwarnings('ignore', category=UserWarning, 
                          message='No positive samples')
    
    parser = argparse.ArgumentParser(
        description='Funding Stage Prediction Pipeline')
    parser.add_argument(
        '--base_dir',
        type=str,
        default='./JSONFolder',
        help='Base directory for data files')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./MainOutput',
        help='Output directory for results')
    parser.add_argument(
        '--archive',
        action='store_true',
        help='Archive existing data before processing')
    parser.add_argument(
        '--schedule',
        action='store_true',
        help='Run the pipeline on a schedule')
    parser.add_argument(
        '--interval',
        type=int,
        default=24,
        help='Interval in hours for scheduled runs')
    parser.add_argument(
        '--reset_db',
        action='store_true',
        help='Reset the database before running')
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Run continuously with scheduling')
    parser.add_argument(
        '--start_time',
        type=str,
        help='Start time in HH:MM format (24h)',
        default='00:00')
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once without scheduling')
    args = parser.parse_args()

    # Make sure we're using the JSONFolder explicitly
    if args.base_dir == './':
        args.base_dir = './JSONFolder'

    logger.info(f"Starting with data directory: {args.base_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Initialize pipeline with JSONFolder explicitly
    pipeline = EnhancedPipeline(
        args.base_dir,
        args.output_dir,
        archive=args.archive)

    if args.reset_db:
        pipeline.data_loader.reset_database()

    # Define the job function
    def scheduled_job():
        logger.info(f"Running scheduled job at {datetime.now()}")
        success = pipeline.run()
        if not success:
            logger.error("Scheduled job failed")

    # Run the job immediately regardless of scheduling options
    logger.info(f"Running funding prediction job at {datetime.now()}")
    success = pipeline.run() 

    if not success:
        logger.error("Initial run failed - check logs for details")
        return

    # Only run once and exit if specifically requested with --once flag
    # Otherwise, default behavior is to schedule future runs
    if args.once:
        logger.info("Job completed - exiting")
    else:
        # Configure scheduling parameters
        interval_hours = args.interval

        # Schedule the job - either at a specific time each day or on an
        # interval
        if args.start_time:
            # Schedule for specific time each day
            schedule.every().day.at(args.start_time).do(scheduled_job)
            logger.info(f"Scheduled to run daily at {args.start_time}")

            # Calculate next run time
            hour, minute = map(int, args.start_time.split(':'))
            next_run = datetime.now().replace(hour=hour, minute=minute)
            if next_run < datetime.now():
                next_run += timedelta(days=1)
        else:
            # Schedule to run on an interval
            schedule.every(interval_hours).hours.do(scheduled_job)
            logger.info(f"Scheduled to run every {interval_hours} hours")
            next_run = datetime.now() + timedelta(hours=interval_hours)

        logger.info(f"Next scheduled run: {next_run}")

        # Keep the script running for scheduled jobs
        try:
            logger.info("Scheduler active - process will remain running")
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")


# Define NumpyEncoder here for use in _save_summary
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None # Handle pandas Nat/NA specifically
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    main()

