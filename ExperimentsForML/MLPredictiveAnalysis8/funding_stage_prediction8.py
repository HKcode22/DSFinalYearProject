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

            # --- Calculate funding vs industry median (NOW uses industry_category) --- #
            # <<< ENSURE industry_category exists before this line >>>
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
        data['industry_category'] = data['industry'].fillna('Unknown') # <<< CREATE industry_category FIRST

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

        # Consolidate Rare Industries
        min_frequency = 10
        industry_counts = data['industry_category'].value_counts()
        rare_industries = industry_counts[industry_counts < min_frequency].index.tolist()
        if 'Other' in data['industry_category'].unique() and 'Other' in rare_industries:
            if industry_counts.get('Other', 0) >= min_frequency:
                rare_industries.remove('Other')
        if rare_industries:
            logger.info(f"Mapping {len(rare_industries)} rare industries (count < {min_frequency}) to 'Other': {rare_industries[:10]}...")
            data['industry_category'] = data['industry_category'].replace(rare_industries, 'Other')

        # Location features (if available)
        if 'location' in data.columns or 'headquarters' in data.columns:
            # Use either location or headquarters column
            location_col = 'location' if 'location' in data.columns else 'headquarters'

            data['location_category'] = data[location_col].fillna('Unknown')

            # Extract country or state
            def extract_location(loc_str):
                if not isinstance(loc_str, str) or pd.isna(loc_str):
                    return 'Unknown'

                # Split by comma and get the last part (usually country/state)
                parts = [p.strip() for p in loc_str.split(',')]

                if len(parts) > 1:
                    return parts[-1]  # Return the last part
                return loc_str

            data['location_category'] = data['location_category'].apply(
                extract_location)

            # Consolidate common locations
            location_mapping = {
                'United States': 'USA',
                'US': 'USA',
                'U.S.': 'USA',
                'U.S.A.': 'USA',
                'UK': 'United Kingdom',
                'U.K.': 'United Kingdom'
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

        # +++ Add Interaction Features +++
        # Ensure required columns exist and are numeric before creating interactions
        if 'funding_amount_log' in data.columns and 'months_since_first_funding' in data.columns:
             data['funding_amount_x_age'] = data['funding_amount_log'] * data['months_since_first_funding']
        else:
             data['funding_amount_x_age'] = 0 # Default if components are missing

        if 'employees' in data.columns and 'previous_rounds' in data.columns:
             data['employees_x_rounds'] = data['employees'] * data['previous_rounds']
        else:
              data['employees_x_rounds'] = 0 # Default if components are missing

        # --- Add New Interaction Features ---
        if 'funding_velocity' in data.columns and 'previous_rounds' in data.columns:
             data['velocity_x_rounds'] = data['funding_velocity'] * data['previous_rounds']
        else:
             data['velocity_x_rounds'] = 0 # Default if components are missing

        if 'months_since_first_funding' in data.columns and 'employees' in data.columns:
             data['age_x_employees'] = data['months_since_first_funding'] * data['employees']
        else:
             data['age_x_employees'] = 0 # Default if components are missing
        # --- End New Interaction Features ---
        # +++ End Interaction Features +++

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

        # Fill missing values for all numeric columns
        numeric_cols = [
            'funding_amount',
            'funding_amount_log',
            'employees',
            'employee_efficiency',
            'previous_rounds',
            'funding_velocity',
            'month_sin', # Include new features
            'month_cos', # Include new features
            'funding_amount_x_age', # Include new features
            'employees_x_rounds', # Include new features
            'velocity_x_rounds', # <<< ADD NEW FEATURE TO LIST
            'age_x_employees'    # <<< ADD NEW FEATURE TO LIST
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
            'funding_amount_log', # Keep original log funding
            'employee_efficiency', # Keep efficiency 
            'funding_year', 
            'funding_month', 
            'previous_rounds', 
            # 'months_since_first_funding', # REMOVED - Use binned version
            'funding_velocity',
            'month_sin', 
            'month_cos', 
            'funding_amount_x_age', 
            'employees_x_rounds', 
            'velocity_x_rounds', 
            'age_x_employees',
            'time_since_last_funding',       # <<< ADDED
            'funding_amount_ratio_vs_prev',  # <<< ADDED
            'funding_vs_industry_median'   # <<< ADDED
            # 'employees' # REMOVED - Use binned version
        ]
        categorical_feature_cols = [
            'industry_category', 
            'company_age_bin', # Binned age
            'employees_bin'      # Binned employees
        ]

        # Only use features that actually exist in the data
        features_to_use_num = [col for col in numeric_feature_cols if col in data.columns]
        features_to_use_cat = [col for col in categorical_feature_cols if col in data.columns] # <<< GET CATEGORICAL TO USE

        # Log features used
        # logger.info(f"Preparing model data with features: {features_to_use_num}")
        logger.info(f"Preparing model data with numeric features: {features_to_use_num}")
        logger.info(f"Preparing model data with categorical features: {features_to_use_cat}")

        # Clean feature data - ensure numeric types
        # X = data[features_to_use_num].copy()
        X_num = data[features_to_use_num].copy()

        # Convert all features to numeric
        # for col in X.columns:
        #     X[col] = pd.to_numeric(X[col], errors='coerce')
        for col in X_num.columns:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce')

        # Fill missing values for numeric columns - using median from actual
        # data
        # numeric_cols = X.select_dtypes(include=np.number).columns
        numeric_cols = X_num.select_dtypes(include=np.number).columns # Use X_num

        for col in numeric_cols:
            # if X[col].isna().any():
            #     median_value = X[col].median()
            #     X[col] = X[col].fillna(median_value)
            if X_num[col].isna().any():
                median_value = X_num[col].median()
                X_num[col] = X_num[col].fillna(median_value)
                logger.info(
                    f"Filled NaN values in {col} with median: {median_value}")

        # --- One-Hot Encode Categorical Features --- #
        X_cat_encoded = pd.DataFrame() # Initialize empty DataFrame
        if features_to_use_cat:
            X_cat = data[features_to_use_cat].copy()
            # Ensure categorical columns are treated as strings/categories
            for col in X_cat.columns:
                 X_cat[col] = X_cat[col].astype('category')
            X_cat_encoded = pd.get_dummies(X_cat, prefix=features_to_use_cat, dummy_na=False) # Avoid NaN columns
            logger.info(f"Created {X_cat_encoded.shape[1]} features from one-hot encoding: {X_cat_encoded.columns.tolist()}")

        # --- Combine Numeric and Encoded Categorical Features --- #
        X = pd.concat([X_num, X_cat_encoded], axis=1)

        # --- Sanitize feature names for compatibility (e.g., XGBoost) --- #
        try:
            original_columns = X.columns.tolist()
            sanitized_columns = []
            # Simple sanitization: replace problematic characters with underscores
            # Consider more robust regex if needed: re.sub(r'[^\w\s]+', '', col)
            for col in original_columns:
                sanitized_col = str(col).replace('[', '_').replace(']', '_').replace('<', '_') \
                                         .replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '')
                sanitized_columns.append(sanitized_col)

            X.columns = sanitized_columns
            logger.info(f"Sanitized {len(original_columns)} feature names. Example: '{original_columns[-1]}' -> '{X.columns[-1]}'")
        except Exception as sanitize_err:
            logger.error(f"Error sanitizing feature names: {sanitize_err}")
            # Decide how to handle error - maybe revert to original names or raise?
            # For now, log and continue with potentially problematic names
            pass
        # --- End Sanitization --- #

        # Target variable processing
        y = pd.to_numeric(data['funding_stage_numeric'], errors='coerce')

        # Check we have enough data
        if y.notna().sum() < 10:
            logger.warning(
                f"Very few valid target values: {
                    y.notna().sum()} out of {
                    len(y)}")

        X = X[y.notna()]
        y = y[y.notna()].astype(int)

        # Log data shapes and class distribution
        logger.info(
            f"Prepared model data: X shape={
                X.shape}, y shape={
                y.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y


class ModelTrainer:
    def __init__(self, output_dir="./models"):
        """Initialize with output directory for saving models"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def train_random_forest(self, X, y):
        """Train a Random Forest model for funding stage prediction with anomaly detection"""
        logger.info("Training Random Forest model...")
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Add anomaly detection using isolation forest to identify outliers
            # Use a randomized seed to prevent predictable detection patterns
            isolation_forest = IsolationForest(
                contamination=0.05,
                random_state=np.random.randint(
                    0,
                    10000))
            outlier_scores = isolation_forest.fit_predict(X_train)
            outliers_mask = outlier_scores == -1

            if outliers_mask.any():
                logger.info(
                    f"Identified {
                        outliers_mask.sum()} potential outliers in training data")

                # Log some example outliers
                outlier_examples = X_train.iloc[outliers_mask].head(
                    3) if isinstance(X_train, pd.DataFrame) else None
                if outlier_examples is not None:
                    logger.info(
                        f"Outlier examples:\n{
                            outlier_examples.to_dict(
                                orient='records')}")

            # Check class imbalance
            class_counts = np.bincount(y_train)
            logger.info(f"Class counts before training: {class_counts}")

            if np.min(class_counts) < 10:
                logger.warning(
                    f"Class imbalance detected, but continuing without resampling")

            # Define model with hyperparameters - use bootstrap aggregating for robustness
            # and randomized state for unpredictability
            rf = RandomForestClassifier(
                n_estimators=900,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                bootstrap=True,
                oob_score=True,  # Use out-of-bag to assess model quality
                random_state=np.random.randint(
                    0, 10000),  # Random seed for each run
                n_jobs=-1
            )

            # Train model
            rf.fit(X_train, y_train)

            # Check out-of-bag score as additional validation
            if hasattr(rf, 'oob_score_'):
                logger.info(f"Out-of-bag score: {rf.oob_score_:.4f}")

            # Evaluate
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            logger.info(f"Random Forest accuracy: {accuracy:.4f}")
            logger.info(f"Classification report:\n{report}")

            # Save model and metadata including anomaly detector
            model_path = os.path.join(
                self.output_dir, f"random_forest_{
                    self.timestamp}.joblib")
            model_metadata = {
                'model': rf,
                'isolation_forest': isolation_forest,
                'training_date': self.timestamp,
                'feature_names': X.columns.tolist() if hasattr(
                    X,
                    'columns') else None,
                'accuracy': accuracy,
                'class_mapping': {
                    i: label for i,
                    label in enumerate(
                        np.unique(y))}}
            joblib.dump(model_metadata, model_path)

            # Return model and evaluation data dictionary
            return {
                'status': 'success', # Add status
                'model': rf, # Return model object
                'accuracy': accuracy,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': rf.predict_proba(X_test) if hasattr(rf, 'predict_proba') else None,
                'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None,
                'model_path': model_path,
                'isolation_forest': isolation_forest
            }
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}

    def train_xgboost(self, X, y):
        """
        Train an XGBoost classifier with enhanced hyperparameters.
        
        Args:
            X: Features matrix
            y: Target vector
            
        Returns:
            Dictionary with model, predictions, probabilities and metrics, or failure dict
        """
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train the model with tuned hyperparameters
            model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=500,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0,
                reg_lambda=1,
                objective='multi:softproba',
                num_class=len(np.unique(y)),
                random_state=42,
                # Removed the use_label_encoder parameter as it's deprecated
                verbosity=0
            )

            # Train the model
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[
                    (X_train_scaled,
                     y_train),
                    (X_test_scaled,
                     y_test)],
                eval_metric=['mlogloss',
                             'merror'],
                early_stopping_rounds=10,
                verbose=False)

            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            confusion = confusion_matrix(y_test, y_pred)

            # Log the performance
            logger.info(
                f"XGBoost Performance: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            # Return the results
            return {
                'status': 'success', # Add status indicator
                'model': model,
                'scaler': scaler,
                'predictions': y_pred,
                'probabilities': y_proba,
                'confusion_matrix': confusion,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'feature_importance': model.feature_importances_
            }
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            logger.error(traceback.format_exc())
            # return None # Original line
            return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()} # Modified line

    def train_gradient_boosting(self, X, y):
        """Train a Gradient Boosting model with optimized parameters"""
        logger.info("Training Gradient Boosting model...") # Add logging
        try: # <<< Add try block
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            # Define model with carefully selected parameters
            gb = GradientBoostingClassifier(
                n_estimators=900,
                learning_rate=0.03,
                max_depth=10,
                min_samples_split=9,
                min_samples_leaf=4,
                subsample=0.8,
                max_features='sqrt',
                random_state=42,
                verbose=0
            )

            # Fit the model
            gb.fit(X_train, y_train)

            # Evaluate
            y_pred = gb.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Calculate additional metrics
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) # Added zero_division
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) # Added zero_division
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) # Added zero_division

            # Log the performance
            logger.info(f"Gradient Boosting Performance: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            # Return results dictionary
            return {
                'status': 'success', # Add status
                'model': gb, # Return model object
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'feature_importance': gb.feature_importances_
            }
        except Exception as e: # <<< Add except block
            logger.error(f"Error training Gradient Boosting model: {e}")
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}


class EnhancedModelTrainer(ModelTrainer):
    def tune_random_forest(self, X, y):
        """Use RandomizedSearchCV to find optimal Random Forest parameters"""
        try: # Add try block
            param_grid = {
                'n_estimators': randint(100, 2500), # Further expanded upper bound
                'max_depth': [None, 5, 10, 15, 20, 25, 30, 40, 50, 60], # More discrete values
                'min_samples_split': randint(2, 40), # Expanded upper bound
                'min_samples_leaf': randint(1, 30),  # Expanded upper bound
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_leaf_nodes': [None] + list(range(30, 401, 30)), # Expanded range and steps
                'min_impurity_decrease': [0.0, 0.0001, 0.001, 0.01, 0.02, 0.05, 0.1], # Added more fine-grained values
                'ccp_alpha': uniform(0.0, 0.1), # Expanded ccp_alpha for pruning
                'min_weight_fraction_leaf': uniform(0.0, 0.4) # Added
            }

            rf = RandomForestClassifier(random_state=42)
            grid_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_grid,
                # n_iter=8,  # <<< INCREASED FROM 4 to 8
                
                n_iter=2, # <<< Reduced n_iter back to 2
                cv=3,     # (Keep cv=3)
                verbose=1,
                random_state=42,
                n_jobs=-1,
                scoring='accuracy',
                refit=True
            )
            grid_search.fit(X, y)

            logger.info(f"Best Random Forest params: {grid_search.best_params_}")
            logger.info(f"Best Random Forest score: {grid_search.best_score_:.4f}")

            return grid_search.best_estimator_
        except Exception as e: # Add except block
            logger.error(f"Error tuning Random Forest: {str(e)}")
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}
    
    def tune_xgboost(self, X, y):
        """Use RandomizedSearchCV to find optimal XGBoost parameters"""
        # Make sure we're using the right import
        from xgboost import XGBClassifier
        
        # Define parameter grid (Enhanced Version)
        param_grid = {
            'n_estimators': randint(300, 4000), # Expanded upper bound
            'max_depth': randint(3, 30), # More focused but still broad range for depth
            'learning_rate': [0.0005, 0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3], # Discrete values, added more
            'subsample': uniform(0.5, 0.5), # Samples from 0.5 to 1.0 (corrected range)
            'colsample_bytree': uniform(0.5, 0.5), # Samples from 0.5 to 1.0 (corrected range)
            'colsample_bylevel': uniform(0.5, 0.5), # Samples from 0.5 to 1.0 (corrected range)
            'colsample_bynode': uniform(0.5, 0.5),  # Samples from 0.5 to 1.0 (corrected range)
            'min_child_weight': randint(1, 20), # Expanded range
            'gamma': uniform(0, 3.0), # Expanded range
            'reg_alpha': [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2], # More values
            'reg_lambda': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50],    # More values
            'max_delta_step': [0, 1, 3, 5, 7, 10], # Added 0, common default, more options
            'objective': ['multi:softprob'], # Focused on softprob for probabilities
            'booster': ['gbtree', 'dart'], # Focused on tree-based boosters
            'tree_method': ['auto', 'hist', 'approx'], # Added tree_method
            'scale_pos_weight': [1] + list(uniform(1, 10).rvs(3)) # For imbalanced data, 1 is default
        }
        
        # Create a base model to tune
        n_classes = len(np.unique(y))
        # Ensure y is integer type for XGBoost
        y = y.astype(int)
        model = XGBClassifier(
            num_class=n_classes,
            eval_metric='mlogloss', # Add eval_metric
            random_state=42,
            verbosity=0
        ) 
        
        # Set up the search (REMOVED n_iter argument)
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=2, # <<< Reduced n_iter back to 2
            cv=3,    # (Keep cv=3)
            verbose=2,
            random_state=42,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        try:
            # Fit the model
            search.fit(X, y)
            
            # Get the best parameters and score
            best_params = search.best_params_
            best_score = search.best_score_
            
            logger.info(f"Best XGBoost params: {best_params}")
            logger.info(f"Best XGBoost score: {best_score:.4f}")
            
            # Create a new model with the best parameters
            best_model = XGBClassifier(
                **best_params,
                num_class=n_classes,
                eval_metric='mlogloss', # Add eval_metric
                random_state=42,
                verbosity=0
            )
            
            # Fit the model on the full dataset (optional, often best estimator is returned)
            # best_model.fit(X, y)
            
            # Return the best estimator found by the search
            return search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error tuning XGBoost: {str(e)}")
            logger.error(traceback.format_exc())
            # return None # Original line
            return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()} # Modified line

    def tune_gradient_boosting(self, X, y, n_iter): # <<< Changed default n_iter back to 4
        """Tune Gradient Boosting Classifier using RandomizedSearchCV"""
        logger.info(f"Starting Gradient Boosting tuning with n_iter={n_iter}...")
        gb = GradientBoostingClassifier(random_state=42)

        # Define parameter grid (reduced)
        param_grid = {
            'n_estimators': randint(100, 2500),       # Expanded range
            'learning_rate': [0.0005, 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15, 0.2], # Discrete values, more options
            'max_depth': randint(3, 20),             # Expanded range
            'min_samples_split': randint(2, 40),    # Expanded range
            'min_samples_leaf': randint(1, 40),     # Expanded range
            'subsample': uniform(0.5, 0.5), # Range from 0.5 to 1.0 (corrected range)
            'max_features': ['sqrt', 'log2', None], # Streamlined options
            'loss': ['log_loss'], # For classification
            'criterion': ['friedman_mse', 'squared_error'], # Added criterion
            'min_impurity_decrease': [0.0, 0.0001, 0.001, 0.01, 0.02, 0.05], # Added more values
            'max_leaf_nodes': [None] + list(range(20, 301, 30)), # Added, expanded range
            'ccp_alpha': uniform(0.0, 0.1), # Expanded ccp_alpha
            'init': [None] # Added 'init'
        }

        # Perform randomized search
        search = RandomizedSearchCV(
            estimator=gb,
            param_distributions=param_grid,
            n_iter=n_iter,  # <<< Use passed n_iter value (now defaults to 4, will be overridden in run)
            cv=3,     # Keep cv=3 for speed
            verbose=1,
            random_state=42,
            scoring='accuracy', # Optimize for accuracy
            n_jobs=-1
        )
        
        try:
            search.fit(X, y)
            logger.info(f"Best Gradient Boosting params: {search.best_params_}")
            logger.info(f"Best Gradient Boosting score: {search.best_score_:.4f}")
            return search.best_estimator_
        except Exception as e:
            logger.error(f"Error during Gradient Boosting tuning: {e}")
            # Fallback to default if tuning fails
            logger.warning("Falling back to default Gradient Boosting parameters.")
            return GradientBoostingClassifier(random_state=42)
    
    def train_catboost(self, X, y):
        """Train a CatBoost model with hyperparameter tuning"""
        try:
            from catboost import CatBoostClassifier
            
            # Define parameter grid
            param_grid = {
                'iterations': randint(100, 500),
                'learning_rate': uniform(0.01, 0.3),
                'depth': randint(4, 10),
                'l2_leaf_reg': uniform(1, 9),
                'bagging_temperature': uniform(0, 1),
                'random_strength': uniform(0, 1),
                'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
            }
            
            # Initialize the classifier
            cat_clf = CatBoostClassifier(
                random_seed=42,
                thread_count=-1,
                verbosity=0
            )
            
            # Perform random search
            search = RandomizedSearchCV(
                cat_clf, param_grid, n_iter=2, cv=3, # <<< Reduced n_iter back to 2
                scoring='accuracy', random_state=42, n_jobs=-1, verbose=1
            )
            
            search.fit(X, y)
            
            logger.info(f"Best CatBoost params: {search.best_params_}")
            logger.info(f"Best CatBoost score: {search.best_score_:.4f}")
            
            # Evaluate on train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            best_model = search.best_estimator_
            best_model.fit(X_train, y_train)
            
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"CatBoost test accuracy: {accuracy:.4f}")
            
            # Return success dictionary format (modify existing return)
            return {
                'status': 'success',
                'model': best_model,
                'accuracy': accuracy,
                # Add other relevant metrics if needed, e.g., X_test, y_test, y_pred, scaler
                # Note: This basic method doesn't use/return a scaler unlike _train_final_model
            }
        except ImportError:
            logger.warning("CatBoost not installed. Skipping CatBoost model.")
            # Return failure dictionary for consistency
            return {'status': 'skipped', 'error': 'CatBoost not installed', 'accuracy': 0.0}
        except Exception as e: # <<< Add general exception handler
            logger.error(f"Error training CatBoost model: {e}")
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}
            
    def train_stacked_ensemble(self, X, y, base_models_dict): # Updated to accept a dict of named models
        """Train a stacked ensemble model with cross-validation using RF as meta-learner."""
        from sklearn.ensemble import StackingClassifier, RandomForestClassifier
        from sklearn.base import clone
        
        if not base_models_dict:
            logger.error("No base models provided for stacking ensemble.")
            return None, 0.0, None, None # <<< Return scaler as None

        base_estimators = []
        for name, model_instance in base_models_dict.items():
            # --- Directly try cloning, skip the 'if model_instance' check --- #
            try:
                unfitted_clone = clone(model_instance)
                base_estimators.append((name, unfitted_clone))
                logger.info(f"Added unfitted clone of {name} to stacking base estimators.")
            except Exception as clone_err:
                 logger.error(f"Could not clone base model {name}: {clone_err}. Skipping for stacking.")

        if len(base_estimators) < 2:
            logger.error(f"Not enough valid base model instances ({len(base_estimators)}) available for stacking.")
            return None, 0.0, None, None # <<< Return scaler as None

        # Define meta-learner: RandomForestClassifier with limited complexity
        meta_learner = RandomForestClassifier(
            n_estimators=50, 
            max_depth=5, 
            min_samples_leaf=5,
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Create the stacking ensemble
        # --- Use default CV (e.g., 5-fold) and let StackingClassifier fit the clones ---
        stacked_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            stack_method='predict_proba', # Use probabilities for meta-learner
            passthrough=False, # Do not pass original features to meta-learner
            cv=3, # Use 3-fold CV for speed
            n_jobs=-1,
            verbose=1
        )
        
        # Evaluate on train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # --- Scale data BEFORE fitting the StackingClassifier ---
        # StackingClassifier does not inherently handle scaling for its base models during CV.
        # A common pattern is to use Pipelines within the stacking estimators,
        # OR scale the entire dataset first (but apply scaling fitted *only* on train).
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # The StackingClassifier will now operate on the scaled data.

        logger.info(f"Fitting Stacking Ensemble with base models: {[name for name, _ in base_estimators]}")
        # Fit on the SCALED training data
        stacked_model.fit(X_train_scaled, y_train)
        
        # Predict on the SCALED test data
        y_pred = stacked_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Stacked Ensemble (RF Meta) test accuracy: {accuracy:.4f}")
        
        y_proba = None
        if hasattr(stacked_model, 'predict_proba'):
            try:
                y_proba = stacked_model.predict_proba(X_test_scaled)
            except Exception as e:
                logger.warning(f"Couldn't get probabilities from stacked model: {str(e)}")
        
        # Return the fitted model, accuracy, probabilities, and the scaler used
        return stacked_model, accuracy, y_proba, scaler # <<< Return scaler
        
    def train_voting_ensemble(self, X, y, estimators, voting='soft'):
        """Train a voting ensemble model with multiple base estimators"""
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        
        # Evaluate on train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        voting_clf.fit(X_train, y_train)
        
        y_pred = voting_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Voting Ensemble ({voting}) test accuracy: {accuracy:.4f}")
        
        return voting_clf, accuracy

    # +++ Add tune_lightgbm method +++
    def tune_lightgbm(self, X, y, n_iter=2): # <<< Changed default n_iter back to 2
        """Tune LightGBM Classifier using RandomizedSearchCV"""
        if not LGBMClassifier:
             logger.warning("LightGBM not installed, cannot tune.")
             return None

        logger.info(f"Starting LightGBM tuning with n_iter={n_iter}...")
        lgbm = LGBMClassifier(random_state=42, n_jobs=-1)

        # Define parameter grid
        param_grid = {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.2),
            'num_leaves': randint(20, 60),
            'max_depth': [-1] + list(range(5, 16)), # -1 means no limit
            'min_child_samples': randint(5, 30),
            'subsample': uniform(0.6, 0.4), # range 0.6 to 1.0
            'colsample_bytree': uniform(0.5, 0.5), # range 0.5 to 1.0
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.01, 0.1, 0.5, 1],
        }

        # Perform randomized search
        search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_grid,
            n_iter=n_iter,  # <<< Use passed n_iter value (now defaults to 2)
            cv=3,     # Keep cv=3 for speed
            verbose=1,
            random_state=42,
            scoring='accuracy',
            n_jobs=-1
        )

        try:
            search.fit(X, y)
            logger.info(f"Best LightGBM params: {search.best_params_}")
            logger.info(f"Best LightGBM score: {search.best_score_:.4f}")
            return search.best_estimator_
        except Exception as e:
            logger.error(f"Error during LightGBM tuning: {e}")
            logger.error(traceback.format_exc())
            return None # Return None on failure
    # +++ End tune_lightgbm method +++

    # +++ Add tune_decision_tree method +++
    def tune_decision_tree(self, X, y, n_iter=2): # <<< Changed default n_iter back to 2
        """Tune Decision Tree Classifier using RandomizedSearchCV"""
        from sklearn.tree import DecisionTreeClassifier
        logger.info(f"Starting Decision Tree tuning with n_iter={n_iter}...")
        
        dt = DecisionTreeClassifier(random_state=42)

        param_grid = {
            'criterion': ['gini', 'entropy', 'log_loss', 'gini_impurity', 'entropy_impurity', 'gini_entropy', 'entropy_gini'],
            'splitter': ['best', 'random', 'gini', 'entropy', 'gini_entropy', 'entropy_gini', 'gini_entropy_gini', 'entropy_gini_gini', 'gini_entropy_entropy', 'entropy_gini_entropy'],
            'max_depth': [None] + list(range(5, 51, 5)),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 50),
            'max_features': ['sqrt', 'log2', None],
            'max_leaf_nodes': [None] + list(range(20, 201, 20)),
            'min_impurity_decrease': uniform(0.0, 0.1),
            'ccp_alpha': uniform(0.0, 0.1), # Cost-complexity pruning
            'class_weight': ['balanced', None]
        }

        search = RandomizedSearchCV(
            estimator=dt,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,
            verbose=1,
            random_state=42,
            scoring='accuracy',
            n_jobs=-1
        )
        
        try:
            search.fit(X, y)
            logger.info(f"Best Decision Tree params: {search.best_params_}")
            logger.info(f"Best Decision Tree score: {search.best_score_:.4f}")
            return search.best_estimator_
        except Exception as e:
            logger.error(f"Error during Decision Tree tuning: {e}")
            logger.error(traceback.format_exc())
            return None
    # +++ End tune_decision_tree method +++


class ModelManager:
    """Manages machine learning models for funding stage prediction with validation and audit"""

    def __init__(self, model_dir='models/'):
        """Initialize with model directory and setup audit logging"""
        self.model_dir = model_dir
        self.model = None
        self.metadata = {}
        self.scaler = None
        self.feature_names = []
        # Initialize AnomalyDetector here if needed, or ensure it's passed/set
        self.anomaly_detector = AnomalyDetector(contamination=0.05) # Keep initialization
        self.audit_log_file = os.path.join(model_dir, 'prediction_audit.csv')
        self.init_audit_log()

    def init_audit_log(self):
        """Initialize audit log file if it doesn't exist"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.audit_log_file), exist_ok=True)

            # Create audit log with headers if it doesn't exist
            if not os.path.exists(self.audit_log_file):
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
                with open(self.audit_log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)

            logger.info(f"Audit log initialized at {self.audit_log_file}")
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
                'feature_names'] # REMOVED 'anomaly_detector'
            if not all(key in model_data for key in required_keys):
                # Be more specific about missing keys
                missing = [key for key in required_keys if key not in model_data]
                logger.error(
                    f"Invalid model file format from {model_path}, missing required components: {missing}")
                return False

            # Check model integrity and assign to class properties
            self.model = model_data['model']
            self.metadata = model_data['metadata']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']

            # Log successful load
            version_info = self.metadata.get('version', 'unknown')
            created_at = self.metadata.get('created_at', 'unknown')
            logger.info(
                f"Loaded {model_name} v{version_info} (created {created_at})")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def fit_anomaly_detector(self, training_data, feature_names):
        """Train the anomaly detector"""
        if training_data is not None and self.anomaly_detector is not None:
            logger.info("Training anomaly detector...")
            # Ensure training_data matches expected feature names/order if applicable
            # Assuming training_data is ready (e.g., scaled features)
            self.anomaly_detector.fit(training_data, feature_names) # Pass feature names if AnomalyDetector uses them
            logger.info("Anomaly detector trained.")
        else:
            logger.warning("Could not train anomaly detector: No data or detector instance.")

    def save_model(
            self,
            model_name,
            model,
            scaler,
            feature_names,
            # training_data=None, # Removed: anomaly detector training is separate
            metadata=None,
            anomaly_detector=None): # <<< Add anomaly_detector parameter
        """Save a trained model with important metadata"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)

            # Generate version
            version = datetime.now().strftime("%Y%m%d%H%M")
            # Ensure model_name doesn't already contain versioning info if passed
            base_model_name = model_name.split('_v')[0]
            # --- Save with .joblib extension --- #
            model_path = os.path.join(
                self.model_dir, f"{base_model_name}_v{version}.joblib")

            # Set up metadata
            if metadata is None:
                metadata = {}

            metadata.update({
                'version': version,
                'created_at': datetime.now().isoformat(),
                'feature_names': feature_names,
                'model_type': type(model).__name__
            })

            # Train anomaly detector if training data provided # REMOVED this logic
            # if training_data is not None:
            #     self.anomaly_detector.fit(training_data, feature_names)

            # Package everything together
            model_data = {
                'model': model,
                'metadata': metadata,
                'scaler': scaler,
                'feature_names': feature_names,
                'anomaly_detector': anomaly_detector, # <<< Save the passed detector
                # --- Extract and save specific keys --- #
                'training_metadata': metadata.get('training_metadata', {}), # Extract inner dict
                'class_mapping': metadata.get('class_mapping', {}) # Extract inner dict
            }

            # Save to disk using joblib for potentially better compatibility with sklearn models
            joblib.dump(model_data, model_path) # Changed from pickle to joblib

            logger.info(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            logger.error(traceback.format_exc()) # Add traceback
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
            with open(self.audit_log_file, 'a', newline='') as f:
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

    # Add a method to load specifically using joblib
    def load_model_joblib(self, model_name, version='latest'):
        """Load a trained model from disk (saved with joblib) with checks and validation"""
        try:
            # Determine file path
            if version == 'latest':
                # +++ Add Debug Logging +++
                logger.info(f"[load_model_joblib] Searching in directory: {self.model_dir}")
                pkl_pattern = os.path.join(self.model_dir, f"{model_name}*.pkl")
                logger.info(f"[load_model_joblib] Using PKL glob pattern: {pkl_pattern}")
                # +++ End Debug Logging +++
                model_files = glob.glob(pkl_pattern) # Use full path pattern
                logger.info(f"[load_model_joblib] Found PKL files: {model_files}") # Log found files

                if not model_files:
                    # Try finding .joblib files as well
                     # +++ Add Debug Logging +++
                     joblib_pattern = os.path.join(self.model_dir, f"{model_name}*.joblib")
                     logger.info(f"[load_model_joblib] Using JOBLIB glob pattern: {joblib_pattern}")
                     # +++ End Debug Logging +++
                     model_files = glob.glob(joblib_pattern) # Use full path pattern
                     logger.info(f"[load_model_joblib] Found JOBLIB files: {model_files}") # Log found files
                     if not model_files:
                        logger.error(f"No models found matching pattern {model_name}* in {self.model_dir}")
                        return False

                # Sort by name (which should include version/timestamp)
                model_files.sort(reverse=True)
                model_path = model_files[0]
            else:
                # Try finding .pkl first, then .joblib
                model_path_pkl = os.path.join(
                    self.model_dir, f"{model_name}_v{version}.pkl")
                model_path_joblib = os.path.join(
                    self.model_dir, f"{model_name}_v{version}.joblib")

                if os.path.exists(model_path_pkl):
                    model_path = model_path_pkl
                elif os.path.exists(model_path_joblib):
                     model_path = model_path_joblib
                else:
                    logger.error(f"Model file not found for version {version}: {model_path_pkl} or {model_path_joblib}")
                    return False

            # Load model data using joblib
            model_data = joblib.load(model_path)

            # Validate model data structure
            required_keys = [
                'model',
                'metadata',
                'scaler',
                'feature_names'] # REMOVED 'anomaly_detector'
            if not all(key in model_data for key in required_keys):
                # Be more specific about missing keys
                missing = [key for key in required_keys if key not in model_data]
                logger.error(
                    f"Invalid model file format from {model_path}, missing required components: {missing}")
                return False

            # Check model integrity and assign to class properties
            self.model = model_data['model']
            self.metadata = model_data['metadata']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']

            # Log successful load
            version_info = self.metadata.get('version', 'unknown')
            created_at = self.metadata.get('created_at', 'unknown')
            logger.info(
                f"Loaded {model_name} v{version_info} (created {created_at}) from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            logger.error(traceback.format_exc()) # Add traceback
            return False


class Visualizer:
    def __init__(self, output_dir="./visualizations", interactive=False):
        """Initialize visualizer with output directory"""
        self.output_dir = output_dir
        self.interactive = interactive  # Set to False to prevent blocking terminal

        # Ensure output directory exists - critical fix
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created visualization directory: {output_dir}")
        except Exception as e:
            logger.error(f"Error creating visualization directory: {e}")
            # Fallback to a directory we know exists
            self.output_dir = "./MainOutput/visualizations"
            os.makedirs(self.output_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.palette = sns.color_palette("husl", 8)
        self.feature_palettes = {
            'categorical': sns.color_palette("Set3", 12),
            'sequential': sns.color_palette("viridis", 8),
            'diverging': sns.color_palette("RdYlBu", 11)
        }

    def plot_funding_stage_distribution(self, data, stage_mapping_rev=None): # Add stage_mapping_rev parameter
        """Visualize the distribution of funding stages"""
        plt.figure(figsize=(12, 6))

        # Use provided reverse mapping if available, otherwise use default
        if stage_mapping_rev is None:
            # Fallback to a default/generic mapping if none provided
            logger.warning("No stage mapping provided to plot_funding_stage_distribution. Using generic labels.")
            # Example fallback - adjust as needed or remove if mapping is always expected
            stage_map_rev_local = {i: f'Stage {i}' for i in sorted(data['funding_stage_numeric'].unique())}
        else:
            # Ensure keys in provided mapping are integers
            stage_map_rev_local = {int(k): str(v) for k, v in stage_mapping_rev.items()}


        # Count by stage using the numeric column
        # Map numeric stages back to names using the determined mapping
        stage_counts = data['funding_stage_numeric'].map(
            lambda x: stage_map_rev_local.get(int(x), f'Unknown ({x})') # Handle missing keys
        ).value_counts()

        # Sort counts by the original numeric order if possible, else alphabetically
        # Create a temporary series with numeric index for sorting
        try:
            num_order = {v: k for k, v in stage_map_rev_local.items()}
            sort_order = stage_counts.index.map(lambda x: num_order.get(x, 999)) # Assign large number for unknowns
            stage_counts = stage_counts.loc[sort_order.sort_values().index]
        except Exception:
            logger.warning("Could not sort stage distribution numerically, sorting alphabetically.")
            stage_counts = stage_counts.sort_index()


        # Plot
        ax = stage_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribution of Funding Stages (After Remapping & Merging)', fontsize=14) # Update title
        plt.xlabel('Funding Stage')
        plt.ylabel('Number of Companies')
        plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add count labels on bars
        for i, v in enumerate(stage_counts):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom') # Adjust label position slightly

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"funding_stage_dist_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_feature_importance(self, model, feature_names):
        """Visualize feature importance from model"""
        plt.figure(figsize=(12, 8))

        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-15:]  # Top 15 features

            # Plot horizontal bar chart
            plt.barh(range(len(indices)), importances[indices], color='coral')
            plt.yticks(range(len(indices)),
                       [feature_names[i] for i in indices])
            plt.title('Top Feature Importances', fontsize=14)
            plt.xlabel('Relative Importance')

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    self.output_dir,
                    f"feature_importance_{
                        type(model).__name__}_{
                        self.timestamp}.png"))
            if self.interactive:
                plt.show()
            plt.close()

    def plot_model_comparison(self, model_results):
        """Compare performance of different models"""
        plt.figure(figsize=(10, 6))

        # Extract accuracies and find the best model
        models = list(model_results.keys())
        accuracies = [model_results[m].get('accuracy', 0.0) for m in models] # Use .get() for safety
        best_model_index = np.argmax(accuracies)

        # Use a consistent color, highlight the best
        colors = ['skyblue'] * len(models)
        if accuracies: # Check if accuracies list is not empty
             colors[best_model_index] = 'dodgerblue' # Highlight best model

        # Plot bar chart
        bars = plt.bar(models, accuracies, color=colors) # Use the colors list
        plt.title('Model Accuracy Comparison', fontsize=14)
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.xticks(rotation=15, ha='right') # Slight rotation for model names
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"model_comparison_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_confusion_matrices(self, model_results):
        """Plot confusion matrices for all models"""
        plt.figure(figsize=(15, 10))

        for i, (model_name, results) in enumerate(model_results.items(), 1):
            plt.subplot(1, len(model_results), i)
            cm = confusion_matrix(results['y_test'], results['y_pred'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"confusion_matrices_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_funding_vs_employees(self, data):
        """Visualize relationship between funding amount and employee count"""
        plt.figure(figsize=(12, 8))

        # Prepare data
        plot_data = data[['funding_amount', 'employees',
                          'funding_stage_numeric']].dropna()

        # Map numeric stages to colors
        colors = plt.cm.viridis(np.linspace(0, 1, 12))
        plot_data['color'] = plot_data['funding_stage_numeric'].apply(
            lambda x: colors[int(x)] if 0 <= x < 12 else colors[0]
        )

        # Create scatter plot
        plt.scatter(
            plot_data['employees'],
            plot_data['funding_amount'],
            c=plot_data['color'],
            alpha=0.6,
            s=50
        )

        plt.title('Funding Amount vs. Employee Count by Stage', fontsize=14)
        plt.xlabel('Number of Employees')
        plt.ylabel('Funding Amount (USD)')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(linestyle='--', alpha=0.7)

        # Add legend
        stage_map_rev = {
            0: 'Pre-Seed', 1: 'Seed', 2: 'Series A',
            3: 'Series B', 4: 'Series C', 5: 'Series D+'
        }
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=colors[i], markersize=10,
                                      label=stage_map_rev.get(i, f'Stage {i}'))
                           for i in range(min(6, len(colors)))]
        plt.legend(handles=legend_elements, loc='upper left')

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"funding_vs_employees_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_feature_comparison_matrix(self, data, features):
        """Create a grid of scatterplots for feature comparisons"""
        plt.figure(figsize=(20, 15))
        g = sns.PairGrid(data[features], palette=self.palette)
        g.map_upper(sns.scatterplot, alpha=0.6)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)
        plt.suptitle('Feature Comparison Matrix', y=1.02)
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"feature_matrix_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_correlation_heatmap(self, data):
        """Visualize feature correlations with funding stage"""
        plt.figure(figsize=(15, 12))
        corr = data.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0)
        plt.title("Feature Correlation Heatmap")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"correlation_heatmap_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_temporal_trends(self, data):
        """Analyze funding trends over time with industry breakdown"""
        plt.figure(figsize=(18, 8))

        plt.subplot(1, 2, 1)
        sns.lineplot(
            data=data,
            x='funding_year',
            y='funding_amount',
            hue='industry_category',
            estimator='median',
            errorbar=None)
        plt.title('Median Funding Amount by Year')
        plt.ylabel('USD (log scale)')
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        funding_counts = data.groupby(
            ['funding_year', 'industry_category']).size().reset_index()
        sns.lineplot(
            data=funding_counts,
            x='funding_year',
            y=0,
            hue='industry_category')
        plt.title('Funding Round Frequency by Year')
        plt.ylabel('Number of Rounds')

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"temporal_trends_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_industry_distributions(self, data):
        """Compare funding patterns across industries"""
        plt.figure(figsize=(15, 8))

        plt.subplot(1, 2, 1)
        sns.boxplot(
            data=data,
            x='industry_category',
            y='funding_amount',
            showfliers=False)
        plt.yscale('log')
        plt.title('Funding Amount Distribution by Industry')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        stage_dist = data.groupby(
            ['industry_category', 'funding_stage']).size().unstack()
        stage_dist.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Funding Stage Distribution by Industry')
        plt.ylabel('Number of Companies')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"industry_analysis_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_advanced_feature_correlations(self, data, features):
        """Create detailed feature correlation visualizations"""
        plt.figure(figsize=(20, 16))

        # Advanced correlation heatmap
        plt.subplot(2, 2, 1)
        corr = data[features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True)
        plt.title('Feature Correlation Matrix')

        # Feature clustering
        plt.subplot(2, 2, 2)
        from scipy.cluster import hierarchy
        corr_linkage = hierarchy.ward(corr)
        sns.clustermap(corr, method='ward', cmap='coolwarm',
                       annot=True, fmt='.2f', figsize=(10, 10))

        # 3D scatter plot of top 3 features
        plt.subplot(2, 2, 3, projection='3d')
        top_features = features[:3]  # Use first 3 features
        ax = plt.gca()
        scatter = ax.scatter(data[top_features[0]],
                             data[top_features[1]],
                             data[top_features[2]],
                             c=data['funding_stage_numeric'],
                             cmap='viridis')
        ax.set_xlabel(top_features[0])
        ax.set_ylabel(top_features[1])
        ax.set_zlabel(top_features[2])
        plt.colorbar(scatter, label='Funding Stage')

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"advanced_correlations_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_feature_distributions(self, data, features):
        """Plot detailed feature distributions"""
        n_features = len(features)
        fig = plt.figure(figsize=(15, n_features * 3))

        for idx, feature in enumerate(features, 1):
            # Distribution plot
            plt.subplot(n_features, 2, 2 * idx - 1)
            sns.histplot(
                data=data,
                x=feature,
                hue='funding_stage',
                multiple="stack",
                palette=self.feature_palettes['categorical'])
            plt.title(f'{feature} Distribution by Funding Stage')
            plt.xticks(rotation=45)

            # Box plot
            plt.subplot(n_features, 2, 2 * idx)
            sns.boxplot(data=data, y=feature, x='funding_stage',
                        palette=self.feature_palettes['sequential'])
            plt.xticks(rotation=45)
            plt.title(f'{feature} Range by Funding Stage')

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"feature_distributions_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_funding_patterns(self, data):
        """Visualize complex funding patterns with interactive display"""
        plt.figure(figsize=(20, 12))  # Increased figure height from 10 to 12

        # Funding amount distribution over time
        plt.subplot(2, 2, 1)
        sns.boxenplot(data=data, x='funding_year', y='funding_amount_log',
                      palette=self.feature_palettes['sequential'])
        plt.title('Funding Amount Distribution Over Time')
        plt.xticks(rotation=45)

        # Employee count vs Funding amount
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=data, x='employees', y='funding_amount',
                        hue='funding_stage', size='employee_efficiency',
                        sizes=(20, 200), alpha=0.6,
                        palette=self.feature_palettes['categorical'])
        plt.yscale('log')
        plt.xscale('log')
        plt.title('Funding Amount vs Employee Count')

        # Industry funding distribution
        plt.subplot(2, 2, (3, 4))
        industry_funding = data.groupby('industry_category')[
            'funding_amount'].sum()
        industry_funding.sort_values(
            ascending=True).plot(
            kind='barh',
            color=self.feature_palettes['sequential'])
        plt.title('Total Funding by Industry')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)  # Explicitly set margins instead of tight_layout
        plt.savefig(os.path.join(self.output_dir,
                                 f"funding_patterns_{self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_pairwise_features(self, data, features):
        """Plot pairwise feature relationships (scatter matrix)"""
        sns.set(style="ticks")
        pairplot = sns.pairplot(data[features + ['funding_stage']],
                                hue='funding_stage', palette='tab10', diag_kind='kde')
        plt.suptitle('Pairwise Feature Relationships', y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"pairwise_features_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_full_correlation_heatmap(self, data):
        """Plot a full correlation heatmap for all numeric features"""
        plt.figure(figsize=(18, 14))
        corr = data.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0)
        plt.title("Full Feature Correlation Heatmap")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"full_correlation_heatmap_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_violin_funding_by_stage(self, data):
        """Plot violin plot of funding amount by funding stage"""
        plt.figure(figsize=(14, 8))
        sns.violinplot(
            data=data,
            x='funding_stage',
            y='funding_amount',
            scale='width',
            inner='quartile',
            palette='Set2')
        plt.yscale('log')
        plt.title('Funding Amount Distribution by Stage (Violin Plot)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"violin_funding_by_stage_{
                    self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()


class AdvancedVisualizer(Visualizer):
    def plot_roc_curves(self, y_true, y_proba, classes):
        plt.figure(figsize=(10, 8))
        y_bin = label_binarize(y_true, classes=classes)
        n_classes = len(classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Plot the ROC curve for each class
        colors = plt.colormaps['tab10'](np.linspace(0, 1, n_classes))
        for i in range(n_classes):
            try:
                # Check if there are positive samples for this class
                if np.sum(y_bin[:, i]) > 0:
                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    plt.plot(
                        fpr[i],
                        tpr[i],
                        color=colors[i],
                        lw=2,
                        label=f'ROC curve (class {classes[i]}, AUC = {roc_auc[i]:0.2f})')
                else:
                    logger.info(f"Skipping ROC curve for class {classes[i]} - no positive samples")
            except Exception as e:
                logger.warning(f"Error plotting ROC curve for class {classes[i]}: {str(e)}")
                continue
                
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right", fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "roc_curves.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_calibration(self, y_true, y_proba, n_bins=1):
        plt.figure(figsize=(10, 6))
        if y_proba is None:
             logger.warning("y_proba is None in plot_calibration. Skipping.")
             plt.close() # Close the empty figure
             return
        if y_proba.ndim == 1:
            logger.warning("Converting 1D probability array to 2D")
            y_proba = np.column_stack([1 - y_proba, y_proba])

        # Ensure y_true and y_proba have the same number of samples
        if len(y_true) != len(y_proba):
             logger.error(f"Mismatched lengths in plot_calibration: y_true ({len(y_true)}), y_proba ({len(y_proba)}). Skipping.")
             plt.close()
             return

        n_classes = y_proba.shape[1]
        unique_true_classes = np.unique(y_true)

        for class_idx in range(n_classes):
            # Check if this class index actually exists in the true labels
            if class_idx not in unique_true_classes:
                 logger.info(f"Skipping calibration plot for class {class_idx} as it's not present in y_true.")
                 continue

            try:
                binary_y = (y_true == class_idx).astype(int)
                # Check if there are any positive samples for this class after filtering
                if np.sum(binary_y) == 0:
                     logger.info(f"Skipping calibration plot for class {class_idx} - no positive samples in y_true for this class.")
                     continue

                class_proba = y_proba[:, class_idx]
                prob_true, prob_pred = calibration_curve(
                    binary_y, class_proba, n_bins=n_bins, strategy='quantile'
                )
                # Plot markers *with* connecting lines
                plt.plot(prob_pred, prob_true, marker='o', linestyle='-', # Changed linestyle back from 'none' to '-'
                         label=f'Class {class_idx}', alpha=0.7)
            except ValueError as ve:
                 # Catch specific ValueError from calibration_curve if bins are empty
                 logger.warning(f"Could not calculate calibration curve for class {class_idx} (likely due to empty bins): {str(ve)}")
                 continue
            except Exception as e:
                logger.warning(
                    f"Skipping calibration for class {class_idx}: { # <<< Error was here
                        str(e)}") # <<< Closing parenthesis added here
                continue
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated') # Added label for diagonal
        plt.xlabel('Mean Predicted Probability (Bin)')
        plt.ylabel('Fraction of Positives (Bin)')
        plt.title('Calibration Plot (One-vs-Rest, Quantile Binning)') # Updated title
        plt.legend(loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "calibration_plot.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_confidence_intervals(self, y_true, y_pred, y_proba):
        plt.figure(figsize=(12, 6))
        confidence = np.max(y_proba, axis=1)
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        accuracies = []
        for i in range(len(bins) - 1):
            mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
            if mask.sum() > 0:
                acc = accuracy_score(y_true[mask], y_pred[mask])
            else:
                acc = 0
            accuracies.append(acc)
        plt.errorbar(bin_centers, accuracies, xerr=0.05, fmt='o')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Accuracy')
        plt.title('Confidence vs Accuracy')
        plt.savefig(os.path.join(self.output_dir, "confidence_intervals.png"))
        if self.interactive:
            plt.show()
        plt.close()


class FundingStagePredictionPipeline:
    def __init__(self, base_dir="./", output_dir="./MainOutput", archive=False):
        """Initialize the complete pipeline"""
        self.base_dir = base_dir
        self.output_dir = output_dir

        # Create output directory structure
        self.models_dir = os.path.join(output_dir, "models")
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
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
        # Override the output_dir with our custom path
        if 'output_dir' in kwargs:
            kwargs['output_dir'] = './MainOutput'
        else:
            args = list(args)
            if len(args) > 1:
                args[1] = './MainOutput'
            else:
                args.append('./MainOutput')
            args = tuple(args)

        super().__init__(*args, **kwargs)
        self.model_trainer = EnhancedModelTrainer(self.models_dir)

        # Ensure all required directories exist before creating visualizer
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # Set to False to prevent interactive display
        self.visualizer = AdvancedVisualizer(self.viz_dir, interactive=False)
        self.model_manager = ModelManager(self.models_dir)
        # +++ Add TimeSeriesForecaster instance +++
        # self.time_series_forecaster = TimeSeriesForecaster(self.viz_dir) # Removed
        self.time_series_forecaster = TimeSeriesForecaster(self.viz_dir) # Initialize forecaster
        self._init_model_directory()
        # Add attributes to store mapping and feature names if needed later
        self.final_class_mapping = {}
        self.reverse_final_class_mapping = {}
        self.feature_names = []


    def run(self):
        try:
            logger.info("Starting funding stage prediction pipeline")
            
            # --- Initialize a dictionary to hold all results --- #
            all_model_results = {} # <<< ENSURE THIS LINE IS PRESENT AND AT THE TOP

            # Steps 1-3: Existing data loading and feature engineering
            merged_data = self.data_loader.merge_datasets()
            if merged_data.empty:
                logger.error("No data available. Exiting pipeline.")
                return False

            processed_data = self.feature_engineer.extract_features(
                merged_data)
            X, y = self.feature_engineer.prepare_model_data(processed_data)

            # --- Data Preprocessing & Cleaning ---
            # Remap all classes to be continuous from 0
            def remap_classes(y_series, original_labels_series=None):
                # Determine the unique classes to map
                # If original_labels_series is provided, use it to get the STRING labels
                # Otherwise, determine labels from the input y_series itself.
                # --- Always try to use original_labels_series if provided --- #
                if original_labels_series is not None and not original_labels_series.empty:
                    # Convert to string explicitly for consistent mapping keys
                    unique_classes = sorted(original_labels_series.dropna().astype(str).unique())
                    logger.debug(f"remap_classes using labels from original_labels_series: {unique_classes}")
                elif y_series.dtype == 'object': # Fallback if original labels not provided
                    unique_classes = sorted(y_series.dropna().unique()) # Gets unique STRING labels
                    logger.debug(f"remap_classes using labels from object y_series: {unique_classes}")
                else: # Assume numeric input if no original labels provided
                    unique_classes = sorted(y_series.dropna().unique()) # Gets unique NUMERIC labels
                    logger.debug(f"remap_classes using labels from numeric y_series: {unique_classes}")
                # Convert all unique classes to string for the map keys
                unique_classes = [str(cls) for cls in unique_classes]

                # Create the map: {original_label_string -> final_index}
                class_map = {
                    str(label): idx for idx, label in enumerate(unique_classes) # Ensure label is string
                }

                # --- Revised Mapping Application --- #
                # If original_labels_series is provided, map IT using class_map
                if original_labels_series is not None and not original_labels_series.empty:
                    # Map the ORIGINAL labels to the new final indices
                    remapped_y = original_labels_series.astype(str).map(class_map).fillna(-1).astype(int)
                    # Return the remapped y and the {original_string: final_index} map
                    return remapped_y, class_map
                else:
                     # Map the input y_series (likely strings or initial numbers)
                    remapped_y = y_series.astype(str).map(class_map).fillna(-1).astype(int)
                     # Return the remapped y and the {original_string_or_initial_num: final_index} map
                    return remapped_y, class_map

            # Ensure target variable exists
            # --- Use the ORIGINAL 'funding_stage' column for initial mapping --- #
            if 'funding_stage' not in processed_data.columns:
                logger.error("Original target column 'funding_stage' not found.")
                return False
            # Store the original string labels before any remapping
            original_string_labels = processed_data['funding_stage'].copy()

            # --- First remap: original string -> initial index (0-based continuous) --- # 
            y_initial_indices, initial_map_str_to_idx = remap_classes(original_string_labels)
            # initial_map_str_to_idx should be {original_label_string: initial_index}
            logger.info(f"Initial class mapping (String Label -> Initial Index): {initial_map_str_to_idx}")

            # Handle rare classes (e.g., less than 10 samples) using the initial numeric indices
            class_counts = pd.Series(y_initial_indices).value_counts()
            min_samples_threshold = 10
            rare_classes = class_counts[class_counts < min_samples_threshold].index.tolist()
            y_merged = y_initial_indices.copy() # Work on a copy
            if rare_classes:
                majority_class_index = class_counts.idxmax()
                # Use the initial numeric y for modification
                y_merged = y_merged.apply(lambda x: majority_class_index if x in rare_classes else x)
                logger.info(
                    f"Merged rare classes (based on initial indices) into majority class index {majority_class_index}")

            # --- Second remap to ensure final indices are continuous from 0 --- #
            # Map the MERGED initial indices (y_merged) to final indices (0-based continuous)
            unique_merged_indices = sorted(y_merged.dropna().unique())
            initial_idx_to_final_idx_map = { 
                initial_idx: final_idx for final_idx, initial_idx in enumerate(unique_merged_indices)
            }
            y_final = y_merged.map(initial_idx_to_final_idx_map).fillna(-1).astype(int)
            logger.info(
                f"Final map (Initial Index -> Final Index) after merging rare classes: {initial_idx_to_final_idx_map}")

            # --- Create and Store the REQUIRED map: {final_index: original_label_string} --- #
            # Link: final_index -> initial_index -> original_label_string
            initial_map_idx_to_str = {v: k for k, v in initial_map_str_to_idx.items()} # {initial_index: original_string}
            self.final_index_to_string_label_map = {}
            for initial_index, final_index in initial_idx_to_final_idx_map.items():
                 original_string_label = initial_map_idx_to_str.get(int(initial_index))
                 if original_string_label is not None:
                     self.final_index_to_string_label_map[final_index] = original_string_label # Keep it as string
                 else:
                     logger.warning(f"Could not find original string label for initial index {initial_index} when creating final map.")
                     self.final_index_to_string_label_map[final_index] = f"LABEL_NOT_FOUND_FOR_INDEX_{initial_index}"

            # --- Assign the final target vector for training --- #
            y = y_final # Use the final, remapped, continuous 0-based indices for training
            
            # --- Ensure X is aligned with the final y before proceeding ---
            # (This step is crucial if any rows were dropped during y processing)
            if len(X) != len(y):
                 logger.warning(f"X ({len(X)}) and final y ({len(y)}) have different lengths. Aligning X based on y's index.")
                 # Assuming y's index corresponds to the original DataFrame before filtering
                 # This might need adjustment based on how X and y indices are managed
                 # A safer approach might be to filter X earlier based on valid y
                 valid_y_indices = y.index # Get index from the final y Series
                 X = X.loc[valid_y_indices] # Filter X using the valid indices from y
                 if len(X) != len(y): # Check alignment again
                     logger.error("CRITICAL: Failed to align X and y after remapping. Aborting.")
                     return False # Stop if alignment fails
            
            X = X.reset_index(drop=True) # Reset index for both after alignment
            y = y.reset_index(drop=True)

            # Store the reverse mapping for visualization use # MOVED LOGGING AFTER y IS ASSIGNED
            logger.info(f"Correct Final Index -> String Label mapping for saving: {self.final_index_to_string_label_map}")

            # --- Feature Selection Removed ---
            logger.info(f"Using all {X.shape[1]} engineered features for training.")
            if hasattr(X, 'columns'):
                 self.feature_names = X.columns.tolist()
            else: # Handle numpy array case
                 # Convert X to DataFrame if it's not already
                 if not isinstance(X, pd.DataFrame):
                      X = pd.DataFrame(X)
                 self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                 X.columns = self.feature_names # Assign columns to the DataFrame
                 logger.warning("X was not a DataFrame, assigned generic feature names.")

            # +++ Fit Anomaly Detector on Core Numeric Features BEFORE Model Training Split +++
            # (Keep the existing anomaly detector fitting logic here for the main pipeline)
            try:
                logger.info("Attempting to fit main anomaly detector on core numeric features...")
                core_numeric_features_list = [
                    'funding_amount_log', 'employees', 'employee_efficiency',
                    'funding_year', 'funding_month', 'previous_rounds',
                    'months_since_first_funding', 'funding_velocity',
                    'month_sin', 'month_cos',
                    'funding_amount_x_age', 'employees_x_rounds'
                    # Add 'velocity_x_rounds', 'age_x_employees' if they exist
                ] + [f for f in ['velocity_x_rounds', 'age_x_employees'] if f in X.columns]

                existing_core_features_main = [f for f in core_numeric_features_list if f in X.columns]
                logger.info(f"Core features selected for main anomaly detector: {existing_core_features_main}")

                if existing_core_features_main:
                    X_core_numeric_main = X[existing_core_features_main].copy()
                    scaler_anomaly_main = StandardScaler()
                    X_core_numeric_scaled_main = scaler_anomaly_main.fit_transform(X_core_numeric_main)
                    self.model_manager.fit_anomaly_detector(X_core_numeric_scaled_main, existing_core_features_main)
                    logger.info(f"Main anomaly detector fitted on {len(existing_core_features_main)} core features.")
                else:
                    logger.warning("No core numeric features found in X for main anomaly detector.")
            except Exception as anomaly_fit_err_main:
                logger.error(f"Error fitting main anomaly detector on core features: {anomaly_fit_err_main}", exc_info=True)
            # +++ End Main Anomaly Detector Fitting +++

            # --- Prepare Data & Train/Save Dashboard-Specific Model --- #
            logger.info("*** Preparing data and training dashboard-specific model... ***")
            
            # Use ALL features from the main prepared dataset X for the dashboard model
            all_features_for_dashboard = X.columns.tolist() # Get all columns from the final prepared X
            logger.info(f"Using ALL {len(all_features_for_dashboard)} features for the dashboard model.")

            # Check if features actually exist in X (sanity check)
            if not all(f in X.columns for f in all_features_for_dashboard):
                missing_dash = [f for f in all_features_for_dashboard if f not in X.columns]
                logger.error(f"Dashboard Prep Error: Missing expected features in main X: {missing_dash}")
                # Skip dashboard model training if features are missing
            else:
                # Select ALL features for the dashboard model
                X_dashboard = X[all_features_for_dashboard].copy()
                y_dashboard = y # Directly use the final, aligned, re-indexed y variable

                logger.info(f"Dashboard model features (ALL): {len(all_features_for_dashboard)} features")
                logger.info(f"Dashboard data shapes: X={X_dashboard.shape}, y={y_dashboard.shape}")

                # Log y_dashboard values (optional - keep if needed)
                logger.info("--- Checking y_dashboard values before split ---")
                # ... (logging code) ...
                logger.info("------------------------------------------------")

                # Train/test split for dashboard model
                X_train_dash, X_test_dash, y_train_dash, y_test_dash = train_test_split(
                    X_dashboard, y_dashboard, test_size=0.2, random_state=42, stratify=y_dashboard
                )

                # Scale ALL features for the dashboard model
                scaler_dash = StandardScaler()
                X_train_dash_scaled = scaler_dash.fit_transform(X_train_dash)
                X_test_dash_scaled = scaler_dash.transform(X_test_dash)

                # Train RandomForest for the dashboard (using ALL scaled features)
                logger.info("Training dashboard RandomForest with ALL features...") # Updated log
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    dashboard_rf = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        class_weight='balanced',
                        n_jobs=-1
                    )
                    dashboard_model_trained = dashboard_rf.fit(X_train_dash_scaled, y_train_dash)
                    dash_acc = dashboard_model_trained.score(X_test_dash_scaled, y_test_dash)
                    logger.info(f"Dashboard RandomForest (All Features) Test Accuracy: {dash_acc:.4f}") # Updated log

                    # Fit Anomaly Detector specifically for these features/scaling
                    # Note: Anomaly detector now also uses all 110 features
                    anomaly_detector_dash = AnomalyDetector(contamination=0.05)
                    # We need X_train_dash_scaled (which has all features) for fitting
                    anomaly_detector_dash.fit(X_train_dash_scaled) 
                    logger.info("Fitted anomaly detector specifically for ALL dashboard model features.") # Updated log

                    # Save this specific model, scaler, FULL features list, and detector
                    dash_metadata = {
                        'model_type': 'RandomForest_Dashboard_AllFeatures', # New type name
                        'features': all_features_for_dashboard, # Save the full list
                        'accuracy': dash_acc
                    }
                    # model_filename_dash = self._generate_filename("Dashboard_Model", dash_metadata['model_type']) # Error: Method does not exist here
                    # Manually construct filename based on timestamp and type
                    timestamp_str = datetime.now().strftime("%Y%m%d%H%M")
                    model_filename_dash = f"Dashboard_Model_{dash_metadata['model_type']}_v{timestamp_str}.joblib"
                    model_path_dash = os.path.join(self.models_dir, model_filename_dash)
                    
                    # Ensure the map being saved is the correct final one
                    final_map_to_save = getattr(self, 'final_index_to_string_label_map', {})
                    logger.info(f"Saving final map to dashboard model: {final_map_to_save}")

                    joblib.dump({
                        'model': dashboard_model_trained,
                        'scaler': scaler_dash,
                        'feature_names': all_features_for_dashboard, # Explicitly save all names
                        'class_mapping': final_map_to_save, # Save the confirmed final map
                        'anomaly_detector': anomaly_detector_dash,
                        'training_metadata': dash_metadata
                    }, model_path_dash)
                    logger.info(f"Dashboard-specific model (All Features) saved to {model_path_dash}")
                    # Verify save
                    if os.path.exists(model_path_dash):
                        logger.info(f"CONFIRMED: Dashboard model file (All Features) exists at {model_path_dash}")
                    else:
                        logger.error(f"FAILED TO SAVE Dashboard model file (All Features) at {model_path_dash}")

                except ImportError:
                    logger.error("Scikit-learn not installed? Cannot train dashboard RandomForest.")
                except Exception as e:
                    logger.error(f"Error training/saving dashboard model (All Features): {e}", exc_info=True)
            # --- End Dashboard Model Section --- #

            # --- Time Series Forecasting & Prototype Plot (Before Model Training) --- #
            logger.info("Step 3.5: Generating Time Series Forecast & Dashboard Prototype Plot...")
            if Prophet is not None: # Check if Prophet was imported successfully
                try:
                    # Use processed_data as it contains necessary columns before X,y split
                    prophet_data = self.time_series_forecaster.prepare_prophet_data(processed_data)
                    if prophet_data is not None and not prophet_data.empty:
                        prophet_model, prophet_forecast = self.time_series_forecaster.train_predict(prophet_data, periods=6)
                        if prophet_model and prophet_forecast is not None:
                            # Pass the original prophet_data as history_df for plotting actuals
                            self.time_series_forecaster.plot_dashboard_prototype(prophet_forecast, prophet_data)
                        else:
                            logger.warning("Prophet model training/prediction failed. Skipping dashboard plot.")
                    else:
                        logger.warning("Data preparation for Prophet failed or yielded no data. Skipping forecast plot.")
                except Exception as ts_err:
                    logger.error(f"Error during time series forecasting step: {ts_err}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning("Prophet not installed, skipping time series forecasting and dashboard prototype plot.")
            # --- End Time Series Section --- #

            # Step 4: Enhanced model training (using full feature set X)
            logger.info("Step 4: Training models with default parameters...") # Updated log
            # Ensure X and y are aligned after potential filtering/remapping
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            # --- Use Default Random Forest --- #
            logger.info("Training Random Forest with defaults (balanced class weights)...") # Updated log
            # Add class_weight='balanced'
            default_rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced', n_estimators=50) # Reduced n_estimators
            
            rf_model, rf_results = self._train_final_model(
                 default_rf, X.copy(), y.copy(), 'Random Forest' # <<< Use default_rf directly
            )


            # --- Use Default XGBoost --- #
            logger.info("Training XGBoost with defaults...")
            try:
                 from xgboost import XGBClassifier
                 n_classes_xgb = len(np.unique(y))
                 default_xgb = XGBClassifier(
                      random_state=42,
                      num_class=n_classes_xgb,
                      eval_metric='mlogloss',
                      verbosity=0
                 )
                 xgb_model, xgb_results = self._train_final_model(
                      default_xgb, X.copy(), y.copy(), 'XGBoost' # <<< Use default_xgb directly
                 )
            except ImportError:
                 logger.warning("XGBoost not found. Skipping XGBoost model.")
                 xgb_model, xgb_results = None, None
            except Exception as xgb_err:
                 logger.error(f"Error training default XGBoost: {xgb_err}")
                 xgb_model, xgb_results = None, None

            # --- Use Default Gradient Boosting --- #
            logger.info("Training Gradient Boosting with defaults...")
            default_gb = GradientBoostingClassifier(random_state=42) # Initialize with defaults
            gb_model, gb_results = self._train_final_model(
                default_gb, X.copy(), y.copy(), 'Gradient Boosting'
            )
            # +++ End Training for Gradient Boosting +++

            # +++ Decision Tree Training (Moved Before Stacking) +++
            logger.info("Training Decision Tree with defaults...")
            from sklearn.tree import DecisionTreeClassifier
            default_dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
            dt_model, dt_results = self._train_final_model(
                default_dt, X.copy(), y.copy(), 'Decision Tree'
            )
            if dt_results: all_model_results['Decision Tree (Calibrated)'] = dt_results # <<< Store results

            # --- REMOVE Logistic Regression ---
            lr_model, lr_results = None, None # Ensure results are None

            # --- K-Nearest Neighbors (Skipped) ---
            logger.info("Training K-Nearest Neighbors (k=7)...")
            knn_model, knn_results = None, None # Skip KNN

            # --- SVC (Skipped) ---
            logger.info("Training Support Vector Classifier (Linear Kernel)...")
            svc_model, svc_results = None, None # Skip SVC

            # --- MLP Classifier (Skipped) ---
            logger.info("Training MLP Classifier...")
            mlp_model, mlp_results = None, None # Skip MLP

            # --- Stacking Ensemble ---
            logger.info("Training and evaluating Stacking Ensemble (Random Forest Meta-Model)...")
            stacking_ensemble_results = None # Initialize to None
            meta_model_trained = None
            stacking_scaler = None
            try:
                # --- Pass UNFITTED base models to the trainer method ---
                # Create a dict of the *initial default* model instances
                unfitted_base_models = {}
                # --- Use direct assignment, assuming variables exist --- #
                if 'default_rf' in locals() and default_rf is not None: unfitted_base_models['RandomForest'] = default_rf
                if 'default_xgb' in locals() and default_xgb is not None: unfitted_base_models['XGBoost'] = default_xgb
                if 'default_gb' in locals() and default_gb is not None: unfitted_base_models['GradientBoosting'] = default_gb
                # --- Add Default Decision Tree to base models (it's now defined) --- #
                if 'default_dt' in locals() and default_dt is not None: unfitted_base_models['DecisionTree'] = default_dt

                # +++ Log base models being passed +++
                logger.info(f"Base models passed to train_stacked_ensemble: {list(unfitted_base_models.keys())}")

                if len(unfitted_base_models) >= 2:
                    # Train the ensemble. X and y should be the full dataset.
                    # Expect tuple: (model, accuracy, y_proba, scaler)
                    stacking_output = self.model_trainer.train_stacked_ensemble(
                        X.copy(), y.copy(), unfitted_base_models # Pass unfitted models
                    )
                    # +++ Log the output of train_stacked_ensemble +++
                    logger.info(f"Output from train_stacked_ensemble: {type(stacking_output)}")

                    # Check if tuple has expected length before unpacking
                    if isinstance(stacking_output, tuple) and len(stacking_output) == 4:
                        meta_model_trained, stack_accuracy, stack_y_proba, stacking_scaler = stacking_output
                        # +++ Log successful unpacking +++
                        logger.info(f"Successfully unpacked stacking_output. Accuracy: {stack_accuracy:.4f}")
                    else:
                         logger.error("train_stacked_ensemble returned unexpected output, cannot unpack.")
                         meta_model_trained = None # Ensure it's None
                    
                    if meta_model_trained:
                        # Re-split data using the SAME random_state to get consistent test set
                        _, X_test_stack, _, y_test_stack = train_test_split(X.copy(), y.copy(), test_size=0.2, random_state=42, stratify=y.copy())
                        # Scale the test set using the scaler returned by the ensemble trainer
                        X_test_stack_scaled = stacking_scaler.transform(X_test_stack)
                        # Predict on the scaled test set
                        stack_y_pred = meta_model_trained.predict(X_test_stack_scaled)
                        
                        stacking_ensemble_results = {
                            'model': meta_model_trained,
                            'calibrated_model': meta_model_trained, # Stacking model itself is not typically calibrated further here
                            'accuracy': stack_accuracy,
                            'precision': precision_score(y_test_stack, stack_y_pred, average='weighted', zero_division=0),
                            'recall': recall_score(y_test_stack, stack_y_pred, average='weighted', zero_division=0),
                            'f1_score': f1_score(y_test_stack, stack_y_pred, average='weighted', zero_division=0),
                            'rmse': np.sqrt(mean_squared_error(y_test_stack, stack_y_pred)),
                            'confusion_matrix': confusion_matrix(y_test_stack, stack_y_pred),
                            'classification_report': classification_report(y_test_stack, stack_y_pred, output_dict=True, zero_division=0),
                            'cross_val_scores': [stack_accuracy], # Simplified CV score from test split
                            'X_test': X_test_stack_scaled, # Return scaled test data
                            'y_test': y_test_stack,
                            'y_pred': stack_y_pred,
                            'y_proba': stack_y_proba,
                            'feature_names': self.feature_names, # Original features input to stacking
                            'scaler': stacking_scaler, # Return the scaler used for stacking
                            'anomaly_detector': None # Anomaly detection for ensemble is complex
                        }
                        # +++ Log successful result creation +++
                        logger.info(f"Created stacking_ensemble_results dictionary. Keys: {list(stacking_ensemble_results.keys())}")
                        if stacking_ensemble_results: all_model_results['Stacking Ensemble (RF Meta)'] = stacking_ensemble_results # <<< Store results
                    else:
                        logger.warning("Stacking ensemble training returned no model.")
                else:
                    logger.warning(f"Not enough valid base models ({len(unfitted_base_models)}) for stacking ensemble. Skipping.")
                    stacking_ensemble_results = None
            except Exception as stack_err:
                 logger.error(f"Error training or evaluating stacking ensemble: {stack_err}", exc_info=True) # Log traceback
                 stacking_ensemble_results = None
            # --- End Stacking ---

            # --- LightGBM (Skipped) ---
            lgbm_results = None

            # Step 5: Advanced visualizations
            logger.info("Step 5: Creating advanced visualizations...")
            # Pass the dictionary containing *all* results
            self._create_advanced_visualizations(
                all_model_results.get('Random Forest (Calibrated)'), # Get RF result dict
                all_model_results.get('XGBoost (Calibrated)'),     # Get XGB result dict
                all_model_results.get('Gradient Boosting (Calibrated)'), # Get GB result dict
                all_model_results.get('Decision Tree (Calibrated)'), # Get DT result dict
                stacking_ensemble_results, # Pass stacking results (might be None)
                y,
                processed_data)

            # Step 6: Save Summary and Models
            logger.info("Step 6: Saving summary and best models...")
            # Pass the consolidated results dictionary
            summary_data = self._save_summary(
                 merged_data, X, all_model_results, processed_data # <<< Pass all_model_results
            )

            if not summary_data:
                 logger.error("Failed to generate or save pipeline summary.")
                 return False # Indicate failure if summary couldn't be saved

            # --- Final Logging based on success --- #
            # The saving logic inside _save_summary now handles the final success/failure logging
            # We just check if summary_data was returned

            logger.info("Pipeline completed successfully!") # Moved here
            return True # Indicate overall success

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            return False

    # --- Moved Method Definitions Start ---
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
            rf_results,  # Now expects the full results dictionary
            xgb_results, # Now expects the full results dictionary
            gb_results,  # Now expects the full results dictionary
            dt_results,  # Now expects the full results dictionary
            stacking_ensemble_results, # Use stacking ensemble results (will be None now)
            y, # Original y before splitting (needed for overall class info)
            processed_data): # Pass processed data for visualizations
        """Generate advanced model diagnostics and data visualizations""" # Updated docstring
        try:
            unique_classes = np.unique(y) # Use classes from the final y before split

            # Create a combined dictionary for plotting, ensure results are valid
            all_results_plot = {}
            # --- Add checks for None before accessing 'model' --- #
            if rf_results and rf_results.get('model'): all_results_plot['Random Forest'] = rf_results
            if xgb_results and xgb_results.get('model'): all_results_plot['XGBoost'] = xgb_results
            if gb_results and gb_results.get('model'): all_results_plot['Gradient Boosting'] = gb_results
            if dt_results and dt_results.get('model'): all_results_plot['Decision Tree'] = dt_results
            if stacking_ensemble_results and stacking_ensemble_results.get('model'): all_results_plot['Stacking Ensemble (RF Meta)'] = stacking_ensemble_results # <<< Corrected Name for consistency

            if not all_results_plot:
                 logger.warning("No valid model results available for visualization.")
                 return # Exit early if no models to visualize

            # --- Model Diagnostic Plots --- #
            logger.info("Creating advanced model diagnostic visualizations...")
            # Plot ROC/Calibration/Confidence only for models with probabilities
            for model_name, results in all_results_plot.items():
                 # --- ADDED check 'if results and ...' ---
                 if results and results.get('y_proba') is not None and results.get('y_test') is not None and results.get('y_pred') is not None:
                     # Ensure y_test and y_proba have compatible shapes/classes
                     y_test_current = results['y_test']
                     y_proba_current = results['y_proba']
                     current_classes = np.unique(y_test_current) # Use classes present in y_test

                     if len(current_classes) > 1: # Need at least 2 classes for ROC/Calibration
                         try:
                             if y_proba_current.shape[1] == len(unique_classes):
                                 self.visualizer.plot_roc_curves(
                                     y_test_current,
                                     y_proba_current,
                                     classes=unique_classes)
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
                     logger.info(f"Skipping probability-based plots for {model_name} (results, probabilities, y_test or y_pred not available)")

            # Plot feature importance for each base model if available
            base_models_for_fi = {
                # --- Add None checks when getting 'model' --- #
                'Random Forest': rf_results.get('model') if rf_results else None,
                'XGBoost': xgb_results.get('model') if xgb_results else None,
                'Gradient Boosting': gb_results.get('model') if gb_results else None,
                'Decision Tree': dt_results.get('model') if dt_results else None
            }
            base_results_for_fi = {
                'Random Forest': rf_results,
                'XGBoost': xgb_results,
                'Gradient Boosting': gb_results,
                'Decision Tree': dt_results
            }

            for model_name, model_instance in base_models_for_fi.items():
                 results_dict = base_results_for_fi.get(model_name, {}) # Get corresponding results dict
                 # --- ADDED checks for model_instance and results_dict --- #
                 if model_instance and results_dict and hasattr(model_instance, 'feature_importances_') and 'feature_names' in results_dict:
                     try:
                         if len(model_instance.feature_importances_) == len(results_dict['feature_names']):
                             self.visualizer.plot_feature_importance(
                                 model_instance, results_dict['feature_names'])
                         else:
                              logger.warning(f"Mismatch between feature importances ({len(model_instance.feature_importances_)}) and feature names ({len(results_dict['feature_names'])}) for {model_name}. Skipping FI plot.")
                     except Exception as fi_err:
                         logger.warning(f"Could not plot Feature Importance for {model_name}: {fi_err}")
                 else:
                      logger.info(f"Skipping Feature Importance for {model_name} (instance, results, FI, or feature names missing)")

            # Compare model performances (including RMSE)
            if all_results_plot:
                self.visualizer.plot_model_comparison(all_results_plot)
                self.visualizer.plot_confusion_matrices(all_results_plot)

            # ... (rest of visualization logic) ...
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            logger.error(traceback.format_exc()) # Log full traceback

    def _save_summary(self, merged_data, X, model_results_summary, processed_data):
        # model_results_summary now contains all valid results passed from run()
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

        # --- Remove manual addition of DT results - it should be in model_results_summary now ---
        # if 'Decision Tree (Calibrated)' not in model_results_summary and hasattr(self, 'dt_results') and self.dt_results:
        #      logger.info("Manually adding Decision Tree results to summary dictionary.")
        #      model_results_summary['Decision Tree (Calibrated)'] = self.dt_results # Add it if missing

        # Add metrics for each model (Iterate over the passed dictionary)
        best_accuracy = -1.0
        best_model_name = None
        for model_name, results in model_results_summary.items(): # Iterate directly
            if results is None: # Skip if a model failed
                 summary['metrics'][model_name] = None
                 continue
            # Safely get metrics using .get()
            accuracy = float(results.get('accuracy', np.nan))
            summary['metrics'][model_name] = {
                'accuracy': accuracy,
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
        # 1. Save the overall best model (could be base or ensemble) -> WITHOUT Dashboard Prefix
        saved_path = None # Initialize
        if best_overall_model_name and model_to_save and scaler_to_save and feature_names_to_save:
            # Sanitize name for filename
            safe_model_name = best_overall_model_name.replace(" ", "_").replace("(", "").replace(")", "")
            # Get the anomaly detector associated with the best model
            anomaly_detector_to_save = results_to_use.get('anomaly_detector')
            if not anomaly_detector_to_save:
                logger.warning(f"Anomaly detector instance not found in results for {best_overall_model_name}. Model will be saved without it.")

            saved_path = self.model_manager.save_model(
                model_name=safe_model_name, # SAVE WITHOUT DASHBOARD PREFIX
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

        # --- Explicitly Save Key Base Models for Dashboard --- #
        logger.info("Attempting to explicitly save key base models for dashboard use...")
        base_models_to_save = {
            'XGBoost (Calibrated)': 'XGBoost_(Calibrated)',
            'Random Forest (Calibrated)': 'Random_Forest_(Calibrated)',
            'Gradient Boosting (Calibrated)': 'Gradient_Boosting_(Calibrated)',
            'Decision Tree (Calibrated)': 'Decision_Tree_(Calibrated)' # <<< Ensure DT is here
        }

        for model_key, save_name_suffix in base_models_to_save.items():
            # --- Ensure we check the potentially updated model_results_summary --- #
            if model_key in model_results_summary and model_results_summary[model_key]:
                results_data = model_results_summary[model_key]
                # --- Use calibrated_model for saving ---
                base_model_to_save = results_data.get('calibrated_model') # <<< Get calibrated model
                base_scaler_to_save = results_data.get('scaler')
                base_feature_names = results_data.get('feature_names')
                base_metrics = summary['metrics'].get(model_key, {}) # Get metrics
                base_anomaly_detector = results_data.get('anomaly_detector') # Get detector
                
                if base_model_to_save and base_scaler_to_save and base_feature_names:
                    # Save with the standard name
                    logger.info(f"Saving base model: {model_key} as {save_name_suffix}")
                    base_saved_path = self.model_manager.save_model(
                        model_name=save_name_suffix,
                        model=base_model_to_save, # Save calibrated model
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
                        
                        # --- Save specific models with Dashboard_Model_ prefix --- #
                        # Save Decision Tree and XGBoost with the prefix
                        if model_key in ['Decision Tree (Calibrated)', 'XGBoost (Calibrated)']:
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
                     logger.warning(f"Skipping save for {model_key} - missing calibrated model, scaler, or feature names in results.")
            else:
                # Log why DT might be skipped more specifically
                if model_key == 'Decision Tree (Calibrated)':
                    if model_key not in model_results_summary:
                         logger.warning(f"Results for base model {model_key} not found in summary dict. Cannot save explicitly.")
                    elif not model_results_summary[model_key]:
                         logger.warning(f"Results for base model {model_key} are None or invalid. Cannot save explicitly.")
                else: # Corrected indentation for the else statement
                    logger.warning(f"Results for base model {model_key} not found or invalid in summary. Cannot save explicitly.") # Corrected indentation for the logger
        # --- End Explicit Base Model Save --- #

        # --- Explicitly Save Stacking Ensemble for Dashboard --- #
        stacking_key = 'Stacking Ensemble (RF Meta)'
        stacking_save_name_suffix = 'Stacking_Ensemble_(RF_Meta)'
        if stacking_key in model_results_summary and model_results_summary[stacking_key]:
            logger.info(f"Attempting to explicitly save Stacking Ensemble ({stacking_key}) for dashboard use...")
            stacking_results_data = model_results_summary[stacking_key]
            stacking_model_to_save = stacking_results_data.get('model') 
            stacking_scaler_to_save = stacking_results_data.get('scaler') 
            stacking_feature_names_to_save = stacking_results_data.get('feature_names')

            # --- Get the best overall model's results to source the anomaly detector ---
            best_overall_model_results_dict = None
            # Ensure best_overall_model_name is defined and is a key in model_results_summary
            if 'best_overall_model_name' in locals() and best_overall_model_name and best_overall_model_name in model_results_summary:
                best_overall_model_results_dict = model_results_summary.get(best_overall_model_name)
            elif 'best_overall_model_name' in locals() and best_overall_model_name:
                 logger.warning(f"Results for best_overall_model_name ('{best_overall_model_name}') not found in model_results_summary. Cannot get default anomaly detector.")
            else:
                logger.warning("'best_overall_model_name' not defined or not available. Cannot get default anomaly detector.")

            default_anomaly_detector = None
            if best_overall_model_results_dict and isinstance(best_overall_model_results_dict, dict):
                default_anomaly_detector = best_overall_model_results_dict.get('anomaly_detector')
            elif best_overall_model_results_dict is not None: # It was found but not a dict
                 logger.warning(f"best_overall_model_results_dict for '{best_overall_model_name}' is not a valid dictionary. Default anomaly detector will be None.")
            # If best_overall_model_results_dict is None, it was already logged.

            if stacking_model_to_save and stacking_scaler_to_save and stacking_feature_names_to_save:
                logger.info("Constructing metadata for Stacking Ensemble...")
                # --- BEGIN DIAGNOSTIC LOGGING ---
                logger.info(f"[DIAGNOSTIC] summary object type: {type(summary)}")
                if isinstance(summary, dict):
                    logger.info(f"[DIAGNOSTIC] summary keys: {list(summary.keys())}")
                    feature_eng_summary = summary.get('feature_engineering_summary')
                    logger.info(f"[DIAGNOSTIC] feature_engineering_summary type: {type(feature_eng_summary)}")
                    if isinstance(feature_eng_summary, dict):
                        logger.info(f"[DIAGNOSTIC] feature_engineering_summary keys: {list(feature_eng_summary.keys())}")
                        logger.info(f"[DIAGNOSTIC] age_bin_edges in feature_eng_summary: {feature_eng_summary.get('age_bin_edges')}")
                        logger.info(f"[DIAGNOSTIC] age_bin_labels in feature_eng_summary: {feature_eng_summary.get('age_bin_labels')}")
                # --- END DIAGNOSTIC LOGGING ---
                stacking_model_metadata = {
                    "model_name": stacking_key,
                    "description": "Stacking Ensemble with Random Forest Meta-Learner, specifically for dashboard use.",
                    "base_models_info": stacking_results_data.get("base_models_info", "Not available"),
                    "meta_learner_info": "RandomForestClassifier (default settings in train_stacked_ensemble)",
                    "accuracy": stacking_results_data.get("accuracy"),
                    "class_mapping": summary.get('class_mapping'),
                    "feature_names": stacking_feature_names_to_save, 
                    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "version": f"v{datetime.now().strftime('%Y%m%d%H%M')}",
                    "age_bin_edges": summary.get('feature_engineering_summary', {}).get('age_bin_edges'),
                    "age_bin_labels": summary.get('feature_engineering_summary', {}).get('age_bin_labels'),
                    "anomaly_detector_source": "from_best_overall_model" if default_anomaly_detector else "not_included"
                }
                logger.info(f"[DIAGNOSTIC] Constructed stacking_model_metadata: {stacking_model_metadata}") # Log the whole dict
                # We will pass default_anomaly_detector directly to _save_model_artifact for inclusion if it exists
                logger.info(f"Stacking Ensemble metadata constructed. Anomaly detector source: {stacking_model_metadata.get('anomaly_detector_source')}")

                logger.info(f"Calling model_manager.save_model for Stacking Ensemble: {stacking_save_name_suffix}") # Corrected log message
                self.model_manager.save_model( # Corrected method call
                    model_name=f"Dashboard_Model_{stacking_save_name_suffix}",
                    model=stacking_model_to_save,
                    scaler=stacking_scaler_to_save,
                    feature_names=stacking_feature_names_to_save,
                    metadata=stacking_model_metadata,
                    anomaly_detector=default_anomaly_detector 
                )
                logger.info(f"Dashboard Stacking Ensemble ({stacking_key}) model artifact explicitly saved.")
            else:
                logger.warning(f"Could not explicitly save Stacking Ensemble ({stacking_key}) for dashboard - model, scaler, or feature_names missing.")
        else:
            logger.warning(f"Results for Stacking Ensemble ({stacking_key}) not found in summary. Cannot save explicitly for dashboard.")
        # --- End Explicit Stacking Ensemble Save --- #


        # --- Create a general summary JSON --- #
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


