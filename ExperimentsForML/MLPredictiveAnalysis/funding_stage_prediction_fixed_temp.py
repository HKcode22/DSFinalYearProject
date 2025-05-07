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
    mean_squared_error,
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
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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
        """Prepare feature matrix with proper type handling and no dummy data"""
        # Select relevant features that exist in the original data
        feature_cols = [
            'funding_amount_log', 'employees', 'employee_efficiency',
            'funding_year', 'funding_month', 'previous_rounds',
            'months_since_first_funding', 'funding_velocity'
        ]

        # Only use features that actually exist in the data
        features_to_use = [col for col in feature_cols if col in data.columns]

        # Log features used
        logger.info(f"Preparing model data with features: {features_to_use}")

        # Clean feature data - ensure numeric types
        X = data[features_to_use].copy()

        # Convert all features to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Fill missing values for numeric columns - using median from actual
        # data
        numeric_cols = X.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            if X[col].isna().any():
                median_value = X[col].median()
                X[col] = X[col].fillna(median_value)
                logger.info(
                    f"Filled NaN values in {col} with median: {median_value}")

        # Target variable processing
        y = pd.to_numeric(data['funding_stage_numeric'], errors='coerce')
        valid_mask = y.notna()

        # Check we have enough data
        if valid_mask.sum() < 10:
            logger.warning(
                f"Very few valid target values: {
                    valid_mask.sum()} out of {
                    len(y)}")

        X = X[valid_mask]
        y = y[valid_mask].astype(int)

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
            # SMOTE has been removed to avoid errors with small sample sizes

        # Define model with hyperparameters - use bootstrap aggregating for robustness
        # and randomized state for unpredictability
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
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

        # Return model and evaluation data
        return rf, {
            'accuracy': accuracy,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': rf.predict_proba(X_test) if hasattr(rf, 'predict_proba') else None,
            'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None,
            'model_path': model_path,
            'isolation_forest': isolation_forest
        }

    def train_xgboost(self, X, y):
        """
        Train an XGBoost classifier with enhanced hyperparameters.
        
        Args:
            X: Features matrix
            y: Target vector
            
        Returns:
            Dictionary with model, predictions, probabilities and metrics
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
                n_estimators=200,
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
            return None

    def train_gradient_boosting(self, X, y):
        """Train a Gradient Boosting model with optimized parameters"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define model with carefully selected parameters
        gb = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
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
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log the performance
        logger.info(f"Gradient Boosting Performance: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return gb, {
            'model': gb,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': gb.feature_importances_
        }


class EnhancedModelTrainer(ModelTrainer):
    def tune_random_forest(self, X, y):
        """Use RandomizedSearchCV to find optimal Random Forest parameters"""
        param_grid = {
            'n_estimators': randint(200, 800),
            'max_depth': [None] + list(range(10, 50, 5)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_leaf_nodes': [None] + list(range(50, 200, 50)),
            'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=50,  # Increased from 30 to 50 for more thorough search
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1,
            scoring='accuracy'
        )
        grid_search.fit(X, y)

        logger.info(f"Best Random Forest params: {grid_search.best_params_}")
        logger.info(f"Best Random Forest score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_
    
    def tune_xgboost(self, X, y):
        """Use RandomizedSearchCV to find optimal XGBoost parameters"""
        # First split data for proper tuning
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        n_classes = len(np.unique(y))
        
        param_grid = {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 10),
            'gamma': uniform(0, 1),
            'reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10],
            'reg_lambda': [0.01, 0.1, 1, 10, 100],
            'scale_pos_weight': [1, 3, 5, 10],
            'max_delta_step': [0, 1, 5, 10],
            'tree_method': ['auto', 'exact', 'approx', 'hist']
        }
        
        # Define base model
        xgb_model = xgb.XGBClassifier(
            objective='multi:softproba',
            eval_metric=['mlogloss', 'merror'],
            num_class=n_classes,
            use_label_encoder=False,
            random_state=42,
            verbosity=0
        )
        
        # Advanced search with more iterations
        grid_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50,  # More iterations for thorough search
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        grid_search.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        logger.info(f"Best XGBoost params: {grid_search.best_params_}")
        logger.info(f"Best XGBoost score: {grid_search.best_score_:.4f}")
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Retrain with the best params on the full dataset with early stopping
        best_model.fit(
            X, y,
            verbose=False
        )
        
        return best_model
    
    def train_lightgbm(self, X, y):
        """Train a LightGBM model with hyperparameter tuning"""
        try:
            import lightgbm as lgb
            
            # Define parameter grid
            param_grid = {
                'n_estimators': randint(100, 500),
                'learning_rate': uniform(0.01, 0.3),
                'num_leaves': randint(20, 150),
                'max_depth': randint(3, 12),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'min_child_samples': randint(1, 50),
                'reg_alpha': uniform(0, 1),
                'reg_lambda': uniform(0, 1)
            }
            
            # Initialize the classifier
            lgb_clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
            
            # Perform random search
            search = RandomizedSearchCV(
                lgb_clf, param_grid, n_iter=20, cv=5, 
                scoring='accuracy', random_state=42, n_jobs=-1, verbose=1
            )
            
            search.fit(X, y)
            
            logger.info(f"Best LightGBM params: {search.best_params_}")
            logger.info(f"Best LightGBM score: {search.best_score_:.4f}")
            
            # Evaluate on train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            best_model = search.best_estimator_
            best_model.fit(X_train, y_train)
            
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"LightGBM test accuracy: {accuracy:.4f}")
            
            return best_model, accuracy
        except ImportError:
            logger.warning("LightGBM not installed. Skipping LightGBM model.")
            return None, 0.0
    
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
                verbose=0
            )
            
            # Perform random search
            search = RandomizedSearchCV(
                cat_clf, param_grid, n_iter=15, cv=5, 
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
            
            return best_model, accuracy
        except ImportError:
            logger.warning("CatBoost not installed. Skipping CatBoost model.")
            return None, 0.0
            
    def train_stacked_ensemble(self, X, y, base_models):
        """Train a stacked ensemble model with cross-validation"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import StackingClassifier
        
        # Define meta-learner
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        
        # Create the stacking ensemble
        stacked_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        # Evaluate on train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        stacked_model.fit(X_train, y_train)
        
        y_pred = stacked_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Stacked Ensemble test accuracy: {accuracy:.4f}")
        
        # Calculate more detailed metrics
        if hasattr(stacked_model, 'predict_proba'):
            try:
                # Get predicted probabilities
                y_proba = stacked_model.predict_proba(X_test)
                
                # For multi-class, calculate ROC AUC using One-vs-Rest approach
                if len(np.unique(y)) > 2:
                    # One-hot encode the target
                    y_test_bin = label_binarize(y_test, classes=np.unique(y))
                    n_classes = y_test_bin.shape[1]
                    
                    # Calculate ROC AUC for each class
                    roc_auc_scores = []
                    for i in range(n_classes):
                        if y_test_bin[:, i].sum() > 0:  # Only if class exists in test set
                            roc_auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
                            roc_auc_scores.append(roc_auc)
                    
                    # Average ROC AUC across all classes
                    avg_roc_auc = np.mean(roc_auc_scores)
                    logger.info(f"Average ROC AUC (One-vs-Rest): {avg_roc_auc:.4f}")
                else:
                    # Binary case
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    logger.info(f"ROC AUC: {roc_auc:.4f}")
            except Exception as e:
                logger.warning(f"Couldn't calculate ROC AUC: {str(e)}")
        
        # Calculate RMSE if applicable (for regression tasks or converting to numeric)
        try:
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            logger.info(f"RMSE: {rmse:.4f}")
        except Exception as e:
            logger.warning(f"Couldn't calculate RMSE: {str(e)}")
            
        # Calculate classification report
        try:
            report = classification_report(y_test, y_pred, output_dict=True)
            logger.info(f"Classification Report: {json.dumps(report, indent=2)}")
        except Exception as e:
            logger.warning(f"Couldn't generate classification report: {str(e)}")
            
        return stacked_model, accuracy, y_proba if hasattr(stacked_model, 'predict_proba') else None
        
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


class ModelManager:
    """Manages machine learning models for funding stage prediction with validation and audit"""

    def __init__(self, model_dir='models/'):
        """Initialize with model directory and setup audit logging"""
        self.model_dir = model_dir
        self.model = None
        self.metadata = {}
        self.scaler = None
        self.feature_names = []
        self.anomaly_detector = AnomalyDetector(contamination=0.05)
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

    def plot_funding_stage_distribution(self, data):
        """Visualize the distribution of funding stages"""
        plt.figure(figsize=(12, 6))

        # Map numeric stages back to names
        stage_map_rev = {
            0: 'Pre-Seed', 1: 'Seed', 2: 'Series A',
            3: 'Series B', 4: 'Series C', 5: 'Series D+',
            6: 'Series E+', 7: 'Series F+', 8: 'Private Equity',
            9: 'Venture', 10: 'Debt', 11: 'Grant'
        }

        # Count by stage
        stage_counts = data['funding_stage_numeric'].map(
            lambda x: stage_map_rev.get(x, 'Other')
        ).value_counts().sort_index()

        # Plot
        ax = stage_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribution of Funding Stages', fontsize=14)
        plt.xlabel('Funding Stage')
        plt.ylabel('Number of Companies')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add count labels on bars
        for i, v in enumerate(stage_counts):
            ax.text(i, v + 0.5, str(v), ha='center')

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

        # Extract accuracies
        models = list(model_results.keys())
        accuracies = [model_results[m]['accuracy'] for m in models]

        # Plot bar chart
        bars = plt.bar(models, accuracies, color=['dodgerblue', 'forestgreen'])
        plt.title('Model Accuracy Comparison', fontsize=14)
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
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

    def plot_calibration(self, y_true, y_proba, n_bins=10):
        plt.figure(figsize=(10, 6))
        if y_proba.ndim == 1:
            logger.warning("Converting 1D probability array to 2D")
            y_proba = np.column_stack([1 - y_proba, y_proba])
        n_classes = y_proba.shape[1]
        for class_idx in range(n_classes):
            try:
                binary_y = (y_true == class_idx).astype(int)
                class_proba = y_proba[:, class_idx]
                prob_true, prob_pred = calibration_curve(
                    binary_y, class_proba, n_bins=n_bins, strategy='quantile'
                )
                plt.plot(prob_pred, prob_true, marker='o',
                         label=f'Class {class_idx}', alpha=0.7)
            except Exception as e:
                logger.warning(
                    f"Skipping calibration for class {class_idx}: {
                        str(e)}")
                continue
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Observed Fraction')
        plt.title('Calibration Plot (One-vs-Rest)')
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
        self._init_model_directory()

    def run(self):
        try:
            logger.info("Starting funding stage prediction pipeline")

            # Steps 1-3: Existing data loading and feature engineering
            merged_data = self.data_loader.merge_datasets()
            if merged_data.empty:
                logger.error("No data available. Exiting pipeline.")
                return False

            processed_data = self.feature_engineer.extract_features(
                merged_data)
            X, y = self.feature_engineer.prepare_model_data(processed_data)

            # First remap all classes to be continuous from 0
            def remap_classes(y_series):
                unique_classes = sorted(y_series.unique())
                class_map = {
                    old_label: idx for idx,
                    old_label in enumerate(unique_classes)}
                return y_series.map(class_map), class_map

            y, initial_map = remap_classes(y)
            logger.info(f"Initial class mapping: {initial_map}")

            # Handle rare classes
            class_counts = pd.Series(y).value_counts()
            rare_classes = class_counts[class_counts < 5].index.tolist()
            if rare_classes:
                majority_class = class_counts.idxmax()
                y = y.apply(
                    lambda x: majority_class if x in rare_classes else x)
                logger.info(
                    f"Merged rare classes into majority class {majority_class}")

            # Remap again after merging rare classes to ensure continuous
            # labels
            y, final_map = remap_classes(y)
            logger.info(
                f"Final class mapping after merging rare classes: {final_map}")

            # No SMOTE or synthetic data generation

            # Feature selection with proper classes
            n_classes = len(np.unique(y))
            selector = SelectFromModel(
                xgb.XGBClassifier(
                    objective='multi:softmax',
                    num_class=n_classes,
                    random_state=42
                ),
                threshold="median"
            )
            X_selected = selector.fit_transform(X, y)
            logger.info(f"Selected features shape: {X_selected.shape}")

            # Keep track of selected features if X has column names
            if hasattr(X, 'columns'):
                selected_indices = selector.get_support()
                selected_features = [
                    feature for feature, selected in zip(
                        X.columns, selected_indices) if selected]
                logger.info(f"Selected features: {selected_features}")

            # Step 4: Enhanced model training
            logger.info("Step 4: Tuning and training models...")
            best_rf = self.model_trainer.tune_random_forest(X_selected, y)
            best_xgb = self.model_trainer.tune_xgboost(X_selected, y)
            
            # Also train gradient boosting model
            gb_model, gb_results = self.model_trainer.train_gradient_boosting(X_selected, y)
            
            # Create better ensemble with weights based on cross-validation performance
            ensemble = VotingClassifier(
                estimators=[
                    ('rf', best_rf), 
                    ('xgb', best_xgb),
                    ('gb', gb_model)
                ],
                voting='soft',
                weights=[1, 1.2, 1]  # Give slightly more weight to XGBoost if it performs better
            )

            # Train models
            rf_model, rf_results = self._train_final_model(
                best_rf, X_selected, y, 'Random Forest')
            xgb_model, xgb_results = self._train_final_model(
                best_xgb, X_selected, y, 'XGBoost')
            ensemble_model, ensemble_results = self._train_final_model(
                ensemble, X_selected, y, 'Ensemble')
            
            # Determine best weights based on validation performance
            cv_scores = {
                'rf': np.mean(rf_results['cross_val_scores']),
                'xgb': np.mean(xgb_results['cross_val_scores']),
                'gb': np.mean(gb_results['cross_val_scores'])
            }
            logger.info(f"Cross-validation scores: {cv_scores}")
            
            # Create optimized ensemble with dynamic weights
            best_weights = []
            for model_name in ['rf', 'xgb', 'gb']:
                weight = cv_scores[model_name] / sum(cv_scores.values()) * 3  # Normalize weights
                best_weights.append(round(weight, 2))
                
            logger.info(f"Using optimized ensemble weights: {best_weights}")
            
            optimized_ensemble = VotingClassifier(
                estimators=[
                    ('rf', rf_model), 
                    ('xgb', xgb_model),
                    ('gb', gb_model)
                ],
                voting='soft',
                weights=best_weights
            )
            
            # Train the optimized ensemble with the best weights
            opt_ensemble_model, opt_ensemble_results = self._train_final_model(
                optimized_ensemble, X_selected, y, 'Optimized Ensemble')

            # Step 5: Advanced visualizations
            logger.info("Step 5: Creating advanced visualizations...")
            self._create_advanced_visualizations(
                rf_model, xgb_model, gb_model, rf_results, xgb_results, gb_results, ensemble_results, opt_ensemble_results, y)

            # No dummy/fabricated data visualizations
            key_features = [col for col in [
                'funding_amount_log', 'employees',
                'employee_efficiency', 'previous_rounds',
                'months_since_first_funding', 'funding_year', 'funding_month'
            ] if col in processed_data.columns]

            self.visualizer.plot_temporal_trends(processed_data)
            self.visualizer.plot_funding_vs_employees(processed_data)

            # Save results with all models
            model_results = {
                'Random Forest': rf_results,
                'XGBoost': xgb_results,
                'Ensemble': ensemble_results
            }

            self._save_summary(merged_data, X, model_results)

            logger.info("Pipeline completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Enhanced pipeline error: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _train_final_model(self, model, X, y, name):
        """Helper method to train final models with detailed metrics including RMSE"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Standardize features for better model performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Ensure we have column names for the feature importance analysis
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Train the model with cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        logger.info(f"{name} cross-validation accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        
        # Fit the model on the training data
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # Calculate detailed metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate RMSE - treat classes as numeric values for regression-like evaluation
        # This helps measure how far off predictions are in terms of funding stages
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Calculate class-specific metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Print detailed metrics
        logger.info(f"--- {name} Model Performance ---")
        logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        logger.info(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")
        logger.info(f"RMSE: {rmse:.4f} (lower is better)")
        logger.info("-------------------------")
        
        # Also print to stdout for visibility
        print(f"--- {name} Model Performance ---")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")
        print(f"RMSE: {rmse:.4f} (lower is better)")
        print("-------------------------")

        return model, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'rmse': rmse,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'cross_val_scores': cv_scores,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'feature_names': feature_names,
            'scaler': scaler
        }

    def _create_advanced_visualizations(
            self,
            rf_model,
            xgb_model,
            gb_model,
            rf_results,
            xgb_results,
            gb_results,
            ensemble_results,
            opt_ensemble_results,
            y):
        """Generate advanced model diagnostics"""
        try:
            # Plot ROC curves for all models
            self.visualizer.plot_roc_curves(
                rf_results['y_test'],
                rf_results['y_proba'],
                classes=np.unique(y))
            
            self.visualizer.plot_roc_curves(
                xgb_results['y_test'],
                xgb_results['y_proba'],
                classes=np.unique(y))
                
            self.visualizer.plot_roc_curves(
                gb_results['y_test'],
                gb_results['y_proba'],
                classes=np.unique(y))
                
            # Plot calibration curves for all models
            self.visualizer.plot_calibration(
                rf_results['y_test'], 
                rf_results['y_proba'])
                
            self.visualizer.plot_calibration(
                xgb_results['y_test'], 
                xgb_results['y_proba'])
                
            self.visualizer.plot_calibration(
                gb_results['y_test'], 
                gb_results['y_proba'])
                
            # Plot confidence intervals
            self.visualizer.plot_confidence_intervals(
                rf_results['y_test'],
                rf_results['y_pred'],
                rf_results['y_proba'])
                
            # Plot feature importance for each model
            if hasattr(rf_model, 'feature_importances_'):
                self.visualizer.plot_feature_importance(
                    rf_model, rf_results['feature_names'])
                    
            if hasattr(xgb_model, 'feature_importances_'):
                self.visualizer.plot_feature_importance(
                    xgb_model, xgb_results['feature_names'])
                    
            if hasattr(gb_model, 'feature_importances_'):
                self.visualizer.plot_feature_importance(
                    gb_model, gb_results['feature_names'])
            
            # Compare model performances with RMSE included
            all_results = {
                'Random Forest': rf_results,
                'XGBoost': xgb_results,
                'Gradient Boosting': gb_results,
                'Ensemble': ensemble_results,
                'Optimized Ensemble': opt_ensemble_results
            }
            
            # Plot comparative metrics for all models
            self.visualizer.plot_model_comparison(all_results)
            self.visualizer.plot_confusion_matrices(all_results)
            
            # Plot RMSE comparison
            self._plot_rmse_comparison(all_results)
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            logger.error(traceback.format_exc())
            
    def _plot_rmse_comparison(self, model_results):
        """Create a bar chart comparing RMSE values across models"""
        try:
            plt.figure(figsize=(10, 6))
            
            models = list(model_results.keys())
            rmse_values = [model_results[m]['rmse'] for m in models]
            
            # Create bar chart with reversed color scale (since lower RMSE is better)
            bars = plt.bar(models, rmse_values, color=plt.cm.viridis(np.linspace(0.8, 0, len(models))))
            plt.title('Model RMSE Comparison (Lower is Better)', fontsize=14)
            plt.ylabel('RMSE')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualizer.output_dir, f"rmse_comparison_{self.timestamp}.png"))
            plt.close()
            
            logger.info("RMSE comparison visualization created successfully")
        except Exception as e:
            logger.error(f"Error creating RMSE comparison: {e}")
            logger.error(traceback.format_exc())

    def _save_summary(self, merged_data, X, model_results):
        """Save a summary of the pipeline run"""
        # Save metrics as JSON
        summary = {
            'timestamp': self.timestamp,
            'data_shape': {
                'records': len(merged_data),
                'features': X.shape[1]
            },
            'metrics': {}
        }

        # Add metrics for each model
        for model_name, results in model_results.items():
            summary['metrics'][model_name] = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'rmse': float(results['rmse']) if 'rmse' in results else None,
                'cross_val_accuracy': float(np.mean(results['cross_val_scores'])) if 'cross_val_scores' in results else None
            }

        # Include feature importance if available
        if 'Random Forest' in model_results and hasattr(model_results['Random Forest'], 'feature_importances_'):
            importance = model_results['Random Forest']['model'].feature_importances_
            feature_names = model_results['Random Forest']['feature_names']
            
            # Create feature importance dict
            feature_importance = {
                feature: float(imp) for feature, imp in zip(feature_names, importance)
            }
            
            # Sort by importance (descending)
            summary['feature_importance'] = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        
        # Save comparisons
        if len(model_results) > 1:
            best_model = max(
                summary['metrics'].items(),
                key=lambda x: x[1]['accuracy']
            )[0]
            summary['best_model'] = best_model
            
            # Also determine best model by RMSE (lower is better)
            if all('rmse' in m and m['rmse'] is not None for m in summary['metrics'].values()):
                best_by_rmse = min(
                    summary['metrics'].items(),
                    key=lambda x: x[1]['rmse']
                )[0]
                summary['best_model_by_rmse'] = best_by_rmse

        # Save as JSON
        summary_path = os.path.join(self.output_dir, f"summary_{self.timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
            
        logger.info(f"Pipeline summary saved to {summary_path}")
        
        # Print comparison table to console
        print("\n===== MODEL PERFORMANCE COMPARISON =====")
        print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'RMSE':<12}")
        print("-" * 75)
        
        for model_name, metrics in summary['metrics'].items():
            acc = metrics['accuracy'] * 100
            prec = metrics['precision'] * 100
            rec = metrics['recall'] * 100
            f1 = metrics['f1_score'] * 100
            rmse = metrics['rmse'] if metrics['rmse'] is not None else 'N/A'
            
            print(f"{model_name:<15} {acc:<10.2f} % {prec:<10.2f} % {rec:<10.2f} % {f1:<10.2f} % {rmse:<10.4f}")
            
        print("=" * 75)
            
        return summary

    def make_prediction(self, sample_data):
        """Make prediction with best available model"""
                    'accuracy_percent': f"{float(results['accuracy'])*100:.2f}%",
                    'precision_percent': f"{float(results['precision'])*100:.2f}%",
                    'recall_percent': f"{float(results['recall'])*100:.2f}%",
                    'f1_score_percent': f"{float(results['f1_score'])*100:.2f}%",
                    'model_path': results.get('model_path', 'Not saved')
                }
                for name, results in model_results.items()
            }
        }

        # Generate a comparison table of model metrics
        print("\n===== MODEL PERFORMANCE COMPARISON =====")
        print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print("-" * 65)
        for name, results in model_results.items():
            acc = float(results['accuracy']) * 100
            prec = float(results['precision']) * 100
            rec = float(results['recall']) * 100
            f1 = float(results['f1_score']) * 100
            print(f"{name:<15} {acc:<12.2f}% {prec:<12.2f}% {rec:<12.2f}% {f1:<12.2f}%")
        print("=" * 65)

        with open(os.path.join(self.output_dir, f"summary_{self.timestamp}.json"), 'w') as f:
            json.dump(summary, f, indent=4)

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
                        f"Company data changed by >{
                            np.max(pct_change) * 100:.1f}%")

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
                    f"Range violations in {
                        len(range_violations)} features")
                anomalies['score'] = max(anomalies['score'], 0.7)

            if iqr_violations:
                anomalies['is_anomaly'] = True
                anomalies['reasons'].append(
                    f"IQR violations in {
                        len(iqr_violations)} features")
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
                'reason': f'detection_error: {
                    str(e)}',
                'score': 1.0}



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


if __name__ == "__main__":
    main()


