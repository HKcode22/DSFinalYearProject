# import tools for evaluating machine learning models
from sklearn.metrics import (
    accuracy_score,  # how many predictions were correct
    classification_report,  # summary of precision, recall, f1, etc.
    confusion_matrix,  # table showing correct and incorrect predictions
    roc_auc_score,  # measures how well the model separates classes
    roc_curve,  # data for plotting true vs false positive rates
    auc,  # area under the curve, a summary score
    precision_score,  # how many selected items are relevant
    recall_score,  # how many relevant items are selected
    f1_score,  # balance between precision and recall
    mean_squared_error)  # average squared difference between predictions and actual values

from sklearn.base import clone  # lets us copy models easily

# import tools for splitting data and searching for best model settings
from sklearn.model_selection import (
    train_test_split,  # splits data into train and test sets
    GridSearchCV,  # tries all parameter combinations
    RandomizedSearchCV,  # tries random parameter combinations
    cross_val_score)  # checks model performance on different splits

# import standard python libraries for data and file handling
import pickle  # for saving/loading python objects
import csv  # for reading/writing csv files
import random  # for random numbers
import uuid  # for unique ids
import re  # for text pattern matching
import shutil  # for copying files
import glob  # for finding files by pattern
import traceback  # for detailed error messages
import sqlite3  # for working with sqlite databases
import joblib  # for saving/loading large objects

# import machine learning libraries
import xgboost as xgb  # advanced boosting model
from sklearn.svm import OneClassSVM  # for detecting outliers
from scipy.stats import randint, uniform  # for random numbers in parameter search
from sklearn.preprocessing import label_binarize, StandardScaler, MinMaxScaler  # for preparing data
from sklearn.feature_selection import SelectFromModel  # for picking important features
from sklearn.calibration import calibration_curve, CalibratedClassifierCV  # for adjusting model probabilities
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest, GradientBoostingClassifier  # ensemble models
from sklearn.linear_model import LogisticRegression  # simple linear model

# try to import lightgbm, a fast machine learning library; if not installed, set to None and warn
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
    logger.warning("lightgbm not installed. skipping lightgbm model. install with: pip install lightgbm")

# import libraries for plotting and data manipulation
import seaborn as sns  # for pretty plots
import matplotlib.pyplot as plt  # for plotting
import os  # for file and directory operations
import json  # for working with json files
import pandas as pd  # for data tables
import numpy as np  # for math and arrays
from datetime import datetime  # for dates and times
import logging  # for logging messages
import matplotlib  # for controlling matplotlib settings

matplotlib.use('Agg')  # set matplotlib to non-interactive mode (for saving plots to files)

# try to import prophet for time series forecasting; if not installed, set to None and warn
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
    logger.warning("prophet library not found. time series forecasting features will be disabled. install with: pip install prophet")

# set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("funding_prediction.log"),
        logging.StreamHandler()])
logger = logging.getLogger(__name__)

# this class loads, saves, and manages all the data files and database for the project
class DataLoader:
    def __init__(self, base_dir="./", archive=False, output_dir_for_db="./MainOutput"):
        # set up the main directory for data files
        self.base_dir = os.path.abspath(base_dir)
        logger.info(f"dataloader using absolute base_dir for json files: {self.base_dir}")

        # set up archiving and database paths
        self.archive = archive
        self.archive_dir = None
        os.makedirs(output_dir_for_db, exist_ok=True)
        self.historical_db = os.path.join(output_dir_for_db, "historical_funding_data.db")

        # set up paths for the different json data sources
        self.json_folder = self.base_dir
        self.fundraiser_path = os.path.join(self.json_folder, "fundraisestartup50.json")
        self.growthlist_path = os.path.join(self.json_folder, "growthlistscrapper.json")
        self.topstartup_path = os.path.join(self.json_folder, "topstartupio50.json")
        
        # log the paths for debugging
        logger.info(f"fundraiser path: {self.fundraiser_path}")
        logger.info(f"growthlist path: {self.growthlist_path}")
        logger.info(f"topstartup path: {self.topstartup_path}")

        # initialize the database and archive if needed
        self._init_historical_db()
        if self.archive:
            self.archive_dir = self._create_archive_dir()
            self._archive_current_data()

    def _create_archive_dir(self):
        
        archive_root = os.path.join(self.base_dir, "data_archive")
        os.makedirs(archive_root, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(archive_root, timestamp)
        os.makedirs(archive_dir, exist_ok=True)
        return archive_dir

    def _archive_current_data(self):
        
        try:
            if not self.archive_dir:
                return
            
            if os.path.isfile(self.fundraiser_path):
                shutil.copy2(self.fundraiser_path, os.path.join(self.archive_dir, "fundraiser.json"))
            if os.path.isfile(self.growthlist_path):
                shutil.copy2(self.growthlist_path, os.path.join(self.archive_dir, "growthlist.json"))
            if os.path.isfile(self.topstartup_path):
                shutil.copy2(self.topstartup_path, os.path.join(self.archive_dir, "topstartup.json"))
            logger.info(f"archived data files to {self.archive_dir}")
        except Exception as e:
            logger.error(f"error archiving data: {e}")

    def _init_historical_db(self):
        
        conn = sqlite3.connect(self.historical_db)
        cursor = conn.cursor()

        
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
        logger.info("historical database initialized")

    def reset_database(self):
        
        try:
            if os.path.exists(self.historical_db):
                os.remove(self.historical_db)
                logger.info("existing database removed")
            self._init_historical_db()
            logger.info("database reset complete")
        except Exception as e:
            logger.error(f"error resetting database: {e}")

    def _limit_dataframe_size(self, df, max_rows=10000, stratify_column=None):
        
        if len(df) > max_rows:
            logger.info(f"dataset has {len(df)} rows, exceeding max_rows={max_rows}. attempting to sample down.")
            if stratify_column and stratify_column in df.columns and df[stratify_column].nunique() > 1:
                
                df_copy = df.copy()
                if df_copy[stratify_column].isnull().any():
                    logger.warning(f"stratification column '{stratify_column}' contains nans. filling with 'unknown' for sampling.")
                    df_copy[stratify_column] = df_copy[stratify_column].fillna('Unknown_Stratify')

                
                class_counts = df_copy[stratify_column].value_counts()
                if (class_counts >= 2).all():
                    try:
                        
                        sampled_df = df_copy.groupby(stratify_column, group_keys=False).apply(
                            lambda x: x.sample(min(len(x), int(max_rows * len(x) / len(df_copy)) + 1), random_state=42))
                        if len(sampled_df) > max_rows:
                            sampled_df = sampled_df.sample(n=max_rows, random_state=42)
                        logger.info(f"stratified sampling on '{stratify_column}' resulted in {len(sampled_df)} rows.")
                        return sampled_df.reset_index(drop=True)
                    except ValueError as e:
                        logger.warning(f"stratified sampling failed ({e}). falling back to random sampling.")
                        sampled_df = df.sample(n=max_rows, random_state=42)
                        logger.info(f"random sampling resulted in {len(sampled_df)} rows.")
                        return sampled_df.reset_index(drop=True)
                else:
                    logger.warning(f"not all classes in '{stratify_column}' have at least 2 members. falling back to random sampling.")
                    sampled_df = df.sample(n=max_rows, random_state=42)
                    logger.info(f"random sampling resulted in {len(sampled_df)} rows.")
                    return sampled_df.reset_index(drop=True)
            else:
                if stratify_column:
                    logger.warning(f"stratification column '{stratify_column}' not suitable or not found. performing random sampling.")
                sampled_df = df.sample(n=max_rows, random_state=42)
                logger.info(f"random sampling resulted in {len(sampled_df)} rows.")
                return sampled_df.reset_index(drop=True)
        return df

    def validate_dataset(self, df, source_name):
        
        required_columns = ['company_name', 'funding_date', 'funding_amount']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"missing columns in {source_name}: {missing_cols}")
            return False
        return True

    def validate_merged_columns(self, df):
        
        required_columns = [
            'company_name', 'funding_date', 'funding_amount',
            'funding_stage', 'industry', 'employees'
        ]

        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"missing critical columns: {missing}")
            return False

        
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            logger.error(f"duplicate columns detected: {duplicates}")
            return False

        return True

    def validate_company(self, company_name, funding_amount=None, 
                        funding_stage=None, employees=None):
        
        
        is_valid = True
        confidence_score = 1.0
        messages = []
        
        
        if pd.isna(company_name) or not company_name or len(str(company_name).strip()) < 2:
            is_valid = False
            confidence_score = 0.0
            return is_valid, confidence_score, "invalid company name"
        
        
        valid_funding_stages = [
            'pre-seed', 'seed', 'angel', 'series a', 'series b', 'series c', 
            'series d', 'series e', 'series f', 'series g', 'series h', 
            'venture - series unknown', 'private equity', 'grant', 
            'debt financing', 'undisclosed', 'post-ipo'
        ]
        
        
        if funding_stage is not None and pd.notna(funding_stage):
            normalized_stage = str(funding_stage).lower().strip()
            if normalized_stage not in valid_funding_stages:
                
                stage_found = False
                for valid_stage in valid_funding_stages:
                    if valid_stage in normalized_stage or normalized_stage in valid_stage:
                        stage_found = True
                        break
                
                if not stage_found:
                    messages.append(f"unknown funding stage: {funding_stage}")
                    confidence_score *= 0.7
        else:
            messages.append("missing funding stage")
            confidence_score *= 0.8
        
        
        if funding_amount is not None:
            if pd.isna(funding_amount):
                messages.append("Missing funding amount")
                confidence_score *= 0.9
            elif funding_amount <= 0:
                messages.append("Invalid negative or zero funding amount")
                confidence_score *= 0.6
            elif funding_amount > 10e9:  
                messages.append(f"Suspicious high funding amount: ${funding_amount:,.2f}")
                confidence_score *= 0.5
        
        
        if employees is not None and pd.notna(employees):
            try:
                emp_count = float(employees)
                if emp_count <= 0:
                    messages.append("Invalid negative or zero employee count")
                    confidence_score *= 0.8
                elif emp_count > 1000000:  
                    messages.append(f"Suspicious high employee count: {emp_count:,.0f}")
                    confidence_score *= 0.7
            except (ValueError, TypeError):
                messages.append(f"Invalid employee data format: {employees}")
                confidence_score *= 0.7
        
        
        if (funding_stage is not None and funding_amount is not None and 
            pd.notna(funding_stage) and pd.notna(funding_amount)):
            
            normalized_stage = str(funding_stage).lower().strip()
            
            
            if ('seed' in normalized_stage or 'pre-seed' in normalized_stage) and funding_amount > 50e6:
                messages.append(f"Unusual high funding for {funding_stage}: ${funding_amount:,.2f}")
                confidence_score *= 0.6
            elif 'series a' in normalized_stage and funding_amount > 500e6:
                messages.append(f"Unusual high funding for {funding_stage}: ${funding_amount:,.2f}")
                confidence_score *= 0.7
        
        
        if confidence_score < 0.3:
            is_valid = False
        
        
        if messages:
            final_message = "; ".join(messages)
        else:
            final_message = "Valid company data"
            
        return is_valid, confidence_score, final_message

    def load_fundraiser_data(self):
        """Load and process fundraiser insider data"""
        try:
            if not os.path.exists(self.fundraiser_path):
                logger.error(f"Fundraiser data file not found: {self.fundraiser_path}")
                return pd.DataFrame()
            with open(self.fundraiser_path, 'r') as file:
                data = json.load(file)

            
            companies = data.get('companies', [])
            df = pd.DataFrame(companies)

            
            df['data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            
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
            if not os.path.exists(self.growthlist_path):
                logger.error(f"Growthlist data file not found: {self.growthlist_path}")
                return pd.DataFrame()
            with open(self.growthlist_path, 'r') as file:
                data = json.load(file)

            
            df = pd.DataFrame(data)

            
            df['data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            
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
            if not os.path.exists(self.topstartup_path):
                logger.error(f"Topstartup data file not found: {self.topstartup_path}")
                return pd.DataFrame()
            with open(self.topstartup_path, 'r') as file:
                data = json.load(file)

            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame(data.get('startups', []))
            else:
                logger.error("Unexpected JSON format in topstartup data")
                return pd.DataFrame()

            
            if 'funding' in df.columns:
                
                def extract_funding_info(funding_str):
                    if not funding_str or pd.isna(funding_str):
                        return None, None, None

                    
                    
                    

                    amount = None
                    stage = None
                    date = None

                    
                    amount_match = re.search(
                        r'\$(\d+(?:\.\d+)?[KMB]?)', funding_str)
                    if amount_match:
                        amount = amount_match.group(0)  

                    
                    
                    
                    stage_pattern = r'(Pre[-\s]?Seed|Seed|Angel|Series\s+[A-Z]|Venture[\s\-]+Series\s+Unknown|Initial\s+Coin\s+Offering|ICO|Private\s+Equity|Grant|Debt\s+Financing|Undisclosed|Post[-\s]?IPO)'
                    stage_match = re.search(
                        stage_pattern, funding_str, re.IGNORECASE)

                    if stage_match:
                        
                        raw_stage = stage_match.group(1)

                        
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
                            stage = raw_stage  
                    else:
                        
                        
                        funding_lower = funding_str.lower()

                        if 'seed' in funding_lower and not stage:
                            stage = 'Seed'
                        elif 'angel' in funding_lower and not stage:
                            stage = 'Angel'
                        elif 'raised' in funding_lower and not stage:
                            
                            
                            if 'series' in funding_lower:
                                
                                series_match = re.search(
                                    r'series\s+([a-z])', funding_lower)
                                if series_match:
                                    stage = f'Series {
                                        series_match.group(1).upper()}'
                                else:
                                    stage = 'venture - series unknown'
                            else:
                                
                                
                                stage = 'venture - series unknown'
                        elif 'valuation' in funding_lower and not stage:
                            if 'post-ipo' in funding_lower or 'post ipo' in funding_lower:
                                stage = 'Post-IPO'
                            else:
                                
                                
                                stage = 'venture - series unknown'

                    
                    date_match = re.search(r'in (\d{4})', funding_str)
                    if date_match:
                        date = date_match.group(1)
                    else:
                        
                        year_match = re.search(r'\b(20\d{2})\b', funding_str)
                        if year_match:
                            date = year_match.group(1)

                    return amount, stage, date

                
                funding_details = df['funding'].apply(extract_funding_info)

                
                df['funding_amount'] = funding_details.apply(
                    lambda x: x[0] if x else None)
                df['funding_stage'] = funding_details.apply(
                    lambda x: x[1] if x else None)
                df['funding_date'] = funding_details.apply(
                    lambda x: x[2] if x else None)

            
            column_mapping = {
                'name': 'company_name',
                'funding_type': 'funding_stage',  
                'category': 'industry'
            }

            
            df = df.rename(columns={k: v for k, v in column_mapping.items()
                                    if k in df.columns})

            
            df['data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            logger.info(f"Loaded {len(df)} records from topstartup data")
            return df

        except Exception as e:
            logger.error(f"Error loading topstartup data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _parse_funding_amount(self, amount_str):
        
        if not amount_str or pd.isna(amount_str) or amount_str == "":
            return np.nan

        try:
            
            amount_str = str(amount_str).replace(
                '$', '').replace(',', '').strip()

            
            max_reasonable_amount = 1e10  

            
            if 'B' in amount_str:
                value = float(amount_str.replace('B', '')) * 1e9
            elif 'M' in amount_str:
                value = float(amount_str.replace('M', '')) * 1e6
            elif 'K' in amount_str:
                value = float(amount_str.replace('K', '')) * 1e3
            else:
                value = float(amount_str)

            
            if value > max_reasonable_amount:
                logger.warning(
                    f"unreasonably large funding amount detected: ${value:,.2f}")
                return np.nan
            elif value < 0:
                logger.warning(
                    f"negative funding amount detected: ${value:,.2f}")
                return np.nan

            return value
        except Exception as e:
            logger.warning(
                f"error parsing funding amount '{amount_str}': {str(e)}")
            return np.nan

    def save_historical_data(self, df, table_name):
        
        try:
            conn = sqlite3.connect(self.historical_db)
            df.to_sql(table_name, conn, if_exists='append', index=False)
            conn.close()
            logger.info(f"saved {len(df)} records to historical {table_name}")
        except Exception as e:
            logger.error(f"error saving historical data: {e}")

    def load_historical_data(self, table_name):
        
        try:
            conn = sqlite3.connect(self.historical_db)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(
                f"loaded {len(df)} historical records from {table_name}")
            return df
        except Exception as e:
            logger.error(f"error loading historical data: {e}")
            return pd.DataFrame()

    def merge_datasets(self):
        
        try:
            
            fundraiser_df = self.load_fundraiser_data()
            growthlist_df = self.load_growthlist_data()
            topstartup_df = self.load_topstartup_data()

            
            logger.info(
                f"loaded datasets - fundraiser: {len(fundraiser_df)} rows, growthlist: {len(growthlist_df)} rows, topstartup: {len(topstartup_df)} rows")

            
            all_records = []
            audit_log = []
            rejected_records = []

            
            if not fundraiser_df.empty:
                for _, row in fundraiser_df.iterrows():
                    if pd.notna(row.get('Company')):  
                        
                        try:
                            funding_amount = pd.to_numeric(
                                row.get('Funding_Amount_USD'), errors='coerce')
                            employees = pd.to_numeric(
                                row.get('Total_Employees'), errors='coerce')
                        except BaseException:
                            funding_amount = np.nan
                            employees = np.nan

                        
                        is_valid, confidence, message = self.validate_company(
                            row.get('Company'),
                            funding_amount=funding_amount,
                            funding_stage=row.get('Funding_Type'),
                            employees=employees
                        )

                        
                        audit_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'company': row.get('Company'),
                            'source': 'fundraiser',
                            'validation_result': is_valid,
                            'confidence_score': confidence,
                            'message': message
                        }
                        audit_log.append(audit_entry)

                        if is_valid and confidence >= 0.3:  
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
                    f"processed {len(fundraiser_df)} records from fundraiser data, rejected {len(rejected_records)} suspicious records")

            
            if not growthlist_df.empty:
                growthlist_rejected = len(rejected_records)
                for _, row in growthlist_df.iterrows():
                    if pd.notna(row.get('name')):  
                        
                        funding_amount = row.get('funding_amount_numeric')
                        if pd.isna(funding_amount) and pd.notna(
                                row.get('funding_amount')):
                            funding_amount = self._parse_funding_amount(
                                row.get('funding_amount'))

                        
                        is_valid, confidence, message = self.validate_company(
                            row.get('name'),
                            funding_amount=funding_amount,
                            funding_stage=row.get('funding_type')
                        )

                        
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

            
            if not topstartup_df.empty:
                topstartup_rejected = len(rejected_records)
                for _, row in topstartup_df.iterrows():
                    company_name = row.get('company_name') or row.get('name')

                    if pd.notna(company_name):  
                        
                        funding_stage = row.get('funding_stage') or row.get('funding_round')

                        
                        funding_amount = row.get('funding_amount')
                        if isinstance(funding_amount, str):
                            funding_amount = self._parse_funding_amount(funding_amount)

                        
                        employee_count = None
                        if pd.notna(row.get('employees')):
                            
                            emp_str = str(row.get('employees'))
                            match = re.search(r'(\d+)-(\d+)', emp_str)
                            if match:
                                
                                employee_count = (
                                    int(match.group(1)) + int(match.group(2))) / 2

                        
                        is_valid, confidence, message = self.validate_company(
                            company_name,
                            funding_amount=funding_amount,
                            funding_stage=funding_stage,
                            employees=employee_count
                        )

                        
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
                
                merged_data = pd.DataFrame(all_records)

                
                
                
                merged_data['source'] = merged_data['source'].fillna('Unknown_Source')
                if merged_data['source'].nunique() > 1:
                    merged_data = self._limit_dataframe_size(merged_data, max_rows=10000, stratify_column='source')
                else:
                    
                    merged_data['industry'] = merged_data['industry'].fillna('Unknown_Industry')
                    if merged_data['industry'].nunique() > 1:
                         merged_data = self._limit_dataframe_size(merged_data, max_rows=10000, stratify_column='industry')
                    else:
                         merged_data = self._limit_dataframe_size(merged_data, max_rows=10000) 
                

                
                logger.info(
                    f"Merged data columns: {
                        merged_data.columns.tolist()}")
                logger.info(
                    f"Non-null counts: {merged_data.count().to_dict()}")

                
                pre_dedup_count = len(merged_data)
                merged_data = merged_data.drop_duplicates(
                    subset=['company_name', 'funding_date']
                ).reset_index(drop=True)
                logger.info(
                    f"Removed {
                        pre_dedup_count -
                        len(merged_data)} duplicate records")

                
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

                
                merged_data = merged_data.fillna({
                    'industry': 'Unknown',
                    'employees': 0,
                    'funding_stage': 'Unknown',
                    'confidence_score': 0.5  
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
        
        self.funding_stage_map = {}  
        
        self.standard_stages = [
            'Pre-Seed', 'Seed', 'Angel',
            'Series A', 'Series B', 'Series C',
            'Series D', 'Series E', 'Series F',
            'Series G', 'Series H'
        ]
        
        self.age_bin_edges = None
        self.age_bin_labels = None

    def extract_features(self, df):
        
        logger.info(f"starting feature extraction for {len(df)} records")
        data = df.copy()

        
        valid_stages = data['funding_stage'].dropna().unique()
        logger.info(f"found {len(valid_stages)} unique funding stages: {valid_stages}")

        
        known_stages = {}
        for i, stage in enumerate(self.standard_stages):
            known_stages[stage] = i

        
        self.funding_stage_map = known_stages.copy()

        
        next_value = max(known_stages.values()) + 1 if known_stages else 0
        for stage in valid_stages:
            if stage not in self.funding_stage_map:
                self.funding_stage_map[stage] = next_value
                next_value += 1

        
        if 'Unknown' not in self.funding_stage_map:
            self.funding_stage_map['Unknown'] = next_value

        logger.info(f"created funding stage mapping: {self.funding_stage_map}")

        
        data['funding_stage_numeric'] = data['funding_stage'].map(
            lambda x: self.funding_stage_map.get(x, self.funding_stage_map['Unknown'])
        )

        
        try:
            
            data['funding_date'] = pd.to_datetime(
                data['funding_date'], format='mixed', errors='coerce')

            
            if data['funding_date'].isna().sum() > len(data) * 0.3:
                logger.warning(
                    f"Many dates failed to parse ({
                        data['funding_date'].isna().sum()} NaN values)")

                
                for fmt in ['%d-%b-%y', '%b %Y', '%Y', '%m/%d/%Y', '%Y-%m-%d']:
                    try:
                        
                        na_before = data['funding_date'].isna().sum()

                        
                        mask = data['funding_date'].isna()
                        data.loc[mask, 'funding_date'] = pd.to_datetime(
                            df.loc[mask, 'funding_date'],
                            format=fmt,
                            errors='coerce'
                        )

                        
                        na_after = data['funding_date'].isna().sum()
                        if na_before > na_after:
                            logger.info(
                                f"Format '{fmt}' parsed {
                                    na_before - na_after} dates")
                    except BaseException:
                        pass

            
            data['funding_year'] = data['funding_date'].dt.year
            data['funding_month'] = data['funding_date'].dt.month

            
            current_year = datetime.now().year
            current_month = datetime.now().month

            data['funding_year'] = data['funding_year'].fillna(current_year)
            data['funding_month'] = data['funding_month'].fillna(current_month)

            
            data['month_sin'] = np.sin(2 * np.pi * data['funding_month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['funding_month'] / 12)
            

            
            company_first_funding = data.groupby(
                'company_name')['funding_date'].min()
            data['company_first_funding'] = data['company_name'].map(
                company_first_funding)

            
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
            
            data['funding_year'] = datetime.now().year
            data['funding_month'] = datetime.now().month
            data['months_since_first_funding'] = 0

        
        data['industry_category'] = data['industry'].fillna('Unknown') 

        
        try:
            data = data.sort_values(by=['company_name', 'funding_date'])

            
            data['time_since_last_funding'] = data.groupby('company_name')['funding_date'].diff().dt.days / 30.44 
            data['time_since_last_funding'] = data['time_since_last_funding'].fillna(0) 

            
            data['prev_funding_amount'] = data.groupby('company_name')['funding_amount'].shift(1)
            data['funding_amount_ratio_vs_prev'] = data.apply(
                 lambda row: (row['funding_amount'] / row['prev_funding_amount'])
                             if pd.notna(row['funding_amount']) and pd.notna(row['prev_funding_amount']) and row['prev_funding_amount'] > 0
                             else 1.0,
                 axis=1
            )

            
            
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
            
            data['time_since_last_funding'] = 0
            data['funding_amount_ratio_vs_prev'] = 1.0
            data['funding_vs_industry_median'] = 1.0
        

        
        
        data['funding_amount'] = pd.to_numeric(data['funding_amount'], errors='coerce')

        
        funding_cap = 2e9
        data['funding_amount'] = data['funding_amount'].clip(upper=funding_cap)
        logger.info(f"Capped funding_amount at ${funding_cap:,.0f}")
        

        
        
        if (data['funding_amount'] > 1e11).any(): 
            large_values = data[data['funding_amount'] > 1e11]
            logger.warning(
                f"Found {len(large_values)} extremely large funding amounts (>$100B)")
            logger.warning(
                f"Sample: {large_values[['company_name', 'funding_amount', 'source']].head().to_dict()}")

        
        data['funding_amount_log'] = np.log1p(data['funding_amount'].fillna(0))

        
        if 'employees' in data.columns:
            data['employees'] = pd.to_numeric(
                data['employees'], errors='coerce')

            
            data['employee_efficiency'] = data.apply(
                lambda row: row['funding_amount'] /
                row['employees'] if row['employees'] > 0 else np.nan,
                axis=1)

            
            efficiency_medians = data.groupby('funding_stage')[
                'employee_efficiency'].median()

            for stage in data['funding_stage'].unique():
                stage_median = efficiency_medians.get(
                    stage, data['employee_efficiency'].median())
                mask = (
                    data['funding_stage'] == stage) & (
                    data['employee_efficiency'].isna())
                data.loc[mask, 'employee_efficiency'] = stage_median

            
            data['employee_efficiency'] = data['employee_efficiency'].fillna(
                data['employee_efficiency'].median())
        else:
            data['employees'] = np.nan
            data['employee_efficiency'] = np.nan

        
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

            
            return industry_str.title()

        
        data['industry_category'] = data['industry_category'].apply(map_industry)

        
        min_frequency = 10
        industry_counts = data['industry_category'].value_counts()
        rare_industries = industry_counts[industry_counts < min_frequency].index.tolist()
        
        
        if 'Other' in data['industry_category'].unique() and 'Other' in rare_industries:
            if industry_counts.get('Other', 0) >= min_frequency:
                rare_industries.remove('Other')
                
        
        if rare_industries:
            logger.info(f"mapping {len(rare_industries)} rare industries (count < {min_frequency}) to 'Other': {rare_industries[:10]}...")
            data['industry_category'] = data['industry_category'].replace(rare_industries, 'Other')

        
        if 'location' in data.columns or 'headquarters' in data.columns:
            
            location_col = 'location' if 'location' in data.columns else 'headquarters'

            data['location_category'] = data[location_col].fillna('Unknown')

            
            def extract_location(loc_str):
                if not isinstance(loc_str, str) or pd.isna(loc_str):
                    return 'Unknown'

                
                parts = [p.strip() for p in loc_str.split(',')]

                if len(parts) > 1:
                    return parts[-1]  
                return loc_str

            data['location_category'] = data['location_category'].apply(
                extract_location)

            
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

        
        company_funding_counts = data.groupby('company_name').size()
        data['previous_rounds'] = data['company_name'].map(
            company_funding_counts) - 1
        data['previous_rounds'] = data['previous_rounds'].clip(lower=0)

        
        def calc_funding_velocity(dates):
            
            if not dates or len(dates) < 2:
                return 0

            
            sorted_dates = sorted(dates)
            
            
            time_diffs = []
            for i in range(1, len(sorted_dates)):
                diff = (sorted_dates[i] - sorted_dates[i-1]).days
                if diff > 0:  
                    time_diffs.append(diff)

            
            return np.mean(time_diffs) if time_diffs else 0

        
        company_funding_dates = data.groupby('company_name')['funding_date'].apply(list)
        data['funding_velocity'] = data['company_name'].map(
            lambda x: calc_funding_velocity(company_funding_dates.get(x, []))
        )

        
        median_velocity = data['funding_velocity'].median()
        data['funding_velocity'] = data['funding_velocity'].fillna(median_velocity)

        
        
        if 'funding_amount_log' in data.columns and 'months_since_first_funding' in data.columns:
             data['funding_amount_x_age'] = data['funding_amount_log'] * data['months_since_first_funding']
        else:
             data['funding_amount_x_age'] = 0 

        if 'employees' in data.columns and 'previous_rounds' in data.columns:
             data['employees_x_rounds'] = data['employees'] * data['previous_rounds']
        else:
              data['employees_x_rounds'] = 0 

        
        if 'funding_velocity' in data.columns and 'previous_rounds' in data.columns:
             data['velocity_x_rounds'] = data['funding_velocity'] * data['previous_rounds']
        else:
             data['velocity_x_rounds'] = 0 

        if 'months_since_first_funding' in data.columns and 'employees' in data.columns:
             data['age_x_employees'] = data['months_since_first_funding'] * data['employees']
        else:
             data['age_x_employees'] = 0 
        
        

        
        age_feature = 'months_since_first_funding'
        if age_feature in data.columns:
            try:
                
                age_bins = [-np.inf, 12, 24, 48, 72, np.inf] 
                age_labels = ['0-12m', '13-24m', '25-48m', '49-72m', '73m+']
                data['company_age_bin'] = pd.cut(
                    data[age_feature],
                    bins=age_bins,
                    labels=age_labels,
                    right=True 
                ).astype(str) 

                
                self.age_bin_edges = age_bins
                self.age_bin_labels = age_labels
                logger.info(f"Created 'company_age_bin' feature with fixed bins: {age_labels}")

                
                data['company_age_bin'] = data['company_age_bin'].fillna('Unknown_Age').astype(str) 

                
                
            except Exception as bin_err:
                 logger.warning(f"Could not create 'company_age_bin': {bin_err}")
                 data['company_age_bin'] = 'Unknown_Age' 
                 
                 self.age_bin_edges = None
                 self.age_bin_labels = None
        else:
            data['company_age_bin'] = 'Unknown_Age' 
            
            self.age_bin_edges = None
            self.age_bin_labels = None
        

        
        emp_feature = 'employees'
        if emp_feature in data.columns:
            try:
                emp_bins = [-np.inf, 10, 50, 200, 1000, np.inf] 
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
            'age_x_employees'    
        ]

        for col in numeric_cols:
            if col in data.columns:
                
                if 'funding_stage' in data.columns:
                    data[col] = data.groupby('funding_stage')[col].transform(
                        lambda x: x.fillna(x.median())
                    )

                
                data[col] = data[col].fillna(data[col].median())

        logger.info(
            f"Feature engineering complete: {
                data.shape[1]} features created")
        return data

    def prepare_model_data(self, data):
        
        logger.info("preparing data for model training")

        
        feature_columns = [
            'funding_amount_log',  
            'employees',  
            'employee_efficiency',  
            'funding_velocity',  
            'time_since_last_funding',  
            'funding_amount_ratio_vs_prev',  
            'funding_vs_industry_median',  
            'months_since_first_funding',  
            'month_sin',  
            'month_cos'  
        ]

        
        X = data[feature_columns].copy()

        
        for col in X.columns:
            if X[col].isna().any():
                
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median())
                else:
                    
                    X[col] = X[col].fillna(X[col].mode()[0])

        
        y = data['funding_stage_numeric']

        
        logger.info(f"prepared data with {len(X)} samples and {len(feature_columns)} features")
        logger.info(f"feature columns: {feature_columns}")
        logger.info(f"target distribution: {y.value_counts().to_dict()}")

        return X, y, feature_columns


class ModelTrainer:
    def __init__(self, output_dir="./models"):
        
        self.output_dir = output_dir
        
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        
        logger.info(f"initialized model trainer with output directory: {output_dir}")
        
        
        np.random.seed(42)
        random.seed(42)

    def train_random_forest(self, X, y):
        """
        train a random forest classifier with optimized parameters and anomaly detection
        
        args:
            x: features matrix for training
            y: target vector for training
            
        returns:
            dictionary containing model, predictions, metrics, and anomaly detector
        """
        try:
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            
            isolation_forest = IsolationForest(
                contamination=0.01,  
                random_state=np.random.randint(
                    0, 10000),  
                n_jobs=-1  
            )

            
            rf = RandomForestClassifier(
                n_estimators=1000,  
                max_depth=20,  
                min_samples_split=5,  
                min_samples_leaf=2,  
                max_features='sqrt',  
                bootstrap=True,  
                oob_score=True,  
                random_state=np.random.randint(
                    0, 10000),  
                n_jobs=-1  
            )

            
            rf.fit(X_train, y_train)

            
            if hasattr(rf, 'oob_score_'):
                logger.info(f"Out-of-bag score: {rf.oob_score_:.4f}")

            
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            
            logger.info(f"Random Forest accuracy: {accuracy:.4f}")
            logger.info(f"Classification report:\n{report}")

            
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

            
            return {
                'status': 'success',  
                'model': rf,  
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
        train an xgboost classifier with enhanced hyperparameters and feature scaling
        
        args:
            x: features matrix for training
            y: target vector for training
            
        returns:
            dictionary containing model, predictions, probabilities and metrics
        """
        try:
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            
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
                verbosity=0  
            )

            
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

            
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)

            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            confusion = confusion_matrix(y_test, y_pred)

            
            logger.info(
                f"XGBoost Performance: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            
            return {
                'status': 'success',  
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
            return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}

    def train_gradient_boosting(self, X, y):
        """
        train a gradient boosting classifier with optimized parameters
        
        args:
            x: features matrix for training
            y: target vector for training
            
        returns:
            dictionary containing model, predictions, and performance metrics
        """
        logger.info("Training Gradient Boosting model...")
        try:
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            
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

            
            gb.fit(X_train, y_train)

            
            y_pred = gb.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            
            logger.info(f"Gradient Boosting Performance: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            
            return {
                'status': 'success',  
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
        except Exception as e:
            
            logger.error(f"Error training Gradient Boosting model: {e}")
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}


class EnhancedModelTrainer(ModelTrainer):
    def tune_random_forest(self, X, y):
        """Use RandomizedSearchCV to find optimal Random Forest parameters"""
        try: 
            param_grid = {
                'n_estimators': randint(100, 1200), 
                'max_depth': [10, 20, 30, 40, None], 
                'min_samples_split': randint(2, 20), 
                'min_samples_leaf': randint(1, 10),  
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True],
                'class_weight': ['balanced', 'balanced_subsample'],
                
                
            }

            rf = RandomForestClassifier(random_state=42)
            grid_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_grid,
                n_iter=2, 
                cv=3,     
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
        except Exception as e: 
            logger.error(f"Error tuning Random Forest: {str(e)}")
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}
    
    def tune_xgboost(self, X, y):
        """
        use randomized search to find optimal xgboost parameters
        
        args:
            x: features matrix for training
            y: target vector for training
            
        returns:
            best performing xgboost model with optimized parameters
        """
        
        from xgboost import XGBClassifier
        
        
        param_grid = {
            'n_estimators': randint(100, 700),  
            'max_depth': randint(3, 12),  
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],  
            'subsample': uniform(0.6, 0.4),  
            'colsample_bytree': uniform(0.6, 0.4),  
            'min_child_weight': randint(1, 15),  
            'gamma': uniform(0, 1.0),  
            'reg_alpha': [0, 0.005, 0.01, 0.05, 0.1, 0.5, 1],  
            'reg_lambda': [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],  
            'max_delta_step': [0, 1, 5],  
            'objective': ['multi:softprob'],  
            'booster': ['gbtree', 'dart'],  
            'tree_method': ['hist'],  
            'scale_pos_weight': [1] + list(uniform(1, 5).rvs(2))  
        }
        
        
        n_classes = len(np.unique(y))
        
        y = y.astype(int)
        model = XGBClassifier(
            num_class=n_classes,
            eval_metric='mlogloss',  
            random_state=42,
            verbosity=0  
        ) 
        
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=1,  
            cv=3,  
            verbose=2,  
            random_state=42,  
            n_jobs=-1,  
            scoring='accuracy'  
        )
        
        try:
            
            search.fit(X, y)
            
            
            best_params = search.best_params_
            best_score = search.best_score_
            
            
            logger.info(f"Best XGBoost params: {best_params}")
            logger.info(f"Best XGBoost score: {best_score:.4f}")
            
            
            best_model = XGBClassifier(
                **best_params,
                num_class=n_classes,
                eval_metric='mlogloss',  
                random_state=42,
                verbosity=0  
            )
            
            
            return search.best_estimator_
            
        except Exception as e:
            
            logger.error(f"Error tuning XGBoost: {str(e)}")
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}

    def tune_gradient_boosting(self, X, y, n_iter):
        """
        tune gradient boosting classifier using randomized search
        
        args:
            x: features matrix for training
            y: target vector for training
            n_iter: number of parameter settings to try
            
        returns:
            best performing gradient boosting model with optimized parameters
        """
        logger.info(f"Starting Gradient Boosting tuning with n_iter={n_iter}...")
        
        gb = GradientBoostingClassifier(random_state=42)

        
        param_grid = {
            'n_estimators': randint(100, 2500),  
            'learning_rate': [0.0005, 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15, 0.2],  
            'max_depth': randint(3, 20),  
            'min_samples_split': randint(2, 40),  
            'min_samples_leaf': randint(1, 40),  
            'subsample': uniform(0.5, 0.5),  
            'max_features': ['sqrt', 'log2', None],  
            'loss': ['log_loss'],  
            'criterion': ['friedman_mse', 'squared_error'],  
            'min_impurity_decrease': [0.0, 0.0001, 0.001, 0.01, 0.02, 0.05],  
            'max_leaf_nodes': [None] + list(range(20, 301, 30)),  
            'ccp_alpha': uniform(0.0, 0.1),  
            'init': [None]  
        }

        
        search = RandomizedSearchCV(
            estimator=gb,
            param_distributions=param_grid,
            n_iter=2,  
            cv=3,  
            verbose=1,  
            random_state=42,  
            scoring='accuracy',  
            n_jobs=-1  
        )
        
        try:
            
            search.fit(X, y)
            
            logger.info(f"Best Gradient Boosting params: {search.best_params_}")
            logger.info(f"Best Gradient Boosting score: {search.best_score_:.4f}")
            
            return search.best_estimator_
        except Exception as e:
            
            logger.error(f"Error during Gradient Boosting tuning: {e}")
            
            logger.warning("Falling back to default Gradient Boosting parameters.")
            return GradientBoostingClassifier(random_state=42)
    
    def train_catboost(self, X, y):
        """
        train a catboost classifier with hyperparameter tuning
        
        args:
            x: features matrix for training
            y: target vector for training
            
        returns:
            dictionary containing model, predictions, and performance metrics
        """
        try:
            
            from catboost import CatBoostClassifier
            
            
            param_grid = {
                'iterations': randint(100, 500),  
                'learning_rate': uniform(0.01, 0.3),  
                'depth': randint(4, 10),  
                'l2_leaf_reg': uniform(1, 9),  
                'bagging_temperature': uniform(0, 1),  
                'random_strength': uniform(0, 1),  
                'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']  
            }
            
            
            cat_clf = CatBoostClassifier(
                random_seed=42,  
                thread_count=-1,  
                verbosity=0  
            )
            
            
            search = RandomizedSearchCV(
                cat_clf, 
                param_grid, 
                n_iter=2,  
                cv=3,  
                scoring='accuracy',  
                random_state=42,  
                n_jobs=-1,  
                verbose=1  
            )
            
            
            search.fit(X, y)
            
            
            logger.info(f"Best CatBoost params: {search.best_params_}")
            logger.info(f"Best CatBoost score: {search.best_score_:.4f}")
            
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            best_model = search.best_estimator_
            best_model.fit(X_train, y_train)
            
            
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            
            logger.info(f"CatBoost test accuracy: {accuracy:.4f}")
            
            
            return {
                'status': 'success',  
                'model': best_model,  
                'accuracy': accuracy,  
                
            }
        except ImportError:
            
            logger.warning("CatBoost not installed. Skipping CatBoost model.")
            return {'status': 'skipped', 'error': 'CatBoost not installed', 'accuracy': 0.0}
        except Exception as e:
            
            logger.error(f"Error training CatBoost model: {e}")
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}

    def train_stacked_ensemble(self, X, y, base_models_dict):
        """
        train a stacked ensemble model with cross-validation using random forest as meta-learner
        
        args:
            x: features matrix for training
            y: target vector for training
            base_models_dict: dictionary of named base models to use in the ensemble
            
        returns:
            tuple containing model, accuracy, scaler, and feature names
        """
        
        from sklearn.ensemble import StackingClassifier, RandomForestClassifier
        from sklearn.base import clone
        
        
        if not base_models_dict:
            logger.error("No base models provided for stacking ensemble.")
            return None, 0.0, None, None

        
        base_estimators = []
        for name, model_instance in base_models_dict.items():
            try:
                
                unfitted_clone = clone(model_instance)
                base_estimators.append((name, unfitted_clone))
            except Exception as e:
                logger.warning(f"Failed to clone model {name}: {str(e)}")
                continue

        if not base_estimators:
            logger.error("No valid base models available for stacking.")
            return None, 0.0, None, None

        
        meta_learner = RandomForestClassifier(
            n_estimators=100,  
            max_depth=10,  
            random_state=42  
        )

        
        stacking_clf = StackingClassifier(
            estimators=base_estimators,  
            final_estimator=meta_learner,  
            cv=5,  
            stack_method='predict_proba',  
            n_jobs=-1  
        )

        try:
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            
            stacking_clf.fit(X_train, y_train)

            
            y_pred = stacking_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            
            logger.info(f"Stacking Ensemble accuracy: {accuracy:.4f}")

            
            return stacking_clf, accuracy, None, X.columns.tolist()

        except Exception as e:
            
            logger.error(f"Error training Stacking Ensemble: {str(e)}")
            logger.error(traceback.format_exc())
            return None, 0.0, None, None

    def train_voting_ensemble(self, X, y, estimators, voting='soft'):
        """
        train a voting ensemble classifier that combines predictions from multiple models
        
        args:
            x: features matrix for training
            y: target vector for training
            estimators: list of (name, model) tuples to use in the ensemble
            voting: voting strategy ('soft' for probabilities, 'hard' for class labels)
            
        returns:
            tuple containing model, accuracy, and feature names
        """
        
        from sklearn.ensemble import VotingClassifier

        
        voting_clf = VotingClassifier(
            estimators=estimators,  
            voting=voting,  
            n_jobs=-1  
        )

        try:
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            
            voting_clf.fit(X_train, y_train)

            
            y_pred = voting_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            
            logger.info(f"Voting Ensemble accuracy: {accuracy:.4f}")

            
            return voting_clf, accuracy, X.columns.tolist()

        except Exception as e:
            
            logger.error(f"Error training Voting Ensemble: {str(e)}")
            logger.error(traceback.format_exc())
            return None, 0.0, None

    
    def tune_lightgbm(self, X, y, n_iter=2):
        """
        tune lightgbm classifier using randomized search
        
        args:
            x: features matrix for training
            y: target vector for training
            n_iter: number of parameter settings to try
            
        returns:
            best performing lightgbm model with optimized parameters
        """
        
        import lightgbm as lgb

        
        param_grid = {
            'n_estimators': randint(100, 1000),  
            'num_leaves': randint(20, 100),  
            'learning_rate': uniform(0.01, 0.3),  
            'max_depth': randint(3, 12),  
            'min_child_samples': randint(5, 100),  
            'subsample': uniform(0.6, 0.4),  
            'colsample_bytree': uniform(0.6, 0.4),  
            'reg_alpha': uniform(0, 1),  
            'reg_lambda': uniform(0, 1),  
            'min_split_gain': uniform(0, 0.1),  
            'min_child_weight': uniform(0, 10),  
            'boosting_type': ['gbdt', 'dart', 'goss'],  
            'objective': ['multiclass'],  
            'metric': ['multi_logloss']  
        }

        
        lgb_clf = lgb.LGBMClassifier(
            random_state=42,  
            n_jobs=-1,  
            verbose=-1  
        )

        
        search = RandomizedSearchCV(
            estimator=lgb_clf,
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

            
            logger.info(f"Best LightGBM params: {search.best_params_}")
            logger.info(f"Best LightGBM score: {search.best_score_:.4f}")

            
            return search.best_estimator_

        except Exception as e:
            
            logger.error(f"Error tuning LightGBM: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    

    
    def tune_decision_tree(self, X, y, n_iter):
        """
        tune decision tree classifier using randomized search
        
        args:
            x: features matrix for training
            y: target vector for training
            n_iter: number of parameter settings to try
            
        returns:
            best performing decision tree model with optimized parameters
        """
        
        from sklearn.tree import DecisionTreeClassifier

        
        param_grid = {
            'max_depth': randint(3, 20),  
            'min_samples_split': randint(2, 20),  
            'min_samples_leaf': randint(1, 10),  
            'max_features': ['sqrt', 'log2', None],  
            'criterion': ['gini', 'entropy'],  
            'splitter': ['best', 'random'],  
            'max_leaf_nodes': [None] + list(range(10, 100, 10)),  
            'min_weight_fraction_leaf': uniform(0, 0.5),  
            'min_impurity_decrease': uniform(0, 0.1),  
            'ccp_alpha': uniform(0, 0.1)  
        }

        
        dt_clf = DecisionTreeClassifier(
            random_state=42  
        )

        
        search = RandomizedSearchCV(
            estimator=dt_clf,
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
            
            logger.error(f"Error tuning Decision Tree: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    


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
            
            os.makedirs(os.path.dirname(self.audit_log_file), exist_ok=True)

            
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
            
            if version == 'latest':
                
                model_files = glob.glob(
                    os.path.join(
                        self.model_dir,
                        f"{model_name}*.pkl"))
                if not model_files:
                    logger.error(f"No models found for {model_name}")
                    return False
                
                model_files.sort(reverse=True)
                model_path = model_files[0]
            else:
                model_path = os.path.join(
                    self.model_dir, f"{model_name}_v{version}.pkl")
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    return False

            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            
            required_keys = [
                'model',
                'metadata',
                'scaler',
                'feature_names'] 
            if not all(key in model_data for key in required_keys):
                
                missing = [key for key in required_keys if key not in model_data]
                logger.error(
                    f"Invalid model file format from {model_path}, missing required components: {missing}")
                return False

            
            self.model = model_data['model']
            self.metadata = model_data['metadata']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']

            
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
            
            
            self.anomaly_detector.fit(training_data, feature_names) 
            logger.info("Anomaly detector trained.")
        else:
            logger.warning("Could not train anomaly detector: No data or detector instance.")

    def save_model(
            self,
            model_name,
            model,
            scaler,
            feature_names,
            
            metadata=None,
            anomaly_detector=None): 
        """Save a trained model with important metadata"""
        try:
            
            os.makedirs(self.model_dir, exist_ok=True)

            
            version = datetime.now().strftime("%Y%m%d%H%M")
            
            base_model_name = model_name.split('_v')[0]
            
            model_path = os.path.join(
                self.model_dir, f"{base_model_name}_v{version}.joblib")

            
            if metadata is None:
                metadata = {}

            metadata.update({
                'version': version,
                'created_at': datetime.now().isoformat(),
                'feature_names': feature_names,
                'model_type': type(model).__name__
            })

            
            
            

            
            model_data = {
                'model': model,
                'metadata': metadata,
                'scaler': scaler,
                'feature_names': feature_names,
                'anomaly_detector': anomaly_detector, 
                
                'training_metadata': metadata.get('training_metadata', {}), 
                'class_mapping': metadata.get('class_mapping', {}) 
            }

            
            joblib.dump(model_data, model_path) 

            logger.info(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            logger.error(traceback.format_exc()) 
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

            
            if isinstance(features, dict):
                
                missing_features = [
                    f for f in self.feature_names if f not in features]
                if missing_features:
                    return {
                        'error': f'Missing features: {missing_features}',
                        'is_valid': False,
                        'confidence': 0.0
                    }

                
                X = np.array([features[f]
                             for f in self.feature_names]).reshape(1, -1)
            elif isinstance(features, pd.Series):
                
                X = features[self.feature_names].values.reshape(1, -1)
            else:
                X = features

            
            request_id = str(uuid.uuid4())

            
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            
            anomaly_result = self.anomaly_detector.detect_anomalies(
                X_scaled, company_name)
            is_anomaly = anomaly_result.get('is_anomaly', False)
            anomaly_score = anomaly_result.get('score', 0.0)
            anomaly_reasons = anomaly_result.get('reasons', [])

            
            prediction = int(self.model.predict(X_scaled)[0])
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probabilities))

            
            if is_anomaly:
                
                adjusted_confidence = confidence * \
                    (1 - min(anomaly_score, 0.9))
            else:
                adjusted_confidence = confidence

            
            result = {
                'prediction': prediction,
                'confidence': round(adjusted_confidence, 4),
                'is_valid': not is_anomaly,
                'request_id': request_id
            }

            
            if is_anomaly:
                result['validation'] = {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': round(anomaly_score, 4),
                    'reasons': anomaly_reasons,
                }

            
            self._log_prediction(
                request_id=request_id,
                company_name=company_name,
                prediction=prediction,
                confidence=adjusted_confidence,
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

            
            if isinstance(features, dict):
                
                missing_features = [
                    f for f in self.feature_names if f not in features]
                if missing_features:
                    return {
                        'error': f'Missing features: {missing_features}',
                        'is_valid': False
                    }

                
                X = np.array([features[f]
                             for f in self.feature_names]).reshape(1, -1)
            elif isinstance(features, pd.Series):
                
                X = features[self.feature_names].values.reshape(1, -1)
            else:
                X = features

            
            request_id = str(uuid.uuid4())

            
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            
            anomaly_result = self.anomaly_detector.detect_anomalies(
                X_scaled, company_name)
            is_anomaly = anomaly_result.get('is_anomaly', False)
            anomaly_score = anomaly_result.get('score', 0.0)
            anomaly_reasons = anomaly_result.get('reasons', [])

            
            probabilities = self.model.predict_proba(X_scaled)[0].tolist()
            classes = self.model.classes_.tolist() if hasattr(
                self.model, 'classes_') else list(range(len(probabilities)))

            
            if is_anomaly:
                
                
                adjustment_factor = 1.0 - \
                    min(anomaly_score, 0.8) 

                
                uniform_prob = 1.0 / len(probabilities)
                adjusted_probs = [
                    p * adjustment_factor + uniform_prob * (1 - adjustment_factor)
                    for p in probabilities
                ]

                
                total = sum(adjusted_probs)
                adjusted_probs = [p / total for p in adjusted_probs]
            else:
                adjusted_probs = probabilities

            
            result = {
                'probabilities': {
                    str(c): round(
                        p,
                        4) for c,
                    p in zip(
                        classes,
                        adjusted_probs)},
                
                'is_valid': not is_anomaly,
                'request_id': request_id}

            
            if is_anomaly:
                result['validation'] = {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': round(anomaly_score, 4),
                    'reasons': anomaly_reasons,
                }

            
            max_prob_idx = np.argmax(adjusted_probs)
            prediction = classes[max_prob_idx]
            confidence = adjusted_probs[max_prob_idx]

            
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
            
            timestamp = datetime.now().isoformat()
            model_version = self.metadata.get('version', 'unknown')

            
            if isinstance(feature_values, dict):
                feature_str = json.dumps({k: float(v) if isinstance(
                    v, (int, float, np.number)) else str(v) for k, v in feature_values.items()})
            else:
                feature_str = str(feature_values)

            
            if isinstance(anomaly_reasons, list):
                anomaly_reasons_str = '; '.join(anomaly_reasons)
            else:
                anomaly_reasons_str = str(anomaly_reasons)

            
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
            

    
    def load_model_joblib(self, model_name, version='latest'):
        """Load a trained model from disk (saved with joblib) with checks and validation"""
        model_path = "Not_Determined_Yet" 
        try:
            print(f"[ModelManager.load_model_joblib] Attempting to load: model_name='{model_name}', version='{version}'") 
            print(f"[ModelManager.load_model_joblib] Model directory (self.model_dir): '{self.model_dir}'") 

            
            if version == 'latest':
                
                logger.info(f"[load_model_joblib] Searching in directory: {self.model_dir}")
                
                
                
                if os.path.isabs(model_name) or model_name.endswith(".joblib") or model_name.endswith(".pkl"):
                    
                    
                    glob_pattern_base = model_name 
                else:
                    glob_pattern_base = model_name 

                pkl_pattern = os.path.join(self.model_dir, f"{glob_pattern_base}*.pkl")
                joblib_pattern = os.path.join(self.model_dir, f"{glob_pattern_base}*.joblib")
                print(f"[ModelManager.load_model_joblib] PKL glob pattern: '{pkl_pattern}'") 
                print(f"[ModelManager.load_model_joblib] JOBLIB glob pattern: '{joblib_pattern}'") 
                
                model_files_pkl = glob.glob(pkl_pattern)
                model_files_joblib = glob.glob(joblib_pattern)
                model_files = model_files_pkl + model_files_joblib 

                print(f"[ModelManager.load_model_joblib] Files found by glob (PKL): {model_files_pkl}") 
                print(f"[ModelManager.load_model_joblib] Files found by glob (JOBLIB): {model_files_joblib}") 
                print(f"[ModelManager.load_model_joblib] All files found: {model_files}") 

                if not model_files:
                    logger.error(f"No models found matching pattern {glob_pattern_base}* in {self.model_dir}")
                    print(f"[ModelManager.load_model_joblib] No model files found for pattern '{glob_pattern_base}*' in '{self.model_dir}'") 
                    return False

                
                model_files.sort(key=os.path.getmtime, reverse=True) 
                model_path = model_files[0]
                print(f"[ModelManager.load_model_joblib] Chosen model path (latest): '{model_path}'") 
            else:
                
                model_path_pkl = os.path.join(
                    self.model_dir, f"{model_name}_v{version}.pkl")
                model_path_joblib = os.path.join(
                    self.model_dir, f"{model_name}_v{version}.joblib")
                print(f"[ModelManager.load_model_joblib] Explicit version paths: PKL='{model_path_pkl}', JOBLIB='{model_path_joblib}'") 

                if os.path.exists(model_path_joblib): 
                    model_path = model_path_joblib
                elif os.path.exists(model_path_pkl):
                    model_path = model_path_pkl
                else:
                    logger.error(f"Model file not found for version {version}: {model_path_joblib} or {model_path_pkl}")
                    print(f"[ModelManager.load_model_joblib] Model file not found for version '{version}'.") 
                    return False
                print(f"[ModelManager.load_model_joblib] Chosen model path (specific version): '{model_path}'") 

            
            print(f"[ModelManager.load_model_joblib] Loading model from: '{model_path}'") 
            model_data = joblib.load(model_path)
            print(f"[ModelManager.load_model_joblib] Successfully loaded data from path. Keys in model_data: {list(model_data.keys())}") 

            
            required_keys = [
                'model',
                'metadata',
                'scaler',
                'feature_names'] 
            if not all(key in model_data for key in required_keys):
                
                missing = [key for key in required_keys if key not in model_data]
                logger.error(
                    f"Invalid model file format from {model_path}, missing required components: {missing}")
                return False

            
            self.model = model_data['model']
            self.metadata = model_data['metadata']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            
            self.anomaly_detector = model_data.get('anomaly_detector')
            if self.anomaly_detector:
                print(f"[ModelManager.load_model_joblib] Anomaly detector loaded: {type(self.anomaly_detector)}")
            else:
                print("[ModelManager.load_model_joblib] Anomaly detector not found in model artifact or is None.")
            


            
            version_info = self.metadata.get('version', 'unknown')
            created_at = self.metadata.get('created_at', 'unknown')
            logger.info(
                f"Loaded {model_name} v{version_info} (created {created_at}) from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            logger.error(traceback.format_exc()) 
            print(f"[ModelManager.load_model_joblib] Exception during load: {str(e)}, Path attempted: {model_path}") 
            return False


class Visualizer:
    def __init__(self, output_dir="./visualizations", interactive=False):
        """Initialize visualizer with output directory"""
        self.output_dir = output_dir
        self.interactive = interactive  

        
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created visualization directory: {output_dir}")
        except Exception as e:
            logger.error(f"Error creating visualization directory: {e}")
            
            self.output_dir = "./MainOutput/visualizations"
            os.makedirs(self.output_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.palette = sns.color_palette("husl", 8)
        self.feature_palettes = {
            'categorical': sns.color_palette("Set3", 12),
            'sequential': sns.color_palette("viridis", 8),
            'diverging': sns.color_palette("RdYlBu", 11)
        }

    def plot_funding_stage_distribution(self, data, stage_mapping_rev=None): 
        """Visualize the distribution of funding stages"""
        plt.figure(figsize=(12, 6))

        
        if stage_mapping_rev is None:
            
            logger.warning("No stage mapping provided to plot_funding_stage_distribution. Using generic labels.")
            
            stage_map_rev_local = {i: f'Stage {i}' for i in sorted(data['funding_stage_numeric'].unique())}
        else:
            
            stage_map_rev_local = {int(k): str(v) for k, v in stage_mapping_rev.items()}


        
        
        stage_counts = data['funding_stage_numeric'].map(
            lambda x: stage_map_rev_local.get(int(x), f'Unknown ({x})') 
        ).value_counts()

        
        
        try:
            num_order = {v: k for k, v in stage_map_rev_local.items()}
            sort_order = stage_counts.index.map(lambda x: num_order.get(x, 999)) 
            stage_counts = stage_counts.loc[sort_order.sort_values().index]
        except Exception:
            logger.warning("Could not sort stage distribution numerically, sorting alphabetically.")
            stage_counts = stage_counts.sort_index()


        
        ax = stage_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribution of Funding Stages (After Remapping & Merging)', fontsize=14) 
        plt.xlabel('Funding Stage')
        plt.ylabel('Number of Companies')
        plt.xticks(rotation=45, ha='right') 
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        
        for i, v in enumerate(stage_counts):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom') 

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

        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-15:]  

            
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
        """Compare performance of different models, including Accuracy and RMSE."""
        plt.figure(figsize=(15, 8)) 

        model_names = list(model_results.keys())
        accuracies = [model_results[m].get('accuracy', 0.0) if model_results[m] else 0.0 for m in model_names]
        rmses = [model_results[m].get('rmse', 0.0) if model_results[m] else 0.0 for m in model_names] 

        if not model_names:
            logger.warning("No model results to plot in model_comparison.")
            plt.close()
            return

        x = np.arange(len(model_names))  
        width = 0.35  

        fig, ax1 = plt.subplots(figsize=(15,8)) 

        
        rects1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')

        
        ax2 = ax1.twinx()
        rects2 = ax2.bar(x + width/2, rmses, width, label='RMSE', color='coral')

        
        ax1.set_ylabel('Accuracy', color='skyblue')
        ax2.set_ylabel('RMSE (Lower is Better)', color='coral') 
        ax1.set_title('Model Performance Comparison: Accuracy & RMSE', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=25, ha="right") 
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        
        ax1.set_ylim(0, max(1.0, max(accuracies) * 1.1 if accuracies else 1.0)) 
        ax2.set_ylim(0, max(rmses) * 1.2 if rmses and max(rmses) > 0 else 1.0) 

        
        def autolabel(rects, axis):
            for rect in rects:
                height = rect.get_height()
                axis.annotate(f'{height:.3f}', 
                              xy=(rect.get_x() + rect.get_width() / 2, height),
                              xytext=(0, 3),  
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=8)

        autolabel(rects1, ax1)
        autolabel(rects2, ax2)

        fig.tight_layout()  
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"model_comparison_accuracy_rmse_{self.timestamp}.png")) 
        if self.interactive:
            plt.show()
        plt.close(fig) 

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

        
        plot_data = data[['funding_amount', 'employees',
                          'funding_stage_numeric']].dropna()

        
        colors = plt.cm.viridis(np.linspace(0, 1, 12))
        plot_data['color'] = plot_data['funding_stage_numeric'].apply(
            lambda x: colors[int(x)] if 0 <= x < 12 else colors[0]
        )

        
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

        
        plt.subplot(2, 2, 1)
        corr = data[features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True)
        plt.title('Feature Correlation Matrix')

        
        plt.subplot(2, 2, 2)
        from scipy.cluster import hierarchy
        corr_linkage = hierarchy.ward(corr)
        sns.clustermap(corr, method='ward', cmap='coolwarm',
                       annot=True, fmt='.2f', figsize=(10, 10))

        
        plt.subplot(2, 2, 3, projection='3d')
        top_features = features[:3]  
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
            
            plt.subplot(n_features, 2, 2 * idx - 1)
            sns.histplot(
                data=data,
                x=feature,
                hue='funding_stage',
                multiple="stack",
                palette=self.feature_palettes['categorical'])
            plt.title(f'{feature} Distribution by Funding Stage')
            plt.xticks(rotation=45)

            
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
        plt.figure(figsize=(20, 12))  

        
        plt.subplot(2, 2, 1)
        sns.boxenplot(data=data, x='funding_year', y='funding_amount_log',
                      palette=self.feature_palettes['sequential'])
        plt.title('Funding Amount Distribution Over Time')
        plt.xticks(rotation=45)

        
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=data, x='employees', y='funding_amount',
                        hue='funding_stage', size='employee_efficiency',
                        sizes=(20, 200), alpha=0.6,
                        palette=self.feature_palettes['categorical'])
        plt.yscale('log')
        plt.xscale('log')
        plt.title('Funding Amount vs Employee Count')

        
        plt.subplot(2, 2, (3, 4))
        industry_funding = data.groupby('industry_category')[
            'funding_amount'].sum()
        industry_funding.sort_values(
            ascending=True).plot(
            kind='barh',
            color=self.feature_palettes['sequential'])
        plt.title('Total Funding by Industry')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)  
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

    def plot_predicted_stages_by_industry(self, data, stage_mapping_rev, top_n=7):
        """Visualize the distribution of true funding stages for the top N industries."""
        if 'industry_category' not in data.columns or 'funding_stage_numeric' not in data.columns:
            logger.warning("Required columns for industry stage plot are missing.")
            return

        plt.figure(figsize=(15, 8))

        
        top_industries = data['industry_category'].value_counts().nlargest(top_n).index

        
        top_industry_data = data[data['industry_category'].isin(top_industries)]

        if top_industry_data.empty:
            logger.warning(f"No data found for the top {top_n} industries.")
            plt.close()
            return

        
        
        safe_stage_mapping_rev = {int(k): str(v) for k, v in stage_mapping_rev.items()}
        plot_data = top_industry_data.copy()
        plot_data['funding_stage_label'] = plot_data['funding_stage_numeric'].map(
            lambda x: safe_stage_mapping_rev.get(int(x), f'Unknown_{x}')
        )

        
        industry_stage_counts = pd.crosstab(plot_data['industry_category'], plot_data['funding_stage_label'])

        
        
        try:
            
            
            
            sorted_stage_labels = sorted(plot_data['funding_stage_label'].unique())
            industry_stage_counts = industry_stage_counts.reindex(columns=sorted_stage_labels, fill_value=0)
        except Exception as e:
            logger.warning(f"Could not fully sort stage labels for industry plot: {e}")
            industry_stage_counts = industry_stage_counts.sort_index(axis=1)


        
        industry_stage_counts.plot(kind='bar', stacked=True, figsize=(15, 8), colormap='viridis') 

        plt.title(f'Funding Stage Distribution for Top {top_n} Industries', fontsize=14)
        plt.xlabel('Industry Category', fontsize=12)
        plt.ylabel('Number of Companies', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Funding Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) 

        filename = f"top_industries_stage_dist_{self.timestamp}.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        if self.interactive:
            plt.show()
        plt.close()
        logger.info(f"Saved top industries by funding stage plot to {filename}")

    def plot_top_n_companies(self, data, n=10, column='funding_amount', title_suffix='by Total Funding'):
        """Visualize the top N companies by a specified column."""
        if column not in data.columns or 'company_name' not in data.columns:
            logger.warning(f"Required columns ('{column}', 'company_name') not in data for top N companies plot.")
            return

        plt.figure(figsize=(12, 8))

        
        top_companies = data.groupby('company_name')[column].sum().nlargest(n)

        if top_companies.empty:
            logger.warning(f"No data found for top {n} companies by '{column}'.")
            plt.close()
            return

        top_companies.sort_values(ascending=True).plot(kind='barh', color='teal')
        
        plt.title(f'Top {n} Companies {title_suffix}', fontsize=14)
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel('Company Name')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        filename = f"top_{n}_companies_{column}_{self.timestamp}.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        if self.interactive:
            plt.show()
        plt.close()
        logger.info(f"Saved top {n} companies plot to {filename}")


class AdvancedVisualizer(Visualizer):
    def plot_roc_curves(self, y_true, y_proba, classes):
        plt.figure(figsize=(10, 8))
        y_bin = label_binarize(y_true, classes=classes)
        n_classes = len(classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        
        colors = plt.colormaps['tab10'](np.linspace(0, 1, n_classes))
        for i in range(n_classes):
            try:
                
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

    def plot_calibration(self, y_true, y_proba, model_name, n_bins=5):
        plt.figure(figsize=(10, 6))
        try: 
            if y_proba is None:
                 logger.warning(f"y_proba is None in plot_calibration for {model_name}. Skipping.")
                 plt.close() 
                 return
            if y_proba.ndim == 1:
                logger.warning(f"Converting 1D probability array to 2D for {model_name}")
                y_proba = np.column_stack([1 - y_proba, y_proba])

            
            if len(y_true) != len(y_proba):
                 logger.error(f"Mismatched lengths in plot_calibration for {model_name}: y_true ({len(y_true)}), y_proba ({len(y_proba)}). Skipping.")
                 plt.close()
                 return

            n_classes = y_proba.shape[1]
            unique_true_classes = np.unique(y_true)
            plotted_anything = False 

            for class_idx in range(n_classes):
                
                if class_idx not in unique_true_classes:
                     
                     continue

                try:
                    binary_y = (y_true == class_idx).astype(int)
                    
                    if np.sum(binary_y) == 0:
                         
                         continue

                    class_proba = y_proba[:, class_idx]
                    prob_true, prob_pred = calibration_curve(
                        binary_y, class_proba, n_bins=n_bins, strategy='quantile'
                    )
                    
                    plt.plot(prob_pred, prob_true, marker='o', linestyle='-', 
                             label=f'Class {class_idx}', alpha=0.7)
                    plotted_anything = True 
                except ValueError as ve:
                     
                     logger.warning(f"Could not calculate calibration curve for class {class_idx} of {model_name} (likely due to empty bins): {str(ve)}")
                     continue
                except Exception as e:
                    logger.warning(
                        f"Error calculating or plotting calibration for class {class_idx} of {model_name}: {str(e)}") 
                    continue 

            if not plotted_anything:
                logger.warning(f"No calibration curves were successfully plotted for {model_name}. Skipping save.")
                plt.close()
                return

            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated') 
            plt.xlabel('Mean Predicted Probability (Bin)')
            plt.ylabel('Fraction of Positives (Bin)')
            plt.title(f'Calibration Plot - {model_name} (One-vs-Rest, Quantile Binning)') 
            plt.legend(loc='upper left', fontsize='small')
            plt.grid(True, alpha=0.3)
            
            safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
            filename = f"calibration_plot_{safe_model_name}_{self.timestamp}.png"
            filepath = os.path.join(self.output_dir, filename) 
            plt.savefig(filepath)
            logger.info(f"Successfully saved calibration plot for {model_name} to {filepath}") 
            if self.interactive:
                plt.show()
            plt.close()
        except Exception as plot_err: 
            logger.error(f"Failed to generate or save calibration plot for {model_name}: {plot_err}", exc_info=True)
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

        
        self.models_dir = os.path.join(output_dir, "models")
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        
        self.data_loader = DataLoader(base_dir, archive=archive, output_dir_for_db=self.output_dir)
        self.feature_engineer = FeatureEngineering()
        self.model_trainer = EnhancedModelTrainer(self.models_dir)
        self.visualizer = Visualizer(self.viz_dir)

        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self):
        """Execute the full pipeline"""
        try:
            logger.info("Starting funding stage prediction pipeline")

            
            logger.info("Step 1: Loading and merging data...")
            merged_data = self.data_loader.merge_datasets()

            if merged_data.empty:
                logger.error("No data available. Exiting pipeline.")
                return False

            
            logger.info("Step 2: Extracting features...")
            processed_data = self.feature_engineer.extract_features(
                merged_data)

            
            logger.info("Step 3: Preparing model data...")
            X, y, feature_columns = self.feature_engineer.prepare_model_data(processed_data)

            
            logger.info("Step 4: Training models...")
            rf_model, rf_results = self.model_trainer.train_random_forest(X, y)
            xgb_model, xgb_results = self.model_trainer.train_xgboost(X, y)

            
            model_results = {
                'Random Forest': rf_results,
                'XGBoost': xgb_results
            }

            
            logger.info("Step 5: Creating visualizations...")
            self.visualizer.plot_funding_stage_distribution(processed_data)
            self.visualizer.plot_feature_importance(rf_model, feature_columns)
            self.visualizer.plot_feature_importance(xgb_model, feature_columns)
            self.visualizer.plot_model_comparison(model_results)
            self.visualizer.plot_funding_vs_employees(processed_data)

            
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
            
            self.visualizer.plot_pairwise_features(
                processed_data, key_features)
            self.visualizer.plot_full_correlation_heatmap(processed_data)
            self.visualizer.plot_violin_funding_by_stage(processed_data)

            
            logger.info("Step 6: Saving summary report...")
            summary = {
                'timestamp': self.timestamp,
                'data_records': len(merged_data),
                'features': feature_columns,
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

        
        schedule.every(interval_hours).hours.do(job)

        
        job()

        
        logger.info(
            f"Scheduler started. Will run every {interval_hours} hours")
        while True:
            schedule.run_pending()
            time.sleep(60)

    def _init_model_directory(self):
        """Create organized model directory structure"""
        
        os.makedirs(self.models_dir, exist_ok=True)

        
        model_types = ['random_forest', 'xgboost', 'ensemble']
        for model_type in model_types:
            os.makedirs(
                os.path.join(
                    self.models_dir,
                    model_type),
                exist_ok=True)

        
        os.makedirs(os.path.join(self.models_dir, 'evaluation'), exist_ok=True)

        logger.info(
            f"Initialized model directory structure at {
                self.models_dir}")


class EnhancedPipeline(FundingStagePredictionPipeline):
    def __init__(self, *args, **kwargs):
        
        if 'output_dir' in kwargs:
            kwargs['output_dir'] = './cs163-main/backend/MainOutput'  
        else:
            args = list(args)
            if len(args) > 1:
                args[1] = './cs163-main/backend/MainOutput'  
            else:
                
                if len(args) == 0: 
                    args.append(None) 
                args.append('./cs163-main/backend/MainOutput') 
            args = tuple(args)

        super().__init__(*args, **kwargs)
        self.model_trainer = EnhancedModelTrainer(self.models_dir)

        
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        
        self.visualizer = AdvancedVisualizer(self.viz_dir, interactive=False)
        self.model_manager = ModelManager(self.models_dir)
        
        
        self.time_series_forecaster = TimeSeriesForecaster(self.viz_dir) 
        self._init_model_directory()
        
        self.final_class_mapping = {}
        self.reverse_final_class_mapping = {}
        self.feature_names = []


    def run(self):
        try:
            logger.info("Starting funding stage prediction pipeline")
            
            
            all_model_results = {} 

            
            merged_data = self.data_loader.merge_datasets()
            if merged_data.empty:
                logger.error("No data available. Exiting pipeline.")
                return False

            processed_data = self.feature_engineer.extract_features(
                merged_data)
            X, y, feature_columns = self.feature_engineer.prepare_model_data(processed_data)

            
            
            def remap_classes(y_series, original_labels_series=None):
                
                
                
                
                if original_labels_series is not None and not original_labels_series.empty:
                    
                    unique_classes = sorted(original_labels_series.dropna().astype(str).unique())
                    logger.debug(f"remap_classes using labels from original_labels_series: {unique_classes}")
                elif y_series.dtype == 'object': 
                    unique_classes = sorted(y_series.dropna().unique()) 
                    logger.debug(f"remap_classes using labels from object y_series: {unique_classes}")
                else: 
                    unique_classes = sorted(y_series.dropna().unique()) 
                    logger.debug(f"remap_classes using labels from numeric y_series: {unique_classes}")
                
                unique_classes = [str(cls) for cls in unique_classes]

                
                class_map = {
                    str(label): idx for idx, label in enumerate(unique_classes) 
                }

                
                
                if original_labels_series is not None and not original_labels_series.empty:
                    
                    remapped_y = original_labels_series.astype(str).map(class_map).fillna(-1).astype(int)
                    
                    return remapped_y, class_map
                else:
                     
                    remapped_y = y_series.astype(str).map(class_map).fillna(-1).astype(int)
                     
                    return remapped_y, class_map

            
            
            if 'funding_stage' not in processed_data.columns:
                logger.error("Original target column 'funding_stage' not found.")
                return False
            
            original_string_labels = processed_data['funding_stage'].copy()

            
            y_initial_indices, initial_map_str_to_idx = remap_classes(original_string_labels)
            
            logger.info(f"Initial class mapping (String Label -> Initial Index): {initial_map_str_to_idx}")

            
            class_counts = pd.Series(y_initial_indices).value_counts()
            min_samples_threshold = 10
            rare_classes = class_counts[class_counts < min_samples_threshold].index.tolist()
            y_merged = y_initial_indices.copy() 
            if rare_classes:
                majority_class_index = class_counts.idxmax()
                
                y_merged = y_merged.apply(lambda x: majority_class_index if x in rare_classes else x)
                logger.info(
                    f"Merged rare classes (based on initial indices) into majority class index {majority_class_index}")

            
            
            unique_merged_indices = sorted(y_merged.dropna().unique())
            initial_idx_to_final_idx_map = { 
                initial_idx: final_idx for final_idx, initial_idx in enumerate(unique_merged_indices)
            }
            y_final = y_merged.map(initial_idx_to_final_idx_map).fillna(-1).astype(int)
            logger.info(
                f"Final map (Initial Index -> Final Index) after merging rare classes: {initial_idx_to_final_idx_map}")

            
            
            initial_map_idx_to_str = {v: k for k, v in initial_map_str_to_idx.items()} 
            self.final_index_to_string_label_map = {}
            for initial_index, final_index in initial_idx_to_final_idx_map.items():
                 original_string_label = initial_map_idx_to_str.get(int(initial_index))
                 if original_string_label is not None:
                     self.final_index_to_string_label_map[final_index] = original_string_label 
                 else:
                     logger.warning(f"Could not find original string label for initial index {initial_index} when creating final map.")
                     self.final_index_to_string_label_map[final_index] = f"LABEL_NOT_FOUND_FOR_INDEX_{initial_index}"

            
            y = y_final 
            
            
            
            if len(X) != len(y):
                 logger.warning(f"X ({len(X)}) and final y ({len(y)}) have different lengths. Aligning X based on y's index.")
                 
                 
                 
                 valid_y_indices = y.index 
                 X = X.loc[valid_y_indices] 
                 if len(X) != len(y): 
                     logger.error("CRITICAL: Failed to align X and y after remapping. Aborting.")
                     return False 
            
            X = X.reset_index(drop=True) 
            y = y.reset_index(drop=True)

            
            logger.info(f"Correct Final Index -> String Label mapping for saving: {self.final_index_to_string_label_map}")

            
            logger.info(f"Using all {X.shape[1]} engineered features for training.")
            if hasattr(X, 'columns'):
                 self.feature_names = X.columns.tolist()
            else: 
                 
                 if not isinstance(X, pd.DataFrame):
                      X = pd.DataFrame(X)
                 self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                 X.columns = self.feature_names 
                 logger.warning("X was not a DataFrame, assigned generic feature names.")

            
            
            try:
                logger.info("Attempting to fit main anomaly detector on core numeric features...")
                core_numeric_features_list = [
                    'funding_amount_log', 'employees', 'employee_efficiency',
                    'funding_year', 'funding_month', 'previous_rounds',
                    'months_since_first_funding', 'funding_velocity',
                    'month_sin', 'month_cos',
                    'funding_amount_x_age', 'employees_x_rounds'
                    
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
            

            
            logger.info("*** Preparing data and training dashboard-specific model... ***")
            
            
            all_features_for_dashboard = X.columns.tolist() 
            logger.info(f"Using ALL {len(all_features_for_dashboard)} features for the dashboard model.")

            
            if not all(f in X.columns for f in all_features_for_dashboard):
                missing_dash = [f for f in all_features_for_dashboard if f not in X.columns]
                logger.error(f"Dashboard Prep Error: Missing expected features in main X: {missing_dash}")
                
            else:
                
                X_dashboard = X[all_features_for_dashboard].copy()
                y_dashboard = y 

                logger.info(f"Dashboard model features (ALL): {len(all_features_for_dashboard)} features")
                logger.info(f"Dashboard data shapes: X={X_dashboard.shape}, y={y_dashboard.shape}")

                
                logger.info("--- Checking y_dashboard values before split ---")
                
                logger.info("------------------------------------------------")

                
                X_train_dash, X_test_dash, y_train_dash, y_test_dash = train_test_split(
                    X_dashboard, y_dashboard, test_size=0.2, random_state=42, stratify=y_dashboard
                )

                
                scaler_dash = StandardScaler()
                X_train_dash_scaled = scaler_dash.fit_transform(X_train_dash)
                X_test_dash_scaled = scaler_dash.transform(X_test_dash)

                
                logger.info("Training dashboard RandomForest with ALL features...") 
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
                    logger.info(f"Dashboard RandomForest (All Features) Test Accuracy: {dash_acc:.4f}") 

                    
                    
                    anomaly_detector_dash = AnomalyDetector(contamination=0.05)
                    
                    anomaly_detector_dash.fit(X_train_dash_scaled) 
                    logger.info("Fitted anomaly detector specifically for ALL dashboard model features.") 

                    
                    dash_metadata = {
                        'model_type': 'RandomForest_Dashboard_AllFeatures', 
                        'features': all_features_for_dashboard, 
                        'accuracy': dash_acc
                    }
                    
                    
                    timestamp_str = datetime.now().strftime("%Y%m%d%H%M")
                    model_filename_dash = f"Dashboard_Model_{dash_metadata['model_type']}_v{timestamp_str}.joblib"
                    model_path_dash = os.path.join(self.models_dir, model_filename_dash)
                    
                    
                    final_map_to_save = getattr(self, 'final_index_to_string_label_map', {})
                    logger.info(f"Saving final map to dashboard model: {final_map_to_save}")

                    joblib.dump({
                        'model': dashboard_model_trained,
                        'scaler': scaler_dash,
                        'feature_names': all_features_for_dashboard, 
                        'class_mapping': final_map_to_save, 
                        'anomaly_detector': anomaly_detector_dash,
                        'training_metadata': dash_metadata
                    }, model_path_dash)
                    logger.info(f"Dashboard-specific model (All Features) saved to {model_path_dash}")
                    
                    if os.path.exists(model_path_dash):
                        logger.info(f"CONFIRMED: Dashboard model file (All Features) exists at {model_path_dash}")
                    else:
                        logger.error(f"FAILED TO SAVE Dashboard model file (All Features) at {model_path_dash}")

                except ImportError:
                    logger.error("Scikit-learn not installed? Cannot train dashboard RandomForest.")
                except Exception as e:
                    logger.error(f"Error training/saving dashboard model (All Features): {e}", exc_info=True)
            

            
            logger.info("Step 3.5: Generating Time Series Forecast & Dashboard Prototype Plot...")
            if Prophet is not None: 
                try:
                    
                    prophet_data = self.time_series_forecaster.prepare_prophet_data(processed_data)
                    if prophet_data is not None and not prophet_data.empty:
                        prophet_model, prophet_forecast = self.time_series_forecaster.train_predict(prophet_data, periods=6)
                        if prophet_model and prophet_forecast is not None:
                            
                            self.time_series_forecaster.plot_dashboard_prototype(prophet_forecast, prophet_data)
                            self.time_series_forecaster.plot_forecast(prophet_model, prophet_forecast) 
                        else:
                            logger.warning("Prophet model training/prediction failed. Skipping dashboard plot.")
                    else:
                        logger.warning("Data preparation for Prophet failed or yielded no data. Skipping forecast plot.")
                except Exception as ts_err:
                    logger.error(f"Error during time series forecasting step: {ts_err}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning("Prophet not installed, skipping time series forecasting and dashboard prototype plot.")
            

            
            logger.info("Step 4: Tuning and Training models...") 
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            
            logger.info("Tuning Random Forest model...")
            tuned_rf_model_candidate = self.model_trainer.tune_random_forest(X.copy(), y.copy())
            if tuned_rf_model_candidate and not isinstance(tuned_rf_model_candidate, dict): 
                logger.info("Training final Random Forest model with tuned parameters...")
                rf_model_to_train = tuned_rf_model_candidate
                rf_model_name = 'Random Forest (Tuned)'
            else:
                logger.warning("Random Forest tuning failed or skipped. Training with defaults.")
                rf_model_to_train = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced', n_estimators=150) 
                rf_model_name = 'Random Forest (Default)'
            rf_model, rf_results = self._train_final_model(
                rf_model_to_train, X.copy(), y.copy(), rf_model_name
            )
            if rf_results: all_model_results[rf_model_name] = rf_results


            
            xgb_model, xgb_results = None, None 
            logger.info("Tuning XGBoost model...")
            tuned_xgb_model_candidate = self.model_trainer.tune_xgboost(X.copy(), y.copy())
            if tuned_xgb_model_candidate and not isinstance(tuned_xgb_model_candidate, dict):
                logger.info("Training final XGBoost model with tuned parameters...")
                xgb_model_to_train = tuned_xgb_model_candidate
                xgb_model_name = 'XGBoost (Tuned)'
            else:
                logger.warning("XGBoost tuning failed or skipped. Training with defaults.")
                try:
                    from xgboost import XGBClassifier
                    n_classes_xgb = len(np.unique(y))
                    xgb_model_to_train = XGBClassifier(
                        random_state=42, num_class=n_classes_xgb, eval_metric='mlogloss', verbosity=0
                    )
                    xgb_model_name = 'XGBoost (Default)'
                except ImportError:
                    logger.warning("XGBoost not installed. Skipping XGBoost model.")
                    xgb_model_to_train = None
                    xgb_model_name = 'XGBoost (Skipped)'
                except Exception as xgb_init_err:
                    logger.error(f"Error initializing default XGBoost: {xgb_init_err}")
                    xgb_model_to_train = None
                    xgb_model_name = 'XGBoost (Error)'
            
            if xgb_model_to_train:
                xgb_model, xgb_results = self._train_final_model(
                    xgb_model_to_train, X.copy(), y.copy(), xgb_model_name
                )
                if xgb_results: all_model_results[xgb_model_name] = xgb_results
            elif xgb_model_name not in all_model_results : 
                 all_model_results[xgb_model_name] = None


            
            logger.info("Tuning Gradient Boosting model (n_iter=4)...") 
            tuned_gb_model_candidate = self.model_trainer.tune_gradient_boosting(X.copy(), y.copy(), n_iter=2) 
            if tuned_gb_model_candidate and not isinstance(tuned_gb_model_candidate, dict):
                logger.info("Training final Gradient Boosting model with tuned parameters...")
                gb_model_to_train = tuned_gb_model_candidate
                gb_model_name = 'Gradient Boosting (Tuned)'
            else:
                logger.warning("Gradient Boosting tuning failed or skipped. Training with defaults.")
                gb_model_to_train = GradientBoostingClassifier(random_state=42)
                gb_model_name = 'Gradient Boosting (Default)'
            gb_model, gb_results = self._train_final_model(
                gb_model_to_train, X.copy(), y.copy(), gb_model_name
            )
            if gb_results: all_model_results[gb_model_name] = gb_results
            

            
            logger.info("Tuning Decision Tree model (n_iter=4)...") 
            from sklearn.tree import DecisionTreeClassifier 
            tuned_dt_model_candidate = self.model_trainer.tune_decision_tree(X.copy(), y.copy(), n_iter=2) 
            if tuned_dt_model_candidate and not isinstance(tuned_dt_model_candidate, dict):
                logger.info("Training final Decision Tree model with tuned parameters...")
                dt_model_to_train = tuned_dt_model_candidate
                dt_model_name = 'Decision Tree (Tuned)'
            else:
                logger.warning("Decision Tree tuning failed or skipped. Training with defaults.")
                dt_model_to_train = DecisionTreeClassifier(random_state=42, class_weight='balanced')
                dt_model_name = 'Decision Tree (Default)'
            dt_model, dt_results = self._train_final_model(
                dt_model_to_train, X.copy(), y.copy(), dt_model_name
            )
            if dt_results: all_model_results[dt_model_name] = dt_results


            
            lr_model, lr_results = None, None 

            
            
            knn_model, knn_results = None, None 

            
            
            svc_model, svc_results = None, None 

            
            
            mlp_model, mlp_results = None, None 

            
            logger.info("Training and evaluating Stacking Ensemble (Random Forest Meta-Model)...")
            stacking_ensemble_results = None 
            meta_model_trained = None
            stacking_scaler = None 

            unfitted_base_models_for_stacking = {}
            
            rf_stack_candidate = all_model_results.get('Random Forest (Tuned)', {}).get('model') or all_model_results.get('Random Forest (Default)', {}).get('model')
            if rf_stack_candidate: unfitted_base_models_for_stacking['RandomForest'] = clone(rf_stack_candidate) 
            
            
            xgb_stack_candidate = all_model_results.get('XGBoost (Tuned)', {}).get('model') or all_model_results.get('XGBoost (Default)', {}).get('model')
            if xgb_stack_candidate: unfitted_base_models_for_stacking['XGBoost'] = clone(xgb_stack_candidate)

            
            gb_stack_candidate = all_model_results.get('Gradient Boosting (Tuned)', {}).get('model') or all_model_results.get('Gradient Boosting (Default)', {}).get('model')
            if gb_stack_candidate: unfitted_base_models_for_stacking['GradientBoosting'] = clone(gb_stack_candidate)

            
            dt_stack_candidate = all_model_results.get('Decision Tree (Tuned)', {}).get('model') or all_model_results.get('Decision Tree (Default)', {}).get('model')
            if dt_stack_candidate: unfitted_base_models_for_stacking['DecisionTree'] = clone(dt_stack_candidate)
            
            logger.info(f"Base models for Stacking Ensemble: {list(unfitted_base_models_for_stacking.keys())}")

            if len(unfitted_base_models_for_stacking) >= 2:
                try:
                    stacking_output = self.model_trainer.train_stacked_ensemble(
                        X.copy(), y.copy(), unfitted_base_models_for_stacking
                    )
                    if isinstance(stacking_output, tuple) and len(stacking_output) == 4:
                        meta_model_trained, stack_accuracy, stack_y_proba, stacking_scaler = stacking_output
                        
                        if meta_model_trained:
                            _, X_test_stack, _, y_test_stack = train_test_split(X.copy(), y.copy(), test_size=0.2, random_state=42, stratify=y.copy())
                            X_test_stack_scaled = stacking_scaler.transform(X_test_stack) 
                            stack_y_pred = meta_model_trained.predict(X_test_stack_scaled)
                            
                            stacking_ensemble_results = {
                                'model': meta_model_trained,
                                'calibrated_model': meta_model_trained, 
                                'accuracy': stack_accuracy,
                                'precision': precision_score(y_test_stack, stack_y_pred, average='weighted', zero_division=0),
                                'recall': recall_score(y_test_stack, stack_y_pred, average='weighted', zero_division=0),
                                'f1_score': f1_score(y_test_stack, stack_y_pred, average='weighted', zero_division=0),
                                'rmse': np.sqrt(mean_squared_error(y_test_stack, stack_y_pred)),
                                'confusion_matrix': confusion_matrix(y_test_stack, stack_y_pred),
                                'classification_report': classification_report(y_test_stack, stack_y_pred, output_dict=True, zero_division=0),
                                'cross_val_scores': [stack_accuracy], 
                                'X_test': X_test_stack_scaled, 
                                'y_test': y_test_stack,
                                'y_pred': stack_y_pred,
                                'y_proba': stack_y_proba,
                                'feature_names': self.feature_names, 
                                'scaler': stacking_scaler, 
                                'anomaly_detector': None 
                            }
                            all_model_results['Stacking Ensemble (RF Meta)'] = stacking_ensemble_results
                        else:
                            logger.warning("Stacking ensemble training returned no model.")
                    else:
                         logger.error("train_stacked_ensemble returned unexpected output.")
                except Exception as stack_err:
                     logger.error(f"Error training or evaluating stacking ensemble: {stack_err}", exc_info=True)
                     all_model_results['Stacking Ensemble (RF Meta)'] = None 
            else:
                logger.warning(f"Not enough valid base models ({len(unfitted_base_models_for_stacking)}) for stacking. Skipping.")
                all_model_results['Stacking Ensemble (RF Meta)'] = None
            

            
            lgbm_results = None 
            lgbm_model_name_final = 'LightGBM (Skipped)' 
            if LGBMClassifier is not None:
                logger.info("Tuning LightGBM model (n_iter=1)...") 
                tuned_lgbm_model_candidate = self.model_trainer.tune_lightgbm(X.copy(), y.copy(), n_iter=1) 
                if tuned_lgbm_model_candidate and not isinstance(tuned_lgbm_model_candidate, dict):
                    logger.info("Training final LightGBM model with tuned parameters...")
                    lgbm_model_to_train = tuned_lgbm_model_candidate
                    lgbm_model_name_final = 'LightGBM (Tuned)'
                else:
                    logger.warning("LightGBM tuning failed or skipped. Training with defaults.")
                    try:
                        lgbm_model_to_train = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1) 
                        lgbm_model_name_final = 'LightGBM (Default)'
                    except Exception as lgbm_init_err:
                        logger.error(f"Error initializing default LightGBM: {lgbm_init_err}")
                        lgbm_model_to_train = None
                        lgbm_model_name_final = 'LightGBM (Error)'
                
                if lgbm_model_to_train:
                    lgbm_model, lgbm_results = self._train_final_model(
                        lgbm_model_to_train, X.copy(), y.copy(), lgbm_model_name_final
                    )
                    if lgbm_results: all_model_results[lgbm_model_name_final] = lgbm_results
                elif lgbm_model_name_final not in all_model_results:
                     all_model_results[lgbm_model_name_final] = None
            else:
                logger.warning("LightGBM not installed or LGBMClassifier is None. Skipping LightGBM model.")
                all_model_results[lgbm_model_name_final] = None


            
            logger.info("Step 5: Creating advanced visualizations...")
            
            self._create_advanced_visualizations(
                all_model_results=all_model_results, 
                y=y, 
                processed_data=processed_data 
            )

            
            logger.info("Step 6: Saving summary and best models...")
            
            summary_data = self._save_summary(
                 merged_data, X, all_model_results, processed_data 
            )

            if not summary_data:
                 logger.error("Failed to generate or save pipeline summary.")
                 return False 

            
            
            

            logger.info("Pipeline completed successfully!") 
            return True 

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            return False

    
    def _train_final_model(self, model, X, y, name):
        """Helper method to train final models with detailed metrics including RMSE and Calibration""" 
        
        from sklearn.base import clone
        
        if not isinstance(X, pd.DataFrame):
            logger.warning(f"Input X for {name} is not a DataFrame. Creating one with generic feature names.")
            feature_names_list = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names_list)
        else:
            X_df = X.copy() 
            feature_names_list = X_df.columns.tolist()

        
        if not isinstance(y, pd.Series):
             y = pd.Series(y)

        
        X_df = X_df.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42, stratify=y)

        
        scaler = StandardScaler() 
        
        X_train_scaled = scaler.fit_transform(X_train)
        
        X_test_scaled = scaler.transform(X_test)

        
        
        

        
        cv_scores = np.array([np.nan]) 
        try:
            
            
            if hasattr(model, 'random_state') or isinstance(model, (VotingClassifier)):
                 model_clone = clone(model)
            else:
                 
                 
                 
                 try:
                     model_clone = clone(model)
                 except TypeError: 
                     logger.warning(f"Could not clone model {name} for CV. Using original instance.")
                     model_clone = model

            
            
            
            logger.info(f"Skipping cross-validation for {name} in this run.") 
            cv_scores = np.array([np.nan]) 

        except Exception as cv_err:
            logger.error(f"Cross-validation failed for {name}: {cv_err}")
            cv_scores = np.array([np.nan]) 

        
        
        try:
            model_final = clone(model) 
        except TypeError:
             logger.warning(f"Could not clone model {name} for final fit. Using original instance.")
             model_final = model

        
        fitted_model = None
        calibrated_model = None 
        try:
             fitted_model = model_final.fit(X_train_scaled, y_train)

             
             
             calibration_method = 'isotonic' 
             if name == 'Decision Tree':
                 calibration_method = 'sigmoid'
                 logger.info(f"Using 'sigmoid' calibration for {name} model.")
             else:
                 logger.info(f"Using 'isotonic' calibration for {name} model.")

             if fitted_model:
                 calibrated_model = CalibratedClassifierCV(fitted_model, method=calibration_method, cv='prefit')
                 calibrated_model.fit(X_train_scaled, y_train)
                 logger.info(f"Calibration complete for {name} using {calibration_method} method.")
             else: 
                 logger.warning(f"Cannot calibrate {name} as the base model was not fitted successfully.")
                 calibrated_model = None 
             

             
             anomaly_detector_instance = None
             
             
             
             if fitted_model: 
                 try:
                     logger.info(f"Fitting anomaly detector for {name}...")
                     anomaly_detector_instance = AnomalyDetector(contamination=0.05) 
                     
                     anomaly_detector_instance.fit(X_train_scaled)
                     logger.info(f"Anomaly detector fitted for {name}.")
                 except Exception as ad_fit_err:
                     logger.error(f"Failed to fit anomaly detector for {name}: {ad_fit_err}")
                     anomaly_detector_instance = None 
             

        except Exception as fit_err:
             logger.error(f"Fitting or Calibrating final model failed for {name}: {fit_err}")
             
             return None, { 
                 'model': None,
                 'calibrated_model': None, 
                 'anomaly_detector': None, 
                 'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan,
                 'rmse': np.nan, 'confusion_matrix': None, 'classification_report': None,
                 'cross_val_scores': cv_scores, 'X_test': X_test_scaled, 'y_test': y_test,
                 'y_pred': None, 'y_proba': None, 'feature_names': feature_names_list, 'scaler': scaler
             }


        
        y_pred = calibrated_model.predict(X_test_scaled)
        y_proba = calibrated_model.predict_proba(X_test_scaled) 

        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)

        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        
        
        report_labels = np.unique(np.concatenate((y_test, y_pred)))
        class_report = classification_report(y_test, y_pred, labels=report_labels, output_dict=True, zero_division=0)

        
        logger.info(f"--- {name} Model Performance ---")
        logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        logger.info(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")
        logger.info(f"RMSE: {rmse:.4f} (lower is better)")
        logger.info("-------------------------")

        
        print(f"\\n--- {name} Model Performance ---")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")
        print(f"RMSE: {rmse:.4f} (lower is better)")
        print("-------------------------")

        
        return fitted_model, { 
            'model': fitted_model, 
            'calibrated_model': calibrated_model, 
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
            'feature_names': feature_names_list, 
            'scaler': scaler, 
            'anomaly_detector': anomaly_detector_instance 
        }

    def _create_advanced_visualizations(
            self,
            all_model_results, 
            y, 
            processed_data): 
        """Generate advanced model diagnostics and data visualizations""" 
        try:
            unique_classes = np.unique(y) 

            
            
            all_results_plot = {k: v for k, v in all_model_results.items() if v is not None} if all_model_results else {}

            if not all_results_plot:
                 logger.warning("No valid model results available for visualization.")
                 return 

            
            logger.info("Creating advanced model diagnostic visualizations...")
            
            for model_name, results in all_results_plot.items():
                 
                 if results and results.get('y_proba') is not None and results.get('y_test') is not None and results.get('y_pred') is not None:
                     
                     y_test_current = results['y_test']
                     y_proba_current = results['y_proba']
                     current_classes = np.unique(y_test_current) 

                     if len(current_classes) > 1: 
                         try:
                             
                             
                             if y_proba_current.shape[1] >= len(unique_classes): 
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
                                 y_proba_current,
                                 model_name) 
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

            
            
            for model_name, results_dict in all_results_plot.items():
                 if 'Stacking Ensemble' in model_name: 
                     continue

                 model_instance = results_dict.get('model') 

                 if not model_instance:
                     logger.info(f"Skipping FI for {model_name}: model_instance (original model) is None in results_dict.")
                     continue
                 if not hasattr(model_instance, 'feature_importances_'):
                     logger.info(f"Skipping FI for {model_name}: model_instance (type: {type(model_instance)}) has no 'feature_importances_' attribute.")
                     continue
                 if 'feature_names' not in results_dict or not results_dict['feature_names']:
                     logger.info(f"Skipping FI for {model_name}: 'feature_names' missing or empty in results_dict.")
                     continue

                 try:
                     feature_importances = model_instance.feature_importances_
                     feature_names_list = results_dict['feature_names']
                     if len(feature_importances) == len(feature_names_list):
                         self.visualizer.plot_feature_importance(
                             model_instance, feature_names_list)
                     else:
                          logger.warning(f"Mismatch between feature importances length ({len(feature_importances)}) and feature names length ({len(feature_names_list)}) for {model_name}. Skipping FI plot.")
                 except Exception as fi_err:
                     logger.warning(f"Could not plot Feature Importance for {model_name}: {fi_err}")

            
            if all_results_plot:
                self.visualizer.plot_model_comparison(all_results_plot)
                self.visualizer.plot_confusion_matrices(all_results_plot)

            
            if processed_data is not None and not processed_data.empty:
                logger.info("Creating data-specific visualizations...")
                try:
                    self.visualizer.plot_funding_stage_distribution(processed_data, getattr(self, 'final_index_to_string_label_map', None))
                    self.visualizer.plot_funding_vs_employees(processed_data) 

                    
                    
                    potential_key_features = [
                        'funding_amount_log', 'employees', 'employee_efficiency',
                        'previous_rounds', 'months_since_first_funding',
                        'funding_year', 'funding_month',
                        'time_since_last_funding', 'funding_amount_ratio_vs_prev',
                        'funding_vs_industry_median'
                    ]
                    key_features = [f for f in potential_key_features if f in processed_data.columns]
                    if not key_features:
                        logger.warning("No key features found in processed_data for detailed visualizations.")
                    else:
                        logger.info(f"Using key features for plots: {key_features}")
                        
                        self.visualizer.plot_pairwise_features(processed_data, key_features) 
                        self.visualizer.plot_feature_distributions(processed_data, key_features)
                        
                        
                        if len(key_features) >=3: 
                            self.visualizer.plot_advanced_feature_correlations(processed_data, key_features)

                    self.visualizer.plot_correlation_heatmap(processed_data) 
                    self.visualizer.plot_full_correlation_heatmap(processed_data) 
                    self.visualizer.plot_temporal_trends(processed_data)
                    self.visualizer.plot_industry_distributions(processed_data)
                    self.visualizer.plot_funding_patterns(processed_data)
                    self.visualizer.plot_violin_funding_by_stage(processed_data)

                except Exception as data_viz_err:
                    logger.error(f"Error during data-specific visualizations: {data_viz_err}", exc_info=True)
            else:
                logger.warning("Skipping data-specific visualizations as processed_data is empty or None.")
            
            
            if processed_data is not None and not processed_data.empty and hasattr(self, 'final_index_to_string_label_map'):
                try:
                    logger.info("Creating visualization for top industries by funding stage...")
                    self.visualizer.plot_predicted_stages_by_industry(processed_data, self.final_index_to_string_label_map)
                    logger.info("Creating visualization for top N companies...")
                    self.visualizer.plot_top_n_companies(processed_data, n=10, column='funding_amount')
                except Exception as industry_plot_err:
                    logger.error(f"Could not generate top industries plot: {industry_plot_err}", exc_info=True)
            else:
                logger.warning("Skipping top industries plot: processed_data or final_index_to_string_label_map not available.")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            logger.error(traceback.format_exc()) 

    def _save_summary(self, merged_data, X, model_results_summary, processed_data):
        
        
        summary = {
            'run_timestamp': self.timestamp,
            'data_shape': {
                'initial_records_merged': len(merged_data),
                'records_after_preprocessing': X.shape[0] if hasattr(X, 'shape') else 'N/A',
                'features_used': X.shape[1] if hasattr(X, 'shape') else 'N/A'
            },
            'feature_names': self.feature_names,
             'class_mapping': {str(k): str(v) for k, v in getattr(self, 'final_index_to_string_label_map', {}).items()},
            
            'age_bin_edges': getattr(self.feature_engineer, 'age_bin_edges', None),
            'age_bin_labels': getattr(self.feature_engineer, 'age_bin_labels', None),
            
            'metrics': {}
        }

        
        
        
        

        
        best_accuracy = -1.0
        best_model_name = None
        for model_name, results in model_results_summary.items(): 
            if results is None: 
                 summary['metrics'][model_name] = None
                 continue
            
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
            
            if not pd.isna(accuracy) and accuracy > best_accuracy:
                 best_accuracy = accuracy
                 best_model_name = model_name

        
        summary['best_model_by_accuracy'] = best_model_name

        
        try:
             benchmarks = {}
             
             
             if 'funding_stage' in processed_data.columns and \
                'funding_stage_numeric' in processed_data.columns and \
                'funding_amount' in processed_data.columns and \
                'employees' in processed_data.columns:

                 
                 
                 
                 
                 grouped_by_string_label = processed_data.groupby('funding_stage')

                 for stage_label, group_data in grouped_by_string_label:
                     
                     stage_key = str(stage_label) 
                     benchmarks[stage_key] = {
                         'funding_amount_median': float(group_data['funding_amount'].median()) if pd.notna(group_data['funding_amount'].median()) else None,
                         'funding_amount_q1': float(group_data['funding_amount'].quantile(0.25)) if pd.notna(group_data['funding_amount'].quantile(0.25)) else None,
                         'funding_amount_q3': float(group_data['funding_amount'].quantile(0.75)) if pd.notna(group_data['funding_amount'].quantile(0.75)) else None,
                         'employees_median': float(group_data['employees'].median()) if pd.notna(group_data['employees'].median()) else None,
                         'employees_q1': float(group_data['employees'].quantile(0.25)) if pd.notna(group_data['employees'].quantile(0.25)) else None,
                         'employees_q3': float(group_data['employees'].quantile(0.75)) if pd.notna(group_data['employees'].quantile(0.75)) else None,
                         'count': int(len(group_data)) 
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
        


        
        
        best_base_model_importance = None
        importance_source_model = None
        feature_names_imp = self.feature_names 

        
        base_model_accuracies = {
             name: metrics['accuracy']
             for name, metrics in summary['metrics'].items()
             if metrics is not None and name != 'Stacking Ensemble (LR Meta)' and not np.isnan(metrics.get('accuracy', np.nan))
        }

        best_base_model_name = None
        if base_model_accuracies:
             
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

        
        best_overall_model_name = summary.get('best_model_by_accuracy')
        model_to_save = None
        scaler_to_save = None
        metadata_to_save = {}
        feature_names_to_save = self.feature_names 

        if best_overall_model_name and best_overall_model_name in model_results_summary:
            results_to_use = model_results_summary[best_overall_model_name]
            if results_to_use:
                metadata_to_save = summary['metrics'].get(best_overall_model_name, {})

                
                is_ensemble = 'Stacking Ensemble' in best_overall_model_name

                if is_ensemble:
                    model_to_save = results_to_use.get('model') 
                    feature_names_to_save = self.feature_names 
                    
                    scaler_found = False
                    
                    for base_key_lookup in [
                        'LightGBM', 
                        'XGBoost (Calibrated)',
                        'Gradient Boosting (Calibrated)',
                        'Random Forest (Calibrated)'
                        
                        
                        
                        
                        
                    ]:
                        if base_key_lookup in model_results_summary and model_results_summary[base_key_lookup]:
                            scaler_candidate = model_results_summary[base_key_lookup].get('scaler')
                            if scaler_candidate is not None:
                                scaler_to_save = scaler_candidate
                                logger.info(f"Using scaler from base model '{base_key_lookup}' for saving with ensemble.")
                                scaler_found = True
                                break 
                    if not scaler_found:
                         logger.error("Scaler not found from any base model for ensemble saving.")
                         scaler_to_save = None 
                else: 
                    model_to_save = results_to_use.get('calibrated_model') 
                    scaler_to_save = results_to_use.get('scaler')
                    feature_names_to_save = results_to_use.get('feature_names') 
                

                
                if not model_to_save: logger.error(f"Model object missing for {best_overall_model_name}")
                if not scaler_to_save: logger.error(f"Scaler object missing for {best_overall_model_name}")
                if not feature_names_to_save: logger.error(f"Feature names missing for {best_overall_model_name}")

            else: 
                 logger.error(f"Results dict missing for {best_overall_model_name}")
                 best_overall_model_name = None 
        else: 
            logger.error(f"Best overall model name '{best_overall_model_name}' invalid. Cannot save.")
            best_overall_model_name = None

        
        
        saved_path = None 
        if best_overall_model_name and model_to_save and scaler_to_save and feature_names_to_save:
            
            safe_model_name = best_overall_model_name.replace(" ", "_").replace("(", "").replace(")", "")
            
            anomaly_detector_to_save = results_to_use.get('anomaly_detector')
            if not anomaly_detector_to_save:
                logger.warning(f"Anomaly detector instance not found in results for {best_overall_model_name}. Model will be saved without it.")

            saved_path = self.model_manager.save_model(
                model_name=safe_model_name, 
                model=model_to_save,
                scaler=scaler_to_save,
                feature_names=feature_names_to_save,
                metadata={'training_metadata': metadata_to_save, 
                          'class_mapping': getattr(self, 'final_index_to_string_label_map', {})},
                anomaly_detector=anomaly_detector_to_save 
            )
            if saved_path:
                 logger.info(f"Successfully saved best model ({best_overall_model_name}) and scaler to {saved_path}")
                 
                 if not os.path.exists(saved_path):
                     logger.error(f"!!! FILE NOT FOUND AFTER SAVING: {saved_path} !!!")
                 else:
                     logger.info(f"Confirmed file exists: {saved_path}")
                 
            else:
                 logger.error(f"Failed to save the best model ({best_overall_model_name}).")
        else:
            logger.error(f"Could not save best model '{best_overall_model_name}' - model or scaler missing.")

        
        logger.info("Attempting to explicitly save key base models for dashboard use...")
        base_models_to_save = {
            'XGBoost (Calibrated)': 'XGBoost_(Calibrated)',
            'Random Forest (Calibrated)': 'Random_Forest_(Calibrated)',
            'Gradient Boosting (Calibrated)': 'Gradient_Boosting_(Calibrated)',
            'Decision Tree (Calibrated)': 'Decision_Tree_(Calibrated)' 
        }

        for model_key, save_name_suffix in base_models_to_save.items():
            
            if model_key in model_results_summary and model_results_summary[model_key]:
                results_data = model_results_summary[model_key]
                
                base_model_to_save = results_data.get('calibrated_model') 
                base_scaler_to_save = results_data.get('scaler')
                base_feature_names = results_data.get('feature_names')
                base_metrics = summary['metrics'].get(model_key, {}) 
                base_anomaly_detector = results_data.get('anomaly_detector') 
                
                if base_model_to_save and base_scaler_to_save and base_feature_names:
                    
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
                
                if model_key == 'Decision Tree (Calibrated)':
                    if model_key not in model_results_summary:
                         logger.warning(f"Results for base model {model_key} not found in summary dict. Cannot save explicitly.")
                    elif not model_results_summary[model_key]:
                         logger.warning(f"Results for base model {model_key} are None or invalid. Cannot save explicitly.")
                else: 
                    logger.warning(f"Results for base model {model_key} not found or invalid in summary. Cannot save explicitly.") 
        

        
        stacking_key = 'Stacking Ensemble (RF Meta)'
        stacking_save_name_suffix = 'Stacking_Ensemble_(RF_Meta)'
        if stacking_key in model_results_summary and model_results_summary[stacking_key]:
            logger.info(f"Attempting to explicitly save Stacking Ensemble ({stacking_key}) for dashboard use...")
            stacking_results_data = model_results_summary[stacking_key]
            stacking_model_to_save = stacking_results_data.get('model') 
            stacking_scaler_to_save = stacking_results_data.get('scaler') 
            stacking_feature_names_to_save = stacking_results_data.get('feature_names')

            
            best_overall_model_results_dict = None
            
            if 'best_overall_model_name' in locals() and best_overall_model_name and best_overall_model_name in model_results_summary:
                best_overall_model_results_dict = model_results_summary.get(best_overall_model_name)
            elif 'best_overall_model_name' in locals() and best_overall_model_name:
                 logger.warning(f"Results for best_overall_model_name ('{best_overall_model_name}') not found in model_results_summary. Cannot get default anomaly detector.")
            else:
                logger.warning("'best_overall_model_name' not defined or not available. Cannot get default anomaly detector.")

            default_anomaly_detector = None
            if best_overall_model_results_dict and isinstance(best_overall_model_results_dict, dict):
                default_anomaly_detector = best_overall_model_results_dict.get('anomaly_detector')
            elif best_overall_model_results_dict is not None: 
                 logger.warning(f"best_overall_model_results_dict for '{best_overall_model_name}' is not a valid dictionary. Default anomaly detector will be None.")
            

            if stacking_model_to_save and stacking_scaler_to_save and stacking_feature_names_to_save:
                logger.info("Constructing metadata for Stacking Ensemble...")
                
                logger.info(f"[DIAGNOSTIC] summary object type: {type(summary)}")
                if isinstance(summary, dict):
                    logger.info(f"[DIAGNOSTIC] summary keys: {list(summary.keys())}")
                    feature_eng_summary = summary.get('feature_engineering_summary')
                    logger.info(f"[DIAGNOSTIC] feature_engineering_summary type: {type(feature_eng_summary)}")
                    if isinstance(feature_eng_summary, dict):
                        logger.info(f"[DIAGNOSTIC] feature_engineering_summary keys: {list(feature_eng_summary.keys())}")
                        logger.info(f"[DIAGNOSTIC] age_bin_edges in feature_eng_summary: {feature_eng_summary.get('age_bin_edges')}")
                        logger.info(f"[DIAGNOSTIC] age_bin_labels in feature_eng_summary: {feature_eng_summary.get('age_bin_labels')}")
                
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
                logger.info(f"[DIAGNOSTIC] Constructed stacking_model_metadata: {stacking_model_metadata}") 
                
                logger.info(f"Stacking Ensemble metadata constructed. Anomaly detector source: {stacking_model_metadata.get('anomaly_detector_source')}")

                logger.info(f"Calling model_manager.save_model for Stacking Ensemble: {stacking_save_name_suffix}") 
                self.model_manager.save_model( 
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
        


        
        summary_path = os.path.join(self.output_dir, f"summary_{self.timestamp}.json")
        try:
            
            
            final_map_to_save = summary.get('class_mapping', {})
            logger.info(f"Final Class mapping being saved to summary (Index -> Label): {final_map_to_save}")
            
            with open(summary_path, 'w') as f:
                
                json.dump(summary, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Pipeline summary saved to {summary_path}")
        except Exception as json_err:
            logger.error(f"Failed to save summary JSON: {json_err}")
            logger.error(traceback.format_exc())
        

        return summary 

    def make_prediction(self, sample_data):
        """Make prediction with best available model"""
        
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

            
            safe_model_name = best_model_name.replace(" ", "_").replace("(", "").replace(")", "")
            logger.info(f"Attempting to load best model: {best_model_name} (Filename pattern: {safe_model_name})")

        except Exception as e:
            logger.error(f"Error reading latest summary file: {e}")
            return {'error': 'Failed to read summary file'}


        
        
        loaded_ok = self.model_manager.load_model_joblib(model_name=safe_model_name)
        if not loaded_ok:
            
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

        
        if self.model_manager.scaler is None or not self.model_manager.feature_names:
            logger.error("Scaler or feature names not loaded with the model.")
            return {'error': 'Model loaded without scaler/feature names'}

        
        try:
            if isinstance(sample_data, dict):
                
                feature_columns = self.model_manager.feature_names
                
                ordered_data = pd.Series(index=feature_columns, dtype=float)
                missing_input_features = []
                for col in feature_columns:
                     if col in sample_data:
                         ordered_data[col] = sample_data[col]
                     else:
                         ordered_data[col] = 0 
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

        
        
        
        
        company_name_input = sample_data.get('company_name', None) if isinstance(sample_data, dict) else None
        prediction_result = self.model_manager.predict(features_df.iloc[0].to_dict(), company_name=company_name_input)

        
        
        
        reverse_map = getattr(self, 'reverse_final_class_mapping', None)
        if not reverse_map and latest_summary_path:
             try:
                 with open(latest_summary_path, 'r') as f:
                     summary_data = json.load(f)
                 
                 final_map_from_summary = {int(k): v for k, v in summary_data.get('class_mapping', {}).items()}
                 reverse_map = {v: k for k, v in final_map_from_summary.items()} 
             except Exception as map_load_err:
                 logger.error(f"Could not load class mapping from summary for label conversion: {map_load_err}")


        if 'prediction' in prediction_result and reverse_map:
             predicted_class_index = prediction_result['prediction']
             
             predicted_stage_numeric = int(predicted_class_index)
             
             predicted_stage_label = reverse_map.get(predicted_stage_numeric, f"Unknown Stage (Index {predicted_stage_numeric})")

             prediction_result['predicted_stage_label'] = predicted_stage_label
             prediction_result['predicted_stage_numeric'] = predicted_stage_numeric 
        elif 'error' not in prediction_result:
             logger.warning("Could not map prediction index back to stage label (mapping missing?).")
             prediction_result['predicted_stage_label'] = f"Unknown (Class {prediction_result.get('prediction')})"


        return prediction_result


class AnomalyDetector:
    """Detects anomalies and potential manipulation in startup data"""

    def __init__(self, contamination=0.01): 
        """Initialize detector with contamination parameter (expected outlier ratio)"""
        
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("IsolationForest not installed. Skipping anomaly detection.")
            self.isolation_forest = None 
            self.contamination = contamination 
            self.feature_ranges = {}
            self.startup_data_cache = {}
            self.known_companies = set()
            return 

        
        self.isolation_forest = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
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
        
        if self.isolation_forest is None:
             logger.error("IsolationForest not initialized, cannot fit.")
             return False
        try:
            
            self.isolation_forest.fit(X) 

            
            self.feature_ranges = {
                'min': np.min(X, axis=0),
                'max': np.max(X, axis=0),
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0),
                'q1': np.percentile(X, 25, axis=0),
                'q3': np.percentile(X, 75, axis=0)
            }

            
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
            
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            
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

            
            anomalies = {
                'is_anomaly': False, 
                'score': 0.0,        
                'reasons': []        
            }

            
            if self.isolation_forest is not None:
                
                scores = self.isolation_forest.decision_function(X)
                
                predictions = self.isolation_forest.predict(X)

                
                min_score = np.min(scores)
                
                
                if min_score < threshold or np.any(predictions == -1):
                    anomalies['is_anomaly'] = True
                    
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

class TimeSeriesForecaster:
    def __init__(self, output_dir="./visualizations"):
        """Initialize time series forecaster"""
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_prophet_data(self, data):
        """Aggregate data monthly and prepare for Prophet"""
        logger.info("Preparing data for Prophet...")
        
        if not pd.api.types.is_datetime64_any_dtype(data['funding_date']):
             data['funding_date'] = pd.to_datetime(data['funding_date'], errors='coerce')

        
        data['funding_amount_log'] = pd.to_numeric(data['funding_amount_log'], errors='coerce')

        
        data = data.dropna(subset=['funding_date', 'funding_stage_numeric', 'funding_amount_log'])
        if data.empty:
             logger.warning("No valid date data found for time series analysis.")
             return None

        
        
        monthly_data = data.set_index('funding_date')

        
        
        
        
        
        

        
        raw_quarterly_counts = monthly_data.resample('QS').size() 
        logger.info(f"Raw quarterly deal counts (Full History, before aggregation/filtering):\n{raw_quarterly_counts.to_string()}")
        

        
        window_size = 4 
        
        numeric_cols_for_resample = ['funding_stage_numeric', 'funding_amount_log']
        
        quarterly_resampled = monthly_data[numeric_cols_for_resample].resample('QS').median()

        
        quarterly_agg = pd.DataFrame(index=quarterly_resampled.index)
        
        quarterly_agg['y'] = quarterly_resampled['funding_stage_numeric'].ewm(span=4, adjust=False).mean() 
        
        

        
        quarterly_counts = monthly_data.resample('QS').size()
        quarterly_agg['deal_count'] = quarterly_counts.rolling(window=window_size, min_periods=1).sum()

        
        
        quarterly_funding_log = monthly_data['funding_amount_log'].resample('QS').median()
        
        quarterly_agg['median_funding_log'] = quarterly_funding_log.ewm(span=4, adjust=False).mean() 


        
        prophet_df = quarterly_agg.reset_index()
        prophet_df = prophet_df.rename(columns={'funding_date': 'ds'})
        

        
        prophet_df = prophet_df.dropna()

        if prophet_df.empty: 
            logger.warning("No valid quarterly data points remaining after aggregation and dropna.")
            return None

        logger.info(f"Prepared {len(prophet_df)} quarterly data points with regressors for Prophet.")
        return prophet_df

    def train_predict(self, prophet_df, periods=6, freq='MS'):
        """Train Prophet model and make future predictions"""
        if prophet_df is None or prophet_df.empty:
             logger.error("Cannot train Prophet model: Input data is empty.")
             return None, None 

        
        required_regressors = ['median_funding_log', 'deal_count']
        missing_regressors = [reg for reg in required_regressors if reg not in prophet_df.columns]
        if missing_regressors:
             logger.error(f"Missing required regressor columns in Prophet data: {missing_regressors}")
             return None, None
 
        
        min_data_points = 4 
        if len(prophet_df) < min_data_points:
            logger.warning(f"Insufficient historical data ({len(prophet_df)} quarters) for reliable Prophet forecast. Minimum required: {min_data_points}. Skipping forecast.")
            return None, None

        
        forecast_periods = 2 
        forecast_freq = 'QS' 

        logger.info(f"Training Prophet model to forecast {forecast_periods} quarters ({forecast_freq})...") 
        try:
            
            
            n_points = len(prophet_df)
            n_changepoints = min(25, n_points - 1) if n_points > 1 else 0 

            model = Prophet(
                n_changepoints=n_changepoints, 
                
                changepoint_prior_scale=0.01, 
                seasonality_prior_scale=1.0,  
                yearly_seasonality=True,      
                
                weekly_seasonality=False, 
                daily_seasonality=False,
                interval_width=0.95 
             )

            
            model.add_regressor('median_funding_log')
            model.add_regressor('deal_count')

            model.fit(prophet_df)
 
            
            future = model.make_future_dataframe(periods=forecast_periods, freq=forecast_freq) 
 
            
            
            
            
            last_months = 3
            future_median_funding_log = prophet_df['median_funding_log'].iloc[-last_months:].mean()
            future_deal_count = prophet_df['deal_count'].iloc[-last_months:].mean()

            
            future['median_funding_log'] = future_median_funding_log
            future['deal_count'] = future_deal_count

            
            future.loc[future['ds'].isin(prophet_df['ds']), 'median_funding_log'] = prophet_df['median_funding_log']
            future.loc[future['ds'].isin(prophet_df['ds']), 'deal_count'] = prophet_df['deal_count']

            
            future['median_funding_log'] = future['median_funding_log'].fillna(prophet_df['median_funding_log'].median()) 
            future['deal_count'] = future['deal_count'].fillna(prophet_df['deal_count'].median()) 

            
            forecast = model.predict(future)

            
            forecast['yhat'] = forecast['yhat'].clip(lower=0.0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0.0)
            

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
            
            fig = plt.figure(figsize=(15, 6)) 
            ax = fig.add_subplot(111)
            model.plot(forecast, ax=ax) 

            plt.title('Funding Stage Trend Forecast (Median Stage) - 6 Months', fontsize=14) 
            plt.xlabel('Date')
            plt.ylabel('Median Funding Stage (Numeric)')
            plt.grid(True, linestyle='--', alpha=0.7)

            
            last_hist_date = model.history_dates.max()
            plt.axvline(last_hist_date, color='r', linestyle='--', lw=1, label='Forecast Start')
            plt.legend()

            plot_path = os.path.join(self.output_dir, f"prophet_forecast_{self.timestamp}.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            logger.info(f"Prophet forecast plot saved to {plot_path}")
            plt.close(fig) 

            
            
            
            
            
            

        except Exception as e:
            logger.error(f"Error generating Prophet plot: {e}")
            logger.error(traceback.format_exc())
            plt.close() 

    def plot_dashboard_prototype(self, forecast, history_df):
        """Plot historical actuals and forecast (if provided) for the dashboard prototype as an interactive HTML."""
        
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error("Plotly not installed. Cannot create interactive plot. `pip install plotly`")
            return 

        if history_df is None or history_df.empty:
             logger.error("Cannot plot dashboard prototype: Historical data missing for comparison.")
             return

        logger.info("Generating interactive dashboard prototype plot (HTML)...") 
        prototype_output_dir = os.path.join(self.output_dir, "..", "prototype_dashboard") 
        os.makedirs(prototype_output_dir, exist_ok=True)

        try:
            fig = go.Figure()

            
            fig.add_trace(go.Scatter(
                x=history_df['ds'],
                y=history_df['y'],
                mode='lines+markers',
                name='Historical EWMA Stage (4-Qtr Span)', 
                line=dict(color='black'),
                marker=dict(size=4)
            ))

            plot_title = 'Bay Area Startups: Historical EWMA Funding Stage (4-Quarter Span)' 

            
            if forecast is not None and not forecast.empty:
                
                plot_df = history_df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')

                
                fig.add_trace(go.Scatter(
                    x=plot_df['ds'],
                    y=plot_df['yhat'],
                    mode='lines',
                    name='Model Fit (Historical)',
                    line=dict(color='red', dash='dash')
                ))

                
                last_hist_date = history_df['ds'].max()
                forecast_future = forecast[forecast['ds'] > last_hist_date]

                
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat'],
                    mode='lines+markers',
                    name='Forecast Median Stage (2 Qtrs)', 
                    line=dict(color='blue'),
                    marker=dict(size=4)
                ))

                
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat_upper'],
                    mode='lines',
                    line=dict(width=0), 
                    showlegend=False 
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat_lower'],
                    mode='lines',
                    line=dict(width=0), 
                    fillcolor='rgba(0, 0, 255, 0.2)', 
                    fill='tonexty', 
                    name='95% Confidence Interval'
                ))

                plot_title = 'Funding Stage Trend & Forecast' 
            

            
            fig.update_layout(
                title=plot_title,
                xaxis_title='Date (Quarter Start)',
                yaxis_title='Median Funding Stage (Numeric)',
                hovermode="x unified", 
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(0,0,0,0)'), 
                margin=dict(l=40, r=40, t=80, b=40) 
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')


            
            plot_path_html = os.path.join(prototype_output_dir, f"bay_area_funding_trend_interactive_{self.timestamp}.html") 
            fig.write_html(plot_path_html)

            logger.info(f"Interactive dashboard prototype plot saved to {plot_path_html}")
            

        except Exception as e:
            logger.error(f"Error generating interactive dashboard prototype plot: {e}")
            logger.error(traceback.format_exc())
            


def main():
    """Main entry point for the funding stage prediction pipeline"""
    import argparse
    import schedule
    import time
    from datetime import datetime, timedelta
    
    
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
        default='./cs163-main/backend/ADataCollection/JSONFolder',  
        help='Base directory for data files')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./cs163-main/backend/MainOutput',  
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

    
    
    if args.base_dir == './': 
        
        
        logger.warning("Base directory was './', attempting to use 'cs163-main/backend/ADataCollection/JSONFolder'. Verify if this is intended.")
        args.base_dir = './cs163-main/backend/ADataCollection/JSONFolder'

    logger.info(f"Starting with data directory: {args.base_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    
    pipeline = EnhancedPipeline(
        args.base_dir,
        args.output_dir,
        archive=args.archive)

    if args.reset_db:
        pipeline.data_loader.reset_database()

    
    def scheduled_job():
        logger.info(f"Running scheduled job at {datetime.now()}")
        success = pipeline.run()
        if not success:
            logger.error("Scheduled job failed")

    
    logger.info(f"Running funding prediction job at {datetime.now()}")
    success = pipeline.run() 

    if not success:
        logger.error("Initial run failed - check logs for details")
        return

    
    
    if args.once:
        logger.info("Job completed - exiting")
    else:
        
        interval_hours = args.interval

        
        
        if args.start_time:
            
            schedule.every().day.at(args.start_time).do(scheduled_job)
            logger.info(f"Scheduled to run daily at {args.start_time}")

            
            hour, minute = map(int, args.start_time.split(':'))
            next_run = datetime.now().replace(hour=hour, minute=minute)
            if next_run < datetime.now():
                next_run += timedelta(days=1)
        else:
            
            schedule.every(interval_hours).hours.do(scheduled_job)
            logger.info(f"Scheduled to run every {interval_hours} hours")
            next_run = datetime.now() + timedelta(hours=interval_hours)

        logger.info(f"Next scheduled run: {next_run}")


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
            return None 
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    main()
