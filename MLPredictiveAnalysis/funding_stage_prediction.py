import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')  # Use Agg backend which doesn't require a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, auc)
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import label_binarize
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import sqlite3
import traceback
import glob
import shutil
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("funding_prediction.log"),
                            logging.StreamHandler()])
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, base_dir="./", archive=False):
        """Initialize data loader with paths to data sources and historical database"""
        self.base_dir = base_dir
        self.archive = archive
        self.archive_dir = None
        self.historical_db = os.path.join(base_dir, "historical_funding_data.db")
        
        # Define paths to source files in JSONFolder - fix for duplicated path
        # If base_dir already contains JSONFolder, don't add it again
        if os.path.basename(base_dir) == "JSONFolder" or os.path.exists(os.path.join(base_dir, "fundraisestartup50.json")):
            self.json_folder = base_dir
        else:
            self.json_folder = os.path.join(base_dir, "JSONFolder")
        
        # Use the fixed json_folder path for file paths
        self.fundraiser_path = os.path.join(self.json_folder, "fundraisestartup50.json")
        self.growthlist_path = os.path.join(self.json_folder, "growthlistscrapper.json")
        self.topstartup_path = os.path.join(self.json_folder, "topstartupio50.json")
        
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
                shutil.copy2(self.fundraiser_path, os.path.join(self.archive_dir, "fundraiser.json"))
            # Archive growthlist data
            if os.path.isfile(self.growthlist_path):
                shutil.copy2(self.growthlist_path, os.path.join(self.archive_dir, "growthlist.json"))
            # Archive topstartup data
            if os.path.isfile(self.topstartup_path):
                shutil.copy2(self.topstartup_path, os.path.join(self.archive_dir, "topstartup.json"))
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
            'fundraiser_data': ['Company', 'Funding_Amount_USD'],
            'growthlist_data': ['name'],
            'topstartup_data': ['company_name', 'funding_stage']
        }
        
        if source_name in required_columns:
            missing_cols = [col for col in required_columns[source_name] 
                           if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {source_name}: {missing_cols}")
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
                df['Funding_Amount_USD'] = pd.to_numeric(df['Funding_Amount_USD'], errors='coerce')
            
            if 'Total_Employees' in df.columns:
                df['Total_Employees'] = pd.to_numeric(df['Total_Employees'], errors='coerce')
                
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
                df['funding_amount_numeric'] = df['funding_amount'].apply(self._parse_funding_amount)
            
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
                    # or "Andreessen Horowitz $10B Series J in 2024 $62.0B valuation"
                    
                    amount = None
                    stage = None
                    date = None
                    
                    # Extract amount
                    amount_match = re.search(r'\$(\d+(?:\.\d+)?[KMB]?)', funding_str)
                    if amount_match:
                        amount = amount_match.group(0)  # Keep the $ symbol
                    
                    # Extract stage
                    stage_match = re.search(r'(Seed|Series [A-Z]|Pre-Seed|Angel)', funding_str)
                    if stage_match:
                        stage = stage_match.group(0)
                    
                    # Extract date - usually has "in YYYY" format
                    date_match = re.search(r'in (\d{4})', funding_str)
                    if date_match:
                        date = date_match.group(1)
                    
                    return amount, stage, date
                
                # Extract funding details
                funding_details = df['funding'].apply(extract_funding_info)
                
                # Create separate columns for extracted values
                df['funding_amount'] = funding_details.apply(lambda x: x[0] if x else None)
                df['funding_stage'] = funding_details.apply(lambda x: x[1] if x else None)
                df['funding_date'] = funding_details.apply(lambda x: x[2] if x else None)
            
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
        """Improved function to convert funding amount strings to numeric values"""
        if not amount_str or pd.isna(amount_str) or amount_str == "":
            return np.nan
        
        try:
            # Remove currency symbol and commas
            amount_str = str(amount_str).replace('$', '').replace(',', '').strip()
            
            # Handle extremely large numbers (like 50000000000)
            # If number is extremely large, it might be an error
            if amount_str.isdigit() and len(amount_str) > 11:  # More than 100B
                logger.warning(f"Suspiciously large funding amount: ${amount_str}")
                # Take it at face value but log the warning
            
            # Convert based on unit (M=million, B=billion, K=thousand)
            if 'B' in amount_str:
                return float(amount_str.replace('B', '')) * 1e9
            elif 'M' in amount_str:
                return float(amount_str.replace('M', '')) * 1e6
            elif 'K' in amount_str:
                return float(amount_str.replace('K', '')) * 1e3
            else:
                return float(amount_str)
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
            logger.info(f"Loaded {len(df)} historical records from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def merge_datasets(self):
        """Improved merge function with better error handling and validation"""
        try:
            # Load raw data
            fundraiser_df = self.load_fundraiser_data()
            growthlist_df = self.load_growthlist_data()
            topstartup_df = self.load_topstartup_data()
            
            # Log loaded data sizes
            logger.info(f"Loaded datasets - Fundraiser: {len(fundraiser_df)} rows, Growthlist: {len(growthlist_df)} rows, Topstartup: {len(topstartup_df)} rows")
            
            # Initialize list to store all records
            all_records = []
            
            # Process fundraiser data
            if not fundraiser_df.empty:
                for _, row in fundraiser_df.iterrows():
                    if pd.notna(row.get('Company')):  # Only add records with valid company names
                        # Convert numeric values properly
                        try:
                            funding_amount = pd.to_numeric(row.get('Funding_Amount_USD'), errors='coerce')
                            employees = pd.to_numeric(row.get('Total_Employees'), errors='coerce')
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
                            'source': 'fundraiser'
                        })
                logger.info(f"Processed {len(fundraiser_df)} records from fundraiser data")
            
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
                            'funding_stage': row.get('funding_type'),
                            'funding_amount': funding_amount,
                            'funding_date': row.get('last_funding_date'),
                            'industry': row.get('industry'),
                            'employees': None,
                            'source': 'growthlist'
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
                            'industry': row.get('industry') or row.get('category'),
                            'employees': employee_count,
                            'source': 'topstartup'
                        })
                logger.info(f"Processed {len(topstartup_df)} records from topstartup data")
            
            if all_records:
                # Create DataFrame from records
                merged_data = pd.DataFrame(all_records)
                
                # Log column counts to debug missing data
                logger.info(f"Merged data columns: {merged_data.columns.tolist()}")
                logger.info(f"Non-null counts: {merged_data.count().to_dict()}")
                
                # Drop duplicates after creation
                pre_dedup_count = len(merged_data)
                merged_data = merged_data.drop_duplicates(
                    subset=['company_name', 'funding_date']
                ).reset_index(drop=True)
                logger.info(f"Removed {pre_dedup_count - len(merged_data)} duplicate records")
                
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
                        lambda x: stage_mapping.get(x, x) if pd.notna(x) else 'Unknown'
                    )
                
                # Fill missing values with meaningful defaults
                merged_data = merged_data.fillna({
                    'industry': 'Unknown',
                    'employees': 0,
                    'funding_stage': 'Unknown'
                })
                
                logger.info(f"Successfully merged {len(merged_data)} records")
                return merged_data
            
            logger.warning("No records available after merging")
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
        logger.info(f"Found {len(valid_stages)} unique funding stages: {valid_stages}")
        
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
            data['funding_date'] = pd.to_datetime(data['funding_date'], format='mixed', errors='coerce')
            
            # If a lot of dates failed to parse, try more specific formats
            if data['funding_date'].isna().sum() > len(data) * 0.3:
                logger.warning(f"Many dates failed to parse ({data['funding_date'].isna().sum()} NaN values)")
                
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
                            logger.info(f"Format '{fmt}' parsed {na_before - na_after} dates")
                    except:
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
            company_first_funding = data.groupby('company_name')['funding_date'].min()
            data['company_first_funding'] = data['company_name'].map(company_first_funding)
            
            # Calculate months between dates, handling NaT values
            def safe_month_diff(end_date, start_date):
                if pd.isna(end_date) or pd.isna(start_date):
                    return 0
                try:
                    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                except:
                    return 0
                    
            data['months_since_first_funding'] = data.apply(
                lambda row: safe_month_diff(row['funding_date'], row['company_first_funding']), 
                axis=1
            )
            
        except Exception as e:
            logger.warning(f"Error processing dates: {e}")
            # Set default values if date processing fails
            data['funding_year'] = datetime.now().year
            data['funding_month'] = datetime.now().month
            data['months_since_first_funding'] = 0
        
        # Log transform funding amount (handle skewed distribution)
        # First ensure it's numeric
        data['funding_amount'] = pd.to_numeric(data['funding_amount'], errors='coerce')
        
        # Check for extremely large values that might be errors (>$100B)
        if (data['funding_amount'] > 1e11).any():
            large_values = data[data['funding_amount'] > 1e11]
            logger.warning(f"Found {len(large_values)} extremely large funding amounts (>$100B)")
            logger.warning(f"Sample: {large_values[['company_name', 'funding_amount', 'source']].head().to_dict()}")
            
            # Cap extremely large values
            data['funding_amount'] = data['funding_amount'].clip(upper=1e11)
            
        # Apply log transform with offset to handle zeros
        data['funding_amount_log'] = np.log1p(data['funding_amount'].fillna(0))
        
        # Employee efficiency (funding per employee)
        if 'employees' in data.columns:
            data['employees'] = pd.to_numeric(data['employees'], errors='coerce')
            
            # Avoid division by zero
            data['employee_efficiency'] = data.apply(
                lambda row: row['funding_amount'] / row['employees'] if row['employees'] > 0 else np.nan,
                axis=1
            )
            
            # Fill missing values with median by funding stage
            efficiency_medians = data.groupby('funding_stage')['employee_efficiency'].median()
            
            for stage in data['funding_stage'].unique():
                stage_median = efficiency_medians.get(stage, data['employee_efficiency'].median())
                mask = (data['funding_stage'] == stage) & (data['employee_efficiency'].isna())
                data.loc[mask, 'employee_efficiency'] = stage_median
            
            # Fill any remaining NaNs with overall median
            data['employee_efficiency'] = data['employee_efficiency'].fillna(data['employee_efficiency'].median())
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
        
        data['industry_category'] = data['industry_category'].apply(map_industry)
        
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
            
            data['location_category'] = data['location_category'].apply(extract_location)
            
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
        data['previous_rounds'] = data['company_name'].map(company_funding_counts) - 1
        data['previous_rounds'] = data['previous_rounds'].clip(lower=0)
        
        # New feature: Funding velocity (average months between rounds)
        company_funding_dates = data.groupby('company_name')['funding_date'].apply(list)
        
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
                interval = (valid_dates[i].year - valid_dates[i-1].year) * 12 + \
                          (valid_dates[i].month - valid_dates[i-1].month)
                intervals.append(interval)
            
            return np.mean(intervals) if intervals else np.nan
        
        data['funding_velocity'] = data['company_name'].map(
            company_funding_dates.apply(calc_funding_velocity)
        )
        
        # Fill missing values for all numeric columns
        numeric_cols = [
            'funding_amount', 'funding_amount_log', 'employees', 'employee_efficiency',
            'previous_rounds', 'funding_velocity'
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
        
        logger.info(f"Feature engineering complete: {data.shape[1]} features created")
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
        
        # Check we have enough data
        if valid_mask.sum() < 10:
            logger.warning(f"Very few valid target values: {valid_mask.sum()} out of {len(y)}")
        
        X = X[valid_mask]
        y = y[valid_mask].astype(int)
        
        # Log data shapes and class distribution
        logger.info(f"Prepared model data: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y


class ModelTrainer:
    def __init__(self, output_dir="./models"):
        """Initialize with output directory for saving models"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def train_random_forest(self, X, y):
        """Train a Random Forest model for funding stage prediction"""
        logger.info("Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define model with hyperparameters
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"Random Forest accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{report}")
        
        # Save model
        model_path = os.path.join(self.output_dir, f"random_forest_{self.timestamp}.joblib")
        joblib.dump(rf, model_path)
        
        # Return model and evaluation data
        return rf, {
            'accuracy': accuracy,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_names': X.columns.tolist(),
            'model_path': model_path
        }
    
    def train_xgboost(self, X, y):
        """Train an XGBoost model for funding stage prediction"""
        logger.info("Training XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Ensure continuous non-negative class labels starting from 0
        unique_labels = sorted(set(y))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_train_mapped = pd.Series(y_train).map(label_map)
        y_test_mapped = pd.Series(y_test).map(label_map)
        
        # Define model with updated number of classes
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(unique_labels),
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Train model
        xgb_model.fit(X_train, y_train_mapped)
        
        # Evaluate
        y_pred_mapped = xgb_model.predict(X_test)
        
        # Map predictions back to original labels for evaluation
        reverse_map = {v: k for k, v in label_map.items()}
        y_pred = pd.Series(y_pred_mapped).map(reverse_map)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"XGBoost accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{report}")
        
        # Save model and mappings
        model_path = os.path.join(self.output_dir, f"xgboost_{self.timestamp}.joblib")
        joblib.dump({
            'model': xgb_model, 
            'label_map': label_map,
            'reverse_map': reverse_map
        }, model_path)
        
        return xgb_model, {
            'accuracy': accuracy,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_names': X.columns.tolist(),
            'model_path': model_path,
            'label_map': label_map
        }

class EnhancedModelTrainer(ModelTrainer):
    def tune_random_forest(self, X, y):
        """Hyperparameter tuning for Random Forest"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        logger.info(f"Best RF params: {grid_search.best_params_}")
        logger.info(f"Best RF accuracy: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    def tune_xgboost(self, X, y):
        """Handle dynamic class counts in XGBoost"""
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        
        param_dist = {
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'num_class': [num_classes]  # Critical fix
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            n_estimators=200,
            random_state=42
        )
        
        random_search = RandomizedSearchCV(
            xgb_model, param_dist, n_iter=25,
            cv=3, scoring='accuracy', n_jobs=-1,
            error_score='raise'  # Get detailed error reports
        )
        random_search.fit(X, y)
        
        logger.info(f"Best XGB params: {random_search.best_params_}")
        logger.info(f"Best XGB accuracy: {random_search.best_score_:.4f}")
        return random_search.best_estimator_

class ModelManager:
    """Manages model loading, versioning and prediction"""
    def __init__(self, models_dir="./models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.loaded_models = {}

    def load_model(self, model_path=None, model_type="latest"):
        """Load a specific model or latest version"""
        if model_path:
            model = joblib.load(model_path)
            return model

        # Find latest model of specified type
        model_files = glob.glob(os.path.join(self.models_dir, f"{model_type}_*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No {model_type} models found in {self.models_dir}")

        # Sort by timestamp in filename
        latest_model = sorted(model_files)[-1]

        logger.info(f"Loading model: {latest_model}")
        model = joblib.load(latest_model)
        self.loaded_models[model_type] = model
        return model

    def save_model(self, model, model_type, metadata=None):
        """Save model with versioning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"{model_type}_{timestamp}.joblib")

        # Save model with metadata
        model_data = {
            'model': model,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'type': model_type
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Saved {model_type} model to {model_path}")
        return model_path

    def predict(self, features, model_type="ensemble"):
        """Make prediction using loaded model"""
        if model_type not in self.loaded_models:
            self.load_model(model_type=model_type)

        model_data = self.loaded_models[model_type]

        # Handle different model storage formats
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data

        # Handle different model types
        if model_type == 'xgboost' and hasattr(model, 'predict_proba'):
            return model.predict_proba([features])[0]

        return model.predict([features])[0]

    def predict_proba(self, features, model_type="ensemble"):
        """Return prediction probabilities using loaded model"""
        if model_type not in self.loaded_models:
            self.load_model(model_type=model_type)
        model_data = self.loaded_models[model_type]
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
        if hasattr(model, 'predict_proba'):
            return model.predict_proba([features])[0]
        else:
            return None

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
            self.output_dir = "./outputFundingStagePrediction/visualizations"
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
        plt.savefig(os.path.join(self.output_dir, f"funding_stage_dist_{self.timestamp}.png"))
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
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.title('Top Feature Importances', fontsize=14)
            plt.xlabel('Relative Importance')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 
                                     f"feature_importance_{type(model).__name__}_{self.timestamp}.png"))
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
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"model_comparison_{self.timestamp}.png"))
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
        plt.savefig(os.path.join(self.output_dir, f"confusion_matrices_{self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()
    
    def plot_funding_vs_employees(self, data):
        """Visualize relationship between funding amount and employee count"""
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        plot_data = data[['funding_amount', 'employees', 'funding_stage_numeric']].dropna()
        
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
        plt.savefig(os.path.join(self.output_dir, f"funding_vs_employees_{self.timestamp}.png"))
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
        plt.savefig(os.path.join(self.output_dir, f"feature_matrix_{self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_correlation_heatmap(self, data):
        """Visualize feature correlations with funding stage"""
        plt.figure(figsize=(15, 12))
        corr = data.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title("Feature Correlation Heatmap")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"correlation_heatmap_{self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_temporal_trends(self, data):
        """Analyze funding trends over time with industry breakdown"""
        plt.figure(figsize=(18, 8))
        
        plt.subplot(1, 2, 1)
        sns.lineplot(data=data, x='funding_year', y='funding_amount', 
                    hue='industry_category', estimator='median', errorbar=None)
        plt.title('Median Funding Amount by Year')
        plt.ylabel('USD (log scale)')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        funding_counts = data.groupby(['funding_year', 'industry_category']).size().reset_index()
        sns.lineplot(data=funding_counts, x='funding_year', y=0, hue='industry_category')
        plt.title('Funding Round Frequency by Year')
        plt.ylabel('Number of Rounds')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"temporal_trends_{self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_industry_distributions(self, data):
        """Compare funding patterns across industries"""
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=data, x='industry_category', y='funding_amount', showfliers=False)
        plt.yscale('log')
        plt.title('Funding Amount Distribution by Industry')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        stage_dist = data.groupby(['industry_category', 'funding_stage']).size().unstack()
        stage_dist.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Funding Stage Distribution by Industry')
        plt.ylabel('Number of Companies')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"industry_analysis_{self.timestamp}.png"))
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
        plt.savefig(os.path.join(self.output_dir, 
                                f"advanced_correlations_{self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_feature_distributions(self, data, features):
        """Plot detailed feature distributions"""
        n_features = len(features)
        fig = plt.figure(figsize=(15, n_features * 3))
        
        for idx, feature in enumerate(features, 1):
            # Distribution plot
            plt.subplot(n_features, 2, 2*idx-1)
            sns.histplot(data=data, x=feature, hue='funding_stage',
                        multiple="stack", palette=self.feature_palettes['categorical'])
            plt.title(f'{feature} Distribution by Funding Stage')
            plt.xticks(rotation=45)
            
            # Box plot
            plt.subplot(n_features, 2, 2*idx)
            sns.boxplot(data=data, y=feature, x='funding_stage',
                       palette=self.feature_palettes['sequential'])
            plt.xticks(rotation=45)
            plt.title(f'{feature} Range by Funding Stage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                                f"feature_distributions_{self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_funding_patterns(self, data):
        """Visualize complex funding patterns with interactive display"""
        plt.figure(figsize=(20, 10))

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
        industry_funding = data.groupby('industry_category')['funding_amount'].sum()
        industry_funding.sort_values(ascending=True).plot(kind='barh',
            color=self.feature_palettes['sequential'])
        plt.title('Total Funding by Industry')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                                f"funding_patterns_{self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_pairwise_features(self, data, features):
        """Plot pairwise feature relationships (scatter matrix)"""
        sns.set(style="ticks")
        pairplot = sns.pairplot(data[features + ['funding_stage']], hue='funding_stage', palette='tab10', diag_kind='kde')
        plt.suptitle('Pairwise Feature Relationships', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"pairwise_features_{self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_full_correlation_heatmap(self, data):
        """Plot a full correlation heatmap for all numeric features"""
        plt.figure(figsize=(18, 14))
        corr = data.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title("Full Feature Correlation Heatmap")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"full_correlation_heatmap_{self.timestamp}.png"))
        if self.interactive:
            plt.show()
        plt.close()

    def plot_violin_funding_by_stage(self, data):
        """Plot violin plot of funding amount by funding stage"""
        plt.figure(figsize=(14, 8))
        sns.violinplot(data=data, x='funding_stage', y='funding_amount', scale='width', inner='quartile', palette='Set2')
        plt.yscale('log')
        plt.title('Funding Amount Distribution by Stage (Violin Plot)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"violin_funding_by_stage_{self.timestamp}.png"))
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
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        colors = plt.colormaps['tab10'](np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve (class {classes[i]}, AUC = {roc_auc[i]:0.2f})')
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
                logger.warning(f"Skipping calibration for class {class_idx}: {str(e)}")
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
        for i in range(len(bins)-1):
            mask = (confidence >= bins[i]) & (confidence < bins[i+1])
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
            processed_data = self.feature_engineer.extract_features(merged_data)
            
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
            
            self.visualizer.plot_feature_comparison_matrix(processed_data, key_features)
            self.visualizer.plot_correlation_heatmap(processed_data)
            self.visualizer.plot_temporal_trends(processed_data)
            self.visualizer.plot_industry_distributions(processed_data)
            self.visualizer.plot_advanced_feature_correlations(processed_data, key_features)
            self.visualizer.plot_feature_distributions(processed_data, key_features)
            self.visualizer.plot_funding_patterns(processed_data)
            # --- NEW VISUALIZATIONS ---
            self.visualizer.plot_pairwise_features(processed_data, key_features)
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
        logger.info(f"Scheduler started. Will run every {interval_hours} hours")
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
            os.makedirs(os.path.join(self.models_dir, model_type), exist_ok=True)

        # Create evaluation directory for model performance metrics
        os.makedirs(os.path.join(self.models_dir, 'evaluation'), exist_ok=True)

        logger.info(f"Initialized model directory structure at {self.models_dir}")

class EnhancedPipeline(FundingStagePredictionPipeline):
    def __init__(self, *args, **kwargs):
        # Override the output_dir with our custom path
        if 'output_dir' in kwargs:
            kwargs['output_dir'] = './outputFundingStagePrediction'
        else:
            args = list(args)
            if len(args) > 1:
                args[1] = './outputFundingStagePrediction'
            else:
                args.append('./outputFundingStagePrediction')
            args = tuple(args)
            
        super().__init__(*args, **kwargs)
        self.model_trainer = EnhancedModelTrainer(self.models_dir)
        
        # Ensure all required directories exist before creating visualizer
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.visualizer = AdvancedVisualizer(self.viz_dir, interactive=False)  # Set to False to prevent interactive display
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
            
            processed_data = self.feature_engineer.extract_features(merged_data)
            X, y = self.feature_engineer.prepare_model_data(processed_data)
            
            # First remap all classes to be continuous from 0
            def remap_classes(y_series):
                unique_classes = sorted(y_series.unique())
                class_map = {old_label: idx for idx, old_label in enumerate(unique_classes)}
                return y_series.map(class_map), class_map
            
            y, initial_map = remap_classes(y)
            logger.info(f"Initial class mapping: {initial_map}")
            
            # Handle rare classes
            class_counts = pd.Series(y).value_counts()
            rare_classes = class_counts[class_counts < 5].index.tolist()
            if rare_classes:
                majority_class = class_counts.idxmax()
                y = y.apply(lambda x: majority_class if x in rare_classes else x)
                logger.info(f"Merged rare classes into majority class {majority_class}")
            
            # Remap again after merging rare classes to ensure continuous labels
            y, final_map = remap_classes(y)
            logger.info(f"Final class mapping after merging rare classes: {final_map}")
            
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
                selected_features = [feature for feature, selected in zip(X.columns, selected_indices) if selected]
                logger.info(f"Selected features: {selected_features}")
            
            # Step 4: Enhanced model training
            logger.info("Step 4: Tuning and training models...")
            best_rf = self.model_trainer.tune_random_forest(X_selected, y)
            best_xgb = self.model_trainer.tune_xgboost(X_selected, y)
            
            # Create ensemble
            ensemble = VotingClassifier(
                estimators=[('rf', best_rf), ('xgb', best_xgb)],
                voting='soft'
            )
            
            # Train models
            rf_model, rf_results = self._train_final_model(best_rf, X_selected, y, 'Random Forest')
            xgb_model, xgb_results = self._train_final_model(best_xgb, X_selected, y, 'XGBoost')
            ensemble_model, ensemble_results = self._train_final_model(ensemble, X_selected, y, 'Ensemble')
            
            # Step 5: Advanced visualizations
            logger.info("Step 5: Creating advanced visualizations...")
            self._create_advanced_visualizations(rf_model, xgb_model, rf_results, xgb_results, y)
            
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
        """Helper method to train final models"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        return model, {
            'accuracy': accuracy_score(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'feature_names': X.columns if hasattr(X, 'columns') else None
        }

    def _create_advanced_visualizations(self, rf_model, xgb_model, rf_results, xgb_results, y):
        """Generate advanced model diagnostics"""
        try:
            # Plot ROC curves
            self.visualizer.plot_roc_curves(rf_results['y_test'], rf_results['y_proba'], classes=np.unique(y))
            self.visualizer.plot_roc_curves(xgb_results['y_test'], xgb_results['y_proba'], classes=np.unique(y))
            
            # Plot calibration curves (using full probability matrix)
            self.visualizer.plot_calibration(rf_results['y_test'], rf_results['y_proba'])
            self.visualizer.plot_calibration(xgb_results['y_test'], xgb_results['y_proba'])
            
            # Plot confidence intervals
            self.visualizer.plot_confidence_intervals(
                rf_results['y_test'], 
                rf_results['y_pred'],
                rf_results['y_proba']
            )
            self.visualizer.plot_confidence_intervals(
                xgb_results['y_test'],
                xgb_results['y_pred'],
                xgb_results['y_proba']
            )
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            logger.error(traceback.format_exc())

    def _save_summary(self, merged_data, X, model_results):
        """Save pipeline summary with extended metrics"""
        summary = {
            'timestamp': self.timestamp,
            'data_records': len(merged_data),
            'features': X.columns.tolist() if hasattr(X, 'columns') else [],
            'model_results': {
                name: {
                    'accuracy': float(results['accuracy']),
                    'model_path': results.get('model_path', 'Not saved')
                }
                for name, results in model_results.items()
            }
        }
        
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
                'funding_amount_log', 'employees', 'employee_efficiency',
                'previous_rounds', 'funding_year', 'funding_month', 'months_since_first_funding'
            ]
            features = [sample_data.get(col, 0) for col in feature_columns]
        else:
            features = sample_data

        # Make prediction
        prediction = self.model_manager.predict(features)
        # Map back to original funding stage
        if hasattr(self.feature_engineer, 'funding_stage_map'):
            reverse_map = {v: k for k, v in self.feature_engineer.funding_stage_map.items()}
            if isinstance(prediction, (int, float)):
                prediction = reverse_map.get(int(prediction), f"Unknown (Class {prediction})")
        return prediction

def main():
    """Main entry point with command line options"""
    import argparse
    import schedule
    import time
    from datetime import datetime, timedelta
    
    parser = argparse.ArgumentParser(description='Funding Stage Prediction System')
    parser.add_argument('--data-dir', type=str, default='./JSONFolder', help='Base directory with JSON data files')
    parser.add_argument('--output-dir', type=str, default='./outputFundingStagePrediction', help='Output directory')
    parser.add_argument('--schedule', action='store_true', help='Run on a schedule')
    parser.add_argument('--interval', type=int, default=24, help='Hours between runs')
    parser.add_argument('--reset-db', action='store_true', help='Reset the database before running')
    parser.add_argument('--continuous', action='store_true', help='Run continuously with scheduling')
    parser.add_argument('--start-time', type=str, help='Start time in HH:MM format (24h)', default='00:00')
    parser.add_argument('--archive', action='store_true',
                        help='Enable data archiving (required for scheduled runs)')
    parser.add_argument('--once', action='store_true', help='Run once without scheduling')
    
    args = parser.parse_args()
    
    # Make sure we're using the JSONFolder explicitly
    if args.data_dir == './':
        args.data_dir = './JSONFolder'
    
    logger.info(f"Starting with data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize pipeline with JSONFolder explicitly
    pipeline = EnhancedPipeline(args.data_dir, args.output_dir, archive=args.archive)
    
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
        
        # Schedule the job - either at a specific time each day or on an interval
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
