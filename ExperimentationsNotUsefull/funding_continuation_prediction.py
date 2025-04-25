import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
import joblib
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("funding_continuation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- COLUMN STANDARDIZATION, FUNDING CLEANING, DATE PARSING HELPERS ---

COLUMN_MAP = {
    'Funding_Amount_USD': 'funding_amount',
    'funding_amount_numeric': 'funding_amount',
    'funding': 'funding_amount',
    'amount': 'funding_amount',
    'Total_Employees': 'employees',
    'employee_count': 'employees',
    'headcount': 'employees'
}

def standardize_columns(df):
    """Standardize column names for funding amount, employees, and other key fields."""
    return df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

def parse_funding(val):
    """Parse funding amount from various formats."""
    if pd.isna(val) or val == '':
        return np.nan
    if isinstance(val, str):
        val = val.replace('$', '').replace(',', '').strip().upper()
        try:
            if 'M' in val: return float(val.replace('M',''))*1e6
            if 'K' in val: return float(val.replace('K',''))*1e3
            if 'B' in val: return float(val.replace('B',''))*1e9
            if val.replace('.', '', 1).isdigit(): return float(val)
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

DATE_FORMATS = ['%d-%b-%y', '%b %Y', '%Y-%m-%d', '%d-%b-%Y']

def robust_parse_date(date_str):
    """Try multiple formats, fallback to dateutil if needed."""
    if pd.isna(date_str):
        return pd.NaT
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except Exception:
        return pd.NaT

# --- CLASS CONSOLIDATION FOR MINIMUM SAMPLES ---

MIN_SAMPLES = 5

def consolidate_funding_stage(df):
    class_counts = df['funding_stage'].value_counts()
    valid_classes = class_counts[class_counts >= MIN_SAMPLES].index
    df['funding_stage'] = df['funding_stage'].where(
        df['funding_stage'].isin(valid_classes), 'Other'
    )
    return df

# --- MULTI-CLASS CALIBRATION PLOT ---

from sklearn.calibration import calibration_curve

def plot_calibration(y_true, y_proba):
    plt.figure(figsize=(10,8))
    for i in range(y_proba.shape[1]):
        prob_true, prob_pred = calibration_curve(
            (y_true == i), y_proba[:,i], n_bins=10
        )
        plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}')
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.legend()
    plt.title('Multi-Class Calibration Plot')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.show()

class ContinuationDataLoader:
    """Handles loading and merging of raw data sources"""
    def __init__(self, base_dir="./"):
        self.base_dir = base_dir
        self._validate_data_files()
        self.data_paths = {
            'fundraiser': os.path.join(base_dir, "fundraise_data/fundraise_data_20250414_152644/startups_20250414_152644.json"),
            'growthlist': os.path.join(base_dir, "growthlist_data/growthlist_startups.json"),
            'topstartup': os.path.join(base_dir, "topstartiorealtimedata/2025-04-14/topstartups_data.json")
        }
        self.historical_db = os.path.join(base_dir, "historical_continuation.db")
        self._init_database()

    def _validate_data_files(self):
        required = [
            "fundraise_data/fundraise_data_20250414_152644/startups_20250414_152644.json",
            "growthlist_data/growthlist_startups.json",
            "topstartiorealtimedata/2025-04-14/topstartups_data.json"
        ]
        missing = []
        for rel_path in required:
            full_path = os.path.join(self.base_dir, rel_path)
            if not os.path.exists(full_path):
                missing.append(rel_path)
        
        if missing:
            error_msg = f"Missing required data files: {', '.join(missing)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def _init_database(self):
        conn = sqlite3.connect(self.historical_db)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS funding_events (
            company_id TEXT,
            funding_date TEXT,
            amount REAL,
            stage TEXT,
            employees INTEGER,
            industry TEXT,
            next_funding_date TEXT,
            event INTEGER,
            PRIMARY KEY (company_id, funding_date)
        )''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS company_metadata (
            company_id TEXT PRIMARY KEY,
            founded_date TEXT,
            headquarters TEXT,
            last_known_valuation REAL
        )''')
        conn.commit()
        conn.close()

    def load_and_preprocess(self):
        """Load and merge data with enhanced deduplication and validation"""
        # Load individual datasets
        fundraiser_data = self._load_fundraiser()
        growthlist_data = self._load_growthlist()
        topstartup_data = self._load_topstartup()

        # Standardize columns for all dataframes
        fundraiser_data = standardize_columns(fundraiser_data)
        growthlist_data = standardize_columns(growthlist_data)
        topstartup_data = standardize_columns(topstartup_data)

        # Add source column to track data origin
        fundraiser_data['source'] = 'fundraiser'
        growthlist_data['source'] = 'growthlist'
        topstartup_data['source'] = 'topstartup'

        # Convert dates before merging using robust parser
        for df in [fundraiser_data, growthlist_data, topstartup_data]:
            df['funding_date'] = df['funding_date'].apply(robust_parse_date)

        # Parse funding amounts using robust parser
        for df in [fundraiser_data, growthlist_data, topstartup_data]:
            df['funding_amount'] = df['funding_amount'].apply(parse_funding)

        # Merge with priority (keep most complete record)
        merged = pd.concat([fundraiser_data, growthlist_data, topstartup_data])
        merged['completeness'] = merged.notna().sum(axis=1)
        merged = merged.sort_values('completeness', ascending=False)
        merged = merged.drop_duplicates(
            subset=['company_id', 'funding_date'], 
            keep='first'
        )
        merged = merged.drop(columns=['completeness', 'source'])

        # Data validation after merge
        assert merged['funding_amount'].notna().all(), "Missing funding amounts"
        assert merged['funding_date'].notna().all(), "Invalid dates remaining"
        assert merged.index.is_unique, "Duplicate indices detected"

        # Clean and calculate survival metrics
        processed = self._clean_data(merged)
        final_df = self._calculate_survival_metrics(processed)

        # Data validation after cleaning
        assert final_df['funding_amount'].notna().all(), "Missing funding amounts after cleaning"
        assert final_df['funding_date'].notna().all(), "Invalid dates remaining after cleaning"
        assert final_df.index.is_unique, "Duplicate indices detected after cleaning"

        return final_df.dropna(subset=['duration', 'event'])

    def _load_fundraiser(self):
        with open(self.data_paths['fundraiser']) as f:
            data = json.load(f)['companies']
        df = pd.json_normalize(data)
        # Standardize column names for downstream compatibility
        column_mapping = {
            'Funding_Amount_USD': 'funding_amount',
            'funding_amount_numeric': 'funding_amount',
            'funding': 'funding_amount'
        }
        df = df.rename(columns=column_mapping)
        return df.rename(columns={
            'Company': 'company_id',
            'Funding_Date': 'funding_date',
            'Funding_Type': 'funding_stage',
            'Total_Employees': 'employees',
            'Industry': 'industry'
        })[['company_id', 'funding_date', 'funding_amount', 'funding_stage', 'employees', 'industry']]

    def _load_growthlist(self):
        try:
            with open(self.data_paths['growthlist']) as f:
                data = json.load(f)
            df = pd.json_normalize(data)
            column_mapping = {
                'Funding_Amount_USD': 'funding_amount',
                'funding_amount_numeric': 'funding_amount',
                'funding': 'funding_amount'
            }
            df = df.rename(columns=column_mapping)
            df = df.rename(columns={
                'name': 'company_id',
                'company_name': 'company_id',
                'last_funding_date': 'funding_date',
                'funding_date': 'funding_date',
                'funding_round': 'funding_stage',
                'funding_type': 'funding_stage',
                'employee_count': 'employees',
                'headcount': 'employees',
                'industry': 'industry',
                'sector': 'industry'
            })
            required_columns = ['company_id', 'funding_date', 'funding_amount', 'funding_stage', 'employees', 'industry']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = np.nan
            df['employees'] = pd.to_numeric(df['employees'], errors='coerce')
            df['funding_date'] = pd.to_datetime(df['funding_date'], errors='coerce')
            return df[required_columns].copy()
        except Exception as e:
            logger.error(f"Error loading Growthlist data: {e}")
            return pd.DataFrame(columns=['company_id', 'funding_date', 'funding_amount', 'funding_stage', 'employees', 'industry'])

    def _load_topstartup(self):
        try:
            with open(self.data_paths['topstartup']) as f:
                data = json.load(f)
                if isinstance(data, dict) and 'startups' in data:
                    data = data['startups']
                elif not isinstance(data, list):
                    data = [data]
            df = pd.json_normalize(data)
            column_mapping = {
                'Funding_Amount_USD': 'funding_amount',
                'funding_amount_numeric': 'funding_amount',
                'funding': 'funding_amount'
            }
            df = df.rename(columns=column_mapping)
            # Funding amount normalization
            def parse_funding(amount):
                if pd.isna(amount):
                    return np.nan
                s = str(amount).replace('$', '').replace(',', '').strip().upper()
                try:
                    if 'M' in s:
                        return float(s.split('M')[0]) * 1_000_000
                    if 'K' in s:
                        return float(s.split('K')[0]) * 1_000
                    if 'B' in s:
                        return float(s.split('B')[0]) * 1_000_000_000
                    return float(s)
                except Exception:
                    return np.nan
            for col in ['funding_amount', 'funding', 'amount']:
                if col in df.columns:
                    df['funding_amount'] = df[col].apply(parse_funding)
                    break
            df = df.rename(columns={
                'name': 'company_id',
                'founding_year': 'funding_date',
                'headquarters': 'location',
                'employees': 'employees',
                'category': 'industry',
                'funding_round': 'funding_stage',
                'funding_type': 'funding_stage'
            })
            required_columns = ['company_id', 'funding_date', 'funding_amount', 'funding_stage', 'employees', 'industry']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = np.nan
            if 'employees' in df.columns:
                df['employees'] = df['employees'].apply(lambda x: pd.to_numeric(str(x).split('-')[0].strip() if pd.notnull(x) else np.nan, errors='coerce'))
            return df[required_columns].copy()
        except Exception as e:
            logger.error(f"Error loading TopStartup data: {e}")
            return pd.DataFrame(columns=['company_id', 'funding_date', 'funding_amount', 'funding_stage', 'employees', 'industry'])

    def merge_datasets(self, historical_data, recent_data):
        """Merge historical and recent funding datasets with validation"""
        logger.info("Merging datasets...")
        
        # Validate input data
        for df, name in [(historical_data, 'historical'), (recent_data, 'recent')]:
            if df.index.duplicated().any():
                logger.warning(f"Duplicate indices found in {name} data, resetting index")
                df = df.reset_index(drop=True)
                
            null_cols = df.columns[df.isna().any()].tolist()
            if null_cols:
                logger.warning(f"Null values found in {name} data columns: {null_cols}")
        
        # Merge datasets
        merged = pd.concat([historical_data, recent_data], ignore_index=True)
        
        # Validate merged data
        if merged.index.duplicated().any():
            logger.warning("Duplicate indices found after merge, resetting index")
            merged = merged.reset_index(drop=True)
            
        logger.info(f"Merged dataset contains {len(merged)} records")
        
        # Check for key columns
        required_cols = ['company_id', 'funding_date', 'funding_amount']
        missing_cols = [col for col in required_cols if col not in merged.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return merged

    def _clean_data(self, df):
        """Enhanced data cleaning with robust error handling"""
        try:
            # Robust date parsing
            def parse_date(date_str):
                formats = ['%d-%b-%y', '%b %Y', '%Y-%m-%d', '%d-%b-%Y']
                for fmt in formats:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except Exception:
                        continue
                return pd.NaT
            df['funding_date'] = df['funding_date'].apply(parse_date)

            # Enhanced funding amount parsing
            def clean_funding_amount(amt):
                if isinstance(amt, str):
                    amt = amt.replace('$', '').replace(',', '')
                    if 'M' in amt:
                        return float(amt.replace('M', '')) * 1_000_000
                    if 'K' in amt:
                        return float(amt.replace('K', '')) * 1_000
                try:
                    return float(amt)
                except Exception:
                    return np.nan
            df['funding_amount'] = df['funding_amount'].apply(clean_funding_amount)

            # Impute missing funding_amount by industry median
            df['funding_amount'] = df.groupby('industry')['funding_amount'].transform(lambda x: x.fillna(x.median()))

            # Robust employee handling
            def clean_employees(val):
                if pd.isna(val) or val in ['', '0', 0]:
                    return np.nan
                try:
                    return int(float(str(val).replace(',', '')))
                except Exception:
                    return np.nan
            df['employees'] = df['employees'].apply(clean_employees)
            df['employees'] = df.groupby('industry')['employees'].transform(lambda x: x.fillna(x.median()))

            # Standardize stages
            stage_mapping = {
                'series a': 'Series A',
                'series b': 'Series B',
                'series c': 'Series C',
                'series d': 'Series D',
                'series e': 'Series E',
                'series f': 'Series F',
                'seed': 'Seed',
                'angel': 'Angel',
                'pre-seed': 'Pre-Seed',
                'ipo': 'IPO'
            }
            
            if 'funding_stage' in df.columns:
                df['funding_stage'] = df['funding_stage'].astype(str).str.lower()
                df['funding_stage'] = df['funding_stage'].map(lambda x: next((v for k, v in stage_mapping.items() if k in x), 'Other'))
            
            # Clean industry categories
            if 'industry' in df.columns:
                df['industry'] = df['industry'].astype(str).str.lower().str.strip()
                df['industry'] = df['industry'].replace('', 'unknown').fillna('unknown')
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['company_id', 'funding_date'], keep='last')
            df = df.sort_values(['company_id', 'funding_date'])
            
            # Log data quality metrics
            total_rows = len(df)
            missing_dates = df['funding_date'].isna().sum()
            missing_amounts = df['funding_amount'].isna().sum()
            
            logger.info(f"Data cleaning completed - Total rows: {total_rows}")
            logger.info(f"Missing dates: {missing_dates}, Missing amounts: {missing_amounts}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            return df

    def _calculate_survival_metrics(self, df):
        """Calculate time-to-event metrics with robust handling of NaN values and no duplicate indices"""
        df = df.reset_index(drop=True)
        # Impute missing funding_amount by industry median before dropping
        df['funding_amount'] = pd.to_numeric(df['funding_amount'], errors='coerce')
        df['funding_amount'] = df.groupby('industry')['funding_amount'].transform(
            lambda x: x.fillna(x.median())
        )
        df = df.dropna(subset=['funding_amount'])
        # Calculate next funding date
        df['next_funding_date'] = df.groupby('company_id')['funding_date'].shift(-1)
        df['duration'] = (df['next_funding_date'] - df['funding_date']).dt.days
        df['event'] = df['next_funding_date'].notnull().astype(int)
        max_date = df['funding_date'].max()
        # Index-aware assignment for censored data
        mask = df['event'] == 0
        df.loc[mask, 'duration'] = (max_date - df.loc[mask, 'funding_date']).dt.days
        if df['duration'].isna().any():
            logger.warning(f"Found {df['duration'].isna().sum()} null durations")
            df = df.dropna(subset=['duration'])
        logger.info(f"Survival metrics calculated for {len(df)} records")
        return df

# Filter out classes with too few samples to avoid SMOTE errors
def filter_small_classes(df, class_col='funding_stage', min_samples=5):
    """Filter out classes with too few samples."""
    class_counts = df[class_col].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    return df[df[class_col].isin(valid_classes)]

class ContinuationFeatureEngineer:
    """Feature engineering for survival analysis"""
    def __init__(self):
        self.industry_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()

    def fit_transform(self, df):
        # Create a working copy
        data = df.copy()
        
        # Add temporal features first
        data = self._add_temporal_features(data)
        
        # Add industry encoding
        data = self._encode_industries(data)
        
        # Add financial features including amount_scaled
        data = self._add_financial_features(data)
        
        # Add growth metrics
        data = self._add_growth_metrics(data)
        
        # Ensure all required features exist
        required_features = [
            'funding_velocity', 'burn_rate', 'industry_code',
            'stage', 'employee_growth', 'amount_scaled',
            'time_since_last', 'round_number'
        ]
        
        # Fill any missing required features with 0
        for feature in required_features:
            if feature not in data.columns:
                logger.warning(f"Creating missing feature: {feature}")
                data[feature] = 0
                
        # Convert stage to numeric if it exists
        if 'stage' in data.columns and not pd.api.types.is_numeric_dtype(data['stage']):
            stage_map = {stage: idx for idx, stage in enumerate(data['stage'].unique())}
            data['stage'] = data['stage'].map(stage_map).fillna(-1)
        
        # Ensure we keep funding_date for temporal splitting
        required_features.append('funding_date')
        
        # Drop rows with missing target variables, but keep funding_date
        result = data[required_features + ['duration', 'event']].copy()
        result = result.dropna(subset=['duration', 'event'])
        
        logger.info(f"Feature engineering complete. Shape: {result.shape}")
        return result

    def _add_temporal_features(self, df):
        """Add time-based features"""
        df = df.sort_values(['company_id', 'funding_date'])
        df['time_since_last'] = df.groupby('company_id')['funding_date'].diff().dt.days
        df['round_number'] = df.groupby('company_id').cumcount() + 1
        # Fill NaN values for first rounds
        df['time_since_last'] = df['time_since_last'].fillna(0)
        return df

    def _encode_industries(self, df):
        """Create industry encoding including numeric industry_code"""
        # Handle missing values
        df['industry'] = df['industry'].fillna('unknown')
        
        # Create simple numeric encoding for industry
        unique_industries = sorted(df['industry'].unique())
        industry_to_code = {ind: idx for idx, ind in enumerate(unique_industries)}
        df['industry_code'] = df['industry'].map(industry_to_code)
        
        return df

    def _add_financial_features(self, df):
        """Add financial features including scaled amounts"""
        # Use correct column name for all calculations
        amount_col = 'funding_amount' if 'funding_amount' in df.columns else 'amount'
        if amount_col in df.columns:
            # Scale amounts using StandardScaler
            df['amount_scaled'] = self.scaler.fit_transform(df[[amount_col]].fillna(0))
            
            # Calculate funding velocity with increased variance
            df['funding_velocity'] = df.groupby('company_id')[amount_col].transform(
                lambda x: np.log1p((x - x.shift(1)) / (x.shift(1) + 1e-6))  # Log transform for better distribution
            ).fillna(0)
            
            # Add funding growth rate
            df['funding_growth'] = df.groupby('company_id')[amount_col].transform(
                lambda x: x / x.shift(1) - 1
            ).fillna(0).replace([np.inf, -np.inf], 0)
            
            # Add amount relative to company mean
            company_mean = df.groupby('company_id')[amount_col].transform('mean')
            df['amount_relative'] = (df[amount_col] - company_mean) / (company_mean + 1e-6)
            df['amount_relative'] = df['amount_relative'].fillna(0).clip(-5, 5)  # Clip outliers
        else:
            df['amount_scaled'] = 0
            df['funding_velocity'] = 0
            df['funding_growth'] = 0
            df['amount_relative'] = 0
            
        # Calculate burn rate with robust handling
        if 'employees' in df.columns:
            employee_count = df['employees'].clip(lower=1)
            df['burn_rate'] = df[amount_col].fillna(0) / (employee_count * 30)
            df['burn_rate'] = df['burn_rate'].fillna(df['burn_rate'].median())
            df['burn_rate'] = df['burn_rate'].clip(lower=0, upper=df['burn_rate'].quantile(0.99))  # Remove extreme outliers
        else:
            df['burn_rate'] = 0
            
        return df

    def _add_growth_metrics(self, df):
        """Add growth-related metrics"""
        company_groups = df.groupby('company_id')
        
        # Employee growth with explicit fill_method=None
        if 'employees' in df.columns:
            df['employee_growth'] = company_groups['employees'].transform(
                lambda x: x.pct_change(fill_method=None).fillna(0)
            )
        else:
            df['employee_growth'] = 0
            
        return df

class FeatureEngineering:
    def __init__(self):
        """Initialize with dynamic funding stage mapping"""
        self.funding_stage_map = {}  # Will be populated dynamically

    def extract_features(self, df):
        """Dynamically create funding stage mapping with robust Unknown handling"""
        data = df.copy()
        # Dynamic class mapping with 'Unknown' category
        all_stages = sorted(data['funding_stage'].dropna().unique().tolist() + ['Unknown'])
        self.funding_stage_map = {stage: idx for idx, stage in enumerate(all_stages)}
        data['funding_stage'] = data['funding_stage'].fillna('Unknown')
        data['funding_stage_numeric'] = data['funding_stage'].map(
            lambda x: self.funding_stage_map.get(x, self.funding_stage_map['Unknown'])
        )
        # ...existing code for other features...
        return data

class ContinuationModelTrainer:
    """Trains and evaluates survival analysis models"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {}

    def _prepare_training_data(self, train_df, test_df):
        """Prepare data for model training with robust preprocessing"""
        # Get numeric features
        feature_cols = train_df.select_dtypes(include=[np.number]).columns
        feature_cols = feature_cols.drop(['duration', 'event'])
        
        # Initialize scaler
        scaler = StandardScaler()
        
        # Handle infinities and NaN values more robustly
        train_features = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        test_features = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Use more robust imputation strategy
        imputer = SimpleImputer(strategy='median')
        train_features = pd.DataFrame(
            imputer.fit_transform(train_features),
            columns=train_features.columns,
            index=train_features.index
        )
        test_features = pd.DataFrame(
            imputer.transform(test_features),
            columns=test_features.columns,
            index=test_features.index
        )
        
        # Scale features
        train_features = pd.DataFrame(
            scaler.fit_transform(train_features),
            columns=feature_cols,
            index=train_features.index
        )
        test_features = pd.DataFrame(
            scaler.transform(test_features),
            columns=feature_cols,
            index=test_features.index
        )
        
        # Handle target variables
        train_targets = train_df[['duration', 'event']].copy()
        test_targets = test_df[['duration', 'event']].copy()
        
        # Ensure duration is positive and finite
        train_targets['duration'] = train_targets['duration'].clip(lower=1)
        test_targets['duration'] = test_targets['duration'].clip(lower=1)
        
        # Ensure event is binary
        train_targets['event'] = train_targets['event'].astype(int)
        test_targets['event'] = test_targets['event'].astype(int)
        
        return train_features, test_features, train_targets, test_targets

    def train(self, df):
        """Train both Cox and RSF models with comprehensive validation"""
        train_df, test_df = self._temporal_split(df)
        
        # Prepare data with robust preprocessing
        train_features, test_features, train_targets, test_targets = self._prepare_training_data(train_df, test_df)
        
        # Train Cox model
        cox_model, cox_metrics = self.train_cox(train_features, train_targets)
        if cox_model is not None:
            try:
                cox_metrics['test_c_index'] = self._validate_cox(cox_model, pd.concat([test_features, test_targets], axis=1))
            except Exception as e:
                logger.error(f"Cox validation failed: {str(e)}")
                cox_metrics['test_c_index'] = None
            
        # Train Random Survival Forest
        rsf_model, rsf_metrics = self.train_rsf(train_features, train_targets)
        if rsf_model is not None:
            try:
                rsf_metrics['test_c_index'] = self._validate_rsf(rsf_model, test_features, test_targets)
            except Exception as e:
                logger.error(f"RSF validation failed: {str(e)}")
                rsf_metrics['test_c_index'] = None
        
        # Save results
        results = {
            'cox': {'model': cox_model, **(cox_metrics or {})},
            'rsf': {'model': rsf_model, **(rsf_metrics or {})}
        }
        
        # Track experiment
        metadata = {
            'data_size': len(df),
            'train_size': len(train_features),
            'test_size': len(test_features),
            'features': list(train_features.columns)
        }
        self._track_experiment(results, metadata)
        
        return results

    def _temporal_split(self, df, test_size=0.2):
        """Split data based on funding dates to prevent data leakage"""
        # Ensure funding_date is datetime
        df['funding_date'] = pd.to_datetime(df['funding_date'])
        # Find the cutoff date that gives us the desired train/test split
        cutoff_date = df['funding_date'].quantile(1 - test_size)
        
        # Split the data
        train_df = df[df['funding_date'] <= cutoff_date].copy()
        test_df = df[df['funding_date'] > cutoff_date].copy()
        
        # Remove funding_date from features to avoid datetime issues
        train_features = train_df.drop(columns=['funding_date'])
        test_features = test_df.drop(columns=['funding_date'])
        
        logger.info(f"Temporal split at {cutoff_date:%Y-%m-%d}")
        logger.info(f"Train size: {len(train_features)}, Test size: {len(test_features)}")
        
        return train_features, test_features

    def train_cox(self, features, targets):
        """Train Cox Proportional Hazards model with improved stability"""
        try:
            # Scale features before fitting Cox model
            scaler = StandardScaler()
            features_scaled = pd.DataFrame(
                scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
            
            # Remove low variance features
            selector = VarianceThreshold(threshold=0.01)
            features_selected = pd.DataFrame(
                selector.fit_transform(features_scaled),
                columns=features_scaled.columns[selector.get_support()],
                index=features_scaled.index
            )
            
            # Initialize model with increased regularization
            cph = CoxPHFitter(penalizer=0.5, l1_ratio=0.1)
            
            # Prepare training data
            train_data = pd.concat([targets, features_selected], axis=1)
            
            # Fit model with progress tracking
            cph.fit(train_data, 
                   duration_col='duration',
                   event_col='event',
                   show_progress=True)
            
            # Calculate performance metrics with error handling
            metrics = {
                'c_index': float(cph.concordance_index_),
                'log_likelihood': float(cph.log_likelihood_),
                'aic': float(cph.AIC_partial_),
                'feature_importance': dict(zip(features_selected.columns, abs(cph.params_.values)))
            }
            
            return cph, metrics
            
        except Exception as e:
            logger.error(f"Cox model training failed: {str(e)}")
            return None, {'error': str(e), 'c_index': 0.0}

    def train_rsf(self, features, targets):
        """Train Random Survival Forest with optimized parameters"""
        try:
            from sksurv.ensemble import RandomSurvivalForest
            
            # Check for NaN values
            if features.isna().any().any():
                logger.warning("Features contain NaN values, filling with median")
                features = features.fillna(features.median())
                
            if targets.isna().any().any():
                logger.error("Target variables contain NaN values")
                return None, {'error': 'Target variables contain NaN values'}
            
            # Ensure all features are numeric
            non_numeric = features.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric) > 0:
                logger.error(f"Non-numeric features found: {list(non_numeric)}")
                return None, {'error': f'Non-numeric features found: {list(non_numeric)}'}
            
            # Prepare survival data structure with validation
            try:
                y = np.array(
                    [(bool(row['event']), float(row['duration'])) 
                     for _, row in targets.iterrows()],
                    dtype=[('event', '?'), ('time', '<f8')]
                )
            except Exception as e:
                logger.error(f"Failed to create survival data structure: {str(e)}")
                return None, {'error': f'Failed to create survival data structure: {str(e)}'}
            
            # Initialize model with optimized parameters
            rsf = RandomSurvivalForest(
                n_estimators=200,
                min_samples_split=10,
                min_samples_leaf=15,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            )
            
            # Fit model with error handling
            try:
                rsf.fit(features, y)
            except Exception as e:
                logger.error(f"RSF model fitting failed: {str(e)}")
                return None, {'error': f'Model fitting failed: {str(e)}'}
            
            # Calculate performance metrics
            try:
                c_index = rsf.score(features, y)
                
                # Get feature importance safely
                if hasattr(rsf, 'feature_importances_'):
                    feature_importance = dict(zip(features.columns, rsf.feature_importances_))
                else:
                    feature_importance = {}
                    logger.warning("Feature importances not available for RSF model")
                
                metrics = {
                    'c_index': float(c_index),  # Convert to float for JSON serialization
                    'feature_importance': feature_importance,
                    'n_trees': rsf.n_estimators
                }
                
                return rsf, metrics
                
            except Exception as e:
                logger.error(f"Error calculating RSF metrics: {str(e)}")
                return None, {'error': f'Error calculating metrics: {str(e)}'}
            
        except ImportError:
            logger.error("scikit-survival not installed, skipping RSF training")
            return None, {'error': 'scikit-survival not installed'}
        except Exception as e:
            logger.error(f"RSF training failed: {str(e)}")
            return None, {'error': str(e)}

    def _validate_cox(self, model, test_data):
        """Validate Cox model on test set with robust error handling"""
        try:
            # Get model coefficients - this is what we actually use since hazard_col_names_ doesn't exist
            test_features = test_data.drop(columns=['duration', 'event'])
            model_coeffs = model.params_.index
            
            # Ensure test data has all required columns
            missing_cols = set(model_coeffs) - set(test_features.columns)
            if missing_cols:
                logger.error(f"Missing columns in test data: {missing_cols}")
                return None
                
            # Handle potential NaN values
            test_df = test_data.fillna(test_data.median())
            
            # Calculate concordance index
            score = model.score(test_df)
            return float(score)  # Ensure score is JSON serializable
            
        except Exception as e:
            logger.error(f"Cox validation failed: {str(e)}")
            return None

    def _validate_rsf(self, model, features, targets):
        """Validate Random Survival Forest on test set with robust error handling"""
        try:
            # Handle NaN values in features
            features_clean = features.fillna(features.median())
            
            # Create structured array for survival data
            structured_targets = np.array(
                [(bool(e), float(d)) for e, d in zip(targets['event'], targets['duration'])],
                dtype=[('event', '?'), ('time', '<f8')]
            )
            
            # Calculate score with error handling
            try:
                score = model.score(features_clean, structured_targets)
                return float(score)
            except Exception as e:
                logger.error(f"Error calculating RSF score: {str(e)}")
                return 0.0
            
        except Exception as e:
            logger.error(f"RSF validation failed: {str(e)}")
            return None

    def _prepare_survival_data(self, targets):
        """Prepare survival data structure with validation"""
        try:
            # Convert duration to positive float
            duration = pd.to_numeric(targets['duration'], errors='coerce')
            duration = duration.clip(lower=1)
            
            # Ensure event is binary
            event = targets['event'].astype(bool)
            
            # Create structured array
            y = np.array(
                list(zip(event, duration)),
                dtype=[('event', '?'), ('time', '<f8')]
            )
            
            return y
            
        except Exception as e:
            logger.error(f"Failed to prepare survival data: {str(e)}")
            raise

    def _save_best_model(self, cox_model, rsf_model, cox_metrics, rsf_metrics):
        """Save the best performing model based on validation metrics"""
        if cox_model is not None and rsf_model is not None:
            cox_score = cox_metrics.get('test_c_index', 0)
            rsf_score = rsf_metrics.get('test_c_index', 0)
            
            if cox_score > rsf_score:
                best_model = ('cox', cox_model, cox_score)
            else:
                best_model = ('rsf', rsf_model, rsf_score)
            
            model_path = os.path.join(self.output_dir, f"best_{best_model[0]}_model.joblib")
            joblib.dump(best_model[1], model_path)
            logger.info(f"Saved best model ({best_model[0]}) with score {best_model[2]:.3f} to {model_path}")

    def _track_experiment(self, results, metadata):
        """Track experiment results and model versions"""
        experiment_dir = os.path.join(self.output_dir, 'experiments')
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create experiment record
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_data = {
            'timestamp': timestamp,
            'cox_metrics': results['cox'],
            'rsf_metrics': results['rsf'],
            'metadata': metadata,
            'feature_importance': {
                'cox': results['cox'].get('feature_importance', {}),
                'rsf': results['rsf'].get('feature_importance', {})
            }
        }
        
        # Save experiment results
        experiment_path = os.path.join(experiment_dir, f'experiment_{timestamp}.json')
        with open(experiment_path, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
            
        # Update experiment index
        index_path = os.path.join(experiment_dir, 'experiment_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = []
            
        index.append({
            'timestamp': timestamp,
            'cox_c_index': float(results['cox'].get('c_index', 0)),
            'rsf_c_index': float(results['rsf'].get('c_index', 0)),
            'data_size': metadata.get('data_size', 0)
        })
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
            
        logger.info(f"Experiment results saved to {experiment_path}")
        
        # Track model versions
        self._update_model_registry(results, timestamp)

    def _update_model_registry(self, results, timestamp):
        """Update model registry with latest model versions"""
        registry_dir = os.path.join(self.output_dir, 'model_registry')
        os.makedirs(registry_dir, exist_ok=True)
        
        # Save models if they exist
        if results['cox'].get('model') is not None:
            model_path = os.path.join(registry_dir, f'cox_model_{timestamp}.joblib')
            joblib.dump(results['cox']['model'], model_path)
            
        if results['rsf'].get('model') is not None:
            model_path = os.path.join(registry_dir, f'rsf_model_{timestamp}.joblib')
            joblib.dump(results['rsf']['model'], model_path)
            
        # Update registry index
        registry_index = os.path.join(registry_dir, 'model_registry.json')
        if os.path.exists(registry_index):
            with open(registry_index, 'r') as f:
                index = json.load(f)
        else:
            index = {'cox': [], 'rsf': []}
            
        # Add new model versions to index
        for model_type in ['cox', 'rsf']:
            if results[model_type].get('model') is not None:
                model_info = {
                    'timestamp': timestamp,
                    'c_index': float(results[model_type].get('c_index', 0)),
                    'filename': f'{model_type}_model_{timestamp}.joblib',
                    'metrics': {k: v for k, v in results[model_type].items() 
                              if k not in ['model', 'summary'] and not isinstance(v, dict)}
                }
                index[model_type].append(model_info)
                
        # Save updated index
        with open(registry_index, 'w') as f:
            json.dump(index, f, indent=2)

class ContinuationVisualizer:
    """Handles data visualization and analysis outputs"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")  # Modern seaborn styling
        self.palette = sns.color_palette("husl", 8)
        self.feature_palettes = {
            'categorical': sns.color_palette("Set3", 12),
            'sequential': sns.color_palette("viridis", 8),
            'diverging': sns.color_palette("RdYlBu", 11)
        }

    def plot_survival_curves(self, df, model):
        """Plot survival curves with confidence intervals"""
        if len(df) < 10:
            logger.error("Insufficient data for survival curves")
            return
            
        # Filter invalid durations
        valid_mask = (df['duration'] > 0) & (df['duration'] < 365*5)  # 5 years max
        df = df[valid_mask].copy()
        
        plt.figure(figsize=(12, 8))
        
        # Overall survival curve
        kmf = KaplanMeierFitter()
        kmf.fit(df['duration'], df['event'], label='All Companies')
        kmf.plot_survival_function(ci_show=True)
        
        # Plot by industry if available
        if 'industry_category' in df.columns:
            for industry in df['industry_category'].unique()[:5]:  # Top 5 industries
                mask = df['industry_category'] == industry
                if mask.sum() > 10:  # Minimum sample size
                    kmf_ind = KaplanMeierFitter()
                    kmf_ind.fit(
                        df[mask]['duration'],
                        df[mask]['event'],
                        label=f'{industry.title()}'
                    )
                    kmf_ind.plot_survival_function(ci_show=False)
        
        plt.title('Startup Funding Survival Analysis')
        plt.xlabel('Days Since Last Funding Round')
        plt.ylabel('Probability of Not Receiving Next Round')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'survival_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_cox_summary(self, cox_model):
        """Plot Cox model hazard ratios and confidence intervals"""
        try:
            plt.figure(figsize=(10, 6))
            cox_model.plot()
            plt.title('Cox Model Hazard Ratios')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'cox_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot Cox summary: {str(e)}")

    def plot_feature_importance(self, model_results):
        """Plot feature importance from both Cox and RSF models"""
        plt.figure(figsize=(12, 6))
        
        # Get feature importance with error handling
        cox_importance = {}
        rsf_importance = {}
        
        if 'cox' in model_results and model_results['cox'].get('model') is not None:
            cox_model = model_results['cox']['model']
            if hasattr(cox_model, 'params_'):
                cox_importance = dict(zip(cox_model.params_.index, abs(cox_model.params_.values)))
        
        if 'rsf' in model_results and model_results['rsf'].get('model') is not None:
            rsf_model = model_results['rsf']['model']
            if hasattr(rsf_model, 'feature_importances_'):
                rsf_importance = dict(zip(rsf_model.feature_names_in_, rsf_model.feature_importances_))
        
        # Get common features
        features = list(set(cox_importance.keys()) | set(rsf_importance.keys()))
        
        if not features:
            logger.warning("No feature importance data available")
            return
            
        x = np.arange(len(features))
        width = 0.35
        
        # Plot bars for both models
        cox_values = [cox_importance.get(f, 0) for f in features]
        rsf_values = [rsf_importance.get(f, 0) for f in features]
        
        plt.bar(x - width/2, cox_values, width, label='Cox Model')
        plt.bar(x + width/2, rsf_values, width, label='Random Survival Forest')
        
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance Comparison')
        plt.xticks(x, features, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_funding_patterns(self, df):
        """Visualize funding patterns over time with robust column handling"""
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot 1: Funding amounts over time
        df['year'] = pd.to_datetime(df['funding_date']).dt.year
        # Use funding_amount instead of amount column
        yearly_funding = df.groupby('year')['funding_amount'].sum()
        ax1.plot(yearly_funding.index, yearly_funding.values, marker='o')
        ax1.set_title('Total Funding Amount by Year')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Total Funding Amount ($)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success rate by industry with error handling
        if 'industry_category' in df.columns and len(df['industry_category'].unique()) > 1:
            industry_success = df.groupby('industry_category')['event'].mean().sort_values(ascending=True)
            industry_success.plot(kind='barh', ax=ax2)
            ax2.set_title('Funding Success Rate by Industry')
            ax2.set_xlabel('Success Rate')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Insufficient industry data', 
                    horizontalalignment='center',
                    verticalalignment='center')
            ax2.set_title('Industry Analysis Unavailable')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'funding_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_report(self, df, model_results):
        """Generate a comprehensive analysis report with robust error handling"""
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Funding Continuation Analysis Report\n")
            f.write("="*40 + "\n\n")
            
            # Dataset statistics
            f.write("Dataset Statistics:\n")
            f.write("-"*20 + "\n")
            f.write(f"Total companies analyzed: {df['company_id'].nunique()}\n")
            f.write(f"Total funding events: {len(df)}\n")
            f.write(f"Date range: {df['funding_date'].min():%Y-%m-%d} to {df['funding_date'].max():%Y-%m-%d}\n\n")
            
            # Model performance
            f.write("Model Performance:\n")
            f.write("-"*20 + "\n")
            
            # Safely get model metrics
            cox_metrics = model_results.get('cox', {})
            rsf_metrics = model_results.get('rsf', {})
            
            cox_cindex = cox_metrics.get('c_index', 'Not available')
            rsf_cindex = rsf_metrics.get('c_index', 'Not available')
            
            if isinstance(cox_cindex, (int, float)):
                f.write(f"Cox Model C-index: {cox_cindex:.3f}\n")
            else:
                f.write("Cox Model C-index: Not available\n")
                
            if isinstance(rsf_cindex, (int, float)):
                f.write(f"RSF Model C-index: {rsf_cindex:.3f}\n")
            else:
                f.write("RSF Model C-index: Not available\n")
            
            # Model errors if any
            if 'error' in cox_metrics:
                f.write(f"\nCox Model Error: {cox_metrics['error']}\n")
            if 'error' in rsf_metrics:
                f.write(f"\nRSF Model Error: {rsf_metrics['error']}\n")
            
            f.write("\nKey Findings:\n")
            f.write("-"*20 + "\n")
            
            # Average time between rounds
            successful_rounds = df[df['event']==1]
            if len(successful_rounds) > 0:
                avg_duration = successful_rounds['duration'].mean()
                f.write(f"Average time between funding rounds: {avg_duration:.1f} days\n")
            else:
                f.write("No successful funding rounds found in the dataset\n")
            
            # Success rate by stage if available
            if 'stage' in df.columns:
                stage_success = df.groupby('stage')['event'].agg(['count', 'mean'])
                f.write("\nSuccess rate by funding stage:\n")
                for stage, stats in stage_success.iterrows():
                    count = stats['count']
                    rate = stats['mean']
                    f.write(f"{stage} (n={count}): {rate:.1%}\n")
            
            logger.info(f"Analysis report saved to {report_path}")

class FundingContinuationPipeline:
    """End-to-end pipeline for funding continuation prediction"""
    def __init__(self, base_dir="./"):
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "outputFundingContinuation")
        self.loader = ContinuationDataLoader(base_dir)
        self.fe = ContinuationFeatureEngineer()
        self.trainer = ContinuationModelTrainer(self.output_dir)
        self.viz = ContinuationVisualizer(self.output_dir)

    def run(self):
        logger.info("Starting funding continuation prediction pipeline")
        raw_df = self.loader.load_and_preprocess()
        raw_df = standardize_columns(raw_df) 
        
        # Add error handling for feature engineering
        try:
            logger.info("Engineering features...")
            processed_df = self.fe.fit_transform(raw_df)
            if processed_df.empty:
                raise ValueError("Feature engineering produced empty dataset")
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            return
            
        # Filter out classes with too few samples
        processed_df = filter_small_classes(processed_df, class_col='funding_stage', min_samples=5)
        
        # Add validation for processed data
        if len(processed_df) < 10:
            logger.error("Insufficient data after preprocessing")
            return
            
        logger.info("Training models...")
        results = self.trainer.train(processed_df)
        
        if not results:
            logger.error("Model training failed")
            return
            
        logger.info("Generating visualizations...")
        if 'cox' in results and results['cox'].get('model') is not None:
            self.viz.plot_survival_curves(processed_df, results['cox']['model'])
            self.viz.plot_cox_summary(results['cox']['model'])
            
        self.viz.plot_feature_importance(results)
        self.viz.plot_funding_patterns(raw_df)
        self.viz.create_summary_report(raw_df, results)
        self._save_results(results)
        logger.info("Pipeline completed successfully!")

    def _save_results(self, results):
        """Save results with robust error handling"""
        try:
            metrics = {
                'cox_c_index': float(results.get('cox', {}).get('c_index', 0.0)),
                'rsf_c_index': float(results.get('rsf', {}).get('c_index', 0.0))
            }
            
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save metrics
            with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save Cox summary if available
            if results.get('cox', {}).get('model') is not None and 'summary' in results['cox']:
                with open(os.path.join(self.output_dir, 'cox_summary.txt'), 'w') as f:
                    f.write(str(results['cox']['summary']))
                    
            logger.info(f"Results saved successfully to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            # Create minimal metrics file to prevent further errors
            with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
                json.dump({'cox_c_index': 0.0, 'rsf_c_index': 0.0}, f)

if __name__ == "__main__":
    pipeline = FundingContinuationPipeline("./")
    pipeline.run()
