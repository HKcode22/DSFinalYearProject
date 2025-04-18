import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import sqlite3
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("funding_prediction.log"),
                            logging.StreamHandler()])
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, base_dir="./"):
        """Initialize data loader with paths to data sources and historical database"""
        self.base_dir = base_dir
        self.historical_db = os.path.join(base_dir, "historical_funding_data.db")
        
        # Define paths to source files
        self.fundraiser_path = os.path.join(base_dir, "fundraise_data", 
                                          "fundraise_data_20250414_152644", 
                                          "startups_20250414_152644.json")
        self.growthlist_path = os.path.join(base_dir, "growthlist_data", 
                                          "growthlist_startups.json")
        self.topstartup_path = os.path.join(base_dir, "topstartiorealtimedata", 
                                          "2025-04-14", "topstartups_data.json")
        
        # Initialize the database for historical data
        self._init_historical_db()
    
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
        """Handle both list and dict formatted JSON"""
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
            
            # Standardize column names
            column_mapping = {
                'name': 'company_name',
                'funding_round': 'funding_stage',
                'funding_type': 'funding_stage',  # Alternative naming
                'amount': 'funding_amount',
                'date': 'funding_date',
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
            return pd.DataFrame()
    
    def _parse_funding_amount(self, amount_str):
        """Convert funding amount strings (e.g., "$27.6M") to numeric values"""
        if not amount_str or pd.isna(amount_str) or amount_str == "":
            return np.nan
        
        try:
            # Remove currency symbol and commas
            amount_str = amount_str.replace('$', '').replace(',', '').strip()
            
            # Convert based on unit (M=million, B=billion, K=thousand)
            if 'B' in amount_str:
                return float(amount_str.replace('B', '')) * 1e9
            elif 'M' in amount_str:
                return float(amount_str.replace('M', '')) * 1e6
            elif 'K' in amount_str:
                return float(amount_str.replace('K', '')) * 1e3
            else:
                return float(amount_str)
        except Exception:
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
        """Merge datasets using list-based construction"""
        try:
            # Load raw data
            fundraiser_df = self.load_fundraiser_data()
            growthlist_df = self.load_growthlist_data()
            topstartup_df = self.load_topstartup_data()
            
            # Initialize list to store all records
            all_records = []
            
            # Process fundraiser data
            if not fundraiser_df.empty:
                for _, row in fundraiser_df.iterrows():
                    if pd.notna(row.get('Company')):  # Only add records with valid company names
                        all_records.append({
                            'company_name': row.get('Company'),
                            'funding_stage': row.get('Funding_Type'),
                            'funding_amount': row.get('Funding_Amount_USD'),
                            'funding_date': row.get('Funding_Date'),
                            'industry': row.get('Industry'),
                            'employees': row.get('Total_Employees')
                        })
            
            # Process growthlist data
            if not growthlist_df.empty:
                for _, row in growthlist_df.iterrows():
                    if pd.notna(row.get('name')):  # Only add records with valid company names
                        all_records.append({
                            'company_name': row.get('name'),
                            'funding_stage': row.get('funding_type'),
                            'funding_amount': row.get('funding_amount_numeric'),
                            'funding_date': row.get('last_funding_date'),
                            'industry': row.get('industry'),
                            'employees': None
                        })
            
            # Process topstartup data
            if not topstartup_df.empty:
                for _, row in topstartup_df.iterrows():
                    company_name = row.get('company_name') or row.get('name')
                    if pd.notna(company_name):  # Only add records with valid company names
                        all_records.append({
                            'company_name': company_name,
                            'funding_stage': row.get('funding_stage') or row.get('funding_round'),
                            'funding_amount': row.get('funding_amount') or row.get('amount'),
                            'funding_date': row.get('funding_date') or row.get('date'),
                            'industry': row.get('industry') or row.get('category'),
                            'employees': None
                        })
            
            if all_records:
                # Create DataFrame from records
                merged_data = pd.DataFrame(all_records)
                
                # Drop duplicates after creation
                merged_data = merged_data.drop_duplicates(
                    subset=['company_name', 'funding_date']
                ).reset_index(drop=True)
                
                # Fill missing values
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
        """Initialize with funding stage mapping dictionary"""
        self.funding_stage_map = {
            'Pre-Seed': 0,
            'Seed': 1,
            'Series A': 2,
            'Series B': 3,
            'Series C': 4,
            'Series D': 5,
            'Series E': 6,
            'Series F': 7,
            'Private Equity': 8,
            'Venture - Series Unknown': 9,
            'Debt Financing': 10,
            'Grant': 11,
            'Unknown': 12  # Add Unknown stage mapping
        }
        
    def extract_features(self, df):
        """Extract and engineer features from merged dataset"""
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Convert funding stage to numeric (for ordered classification)
        data['funding_stage_numeric'] = data['funding_stage'].apply(
            lambda x: next((v for k, v in self.funding_stage_map.items() 
                           if k in str(x)), self.funding_stage_map['Unknown'])
        )
        
        # Handle dates and extract temporal features
        try:
            data['funding_date'] = pd.to_datetime(data['funding_date'], format='mixed', errors='coerce')
            data['funding_year'] = data['funding_date'].dt.year
            data['funding_month'] = data['funding_date'].dt.month
            
            # Calculate months since first funding (proxy for company age)
            company_first_funding = data.groupby('company_name')['funding_date'].min()
            data['company_first_funding'] = data['company_name'].map(company_first_funding)
            data['months_since_first_funding'] = (data['funding_date'] - 
                                                 data['company_first_funding']).dt.days / 30
        except Exception as e:
            logger.warning(f"Error processing dates: {e}")
            # Set default values if date processing fails
            data['funding_year'] = datetime.now().year
            data['funding_month'] = datetime.now().month
            data['months_since_first_funding'] = 0
        
        # Log transform funding amount (handle skewed distribution)
        data['funding_amount_log'] = np.log1p(pd.to_numeric(data['funding_amount'], errors='coerce').fillna(0))
        
        # Employee efficiency (funding per employee)
        if 'employees' in data.columns:
            data['employees'] = pd.to_numeric(data['employees'], errors='coerce')
            data['employee_efficiency'] = data['funding_amount'] / data['employees'].replace(0, np.nan)
            data['employee_efficiency'] = data['employee_efficiency'].fillna(data['employee_efficiency'].median())
        else:
            data['employees'] = np.nan
            data['employee_efficiency'] = np.nan
        
        # Standardize industry categories
        data['industry_category'] = data['industry'].fillna('Unknown')
        
        # Map to standardized categories
        industry_mapping = {
            'artificial intelligence': 'AI & ML',
            'information technology': 'IT & Software',
            'health': 'Healthcare',
            'biotech': 'Biotech',
            'financial': 'FinTech',
            'education': 'EdTech',
            'retail': 'Retail',
            'energy': 'Energy',
            'food': 'Food & Agriculture'
        }
        
        # Apply mapping for standardization
        for key, value in industry_mapping.items():
            mask = data['industry_category'].str.contains(key, case=False, na=False)
            data.loc[mask, 'industry_category'] = value
        
        # Create dummy variables for industries
        industry_dummies = pd.get_dummies(data['industry_category'], prefix='industry')
        data = pd.concat([data, industry_dummies], axis=1)
        
        # Location features (if available)
        if 'location' in data.columns:
            data['location_category'] = data['location'].fillna('Unknown')
            
            # Extract country or state
            data['location_category'] = data['location_category'].apply(
                lambda x: x.split(',')[-1].strip() if isinstance(x, str) and ',' in x else x
            )
            
            # Create location dummies
            location_dummies = pd.get_dummies(data['location_category'], prefix='location')
            data = pd.concat([data, location_dummies], axis=1)
        
        # Funding frequency features
        company_funding_counts = data.groupby('company_name').size()
        data['previous_rounds'] = data['company_name'].map(company_funding_counts) - 1
        data['previous_rounds'] = data['previous_rounds'].clip(lower=0)
        
        # Fill missing values
        numeric_cols = ['funding_amount', 'funding_amount_log', 'employees', 'employee_efficiency']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
        
        logger.info(f"Feature engineering complete: {data.shape[1]} features created")
        return data
    
    def prepare_model_data(self, data):
        """Prepare feature matrix with proper type handling"""
        # Select relevant features
        feature_cols = [
            'funding_amount_log', 'employees', 'employee_efficiency',
            'funding_year', 'funding_month', 'previous_rounds',
            'months_since_first_funding'
        ]

        # Clean feature data - ensure numeric types
        X = data[feature_cols].copy()
        
        # Convert all features to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill missing values only for numeric columns
        numeric_cols = X.select_dtypes(include=np.number).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Target variable processing
        y = pd.to_numeric(data['funding_stage_numeric'], errors='coerce')
        valid_mask = y.notna()
        
        X = X[valid_mask]
        y = y[valid_mask].astype(int)
        
        logger.info(f"Prepared model data: X shape={X.shape}, y shape={y.shape}")
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

class Visualizer:
    def __init__(self, output_dir="./visualizations"):
        """Initialize visualizer with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
        plt.close()



class FundingStagePredictionPipeline:
    def __init__(self, base_dir="./", output_dir="./output"):
        """Initialize the complete pipeline"""
        self.base_dir = base_dir
        self.output_dir = output_dir
        
        # Create output directory structure
        self.models_dir = os.path.join(output_dir, "models")
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(base_dir)
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



def main():
    """Main entry point with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Funding Stage Prediction System')
    parser.add_argument('--data-dir', type=str, default='./', help='Base directory with data')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--schedule', action='store_true', help='Run on a schedule')
    parser.add_argument('--interval', type=int, default=24, help='Hours between runs')
    parser.add_argument('--reset-db', action='store_true', help='Reset the database before running')
    
    args = parser.parse_args()
    
    # Initialize and run the pipeline
    pipeline = FundingStagePredictionPipeline(args.data_dir, args.output_dir)
    
    if args.reset_db:
        pipeline.data_loader.reset_database()
    
    if args.schedule:
        logger.info(f"Starting scheduled runs every {args.interval} hours")
        pipeline.schedule_run(args.interval)
    else:
        logger.info("Running pipeline once")
        pipeline.run()

if __name__ == "__main__":
    main()
