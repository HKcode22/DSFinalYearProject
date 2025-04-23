import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import pickle
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import time
import threading
import argparse
import sys


class FundingAnomalyDetection:
    """
    Anomaly detection for startup funding patterns.
    Identifies unusual funding patterns that require investigation.
    """
    
    def __init__(self, data_dir=None, output_dir=None, contamination=0.01):
        """
        Initialize the anomaly detection system.
        
        Args:
            data_dir (str): Directory containing funding data JSON files
            output_dir (str): Directory to save output files, models, and visualizations
            contamination (float): Expected proportion of anomalies in the dataset
        """
        self.data_dir = data_dir or os.path.join(os.getcwd(), 'JSONFolder')
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'output')
        self.model = None
        self.scaler = StandardScaler()
        self.industry_means = {}
        self.industry_stds = {}
        self.contamination = contamination
        self.logger = self._setup_logging()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _setup_logging(self):
        """Configure logging for the anomaly detection module"""
        logger = logging.getLogger('FundingAnomalyDetection')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(os.path.join(self.output_dir, 'anomaly_detection.log'))
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger
    
    def load_data_from_json_files(self, base_dir=None):
        """
        Load funding data from JSON files in the specified directory.
        
        Args:
            base_dir (str): Directory containing JSON files
            
        Returns:
            pandas.DataFrame: Combined DataFrame with all funding data
        """
        if base_dir is None:
            base_dir = self.data_dir
            
        self.logger.info(f"Loading data from JSON files in {base_dir}")
        
        all_records = []
        
        # Walk through the directory and process JSON files
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    self.logger.info(f"Processing {file_path}")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # Check if the data is a list of records
                        if isinstance(data, list):
                            # Add source file information
                            for record in data:
                                record['source_file'] = file
                            all_records.extend(data)
                            self.logger.info(f"Added {len(data)} records from {file}")
                        # Check if the data is a dict with a companies key
                        elif isinstance(data, dict) and 'companies' in data and isinstance(data['companies'], list):
                            self.logger.info(f"Found nested companies array with {len(data['companies'])} records in {file}")
                            # Add source file information to each company record
                            for record in data['companies']:
                                record['source_file'] = file
                            all_records.extend(data['companies'])
                            self.logger.info(f"Added {len(data['companies'])} records from {file}")
                        else:
                            self.logger.warning(f"Skipping {file_path} - not a list of records or doesn't have a companies array")
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse JSON in {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {str(e)}")
        
        if not all_records:
            self.logger.error("No records found in JSON files")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        
        # Parse funding amounts
        if 'funding_amount' in df.columns:
            df['funding_amount_numeric'] = df['funding_amount'].apply(self._parse_funding_amount)
        
        # Parse funding dates
        if 'last_funding_date' in df.columns:
            df['funding_date'] = pd.to_datetime(df['last_funding_date'], errors='coerce')
        
        self.logger.info(f"Loaded {len(df)} records from JSON files")
        return df
    
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
        
        # Remove currency symbols and commas
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
        except ValueError:
            return np.nan
    
    def create_anomaly_features(self, data):
        """
        Create features for anomaly detection from the funding data.
        
        Args:
            data (pandas.DataFrame): Input data with funding information
            
        Returns:
            pandas.DataFrame: DataFrame with anomaly detection features
        """
        self.logger.info("Creating features for anomaly detection")
        
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Calculate funding amount per employee or estimate based on industry
        if 'employees' in df.columns:
            # Ensure employees is numeric
            try:
                df['employees'] = pd.to_numeric(df['employees'], errors='coerce')
                df['funding_per_employee'] = df['funding_amount_numeric'] / df['employees'].fillna(1).clip(lower=1)
                self.logger.info("Created funding_per_employee feature")
            except Exception as e:
                self.logger.warning(f"Could not create funding_per_employee feature: {str(e)}")
        
        # Calculate industry average funding amounts
        if 'industry' in df.columns and 'funding_amount_numeric' in df.columns:
            # Split multi-industry strings and create separate rows
            industry_expanded = df.copy()
            industry_expanded['industry'] = industry_expanded['industry'].str.split(',')
            industry_expanded = industry_expanded.explode('industry')
            industry_expanded['industry'] = industry_expanded['industry'].str.strip()
            
            # Calculate industry stats
            industry_stats = industry_expanded.groupby('industry')['funding_amount_numeric'].agg(['mean', 'std'])
            
            # Store industry means and stds for later use
            self.industry_means = dict(industry_stats['mean'])
            self.industry_stds = dict(industry_stats['std'].fillna(1))  # Avoid zero division
            
            # For each company, calculate deviation from industry mean
            df['industry_list'] = df['industry'].str.split(',')
            
            def calc_industry_deviation(row):
                if pd.isna(row['funding_amount_numeric']) or not isinstance(row['industry_list'], list):
                    return np.nan
                
                deviations = []
                for ind in row['industry_list']:
                    ind = ind.strip()
                    if ind in self.industry_means and ind in self.industry_stds:
                        dev = (row['funding_amount_numeric'] - self.industry_means[ind]) / self.industry_stds[ind]
                        deviations.append(dev)
                
                return np.mean(deviations) if deviations else np.nan
            
            df['industry_deviation'] = df.apply(calc_industry_deviation, axis=1)
        
        # Calculate funding type relative position (seed -> series A -> ...)
        funding_type_order = {
            'Pre-Seed': 1,
            'Seed': 2,
            'Series A': 3,
            'Series B': 4,
            'Series C': 5,
            'Series D': 6,
            'Series E': 7,
            'Series F': 8,
            'Series G': 9,
            'Series H': 10,
            'Initial Coin Offering': 3.5,  # Approximate position
            'Venture - Series Unknown': 3.5,  # Approximate position
            'Private Equity': 8,  # Approximate position
        }
        
        if 'funding_type' in df.columns:
            df['funding_type_numeric'] = df['funding_type'].map(lambda x: funding_type_order.get(x, np.nan))
            
            # Calculate expected funding based on funding type
            funding_type_avg = df.groupby('funding_type')['funding_amount_numeric'].mean().to_dict()
            df['expected_funding'] = df['funding_type'].map(funding_type_avg)
            df['funding_type_deviation'] = df['funding_amount_numeric'] / df['expected_funding'].clip(lower=1) - 1
        
        # Create features for anomaly detection
        anomaly_features = pd.DataFrame(index=df.index)
        
        # Feature 1: Funding amount z-score (overall)
        mean_funding = df['funding_amount_numeric'].mean()
        std_funding = df['funding_amount_numeric'].std()
        anomaly_features['funding_amount_zscore'] = (df['funding_amount_numeric'] - mean_funding) / std_funding
        
        # Feature 2: Industry deviation
        if 'industry_deviation' in df.columns:
            anomaly_features['industry_deviation'] = df['industry_deviation']
        
        # Feature 3: Funding type deviation
        if 'funding_type_deviation' in df.columns:
            anomaly_features['funding_type_deviation'] = df['funding_type_deviation']
        
        # Feature 4: Funding per employee deviation
        if 'funding_per_employee' in df.columns:
            mean_fpe = df['funding_per_employee'].mean()
            std_fpe = df['funding_per_employee'].std()
            anomaly_features['funding_per_employee_zscore'] = (df['funding_per_employee'] - mean_fpe) / std_fpe
        
        # Fill NaN values with 0 for the anomaly detection
        anomaly_features = anomaly_features.fillna(0)
        
        self.logger.info(f"Created {len(anomaly_features.columns)} features for anomaly detection")
        return anomaly_features
    
    def fit_isolation_forest(self, features, contamination=None):
        """
        Fit an Isolation Forest model for anomaly detection.
        
        Args:
            features (pandas.DataFrame): Features for anomaly detection
            contamination (float): Expected proportion of anomalies
            
        Returns:
            self
        """
        self.logger.info(f"Fitting Isolation Forest with contamination={contamination}")
        
        # Use specified contamination value or default
        if contamination is None:
            contamination = self.contamination
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(scaled_features)
        
        self.logger.info("Isolation Forest model trained successfully")
        return self
    
    def detect_anomalies(self, data, features=None, contamination=None):
        """
        Detect anomalies in the funding data.
        
        Args:
            data (pandas.DataFrame): Input data with funding information
            features (pandas.DataFrame): Pre-computed features for anomaly detection
            contamination (float): Expected proportion of anomalies
            
        Returns:
            pandas.DataFrame: Original data with anomaly scores and flags
        """
        self.logger.info("Detecting anomalies in funding data")
        
        # Use specified contamination value or default
        if contamination is None:
            contamination = self.contamination
        
        # Create a copy of the original data
        result_df = data.copy()
        
        # Create features if not provided
        if features is None:
            features = self.create_anomaly_features(data)
        
        # Fit the model if not already fitted
        if self.model is None:
            self.fit_isolation_forest(features, contamination)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Predict anomalies using Isolation Forest
        result_df['anomaly_score'] = self.model.decision_function(scaled_features)
        result_df['is_anomaly'] = self.model.predict(scaled_features) == -1
        
        # Calculate anomaly severity (lower score = more anomalous)
        result_df['anomaly_severity'] = 1 / (1 + np.exp(result_df['anomaly_score']))
        
        # Calculate statistical anomalies (3-sigma rule)
        if 'industry_deviation' in features.columns:
            result_df['beyond_3sigma'] = np.abs(features['industry_deviation']) > 3
            
            # Mark as anomalous if beyond 3-sigma from industry mean
            result_df['is_anomaly'] = result_df['is_anomaly'] | result_df['beyond_3sigma']
        
        # Identify extreme funding/employee ratio outliers if available
        if 'funding_per_employee_zscore' in features.columns:
            result_df['funding_per_employee_outlier'] = np.abs(features['funding_per_employee_zscore']) > 3
            
            # Mark as anomalous if extreme funding/employee ratio
            result_df['is_anomaly'] = result_df['is_anomaly'] | result_df['funding_per_employee_outlier']
        
        # Flag unusually large funding amounts for a given funding stage
        if 'funding_type_deviation' in features.columns:
            result_df['funding_type_outlier'] = np.abs(features['funding_type_deviation']) > 3
            
            # Mark as anomalous if extreme funding amount for stage
            result_df['is_anomaly'] = result_df['is_anomaly'] | result_df['funding_type_outlier']
            
        # Assign anomaly types
        def determine_anomaly_type(row):
            if not row['is_anomaly']:
                return np.nan
                
            types = []
            
            if 'beyond_3sigma' in row and row['beyond_3sigma']:
                types.append('Industry Outlier')
                
            if 'funding_per_employee_outlier' in row and row['funding_per_employee_outlier']:
                types.append('Funding/Employee Ratio')
                
            if 'funding_type_outlier' in row and row['funding_type_outlier']:
                types.append('Unusual for Funding Stage')
                
            if len(types) == 0 and row['is_anomaly']:
                types.append('Complex Pattern')
                
            return ', '.join(types)
            
        result_df['anomaly_type'] = result_df.apply(determine_anomaly_type, axis=1)
        
        # Count anomalies
        num_anomalies = result_df['is_anomaly'].sum()
        self.logger.info(f"Detected {num_anomalies} anomalies ({num_anomalies/len(result_df)*100:.2f}%)")
        
        return result_df
    
    def visualize_anomalies(self, data, output_dir=None):
        """
        Create visualizations for the detected anomalies.
        
        Args:
            data (pandas.DataFrame): Data with anomaly detection results
            output_dir (str): Directory to save visualizations
            
        Returns:
            None
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("Creating anomaly visualizations")
        
        # 1. Anomaly score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data['anomaly_score'], kde=True)
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score (lower = more anomalous)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anomaly_score_distribution.png'), dpi=300)
        plt.close()
        
        # 2. Scatter plot of funding amount vs anomaly score
        if 'funding_amount_numeric' in data.columns:
            # Filter out NaN or infinite values
            mask = np.isfinite(data['funding_amount_numeric']) & np.isfinite(data['anomaly_score'])
            if mask.sum() > 0:
                plt.figure(figsize=(10, 6))
                colors = np.where(data.loc[mask, 'is_anomaly'], 'red', 'blue')
                
                scatter = plt.scatter(
                    data.loc[mask, 'funding_amount_numeric'], 
                    data.loc[mask, 'anomaly_score'],
                    c=colors,
                    alpha=0.6
                )
                
                # Create a custom legend instead of colorbar
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Anomaly'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal')
                ]
                plt.legend(handles=legend_elements)
                
                plt.title('Funding Amount vs Anomaly Score')
                plt.xlabel('Funding Amount')
                plt.xscale('log')
                plt.ylabel('Anomaly Score (lower = more anomalous)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'funding_vs_anomaly.png'), dpi=300)
                plt.close()
        
        # 3. PCA visualization
        if 'funding_amount_numeric' in data.columns and 'funding_type_numeric' in data.columns:
            # Replace NaN values and filter finite values for PCA
            features_for_pca = data[['funding_amount_numeric', 'funding_type_numeric']].copy()
            features_for_pca = features_for_pca.fillna(0)
            
            # Make sure we have finite values only
            mask = np.isfinite(features_for_pca['funding_amount_numeric']) & np.isfinite(features_for_pca['funding_type_numeric'])
            if mask.sum() > 2:  # Need at least 3 points for meaningful PCA
                features_for_pca = features_for_pca[mask]
                
                # Standardize
                pca_scaler = StandardScaler()
                scaled_features = pca_scaler.fit_transform(features_for_pca)
                
                # PCA
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(scaled_features)
                
                # Create DataFrame for plotting
                pca_df = pd.DataFrame(
                    data=principal_components, 
                    columns=['PC1', 'PC2']
                )
                
                # Get corresponding anomaly severity values
                severity_values = data.loc[features_for_pca.index, 'anomaly_severity']
                
                # Plot
                plt.figure(figsize=(10, 6))
                scatter = plt.scatter(
                    pca_df['PC1'],
                    pca_df['PC2'],
                    c=severity_values,
                    cmap='coolwarm',
                    alpha=0.7,
                    s=50
                )
                plt.colorbar(scatter, label='Anomaly Severity')
                plt.title('PCA of Funding Features with Anomaly Detection')
                plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
                plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'anomaly_pca.png'), dpi=300)
                plt.close()
            else:
                self.logger.warning("Not enough valid data points for PCA visualization")
        
        # 4. Industry-specific anomaly rates
        if 'industry' in data.columns:
            try:
                # Split multi-industry strings and create separate rows
                industry_expanded = data.copy()
                industry_expanded['industry'] = industry_expanded['industry'].str.split(',')
                industry_expanded = industry_expanded.explode('industry')
                industry_expanded['industry'] = industry_expanded['industry'].str.strip()
                
                # Calculate anomaly rate by industry
                industry_anomaly = industry_expanded.groupby('industry')['is_anomaly'].agg(['mean', 'count'])
                industry_anomaly = industry_anomaly.sort_values('mean', ascending=False)
                
                # Filter to industries with at least 3 companies
                industry_anomaly = industry_anomaly[industry_anomaly['count'] >= 3]
                
                # Plot top 15 industries by anomaly rate or all if less than 15
                if len(industry_anomaly) > 0:
                    plt.figure(figsize=(12, 8))
                    industry_anomaly.head(15)['mean'].plot(kind='bar', color='coral')
                    plt.title('Anomaly Rate by Industry (Top 15)')
                    plt.xlabel('Industry')
                    plt.ylabel('Anomaly Rate')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'industry_anomaly_rate.png'), dpi=300)
                    plt.close()
                else:
                    self.logger.warning("No industries with sufficient data for anomaly rate visualization")
            except Exception as e:
                self.logger.error(f"Error creating industry anomaly visualization: {str(e)}")
        
        # Add new visualization: Model calibration plot
        self.visualize_model_calibration(data, output_dir)
        
        self.logger.info("Anomaly visualizations created successfully")
    
    def visualize_model_calibration(self, data, output_dir=None):
        """
        Create calibration plot and performance metrics visualization.
        
        Args:
            data (pandas.DataFrame): Data with anomaly detection results
            output_dir (str): Directory to save visualizations
            
        Returns:
            dict: Performance metrics
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("Creating model calibration and performance visualizations")
        
        # Initialize metrics
        metrics = {}
        
        # 1. Split data into train/test sets
        try:
            # Check if we have enough anomalies to split
            if data['is_anomaly'].sum() >= 5:  # Need at least a few anomalies in each set
                # Create feature matrix for evaluation
                X = self.create_anomaly_features(data)
                y = data['is_anomaly'].astype(int)
                
                # Stratified split to maintain anomaly ratio
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Recalibrate model on training set
                self.fit_isolation_forest(X_train)
                
                # Get predictions on test set
                # Transform features
                X_test_scaled = self.scaler.transform(X_test)
                
                # Get raw scores
                test_scores = self.model.decision_function(X_test_scaled)
                
                # Convert to severity (higher is more anomalous)
                test_severities = 1 / (1 + np.exp(test_scores))
                
                # Convert decision function to binary predictions
                test_predictions = (self.model.predict(X_test_scaled) == -1).astype(int)
                
                # Calculate metrics
                metrics['accuracy'] = accuracy_score(y_test, test_predictions)
                metrics['precision'] = precision_score(y_test, test_predictions, zero_division=0)
                metrics['recall'] = recall_score(y_test, test_predictions, zero_division=0)
                metrics['f1'] = f1_score(y_test, test_predictions, zero_division=0)
                
                # Calculate RMSE on anomaly severity
                # Convert binary labels to severity scale for RMSE calculation
                y_test_severity = y_test.astype(float)
                metrics['rmse'] = np.sqrt(mean_squared_error(y_test_severity, test_severities))
                
                # Create calibration plot
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 1. Plot calibration curve
                prob_true, prob_pred = calibration_curve(
                    y_test, test_severities, n_bins=10, strategy='uniform'
                )
                
                ax.plot(prob_pred, prob_true, marker='o', linewidth=2, 
                        label='Calibration curve', color='darkblue')
                
                # Plot the ideal calibration curve
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', 
                        label='Perfectly calibrated')
                
                # Add metrics to the plot
                metrics_text = (
                    f"Accuracy: {metrics['accuracy']:.3f}\n"
                    f"Precision: {metrics['precision']:.3f}\n"
                    f"Recall: {metrics['recall']:.3f}\n"
                    f"F1 Score: {metrics['f1']:.3f}\n"
                    f"RMSE: {metrics['rmse']:.3f}"
                )
                
                # Add a box with metrics
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
                
                ax.set_title('Anomaly Detection Calibration Plot')
                ax.set_xlabel('Mean predicted probability')
                ax.set_ylabel('Fraction of positive samples')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'model_calibration.png'), dpi=300)
                plt.close()
                
                # 2. Create metrics visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics_for_plot = {k: v for k, v in metrics.items() if k != 'rmse'}
                
                bars = ax.bar(metrics_for_plot.keys(), metrics_for_plot.values(), color='skyblue')
                
                # Add values on top of each bar
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
                
                ax.set_ylim(0, 1.1)
                ax.set_title('Anomaly Detection Performance Metrics')
                ax.set_ylabel('Score')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300)
                plt.close()
                
                # 3. Create confusion matrix visualization
                cm = confusion_matrix(y_test, test_predictions)
                
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                           xticklabels=['Normal', 'Anomaly'],
                           yticklabels=['Normal', 'Anomaly'])
                
                ax.set_title('Confusion Matrix for Anomaly Detection')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
                plt.close()
                
                # Save metrics to file
                with open(os.path.join(output_dir, 'model_performance.json'), 'w') as f:
                    json.dump(metrics, f, indent=2)
                    
                self.logger.info(f"Model performance: Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, RMSE={metrics['rmse']:.3f}")
            else:
                self.logger.warning("Not enough anomalies for model calibration visualization")
                
                # Create empty plot with explanation
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "Insufficient anomalies for calibration plot\nAt least 5 anomalies required", 
                       ha='center', va='center', fontsize=14)
                ax.set_title('Model Calibration (Unavailable)')
                ax.set_xlabel('Mean predicted probability')
                ax.set_ylabel('Fraction of positive samples')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'model_calibration.png'), dpi=300)
                plt.close()
        except Exception as e:
            self.logger.error(f"Error creating calibration visualization: {str(e)}")
            # Create error plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error creating calibration plot:\n{str(e)}", 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Model Calibration (Error)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'model_calibration.png'), dpi=300)
            plt.close()
        
        return metrics
    
    def generate_anomaly_report(self, data, output_dir=None):
        """
        Generate a report of the detected anomalies.
        
        Args:
            data (pandas.DataFrame): Data with anomaly detection results
            output_dir (str): Directory to save the report
            
        Returns:
            None
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("Generating anomaly detection report")
        
        # Select anomalies
        anomalies = data[data['is_anomaly']].copy()
        
        # Sort by anomaly severity
        anomalies = anomalies.sort_values('anomaly_severity', ascending=False)
        
        # Select columns for report
        report_columns = [
            'name', 'industry', 'funding_amount', 'funding_type', 
            'last_funding_date', 'anomaly_score', 'anomaly_severity', 'anomaly_type'
        ]
        
        report_columns = [col for col in report_columns if col in anomalies.columns]
        
        # Generate report
        report_path = os.path.join(output_dir, 'anomaly_report.csv')
        anomalies[report_columns].to_csv(report_path, index=False)
        
        # Generate summary statistics
        total_companies = len(data)
        total_anomalies = len(anomalies)
        
        with open(os.path.join(output_dir, 'anomaly_summary.txt'), 'w') as f:
            f.write("Funding Anomaly Detection Summary\n")
            f.write("=================================\n\n")
            f.write(f"Total companies analyzed: {total_companies}\n")
            f.write(f"Anomalies detected: {total_anomalies} ({total_anomalies/total_companies*100:.2f}%)\n\n")
            
            if 'funding_type' in anomalies.columns:
                f.write("Anomalies by Funding Type:\n")
                funding_type_counts = anomalies['funding_type'].value_counts()
                for funding_type, count in funding_type_counts.items():
                    f.write(f"  {funding_type}: {count}\n")
                f.write("\n")
            
            if 'industry' in anomalies.columns:
                f.write("Top 10 Industries with Anomalies:\n")
                # Split and count industry anomalies
                industry_expanded = anomalies.copy()
                industry_expanded['industry'] = industry_expanded['industry'].str.split(',')
                industry_expanded = industry_expanded.explode('industry')
                industry_expanded['industry'] = industry_expanded['industry'].str.strip()
                
                industry_counts = industry_expanded['industry'].value_counts().head(10)
                for industry, count in industry_counts.items():
                    f.write(f"  {industry}: {count}\n")
                
                f.write("\n")
            
            if 'anomaly_type' in anomalies.columns:
                f.write("Anomalies by Type:\n")
                # Count by primary anomaly type (first one listed)
                anomaly_types = anomalies['anomaly_type'].str.split(', ').str[0].value_counts()
                for anomaly_type, count in anomaly_types.items():
                    f.write(f"  {anomaly_type}: {count}\n")
                
        self.logger.info(f"Anomaly report saved to {report_path}")
        
        # Create JSON file for integration with other components
        integration_data = {
            'anomalies': anomalies[report_columns].to_dict(orient='records'),
            'summary': {
                'total_companies': total_companies,
                'total_anomalies': total_anomalies,
                'anomaly_rate': total_anomalies/total_companies,
                'anomaly_by_funding_type': anomalies['funding_type'].value_counts().to_dict() if 'funding_type' in anomalies.columns else {},
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(os.path.join(output_dir, 'anomaly_results.json'), 'w') as f:
            json.dump(integration_data, f, indent=2)
        
        self.logger.info(f"Integration data saved to {os.path.join(output_dir, 'anomaly_results.json')}")
    
    def save_model(self, filepath=None):
        """
        Save the trained anomaly detection model to a file.
        
        Args:
            filepath (str): Path to save the model file
            
        Returns:
            str: Path to the saved model
        """
        if self.model is None:
            self.logger.error("No model to save")
            return None
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"anomaly_model_{timestamp}.pkl")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'industry_means': self.industry_means,
            'industry_stds': self.industry_stds,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """
        Load a trained anomaly detection model from a file.
        
        Args:
            filepath (str): Path to the model file
            
        Returns:
            self
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.industry_means = model_data['industry_means']
            self.industry_stds = model_data['industry_stds']
            
            self.logger.info(f"Model loaded from {filepath}")
            return self
        except Exception as e:
            self.logger.error(f"Failed to load model from {filepath}: {str(e)}")
            return None
    
    def run_analysis(self):
        """
        Run the complete anomaly detection analysis pipeline.
        
        Returns:
            pandas.DataFrame: Results with anomaly detection
        """
        self.logger.info("Starting anomaly detection analysis")
        
        # Load data
        data = self.load_data_from_json_files()
        
        if len(data) == 0:
            self.logger.error("No data loaded, aborting analysis")
            return None
        
        # Create features
        features = self.create_anomaly_features(data)
        
        # Detect anomalies
        results = self.detect_anomalies(data, features)
        
        # Visualize results
        self.visualize_anomalies(results)
        
        # Generate report
        self.generate_anomaly_report(results)
        
        # Save model
        self.save_model()
        
        self.logger.info("Anomaly detection analysis completed")
        return results

    def get_risk_score_contribution(self, data):
        """
        Calculate risk score contribution based on anomaly detection.
        This method is used for integration with the meta-model.
        
        Args:
            data (pandas.DataFrame): Data with company information
            
        Returns:
            pandas.DataFrame: DataFrame with company names and risk scores
        """
        # Detect anomalies if not already done
        if 'anomaly_severity' not in data.columns:
            data = self.detect_anomalies(data)
        
        # Calculate risk score (0-100, higher is riskier)
        risk_scores = pd.DataFrame({
            'company_name': data['name'],
            'anomaly_risk_score': data['anomaly_severity'] * 100
        })
        
        return risk_scores
    
    def create_prediction_features(self, new_company_data):
        """
        Create anomaly detection features for a new company.
        Ensures compatibility with the original feature set used for training.
        
        Args:
            new_company_data (DataFrame): Data for new company
            
        Returns:
            DataFrame: Features for anomaly detection
        """
        self.logger.info("Creating prediction features for new company")
        
        # Create basic features
        features = self.create_anomaly_features(new_company_data)
        
        # Ensure all features used during training are present
        required_features = [
            'funding_amount_zscore', 'industry_deviation', 
            'funding_type_deviation', 'funding_per_employee_zscore'
        ]
        
        # Add missing features with default values (0)
        for feature in required_features:
            if feature not in features.columns:
                features[feature] = 0
                self.logger.info(f"Added missing feature: {feature}")
        
        self.logger.info(f"Created {len(features.columns)} prediction features for new company")
        return features
        
    def predict_for_new_company(self, company_data):
        """
        Predict if a new company is an anomaly.
        
        Args:
            company_data (dict or DataFrame): Company data
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            self.logger.error("Model not trained, cannot make predictions")
            return {'error': 'Model not trained'}
        
        # Convert to DataFrame if dict
        if isinstance(company_data, dict):
            company_df = pd.DataFrame([company_data])
        else:
            company_df = company_data.copy()
        
        # Ensure numeric funding amount
        if 'funding_amount' in company_df.columns:
            company_df['funding_amount_numeric'] = company_df['funding_amount'].apply(self._parse_funding_amount)
        
        try:
            # Create features for the new company with all required features
            company_features = self.create_prediction_features(company_df)
            
            # Ensure features are in the same order as during training
            if hasattr(self.model, 'feature_names_in_'):
                # Get feature names from the model
                model_features = self.model.feature_names_in_
                
                # Ensure all required features are present
                for feature in model_features:
                    if feature not in company_features.columns:
                        company_features[feature] = 0
                
                # Reorder to match training order
                company_features = company_features[model_features]
            
            # Scale features
            scaled_features = self.scaler.transform(company_features)
            
            # Predict using isolation forest
            anomaly_score = self.model.decision_function(scaled_features)[0]
            is_anomaly = self.model.predict(scaled_features)[0] == -1
            anomaly_severity = 1 / (1 + np.exp(anomaly_score))
            
            # Handle statistical anomalies
            industry_outlier = False
            funding_type_outlier = False
            funding_employee_outlier = False
            
            if 'industry_deviation' in company_features.columns:
                industry_outlier = abs(company_features['industry_deviation'].iloc[0]) > 3
                is_anomaly = is_anomaly or industry_outlier
                
            if 'funding_type_deviation' in company_features.columns:
                funding_type_outlier = abs(company_features['funding_type_deviation'].iloc[0]) > 3
                is_anomaly = is_anomaly or funding_type_outlier
                
            if 'funding_per_employee_zscore' in company_features.columns:
                funding_employee_outlier = abs(company_features['funding_per_employee_zscore'].iloc[0]) > 3
                is_anomaly = is_anomaly or funding_employee_outlier
            
            # Determine anomaly type
            anomaly_type = []
            if industry_outlier:
                anomaly_type.append('Industry Outlier')
            if funding_type_outlier:
                anomaly_type.append('Unusual for Funding Stage')
            if funding_employee_outlier:
                anomaly_type.append('Funding/Employee Ratio')
            if not anomaly_type and is_anomaly:
                anomaly_type.append('Complex Pattern')
            
            anomaly_type_str = ', '.join(anomaly_type) if anomaly_type else None
            
            # Add results to the company data
            result = {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'anomaly_severity': float(anomaly_severity),
                'anomaly_type': anomaly_type_str,
                'explanation': self._generate_explanation({
                    'name': company_df['name'].iloc[0],
                    'is_anomaly': is_anomaly,
                    'anomaly_severity': anomaly_severity,
                    'anomaly_type': anomaly_type_str,
                    'funding_type': company_df['funding_type'].iloc[0] if 'funding_type' in company_df.columns else None
                })
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting for new company: {str(e)}")
            return {
                'error': str(e),
                'is_anomaly': False,
                'anomaly_score': 0,
                'anomaly_severity': 0.5,
                'explanation': f"Could not analyze company: {str(e)}"
            }
    
    def _generate_explanation(self, result_row):
        """
        Generate a human-readable explanation of why a company is flagged as an anomaly.
        
        Args:
            result_row (Series): Row from the results DataFrame for a single company
            
        Returns:
            str: Explanation of the anomaly
        """
        if not result_row['is_anomaly']:
            return "This company's funding pattern appears normal."
        
        company_name = result_row['name']
        explanation = f"{company_name} has been flagged as an anomaly because: "
        
        if 'anomaly_type' in result_row and not pd.isna(result_row['anomaly_type']):
            anomaly_types = result_row['anomaly_type'].split(', ')
            
            reasons = []
            for anomaly_type in anomaly_types:
                if anomaly_type == 'Industry Outlier':
                    reasons.append(f"its funding amount is unusual compared to other companies in the same industry")
                elif anomaly_type == 'Funding/Employee Ratio':
                    reasons.append(f"its funding amount per employee is significantly different from typical values")
                elif anomaly_type == 'Unusual for Funding Stage':
                    reasons.append(f"its funding amount is unusual for its funding stage ({result_row['funding_type']})")
                elif anomaly_type == 'Complex Pattern':
                    reasons.append(f"it exhibits an unusual combination of funding characteristics")
            
            explanation += ", ".join(reasons) + "."
        else:
            explanation += "it exhibits unusual funding patterns compared to other companies."
        
        # Add severity information
        severity = result_row['anomaly_severity']
        if severity > 0.8:
            explanation += " This is an extreme anomaly that requires immediate investigation."
        elif severity > 0.6:
            explanation += " This is a significant anomaly that should be investigated."
        else:
            explanation += " This is a moderate anomaly worth looking into."
        
        return explanation


def integrate_with_meta_model(anomaly_results, funding_stage_results=None, funding_continuation_results=None):
    """
    Integrate anomaly detection results with other predictive components.
    
    Args:
        anomaly_results (DataFrame): Results from anomaly detection
        funding_stage_results (DataFrame): Results from funding stage prediction
        funding_continuation_results (DataFrame): Results from funding continuation analysis
        
    Returns:
        DataFrame: Integrated results for meta-model
    """
    # Start with anomaly results
    integrated_df = anomaly_results[['name', 'anomaly_severity']].copy()
    integrated_df.rename(columns={'name': 'company_name'}, inplace=True)
    
    # Add funding stage results if available
    if funding_stage_results is not None and 'company_name' in funding_stage_results.columns:
        stage_df = funding_stage_results[['company_name', 'predicted_stage', 'stage_probability']]
        integrated_df = integrated_df.merge(stage_df, on='company_name', how='left')
    
    # Add funding continuation results if available
    if funding_continuation_results is not None and 'company_name' in funding_continuation_results.columns:
        cont_df = funding_continuation_results[['company_name', 'survival_probability', 'expected_duration']]
        integrated_df = integrated_df.merge(cont_df, on='company_name', how='left')
    
    # Calculate meta-model score
    # This is a simple weighted example that could be replaced with a more sophisticated model
    def calculate_success_score(row):
        score = 0
        weight_sum = 0
        
        # Anomaly component (lower anomaly severity = higher success chance)
        if 'anomaly_severity' in row and not pd.isna(row['anomaly_severity']):
            score += (1 - row['anomaly_severity']) * 30
            weight_sum += 30
        
        # Stage component (higher stage = higher success)
        if 'predicted_stage' in row and not pd.isna(row['predicted_stage']):
            stage_map = {
                'Pre-Seed': 10,
                'Seed': 20,
                'Series A': 40, 
                'Series B': 60,
                'Series C': 80,
                'Series D+': 100
            }
            stage_score = stage_map.get(row['predicted_stage'], 0)
            stage_prob = row.get('stage_probability', 1)
            score += stage_score * stage_prob * 30 / 100
            weight_sum += 30
        
        # Continuation component
        if 'survival_probability' in row and not pd.isna(row['survival_probability']):
            score += row['survival_probability'] * 40
            weight_sum += 40
        
        # Normalize by weights
        return (score / weight_sum * 100) if weight_sum > 0 else 50
    
    integrated_df['success_score'] = integrated_df.apply(calculate_success_score, axis=1)
    
    # Classify as success/failure
    integrated_df['predicted_success'] = integrated_df['success_score'] > 50
    
    return integrated_df


def main():
    """Main function to run the anomaly detection analysis"""
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), 'JSONFolder')
    output_dir = os.path.join(base_dir, 'output', 'anomaly_detection')
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize anomaly detection
    anomaly_detector = FundingAnomalyDetection(data_dir=data_dir, output_dir=output_dir)
    
    # Run analysis now and then every 24 hours
    interval_hours = 24
    print(f"\n===== FUNDING ANOMALY DETECTION =====")
    print(f"Starting analysis with automatic 24-hour scheduling")
    print(f"First run starting now, will repeat every 24 hours")
    
    # Run in a loop that repeats every 24 hours
    while True:
        try:
            # Run the analysis
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running anomaly detection analysis...")
            results = anomaly_detector.run_analysis()
            
            if results is not None:
                # Print summary of anomalies
                anomaly_count = results['is_anomaly'].sum()
                total_count = len(results)
                print(f"Analysis complete. Detected {anomaly_count} anomalies out of {total_count} companies ({anomaly_count/total_count*100:.2f}%)")
                
                # Schedule next run
                next_run = datetime.now() + timedelta(hours=interval_hours)
                print(f"Next analysis scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')} (in {interval_hours} hours)")
                print("The program will continue running in the background.")
                print("Close this window or press Ctrl+C to stop the automatic scheduling.")
                
                # Sleep until next run
                time.sleep(interval_hours * 3600)
            else:
                print("Analysis failed. Will retry in 1 hour.")
                time.sleep(3600)  # Wait 1 hour before retrying
        except KeyboardInterrupt:
            print("\nAutomatic scheduling stopped by user.")
            break
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            print("Will retry in 1 hour.")
            time.sleep(3600)  # Wait 1 hour before retrying


# Add this code to run the main function if the script is run directly
if __name__ == "__main__":
    main() 