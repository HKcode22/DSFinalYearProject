#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integrated Predictive Analytics Framework for Startup Funding

This module implements a comprehensive system integrating multiple predictive models:
1. Funding Stage Prediction
2. Funding Continuation Analysis
3. Funding Anomaly Detection
4. Funding Amount Forecast
5. Industry Trend Analysis

Together, these models form a meta-model for startup success/failure prediction.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import seaborn as sns
from datetime import datetime, timedelta
import logging
import argparse
import sys
import time
import joblib
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Optional, Tuple, Union
import threading

# Import individual prediction modules
from MLPredictiveAnalysis.funding_stage_prediction import EnhancedPipeline as FundingStagePipeline
from MLPredictiveAnalysis.funding_continuation import FundingContinuationAnalysis
from MLPredictiveAnalysis.funding_amount_forecast import FundingAmountForecast
from MLPredictiveAnalysis.funding_anomaly_detection import FundingAnomalyDetection
from MLPredictiveAnalysis.industry_trend_analysis import IndustryTrendAnalysis

class IntegratedPredictiveSystem:
    """
    Core Architecture: Meta-model aggregating insights from all components
    
    Input Features Weight Source Model
    Stage transition probability 30% XGBoost Classifier
    18-month survival probability 25% Cox Model
    Funding adequacy score 20% QRF Forecast
    Industry growth momentum 15% STL Decomposition
    Anomaly severity 10% Isolation Forest
    
    Success Definition:
    - Received Series B+ funding within 3 years
    - Not marked as outlier in 2 consecutive quarters
    - Employee growth > industry 75th percentile
    
    Failure Signals:
    - Survival probability <40% for 6 months
    - Burn rate > industry 90th percentile
    - Anomaly score persists >3 months
    """
    
    def __init__(self, data_dir=None, output_dir=None, parallel=True):
        """
        Initialize the integrated predictive system.
        
        Args:
            data_dir (str): Directory containing funding data JSON files
            output_dir (str): Base directory to save output files, models, and visualizations
            parallel (bool): Whether to run components in parallel for faster processing
        """
        self.data_dir = data_dir or os.path.join(os.getcwd(), 'JSONFolder')
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'outputIntegrated')
        self.parallel = parallel
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize component models
        self.stage_predictor = None
        self.continuation_analyzer = None
        self.amount_forecaster = None
        self.anomaly_detector = None
        self.trend_analyzer = None
        
        # Initialize result storage
        self.results = {
            'stage_prediction': {},
            'continuation': {},
            'amount_forecast': {},
            'anomaly_detection': {},
            'industry_trends': {},
            'meta_model': {}
        }
        
        # Meta-model weights
        self.model_weights = {
            'stage_transition_probability': 0.30,
            'survival_probability': 0.25,
            'funding_adequacy': 0.20,
            'industry_momentum': 0.15,
            'anomaly_severity': 0.10
        }
        
        # Set up matplotlib parameters
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        self.logger.info("Integrated Predictive System initialized")
        
    def _setup_logging(self):
        """Set up logging configuration for the integrated system."""
        # Create a logger
        self.logger = logging.getLogger('integrated_predictive_system')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(os.path.join(self.output_dir, 'integrated_system.log'))
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
    
    def initialize_components(self):
        """Initialize all component prediction systems."""
        self.logger.info("Initializing component prediction systems")
        
        # Initialize funding stage prediction
        self.stage_predictor = FundingStagePipeline(
            base_dir=self.data_dir,
            output_dir=os.path.join(self.output_dir, 'stage_prediction')
        )
        
        # Initialize funding continuation analysis
        self.continuation_analyzer = FundingContinuationAnalysis(
            data_dir=self.data_dir,
            output_dir=os.path.join(self.output_dir, 'continuation')
        )
        
        # Initialize funding amount forecast
        self.amount_forecaster = FundingAmountForecast(
            data_dir=self.data_dir,
            output_dir=os.path.join(self.output_dir, 'amount_forecast')
        )
        
        # Initialize funding anomaly detection
        self.anomaly_detector = FundingAnomalyDetection(
            data_dir=self.data_dir,
            output_dir=os.path.join(self.output_dir, 'anomaly_detection')
        )
        
        # Initialize industry trend analysis
        self.trend_analyzer = IndustryTrendAnalysis(
            data_dir=self.data_dir,
            output_dir=os.path.join(self.output_dir, 'industry_trends')
        )
        
        self.logger.info("All component systems initialized")
    
    def run_component_analysis(self, component_name):
        """
        Run analysis for a specific component.
        
        Args:
            component_name (str): Name of the component to run
            
        Returns:
            dict: Results from the component analysis
        """
        self.logger.info(f"Running {component_name} analysis")
        
        try:
            if component_name == 'stage_prediction':
                if not self.stage_predictor:
                    self.stage_predictor = FundingStagePipeline(
                        base_dir=self.data_dir,
                        output_dir=os.path.join(self.output_dir, 'stage_prediction')
                    )
                result = self.stage_predictor.run()
                self.results['stage_prediction'] = result
                return result
                
            elif component_name == 'continuation':
                if not self.continuation_analyzer:
                    self.continuation_analyzer = FundingContinuationAnalysis(
                        data_dir=self.data_dir,
                        output_dir=os.path.join(self.output_dir, 'continuation')
                    )
                result = self.continuation_analyzer.run_analysis()
                self.results['continuation'] = result
                return result
                
            elif component_name == 'amount_forecast':
                if not self.amount_forecaster:
                    self.amount_forecaster = FundingAmountForecast(
                        data_dir=self.data_dir,
                        output_dir=os.path.join(self.output_dir, 'amount_forecast')
                    )
                result = self.amount_forecaster.run_analysis()
                self.results['amount_forecast'] = result
                return result
                
            elif component_name == 'anomaly_detection':
                if not self.anomaly_detector:
                    self.anomaly_detector = FundingAnomalyDetection(
                        data_dir=self.data_dir,
                        output_dir=os.path.join(self.output_dir, 'anomaly_detection')
                    )
                result = self.anomaly_detector.run_analysis()
                self.results['anomaly_detection'] = result
                return result
                
            elif component_name == 'industry_trends':
                if not self.trend_analyzer:
                    self.trend_analyzer = IndustryTrendAnalysis(
                        data_dir=self.data_dir,
                        output_dir=os.path.join(self.output_dir, 'industry_trends')
                    )
                result = self.trend_analyzer.run_analysis()
                self.results['industry_trends'] = result
                return result
                
            else:
                self.logger.error(f"Unknown component: {component_name}")
                return {'success': False, 'error': f"Unknown component: {component_name}"}
                
        except Exception as e:
            self.logger.error(f"Error running {component_name} analysis: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_all_components(self):
        """
        Run all component analyses, either in parallel or sequentially.
        
        Returns:
            dict: Dictionary of results from all components
        """
        self.logger.info("Running all component analyses")
        
        component_names = [
            'stage_prediction',
            'continuation',
            'amount_forecast',
            'anomaly_detection',
            'industry_trends'
        ]
        
        if self.parallel:
            # Run components in parallel using threads
            threads = []
            for component in component_names:
                thread = threading.Thread(target=self.run_component_analysis, args=(component,))
                threads.append(thread)
                thread.start()
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        else:
            # Run components sequentially
            for component in component_names:
                self.run_component_analysis(component)
                
        self.logger.info("All component analyses completed")
        
        return self.results
    
    def integrate_results(self):
        """
        Integrate results from all components to create meta-model predictions.
        
        Returns:
            dict: Integrated results with success/failure predictions
        """
        self.logger.info("Integrating results from all components")
        
        # Check if we have results from all components
        required_components = [
            'stage_prediction',
            'continuation',
            'amount_forecast',
            'anomaly_detection',
            'industry_trends'
        ]
        
        missing_components = [comp for comp in required_components if not self.results.get(comp)]
        if missing_components:
            self.logger.warning(f"Missing results from components: {missing_components}")
            
        # Create integrated dataset with company-level metrics
        integrated_data = self._create_integrated_dataset()
        
        if integrated_data.empty:
            self.logger.error("Could not create integrated dataset - insufficient data")
            return {'success': False, 'error': 'Insufficient data for integration'}
            
        # Calculate success scores
        success_scores = self._calculate_success_scores(integrated_data)
        
        # Classify companies into success/failure categories
        success_classification = self._classify_success_failure(success_scores)
        
        # Generate meta-model insights
        meta_insights = self._generate_meta_insights(success_classification)
        
        # Store results
        self.results['meta_model'] = {
            'success': True,
            'integrated_data': integrated_data,
            'success_scores': success_scores,
            'classification': success_classification,
            'insights': meta_insights
        }
        
        self.logger.info("Results integration completed")
        
        return self.results['meta_model']
    
    def _create_integrated_dataset(self):
        """
        Create an integrated dataset combining metrics from all components.
        
        Returns:
            pandas.DataFrame: Integrated dataset with company-level metrics
        """
        self.logger.info("Creating integrated dataset")
        
        # Start with an empty DataFrame
        integrated_data = pd.DataFrame()
        
        try:
            # Extract company data from stage prediction results
            if self.results.get('stage_prediction') and 'merged_data' in self.results['stage_prediction']:
                # Start with company names and basic information
                base_data = self.results['stage_prediction']['merged_data']
                
                # Select company identification columns
                integrated_data = base_data[['company_name', 'industry', 'funding_stage']].copy()
                
                # Add stage prediction probabilities if available
                if 'stage_probabilities' in self.results['stage_prediction']:
                    stage_probs = self.results['stage_prediction']['stage_probabilities']
                    if not stage_probs.empty and 'company_name' in stage_probs.columns:
                        # Merge with integrated data
                        integrated_data = pd.merge(
                            integrated_data,
                            stage_probs,
                            on='company_name',
                            how='left'
                        )
            
            # Add survival probabilities from continuation analysis
            if self.results.get('continuation') and 'survival_probabilities' in self.results['continuation']:
                survival_probs = self.results['continuation']['survival_probabilities']
                if not survival_probs.empty and 'company_name' in survival_probs.columns:
                    # Merge with integrated data
                    integrated_data = pd.merge(
                        integrated_data,
                        survival_probs,
                        on='company_name',
                        how='left'
                    )
            
            # Add funding amount predictions
            if self.results.get('amount_forecast') and 'predictions' in self.results['amount_forecast']:
                amount_preds = self.results['amount_forecast']['predictions']
                if not amount_preds.empty and 'company_name' in amount_preds.columns:
                    # Merge with integrated data
                    integrated_data = pd.merge(
                        integrated_data,
                        amount_preds,
                        on='company_name',
                        how='left'
                    )
            
            # Add anomaly scores
            if self.results.get('anomaly_detection') and 'anomaly_scores' in self.results['anomaly_detection']:
                anomaly_scores = self.results['anomaly_detection']['anomaly_scores']
                if not anomaly_scores.empty and 'company_name' in anomaly_scores.columns:
                    # Merge with integrated data
                    integrated_data = pd.merge(
                        integrated_data,
                        anomaly_scores,
                        on='company_name',
                        how='left'
                    )
            
            # Add industry trend metrics
            if self.results.get('industry_trends') and 'industry_metrics' in self.results['industry_trends']:
                industry_metrics = self.results['industry_trends']['industry_metrics']
                if not industry_metrics.empty:
                    # Join by industry
                    integrated_data = pd.merge(
                        integrated_data,
                        industry_metrics,
                        on='industry',
                        how='left'
                    )
            
            self.logger.info(f"Created integrated dataset with {len(integrated_data)} companies and {len(integrated_data.columns)} metrics")
            
        except Exception as e:
            self.logger.error(f"Error creating integrated dataset: {str(e)}")
            return pd.DataFrame()
            
        return integrated_data
    
    def _calculate_success_scores(self, integrated_data):
        """
        Calculate success scores for each company based on integrated metrics.
        
        Args:
            integrated_data (pandas.DataFrame): Integrated dataset with company metrics
            
        Returns:
            pandas.DataFrame: Dataset with success scores
        """
        self.logger.info("Calculating success scores")
        
        # Create a copy to avoid modifying the original
        data = integrated_data.copy()
        
        try:
            # Calculate component scores based on available metrics
            # 1. Stage transition score (higher stages = higher score)
            if 'next_stage_probability' in data.columns:
                data['stage_transition_score'] = data['next_stage_probability'] * 100
            elif 'funding_stage_numeric' in data.columns:
                # Normalize funding stage to 0-100 scale
                stage_max = data['funding_stage_numeric'].max()
                data['stage_transition_score'] = (data['funding_stage_numeric'] / stage_max) * 100
            else:
                # Map funding stages to numeric values
                stage_mapping = {
                    'Pre-Seed': 1,
                    'Seed': 2,
                    'Series A': 3,
                    'Series B': 4,
                    'Series C': 5,
                    'Series D': 6,
                    'Series E': 7,
                    'Series F': 8,
                    'Series G': 9,
                    'IPO': 10
                }
                data['stage_transition_score'] = data['funding_stage'].map(
                    lambda x: stage_mapping.get(x, 0) * 10  # Scale to 0-100
                )
            
            # 2. Survival score (18-month survival probability)
            if '18_month_survival' in data.columns:
                data['survival_score'] = data['18_month_survival'] * 100
            elif 'survival_probability' in data.columns:
                data['survival_score'] = data['survival_probability'] * 100
            else:
                # Default based on stage (higher stages have better survival)
                data['survival_score'] = data['stage_transition_score'] * 0.8
            
            # 3. Funding adequacy score
            if 'funding_adequacy' in data.columns:
                data['funding_adequacy_score'] = data['funding_adequacy'] * 100
            elif 'next_round_amount' in data.columns and 'predicted_required_amount' in data.columns:
                # Calculate ratio of predicted amount to required amount
                data['funding_adequacy_score'] = np.minimum(
                    data['next_round_amount'] / data['predicted_required_amount'], 
                    1.5  # Cap at 150%
                ) * 100
            else:
                # No adequacy metrics available
                data['funding_adequacy_score'] = 50  # Neutral score
            
            # 4. Industry momentum score
            if 'industry_momentum' in data.columns:
                data['industry_momentum_score'] = np.minimum(data['industry_momentum'] * 200, 100)  # Scale and cap
            elif 'industry_growth' in data.columns:
                data['industry_momentum_score'] = np.minimum(data['industry_growth'] * 200, 100)
            else:
                # No industry momentum metrics available
                data['industry_momentum_score'] = 50  # Neutral score
            
            # 5. Anomaly score (lower anomaly scores = higher normalized scores)
            if 'anomaly_score' in data.columns:
                # Invert anomaly score (anomaly scores are negative, with -1 being most anomalous)
                data['anomaly_normalized_score'] = (data['anomaly_score'] + 1) * 50  # Convert -1..0 to 0..50
            elif 'is_anomaly' in data.columns:
                # Binary anomaly flag
                data['anomaly_normalized_score'] = data['is_anomaly'].map({False: 100, True: 0})
            else:
                # No anomaly metrics available
                data['anomaly_normalized_score'] = 75  # Somewhat optimistic default
            
            # Calculate weighted success score
            data['success_score'] = (
                data['stage_transition_score'] * self.model_weights['stage_transition_probability'] +
                data['survival_score'] * self.model_weights['survival_probability'] +
                data['funding_adequacy_score'] * self.model_weights['funding_adequacy'] +
                data['industry_momentum_score'] * self.model_weights['industry_momentum'] +
                data['anomaly_normalized_score'] * self.model_weights['anomaly_severity']
            )
            
            self.logger.info("Success scores calculated successfully")
            
        except Exception as e:
            self.logger.error(f"Error calculating success scores: {str(e)}")
            # Add a default score column
            if 'success_score' not in data.columns:
                data['success_score'] = 50  # Neutral score
        
        return data
    
    def _classify_success_failure(self, scored_data):
        """
        Classify companies into success/failure categories based on scores.
        
        Args:
            scored_data (pandas.DataFrame): Data with success scores
            
        Returns:
            pandas.DataFrame: Data with success/failure classifications
        """
        self.logger.info("Classifying companies by success potential")
        
        # Create a copy to avoid modifying the original
        data = scored_data.copy()
        
        # Define classification thresholds
        HIGH_SUCCESS = 75
        MEDIUM_SUCCESS = 60
        LOW_SUCCESS = 40
        
        # Classify based on success score
        data['success_classification'] = pd.cut(
            data['success_score'],
            bins=[0, LOW_SUCCESS, MEDIUM_SUCCESS, HIGH_SUCCESS, 100],
            labels=['High Failure Risk', 'Moderate Failure Risk', 'Moderate Success Potential', 'High Success Potential']
        )
        
        # Add binary classification for simpler analysis
        data['is_likely_success'] = data['success_score'] >= MEDIUM_SUCCESS
        
        # Count by category
        classification_counts = data['success_classification'].value_counts()
        self.logger.info(f"Classification results: {classification_counts.to_dict()}")
        
        return data
    
    def _generate_meta_insights(self, classified_data):
        """
        Generate meta-insights from the integrated analysis.
        
        Args:
            classified_data (pandas.DataFrame): Data with success classifications
            
        Returns:
            dict: Dictionary of insights
        """
        self.logger.info("Generating meta-insights")
        
        insights = {}
        
        try:
            # Overall success rate
            insights['overall_success_rate'] = classified_data['is_likely_success'].mean() * 100
            
            # Success rate by industry
            industry_success = classified_data.groupby('industry')['is_likely_success'].mean() * 100
            insights['industry_success_rates'] = industry_success.to_dict()
            
            # Top and bottom industries by success rate
            min_companies = 3  # Minimum companies per industry to be considered
            industry_counts = classified_data.groupby('industry').size()
            valid_industries = industry_counts[industry_counts >= min_companies].index
            
            valid_industry_success = industry_success[valid_industries]
            if not valid_industry_success.empty:
                insights['top_industries'] = valid_industry_success.nlargest(5).to_dict()
                insights['bottom_industries'] = valid_industry_success.nsmallest(5).to_dict()
            
            # Success rate by funding stage
            stage_success = classified_data.groupby('funding_stage')['is_likely_success'].mean() * 100
            insights['stage_success_rates'] = stage_success.to_dict()
            
            # Component contribution analysis
            # Calculate correlation of component scores with overall success score
            component_scores = [
                'stage_transition_score', 
                'survival_score', 
                'funding_adequacy_score', 
                'industry_momentum_score', 
                'anomaly_normalized_score'
            ]
            
            correlations = {}
            for component in component_scores:
                if component in classified_data.columns:
                    corr = classified_data[component].corr(classified_data['success_score'])
                    correlations[component] = corr
            
            insights['component_correlations'] = correlations
            
            # Identify top success/failure companies
            top_success = classified_data.nlargest(10, 'success_score')[['company_name', 'success_score', 'success_classification']]
            top_failure = classified_data.nsmallest(10, 'success_score')[['company_name', 'success_score', 'success_classification']]
            
            insights['top_success_companies'] = top_success.to_dict(orient='records')
            insights['top_failure_companies'] = top_failure.to_dict(orient='records')
            
            self.logger.info("Meta-insights generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating meta-insights: {str(e)}")
        
        return insights
    
    def generate_comprehensive_report(self, output_path=None):
        """
        Generate a comprehensive report with insights from all components.
        
        Args:
            output_path (str): Path to save the report
            
        Returns:
            str: Path to the generated report
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'comprehensive_report.txt')
            
        self.logger.info(f"Generating comprehensive report at {output_path}")
        
        try:
            with open(output_path, 'w') as f:
                # Report header
                f.write("==========================================================\n")
                f.write("     COMPREHENSIVE STARTUP FUNDING PREDICTION REPORT      \n")
                f.write("==========================================================\n\n")
                f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Meta-model insights
                if 'meta_model' in self.results and 'insights' in self.results['meta_model']:
                    insights = self.results['meta_model']['insights']
                    
                    f.write("----------------------------------------------------------\n")
                    f.write("META-MODEL INSIGHTS\n")
                    f.write("----------------------------------------------------------\n\n")
                    
                    f.write(f"Overall Success Rate: {insights.get('overall_success_rate', 'N/A'):.2f}%\n\n")
                    
                    f.write("Top Industries by Success Rate:\n")
                    for industry, rate in insights.get('top_industries', {}).items():
                        f.write(f"  - {industry}: {rate:.2f}%\n")
                    f.write("\n")
                    
                    f.write("Bottom Industries by Success Rate:\n")
                    for industry, rate in insights.get('bottom_industries', {}).items():
                        f.write(f"  - {industry}: {rate:.2f}%\n")
                    f.write("\n")
                    
                    f.write("Top Success Potential Companies:\n")
                    for company in insights.get('top_success_companies', [])[:5]:
                        f.write(f"  - {company['company_name']}: Score {company['success_score']:.2f}\n")
                    f.write("\n")
                    
                    f.write("High Risk Companies:\n")
                    for company in insights.get('top_failure_companies', [])[:5]:
                        f.write(f"  - {company['company_name']}: Score {company['success_score']:.2f}\n")
                    f.write("\n")
                
                # Add component-specific insights
                component_sections = [
                    ('Funding Stage Prediction', 'stage_prediction'),
                    ('Funding Continuation Analysis', 'continuation'),
                    ('Funding Amount Forecast', 'amount_forecast'),
                    ('Anomaly Detection', 'anomaly_detection'),
                    ('Industry Trend Analysis', 'industry_trends')
                ]
                
                for title, key in component_sections:
                    f.write("----------------------------------------------------------\n")
                    f.write(f"{title.upper()} INSIGHTS\n")
                    f.write("----------------------------------------------------------\n\n")
                    
                    if key in self.results and self.results[key].get('success', False):
                        # Extract key metrics for each component
                        if key == 'stage_prediction' and 'metrics' in self.results[key]:
                            metrics = self.results[key]['metrics']
                            f.write(f"Model Accuracy: {metrics.get('accuracy', 'N/A'):.2f}%\n")
                            f.write(f"Top Stage Transition: {metrics.get('top_transition', 'N/A')}\n\n")
                        
                        elif key == 'continuation' and 'metrics' in self.results[key]:
                            metrics = self.results[key]['metrics']
                            f.write(f"Median Time to Next Funding: {metrics.get('median_time', 'N/A')} months\n")
                            f.write(f"Companies at Risk: {metrics.get('at_risk_count', 'N/A')}\n\n")
                            
                        elif key == 'amount_forecast' and 'metrics' in self.results[key]:
                            metrics = self.results[key]['metrics']
                            f.write(f"Forecast Accuracy (MAPE): {metrics.get('mape', 'N/A'):.2f}%\n")
                            f.write(f"Median Predicted Increase: {metrics.get('median_increase', 'N/A')}%\n\n")
                            
                        elif key == 'anomaly_detection' and 'metrics' in self.results[key]:
                            metrics = self.results[key]['metrics']
                            f.write(f"Anomalies Detected: {metrics.get('anomaly_count', 'N/A')}\n")
                            f.write(f"Most Common Anomaly Type: {metrics.get('common_type', 'N/A')}\n\n")
                            
                        elif key == 'industry_trends' and 'metrics' in self.results[key]:
                            metrics = self.results[key]['metrics']
                            f.write("Emerging Industries:\n")
                            emerging = self.results[key].get('emerging_industries', [])
                            for industry in emerging[:3]:
                                f.write(f"  - {industry['industry']}: {industry['momentum']*100:.2f}% growth\n")
                            f.write("\n")
                            
                            f.write("Saturated Industries:\n")
                            saturated = self.results[key].get('saturated_industries', [])
                            for industry in saturated[:3]:
                                f.write(f"  - {industry['industry']}: {industry['momentum']*100:.2f}% growth\n")
                            f.write("\n")
                    else:
                        f.write(f"No results available for {title}\n\n")
                
                # Add timestamp
                f.write("\n\nAnalysis completed at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            self.logger.info(f"Comprehensive report generated at {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            return None

    def visualize_meta_model_results(self, output_dir=None):
        """
        Generate visualizations for the meta-model results.
        
        Args:
            output_dir (str): Directory to save visualizations
            
        Returns:
            bool: True if visualizations were created successfully
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'meta_visualizations')
            
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Generating meta-model visualizations in {output_dir}")
        
        try:
            if 'meta_model' not in self.results or 'classification' not in self.results['meta_model']:
                self.logger.error("No meta-model results available for visualization")
                return False
                
            data = self.results['meta_model']['classification']
            
            # 1. Success score distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data['success_score'], bins=20, kde=True)
            plt.title('Distribution of Success Scores')
            plt.xlabel('Success Score')
            plt.ylabel('Count')
            plt.axvline(x=60, color='r', linestyle='--', label='Success Threshold')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'success_score_distribution.png'), dpi=300)
            plt.close()
            
            # 2. Success by industry
            plt.figure(figsize=(12, 8))
            industry_success = data.groupby('industry')['success_score'].mean().sort_values(ascending=False)
            industry_counts = data.groupby('industry').size()
            valid_industries = industry_counts[industry_counts >= 3].index
            industry_success = industry_success[valid_industries]
            
            # Plot top 15 industries
            top_industries = industry_success.head(15)
            sns.barplot(x=top_industries.values, y=top_industries.index)
            plt.title('Average Success Score by Industry')
            plt.xlabel('Average Success Score')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'industry_success_scores.png'), dpi=300)
            plt.close()
            
            # 3. Component contribution
            plt.figure(figsize=(10, 6))
            component_names = [
                'Stage Transition',
                'Survival Probability',
                'Funding Adequacy',
                'Industry Momentum',
                'Anomaly Score'
            ]
            component_weights = [
                self.model_weights['stage_transition_probability'],
                self.model_weights['survival_probability'],
                self.model_weights['funding_adequacy'],
                self.model_weights['industry_momentum'],
                self.model_weights['anomaly_severity']
            ]
            
            sns.barplot(x=component_names, y=component_weights)
            plt.title('Component Contribution to Success Score')
            plt.xlabel('Component')
            plt.ylabel('Weight')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'component_weights.png'), dpi=300)
            plt.close()
            
            # 4. Success classification pie chart
            plt.figure(figsize=(10, 8))
            classification_counts = data['success_classification'].value_counts()
            plt.pie(
                classification_counts, 
                labels=classification_counts.index, 
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette('viridis', len(classification_counts))
            )
            plt.axis('equal')
            plt.title('Distribution of Success Classifications')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'classification_distribution.png'), dpi=300)
            plt.close()
            
            # 5. Component scores correlation with success
            if 'insights' in self.results['meta_model'] and 'component_correlations' in self.results['meta_model']['insights']:
                plt.figure(figsize=(10, 6))
                correlations = self.results['meta_model']['insights']['component_correlations']
                corr_components = list(correlations.keys())
                corr_values = list(correlations.values())
                
                # Convert to more readable labels
                readable_components = [c.replace('_score', '').replace('_', ' ').title() for c in corr_components]
                
                sns.barplot(x=corr_values, y=readable_components)
                plt.title('Correlation of Component Scores with Success Score')
                plt.xlabel('Correlation Coefficient')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'component_correlations.png'), dpi=300)
                plt.close()
            
            self.logger.info("Meta-model visualizations created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating meta-model visualizations: {str(e)}")
            return False
            
    def run_analysis(self):
        """
        Run the full integrated predictive analysis pipeline.
        
        Returns:
            dict: Dictionary with analysis results
        """
        self.logger.info("Starting integrated predictive analysis")
        
        try:
            # Step 1: Initialize components if not already done
            if not all([self.stage_predictor, self.continuation_analyzer, self.amount_forecaster, 
                        self.anomaly_detector, self.trend_analyzer]):
                self.initialize_components()
                
            # Step 2: Run all component analyses
            component_results = self.run_all_components()
            
            # Step 3: Integrate results into meta-model
            meta_results = self.integrate_results()
            
            # Step 4: Generate visualizations
            self.visualize_meta_model_results()
            
            # Step 5: Generate comprehensive report
            report_path = self.generate_comprehensive_report()
            
            self.logger.info("Integrated predictive analysis completed successfully")
            
            return {
                'success': True,
                'component_results': component_results,
                'meta_results': meta_results,
                'report_path': report_path
            }
            
        except Exception as e:
            self.logger.error(f"Error in integrated predictive analysis: {str(e)}")
            return {'success': False, 'error': str(e)}

def main():
    """Main function to run the integrated predictive system."""
    parser = argparse.ArgumentParser(description='Run integrated predictive analytics on startup funding data')
    parser.add_argument('--data-dir', type=str, default='./JSONFolder', help='Directory containing funding data files')
    parser.add_argument('--output-dir', type=str, default='./outputIntegrated', help='Directory to save output files')
    parser.add_argument('--parallel', action='store_true', help='Run components in parallel')
    parser.add_argument('--visualize-only', action='store_true', help='Only generate visualizations from existing results')
    parser.add_argument('--report-only', action='store_true', help='Only generate report from existing results')
    args = parser.parse_args()
    
    # Create integrated system
    system = IntegratedPredictiveSystem(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        parallel=args.parallel
    )
    
    if args.visualize_only:
        # Only generate visualizations from existing results
        system.logger.info("Generating visualizations from existing results")
        system.visualize_meta_model_results()
    elif args.report_only:
        # Only generate report from existing results
        system.logger.info("Generating report from existing results")
        system.generate_comprehensive_report()
    else:
        # Run full analysis
        results = system.run_analysis()
        
        if results['success']:
            print("Integrated predictive analysis completed successfully")
            print(f"Results saved to {args.output_dir}")
            
            if 'meta_results' in results and 'insights' in results['meta_results']:
                insights = results['meta_results']['insights']
                
                print(f"\nOverall Success Rate: {insights.get('overall_success_rate', 0):.2f}%")
                
                print("\nTop Industries by Success Rate:")
                for industry, rate in list(insights.get('top_industries', {}).items())[:3]:
                    print(f"  - {industry}: {rate:.2f}%")
                
                print("\nTop Success Potential Companies:")
                for company in insights.get('top_success_companies', [])[:3]:
                    print(f"  - {company['company_name']}: Score {company['success_score']:.2f}")
                
                print(f"\nDetailed report available at: {results.get('report_path', 'N/A')}")
        else:
            print(f"Analysis failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 