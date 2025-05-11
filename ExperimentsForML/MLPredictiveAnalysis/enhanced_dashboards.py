import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import traceback
from datetime import datetime
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.dates as mdates  # Import matplotlib.dates for proper date formatting

# Import necessary components from funding_stage_prediction.py
from MLPredictiveAnalysis.funding_stage_prediction9 import (
    DataLoader, FeatureEngineering, EnhancedPipeline, 
    Visualizer, AdvancedVisualizer, DashboardGenerator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_dashboards.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedDashboards:
    """
    Enhanced dashboards for funding stage prediction and time series analysis.
    This class implements the dashboards described in the user's request.
    """
    
    def __init__(self, output_dir="./enhanced_dashboards"):
        """
        Initialize the enhanced dashboards generator.
        
        Args:
            output_dir (str): Directory to save dashboards
        """
        self.output_dir = output_dir
        
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Enhanced dashboards output directory set to: {self.output_dir}")
        
        # Create subdirectories
        self.classification_dir = os.path.join(self.output_dir, "classification")
        self.timeseries_dir = os.path.join(self.output_dir, "timeseries")
        
        os.makedirs(self.classification_dir, exist_ok=True)
        os.makedirs(self.timeseries_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineering()
        self.visualizer = AdvancedVisualizer(output_dir=self.output_dir)
        
        # Set matplotlib backend to non-interactive for server environments
        plt.switch_backend('Agg')
    
    def load_data(self):
        """
        Load and prepare data for dashboard visualizations.
        
        Returns:
            pandas.DataFrame: Prepared data
        """
        try:
            # Load data from sources
            logger.info("Loading and merging datasets...")
            merged_data = self.data_loader.merge_datasets()
            
            if merged_data.empty:
                logger.error("No data available after merging datasets")
                return None
            
            # Extract features
            logger.info("Extracting features...")
            processed_data = self.feature_engineer.extract_features(merged_data)
            
            logger.info(f"Data prepared successfully with {len(processed_data)} records")
            return processed_data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def generate_classification_dashboard_prototype(self, results):
        """
        Generate a prototype of the classification dashboard.
        
        Args:
            results (dict): Dictionary containing model results
            
        Returns:
            str: Path to the generated prototype
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 20))
            fig.suptitle('Funding Stage Classification Dashboard', fontsize=24, fontweight='bold')
            
            # Define grid layout
            gs = GridSpec(3, 2, figure=fig)
            
            # 1. Model Performance Overview
            ax_perf = fig.add_subplot(gs[0, 0])
            self._plot_model_performance(ax_perf, results)
            
            # 2. Confusion Matrix
            ax_conf = fig.add_subplot(gs[0, 1])
            self._plot_confusion_matrix(ax_conf, results)
            
            # 3. Feature Importance
            ax_imp = fig.add_subplot(gs[1, 0])
            self._plot_feature_importance(ax_imp, results)
            
            # 4. ROC Curves
            ax_roc = fig.add_subplot(gs[1, 1])
            self._plot_roc_curves(ax_roc, results)
            
            # 5. Funding Stage Distribution
            ax_dist = fig.add_subplot(gs[2, 0])
            self._plot_funding_stage_distribution(ax_dist, results)
            
            # 6. Feature Distribution
            ax_feat = fig.add_subplot(gs[2, 1])
            self._plot_feature_distribution(ax_feat, results)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save the prototype
            output_path = os.path.join(self.classification_dir, f'classification_dashboard_prototype_{timestamp}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Classification dashboard prototype saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error generating classification dashboard prototype: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_timeseries_dashboard_prototype(self, results):
        """
        Generate a prototype of the time series dashboard.
        
        Args:
            results (dict): Dictionary containing time series results
            
        Returns:
            str: Path to the generated prototype
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 20))
            fig.suptitle('Funding Trends Time Series Dashboard', fontsize=24, fontweight='bold')
            
            # Define grid layout
            gs = GridSpec(3, 2, figure=fig)
            
            # 1. Funding Trends Forecast
            ax_forecast = fig.add_subplot(gs[0, :])
            self._plot_funding_forecast(ax_forecast, results)
            
            # 2. Forecast Components
            ax_components = fig.add_subplot(gs[1, 0])
            self._plot_forecast_components(ax_components, results)
            
            # 3. Industry Forecasts
            ax_industry = fig.add_subplot(gs[1, 1])
            self._plot_industry_forecasts(ax_industry, results)
            
            # 4. Funding Stage Transitions
            ax_transitions = fig.add_subplot(gs[2, 0])
            self._plot_stage_transitions(ax_transitions, results)
            
            # 5. Stage Composition
            ax_composition = fig.add_subplot(gs[2, 1])
            self._plot_stage_composition(ax_composition, results)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save the prototype
            output_path = os.path.join(self.timeseries_dir, f'timeseries_dashboard_prototype_{timestamp}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Time series dashboard prototype saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error generating time series dashboard prototype: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    # Helper methods for classification dashboard components
    def _plot_model_performance(self, ax, results):
        """Plot model performance comparison"""
        if not results:
            ax.text(0.5, 0.5, 'No model results available', ha='center', va='center')
            ax.set_title('Model Performance Overview')
            return
        
        # Extract metrics
        model_names = []
        accuracy_scores = []
        f1_scores = []
        
        for model_name, model_results in results.items():
            if isinstance(model_results, tuple) and len(model_results) > 1:
                metrics = model_results[1]
                model_names.append(model_name)
                accuracy_scores.append(metrics.get('accuracy', 0))
                f1_scores.append(metrics.get('f1', 0))
        
        if not model_names:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
            ax.set_title('Model Performance Overview')
            return
        
        # Create bar chart
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, accuracy_scores, width, label='Accuracy')
        ax.bar(x + width/2, f1_scores, width, label='F1 Score')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Overview')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(accuracy_scores):
            ax.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
        for i, v in enumerate(f1_scores):
            ax.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    def _plot_confusion_matrix(self, ax, results):
        """Plot confusion matrix for the best model"""
        if not results:
            ax.text(0.5, 0.5, 'No model results available', ha='center', va='center')
            ax.set_title('Confusion Matrix')
            return
        
        # Find best model based on accuracy
        best_model = None
        best_accuracy = -1
        
        for model_name, model_results in results.items():
            if isinstance(model_results, tuple) and len(model_results) > 1:
                metrics = model_results[1]
                accuracy = metrics.get('accuracy', 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
        
        if best_model and 'y_test' in results[best_model] and 'y_pred' in results[best_model]:
            cm = confusion_matrix(results[best_model]['y_test'], results[best_model]['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {best_model}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        else:
            ax.text(0.5, 0.5, 'No confusion matrix data available', ha='center', va='center')
            ax.set_title('Confusion Matrix')
    
    def _plot_feature_importance(self, ax, results):
        """Plot feature importance for the best model"""
        if not results:
            ax.text(0.5, 0.5, 'No model results available', ha='center', va='center')
            ax.set_title('Feature Importance')
            return
        
        # Find model with feature importance
        for model_name, model_results in results.items():
            if isinstance(model_results, tuple) and len(model_results) > 1:
                model = model_results[0]
                
                # Check if we have feature importance data
                if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
                    importances = model.feature_importances_
                    feature_names = model.feature_names_in_
                    
                    # Get top 15 features
                    indices = np.argsort(importances)[-15:]
                    
                    # Plot horizontal bar chart
                    y_pos = np.arange(len(indices))
                    ax.barh(y_pos, importances[indices], align='center', color='coral')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([feature_names[i] for i in indices])
                    ax.invert_yaxis()  # Display highest importance at the top
                    ax.set_xlabel('Importance')
                    ax.set_title(f'Feature Importance - {model_name}')
                    ax.grid(True, alpha=0.3)
                    return
        
        ax.text(0.5, 0.5, 'No feature importance data available', ha='center', va='center')
        ax.set_title('Feature Importance')
        
    def _plot_roc_curves(self, ax, results):
        """Plot ROC curves for models"""
        if not results:
            ax.text(0.5, 0.5, 'No model results available', ha='center', va='center')
            ax.set_title('ROC Curves')
            return
        
        has_data = False
        
        for model_name, model_results in results.items():
            if isinstance(model_results, tuple) and len(model_results) > 1:
                if 'y_test' in model_results[1] and 'y_proba' in model_results[1]:
                    y_test = model_results[1]['y_test']
                    y_proba = model_results[1]['y_proba']
                    
                    # Get unique classes
                    classes = np.unique(y_test)
                    n_classes = len(classes)
                    
                    # Binary classification
                    if n_classes == 2:
                        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
                        has_data = True
                    # Multiclass classification
                    else:
                        # Binarize the output for multiclass ROC
                        from sklearn.preprocessing import label_binarize
                        y_bin = label_binarize(y_test, classes=classes)
                        
                        # Compute ROC curve for each class
                        for i, class_idx in enumerate(range(n_classes)):
                            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                            roc_auc = auc(fpr, tpr)
                            ax.plot(fpr, tpr, lw=2, label=f'{model_name} (Class {i}, AUC = {roc_auc:.2f})')
                            has_data = True
        
        if has_data:
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend(loc="lower right", fontsize='small')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No ROC curve data available', ha='center', va='center')
            ax.set_title('ROC Curves')
        
    def _plot_funding_stage_distribution(self, ax, results):
        """Plot funding stage distribution"""
        # For this plot, we need the actual data, not just model results
        try:
            data = self.load_data()
            if data is None or 'funding_stage' not in data.columns:
                ax.text(0.5, 0.5, 'No funding stage data available', ha='center', va='center')
                ax.set_title('Funding Stage Distribution')
                return
            
            # Count occurrences of each funding stage
            stage_counts = data['funding_stage'].value_counts()
            
            # Create bar chart
            stage_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_xlabel('Funding Stage')
            ax.set_ylabel('Count')
            ax.set_title('Funding Stage Distribution')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Rotate labels for better readability
            ax.set_xticklabels(stage_counts.index, rotation=45, ha='right')
            
            # Add counts as text
            for i, count in enumerate(stage_counts):
                ax.text(i, count + (stage_counts.max() * 0.02), str(count), 
                       ha='center', va='bottom')
                
        except Exception as e:
            logger.error(f"Error plotting funding stage distribution: {str(e)}")
            ax.text(0.5, 0.5, 'Error generating plot', ha='center', va='center')
            ax.set_title('Funding Stage Distribution')
        
    def _plot_feature_distribution(self, ax, results):
        """Plot feature distributions across funding stages"""
        try:
            data = self.load_data()
            if data is None or 'funding_stage' not in data.columns:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                ax.set_title('Feature Distribution')
                return
            
            # Choose an important numeric feature
            numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
            if 'funding_amount' in numeric_features:
                feature = 'funding_amount'
            elif len(numeric_features) > 0:
                feature = numeric_features[0]
            else:
                ax.text(0.5, 0.5, 'No numeric features available', ha='center', va='center')
                ax.set_title('Feature Distribution')
                return
            
            # Create violin plot
            if len(data[feature].dropna()) > 0:
                sns.violinplot(x='funding_stage', y=feature, data=data, ax=ax)
                ax.set_title(f'Distribution of {feature} by Funding Stage')
                ax.set_xlabel('Funding Stage')
                ax.set_ylabel(feature)
                
                # Improve label readability
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Format y-axis for large numbers
                if feature == 'funding_amount':
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
            else:
                ax.text(0.5, 0.5, f'No valid data for {feature}', ha='center', va='center')
                ax.set_title('Feature Distribution')
                
        except Exception as e:
            logger.error(f"Error plotting feature distribution: {str(e)}")
            ax.text(0.5, 0.5, 'Error generating plot', ha='center', va='center')
            ax.set_title('Feature Distribution')
    
    # Helper methods for time series dashboard components
    def _plot_funding_forecast(self, ax, results):
        """Plot funding forecast with historical and predicted values"""
        if not results or not any('funding' in k for k in results.keys()):
            ax.text(0.5, 0.5, 'No forecast data available', ha='center', va='center')
            ax.set_title('Funding Trends Forecast')
            return
        
        # Try to find funding rounds or funding amounts forecast
        forecast_key = None
        if 'funding_rounds' in results:
            forecast_key = 'funding_rounds'
        elif 'funding_amounts' in results:
            forecast_key = 'funding_amounts'
        elif any(k.startswith('funding_') for k in results.keys()):
            # Get the first key that starts with 'funding_'
            forecast_key = next(k for k in results.keys() if k.startswith('funding_'))
        
        if forecast_key is None:
            ax.text(0.5, 0.5, 'No funding forecast available', ha='center', va='center')
            ax.set_title('Funding Trends Forecast')
            return
        
        forecast = results[forecast_key]
        
        # Check if forecast has the necessary columns
        required_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        if not all(col in forecast.columns for col in required_columns):
            ax.text(0.5, 0.5, 'Invalid forecast data format', ha='center', va='center')
            ax.set_title('Funding Trends Forecast')
            return
        
        # Find actual values if available
        has_actuals = 'y' in forecast.columns
        
        # Determine cutoff between historical and future data
        if has_actuals:
            historical_mask = ~forecast['y'].isna()
            historical_data = forecast[historical_mask]
            future_data = forecast[~historical_mask]
        else:
            # Assume the last 12 periods are the future if no actuals
            historical_data = forecast.iloc[:-12]
            future_data = forecast.iloc[-12:]
        
        # Plot historical data
        if has_actuals and len(historical_data) > 0:
            ax.plot(historical_data['ds'], historical_data['y'], 'ko', markersize=4, label='Historical')
        
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
        
        # Plot confidence intervals
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                        color='blue', alpha=0.2, label='95% Confidence Interval')
        
        # Add vertical line to separate historical and future
        if len(historical_data) > 0 and len(future_data) > 0:
            cutoff_date = historical_data['ds'].max()
            ax.axvline(x=cutoff_date, color='r', linestyle='--', label='Forecast Start')
        
        # Format the plot
        ax.set_xlabel('Date')
        if forecast_key == 'funding_rounds':
            ax.set_ylabel('Number of Funding Rounds')
        elif forecast_key == 'funding_amounts':
            ax.set_ylabel('Funding Amount ($)')
            # Format y-axis for large amounts
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
        else:
            ax.set_ylabel('Value')
        
        ax.set_title('Funding Trends Forecast')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Improve date formatting - use mdates instead of ticker.DateFormatter
        date_format = mdates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(date_format)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
    def _plot_forecast_components(self, ax, results):
        """Plot forecast components (trend, seasonality)"""
        if not results or not any('funding' in k for k in results.keys()):
            ax.text(0.5, 0.5, 'No forecast data available', ha='center', va='center')
            ax.set_title('Forecast Components')
            return
        
        # Try to find funding rounds or funding amounts forecast
        forecast_key = None
        if 'funding_rounds' in results:
            forecast_key = 'funding_rounds'
        elif 'funding_amounts' in results:
            forecast_key = 'funding_amounts'
        elif any(k.startswith('funding_') for k in results.keys()):
            forecast_key = next(k for k in results.keys() if k.startswith('funding_'))
        
        if forecast_key is None:
            ax.text(0.5, 0.5, 'No funding forecast available', ha='center', va='center')
            ax.set_title('Forecast Components')
            return
        
        forecast = results[forecast_key]
        
        # Check if forecast has component columns
        component_columns = ['trend', 'yearly', 'monthly']
        available_components = [col for col in component_columns if col in forecast.columns]
        
        if not available_components:
            ax.text(0.5, 0.5, 'No component data available', ha='center', va='center')
            ax.set_title('Forecast Components')
            return
        
        # Use a nested axis for multiple component plots
        if len(available_components) > 1:
            # Create a figure with subplots for each component
            ax.axis('off')  # Turn off the main axis
            gs = ax.get_gridspec()
            subax = [plt.subplot(gs[1, 0]) for _ in range(len(available_components))]
            
            for i, component in enumerate(available_components):
                # Plot the component
                subax[i].plot(forecast['ds'], forecast[component])
                subax[i].set_title(f'{component.capitalize()} Component')
                
                # Format the plot
                date_format = mdates.DateFormatter('%Y-%m')
                subax[i].xaxis.set_major_formatter(date_format)
                plt.setp(subax[i].xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Add grid
                subax[i].grid(True, alpha=0.3)
                
                # Only show x-label for the bottom plot
                if i == len(available_components) - 1:
                    subax[i].set_xlabel('Date')
        else:
            # Plot the single available component
            component = available_components[0]
            ax.plot(forecast['ds'], forecast[component])
            ax.set_title(f'{component.capitalize()} Component')
            
            # Format the plot
            date_format = mdates.DateFormatter('%Y-%m')
            ax.xaxis.set_major_formatter(date_format)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.3)
        
    def _plot_industry_forecasts(self, ax, results):
        """Plot industry-specific forecasts"""
        # Find industry-specific forecasts
        industry_forecasts = {k: v for k, v in results.items() if k.startswith('industry_')}
        
        if not industry_forecasts:
            ax.text(0.5, 0.5, 'No industry forecasts available', ha='center', va='center')
            ax.set_title('Industry-Specific Forecasts')
            return
        
        # Select top industries by forecast value
        top_industries = []
        for key, forecast in industry_forecasts.items():
            if 'yhat' in forecast.columns and len(forecast) > 0:
                industry_name = key.replace('industry_', '').replace('_', ' ').title()
                max_value = forecast['yhat'].max()
                top_industries.append((industry_name, key, max_value))
        
        # Sort by max value and take top 5
        top_industries.sort(key=lambda x: x[2], reverse=True)
        top_5_industries = top_industries[:5]
        
        if not top_5_industries:
            ax.text(0.5, 0.5, 'No valid industry forecasts', ha='center', va='center')
            ax.set_title('Industry-Specific Forecasts')
            return
        
        # Plot forecasts for top industries
        for industry_name, key, _ in top_5_industries:
            forecast = industry_forecasts[key]
            ax.plot(forecast['ds'], forecast['yhat'], label=industry_name)
        
        # Format the plot
        ax.set_xlabel('Date')
        ax.set_ylabel('Forecasted Value')
        ax.set_title('Top Industry Forecasts')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Improve date formatting
        date_format = mdates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(date_format)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
    def _plot_stage_transitions(self, ax, results):
        """Plot funding stage transitions"""
        # Look for stage-specific forecasts
        stage_forecasts = {k: v for k, v in results.items() if k.startswith('stage_')}
        
        if not stage_forecasts:
            ax.text(0.5, 0.5, 'No stage transition data available', ha='center', va='center')
            ax.set_title('Funding Stage Transitions')
            return
        
        # Plot forecasts for each funding stage
        for stage_key, forecast in stage_forecasts.items():
            if 'yhat' in forecast.columns and len(forecast) > 0:
                stage_name = stage_key.replace('stage_', '').replace('_', ' ').title()
                ax.plot(forecast['ds'], forecast['yhat'], label=stage_name)
        
        # Format the plot
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Rounds')
        ax.set_title('Funding Stage Volume Forecast')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Improve date formatting
        date_format = mdates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(date_format)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
    def _plot_stage_composition(self, ax, results):
        """Plot funding stage composition forecast"""
        # Look for stage-specific forecasts
        stage_forecasts = {k: v for k, v in results.items() if k.startswith('stage_')}
        
        if not stage_forecasts:
            ax.text(0.5, 0.5, 'No stage composition data available', ha='center', va='center')
            ax.set_title('Stage Composition Forecast')
            return
        
        # Create a combined DataFrame with all stage forecasts
        composition_data = pd.DataFrame()
        
        for stage_key, forecast in stage_forecasts.items():
            if 'yhat' in forecast.columns and len(forecast) > 0:
                stage_name = stage_key.replace('stage_', '').replace('_', ' ').title()
                if composition_data.empty:
                    composition_data['ds'] = forecast['ds']
                composition_data[stage_name] = forecast['yhat']
        
        if composition_data.empty:
            ax.text(0.5, 0.5, 'No valid stage composition data', ha='center', va='center')
            ax.set_title('Stage Composition Forecast')
            return
        
        # Create a stacked area chart
        # First, set the index to date for easier plotting
        composition_data.set_index('ds', inplace=True)
        
        # Plot the stacked areas
        composition_data.plot.area(ax=ax, stacked=True, alpha=0.7)
        
        # Format the plot
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Rounds')
        ax.set_title('Funding Stage Composition Forecast')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Improve date formatting
        date_format = mdates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(date_format)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Main function to run the dashboard generation
def main():
    """Run the enhanced dashboard generation"""
    try:
        # Initialize dashboard generator
        dashboard_generator = EnhancedDashboards()
        
        # Load data
        data = dashboard_generator.load_data()
        
        if data is None:
            logger.error("No data available to generate dashboards")
            return False
        
        # Initialize and run the pipeline to get real model results
        logger.info("Running machine learning pipeline to get real model results...")
        pipeline = EnhancedPipeline(output_dir="./output")
        pipeline_results = pipeline.run()
        
        if not pipeline_results:
            logger.error("Pipeline did not return any results")
            return False
        
        # Extract classification and time series results from pipeline results
        classification_results = pipeline_results.get('classification_results', {})
        timeseries_results = pipeline_results.get('timeseries_results', {})
        
        if not classification_results:
            logger.warning("No classification results available")
        
        if not timeseries_results:
            logger.warning("No time series results available")
        
        # Generate dashboards using real results
        logger.info("Generating classification dashboard...")
        class_dashboard_path = dashboard_generator.generate_classification_dashboard_prototype(classification_results)
        if class_dashboard_path:
            logger.info(f"Classification dashboard saved to: {class_dashboard_path}")
        
        logger.info("Generating time series dashboard...")
        ts_dashboard_path = dashboard_generator.generate_timeseries_dashboard_prototype(timeseries_results)
        if ts_dashboard_path:
            logger.info(f"Time series dashboard saved to: {ts_dashboard_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main() 