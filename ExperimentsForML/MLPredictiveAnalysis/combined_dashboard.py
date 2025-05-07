import os
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

class CombinedDashboardGenerator:
    """
    Class to generate consolidated dashboards for classification and time series models.
    Instead of creating multiple separate dashboards, this creates one comprehensive dashboard each
    for classification and time series models.
    """
    
    def __init__(self, output_dir="./dashboards"):
        """
        Initialize the dashboard generator.
        
        Args:
            output_dir (str): Directory to save dashboards
        """
        self.output_dir = output_dir
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Dashboard output directory set to: {self.output_dir}")
        
        # Create subdirectories
        self.classification_dir = os.path.join(self.output_dir, "combined_dashboards", "classification")
        self.timeseries_dir = os.path.join(self.output_dir, "combined_dashboards", "timeseries")
        
        os.makedirs(self.classification_dir, exist_ok=True)
        os.makedirs(self.timeseries_dir, exist_ok=True)
        
        # Set matplotlib backend to non-interactive
        try:
            plt.switch_backend('Agg')
            self.logger.info("Set matplotlib backend to Agg")
        except Exception as e:
            self.logger.warning(f"Failed to set matplotlib backend: {str(e)}")
    
    def generate_all_dashboards(self, classification_results, timeseries_results):
        """
        Generate combined dashboards for both classification and time series models.
        
        Args:
            classification_results (dict): Results from classification models
            timeseries_results (dict): Results from time series forecasts
            
        Returns:
            dict: Paths to generated dashboards
        """
        dashboard_paths = {}
        
        # Generate classification dashboard
        self.logger.info("Generating combined classification model dashboard...")
        classification_path = self.generate_classification_dashboard(classification_results)
        dashboard_paths["classification"] = classification_path
        
        # Generate time series dashboard
        self.logger.info("Generating combined time series model dashboard...")
        timeseries_path = self.generate_timeseries_dashboard(timeseries_results)
        dashboard_paths["timeseries"] = timeseries_path
        
        return dashboard_paths
    
    def generate_classification_dashboard(self, results):
        """
        Generate a comprehensive dashboard for classification models.
        
        Args:
            results (dict): Dictionary containing model results
            
        Returns:
            str: Path to generated dashboard
        """
        if not results:
            self.logger.warning("No classification results available to generate dashboard")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Funding Stage Classification Model Analysis', fontsize=22, fontweight='bold')
        
        # Define a grid layout for our dashboard
        gs = GridSpec(3, 2, figure=fig)
        
        # 1. Model Comparison (Top Left)
        ax_comparison = fig.add_subplot(gs[0, 0])
        self._add_model_comparison_plot(ax_comparison, results)
        
        # 2. Confusion Matrix for Best Model (Top Right)
        ax_confusion = fig.add_subplot(gs[0, 1])
        self._add_confusion_matrix_plot(ax_confusion, results)
        
        # 3. Feature Importance (Middle Left)
        ax_importance = fig.add_subplot(gs[1, 0])
        self._add_feature_importance_plot(ax_importance, results)
        
        # 4. Metrics Table (Middle Right)
        ax_metrics = fig.add_subplot(gs[1, 1])
        self._add_metrics_table(ax_metrics, results)
        
        # 5. ROC Curve for Best Model (Bottom Row - spans both columns)
        ax_roc = fig.add_subplot(gs[2, :])
        self._add_roc_curve_plot(ax_roc, results)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save the combined dashboard
        output_path = os.path.join(self.classification_dir, f'classification_dashboard_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Combined classification dashboard saved to: {output_path}")
        return output_path
    
    def _add_model_comparison_plot(self, ax, results):
        """Add model comparison plot to the given axis."""
        # Extract model names and metrics
        model_names = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for model_name, model_results in results.items():
            if isinstance(model_results, tuple) and len(model_results) > 1:
                metrics = model_results[1]
            else:
                metrics = model_results
                
            model_names.append(model_name)
            
            # Extract metrics
            accuracy_scores.append(metrics.get('accuracy', 0))
            
            # Get precision, recall, f1 from different metric formats
            if 'precision' in metrics:
                if isinstance(metrics['precision'], dict):
                    precision_scores.append(metrics['precision'].get('macro', 0))
                else:
                    precision_scores.append(metrics['precision'])
            else:
                precision_scores.append(0)
                
            if 'recall' in metrics:
                if isinstance(metrics['recall'], dict):
                    recall_scores.append(metrics['recall'].get('macro', 0))
                else:
                    recall_scores.append(metrics['recall'])
            else:
                recall_scores.append(0)
                
            if 'f1_scores' in metrics:
                if isinstance(metrics['f1_scores'], dict):
                    f1_scores.append(metrics['f1_scores'].get('macro', 0))
                else:
                    f1_scores.append(metrics['f1_scores'])
            else:
                f1_scores.append(0)
        
        # Set width of bars
        bar_width = 0.2
        index = np.arange(len(model_names))
        
        # Create bars
        ax.bar(index, accuracy_scores, bar_width, label='Accuracy', color='#3274A1')
        ax.bar(index + bar_width, precision_scores, bar_width, label='Precision', color='#E1812C')
        ax.bar(index + 2*bar_width, recall_scores, bar_width, label='Recall', color='#3A923A')
        ax.bar(index + 3*bar_width, f1_scores, bar_width, label='F1 Score', color='#C03D3E')
        
        # Add labels and legend
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14)
        ax.set_xticks(index + 1.5*bar_width)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.0)
        
        # Add value labels on top of the bars
        for i, v in enumerate(accuracy_scores):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    
    def _add_confusion_matrix_plot(self, ax, results):
        """Add confusion matrix plot for the best model to the given axis."""
        # Find the best model based on accuracy
        best_model_name = None
        best_accuracy = -1
        best_conf_matrix = None
        
        for model_name, model_results in results.items():
            if isinstance(model_results, tuple) and len(model_results) > 1:
                metrics = model_results[1]
            else:
                metrics = model_results
                
            accuracy = metrics.get('accuracy', 0)
            if accuracy > best_accuracy and 'confusion_matrix' in metrics:
                best_accuracy = accuracy
                best_model_name = model_name
                best_conf_matrix = metrics['confusion_matrix']
        
        if best_conf_matrix is None:
            ax.text(0.5, 0.5, 'No confusion matrix available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Confusion Matrix', fontsize=14)
            return
            
        # Plot confusion matrix
        im = ax.imshow(best_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        num_classes = best_conf_matrix.shape[0]
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
        ax.set_yticklabels([f'Class {i}' for i in range(num_classes)])
        
        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = best_conf_matrix.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f"{best_conf_matrix[i, j]}",
                       ha="center", va="center", 
                       color="white" if best_conf_matrix[i, j] > thresh else "black",
                       fontsize=8)
                
        ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
    
    def _add_feature_importance_plot(self, ax, results):
        """Add feature importance plot to the given axis."""
        # Find a model with feature importances
        for model_name, model_results in results.items():
            if isinstance(model_results, tuple):
                model = model_results[0]
                metrics = model_results[1]
            else:
                model = None
                metrics = model_results
                
            if 'feature_importance' in metrics:
                importance = metrics['feature_importance']
                feature_names = metrics.get('feature_names', [f'Feature {i}' for i in range(len(importance))])
                
                # Sort features by importance
                if len(importance) > 0:
                    indices = np.argsort(importance)[::-1]
                    
                    # Plot top 10 features
                    num_features = min(10, len(importance))
                    top_features = [feature_names[i] for i in indices[:num_features]]
                    top_importance = [importance[i] for i in indices[:num_features]]
                    
                    ax.barh(range(num_features), top_importance, align='center', color='#3274A1')
                    ax.set_yticks(range(num_features))
                    ax.set_yticklabels(top_features)
                    ax.set_xlabel('Importance', fontsize=12)
                    ax.set_title(f'Feature Importance - {model_name}', fontsize=14)
                    ax.invert_yaxis()  # Most important at the top
                    ax.grid(True, linestyle='--', alpha=0.7)
                    return
        
        # If no feature importance found
        ax.text(0.5, 0.5, 'No feature importance available', 
               ha='center', va='center', fontsize=12)
        ax.set_title('Feature Importance', fontsize=14)
    
    def _add_metrics_table(self, ax, results):
        """Add model metrics table to the given axis."""
        # Prepare metrics table
        model_names = []
        metrics_dict = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'RMSE': [], 'MAE': []}
        
        for model_name, model_results in results.items():
            if isinstance(model_results, tuple) and len(model_results) > 1:
                metrics = model_results[1]
            else:
                metrics = model_results
                
            model_names.append(model_name)
            
            # Collect metrics
            metrics_dict['Accuracy'].append(f"{metrics.get('accuracy', 0):.4f}")
            
            # Handle precision
            if 'precision' in metrics:
                if isinstance(metrics['precision'], dict):
                    metrics_dict['Precision'].append(f"{metrics['precision'].get('macro', 0):.4f}")
                else:
                    metrics_dict['Precision'].append(f"{metrics['precision']:.4f}")
            else:
                metrics_dict['Precision'].append("N/A")
                
            # Handle recall
            if 'recall' in metrics:
                if isinstance(metrics['recall'], dict):
                    metrics_dict['Recall'].append(f"{metrics['recall'].get('macro', 0):.4f}")
                else:
                    metrics_dict['Recall'].append(f"{metrics['recall']:.4f}")
            else:
                metrics_dict['Recall'].append("N/A")
                
            # Handle F1
            if 'f1_scores' in metrics:
                if isinstance(metrics['f1_scores'], dict):
                    metrics_dict['F1 Score'].append(f"{metrics['f1_scores'].get('macro', 0):.4f}")
                else:
                    metrics_dict['F1 Score'].append(f"{metrics['f1_scores']:.4f}")
            else:
                metrics_dict['F1 Score'].append("N/A")
                
            metrics_dict['RMSE'].append(f"{metrics.get('rmse', 0):.4f}")
            metrics_dict['MAE'].append(f"{metrics.get('mae', 0):.4f}")
        
        # Create a more visually appealing table
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [[metric] + values for metric, values in metrics_dict.items()]
        table = ax.table(cellText=table_data, 
                        rowLabels=None,
                        colLabels=['Metric'] + model_names,
                        cellLoc='center',
                        loc='center',
                        bbox=[0.1, 0.1, 0.8, 0.8])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Add a title
        ax.set_title('Model Metrics Comparison', fontsize=14)
    
    def _add_roc_curve_plot(self, ax, results):
        """Add ROC curve plot to the given axis."""
        # Check if we have ROC curves in results
        has_roc = False
        
        # Color map
        colors = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E', '#9372B2', '#845B53', '#D5BB67', '#8BA8B0']
        color_idx = 0
        
        # For displaying in legend
        best_auc = 0
        best_model = None
        
        for model_name, model_results in results.items():
            if isinstance(model_results, tuple) and len(model_results) > 1:
                metrics = model_results[1]
            else:
                metrics = model_results
                
            if 'roc_auc_scores' in metrics:
                roc_auc_scores = metrics['roc_auc_scores']
                if isinstance(roc_auc_scores, dict) and 'macro_avg' in roc_auc_scores:
                    has_roc = True
                    auc = roc_auc_scores['macro_avg']
                    
                    if auc > best_auc:
                        best_auc = auc
                        best_model = model_name
                        
                    color = colors[color_idx % len(colors)]
                    if 'y_test' in metrics and 'y_proba' in metrics:
                        # Use actual ROC curve if we have predictions and probabilities
                        y_test = metrics['y_test']
                        y_proba = metrics['y_proba']
                        
                        # Convert to one-hot encoding
                        from sklearn.preprocessing import label_binarize
                        from sklearn.metrics import roc_curve, auc
                        
                        # Get unique classes
                        classes = np.unique(y_test)
                        n_classes = len(classes)
                        
                        # Compute ROC curve and ROC area for each class
                        try:
                            y_bin = label_binarize(y_test, classes=classes)
                            
                            # Plot ROC curve for each class
                            fpr = dict()
                            tpr = dict()
                            
                            # Plot ROC curve for a random class
                            for i, cls in enumerate(classes):
                                if i == 0:  # Plot just one class to avoid crowding
                                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
                                    ax.plot(fpr[i], tpr[i], color=color, lw=2, 
                                           label=f'{model_name} (Class {cls}, AUC = {roc_auc_scores.get(f"class_{cls}", 0):.3f})')
                        except Exception as e:
                            # Fall back to just plotting a reference line
                            ax.plot([0, 1], [0, 1], color=color, lw=2, 
                                   label=f'{model_name} (AUC = {auc:.3f})')
                    else:
                        # Just plot the AUC as a straight line if we don't have raw data
                        ax.plot([0, 1], [0, 1], color=color, lw=2, 
                               label=f'{model_name} (AUC = {auc:.3f})')
                    
                    color_idx += 1
        
        if not has_roc:
            ax.text(0.5, 0.5, 'No ROC curve data available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('ROC Curve', fontsize=14)
            return
        
        # Plot the diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        
        # Set labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def generate_timeseries_dashboard(self, results):
        """
        Generate a comprehensive dashboard for time series forecasts.
        
        Args:
            results (dict): Dictionary containing time series results
            
        Returns:
            str: Path to generated dashboard
        """
        if not results:
            self.logger.warning("No time series results available to generate dashboard")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Funding Stage Time Series Analysis', fontsize=22, fontweight='bold')
        
        # Define a grid layout for our dashboard
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Forecast Trends (Top Left)
        ax_forecast = fig.add_subplot(gs[0, 0])
        self._add_forecast_trend_plot(ax_forecast, results)
        
        # 2. Industry Breakdown (Top Right)
        ax_industry = fig.add_subplot(gs[0, 1])
        self._add_industry_breakdown_plot(ax_industry, results)
        
        # 3. Stage Evolution (Bottom Left)
        ax_evolution = fig.add_subplot(gs[1, 0])
        self._add_stage_evolution_plot(ax_evolution, results)
        
        # 4. Seasonality Analysis (Bottom Right)
        ax_seasonality = fig.add_subplot(gs[1, 1])
        self._add_seasonality_plot(ax_seasonality, results)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save the combined dashboard
        output_path = os.path.join(self.timeseries_dir, f'timeseries_dashboard_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Combined time series dashboard saved to: {output_path}")
        return output_path
    
    def _add_forecast_trend_plot(self, ax, results):
        """Add forecast trend plot to the given axis."""
        # Import matplotlib.dates for date formatting
        import matplotlib.dates as mdates
        
        # Look for forecast data
        if 'funding_rounds' in results and isinstance(results['funding_rounds'], pd.DataFrame):
            forecast = results['funding_rounds']
            title = 'Funding Rounds Forecast'
            y_label = 'Number of Rounds'
        elif 'funding_amount' in results and isinstance(results['funding_amount'], pd.DataFrame):
            forecast = results['funding_amount']
            title = 'Funding Amount Forecast'
            y_label = 'Amount ($)'
        else:
            # Find any forecast data
            for key, value in results.items():
                if isinstance(value, pd.DataFrame) and 'ds' in value.columns and 'yhat' in value.columns:
                    forecast = value
                    title = f'{key.replace("_", " ").title()} Forecast'
                    y_label = 'Value'
                    break
            else:
                ax.text(0.5, 0.5, 'No forecast data available', 
                       ha='center', va='center', fontsize=12)
                ax.set_title('Forecast Trend', fontsize=14)
                return
        
        # Plot actual vs predicted
        ax.plot(forecast['ds'], forecast['y'], 'ko', markersize=4, label='Actual')
        ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Predicted')
        
        # Add uncertainty intervals if available
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                          color='blue', alpha=0.2, label='Prediction Interval')
        
        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        # Set labels and title
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add legend
        ax.legend(loc='best')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _add_industry_breakdown_plot(self, ax, results):
        """Add industry breakdown plot to the given axis."""
        # Look for industry-specific forecasts
        industry_keys = [k for k in results.keys() if k.startswith('industry_') and isinstance(results[k], pd.DataFrame)]
        
        if not industry_keys:
            ax.text(0.5, 0.5, 'No industry breakdown data available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Industry Breakdown', fontsize=14)
            return
            
        # Collect the most recent forecast point for each industry
        industries = []
        values = []
        
        for industry_key in industry_keys:
            industry_name = industry_key.replace('industry_', '').replace('_', ' ').title()
            forecast = results[industry_key]
            
            if len(forecast) > 0:
                # Get the most recent actual value or the first forecast value
                if 'y' in forecast.columns and not forecast['y'].isna().all():
                    # Find the most recent non-NaN actual value
                    valid_rows = forecast[~forecast['y'].isna()]
                    if not valid_rows.empty:
                        most_recent = valid_rows.iloc[-1]
                        industries.append(industry_name)
                        values.append(most_recent['y'])
                elif 'yhat' in forecast.columns:
                    # Use the first forecast value
                    industries.append(industry_name)
                    values.append(forecast.iloc[0]['yhat'])
        
        if not industries:
            ax.text(0.5, 0.5, 'No valid industry data available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Industry Breakdown', fontsize=14)
            return
        
        # Sort by value for better visualization
        sorted_indices = np.argsort(values)[::-1]  # Descending order
        industries = [industries[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        bars = ax.barh(industries, values, color='#3274A1')
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.1, i, f'{v:.1f}', va='center')
        
        # Set labels and title
        ax.set_xlabel('Value', fontsize=12)
        ax.set_title('Industry Breakdown', fontsize=14)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _add_stage_evolution_plot(self, ax, results):
        """Add funding stage evolution plot to the given axis."""
        import matplotlib.dates as mdates
        
        # Look for stage-specific forecasts
        stage_keys = [k for k in results.keys() if k.startswith('stage_') and isinstance(results[k], pd.DataFrame)]
        
        if not stage_keys:
            ax.text(0.5, 0.5, 'No funding stage evolution data available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Funding Stage Evolution', fontsize=14)
            return
        
        # Get common dates across all forecasts
        common_dates = None
        
        for stage_name in stage_keys:
            forecast = results[stage_name]
            dates = set(forecast['ds'])
            if common_dates is None:
                common_dates = dates
            else:
                common_dates &= dates
        
        if not common_dates:
            ax.text(0.5, 0.5, 'No common dates across stage forecasts', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Funding Stage Evolution', fontsize=14)
            return
            
        common_dates = sorted(list(common_dates))
        
        # Prepare data for stacked area chart
        data = {}
        for date in common_dates:
            data[date] = {}
        
        for stage_name in stage_keys:
            forecast = results[stage_name]
            for _, row in forecast[forecast['ds'].isin(common_dates)].iterrows():
                data[row['ds']][stage_name] = row['yhat']
        
        # Convert to DataFrame
        evolution_df = pd.DataFrame.from_dict(data, orient='index')
        evolution_df.index.name = 'date'
        evolution_df = evolution_df.fillna(0)
        
        # Fix for mixed positive/negative values - ensure all values are positive
        # Check if any column has mixed positive and negative values
        for col in evolution_df.columns:
            if (evolution_df[col] > 0).any() and (evolution_df[col] < 0).any():
                self.logger.warning(f"Column {col} has mixed positive and negative values. Converting to absolute values.")
                evolution_df[col] = evolution_df[col].abs()
        
        # Plot stacked area chart
        try:
            evolution_df.plot.area(ax=ax, stacked=True, alpha=0.7, linewidth=0)
        except ValueError as e:
            self.logger.error(f"Error plotting stacked area chart: {str(e)}")
            # Fallback to line plot if area plot fails
            self.logger.info("Falling back to line plot visualization")
            evolution_df.plot(ax=ax, alpha=0.7, linewidth=2)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        # Set labels and title
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of Startups', fontsize=12)
        ax.set_title('Funding Stage Distribution Evolution', fontsize=14)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _add_seasonality_plot(self, ax, results):
        """Add seasonality analysis plot to the given axis."""
        # Look for forecast with seasonality components
        forecast_with_components = None
        
        # Try to find a forecast with yearly or monthly seasonality
        for key, forecast in results.items():
            if isinstance(forecast, pd.DataFrame):
                if 'yearly' in forecast.columns or 'monthly' in forecast.columns:
                    forecast_with_components = forecast
                    component_name = 'yearly' if 'yearly' in forecast.columns else 'monthly'
                    break
        
        if forecast_with_components is None:
            ax.text(0.5, 0.5, 'No seasonality data available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Seasonality Analysis', fontsize=14)
            return
            
        # Define x and y based on the component
        if component_name == 'yearly':
            # For yearly seasonality, use day of year
            yearly_pattern = forecast_with_components.copy()
            yearly_pattern['day_of_year'] = yearly_pattern['ds'].dt.dayofyear
            
            # Sort by day of year for a continuous plot
            yearly_pattern = yearly_pattern.sort_values('day_of_year')
            
            x = yearly_pattern['day_of_year']
            y = yearly_pattern['yearly']
            
            # Map day of year to month names for better labeling
            month_positions = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]  # Middle of each month
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            ax.set_xticks(month_positions)
            ax.set_xticklabels(month_labels)
            
            title_component = "Yearly"
            x_label = "Month"
            
        else:  # monthly
            # For monthly seasonality, use day of month
            monthly_pattern = forecast_with_components.copy()
            monthly_pattern['day_of_month'] = monthly_pattern['ds'].dt.day
            
            # Sort by day of month for a continuous plot
            monthly_pattern = monthly_pattern.sort_values('day_of_month')
            
            x = monthly_pattern['day_of_month']
            y = monthly_pattern['monthly']
            
            title_component = "Monthly"
            x_label = "Day of Month"
        
        # Plot the seasonality pattern
        ax.plot(x, y, 'b-', linewidth=2)
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Effect on Outcome', fontsize=12)
        ax.set_title(f'{title_component} Seasonality Pattern', fontsize=14)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7) 