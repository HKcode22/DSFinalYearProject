import os
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from datetime import datetime

class WebsiteDashboardGenerator:
    """
    Class to generate focused dashboards for website display.
    Creates single-purpose dashboards optimized for web interfaces.
    """
    
    def __init__(self, output_dir="./website_dashboards"):
        """
        Initialize the website dashboard generator.
        
        Args:
            output_dir (str): Base directory to save website-ready dashboards
        """
        self.output_dir = output_dir
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Website dashboard output directory set to: {self.output_dir}")
        
        # Create specific subdirectories for classification and time series dashboards
        self.classification_dir = os.path.join(self.output_dir, "classification")
        self.timeseries_dir = os.path.join(self.output_dir, "timeseries")
        
        os.makedirs(self.classification_dir, exist_ok=True)
        os.makedirs(self.timeseries_dir, exist_ok=True)
        
        # Configure matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for server environments
            self.logger.info("Set matplotlib backend to Agg for website dashboards")
        except Exception as e:
            self.logger.warning(f"Failed to set matplotlib backend: {str(e)}")
    
    def generate_model_performance_dashboard(self, model_results, model_name="best_model"):
        """
        Generate a comprehensive single-model performance dashboard for website display.
        Shows accuracy metrics, feature importance, prediction distribution, and confidence levels.
        
        Args:
            model_results (dict): Dictionary containing model results
            model_name (str): Name of the model to display
        
        Returns:
            str: Path to the generated dashboard
        """
        self.logger.info(f"Generating website performance dashboard for {model_name}...")
        
        if not model_results or model_name not in model_results:
            self.logger.error(f"Model results for {model_name} not available")
            return None
        
        # Extract model data
        results = model_results[model_name]
        
        if not isinstance(results, tuple) or len(results) < 3:
            self.logger.error(f"Invalid results format for {model_name}")
            return None
        
        model = results[0]
        metrics = results[1]
        predictions = results[2]
        
        # Check if we have feature importance
        feature_importance = None
        if len(results) > 3 and results[3] is not None:
            feature_importance = results[3]
        
        # Create figure with 4 subplots
        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Accuracy and Confidence Metrics Panel (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Extract metrics
        accuracy = metrics.get('accuracy', 0)
        
        # Unpack precision, recall, f1 from different formats
        precision = 0
        if 'precision' in metrics:
            if isinstance(metrics['precision'], dict):
                precision = metrics['precision'].get('weighted', 0)
            else:
                precision = metrics['precision']
                
        recall = 0
        if 'recall' in metrics:
            if isinstance(metrics['recall'], dict):
                recall = metrics['recall'].get('weighted', 0)
            else:
                recall = metrics['recall']
                
        f1 = 0
        if 'f1_scores' in metrics:
            if isinstance(metrics['f1_scores'], dict):
                f1 = metrics['f1_scores'].get('weighted', 0)
            else:
                f1 = metrics['f1_scores']
        
        # Create a bar chart with metrics
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [accuracy, precision, recall, f1]
        
        bars = ax1.bar(metric_names, metric_values, color=['#3274A1', '#E1812C', '#3A923A', '#C03D3E'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)
        
        ax1.set_ylim(0, 1.05)
        ax1.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 2. Feature Importance Visualization (Top Right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        if feature_importance is not None and hasattr(model, 'feature_names_in_'):
            # Extract feature names and importance values
            feature_names = model.feature_names_in_
            importances = feature_importance
            
            # Sort features by importance
            sorted_idx = importances.argsort()[-10:]  # Top 10 features
            
            # Plot horizontal bar chart
            y_pos = np.arange(len(sorted_idx))
            ax2.barh(y_pos, importances[sorted_idx], align='center', color='#4C72B0')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([feature_names[i] for i in sorted_idx])
            ax2.invert_yaxis()  # Display most important at the top
            ax2.set_title('Top Feature Importance', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Importance', fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7, axis='x')
        else:
            ax2.text(0.5, 0.5, 'Feature importance data not available for this model',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=12)
        
        # 3. Prediction Distribution Visualization (Bottom Left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        if 'y_pred' in predictions and 'y_test' in predictions:
            y_pred = predictions['y_pred']
            
            # Count predictions by class
            pred_counts = pd.Series(y_pred).value_counts().sort_index()
            
            # Create bar chart
            ax3.bar(pred_counts.index.astype(str), pred_counts.values, color='#55A868')
            
            ax3.set_title('Prediction Distribution by Funding Stage', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Funding Stage', fontsize=12)
            ax3.set_ylabel('Count', fontsize=12)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
        else:
            ax3.text(0.5, 0.5, 'Prediction distribution data not available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=12)
        
        # 4. Confidence Level Indication (Bottom Right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        if 'y_proba' in predictions:
            y_proba = predictions['y_proba']
            
            # Get the max probability for each prediction (confidence)
            confidences = np.max(y_proba, axis=1)
            
            # Create histogram of confidence scores
            bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
            ax4.hist(confidences, bins=bins, color='#C44E52', alpha=0.8)
            
            ax4.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Confidence Level', fontsize=12)
            ax4.set_ylabel('Count', fontsize=12)
            ax4.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add vertical lines for confidence thresholds
            ax4.axvline(x=0.9, color='green', linestyle='--', label='High Confidence (>0.9)')
            ax4.axvline(x=0.7, color='orange', linestyle='--', label='Medium Confidence (>0.7)')
            ax4.axvline(x=0.5, color='red', linestyle='--', label='Low Confidence (>0.5)')
            ax4.legend(fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'Confidence level data not available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
        
        # Add overall title
        fig.suptitle(f'Model Performance Dashboard: {model_name}', fontsize=18, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
        
        # Save dashboard
        output_path = os.path.join(self.classification_dir, f'model_performance_{model_name}_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Website classification dashboard saved to: {output_path}")
        return output_path
    
    def generate_funding_growth_dashboard(self, timeseries_results):
        """
        Generate a comprehensive funding growth trajectory dashboard for website display.
        Shows historical vs predicted funding, risk-adjusted forecasts, industry-specific growth,
        and trend inflection points.
        
        Args:
            timeseries_results (dict): Dictionary containing time series forecast results
        
        Returns:
            str: Path to the generated dashboard
        """
        self.logger.info(f"Generating website funding growth dashboard...")
        
        if not timeseries_results or not isinstance(timeseries_results, dict) or len(timeseries_results) == 0:
            self.logger.warning("No time series results available for funding growth dashboard")
            return None
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create figure with 4 panels
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # 1. Historical vs Predicted Funding (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        
        if 'funding_rounds' in timeseries_results and isinstance(timeseries_results['funding_rounds'], pd.DataFrame):
            forecast_df = timeseries_results['funding_rounds']
            
            # Identify historical and forecast periods
            historical_end = forecast_df['ds'].max() - pd.Timedelta(days=365)
            historical_data = forecast_df[forecast_df['ds'] <= historical_end]
            forecast_data = forecast_df[forecast_df['ds'] > historical_end]
            
            # Plot historical data
            if not historical_data.empty:
                ax1.plot(historical_data['ds'], historical_data['yhat'], 'b-', label='Historical')
            
            # Plot forecast data
            ax1.plot(forecast_data['ds'], forecast_data['yhat'], 'r-', label='Predicted')
            
            # Add confidence interval for the forecast
            ax1.fill_between(
                forecast_data['ds'], 
                forecast_data['yhat_lower'], 
                forecast_data['yhat_upper'], 
                color='red', 
                alpha=0.2, 
                label='Confidence Interval'
            )
            
            ax1.set_title('Historical vs Predicted Funding Rounds', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Number of Rounds', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Format x-axis dates
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        else:
            ax1.text(0.5, 0.5, 'Funding rounds forecast data not available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=12)
        
        # 2. Risk-Adjusted Forecasts (Top Right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        if 'funding_amounts' in timeseries_results and isinstance(timeseries_results['funding_amounts'], pd.DataFrame):
            forecast_df = timeseries_results['funding_amounts']
            
            # Filter to forecast period only
            forecast_start = forecast_df['ds'].max() - pd.Timedelta(days=365)
            forecast_data = forecast_df[forecast_df['ds'] > forecast_start]
            
            if not forecast_data.empty:
                # Plot expected forecast (yhat)
                ax2.plot(forecast_data['ds'], forecast_data['yhat'], 'g-', label='Expected Forecast')
                
                # Plot optimistic forecast (upper bound)
                ax2.plot(forecast_data['ds'], forecast_data['yhat_upper'], 'b--', label='Optimistic (Best Case)')
                
                # Plot pessimistic forecast (lower bound)
                ax2.plot(forecast_data['ds'], forecast_data['yhat_lower'], 'r--', label='Pessimistic (Worst Case)')
                
                ax2.set_title('Risk-Adjusted Funding Forecasts', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Date', fontsize=12)
                ax2.set_ylabel('Funding Amount (USD)', fontsize=12)
                
                # Format y-axis labels for currency
                ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
                
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Format x-axis dates
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            else:
                ax2.text(0.5, 0.5, 'Forecast data not available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=12)
        else:
            ax2.text(0.5, 0.5, 'Funding amounts forecast data not available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=12)
        
        # 3. Industry-Specific Growth Rates (Bottom Left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Check for industry forecasts
        industry_keys = [k for k in timeseries_results.keys() if k.startswith('industry_') and isinstance(timeseries_results[k], pd.DataFrame)]
        
        if len(industry_keys) >= 3:
            # Select top industries
            top_industries = industry_keys[:5]  # Limit to top 5
            
            # Calculate growth rates for each industry
            industry_growth = []
            
            for key in top_industries:
                forecast_df = timeseries_results[key]
                
                # Get first and last points of forecast period
                forecast_start = forecast_df['ds'].max() - pd.Timedelta(days=365)
                forecast_period = forecast_df[forecast_df['ds'] > forecast_start]
                
                if len(forecast_period) > 1:
                    start_value = forecast_period.iloc[0]['yhat']
                    end_value = forecast_period.iloc[-1]['yhat']
                    
                    # Calculate annualized growth rate
                    if start_value > 0:
                        growth_rate = (end_value / start_value - 1) * 100
                    else:
                        growth_rate = 0
                        
                    industry_name = key.replace('industry_', '').replace('_', ' ').title()
                    industry_growth.append((industry_name, growth_rate))
            
            # Sort by growth rate
            industry_growth.sort(key=lambda x: x[1], reverse=True)
            
            # Create bar chart
            industries = [x[0] for x in industry_growth]
            growth_rates = [x[1] for x in industry_growth]
            
            # Use different colors based on positive/negative growth
            colors = ['green' if rate >= 0 else 'red' for rate in growth_rates]
            
            ax3.bar(industries, growth_rates, color=colors)
            
            ax3.set_title('Industry-Specific Growth Rates (Annual)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Industry', fontsize=12)
            ax3.set_ylabel('Growth Rate (%)', fontsize=12)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add value labels
            for i, v in enumerate(growth_rates):
                ax3.text(i, v + (0.1 if v >= 0 else -2.0), 
                        f"{v:.1f}%", 
                        ha='center', 
                        fontsize=10,
                        fontweight='bold')
            
        else:
            ax3.text(0.5, 0.5, 'Industry forecast data not available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=12)
        
        # 4. Trend Inflection Points (Bottom Right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        if 'funding_rounds' in timeseries_results and isinstance(timeseries_results['funding_rounds'], pd.DataFrame):
            forecast_df = timeseries_results['funding_rounds']
            
            # Identify forecast period
            forecast_start = forecast_df['ds'].max() - pd.Timedelta(days=365)
            forecast_data = forecast_df[forecast_df['ds'] > forecast_start]
            
            if not forecast_data.empty:
                # Plot the forecast line
                ax4.plot(forecast_data['ds'], forecast_data['yhat'], 'b-')
                
                # Calculate month-over-month changes
                forecast_data['mom_change'] = forecast_data['yhat'].pct_change() * 100
                
                # Find inflection points where trend changes direction (changes sign)
                inflection_points = []
                for i in range(1, len(forecast_data) - 1):
                    current = forecast_data.iloc[i]
                    prev = forecast_data.iloc[i-1]
                    next_point = forecast_data.iloc[i+1]
                    
                    # Check if trend direction changes
                    if (current['mom_change'] > 0 and next_point['mom_change'] < 0) or \
                       (current['mom_change'] < 0 and next_point['mom_change'] > 0):
                        inflection_points.append(current)
                
                # Highlight inflection points
                if inflection_points:
                    inflection_df = pd.DataFrame(inflection_points)
                    ax4.scatter(inflection_df['ds'], inflection_df['yhat'], 
                              color='red', s=100, zorder=5, label='Trend Inflection Points')
                    
                    # Add annotations for inflection points
                    for _, point in inflection_df.iterrows():
                        date_str = point['ds'].strftime('%b %Y')
                        value = point['yhat']
                        ax4.annotate(f"{date_str}\n{value:.1f}", 
                                   (point['ds'], value),
                                   xytext=(0, 10),
                                   textcoords="offset points",
                                   ha='center',
                                   fontsize=9,
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                ax4.set_title('Trend Inflection Points in Funding Activity', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Date', fontsize=12)
                ax4.set_ylabel('Number of Rounds', fontsize=12)
                ax4.grid(True, alpha=0.3)
                if inflection_points:
                    ax4.legend()
                
                # Format x-axis dates
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            else:
                ax4.text(0.5, 0.5, 'Forecast data not available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'Funding rounds forecast data not available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
        
        # Add overall title
        fig.suptitle('Funding Growth Trajectory Dashboard', fontsize=18, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
        
        # Save dashboard
        output_path = os.path.join(self.timeseries_dir, f'funding_growth_dashboard_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Website time series dashboard saved to: {output_path}")
        return output_path
    
    def generate_all_website_dashboards(self, classification_results, timeseries_results, best_model="Random Forest"):
        """
        Generate all dashboards optimized for website display.
        
        Args:
            classification_results (dict): Dictionary containing classification model results
            timeseries_results (dict): Dictionary containing time series forecast results
            best_model (str): Name of the best model to display for classification dashboard
            
        Returns:
            dict: Paths to generated dashboards
        """
        dashboard_paths = {}
        
        # Generate classification dashboard
        try:
            self.logger.info(f"Generating website classification dashboard for {best_model}...")
            classification_path = self.generate_model_performance_dashboard(classification_results, best_model)
            if classification_path:
                dashboard_paths["classification"] = classification_path
        except Exception as e:
            self.logger.error(f"Error generating classification dashboard: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        # Generate time series dashboard
        try:
            self.logger.info("Generating website time series dashboard...")
            timeseries_path = self.generate_funding_growth_dashboard(timeseries_results)
            if timeseries_path:
                dashboard_paths["timeseries"] = timeseries_path
        except Exception as e:
            self.logger.error(f"Error generating time series dashboard: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        return dashboard_paths 