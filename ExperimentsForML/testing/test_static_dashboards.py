#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for creating static HTML dashboards.
This script demonstrates how to create interactive dashboards as static HTML files.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import traceback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_classification_dashboard(output_dir):
    """Create a classification dashboard with multiple plots"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create mock data
        classes = ['Seed', 'Series A', 'Series B', 'Series C']
        model_names = ['Random Forest', 'XGBoost', 'Neural Network', 'Ensemble']
        metrics = {
            'accuracy': [0.85, 0.87, 0.84, 0.89],
            'precision': [0.83, 0.86, 0.82, 0.88],
            'recall': [0.82, 0.84, 0.81, 0.87],
            'f1': [0.825, 0.85, 0.815, 0.875]
        }
        
        # Create confusion matrix
        confusion_matrix = np.array([
            [45, 5, 0, 0],
            [7, 38, 5, 0],
            [0, 4, 36, 10],
            [0, 0, 8, 42]
        ])
        
        # Create feature importance data
        features = ['funding_amount_log', 'employees', 'employee_efficiency', 'funding_velocity']
        importance = [0.45, 0.25, 0.15, 0.15]
        
        # 1. Model Performance Comparison
        fig1 = go.Figure()
        
        # Add bars for each metric
        for metric, values in metrics.items():
            fig1.add_trace(go.Bar(
                x=model_names,
                y=values,
                name=metric.capitalize()
            ))
        
        fig1.update_layout(
            title='Model Performance Metrics',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            template='plotly_white'
        )
        
        # 2. Confusion Matrix
        fig2 = px.imshow(
            confusion_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=classes, 
            y=classes,
            color_continuous_scale='Blues',
            title='Confusion Matrix - Random Forest'
        )
        
        # Add text annotations
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[i])):
                fig2.add_annotation(
                    x=j, y=i, 
                    text=str(confusion_matrix[i][j]), 
                    showarrow=False, 
                    font=dict(color="white" if confusion_matrix[i][j] > confusion_matrix.max()/2 else "black")
                )
        
        # 3. Feature Importance
        fig3 = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h'
        ))
        
        fig3.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature'
        )
        
        # 4. ROC Curve
        fig4 = go.Figure()
        
        # Add ROC curves for each class
        for i, cls in enumerate(classes):
            # Mock ROC curve data
            fpr = np.linspace(0, 1, 100)
            tpr = np.linspace(0, 1, 100) ** (1.5 - i * 0.3)  # Different curve for each class
            
            fig4.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'Class {cls}'
            ))
        
        # Add diagonal line
        fig4.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Chance',
            line=dict(dash='dash', color='gray')
        ))
        
        fig4.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend_title='Class'
        )
        
        # Create HTML files
        metrics_path = os.path.join(output_dir, 'model_metrics.html')
        confusion_path = os.path.join(output_dir, 'confusion_matrix.html')
        importance_path = os.path.join(output_dir, 'feature_importance.html')
        roc_path = os.path.join(output_dir, 'roc_curves.html')
        
        # Write figures to HTML files
        fig1.write_html(metrics_path, include_plotlyjs='cdn', full_html=True)
        fig2.write_html(confusion_path, include_plotlyjs='cdn', full_html=True)
        fig3.write_html(importance_path, include_plotlyjs='cdn', full_html=True)
        fig4.write_html(roc_path, include_plotlyjs='cdn', full_html=True)
        
        # Create an index file linking all dashboards
        index_path = os.path.join(output_dir, 'index.html')
        
        with open(index_path, 'w') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Classification Dashboard</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                        h1 { color: #333; }
                        .dashboard-links { display: flex; flex-wrap: wrap; }
                        .dashboard-card { 
                            margin: 10px; 
                            padding: 15px; 
                            border: 1px solid #ddd; 
                            border-radius: 8px;
                            width: 300px;
                            background-color: #f9f9f9;
                        }
                        .dashboard-card h2 { margin-top: 0; }
                        .dashboard-card a { 
                            display: block; 
                            margin-top: 10px; 
                            padding: 8px 15px; 
                            background-color: #4CAF50; 
                            color: white; 
                            text-decoration: none; 
                            border-radius: 4px;
                            text-align: center;
                        }
                        .dashboard-card a:hover { background-color: #45a049; }
                    </style>
                </head>
                <body>
                    <h1>Classification Dashboard</h1>
                    <div class="dashboard-links">
                        <div class="dashboard-card">
                            <h2>Model Performance</h2>
                            <p>Compare accuracy, precision, recall, and F1 scores across different models.</p>
                            <a href="model_metrics.html">View Dashboard</a>
                        </div>
                        <div class="dashboard-card">
                            <h2>Confusion Matrix</h2>
                            <p>Visualize model predictions across different classes.</p>
                            <a href="confusion_matrix.html">View Dashboard</a>
                        </div>
                        <div class="dashboard-card">
                            <h2>Feature Importance</h2>
                            <p>Explore which features have the most impact on predictions.</p>
                            <a href="feature_importance.html">View Dashboard</a>
                        </div>
                        <div class="dashboard-card">
                            <h2>ROC Curves</h2>
                            <p>Analyze model performance with ROC curves for each class.</p>
                            <a href="roc_curves.html">View Dashboard</a>
                        </div>
                    </div>
                </body>
            </html>
            ''')
        
        logger.info(f"Classification dashboard created at {output_dir}")
        return {
            'index': index_path,
            'metrics': metrics_path,
            'confusion': confusion_path,
            'importance': importance_path,
            'roc': roc_path
        }
    
    except Exception as e:
        logger.error(f"Error creating classification dashboard: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_timeseries_dashboard(output_dir):
    """Create a time series dashboard with Prophet-like visualizations"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create mock time series data
        dates = pd.date_range(start='2018-01-01', end='2025-12-31', freq='MS')  # Month start
        n_dates = len(dates)
        
        # Split into historical and future
        today = pd.Timestamp.now()
        historical_cutoff = '2023-01-01'
        historical_mask = dates < historical_cutoff
        
        # Create forecast dataframe with Prophet-like structure
        trend = np.linspace(5, 15, n_dates) + np.random.normal(0, 0.5, n_dates)
        seasonal = 1 + 0.1 * np.sin(np.linspace(0, 8 * np.pi, n_dates))
        forecast = trend * seasonal + np.random.normal(0, 1, n_dates)
        
        forecast_df = pd.DataFrame({
            'ds': dates,
            'trend': trend,
            'yhat': forecast,
            'yhat_lower': forecast - 2,
            'yhat_upper': forecast + 2,
            'yearly': 1.5 * np.sin(np.linspace(0, 8 * np.pi, n_dates)),
            'monthly': 0.5 * np.sin(np.linspace(0, 50 * np.pi, n_dates)),
        })
        
        # Add historical data
        forecast_df['historical'] = historical_mask
        forecast_df['y'] = np.nan
        forecast_df.loc[historical_mask, 'y'] = forecast_df.loc[historical_mask, 'yhat'] + np.random.normal(0, 0.8, historical_mask.sum())
        
        # 1. Funding Trends Forecast
        fig1 = go.Figure()
        
        # Add historical data
        historical_df = forecast_df[forecast_df['historical']]
        fig1.add_trace(go.Scatter(
            x=historical_df['ds'],
            y=historical_df['y'],
            mode='markers',
            name='Historical Data',
            marker=dict(color='rgba(0, 0, 255, 0.8)', size=6)
        ))
        
        # Add forecast
        fig1.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='rgba(31, 119, 180, 1)', width=2)
        ))
        
        # Add prediction intervals
        fig1.add_trace(go.Scatter(
            x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
            y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='95% Confidence Interval'
        ))
        
        # Create a special trace for the vertical line for "today"
        # Find a date close to today in our range
        today_idx = (forecast_df['ds'] - today).abs().idxmin()
        today_date = forecast_df.loc[today_idx, 'ds']
        y_values = forecast_df['yhat'].values
        y_min, y_max = min(y_values) - 2, max(y_values) + 2
        
        # Add vertical line as a scatter trace
        fig1.add_trace(go.Scatter(
            x=[today_date, today_date],
            y=[y_min, y_max],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Today'
        ))
        
        # Add annotation for "Today"
        fig1.add_annotation(
            x=today_date,
            y=y_max,
            text="Today",
            showarrow=False,
            yshift=10
        )
        
        fig1.update_layout(
            title='Funding Trend Forecast',
            xaxis_title='Date',
            yaxis_title='Funding Amount (Millions USD)',
            template='plotly_white'
        )
        
        # 2. Forecast Components
        fig2 = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=["Trend Component", "Yearly Seasonality", "Monthly Seasonality"],
            shared_xaxes=True
        )
        
        # Add trend component
        fig2.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['trend'],
                mode='lines',
                name='Trend'
            ),
            row=1, col=1
        )
        
        # Add yearly seasonality
        fig2.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yearly'],
                mode='lines',
                name='Yearly Pattern'
            ),
            row=2, col=1
        )
        
        # Add monthly seasonality
        fig2.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['monthly'],
                mode='lines',
                name='Monthly Pattern'
            ),
            row=3, col=1
        )
        
        fig2.update_layout(
            title='Forecast Components',
            height=800,
            template='plotly_white'
        )
        
        # 3. Industry Comparison
        # Create mock data for different industries
        industries = ['Software', 'Fintech', 'Healthcare', 'E-commerce']
        industry_multipliers = [1.2, 0.9, 1.1, 1.0]
        
        fig3 = go.Figure()
        
        for i, industry in enumerate(industries):
            # Scale the baseline forecast for this industry
            industry_forecast = forecast_df['yhat'] * industry_multipliers[i]
            
            fig3.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=industry_forecast,
                mode='lines',
                name=industry
            ))
        
        # Add vertical line as a scatter trace
        y_values = forecast_df['yhat'].values
        y_min, y_max = min(y_values) * 0.9, max(y_values) * 1.2
        
        fig3.add_trace(go.Scatter(
            x=[today_date, today_date],
            y=[y_min, y_max],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Today'
        ))
        
        # Add annotation for "Today"
        fig3.add_annotation(
            x=today_date,
            y=y_max,
            text="Today",
            showarrow=False,
            yshift=10
        )
        
        fig3.update_layout(
            title='Industry Funding Forecast Comparison',
            xaxis_title='Date',
            yaxis_title='Funding Amount (Millions USD)',
            template='plotly_white'
        )
        
        # 4. Funding Stage Transitions
        # Create mock data for different funding stages
        stages = ['Seed', 'Series A', 'Series B', 'Series C', 'Late Stage']
        stage_data = np.zeros((len(dates), len(stages)))
        
        # Create evolving distribution of funding stages
        for i in range(len(dates)):
            progress = i / len(dates)
            # Seed decreases over time, late stage increases
            stage_data[i, 0] = 10 * (1 - progress * 0.8)  # Seed
            stage_data[i, 1] = 8 + progress * 4 - progress**2 * 8  # Series A
            stage_data[i, 2] = 5 + progress * 8 - progress**2 * 6  # Series B
            stage_data[i, 3] = 3 + progress**2 * 10  # Series C
            stage_data[i, 4] = 1 + progress**3 * 15  # Late Stage
            
            # Add some noise
            stage_data[i] += np.random.normal(0, 0.5, len(stages))
            
            # Ensure non-negative
            stage_data[i] = np.maximum(stage_data[i], 0.5)
        
        # Convert to percentage
        stage_pct = stage_data / stage_data.sum(axis=1, keepdims=True) * 100
        
        fig4 = go.Figure()
        
        # Create stacked area chart
        for i, stage in enumerate(stages):
            fig4.add_trace(go.Scatter(
                x=dates,
                y=stage_pct[:, i],
                mode='lines',
                stackgroup='one',
                name=stage
            ))
        
        # Add vertical line as scatter
        fig4.add_trace(go.Scatter(
            x=[today_date, today_date],
            y=[0, 100],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Today'
        ))
        
        # Add annotation
        fig4.add_annotation(
            x=today_date,
            y=100,
            text="Today",
            showarrow=False,
            yshift=10
        )
        
        fig4.update_layout(
            title='Funding Stage Composition Over Time',
            xaxis_title='Date',
            yaxis_title='Percentage of Funding',
            template='plotly_white'
        )
        
        # Write figures to HTML files
        trends_path = os.path.join(output_dir, 'funding_trends.html')
        components_path = os.path.join(output_dir, 'forecast_components.html')
        industry_path = os.path.join(output_dir, 'industry_comparison.html')
        stages_path = os.path.join(output_dir, 'stage_transitions.html')
        
        fig1.write_html(trends_path, include_plotlyjs='cdn', full_html=True)
        fig2.write_html(components_path, include_plotlyjs='cdn', full_html=True)
        fig3.write_html(industry_path, include_plotlyjs='cdn', full_html=True)
        fig4.write_html(stages_path, include_plotlyjs='cdn', full_html=True)
        
        # Create an index file linking all dashboards
        index_path = os.path.join(output_dir, 'index.html')
        
        with open(index_path, 'w') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Time Series Dashboard</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                        h1 { color: #333; }
                        .dashboard-links { display: flex; flex-wrap: wrap; }
                        .dashboard-card { 
                            margin: 10px; 
                            padding: 15px; 
                            border: 1px solid #ddd; 
                            border-radius: 8px;
                            width: 300px;
                            background-color: #f9f9f9;
                        }
                        .dashboard-card h2 { margin-top: 0; }
                        .dashboard-card a { 
                            display: block; 
                            margin-top: 10px; 
                            padding: 8px 15px; 
                            background-color: #4CAF50; 
                            color: white; 
                            text-decoration: none; 
                            border-radius: 4px;
                            text-align: center;
                        }
                        .dashboard-card a:hover { background-color: #45a049; }
                    </style>
                </head>
                <body>
                    <h1>Time Series Dashboard</h1>
                    <div class="dashboard-links">
                        <div class="dashboard-card">
                            <h2>Funding Trends</h2>
                            <p>View historical data and forecasted funding trends with confidence intervals.</p>
                            <a href="funding_trends.html">View Dashboard</a>
                        </div>
                        <div class="dashboard-card">
                            <h2>Forecast Components</h2>
                            <p>Explore the trend and seasonal components of the forecast.</p>
                            <a href="forecast_components.html">View Dashboard</a>
                        </div>
                        <div class="dashboard-card">
                            <h2>Industry Comparison</h2>
                            <p>Compare funding forecasts across different industries.</p>
                            <a href="industry_comparison.html">View Dashboard</a>
                        </div>
                        <div class="dashboard-card">
                            <h2>Stage Transitions</h2>
                            <p>Visualize how funding stage composition changes over time.</p>
                            <a href="stage_transitions.html">View Dashboard</a>
                        </div>
                    </div>
                </body>
            </html>
            ''')
        
        logger.info(f"Time series dashboard created at {output_dir}")
        return {
            'index': index_path,
            'trends': trends_path,
            'components': components_path,
            'industry': industry_path,
            'stages': stages_path
        }
    
    except Exception as e:
        logger.error(f"Error creating time series dashboard: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_main_index(output_dir, classification_dir, timeseries_dir):
    """Create a main index page linking to both dashboards"""
    try:
        index_path = os.path.join(output_dir, 'index.html')
        
        with open(index_path, 'w') as f:
            f.write(f'''
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Funding Stage Prediction Dashboards</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                        h1 {{ color: #333; }}
                        .dashboard-container {{ display: flex; flex-wrap: wrap; justify-content: center; }}
                        .dashboard-box {{ 
                            margin: 20px; 
                            padding: 30px; 
                            border: 1px solid #ddd; 
                            border-radius: 8px;
                            width: 400px;
                            background-color: #f9f9f9;
                            text-align: center;
                        }}
                        .dashboard-box h2 {{ margin-top: 0; color: #2C3E50; }}
                        .dashboard-box p {{ color: #7F8C8D; margin-bottom: 20px; }}
                        .dashboard-box a {{ 
                            display: inline-block; 
                            margin-top: 10px; 
                            padding: 12px 30px; 
                            background-color: #3498DB; 
                            color: white; 
                            text-decoration: none; 
                            border-radius: 4px;
                            font-weight: bold;
                            transition: background-color 0.3s;
                        }}
                        .dashboard-box a:hover {{ background-color: #2980B9; }}
                    </style>
                </head>
                <body>
                    <h1 style="text-align: center;">Funding Stage Prediction Dashboards</h1>
                    <div class="dashboard-container">
                        <div class="dashboard-box">
                            <h2>Classification Dashboard</h2>
                            <p>Explore model performance metrics, confusion matrices, feature importance, and more for funding stage classification models.</p>
                            <a href="{os.path.relpath(os.path.join(classification_dir, 'index.html'), output_dir)}">View Dashboard</a>
                        </div>
                        
                        <div class="dashboard-box">
                            <h2>Time Series Dashboard</h2>
                            <p>Analyze funding trends, forecast components, industry comparisons, and funding stage transitions over time.</p>
                            <a href="{os.path.relpath(os.path.join(timeseries_dir, 'index.html'), output_dir)}">View Dashboard</a>
                        </div>
                    </div>
                </body>
            </html>
            ''')
        
        logger.info(f"Main index page created at {index_path}")
        return index_path
    
    except Exception as e:
        logger.error(f"Error creating main index page: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def run_dashboard_test():
    """Run tests for creating static dashboards"""
    try:
        # Create output directories
        base_dir = './static_dashboards'
        classification_dir = os.path.join(base_dir, 'classification')
        timeseries_dir = os.path.join(base_dir, 'timeseries')
        
        os.makedirs(base_dir, exist_ok=True)
        
        # Create dashboards
        logger.info("Creating classification dashboard...")
        classification_paths = create_classification_dashboard(classification_dir)
        
        logger.info("Creating time series dashboard...")
        timeseries_paths = create_timeseries_dashboard(timeseries_dir)
        
        # Create main index
        if classification_paths and timeseries_paths:
            logger.info("Creating main index page...")
            main_index = create_main_index(base_dir, classification_dir, timeseries_dir)
            
            logger.info(f"All dashboards created successfully at {base_dir}")
            logger.info(f"Main index page: {main_index}")
            
            # Open the index page in a browser
            try:
                import webbrowser
                webbrowser.open('file://' + os.path.abspath(main_index))
                logger.info("Opened dashboard in browser")
            except:
                logger.warning("Could not open dashboard in browser")
                
            return True
        else:
            logger.error("Failed to create dashboards")
            return False
    
    except Exception as e:
        logger.error(f"Error in dashboard test: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    run_dashboard_test() 