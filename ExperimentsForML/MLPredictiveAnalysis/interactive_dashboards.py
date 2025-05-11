import os
import sys
import numpy as np
import pandas as pd
import traceback
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("interactive_dashboards.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import components from funding_stage_prediction
try:
    from MLPredictiveAnalysis.funding_stage_prediction9 import (
        DataLoader, FeatureEngineering, EnhancedPipeline
    )
    from MLPredictiveAnalysis.enhanced_dashboards import EnhancedDashboards
except ImportError:
    logger.warning("Importing from package failed. Trying direct import...")
    sys.path.append('.')
    try:
        from funding_stage_prediction import (
            DataLoader, FeatureEngineering, EnhancedPipeline
        )
        from enhanced_dashboards import EnhancedDashboards
    except ImportError as e:
        logger.error(f"Error importing required modules: {str(e)}")
        logger.error("Please ensure you're running from the correct directory")
        sys.exit(1)

# Try to import plotly and dash
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html, callback
    from dash.dependencies import Input, Output, State
    DASH_AVAILABLE = True
except ImportError:
    logger.warning("Dash or Plotly not available. Only static dashboards will be generated.")
    DASH_AVAILABLE = False

class InteractiveClassificationDashboard:
    """
    Class for creating an interactive classification dashboard using Dash and Plotly.
    """
    
    def __init__(self, port=8050, output_dir="./interactive_dashboards"):
        """
        Initialize the interactive classification dashboard.
        
        Args:
            port (int): Port to run the Dash server on
            output_dir (str): Directory to save dashboard files
        """
        if not DASH_AVAILABLE:
            logger.error("Dash or Plotly not available. Cannot create interactive dashboard.")
            return
        
        self.port = port
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.app = dash.Dash(__name__)
        self.app.title = "Funding Stage Classification Dashboard"
        
        # Load data
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineering()
        
        # Will be populated later
        self.data = None
        self.model_results = None
        
    def load_data_and_results(self):
        """
        Load the data and model results.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load and process data
            merged_data = self.data_loader.merge_datasets()
            if merged_data.empty:
                logger.error("No data available after merging datasets")
                return False
            
            self.data = self.feature_engineer.extract_features(merged_data)
            
            # Load model results from the pipeline
            pipeline = EnhancedPipeline(output_dir="./output")
            model_dir = os.path.join(pipeline.output_dir, "model_results")
            
            self.model_results = {}
            
            if os.path.exists(model_dir):
                for model_file in os.listdir(model_dir):
                    if model_file.endswith('.pkl'):
                        model_name = os.path.splitext(model_file)[0]
                        model_path = os.path.join(model_dir, model_file)
                        try:
                            import pickle
                            with open(model_path, 'rb') as f:
                                self.model_results[model_name] = pickle.load(f)
                        except Exception as e:
                            logger.warning(f"Error loading model results from {model_path}: {str(e)}")
            
            if not self.model_results:
                logger.warning("No model results loaded. Dashboard will have limited functionality.")
                return False
            
            logger.info(f"Data and model results loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading data and results: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _create_model_performance_figure(self):
        """
        Create a figure for model performance comparison.
        
        Returns:
            plotly.graph_objects.Figure: Model performance figure
        """
        # Initialize metrics arrays
        model_names = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        # Extract metrics from model results
        for model_name, model_results in self.model_results.items():
            if isinstance(model_results, tuple) and len(model_results) > 1:
                metrics = model_results[1]
                model_names.append(model_name)
                accuracy_scores.append(metrics.get('accuracy', 0))
                precision_scores.append(metrics.get('precision', 0))
                recall_scores.append(metrics.get('recall', 0))
                f1_scores.append(metrics.get('f1', 0))
        
        if not model_names:
            # Return empty figure if no data
            return go.Figure().update_layout(
                title="No Model Performance Data Available",
                xaxis_title="",
                yaxis_title=""
            )
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for each metric
        fig.add_trace(go.Bar(x=model_names, y=accuracy_scores, name="Accuracy",
                             marker_color='rgb(55, 83, 109)'))
        fig.add_trace(go.Bar(x=model_names, y=precision_scores, name="Precision",
                             marker_color='rgb(26, 118, 255)'))
        fig.add_trace(go.Bar(x=model_names, y=recall_scores, name="Recall",
                             marker_color='rgb(58, 200, 225)'))
        fig.add_trace(go.Bar(x=model_names, y=f1_scores, name="F1 Score",
                             marker_color='rgb(0, 204, 150)'))
        
        # Update layout
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis=dict(
                tickformat=".2f",
                range=[0, 1]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            barmode='group'
        )
        
        return fig
    
    def _create_confusion_matrix_figure(self, model_name):
        """
        Create a confusion matrix figure for the specified model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            plotly.graph_objects.Figure: Confusion matrix figure
        """
        if model_name not in self.model_results:
            return go.Figure().update_layout(
                title=f"No Confusion Matrix Data Available for {model_name}",
                xaxis_title="",
                yaxis_title=""
            )
        
        model_results = self.model_results[model_name]
        
        # Check if we have the necessary data
        if isinstance(model_results, tuple) and len(model_results) > 1:
            if 'y_test' in model_results[1] and 'y_pred' in model_results[1]:
                y_test = model_results[1]['y_test']
                y_pred = model_results[1]['y_pred']
                
                # Get unique classes
                classes = np.unique(np.concatenate((y_test, y_pred)))
                
                # Create confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred, labels=classes)
                
                # Create heatmap
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=classes,
                    y=classes,
                    color_continuous_scale='Blues',
                    title=f'Confusion Matrix - {model_name}'
                )
                
                # Add text annotations
                for i in range(len(cm)):
                    for j in range(len(cm[i])):
                        fig.add_annotation(
                            x=j, y=i,
                            text=str(cm[i][j]),
                            showarrow=False,
                            font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
                        )
                
                return fig
        
        return go.Figure().update_layout(
            title=f"No Confusion Matrix Data Available for {model_name}",
            xaxis_title="",
            yaxis_title=""
        )
    
    def _create_feature_importance_figure(self, model_name):
        """
        Create a feature importance figure for the specified model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            plotly.graph_objects.Figure: Feature importance figure
        """
        if model_name not in self.model_results:
            return go.Figure().update_layout(
                title=f"No Feature Importance Data Available for {model_name}",
                xaxis_title="",
                yaxis_title=""
            )
        
        model_results = self.model_results[model_name]
        
        # Check if we have the necessary data
        if isinstance(model_results, tuple) and len(model_results) > 0:
            model = model_results[0]
            
            if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
                importances = model.feature_importances_
                feature_names = model.feature_names_in_
                
                # Sort features by importance
                indices = np.argsort(importances)
                
                # Get top 15 features
                top_indices = indices[-15:]
                top_importances = importances[top_indices]
                top_features = [feature_names[i] for i in top_indices]
                
                # Create horizontal bar chart
                fig = go.Figure(go.Bar(
                    y=top_features,
                    x=top_importances,
                    orientation='h',
                    marker=dict(color='rgba(50, 171, 96, 0.6)')
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'Feature Importance - {model_name}',
                    xaxis_title='Importance',
                    yaxis_title='Feature',
                    height=600,
                    yaxis=dict(autorange="reversed")  # Highest importance at top
                )
                
                return fig
        
        return go.Figure().update_layout(
            title=f"No Feature Importance Data Available for {model_name}",
            xaxis_title="",
            yaxis_title=""
        )
    
    def _create_roc_curve_figure(self, model_name):
        """
        Create a ROC curve figure for the specified model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            plotly.graph_objects.Figure: ROC curve figure
        """
        if model_name not in self.model_results:
            return go.Figure().update_layout(
                title=f"No ROC Curve Data Available for {model_name}",
                xaxis_title="",
                yaxis_title=""
            )
        
        model_results = self.model_results[model_name]
        
        # Check if we have the necessary data
        if isinstance(model_results, tuple) and len(model_results) > 1:
            if 'y_test' in model_results[1] and 'y_proba' in model_results[1]:
                y_test = model_results[1]['y_test']
                y_proba = model_results[1]['y_proba']
                
                # Get unique classes
                classes = np.unique(y_test)
                n_classes = len(classes)
                
                # Create figure
                fig = go.Figure()
                
                # Add diagonal reference line
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Random Classifier'
                ))
                
                # Binary classification
                if n_classes == 2:
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        line=dict(width=2, color='blue'),
                        name=f'ROC Curve (AUC = {roc_auc:.3f})'
                    ))
                
                # Multiclass classification
                else:
                    from sklearn.metrics import roc_curve, auc
                    from sklearn.preprocessing import label_binarize
                    
                    # Binarize the output
                    y_bin = label_binarize(y_test, classes=classes)
                    
                    # Compute ROC curve for each class
                    for i, class_idx in enumerate(range(n_classes)):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            line=dict(width=2),
                            name=f'Class {classes[i]} (AUC = {roc_auc:.3f})'
                        ))
                
                # Update layout
                fig.update_layout(
                    title=f'ROC Curve - {model_name}',
                    xaxis=dict(title='False Positive Rate', range=[0, 1]),
                    yaxis=dict(title='True Positive Rate', range=[0, 1.05]),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                return fig
        
        return go.Figure().update_layout(
            title=f"No ROC Curve Data Available for {model_name}",
            xaxis_title="",
            yaxis_title=""
        )
    
    def _create_funding_stage_distribution_figure(self):
        """
        Create a funding stage distribution figure.
        
        Returns:
            plotly.graph_objects.Figure: Funding stage distribution figure
        """
        if self.data is None or 'funding_stage' not in self.data.columns:
            return go.Figure().update_layout(
                title="No Funding Stage Data Available",
                xaxis_title="",
                yaxis_title=""
            )
        
        # Count occurrences of each funding stage
        stage_counts = self.data['funding_stage'].value_counts().reset_index()
        stage_counts.columns = ['Funding Stage', 'Count']
        
        # Create bar chart
        fig = px.bar(
            stage_counts,
            x='Funding Stage',
            y='Count',
            color='Count',
            color_continuous_scale='Viridis',
            text='Count',
            title='Funding Stage Distribution'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Funding Stage',
            yaxis_title='Count',
            coloraxis_showscale=False
        )
        
        # Improve text position
        fig.update_traces(textposition='outside')
        
        return fig
    
    def setup_layout(self):
        """
        Set up the Dash app layout.
        """
        if not DASH_AVAILABLE:
            logger.error("Dash not available. Cannot set up layout.")
            return
        
        # Load data and results if not already loaded
        if self.data is None or self.model_results is None:
            self.load_data_and_results()
        
        # Get available models
        available_models = list(self.model_results.keys()) if self.model_results else []
        default_model = available_models[0] if available_models else None
        
        # Set up the app layout
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1('Funding Stage Classification Dashboard', className='dashboard-title'),
                html.P('Interactive analysis of funding stage prediction models', className='dashboard-subtitle')
            ], className='header'),
            
            # Main content container
            html.Div([
                # Model Performance Overview
                html.Div([
                    html.H2('Model Performance Overview'),
                    dcc.Graph(id='model-performance-graph', figure=self._create_model_performance_figure())
                ], className='dashboard-card'),
                
                # Confusion Matrix
                html.Div([
                    html.H2('Confusion Matrix'),
                    html.Label('Select Model:'),
                    dcc.Dropdown(
                        id='confusion-model-dropdown',
                        options=[{'label': name, 'value': name} for name in available_models],
                        value=default_model,
                        clearable=False
                    ),
                    dcc.Graph(id='confusion-matrix-graph')
                ], className='dashboard-card'),
                
                # Feature Importance
                html.Div([
                    html.H2('Feature Importance'),
                    html.Label('Select Model:'),
                    dcc.Dropdown(
                        id='feature-importance-model-dropdown',
                        options=[{'label': name, 'value': name} for name in available_models],
                        value=default_model,
                        clearable=False
                    ),
                    dcc.Graph(id='feature-importance-graph')
                ], className='dashboard-card'),
                
                # ROC Curves
                html.Div([
                    html.H2('ROC Curves'),
                    html.Label('Select Model:'),
                    dcc.Dropdown(
                        id='roc-model-dropdown',
                        options=[{'label': name, 'value': name} for name in available_models],
                        value=default_model,
                        clearable=False
                    ),
                    dcc.Graph(id='roc-curve-graph')
                ], className='dashboard-card'),
                
                # Funding Stage Distribution
                html.Div([
                    html.H2('Funding Stage Distribution'),
                    dcc.Graph(id='funding-stage-distribution-graph', figure=self._create_funding_stage_distribution_figure())
                ], className='dashboard-card'),
                
            ], className='dashboard-content')
        ], className='dashboard-container')
        
        # Set up callbacks
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """
        Set up the Dash app callbacks.
        """
        if not DASH_AVAILABLE:
            logger.error("Dash not available. Cannot set up callbacks.")
            return
        
        # Confusion Matrix callback
        @self.app.callback(
            Output('confusion-matrix-graph', 'figure'),
            [Input('confusion-model-dropdown', 'value')]
        )
        def update_confusion_matrix(selected_model):
            return self._create_confusion_matrix_figure(selected_model)
        
        # Feature Importance callback
        @self.app.callback(
            Output('feature-importance-graph', 'figure'),
            [Input('feature-importance-model-dropdown', 'value')]
        )
        def update_feature_importance(selected_model):
            return self._create_feature_importance_figure(selected_model)
        
        # ROC Curve callback
        @self.app.callback(
            Output('roc-curve-graph', 'figure'),
            [Input('roc-model-dropdown', 'value')]
        )
        def update_roc_curve(selected_model):
            return self._create_roc_curve_figure(selected_model)
    
    def save_to_html(self):
        """
        Save the dashboard to an HTML file.
        
        Returns:
            str: Path to the saved file
        """
        if not DASH_AVAILABLE:
            logger.error("Dash not available. Cannot save to HTML.")
            return None
        
        try:
            # Set up the layout if not already done
            if not hasattr(self.app, 'layout') or self.app.layout is None:
                self.setup_layout()
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save to HTML
            output_path = os.path.join(self.output_dir, f'classification_dashboard_{timestamp}.html')
            
            # Try to use dash's write_html method
            try:
                self.app.write_html(output_path)
                logger.info(f"Dashboard saved to: {output_path}")
            except AttributeError:
                # Fallback to plotly's savefig method (limited functionality)
                logger.warning("App.write_html not available. Using basic HTML export.")
                with open(output_path, 'w') as f:
                    f.write(f"""
                    <html>
                    <head>
                        <title>Funding Stage Classification Dashboard</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                            h1 {{ color: #333; }}
                            .container {{ max-width: 1200px; margin: 0 auto; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>Funding Stage Classification Dashboard</h1>
                            <p>This is a static version of the dashboard. For full interactivity, run the Dash app.</p>
                            <div id="model-performance"></div>
                            <div id="funding-stage-distribution"></div>
                        </div>
                        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                        <script>
                            var modelPerformanceFig = {self._create_model_performance_figure().to_json()};
                            Plotly.newPlot('model-performance', modelPerformanceFig.data, modelPerformanceFig.layout);
                            
                            var stageDistributionFig = {self._create_funding_stage_distribution_figure().to_json()};
                            Plotly.newPlot('funding-stage-distribution', stageDistributionFig.data, stageDistributionFig.layout);
                        </script>
                    </body>
                    </html>
                    """)
                logger.info(f"Basic dashboard saved to: {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error saving dashboard to HTML: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def run_server(self):
        """
        Run the Dash app server.
        """
        if not DASH_AVAILABLE:
            logger.error("Dash not available. Cannot run server.")
            return
        
        # Set up the layout if not already done
        if not hasattr(self.app, 'layout') or self.app.layout is None:
            self.setup_layout()
        
        # Run the server
        logger.info(f"Starting dashboard server on port {self.port}...")
        self.app.run_server(debug=True, port=self.port)

class InteractiveTimeSeriesDashboard:
    """
    Class for creating an interactive time series dashboard using Dash and Plotly.
    """
    
    def __init__(self, port=8051, output_dir="./interactive_dashboards"):
        """
        Initialize the interactive time series dashboard.
        
        Args:
            port (int): Port to run the Dash server on
            output_dir (str): Directory to save dashboard files
        """
        # Similar implementation as the classification dashboard
        pass

# Main function to run the interactive dashboards
def main():
    """Run the interactive dashboards"""
    try:
        # Create output directory
        output_dir = "./interactive_dashboards"
        os.makedirs(output_dir, exist_ok=True)
        
        # Classification dashboard
        classification_dashboard = InteractiveClassificationDashboard(port=8050, output_dir=output_dir)
        
        # First, save to HTML for easy sharing
        html_path = classification_dashboard.save_to_html()
        if html_path:
            logger.info(f"Classification dashboard saved to: {html_path}")
        
        # Run the server (this will block until terminated)
        if DASH_AVAILABLE:
            logger.info("Starting interactive dashboard server...")
            classification_dashboard.run_server()
        else:
            logger.info("Dash not available. Created static HTML dashboard only.")
        
        return True
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main() 