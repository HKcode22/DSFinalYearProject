from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from google.cloud import storage
from io import StringIO
import os
import joblib
import numpy as np
import json
import glob
import sys
import uuid
import shutil
from datetime import datetime

# --- Calculate project root and add to sys.path ---
# Assuming app163.py is in cs163-main/appengine/
# Project root (cs163-main) is ONE level up.
try:
    SCRIPT_DIR_APP163 = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_APP163 = os.path.dirname(SCRIPT_DIR_APP163) # This should be cs163-main
    if PROJECT_ROOT_APP163 not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_APP163)
    # Add backend to sys.path as well to help find ML module
    BACKEND_DIR_APP163 = os.path.join(PROJECT_ROOT_APP163, 'backend')
    if BACKEND_DIR_APP163 not in sys.path:
        sys.path.insert(0, BACKEND_DIR_APP163)
    # Add frontend to sys.path for appPredictionEngine utilities if needed (though less likely for direct use here)
    FRONTEND_DIR_APP163 = os.path.join(PROJECT_ROOT_APP163, 'frontend')
    if FRONTEND_DIR_APP163 not in sys.path:
        sys.path.insert(0, FRONTEND_DIR_APP163)
    
    # Test import to see if path is correct
    from ML.funding_stage_predictionORIGINAL import FeatureEngineering, ModelManager, AnomalyDetector, NumpyEncoder
    print("Successfully imported custom ML modules in app163.py")

except ImportError as e:
    print(f"Error importing custom ML modules in app163.py: {e}. Prediction engine will not work.")
    FeatureEngineering = None
    ModelManager = None
    AnomalyDetector = None
    NumpyEncoder = None
except NameError: # __file__ is not defined
    print("__file__ not defined in app163.py, sys.path modification might not be effective.")
    FeatureEngineering = None
    ModelManager = None
    AnomalyDetector = None
    NumpyEncoder = None

# --- Asset Synchronization and Path Helpers ---

# Define source and target directories relative to the project root
# Ensure PROJECT_ROOT_APP163 is correctly defined above
ASSET_TARGET_DIR = os.path.join(PROJECT_ROOT_APP163, 'appengine', 'assets')
VISUALIZATIONS_SOURCE_DIR = os.path.join(PROJECT_ROOT_APP163, 'backend', 'MainOutput', 'visualizations')
PROTOTYPE_SOURCE_DIR = os.path.join(PROJECT_ROOT_APP163, 'backend', 'MainOutput', 'prototype_dashboard')

def find_latest_asset_file(directory, pattern):
    """Finds the latest file in a directory matching a glob pattern."""
    try:
        search_pattern = os.path.join(directory, pattern)
        list_of_files = glob.glob(search_pattern)
        if not list_of_files:
            # print(f"[Assets] No files found matching: {search_pattern}")
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        # print(f"[Assets] Found latest for {pattern}: {os.path.basename(latest_file)}")
        return latest_file
    except Exception as e:
        print(f"[Assets] Error finding latest file for {pattern} in {directory}: {e}")
        return None

def sync_latest_assets():
    """Copies the latest required plot files from backend output to app assets."""
    print("[Assets] Starting asset synchronization...")
    os.makedirs(ASSET_TARGET_DIR, exist_ok=True) # Ensure assets dir exists

    assets_to_sync = [
        # (Source Directory, Source Pattern, Asset Type Name)
        (PROTOTYPE_SOURCE_DIR, 'bay_area_funding_trend_interactive_*.html', 'Forecast HTML'),
        (VISUALIZATIONS_SOURCE_DIR, 'calibration_plot_Random_Forest_Tuned_*.png', 'RF Calibration Plot'), # Specific RF calibration
        (VISUALIZATIONS_SOURCE_DIR, 'model_comparison_accuracy_rmse_*.png', 'Model Comparison Plot'),
        (VISUALIZATIONS_SOURCE_DIR, 'funding_stage_dist_*.png', 'Funding Dist Plot'),
        (VISUALIZATIONS_SOURCE_DIR, 'feature_importance_RandomForestClassifier_*.png', 'RF Feature Importance Plot'),
        # Add patterns for other plots if needed (e.g., logo, other images used in layout)
        # (VISUALIZATIONS_SOURCE_DIR, 'your_other_plot_pattern_*.png', 'Your Other Plot'),
    ]

    copied_files_count = 0
    for src_dir, pattern, name in assets_to_sync:
        latest_src_path = find_latest_asset_file(src_dir, pattern)
        if latest_src_path:
            dest_filename = os.path.basename(latest_src_path)
            dest_path = os.path.join(ASSET_TARGET_DIR, dest_filename)
            try:
                # Copy if dest doesn't exist or src is newer
                if not os.path.exists(dest_path) or os.path.getmtime(latest_src_path) > os.path.getmtime(dest_path):
                    shutil.copy2(latest_src_path, dest_path)
                    print(f"[Assets] Copied/Updated {name} to assets: {dest_filename}")
                    copied_files_count += 1
                # else:
                    # print(f"[Assets] {name} already up-to-date in assets.")
            except Exception as e:
                print(f"[Assets] Failed to copy {name} from {latest_src_path} to {dest_path}: {e}")
        else:
            print(f"[Assets] Warning: Latest source file for {name} (pattern: {pattern}) not found in {src_dir}.")

    print(f"[Assets] Asset synchronization finished. Copied/Updated {copied_files_count} files.")

def get_asset_url_path(pattern_prefix):
    """Finds the latest asset in the assets folder and returns its URL path."""
    latest_asset_file = find_latest_asset_file(ASSET_TARGET_DIR, f"{pattern_prefix}*")
    if latest_asset_file:
        # print(f"[Assets] Using asset: {os.path.basename(latest_asset_file)} for prefix {pattern_prefix}")
        return app.get_asset_url(os.path.basename(latest_asset_file))
    else:
        print(f"[Assets] Warning: No asset found in {ASSET_TARGET_DIR} for prefix '{pattern_prefix}'")
        # Return a placeholder or default image URL if desired
        # return app.get_asset_url('placeholder.png')
        return "" # Return empty string if not found

# --- Run Asset Sync on Startup ---
sync_latest_assets()

# Initialize Google Cloud Storage client
def get_gcs_data(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_text()
    return pd.read_csv(StringIO(data))



# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Startup Analytics Platform"
server = app.server

# Navbar component (fixed the parenthesis issue here)
def get_navbar():
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="/assets/logo.png", height="30px")),
                    dbc.Col(dbc.NavbarBrand("Startup Analytics", className="ms-2")),
                ], align="center", className="g-0"),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Home", href="/")),
                dbc.NavItem(dbc.NavLink("Project Objective", href="/objective")),
                dbc.NavItem(dbc.NavLink("Methods", href="/methods")),
                dbc.NavItem(dbc.NavLink("Findings", href="/findings")),
                dbc.NavItem(dbc.NavLink("Interactive Prediction", href="/interactive-prediction")),
            ], className="ms-auto", navbar=True)
        ]),
        color="primary",
        dark=True,
        sticky="top"
    )


# Corrected Home/Landing Page
home_layout = html.Div([
    get_navbar(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Startup Funding Stage Analytics & Prediction", className="text-center my-4"),
                html.P("Leveraging data science to forecast funding trajectories and understand key success factors in the Bay Area startup ecosystem.", className="lead text-center mb-5"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Bay Area Startup Funding Stage Forecast", className="text-center mb-3"),
                html.Iframe(
                    src=get_asset_url_path('bay_area_funding_trend_interactive_'), 
                    width="100%",
                    height="500px",
                    style={"border": "1px solid #ddd"}
                ),
                html.P("Interactive forecast of median funding stage trends for Bay Area startups, based on historical data and Prophet time-series modeling.", 
                      className="text-center mt-2")
            ], md=12),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.H4("Random Forest Model Calibration", className="text-center mb-3 mt-4"),
                html.Img(
                    src=get_asset_url_path('calibration_plot_Random_Forest_Tuned_'), 
                    className="img-fluid",
                    style={"display": "block", "margin-left": "auto", "margin-right": "auto", "max-height": "500px"}
                ),
                html.P("Calibration plot for our Random Forest model, illustrating the relationship between predicted probabilities and actual outcomes.", 
                      className="text-center mt-2")
            ], md=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("About This Project", className="mt-5"),
                    html.P("This platform leverages data scraped from sources like TopStartup.io, FundraiserInsider, and GrowthList for Bay Area startups. It utilizes machine learning to predict funding stages based on various company attributes and funding history.", className="mb-3"),
                    
                    html.H4("Key Capabilities:", className="mt-4"),
                    html.Ul([
                        html.Li("Analysis of funding stage distributions and trends."),
                        html.Li("Training and evaluation of multiple predictive models (e.g., Random Forest, XGBoost, LightGBM, Ensembles) for funding stage classification."),
                        html.Li("Identification of key features influencing funding stage predictions."),
                        html.Li("An interactive engine to predict the funding stage for user-provided startup data.")
                    ])
                ], className="p-4 bg-light rounded")
            ], width=12)
        ])
    ], className="my-4")
])

# Project Objective Page
objective_layout = html.Div([
    get_navbar(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Project Objective", className="text-center my-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2("Goals"),
                    html.P("The primary objectives of this project are to:"),
                    html.Ul([
                        html.Li("Develop and evaluate robust machine learning models to predict the funding stage of Bay Area startups."),
                        html.Li("Engineer relevant features from diverse data sources to improve prediction accuracy."),
                        html.Li("Analyze historical funding trends and identify key factors influencing startup progression through funding rounds."),
                        html.Li("Provide a platform for exploring these insights and offering a predictive tool for startup analysis.")
                    ]),
                    
                    html.H2("Data Sources", className="mt-5"),
                    html.P("Our analysis leverages data scraped from multiple online sources, focusing on startups in the Bay Area:"),
                    html.Ul([
                        html.Li("TopStartup.io: Information on various startups including funding rounds, industry, and company details."),
                        html.Li("FundraiserInsider: Data on recent fundraising activities, amounts, and investor participation."),
                        html.Li("GrowthList: Lists of growing companies, often indicating recent funding or significant traction."),
                        html.Li("Data collected pertains to company names, funding dates, funding amounts, funding stages/types, industry, employee counts, and headquarters/location.")
                    ]),
                    html.P("This data is aggregated, cleaned, and processed to create a comprehensive dataset for model training and analysis.", className="mt-2 text-muted"),
                    
                    html.H2("Target Audience", className="mt-5"),
                    html.Ul([
                        html.Li("Venture Capitalists and Angel Investors seeking data-driven insights for investment decisions."),
                        html.Li("Startup Founders and Executives aiming to understand funding landscapes and benchmark their progress."),
                        html.Li("Market Analysts and Researchers studying startup ecosystems and economic trends."),
                        html.Li("Students and enthusiasts of data science and machine learning in the context of business and finance.")
                    ])
                ], className="p-4")
            ], md=8),
            
            dbc.Col([
                html.Div([
                    html.H4("Data Overview", className="mb-3"),
                    html.P("Key aspects of our processed dataset:"),
                    # The following numbers are illustrative based on the capabilities of funding_stage_predictionORIGINAL.py
                    # Actual numbers would come from a specific run's summary.json.
                    html.Ul([
                        html.Li(f"Dataset typically comprises several hundred to a few thousand unique funding events after merging and cleaning."),
                        html.Li(f"A comprehensive set of ~30-40 engineered features are created, including temporal, categorical, and interaction terms."),
                        html.Li("Funding stages are mapped to a consistent numerical representation for modeling."),
                        html.Li("Models are trained to predict these numerical funding stages.")
                    ]),
                    html.Img(src=get_asset_url_path('data-flow'), className="img-fluid mt-3") # Assume data-flow.png is copied to assets
                ], className="p-4 border rounded")
            ], md=4)
        ])
    ], className="my-4")
])

# Analytical Methods Page
methods_layout = html.Div([
    get_navbar(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Analytical Methods", className="text-center my-4"),
                html.P("Our approach to predicting startup funding stages involves data preprocessing, feature engineering, model training, and evaluation.", className="text-center text-muted mb-5"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2("Data Preprocessing & Merging"),
                    html.P("Data from various sources (TopStartup.io, FundraiserInsider, GrowthList) is loaded, validated, and merged. This involves:"),
                    html.Ul([
                        html.Li("Standardizing company names and funding stages."),
                        html.Li("Parsing and converting funding amounts to a consistent numerical format (USD)."),
                        html.Li("Handling missing data through imputation or by assigning meaningful defaults (e.g., 'Unknown' for industry)."),
                        html.Li("Validating company data for consistency and quality, with low-confidence records being potentially excluded.")
                    ]),

                    html.H2("Feature Engineering", className="mt-4"),
                    html.P("A rich set of features is engineered to capture various aspects of a startup's profile and funding history:"),
                    html.Ul([
                        html.Li("Temporal Features: Funding year, month (with cyclical encoding), months since first funding, time since last funding."),
                        html.Li("Funding Amount Features: Log-transformed funding amount, ratio of current funding to previous, funding amount vs. industry median."),
                        html.Li("Company-Specific Features: Number of previous rounds, funding velocity (average months between rounds), employee count (binned), employee efficiency (funding per employee)."),
                        html.Li("Categorical Features: Standardized industry categories (with rare categories grouped into 'Other'), binned company age, binned employee counts."),
                        html.Li("Interaction Features: Combinations of features like funding amount x age, employees x rounds.")
                    ]),

                    html.H2("Modeling Approach", className="mt-4"),
                    html.P("Multiple machine learning models are trained and evaluated for predicting the numerically encoded funding stage. This includes:"),
                    html.Ul([
                        html.Li("Models like Random Forest, XGBoost, Gradient Boosting, LightGBM, and Decision Trees."),
                        html.Li("Hyperparameter tuning using RandomizedSearchCV to find optimal model configurations."),
                        html.Li("Training on a scaled feature set (using StandardScaler)."),
                        html.Li("Calibration of models (e.g., using CalibratedClassifierCV) to improve probability estimates."),
                        html.Li("Consideration of ensemble techniques like Stacking (e.g., with a Random Forest meta-learner).")
                    ]),
                                        
                    html.H2("Model Validation & Evaluation", className="mt-4"),
                    html.Ul([
                        html.Li("Stratified train-test split (e.g., 80/20) to preserve class distribution."),
                        html.Li("Cross-validation (e.g., 5-fold) on the training set during tuning and for some models."),
                        html.Li("Evaluation metrics: Accuracy, Precision, Recall, F1-score (weighted), RMSE (Root Mean Squared Error), Confusion Matrix, and Classification Reports."),
                        html.Li("Analysis of feature importances to understand drivers of predictions."),
                        html.Li("Diagnostic plots such as ROC curves and calibration plots to assess model reliability.")
                    ]),
                    html.H4("Technical References", className="mt-4"),
                    html.Ul([
                        html.Li(html.A("Scikit-learn Documentation", href="https://scikit-learn.org/", target="_blank")),
                        html.Li(html.A("XGBoost Documentation", href="https://xgboost.readthedocs.io/", target="_blank")),
                        html.Li(html.A("LightGBM Documentation", href="https://lightgbm.readthedocs.io/en/latest/", target="_blank")),
                        html.Li(html.A("Prophet Documentation", href="https://facebook.github.io/prophet/docs/quick_start.html", target="_blank"))
                    ])
                ], className="p-4 border rounded mb-4")
            ], md=8),
            
            dbc.Col([
                html.Div([
                    html.H4("Model Performance Comparison", className="mb-3"),
                    html.Img(src=get_asset_url_path('model_comparison_accuracy_rmse_'), className="img-fluid mb-3"),
                    html.P("Comparison of different models based on key performance metrics like Accuracy and RMSE.", className="text-muted"),
                    
                    html.H4("Illustrative Model Architecture", className="mt-4"),
                    html.Img(src=get_asset_url_path('model-architecture'), className="img-fluid mb-3"),
                    html.P("Our pipeline involves data ingestion, preprocessing, feature engineering, model training (often including ensembles), and evaluation.", className="text-muted"),

                ], className="p-4")
            ], md=4)
        ])
    ], className="my-4")
])

# Major Findings Page
findings_layout = html.Div([
    get_navbar(),
    dbc.Container([
        dbc.Row([
            dbc.Col([html.H1("Major Findings & Insights", className="text-center my-4")], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Distribution of Funding Stages", className="text-center mb-3"),
                html.Img(src=get_asset_url_path('funding_stage_dist_'), className="img-fluid", 
                         style={"display": "block", "margin-left": "auto", "margin-right": "auto", "max-height": "450px"}),
                html.P("The dataset shows a typical distribution of startups across funding stages, often with larger numbers in earlier stages like Seed and Series A, and fewer in later or specialized stages like Post-IPO or Debt Financing.", className="text-center mt-2")
            ], md=6),
            dbc.Col([
                html.H4("Key Predictive Features (Example: Random Forest)", className="text-center mb-3"),
                html.Img(src=get_asset_url_path('feature_importance_RandomForestClassifier_'), className="img-fluid", 
                         style={"display": "block", "margin-left": "auto", "margin-right": "auto", "max-height": "450px"}),
                html.P("Features like 'funding_amount_log', 'months_since_first_funding', 'previous_rounds', and 'employee_efficiency' frequently emerge as important predictors of a startup's funding stage.", className="text-center mt-2")
            ], md=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2("Key Insights & Observations", className="mt-5"),
                    html.P("Based on the analysis performed by our funding stage prediction models and data exploration:"),
                    html.Ul([
                        html.Li("Predictive models (e.g., Random Forest, XGBoost, LightGBM) typically achieve accuracies in the 80-90% range on this dataset, though performance varies between models and specific runs."),
                        html.Li("While models like XGBoost might achieve higher headline accuracy, they may sometimes overfit and generalize less well to known startups compared to models like Random Forest, indicating the importance of evaluating beyond just accuracy."),
                        html.Li("The current dataset size (~1700 records after processing) is useful, but acquiring significantly more data would likely improve model robustness, predictive power, and the ability to accurately classify less frequent funding stages."),
                        html.Li("Later or less common stages like 'Post-IPO,' 'Series D+,' or 'Venture - Series Unknown' remain challenging for models due to fewer training samples. Prediction accuracy for these specific stages varies significantly between models."),
                        html.Li("Feature importance analysis consistently highlights that a company's financial history (funding amount, previous rounds), operational scale (employee count), and age/maturity (months since first funding) are critical indicators of its funding stage."),
                        html.Li("Model calibration is crucial; an uncalibrated model might be accurate but provide unreliable probability scores."),
                        html.Li("Time-series analysis indicates general trends but should be viewed as high-level forecasts subject to market dynamics.")
                    ]),
                    html.P("Continuous model retraining and evaluation with new data are crucial for maintaining predictive performance in the dynamic startup landscape.", className="mt-3")
                ], className="p-4 bg-light rounded mt-4")
            ], width=12)
        ])
    ], className="my-4")
])

# Placeholder for Interactive Prediction Page
interactive_prediction_layout = html.Div([
    get_navbar(),
    dbc.Container([
        dbc.Row([
            dbc.Col([html.H1("Interactive Prediction Engine", className="text-center my-4")], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Model:"),
                dcc.Dropdown(
                    id='interactive-model-selector',
                    options=[
                        # Updated options based on observed saved models + adding LGBM back
                        {'label': 'Stacking Ensemble (RF Meta)', 'value': 'Dashboard_Model_Stacking_Ensemble_(RF_Meta)'},
                        {'label': 'Random Forest (All Features)', 'value': 'Dashboard_Model_RandomForest_Dashboard_AllFeatures'},
                        {'label': 'XGBoost (Tuned)', 'value': 'XGBoost_Tuned'}, # Use base name
                        {'label': 'LightGBM (Tuned)', 'value': 'LightGBM_Tuned'}, # ADDED BACK - Use likely base name pattern
                        {'label': 'Decision Tree (Calibrated)', 'value': 'Dashboard_Model_Decision_Tree_(Calibrated)'}, # Still commented out as not seen
                        {'label': 'Best Model (from Latest Summary)', 'value': 'BEST_FROM_SUMMARY'}
                    ],
                    value='Dashboard_Model_Stacking_Ensemble_(RF_Meta)', # Default model remains Stacking
                    clearable=False,
                    className="mb-3"
                ),

                dbc.Label("Company Name (Optional for enhanced anomaly check):"),
                dbc.Input(id="interactive-company-name", placeholder="E.g., Innovatech Solutions", type="text", className="mb-2"),

                dbc.Label("Funding Amount (USD) - Raw value, e.g., 5000000 for $5M:"),
                dbc.Input(id="interactive-funding-amount", placeholder="E.g., 5000000", type="number", className="mb-2"),
                
                dbc.Label("Number of Employees - Raw count:"),
                dbc.Input(id="interactive-employees", placeholder="E.g., 50", type="number", className="mb-2"),
                
                dbc.Label("Industry - As specific as possible:"),
                dbc.Input(id="interactive-industry", placeholder="E.g., Artificial Intelligence, FinTech, SaaS", type="text", className="mb-2"),
                                
                dbc.Label("Months Since First Funding - Total months:"),
                dbc.Input(id="interactive-months-since-first", placeholder="E.g., 24", type="number", className="mb-2"),
                
                dbc.Label("Previous Funding Rounds - Count of prior rounds:"),
                dbc.Input(id="interactive-previous-rounds", placeholder="E.g., 2", type="number", className="mb-2"),
                
                # For features like 'funding_year', 'funding_month', 'month_sin', 'month_cos', 
                # these are typically derived from a 'funding_date'. 
                # For this interactive engine, we will use the current date to derive these if FeatureEngineering requires them.
                # Explicit inputs for these derived features are usually not user-friendly.

                dbc.Button("Predict Funding Stage", id="interactive-predict-button", color="primary", className="mt-3 mb-3 n-clicks-0"), # Added n_clicks-0
                
                html.Div(id='interactive-prediction-output', className="mt-4 p-3 border rounded", children="Prediction results will appear here.")
                
            ], md=8, className="mx-auto") # Center the column
        ])
    ], className="my-4", fluid=True)
])

# Globals for holding loaded model and related objects for the interactive engine
# This is a simplified cache; for production, consider a more robust caching/loading strategy
interactive_engine_globals = {
    'model_manager': None, # Will hold instance of ModelManager
    'feature_engineer': None, # Will hold instance of FeatureEngineering
    'loaded_model_name_cache': None, # Tracks which model (pattern) is currently loaded
    'class_mapping_from_summary': {},
    'feature_names_from_summary': [],
    'age_bin_edges_from_summary': None, # For FeatureEngineer
    'age_bin_labels_from_summary': None  # For FeatureEngineer
}

# Function to initialize or get the ModelManager and FeatureEngineer
def get_ml_tools():
    global PROJECT_ROOT_APP163 # Ensure it's accessible if defined globally during sys.path setup
    if 'PROJECT_ROOT_APP163' not in globals():
        # Fallback or re-calculate if not in global scope (e.g. if app runs in a way __file__ was not set initially)
        try:
            SCRIPT_DIR_APP163_FALLBACK = os.path.dirname(os.path.abspath(__file__))
            PROJECT_ROOT_APP163 = os.path.dirname(SCRIPT_DIR_APP163_FALLBACK)
            print("Recalculated PROJECT_ROOT_APP163 in get_ml_tools")
        except NameError:
            PROJECT_ROOT_APP163 = "." # Last resort, might not be correct
            print("Warning: PROJECT_ROOT_APP163 defaulted to '.' in get_ml_tools")

    if FeatureEngineering and interactive_engine_globals['feature_engineer'] is None:
        interactive_engine_globals['feature_engineer'] = FeatureEngineering()
        print("Initialized FeatureEngineering for interactive engine.")
    
    # --- Determine paths relative to PROJECT_ROOT_APP163 (cs163-main) ---
    output_dir_interactive = os.path.join(PROJECT_ROOT_APP163, 'backend', 'MainOutput')
    models_dir_interactive = os.path.join(output_dir_interactive, 'models')
    # Print the directory that will be used for models, for debugging
    print(f"[get_ml_tools] Model directory for ModelManager: {models_dir_interactive}") # ADDED PRINT

    if ModelManager and interactive_engine_globals['model_manager'] is None:
        if not os.path.exists(models_dir_interactive):
            print(f"CRITICAL_INTERACTIVE: Models directory does not exist at {models_dir_interactive}. Predictions may fail if models are not deployed here.")
        interactive_engine_globals['model_manager'] = ModelManager(model_dir=models_dir_interactive)
        print(f"Initialized ModelManager for interactive engine, model_dir: {models_dir_interactive}")

    # Load summary data to get mappings and feature names if not already loaded or if they are empty
    if not interactive_engine_globals['class_mapping_from_summary'] or not interactive_engine_globals['feature_names_from_summary']:
        try:
            summary_files = glob.glob(os.path.join(output_dir_interactive, "summary_*.json"))
            if summary_files:
                latest_summary_path = max(summary_files, key=os.path.getctime)
                with open(latest_summary_path, 'r') as f:
                    summary_data = json.load(f) 
                
                interactive_engine_globals['class_mapping_from_summary'] = {int(k): str(v) for k,v in summary_data.get('class_mapping', {}).items()}
                interactive_engine_globals['feature_names_from_summary'] = summary_data.get('feature_names', [])
                interactive_engine_globals['age_bin_edges_from_summary'] = summary_data.get('age_bin_edges')
                interactive_engine_globals['age_bin_labels_from_summary'] = summary_data.get('age_bin_labels')
                
                print(f"Interactive engine: Loaded class_mapping ({len(interactive_engine_globals['class_mapping_from_summary'])}) and feature_names ({len(interactive_engine_globals['feature_names_from_summary'])}) from {latest_summary_path}")
                print(f"Interactive engine: Age bins from summary - Edges: {interactive_engine_globals['age_bin_edges_from_summary'] is not None}, Labels: {interactive_engine_globals['age_bin_labels_from_summary'] is not None}")

            else:
                print(f"Interactive engine: No summary file found in {output_dir_interactive} to load class mapping/feature names.")
        except Exception as e:
            print(f"Interactive engine: Error loading summary for class mapping/features: {e}")
            interactive_engine_globals['class_mapping_from_summary'] = {}
            interactive_engine_globals['feature_names_from_summary'] = []
            interactive_engine_globals['age_bin_edges_from_summary'] = None
            interactive_engine_globals['age_bin_labels_from_summary'] = None

    return interactive_engine_globals['model_manager'], interactive_engine_globals['feature_engineer']

# Callback for the interactive prediction engine
@callback(
    Output('interactive-prediction-output', 'children'),
    Input('interactive-predict-button', 'n_clicks'),
    [State('interactive-model-selector', 'value'),
     State('interactive-company-name', 'value'),
     State('interactive-funding-amount', 'value'),
     State('interactive-employees', 'value'),
     State('interactive-industry', 'value'),
     State('interactive-months-since-first', 'value'),
     State('interactive-previous-rounds', 'value')]
)
def handle_interactive_prediction(n_clicks, selected_model_pattern, company_name, 
                                funding_amount, employees, industry, 
                                months_since_first, previous_rounds):
    if n_clicks is None or n_clicks == 0:
        return "Please enter features and click predict."

    print(f"[Interactive Predict] Button clicked. Selected model pattern: {selected_model_pattern}")

    # --- Initialize ML Tools --- 
    # Check if classes were imported correctly
    if 'FeatureEngineering' not in globals() or FeatureEngineering is None or \
       'ModelManager' not in globals() or ModelManager is None or \
       'AnomalyDetector' not in globals(): # AnomalyDetector can be None if import failed, handle gracefully
        print("[Interactive Predict] Error: Core ML classes (FeatureEngineering, ModelManager) are not available.")
        return dbc.Alert("Error: Core ML components not loaded. Prediction unavailable. Check server logs.", color="danger")

    model_manager, feature_engineer = get_ml_tools()

    if not model_manager: # Feature engineer might be usable even if model fails, but prediction needs model_manager
        print("[Interactive Predict] Error: ModelManager could not be initialized by get_ml_tools().")
        return dbc.Alert("Error: Model Manager tool could not be initialized. Check server logs.", color="danger")
    if not feature_engineer:
        # Attempt re-initialization if it failed before
        try: 
             interactive_engine_globals['feature_engineer'] = FeatureEngineering()
             feature_engineer = interactive_engine_globals['feature_engineer']
             print("[Interactive Predict] Re-initialized FeatureEngineering in callback.")
        except Exception as fe_init_err:
             print(f"[Interactive Predict] Error: Failed to initialize FeatureEngineer: {fe_init_err}")
             return dbc.Alert("Error: Feature Engineering tool could not be initialized. Check server logs.", color="danger")

    # --- Load Model --- 
    global PROJECT_ROOT_APP163 # Required for path construction
    if 'PROJECT_ROOT_APP163' not in globals() or not PROJECT_ROOT_APP163:
        try:
            SCRIPT_DIR_APP163_CB = os.path.dirname(os.path.abspath(__file__))
            PROJECT_ROOT_APP163 = os.path.dirname(SCRIPT_DIR_APP163_CB)
            print("[Interactive Predict] Re-initialized PROJECT_ROOT_APP163 in callback.")
        except NameError:
            PROJECT_ROOT_APP163 = "."
            print("[Interactive Predict] Warning: PROJECT_ROOT_APP163 defaulted to '.' in callback.")
    output_dir_interactive = os.path.join(PROJECT_ROOT_APP163, 'backend', 'MainOutput')

    actual_model_to_load = selected_model_pattern
    if selected_model_pattern == 'BEST_FROM_SUMMARY':
        try:
            summary_files = glob.glob(os.path.join(output_dir_interactive, "summary_*.json"))
            if not summary_files: return dbc.Alert("Error: Cannot determine BEST_FROM_SUMMARY, no summary file found.", color="danger")
            latest_summary_path = max(summary_files, key=os.path.getctime)
            with open(latest_summary_path, 'r') as f: summary_data = json.load(f)
            best_model_name_from_summary = summary_data.get('best_model_by_accuracy')
            if not best_model_name_from_summary: return dbc.Alert("Error: Best model name not found in summary.", color="danger")
            actual_model_to_load = best_model_name_from_summary.replace(" ", "_").replace("(", "").replace(")", "")
            print(f"[Interactive Predict] BEST_FROM_SUMMARY resolved to: {actual_model_to_load}")
        except Exception as e: return dbc.Alert(f"Error resolving BEST_FROM_SUMMARY: {str(e)}", color="danger")

    if model_manager.model is None or interactive_engine_globals['loaded_model_name_cache'] != actual_model_to_load:
        print(f"[Interactive Predict] Attempting to load model for pattern: {actual_model_to_load}...")
        if hasattr(model_manager, 'model_dir'): print(f"[Interactive Predict] ModelManager loading from: {model_manager.model_dir}")
        load_success = model_manager.load_model_joblib(model_name=actual_model_to_load, version='latest')
        if not load_success: return dbc.Alert(f"Error: Could not load the selected model: {actual_model_to_load}. Check logs.", color="danger")
        interactive_engine_globals['loaded_model_name_cache'] = actual_model_to_load
        # ModelManager's metadata should be populated by load_model_joblib
        loaded_type = model_manager.metadata.get('training_metadata', {}).get('model_type', model_manager.metadata.get('model_type', actual_model_to_load))
        loaded_version = model_manager.metadata.get('version', 'N/A')
        print(f"[Interactive Predict] Successfully loaded model: {loaded_type} (Version: {loaded_version})")
    else:
        loaded_type = model_manager.metadata.get('training_metadata', {}).get('model_type', model_manager.metadata.get('model_type', interactive_engine_globals['loaded_model_name_cache']))
        print(f"[Interactive Predict] Using cached model: {loaded_type}")

    # --- Feature Engineering (Manual Approach for Single Prediction) --- 
    try: 
        # 1. Get Input Values Safely
        raw_company_name = company_name if company_name else f"InteractivePredict_{uuid.uuid4().hex[:8]}"
        funding_amount_val = float(funding_amount) if funding_amount is not None else np.nan
        employees_val = int(employees) if employees is not None else np.nan
        industry_val = industry if industry else 'Unknown'
        months_since_first_val = int(months_since_first) if months_since_first is not None else np.nan
        previous_rounds_val = int(previous_rounds) if previous_rounds is not None else 0 # Default prev_rounds to 0
        funding_date_val = pd.Timestamp.now() # Use current date as proxy

        # Check if essential numeric inputs are missing
        if pd.isna(funding_amount_val) or pd.isna(employees_val) or pd.isna(months_since_first_val):
             print("[Interactive Predict] Warning: Essential numeric inputs (Funding, Employees, Months) have missing values.")
             # Allow proceeding, NaNs will be handled, but prediction quality might suffer

        # 2. Initialize Feature DataFrame using model's expected feature names
        if not model_manager.feature_names:
            return dbc.Alert("Error: Feature names not loaded with the model.", color="danger")
        features_dict = {feature_name: 0.0 for feature_name in model_manager.feature_names} # Default all to 0.0 float

        # 3. Calculate Core Numeric Features (mimicking FeatureEngineering.extract_features)
        # Use np.log1p for stability with potential 0 funding
        features_dict['funding_amount_log'] = np.log1p(funding_amount_val) if pd.notna(funding_amount_val) else 0.0 
        # Keep raw employees if model expects it (check model_manager.feature_names)
        # if 'employees' in features_dict: # Model might use binned employees instead 
        #     features_dict['employees'] = float(employees_val) if pd.notna(employees_val) else 0.0 
        # Keep raw months_since if model expects it
        # if 'months_since_first_funding' in features_dict:
        #     features_dict['months_since_first_funding'] = float(months_since_first_val) if pd.notna(months_since_first_val) else 0.0
        if 'previous_rounds' in features_dict:
            features_dict['previous_rounds'] = float(previous_rounds_val) # Already defaulted to 0
        
        # Derived features (handle division by zero/NaN)
        emp_calc = float(employees_val) if pd.notna(employees_val) else 0.0
        months_calc = float(months_since_first_val) if pd.notna(months_since_first_val) else 0.0
        funding_calc = float(funding_amount_val) if pd.notna(funding_amount_val) else 0.0
        prev_rounds_calc = float(previous_rounds_val) # Already defaulted

        if 'employee_efficiency' in features_dict:
            features_dict['employee_efficiency'] = (funding_calc / max(emp_calc, 1.0)) if emp_calc > 0 else 0.0
        # Funding velocity might not be directly calculable without history
        if 'funding_velocity' in features_dict: features_dict['funding_velocity'] = 0.0 # Default for interactive
        if 'time_since_last_funding' in features_dict: features_dict['time_since_last_funding'] = 0.0 # Default
        if 'funding_amount_ratio_vs_prev' in features_dict: features_dict['funding_amount_ratio_vs_prev'] = 1.0 # Default
        if 'funding_vs_industry_median' in features_dict: features_dict['funding_vs_industry_median'] = 1.0 # Default

        # Temporal features
        if 'funding_year' in features_dict: features_dict['funding_year'] = float(funding_date_val.year)
        if 'funding_month' in features_dict: features_dict['funding_month'] = float(funding_date_val.month)
        if 'month_sin' in features_dict or 'month_cos' in features_dict:
            month_angle = 2. * np.pi * (funding_date_val.month - 1) / 12
            if 'month_sin' in features_dict: features_dict['month_sin'] = np.sin(month_angle)
            if 'month_cos' in features_dict: features_dict['month_cos'] = np.cos(month_angle)

        # Interaction features
        fa_log_calc = features_dict.get('funding_amount_log', 0.0)
        vel_calc = features_dict.get('funding_velocity', 0.0)

        if 'funding_amount_x_age' in features_dict: features_dict['funding_amount_x_age'] = fa_log_calc * months_calc
        if 'employees_x_rounds' in features_dict: features_dict['employees_x_rounds'] = emp_calc * prev_rounds_calc
        if 'velocity_x_rounds' in features_dict: features_dict['velocity_x_rounds'] = vel_calc * prev_rounds_calc
        if 'age_x_employees' in features_dict: features_dict['age_x_employees'] = months_calc * emp_calc

        # 4. Handle Categorical Features (One-Hot Encoding based on model's features)
        # Industry
        industry_prefix = 'industry_category_'
        final_industry = industry_val if industry_val else 'Unknown'
        # --- Apply Consolidation Mapping (Simplified - ideally load mapping) ---
        industry_mapping_simple = { # From FeatureEngineering class
            'artificial intelligence': 'AI & ML', 'machine learning': 'AI & ML', 'information technology': 'IT & Software',
            'software': 'IT & Software', 'health': 'Healthcare', 'healthcare': 'Healthcare', 'biotech': 'Biotech',
            'biotechnology': 'Biotech', 'financial': 'FinTech', 'finance': 'FinTech', 'fintech': 'FinTech',
            'education': 'EdTech', 'edtech': 'EdTech', 'retail': 'Retail', 'ecommerce': 'Retail', 'energy': 'Energy',
            'renewable': 'Energy', 'food': 'Food & Agriculture', 'agriculture': 'Food & Agriculture',
            'transportation': 'Transport & Logistics', 'logistics': 'Transport & Logistics',
            'real estate': 'Real Estate', 'proptech': 'Real Estate'
        }
        mapped_industry = industry_mapping_simple.get(final_industry.lower(), final_industry.title())
        # --- Apply Rare Category Consolidation (Simplified) ---
        # Assume if mapped_industry isn't directly an expected column prefix, map to Other or Unknown
        target_industry_col = industry_prefix + mapped_industry
        if target_industry_col in features_dict:
            features_dict[target_industry_col] = 1.0
        else:
            other_col = industry_prefix + 'Other'
            unknown_col = industry_prefix + 'Unknown'
            if other_col in features_dict: 
                features_dict[other_col] = 1.0
                print(f"[Interactive Predict] Mapped industry '{industry_val}' -> '{mapped_industry}' to 'Other'.")
            elif unknown_col in features_dict:
                features_dict[unknown_col] = 1.0
                print(f"[Interactive Predict] Mapped industry '{industry_val}' -> '{mapped_industry}' to 'Unknown'.")
            else:
                print(f"[Interactive Predict] Warning: Cannot map industry '{industry_val}' to any known column ({target_industry_col}, {other_col}, {unknown_col}).")

        # Company Age Bin
        age_bin_prefix = 'company_age_bin_'
        age_edges = interactive_engine_globals.get('age_bin_edges_from_summary')
        age_labels = interactive_engine_globals.get('age_bin_labels_from_summary')
        if age_edges and age_labels and pd.notna(months_calc):
            try:
                 # Using include_lowest=True to handle edge cases, ensure bins cover the domain
                 cut_result = pd.cut([months_calc], bins=age_edges, labels=age_labels, right=True, include_lowest=True)
                 age_bin_label = cut_result[0]
                 target_age_col = age_bin_prefix + str(age_bin_label) # Ensure label is string
                 if target_age_col in features_dict:
                     features_dict[target_age_col] = 1.0
                 else:
                      print(f"[Interactive Predict] Warning: Calculated age bin label '{age_bin_label}' not found in model columns.")
                      # Try setting Unknown_Age if available
                      unknown_age_col = age_bin_prefix + 'Unknown_Age'
                      if unknown_age_col in features_dict: features_dict[unknown_age_col] = 1.0
            except Exception as age_bin_err:
                 print(f"[Interactive Predict] Error binning age {months_calc}: {age_bin_err}. Attempting Unknown.")
                 unknown_age_col = age_bin_prefix + 'Unknown_Age'
                 if unknown_age_col in features_dict: features_dict[unknown_age_col] = 1.0
        else:
             print("[Interactive Predict] Age bin edges/labels not loaded or months_calc is NaN. Attempting Unknown age bin.")
             unknown_age_col = age_bin_prefix + 'Unknown_Age'
             if unknown_age_col in features_dict: features_dict[unknown_age_col] = 1.0

        # Employee Bin
        emp_bin_prefix = 'employees_bin_'
        emp_bins = [-np.inf, 10, 50, 200, 1000, np.inf] # Fixed bins from FE
        emp_labels = ['1-10', '11-50', '51-200', '201-1000', '1001+']
        if pd.notna(emp_calc):
            try:
                 emp_cut_result = pd.cut([emp_calc], bins=emp_bins, labels=emp_labels, right=True, include_lowest=True)
                 emp_bin_label = emp_cut_result[0]
                 target_emp_col = emp_bin_prefix + str(emp_bin_label)
                 if target_emp_col in features_dict:
                      features_dict[target_emp_col] = 1.0
                 else:
                      print(f"[Interactive Predict] Warning: Calculated employee bin label '{emp_bin_label}' not found in model columns.")
                      unknown_emp_col = emp_bin_prefix + 'Unknown_Emp'
                      if unknown_emp_col in features_dict: features_dict[unknown_emp_col] = 1.0
            except Exception as emp_bin_err:
                 print(f"[Interactive Predict] Error binning employees {emp_calc}: {emp_bin_err}. Attempting Unknown.")
                 unknown_emp_col = emp_bin_prefix + 'Unknown_Emp'
                 if unknown_emp_col in features_dict: features_dict[unknown_emp_col] = 1.0
        else:
             print("[Interactive Predict] Employee count is NaN. Attempting Unknown employee bin.")
             unknown_emp_col = emp_bin_prefix + 'Unknown_Emp'
             if unknown_emp_col in features_dict: features_dict[unknown_emp_col] = 1.0
        
        # 5. Convert final dict to DataFrame in correct order
        X_final_features = pd.DataFrame([features_dict], columns=model_manager.feature_names)
        
        # Final check for NaNs introduced during calculation (should be minimal with defaults)
        if X_final_features.isnull().any().any():
            print(f"[Interactive Predict] Warning: NaNs found in final feature vector before prediction. Filling with 0. Columns: {X_final_features.columns[X_final_features.isnull().any()].tolist()}")
            X_final_features.fillna(0.0, inplace=True)

        print(f"[Interactive Predict] Manual Feature Engineering complete. Final feature count: {len(X_final_features.columns)}")
        # print(f"[Interactive Predict] Data for prediction: {X_final_features.iloc[0].to_dict()}") # DEBUG: Print full vector if needed

    except Exception as fe_manual_error:
        print(f"[Interactive Predict] Error during manual feature engineering: {fe_manual_error}")
        import traceback
        print(traceback.format_exc())
        return dbc.Alert(f"Feature Engineering Error: {str(fe_manual_error)}. Check console for more details.", color="danger")

    if X_final_features.empty:
        return dbc.Alert("Error: Could not prepare features for prediction.", color="danger")

    # --- Prediction using ModelManager --- 
    prediction_input_dict = X_final_features.iloc[0].to_dict()
    print(f"[Interactive Predict] Calling model_manager.predict_proba ...")
    prediction_result = model_manager.predict_proba(features=prediction_input_dict, company_name=raw_company_name)

    # --- Process and Display Results --- 
    if 'error' in prediction_result:
        print(f"[Interactive Predict] Prediction error from model_manager: {prediction_result['error']}")
        return dbc.Alert(f"Prediction Error: {prediction_result['error']}", color="danger")
    
    try:
        class_mapping_for_output = model_manager.metadata.get('class_mapping', interactive_engine_globals['class_mapping_from_summary'])
        if not isinstance(class_mapping_for_output, dict) or not class_mapping_for_output:
             return dbc.Alert("Error: Class mapping invalid or missing.", color="danger")
        class_mapping_for_output = {int(k): str(v) for k, v in class_mapping_for_output.items()} # Ensure int keys
        
        best_class_str_idx = None; max_prob = -1.0; processed_probs = {}
        for str_idx, prob_val in prediction_result.get('probabilities', {}).items():
            try: int_idx = int(str_idx); label = class_mapping_for_output.get(int_idx, f"Unknown_Idx_{int_idx}"); processed_probs[label] = float(prob_val)
            except ValueError: continue
            if float(prob_val) > max_prob: max_prob = float(prob_val); best_class_str_idx = str_idx 
        
        predicted_stage_label = "Mapping Error"; predicted_stage_numeric_final = "N/A"
        if best_class_str_idx is not None:
            try: numeric_prediction_int = int(best_class_str_idx); predicted_stage_label = class_mapping_for_output.get(numeric_prediction_int, f"Label_{numeric_prediction_int}_NF"); predicted_stage_numeric_final = numeric_prediction_int
            except ValueError: predicted_stage_label = f"MapIdxErr_{best_class_str_idx}"
        confidence_val = max_prob * 100 if max_prob != -1.0 else 0.0
        
        # --- Build Display --- 
        result_display_elements = []
        # Prediction Alert
        result_display_elements.append(dbc.Alert([
            html.H4(f"Predicted Funding Stage: {predicted_stage_label} (Idx: {predicted_stage_numeric_final})"), # Simplified label
            html.P(f"Confidence: {confidence_val:.2f}%")
        ], color="success"))
        # Validation Alert
        if 'validation' in prediction_result:
            val_info = prediction_result['validation']; anomaly_color = "warning" if val_info.get('is_anomaly') else "info"
            result_display_elements.append(dbc.Alert([
                html.H5("Validation & Anomaly Check:"),
                html.P(f"Valid: {str(prediction_result.get('is_valid', True))}, Anomaly: {str(val_info.get('is_anomaly', False))} (Score: {val_info.get('anomaly_score', 0.0):.3f})"),
                html.P(f"Reasons: {'; '.join(val_info.get('reasons', [])) if val_info.get('reasons') else 'None'} | Company Valid: {str(val_info.get('company_valid', 'N/A'))}")
            ], color=anomaly_color, className="mt-3"))
        # Probabilities List
        prob_items = [html.Li(f"{lbl}: {prb:.3f}") for lbl, prb in sorted(processed_probs.items(), key=lambda item: item[1], reverse=True)]
        result_display_elements.append(html.Div([html.H5("Probabilities:", className="mt-3"), html.Ul(prob_items)]))
        # Model Info
        current_model_meta = model_manager.metadata
        model_type_disp = current_model_meta.get('training_metadata', {}).get('model_type', current_model_meta.get('model_type', interactive_engine_globals['loaded_model_name_cache']))
        model_version_disp = current_model_meta.get('version', 'N/A'); model_acc_disp = current_model_meta.get('training_metadata', {}).get('accuracy', 'N/A')
        if isinstance(model_acc_disp, (float, int)): model_acc_disp = f"{model_acc_disp:.3f}"
        result_display_elements.append(html.Div([
            html.H5("Model Used:", className="mt-3"),
            html.P(f"Type: {model_type_disp}, Ver: {model_version_disp}, Trained Acc: {model_acc_disp}") # Compact display
        ], className="small text-muted"))

        print("[Interactive Predict] Successfully processed prediction results.")
        return html.Div(result_display_elements)

    except Exception as display_err:
        print(f"[Interactive Predict] Error processing/displaying prediction result: {display_err}")
        import traceback; print(traceback.format_exc())
        return dbc.Alert(f"Error Displaying Results: {str(display_err)}. Check console.", color="danger")


# App layout with routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Routing callback
@callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/':
        return home_layout
    elif pathname == '/objective':
        return objective_layout
    elif pathname == '/methods':
        return methods_layout
    elif pathname == '/findings':
        return findings_layout
    elif pathname == '/interactive-prediction':
        return interactive_prediction_layout
    else:
        return dbc.Container([
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised...")
        ])

if __name__ == '__main__':
    app.run(debug=True)