import os
import sys

# --- Force sys.path setup early using os.getcwd() as App Engine root --- 
# This assumes Gunicorn's CWD is the app root (where app.yaml is)
APP_ENGINE_PROJECT_ROOT = os.getcwd()
print(f"DEBUG: app163.py started. APP_ENGINE_PROJECT_ROOT (from getcwd()): {APP_ENGINE_PROJECT_ROOT}") # Moved initial print here

# Paths to add, in order of preference for imports
paths_to_add = [
    os.path.join(APP_ENGINE_PROJECT_ROOT, 'appengine', 'ML'),
    os.path.join(APP_ENGINE_PROJECT_ROOT, 'appengine'),
    APP_ENGINE_PROJECT_ROOT,
    os.path.join(APP_ENGINE_PROJECT_ROOT, 'backend')
]

for p in reversed(paths_to_add): # Insert at the beginning, so reverse iterate
    if p not in sys.path:
        sys.path.insert(0, p)
        print(f"DEBUG: Early added to sys.path: {p}")
print(f"DEBUG: sys.path after early setup: {sys.path}")
# --- End of forced sys.path setup ---

try:
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

    # --- GCS Configuration ---
    BUCKET_NAME = os.environ.get('BUCKET_NAME')
    if not BUCKET_NAME:
        print("WARNING: BUCKET_NAME environment variable not set. Attempting to use default: 'staging.oval-sunset-450610-h4.appspot.com'")
        BUCKET_NAME = 'staging.oval-sunset-450610-h4.appspot.com' # Default bucket name
        # raise RuntimeError("CRITICAL: BUCKET_NAME environment variable not set. GCS operations will fail. Please set BUCKET_NAME before running the app.")

    # --- Calculate project root and add to sys.path --- 
    # THIS SECTION WILL BE LARGELY REPLACED/SIMPLIFIED OR REMOVED
    # We now use APP_ENGINE_PROJECT_ROOT established earlier.
    PROJECT_ROOT_APP163 = APP_ENGINE_PROJECT_ROOT
    SCRIPT_DIR_APP163 = os.path.join(PROJECT_ROOT_APP163, 'appengine') # Standard assumption
    ML_DIR_FOR_PICKLE = os.path.join(SCRIPT_DIR_APP163, 'ML')
    BACKEND_DIR_APP163 = os.path.join(PROJECT_ROOT_APP163, 'backend')
    FRONTEND_DIR_APP163 = os.path.join(PROJECT_ROOT_APP163, 'frontend')

    print(f"DEBUG: PROJECT_ROOT_APP163 (post early setup): {PROJECT_ROOT_APP163}")
    print(f"DEBUG: SCRIPT_DIR_APP163 (post early setup): {SCRIPT_DIR_APP163}")

    TEMP_ASSET_DIR_NAME = "gcs_assets"
    TEMP_ASSET_ROOT = os.path.join('/tmp', TEMP_ASSET_DIR_NAME)
    TEMP_MODEL_DIR = os.path.join('/tmp', 'gcs_models')
    LOCAL_FALLBACK_MODEL_DIR = os.path.join(PROJECT_ROOT_APP163, 'backend', 'MainOutput', 'models')

    # --- Path modification for allowing pickle to find original module definitions ---
    # This is for models saved when funding_stage_predictionORIGINAL.py was treated as a top-level module.
    # ML_DIR_FOR_PICKLE = os.path.join(SCRIPT_DIR_APP163, 'ML') # Already defined
    # if ML_DIR_FOR_PICKLE not in sys.path:
    #     sys.path.insert(0, ML_DIR_FOR_PICKLE)
    #     print(f"DEBUG: Added to sys.path for pickle: {ML_DIR_FOR_PICKLE}")

    # Add appengine directory to sys.path for relative imports like 'from ML...' used in *this* script
    # if SCRIPT_DIR_APP163 not in sys.path:
        # insert_idx_appengine = 1 if ML_DIR_FOR_PICKLE in sys.path and sys.path.index(ML_DIR_FOR_PICKLE) == 0 else 0
        # sys.path.insert(insert_idx_appengine, SCRIPT_DIR_APP163)
        # print(f"DEBUG: Added to sys.path for app163 'from ML import ...' style imports: {SCRIPT_DIR_APP163} at index {insert_idx_appengine}")

    # Add project root for other potential imports
    # if PROJECT_ROOT_APP163 not in sys.path:
        # insert_idx_project_root = 0
        # Determine safe insertion index after previously added paths
        # if SCRIPT_DIR_APP163 in sys.path:
            # insert_idx_project_root = sys.path.index(SCRIPT_DIR_APP163) + 1
        # elif ML_DIR_FOR_PICKLE in sys.path:
            # insert_idx_project_root = sys.path.index(ML_DIR_FOR_PICKLE) + 1
        # Ensure index is within current bounds of sys.path
        # insert_idx_project_root = min(insert_idx_project_root, len(sys.path))
        # sys.path.insert(insert_idx_project_root, PROJECT_ROOT_APP163)
        # print(f"DEBUG: Added to sys.path for project root: {PROJECT_ROOT_APP163} at index {insert_idx_project_root}")
    
    # Subsequent paths should be inserted after the primary ones already established
    # Find the highest index of the paths we've definitely managed so far
    # managed_paths = [ML_DIR_FOR_PICKLE, SCRIPT_DIR_APP163, PROJECT_ROOT_APP163]
    # current_max_managed_idx = -1
    # for p in managed_paths:
        # if p in sys.path:
            # current_max_managed_idx = max(current_max_managed_idx, sys.path.index(p))
    
    # insert_idx_for_others = current_max_managed_idx + 1

    # BACKEND_DIR_APP163 = os.path.join(PROJECT_ROOT_APP163, 'backend')
    # if BACKEND_DIR_APP163 not in sys.path:
        # sys.path.insert(insert_idx_for_others, BACKEND_DIR_APP163)
        # print(f"DEBUG: Added to sys.path for backend: {BACKEND_DIR_APP163} at index {insert_idx_for_others}")
        # insert_idx_for_others += 1 # Increment for next potential insertion
        
    # FRONTEND_DIR_APP163 = os.path.join(PROJECT_ROOT_APP163, 'frontend')
    # if FRONTEND_DIR_APP163 not in sys.path:
        # sys.path.insert(insert_idx_for_others, FRONTEND_DIR_APP163)
        # print(f"DEBUG: Added to sys.path for frontend: {FRONTEND_DIR_APP163} at index {insert_idx_for_others}")

    # print(f"DEBUG: Final sys.path: {sys.path}") # This can be removed or kept from early setup
    # --- End of sys.path modifications --- (Old section largely commented out/replaced)

    print("DEBUG: Attempting to import ML modules. Current sys.path (after early setup):", sys.path) # Modified for debugging
    try:
        # Let's try importing one by one to isolate
        print("DEBUG: Attempting to import FeatureEngineering...")
        from ML.funding_stage_predictionORIGINAL import FeatureEngineering
        print("DEBUG: Successfully imported FeatureEngineering.")

        print("DEBUG: Attempting to import ModelManager...")
        from ML.funding_stage_predictionORIGINAL import ModelManager
        print("DEBUG: Successfully imported ModelManager.")

        print("DEBUG: Attempting to import AnomalyDetector...")
        from ML.funding_stage_predictionORIGINAL import AnomalyDetector
        print("DEBUG: Successfully imported AnomalyDetector.")

        print("DEBUG: Attempting to import NumpyEncoder...")
        from ML.funding_stage_predictionORIGINAL import NumpyEncoder
        print("DEBUG: Successfully imported NumpyEncoder.")

        print("Successfully imported all custom ML modules in app163.py")

    except ImportError as e:
        import traceback
        print(f"Error importing custom ML modules in app163.py: {e}. Prediction engine will not work.")
        print(f"DETAILED IMPORT ERROR for ML modules: {e}")
        print("FULL TRACEBACK FOR IMPORT ERROR:")
        traceback.print_exc()
        FeatureEngineering = None
        ModelManager = None
        AnomalyDetector = None
        NumpyEncoder = None
    except NameError:
        print("__file__ not defined in app163.py, sys.path modification might not be effective.")
        FeatureEngineering = None
        ModelManager = None
        AnomalyDetector = None
        NumpyEncoder = None

    ASSET_TARGET_DIR = TEMP_ASSET_ROOT
    GCS_VISUALIZATIONS_PREFIX = 'MainOutput/visualizations/'
    GCS_PROTOTYPE_PREFIX = 'MainOutput/prototype_dashboard/'
    GCS_MODELS_PREFIX = 'MainOutput/models/'
    GCS_SUMMARIES_PREFIX = 'MainOutput/'

    storage_client = None
    try:
        storage_client = storage.Client()
    except Exception as e:
        print(f"Failed to initialize Google Cloud Storage client: {e}")

    def find_latest_gcs_blob(bucket_name, gcs_prefix, file_pattern):
        """Finds the latest blob in GCS matching a prefix and pattern."""
        if not storage_client or not bucket_name:
            print(f"GCS: Storage client or bucket name not available for find_latest_gcs_blob (prefix: {gcs_prefix}).")
            return None
        try:
            bucket = storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=gcs_prefix))
            
            matching_blobs = []
            for blob in blobs:
                file_name_part = blob.name[len(gcs_prefix):] if blob.name.startswith(gcs_prefix) else blob.name
                # Simple pattern matching for wildcards
                if '*' in file_pattern:
                    base_pattern = file_pattern.split('*')[0]
                    # Ensure pattern is not empty and filename part starts with base_pattern
                    # and also consider the part after '*' if it exists.
                    parts = file_pattern.split('*')
                    if len(parts) == 1: # e.g. "pattern*"
                        if file_name_part.startswith(base_pattern):
                            matching_blobs.append(blob)
                    elif len(parts) == 2: # e.g. "pattern1*pattern2"
                        if file_name_part.startswith(parts[0]) and file_name_part.endswith(parts[1]):
                            matching_blobs.append(blob)
                    # Add more sophisticated glob handling if needed for more complex patterns
                elif file_pattern == file_name_part: # Exact match
                     matching_blobs.append(blob)
                elif file_pattern in file_name_part: # Substring match as a fallback (use sparingly)
                     matching_blobs.append(blob)

            if not matching_blobs:
                print(f"[GCS Detailed] No blobs found. Bucket: '{bucket_name}', Prefix: '{gcs_prefix}', Pattern: '{file_pattern}', All Blobs listed (up to 50): {[b.name for b in blobs[:50]]}") # Added detailed logging
                # print(f"[GCS] No blobs found in bucket '{bucket_name}' with prefix '{gcs_prefix}' matching pattern '{file_pattern}'")
                return None
            
            # Sort by time_created or updated attribute to find the latest
            latest_blob = max(matching_blobs, key=lambda b: b.updated if b.updated else b.time_created)
            # print(f"[GCS] Found latest for {file_pattern} in {gcs_prefix}: {latest_blob.name}")
            return latest_blob
        except Exception as e:
            print(f"GCS: Error finding latest blob for {file_pattern} in {gcs_prefix} (bucket: {bucket_name}): {e}")
            return None

    def download_blob_to_temp(bucket_name, blob_name, temp_destination_path):
        """Downloads a blob from GCS to a temporary local path."""
        if not storage_client or not bucket_name:
            print(f"GCS: Storage client or bucket name not available for download_blob_to_temp (blob: {blob_name}).")
            return False
        try:
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            os.makedirs(os.path.dirname(temp_destination_path), exist_ok=True)
            blob.download_to_filename(temp_destination_path)
            print(f"GCS: Downloaded {blob_name} to {temp_destination_path}")
            return True
        except Exception as e:
            print(f"GCS: Failed to download {blob_name} to {temp_destination_path}: {e}")
            print(f"DETAILED GCS Download Error: {e}, Bucket: {bucket_name}, Blob: {blob_name}") # Added detailed logging
            return False

    def sync_latest_assets_from_gcs():
        """Copies the latest required plot files from GCS backend output to app temporary assets.
           Also ensures essential local fallback assets are present in the temp assets folder.
        """
        if not BUCKET_NAME:
            print("GCS Assets: BUCKET_NAME not set. Cannot sync assets from GCS.")
            return

        print("GCS Assets: Starting asset synchronization...")
        os.makedirs(TEMP_ASSET_ROOT, exist_ok=True)

        assets_to_sync_from_gcs = [
            (GCS_PROTOTYPE_PREFIX, 'bay_area_funding_trend_interactive_*.html', 'Forecast HTML'),
            (GCS_VISUALIZATIONS_PREFIX, 'calibration_plot_Random_Forest_Tuned_*.png', 'RF Calibration Plot'),
            (GCS_VISUALIZATIONS_PREFIX, 'model_comparison_accuracy_rmse_*.png', 'Model Comparison Plot'),
            (GCS_VISUALIZATIONS_PREFIX, 'funding_stage_dist_*.png', 'Funding Dist Plot'),
            (GCS_VISUALIZATIONS_PREFIX, 'feature_importance_RandomForestClassifier_*.png', 'RF Feature Importance Plot'),
        ]

        copied_files_count = 0
        for gcs_prefix, file_pattern, name in assets_to_sync_from_gcs:
            latest_blob = find_latest_gcs_blob(BUCKET_NAME, gcs_prefix, file_pattern)
            if latest_blob:
                # The destination filename in /tmp/gcs_assets will be the same as the blob's filename part
                dest_filename_in_temp = os.path.basename(latest_blob.name)
                temp_dest_path = os.path.join(TEMP_ASSET_ROOT, dest_filename_in_temp)
                
                needs_download = True
                if os.path.exists(temp_dest_path) and latest_blob.updated:
                    if latest_blob.updated.timestamp() <= os.path.getmtime(temp_dest_path):
                         needs_download = False
                
                if needs_download:
                     if download_blob_to_temp(BUCKET_NAME, latest_blob.name, temp_dest_path):
                        print(f"GCS Assets: Copied/Updated {name}: {dest_filename_in_temp}")
                        copied_files_count += 1
                # else:
                    # print(f"GCS Assets: {name} already up-to-date in temp assets.")
            else:
                print(f"GCS Assets: Warning - Latest source blob for {name} (pattern: {file_pattern}) not found in GCS bucket '{BUCKET_NAME}' prefix '{gcs_prefix}'.")

        print(f"GCS Assets: Synchronization finished. Copied/Updated {copied_files_count} files to {TEMP_ASSET_ROOT}.")

        # --- Ensure essential local fallback/static assets are in TEMP_ASSET_ROOT for Dash --- 
        print(f"DEBUG: Ensuring essential local assets are in {TEMP_ASSET_ROOT}")
        essential_local_assets = ['violin_funding_by_stage_20250512_022535.png', 'placeholder.png'] 
        # Define where these local assets are stored within the app structure
        local_assets_source_dir = os.path.join(SCRIPT_DIR_APP163, 'assets')
        copied_local_essentials = 0
        for asset_filename in essential_local_assets:
            source_path = os.path.join(local_assets_source_dir, asset_filename)
            temp_dest_path = os.path.join(TEMP_ASSET_ROOT, asset_filename)
            if not os.path.exists(temp_dest_path) and os.path.exists(source_path):
                try:
                    shutil.copy(source_path, temp_dest_path)
                    print(f"DEBUG: Copied essential local asset '{asset_filename}' to {TEMP_ASSET_ROOT}")
                    copied_local_essentials += 1
                except Exception as e:
                    print(f"DEBUG: Failed to copy essential local asset '{asset_filename}': {e}")
            elif os.path.exists(temp_dest_path):
                print(f"DEBUG: Essential asset '{asset_filename}' already in {TEMP_ASSET_ROOT} (possibly from GCS sync or previous copy).")
            elif not os.path.exists(source_path):
                print(f"DEBUG: Essential local asset '{asset_filename}' not found at source '{source_path}'.")
        print(f"DEBUG: Copied {copied_local_essentials} essential local assets to {TEMP_ASSET_ROOT}.")

    def sync_models_from_gcs():
        """Downloads all model files (.joblib and _metadata.json) from GCS to a temporary local directory."""
        if not BUCKET_NAME or not storage_client:
            print("GCS Models: BUCKET_NAME or storage_client not available. Cannot sync models.")
            return

        print(f"GCS Models: Starting model synchronization (bucket: {BUCKET_NAME}, prefix: {GCS_MODELS_PREFIX}) to {TEMP_MODEL_DIR}...")
        os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

        try:
            bucket = storage_client.bucket(BUCKET_NAME)
            blobs = list(bucket.list_blobs(prefix=GCS_MODELS_PREFIX))
            
            downloaded_model_files_count = 0
            if not blobs:
                print(f"GCS Models: No blobs found in bucket '{BUCKET_NAME}' with prefix '{GCS_MODELS_PREFIX}'.")
                return

            for blob in blobs:
                if blob.name.endswith('/') or not (blob.name.endswith('.joblib') or blob.name.endswith('_metadata.json')):
                    # Skip "folders" or files not matching model extensions
                    continue

                dest_filename_in_temp = os.path.basename(blob.name)
                temp_dest_path = os.path.join(TEMP_MODEL_DIR, dest_filename_in_temp)

                # Download if it doesn't exist locally or if GCS version is newer
                needs_download = True
                if os.path.exists(temp_dest_path):
                    local_mtime = os.path.getmtime(temp_dest_path)
                    gcs_mtime = blob.updated.timestamp() if blob.updated else (blob.time_created.timestamp() if blob.time_created else 0)
                    if local_mtime >= gcs_mtime:
                        needs_download = False
                        # print(f"[GCS Models] Model file {dest_filename_in_temp} already up-to-date in {TEMP_MODEL_DIR}.")

                if needs_download:
                    if download_blob_to_temp(BUCKET_NAME, blob.name, temp_dest_path):
                        print(f"GCS Models: Downloaded/Updated model file: {dest_filename_in_temp}")
                        downloaded_model_files_count += 1
            
            print(f"GCS Models: Synchronization finished. Downloaded/Updated {downloaded_model_files_count} model files to {TEMP_MODEL_DIR}.")

        except Exception as e:
            print(f"GCS Models: Error during model synchronization: {e}")

    def get_asset_url_path(pattern_prefix):
        """Finds the latest asset in the temporary assets folder and returns its Dash asset URL path."""
        if not os.path.exists(TEMP_ASSET_ROOT):
            print(f"Assets: Temporary asset directory {TEMP_ASSET_ROOT} does not exist.")
            return app.get_asset_url('violin_funding_by_stage_20250512_022535.png') 

        try:
            # Search for files starting with pattern_prefix in TEMP_ASSET_ROOT
            matching_files = []
            for f_name in os.listdir(TEMP_ASSET_ROOT):
                if f_name.startswith(pattern_prefix):
                    matching_files.append(os.path.join(TEMP_ASSET_ROOT, f_name))

            if not matching_files:
                # print(f"[Assets] No files found in {TEMP_ASSET_ROOT} for prefix: {pattern_prefix}")
                return app.get_asset_url('violin_funding_by_stage_20250512_022535.png') 
            
            latest_file_local = max(matching_files, key=os.path.getmtime)
            asset_filename = os.path.basename(latest_file_local)
            # print(f"[Assets] Using asset from temp: {asset_filename} for prefix {pattern_prefix}")
            return app.get_asset_url(asset_filename) # Dash serves this from the `assets_folder`
        except Exception as e:
            print(f"[Assets] Error in get_asset_url_path for {pattern_prefix}: {e}")
            return app.get_asset_url('violin_funding_by_stage_20250512_022535.png')


    # --- Run Asset Sync on Startup ---
    # We will call sync_latest_assets_from_gcs() after app initialization

    # Initialize Google Cloud Storage client
    # Client initialization moved to the top GCS Configuration section

    def get_gcs_data(bucket_name_param, blob_name): # bucket_name_param to avoid conflict with global BUCKET_NAME
        """Downloads CSV data from GCS and returns a pandas DataFrame."""
        if not storage_client or not bucket_name_param:
            print(f"GCS Data: Storage client or bucket name not available for get_gcs_data (blob: {blob_name}).")
            return pd.DataFrame() # Return empty DataFrame on error
        try:
            bucket = storage_client.bucket(bucket_name_param)
            blob = bucket.blob(blob_name)
            data = blob.download_as_text()
            return pd.read_csv(StringIO(data))
        except Exception as e:
            print(f"[GCS Data] Failed to get CSV data for {blob_name} from bucket {bucket_name_param}: {e}")
            return pd.DataFrame()


    # Initialize the Dash app
    # Crucially, set assets_folder to our temporary GCS asset download location
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder=TEMP_ASSET_ROOT)
    app.title = "Startup Analytics Platform"
    server = app.server  # Expose server for Gunicorn (moved to module level)

    # --- Sync GCS assets after app is defined and TEMP_ASSET_ROOT is known by Dash ---
    if storage_client and BUCKET_NAME: # Only attempt if GCS client and bucket name are available
        print("DEBUG: Attempting to sync GCS assets and models...") # Added for debugging
        sync_latest_assets_from_gcs()
        sync_models_from_gcs() # Call to sync models
    else:
        print("Skipping GCS asset/model sync due to missing GCS client or BUCKET_NAME.")


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
                        # src=get_asset_url_path('calibration_plot_Random_Forest_Tuned_'), 
                        src=app.get_asset_url('calibration_plot_Random_Forest_Tuned_20250511_201419.png'), # Hardcoded for testing
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
                        ]),
                        html.Img(src=app.get_asset_url('placeholder.png'), className="img-fluid mt-3")
                    ], className="p-4 border rounded")
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
                        # These are general capabilities. Actual numbers depend on specific pipeline runs.
                        html.Ul([
                            html.Li(f"Dataset typically comprises several hundred to a few thousand unique funding events after merging and cleaning."),
                            html.Li(f"A comprehensive set of ~30-40 engineered features are created, including temporal, categorical, and interaction terms."),
                            html.Li("Funding stages are mapped to a consistent numerical representation for modeling."),
                            html.Li("Models are trained to predict these numerical funding stages.")
                        ]),
                        html.Img(src=app.get_asset_url('violin_funding_by_stage_20250512_022535.png'), className="img-fluid mt-3") # Using specific asset violin_funding_by_stage_20250512_022535.png
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
                        html.Img(src=app.get_asset_url('placeholder.png'), className="img-fluid mb-3"), # Pointing to placeholder as model-architecture.png is not synced from GCS
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
                    
                    # Temporal features like funding_year, month_sin/cos will be derived from current date if needed by FeatureEngineering.
                    
                    dbc.Button("Predict Funding Stage", id="interactive-predict-button", color="primary", className="mt-3 mb-3", n_clicks=0),
                    
                    html.Div(id='interactive-prediction-output', className="mt-4 p-3 border rounded", children="Prediction results will appear here.")
                    
                ], md=8, className="mx-auto") # Center the column
            ])
        ], className="my-4", fluid=True)
    ])

    # Globals for holding loaded model and related objects for the interactive engine
    interactive_engine_globals = {
        'model_manager': None,
        'feature_engineer': None,
        'loaded_model_name_cache': None,
        'class_mapping_from_summary': {},
        'feature_names_from_summary': [],
        'age_bin_edges_from_summary': None,
        'age_bin_labels_from_summary': None
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
                PROJECT_ROOT_APP163 = "." 
                print("Warning: PROJECT_ROOT_APP163 defaulted to '.' in get_ml_tools, this might cause issues if not in App Engine context.")

        if FeatureEngineering and interactive_engine_globals['feature_engineer'] is None:
            interactive_engine_globals['feature_engineer'] = FeatureEngineering()
            print("Initialized FeatureEngineering for interactive engine.")
        
        # Determine paths for ModelManager initialization
        output_dir_interactive_fallback = os.path.join(PROJECT_ROOT_APP163, 'backend', 'MainOutput') # Fallback/reference path
        models_dir_interactive_fallback = os.path.join(output_dir_interactive_fallback, 'models') # Fallback/reference path
        
        target_model_dir_for_manager = TEMP_MODEL_DIR # Models are synced here from GCS
        # Print the directory that will be used for models by ModelManager
        print(f"ML Tools: ModelManager will use model directory: {target_model_dir_for_manager}")

        if ModelManager and interactive_engine_globals['model_manager'] is None:
            # Ensure the target directory exists, though sync_models_from_gcs should create it
            os.makedirs(target_model_dir_for_manager, exist_ok=True)
            
            if not os.listdir(target_model_dir_for_manager): 
                 print(f"ML Tools WARNING: Target model directory {target_model_dir_for_manager} is empty. Model loading will fail if sync from GCS was unsuccessful.")
            
            print(f"ML Tools: Initializing ModelManager with model_dir: {target_model_dir_for_manager}")
            interactive_engine_globals['model_manager'] = ModelManager(model_dir=target_model_dir_for_manager)

        # Load summary data from GCS to get mappings and feature names
        if not interactive_engine_globals['class_mapping_from_summary'] or not interactive_engine_globals['feature_names_from_summary']:
            if storage_client and BUCKET_NAME:
                print(f"Interactive Engine: Loading latest summary from GCS Bucket: {BUCKET_NAME}, Prefix: {GCS_SUMMARIES_PREFIX}")
                latest_summary_blob = find_latest_gcs_blob(BUCKET_NAME, GCS_SUMMARIES_PREFIX, "summary_*.json")
                if latest_summary_blob:
                    summary_filename = os.path.basename(latest_summary_blob.name)
                    # Ensure TEMP_MODEL_DIR exists for downloading summary, or use TEMP_ASSET_ROOT
                    os.makedirs(TEMP_MODEL_DIR, exist_ok=True) 
                    temp_summary_path = os.path.join(TEMP_MODEL_DIR, summary_filename)
                    
                    if download_blob_to_temp(BUCKET_NAME, latest_summary_blob.name, temp_summary_path):
                        try:
                            with open(temp_summary_path, 'r') as f:
                                summary_data = json.load(f)
                            
                            interactive_engine_globals['class_mapping_from_summary'] = {int(k): str(v) for k,v in summary_data.get('class_mapping', {}).items()}
                            interactive_engine_globals['feature_names_from_summary'] = summary_data.get('feature_names', [])
                            interactive_engine_globals['age_bin_edges_from_summary'] = summary_data.get('age_bin_edges')
                            interactive_engine_globals['age_bin_labels_from_summary'] = summary_data.get('age_bin_labels')
                            
                            print(f"Interactive Engine: Loaded class_mapping ({len(interactive_engine_globals['class_mapping_from_summary'])}) and feature_names ({len(interactive_engine_globals['feature_names_from_summary'])}) from GCS summary: {summary_filename}")
                            # print(f"Interactive engine: Age bins from summary - Edges: {interactive_engine_globals['age_bin_edges_from_summary'] is not None}, Labels: {interactive_engine_globals['age_bin_labels_from_summary'] is not None}")
                        except Exception as e:
                            print(f"Interactive Engine: Error processing downloaded summary file {temp_summary_path}: {e}")
                    else:
                        print(f"Interactive Engine: Failed to download latest summary file from GCS: {latest_summary_blob.name}")
                else:
                    print(f"Interactive Engine: No summary file (summary_*.json) found in GCS bucket '{BUCKET_NAME}' with prefix '{GCS_SUMMARIES_PREFIX}'.")
            else:
                print("Interactive Engine: GCS client or BUCKET_NAME not available. Cannot load summary from GCS.")
            
            # Fallback if GCS load failed or was skipped
            if not interactive_engine_globals['class_mapping_from_summary']: 
                print(f"Interactive Engine: class_mapping or feature_names are still empty. Interactive predictions might fail or be inaccurate.")
                interactive_engine_globals['class_mapping_from_summary'] = {} # Ensure they are dict/list
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

        if FeatureEngineering is None or ModelManager is None:
            print("[Interactive Predict] Error: Core ML classes (FeatureEngineering, ModelManager) are not available due to import issues.")
            return dbc.Alert("Error: Core ML components not loaded. Prediction unavailable. Check server logs.", color="danger")

        model_manager, feature_engineer = get_ml_tools()

        if not model_manager:
            print("[Interactive Predict] Error: ModelManager could not be initialized by get_ml_tools().")
            return dbc.Alert("Error: Model Manager tool could not be initialized. Check server logs.", color="danger")
        if not feature_engineer:
            try: 
                 interactive_engine_globals['feature_engineer'] = FeatureEngineering()
                 feature_engineer = interactive_engine_globals['feature_engineer']
                 print("[Interactive Predict] Re-initialized FeatureEngineering in callback.")
            except Exception as fe_init_err:
                 print(f"[Interactive Predict] Error: Failed to initialize FeatureEngineer: {fe_init_err}")
                 return dbc.Alert("Error: Feature Engineering tool could not be initialized. Check server logs.", color="danger")

        global PROJECT_ROOT_APP163
        if 'PROJECT_ROOT_APP163' not in globals() or not PROJECT_ROOT_APP163:
            try:
                SCRIPT_DIR_APP163_CB = os.path.dirname(os.path.abspath(__file__))
                PROJECT_ROOT_APP163 = os.path.dirname(SCRIPT_DIR_APP163_CB)
                print("[Interactive Predict] Re-initialized PROJECT_ROOT_APP163 in callback.")
            except NameError:
                PROJECT_ROOT_APP163 = "." 
                print("Warning: PROJECT_ROOT_APP163 defaulted to '.' in callback.")

        # --- Model Loading Logic ---
        load_success = False
        model_base_name_to_load = selected_model_pattern
        version_to_load = 'latest'
        
        # Helper to parse model name like "BaseName_vYYYYMMDD" or "BaseName_vYYYYMMDD_HHMMSS"
        def _extract_base_and_version(name_str):
            import re
            # Regex to capture base name and version string (e.g., _v20230101 or _v20230101_123045)
            # It tries to find the last occurrence of _v followed by digits and optionally more digits separated by underscore
            match = re.match(r'(.*?)_((v\d{8}(_\d{6})?)|(v\d{8}))$', name_str)
            if match:
                base_name = match.group(1)
                version_str = match.group(2) # This will be like 'v20230101' or 'v20230101_123045'
                if not version_str.startswith('v'): # Ensure 'v' prefix if accidentally captured without it by a broader group
                    version_str = 'v' + version_str
                return base_name, version_str
            return name_str, 'latest' # Default if no version pattern found

        user_preferred_specific_versions = {
            'Dashboard_Model_Stacking_Ensemble_(RF_Meta)': 'v202505110540',
            'Dashboard_Model_RandomForest_Dashboard_AllFeatures': 'v202505110539',
            'XGBoost_Tuned': 'v202505110545'
        }

        if selected_model_pattern == 'BEST_FROM_SUMMARY':
            print(f"[Interactive Predict] 'BEST_FROM_SUMMARY' selected. Resolving from GCS summary.")
            resolved_full_name_from_summary = None
            if storage_client and BUCKET_NAME:
                latest_summary_blob = find_latest_gcs_blob(BUCKET_NAME, GCS_SUMMARIES_PREFIX, "summary_*.json")
                if latest_summary_blob:
                    summary_filename = os.path.basename(latest_summary_blob.name)
                    temp_summary_path = os.path.join(TEMP_MODEL_DIR, summary_filename)
                    if download_blob_to_temp(BUCKET_NAME, latest_summary_blob.name, temp_summary_path):
                        try:
                            with open(temp_summary_path, 'r') as f: summary_data_local = json.load(f)
                            best_model_name_from_summary_raw = summary_data_local.get('best_model_by_accuracy')
                            if not best_model_name_from_summary_raw:
                                return dbc.Alert("Error: Best model name not found in summary.", color="danger")
                            resolved_full_name_from_summary = best_model_name_from_summary_raw.replace(" ", "_").replace("(", "").replace(")", "")
                            print(f"[Interactive Predict] BEST_FROM_SUMMARY resolved to raw name: {resolved_full_name_from_summary}")
                        except Exception as e:
                            return dbc.Alert(f"Error resolving BEST_FROM_SUMMARY from GCS file: {str(e)} (Path: {temp_summary_path})", color="danger")
                    else:
                        return dbc.Alert("Error: Failed to download summary file for BEST_FROM_SUMMARY.", color="danger")
                else:
                    return dbc.Alert("Error: No summary file found in GCS for BEST_FROM_SUMMARY.", color="danger")
            else:
                return dbc.Alert("Error: GCS not available for BEST_FROM_SUMMARY.", color="danger")

            if resolved_full_name_from_summary:
                model_base_name_to_load, version_to_load = _extract_base_and_version(resolved_full_name_from_summary)
                print(f"[Interactive Predict] BEST_FROM_SUMMARY parsed to Base: '{model_base_name_to_load}', Version: '{version_to_load}'")
            else: # Should have returned an error above if resolution failed
                return dbc.Alert("Error: Could not determine model from BEST_FROM_SUMMARY.", color="danger")
        
        # Now, attempt to load the model
        # This block handles the general case and the first attempt for user-preferred versions.
        # Reset model manager state before any load attempt if model changes
        current_model_cache_key = f"{model_base_name_to_load}_{version_to_load}"
        if model_manager.model is None or interactive_engine_globals.get('loaded_model_name_cache') != current_model_cache_key:
            print(f"[Interactive Predict] Attempting to load model from GCS-synced dir: '{model_base_name_to_load}' with version: '{version_to_load}'")
            if hasattr(model_manager, 'model_dir'): print(f"[Interactive Predict] ModelManager primary loading from: {model_manager.model_dir}")
            
            if os.path.exists(TEMP_MODEL_DIR):
                print(f"[Interactive Predict] DEBUG: Contents of {TEMP_MODEL_DIR} before load: {os.listdir(TEMP_MODEL_DIR)}")
            else:
                print(f"[Interactive Predict] DEBUG ERROR: Model directory {TEMP_MODEL_DIR} does not exist.")
            
            if not os.path.exists(TEMP_MODEL_DIR) or not os.listdir(TEMP_MODEL_DIR):
                print(f"[Interactive Predict] ERROR: Model directory {TEMP_MODEL_DIR} is empty/missing. Attempting re-sync.")
                sync_models_from_gcs()
                if os.path.exists(TEMP_MODEL_DIR): print(f"[Interactive Predict] DEBUG: Contents of {TEMP_MODEL_DIR} after re-sync: {os.listdir(TEMP_MODEL_DIR)}")
                if not os.path.exists(TEMP_MODEL_DIR) or not os.listdir(TEMP_MODEL_DIR):
                    return dbc.Alert(f"Error: Model directory {TEMP_MODEL_DIR} empty after re-sync. Cannot load model. Check logs.", color="danger")

            # Clear previous model state from manager before loading new one
            model_manager.model = None
            model_manager.metadata = {} # Important to clear old metadata
            model_manager.feature_names = []
            model_manager.scaler = None
            model_manager.anomaly_detector = None

            load_success = model_manager.load_model_joblib(model_name=model_base_name_to_load, version=version_to_load)
            if load_success:
                interactive_engine_globals['loaded_model_name_cache'] = current_model_cache_key
                print(f"[Interactive Predict] Successfully loaded model from GCS-synced dir: '{model_base_name_to_load}' (Version: '{version_to_load}')")
            else:
                print(f"[Interactive Predict] Failed to load model from GCS-synced dir: '{model_base_name_to_load}' (Version: '{version_to_load}') with first attempt.")
        else: # Model is already cached and matches the target
            load_success = True 
            print(f"[Interactive Predict] Using cached model from GCS-synced dir: {interactive_engine_globals.get('loaded_model_name_cache')}")


        # If the first attempt failed for a user-preferred model, try the specific hardcoded version from GCS-synced dir
        if not load_success and selected_model_pattern in user_preferred_specific_versions and version_to_load == 'latest':
            # This implies the first attempt was with 'latest' for a user-preferred base name.
            specific_version_str = user_preferred_specific_versions[selected_model_pattern]
            print(f"[Interactive Predict] Initial 'latest' load from GCS-synced dir failed for '{selected_model_pattern}'. Trying specific version from GCS-synced dir: '{specific_version_str}'")
            
            current_model_cache_key = f"{selected_model_pattern}_{specific_version_str}" # Update cache key for this attempt
            # Ensure manager is reset for this specific attempt
            model_manager.model = None
            model_manager.metadata = {}
            model_manager.feature_names = []
            model_manager.scaler = None
            model_manager.anomaly_detector = None

            load_success = model_manager.load_model_joblib(model_name=selected_model_pattern, version=specific_version_str)
            if load_success:
                interactive_engine_globals['loaded_model_name_cache'] = current_model_cache_key
                print(f"[Interactive Predict] Successfully loaded model from GCS-synced dir: '{selected_model_pattern}' (Specific Version: '{specific_version_str}')")
            else:
                print(f"[Interactive Predict] Also failed to load model from GCS-synced dir: '{selected_model_pattern}' (Specific Version: '{specific_version_str}')")
        
        # --- Fallback to local directory if GCS attempts failed ---
        if not load_success: 
            print(f"[Interactive Predict] All GCS-synced load attempts failed for '{model_base_name_to_load}' (version: '{version_to_load}').")
            print(f"[Interactive Predict] Now attempting to load from local fallback directory: {LOCAL_FALLBACK_MODEL_DIR}")

            if not os.path.exists(LOCAL_FALLBACK_MODEL_DIR):
                print(f"[Interactive Predict] Fallback model directory {LOCAL_FALLBACK_MODEL_DIR} does not exist. Cannot attempt fallback load.")
            elif not os.listdir(LOCAL_FALLBACK_MODEL_DIR):
                print(f"[Interactive Predict] Fallback model directory {LOCAL_FALLBACK_MODEL_DIR} is empty. Cannot attempt fallback load.")
            else:
                original_model_dir_for_fallback = model_manager.model_dir # Store current (should be TEMP_MODEL_DIR)
                
                print(f"[Interactive Predict] Temporarily setting model_manager.model_dir to {LOCAL_FALLBACK_MODEL_DIR} for fallback attempt.")
                model_manager.model_dir = LOCAL_FALLBACK_MODEL_DIR
                
                # Reset model state in manager before trying to load from fallback
                model_manager.model = None
                model_manager.metadata = {}
                model_manager.feature_names = []
                model_manager.scaler = None
                model_manager.anomaly_detector = None
                
                print(f"[Interactive Predict] DEBUG: Contents of {LOCAL_FALLBACK_MODEL_DIR} before fallback load: {os.listdir(LOCAL_FALLBACK_MODEL_DIR)}")
                
                # Attempt to load the same model_base_name_to_load and version_to_load that failed from GCS
                fallback_load_success_flag = model_manager.load_model_joblib(model_name=model_base_name_to_load, version=version_to_load)
                
                if fallback_load_success_flag:
                    load_success = True # Update the main load_success flag
                    # Update cache key to reflect it's a fallback model. 
                    # The version_to_load here is the one that was attempted from GCS (e.g. 'latest' or specific 'vYYYYMMDD')
                    interactive_engine_globals['loaded_model_name_cache'] = f"{model_base_name_to_load}_{version_to_load}_fallback"
                    print(f"[Interactive Predict] Successfully loaded model from LOCAL FALLBACK: '{model_base_name_to_load}' (Version: '{version_to_load}') from {LOCAL_FALLBACK_MODEL_DIR}")
                else:
                    print(f"[Interactive Predict] Also failed to load model from LOCAL FALLBACK: '{model_base_name_to_load}' (Version: '{version_to_load}') from {LOCAL_FALLBACK_MODEL_DIR}")
                    # load_success remains False

                # IMPORTANT: Restore original model_dir for the model_manager
                model_manager.model_dir = original_model_dir_for_fallback
                print(f"[Interactive Predict] Restored model_manager.model_dir to: {model_manager.model_dir}")

        if not load_success:
             # Construct a more informative error message
            error_msg_detail = f"Could not load model '{selected_model_pattern}'."
            if selected_model_pattern in user_preferred_specific_versions:
                error_msg_detail += f" Tried with 'latest' and specific version '{user_preferred_specific_versions[selected_model_pattern]}'."
            elif selected_model_pattern == 'BEST_FROM_SUMMARY':
                 error_msg_detail += f" Resolved from summary to Base: '{model_base_name_to_load}', Version: '{version_to_load}'."
            else:
                error_msg_detail += f" Tried with version '{version_to_load}'."
            error_msg_detail += " Ensure it exists in GCS at MainOutput/models/ and was synced to /tmp/gcs_models. Check server logs."
            return dbc.Alert(error_msg_detail, color="danger")
        
        # --- Feature Engineering ---
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
                # Attempt to load feature names from the global cache if model manager doesn't have them
                if interactive_engine_globals['feature_names_from_summary']:
                    model_manager.feature_names = interactive_engine_globals['feature_names_from_summary']
                    print("[Interactive Predict] Loaded feature names from GCS summary cache into ModelManager.")
                else:
                    return dbc.Alert("Error: Feature names not loaded with the model and not found in GCS summary cache.", color="danger")

            features_dict = {feature_name: 0.0 for feature_name in model_manager.feature_names} # Default all to 0.0 float

            # 3. Calculate Core Numeric Features
            features_dict['funding_amount_log'] = np.log1p(funding_amount_val) if pd.notna(funding_amount_val) else 0.0 
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
            
            if X_final_features.isnull().any().any():
                print(f"[Interactive Predict] Warning: NaNs found in final feature vector before prediction. Filling with 0. Columns: {X_final_features.columns[X_final_features.isnull().any()].tolist()}")
                X_final_features.fillna(0.0, inplace=True)

            print(f"[Interactive Predict] Manual Feature Engineering complete. Final feature count: {len(X_final_features.columns)}")

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
                html.H4(f"Predicted Funding Stage: {predicted_stage_label} (Idx: {predicted_stage_numeric_final})"),
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
                html.P(f"Type: {model_type_disp}, Ver: {model_version_disp}, Trained Acc: {model_acc_disp}")
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

    # Ensure server is defined for Gunicorn even if the main app init fails before server assignment
    # This is a last resort and ideally should not be hit if the main try block for app setup is robust.
    if 'server' not in globals():
        print("CRITICAL: 'server' was not defined. Attempting to create a minimal fallback server.")
        minimal_fallback_app = Dash(__name__)
        minimal_fallback_app.layout = html.Div("Emergency fallback: Server variable was not defined.")
        server = minimal_fallback_app.server
        print("Minimal fallback server created.")

    if __name__ == "__main__":
        app.run(debug=True)
except Exception as e:
    import traceback
    print("CRITICAL ERROR DURING APP STARTUP:", e)
    print(traceback.format_exc())