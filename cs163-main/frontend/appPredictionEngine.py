import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, exceptions
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import json
import glob
import os
import uuid
from datetime import datetime
import logging
import plotly.io as pio
import sys # <<< ADDED

# --- Calculate project root and add to sys.path ---
# This script is in cs163-main/frontend/
# Project root (cs163-main) is one level up.
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    # project_root is currently 'cs163-main/frontend'
    project_root = os.path.dirname(project_root) # This should now be 'cs163-main'
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError: # __file__ is not defined (e.g. in interactive interpreter)
    logger.warning("__file__ not defined, sys.path modification for project_root might not be effective if not run as script.")
    project_root = "." # Fallback

# Configure logging for the Dash app (Moved Before Import)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import custom class for loading model pickle ---
try:
    # Adjusted import path (assuming project_root 'cs163-main' is now in sys.path)
    from backend.ML.funding_stage_prediction import AnomalyDetector#, apply_post_prediction_rules # <<< Rule function import commented out
except ImportError as e:
    logger.error(f"Failed to import AnomalyDetector from backend.ML.funding_stage_prediction: {e}. Anomaly detection features will be disabled.") # Removed rule func from error msg
    AnomalyDetector = None # Define as None if import fails
    # apply_post_prediction_rules = None # Keep commented out

# --- Constants (now based on project_root and backend structure) ---
# MainOutput is inside cs163-main/backend/MainOutput/
BACKEND_DIR = os.path.join(project_root, 'backend')
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'MainOutput') 
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
VIZ_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
PROTOTYPE_DIR = os.path.join(OUTPUT_DIR, 'prototype_dashboard') 
SUMMARY_GLOB = os.path.join(OUTPUT_DIR, "summary_*.json")
FORECAST_HTML_GLOB = os.path.join(PROTOTYPE_DIR, "bay_area_funding_trend_interactive_*.html")
CALIBRATION_PLOT_GLOB = os.path.join(VIZ_DIR, "calibration_plot*.png")

# Ensure directories exist (especially if MainOutput is now expected at a new common root)
# These are primarily for the backend to create, but app might read from them.
# os.makedirs(OUTPUT_DIR, exist_ok=True) # AppEngine likely won't create these
# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(VIZ_DIR, exist_ok=True)
# os.makedirs(PROTOTYPE_DIR, exist_ok=True)

# --- Global Variables (Load data once on startup) ---
app_data = {
    "model": None,
    "scaler": None,
    "feature_names": [],
    "class_mapping_index_to_label": {},
    "benchmarks": {},
    "feature_importance": {},
    "latest_forecast_html": None,
    "latest_calibration_plot": None,
    "pipeline_summary": None,
    "best_model_name": None,
    "anomaly_detector": None, # Add anomaly detector instance holder
    "model_metadata": None,
    "age_bin_edges": None,
    "age_bin_labels": None
}

# --- Helper Functions ---

def find_latest_file(pattern):
    """Finds the most recently modified file matching a glob pattern."""
    try:
        list_of_files = glob.glob(pattern)
        if not list_of_files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return None
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        logger.error(f"Error finding latest file for pattern {pattern}: {e}")
        return None

def load_pipeline_summary(summary_path):
    """Loads the pipeline summary JSON."""
    if not summary_path or not os.path.exists(summary_path):
        logger.error(f"Summary file not found at {summary_path}")
        return None
    try:
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        logger.info(f"Successfully loaded summary from {summary_path}")
        return summary_data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {summary_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading summary file {summary_path}: {e}")
        return None

def load_best_model_and_scaler(summary_data):
    """Loads the model artifacts, prioritizing the Stacking Ensemble model."""
    # --- Default values ---
    model_to_load = None
    scaler = None
    anomaly_detector = None
    feature_names = []
    model_metadata = None
    loaded_model_type = "Unknown" # Track which model actually got loaded

    # --- Attempt 1: Load Dashboard_Model_Stacking_Ensemble_(RF_Meta)_*.joblib ---
    logger.info("Attempt 1: Loading specific Stacking Ensemble (RF Meta) Dashboard model...")
    model_to_load, scaler, anomaly_detector, feature_names, model_metadata = _try_load_single_model("Dashboard_Model_Stacking_Ensemble_(RF_Meta)")
    if model_to_load:
        loaded_model_type = "Stacking Ensemble (RF Meta, Dashboard)"
        logger.info(f"Successfully loaded prioritized model: {loaded_model_type}")

    # --- Attempt 1.5: Load Stacking_Ensemble_(RF_Meta)_*.joblib (non-Dashboard prefixed) ---
    if not model_to_load:
        logger.info("Attempt 1.5: Loading specific Stacking Ensemble (RF Meta) model (non-Dashboard prefix)...")
        model_to_load, scaler, anomaly_detector, feature_names, model_metadata = _try_load_single_model("Stacking_Ensemble_(RF_Meta)")
        if model_to_load:
            loaded_model_type = "Stacking Ensemble (RF Meta)"
            logger.info(f"Successfully loaded model via non-Dashboard prefix: {loaded_model_type}")

    # --- Attempt 2: Load Dashboard_Model_Decision_Tree*.joblib ---
    if not model_to_load:
        logger.info("Attempt 2: Loading specific Decision Tree Dashboard model...")
        model_to_load, scaler, anomaly_detector, feature_names, model_metadata = _try_load_single_model("Dashboard_Model_Decision_Tree_(Calibrated)")
        if model_to_load:
            loaded_model_type = "Decision Tree (Dashboard)"
            logger.info(f"Successfully loaded second priority model: {loaded_model_type}")

    # --- Attempt 3: Load Dashboard_Model_XGBoost*.joblib ---
    if not model_to_load:
        logger.info("Attempt 3: Loading specific XGBoost Dashboard model...")
        model_to_load, scaler, anomaly_detector, feature_names, model_metadata = _try_load_single_model("Dashboard_Model_XGBoost_(Calibrated)")
        if model_to_load:
            loaded_model_type = "XGBoost (Dashboard)"
            logger.info(f"Successfully loaded third priority model: {loaded_model_type}")

    # --- Attempt 4: Load ANY Dashboard_Model_*.joblib ---
    if not model_to_load:
        logger.info("Attempt 4: Loading any generic Dashboard_Model_* model...")
        model_to_load, scaler, anomaly_detector, feature_names, model_metadata = _try_load_single_model("Dashboard_Model") # Generic prefix
        if model_to_load:
            # Try to infer type from metadata if possible
            meta_type = model_metadata.get('model_type', 'Unknown Type')
            loaded_model_type = f"Dashboard Model ({meta_type})"
            logger.info(f"Successfully loaded generic dashboard model: {loaded_model_type}")

    # --- Attempt 5: Load best model from summary ---
    if not model_to_load:
        logger.info("Attempt 5: Loading best model specified in summary...")
        best_model_name_from_summary = summary_data.get('best_model_by_accuracy')
        if best_model_name_from_summary and best_model_name_from_summary != 'Unknown':
            safe_best_model_name = best_model_name_from_summary.replace(" ", "_").replace("(", "").replace(")", "")
            model_to_load, scaler, anomaly_detector, feature_names, model_metadata = _try_load_single_model(safe_best_model_name)
            if model_to_load:
                loaded_model_type = best_model_name_from_summary
                logger.info(f"Successfully loaded best model from summary: {loaded_model_type}")
            else:
                logger.warning(f"Failed to load model '{best_model_name_from_summary}' specified in summary.")
        else:
             logger.info("No best model specified in summary or it was 'Unknown'.")

    # --- Attempt 6: Fallback to absolute latest .joblib (less reliable) ---
    if not model_to_load:
        logger.warning("Attempt 6: Critical fallback - Loading absolute latest *.joblib model.")
        all_models_pattern = os.path.join(MODELS_DIR, "*.joblib")
        latest_model_path = find_latest_file(all_models_pattern)
        if latest_model_path:
             logger.info(f"Trying to load latest generic model: {latest_model_path}")
             try:
                 logger.info(f"Directly loading artifacts from: {latest_model_path}")
                 model_data = joblib.load(latest_model_path)
                 required_keys = ['model', 'scaler', 'feature_names', 'training_metadata', 'class_mapping']
                 if all(key in model_data for key in required_keys):
                     model_to_load = model_data['model']
                     scaler = model_data['scaler']
                     feature_names = model_data['feature_names']
                     model_metadata = model_data.get('training_metadata', {}) # Use .get for safety
                     anomaly_detector = model_data.get('anomaly_detector') # Use .get for safety

                     # Load class mapping and age bins into app_data directly
                     app_data["class_mapping_index_to_label"] = {int(k): str(v) for k, v in model_data.get('class_mapping', {}).items()}
                     training_meta = model_data.get('training_metadata', {})
                     # Only update from artifact if keys exist and are not None
                     age_bin_edges_artifact = training_meta.get("age_bin_edges")
                     age_bin_labels_artifact = training_meta.get("age_bin_labels")
                     if age_bin_edges_artifact is not None:
                         app_data["age_bin_edges"] = age_bin_edges_artifact
                         logger.debug(f"_try_load: Loaded age_bin_edges from artifact: {app_data['age_bin_edges']}")
                     else:
                         logger.debug(f"_try_load: age_bin_edges not found in model artifact's training_metadata. Existing app_data value retained: {app_data.get('age_bin_edges')}")
                     if age_bin_labels_artifact is not None:
                         app_data["age_bin_labels"] = age_bin_labels_artifact
                         logger.debug(f"_try_load: Loaded age_bin_labels from artifact: {app_data['age_bin_labels']}")
                     else:
                         logger.debug(f"_try_load: age_bin_labels not found in model artifact's training_metadata. Existing app_data value retained: {app_data.get('age_bin_labels')}")

                     # Validate anomaly detector if present
                     if anomaly_detector is not None and not hasattr(anomaly_detector, 'detect_anomalies'):
                         logger.warning(f"Loaded anomaly_detector from {latest_model_path} lacks 'detect_anomalies' method. Disabling.")
                         anomaly_detector = None

                     loaded_model_type = f"Latest Generic ({os.path.basename(latest_model_path)})"
                     logger.info(f"Successfully loaded latest generic model via direct load: {loaded_model_type}")
                 else:
                     missing_keys = [key for key in required_keys if key not in model_data]
                     logger.error(f"Required keys {missing_keys} missing in fallback model file {latest_model_path}.")
             except Exception as e:
                 logger.error(f"Error loading latest generic model directly from {latest_model_path}: {e}", exc_info=True)
        else:
            logger.warning("No *.joblib files found in models directory for fallback.")

    # --- Final Check and Store ---
    if model_to_load and scaler and feature_names:
        app_data["model"] = model_to_load
        app_data["scaler"] = scaler
        app_data["feature_names"] = feature_names
        app_data["anomaly_detector"] = anomaly_detector # Might be None if not in artifact
        app_data["model_metadata"] = model_metadata
        # Class mapping and age bins are loaded within _try_load_single_model now
        app_data["best_model_name"] = loaded_model_type # Store the name of the model actually loaded

        logger.info(f"Stored loaded model artifacts for '{loaded_model_type}'. Features: {len(feature_names)}.")
        # Log loaded age bin info from app_data for confirmation
        logger.info(f"App Data - Age Bin Edges: {app_data.get('age_bin_edges')}")
        logger.info(f"App Data - Age Bin Labels: {app_data.get('age_bin_labels')}")
        logger.info(f"App Data - Class Mapping: {app_data.get('class_mapping_index_to_label')}")
        logger.info(f"App Data - Anomaly Detector Loaded: {'Yes' if app_data.get('anomaly_detector') else 'No'}")

        return model_to_load, scaler, anomaly_detector, feature_names, model_metadata
    else:
        logger.critical("CRITICAL: Failed to load any suitable model/scaler/features after all attempts. Dashboard predictions will not work.")
        # Reset critical components
        app_data["model"] = None
        app_data["scaler"] = None
        app_data["feature_names"] = []
        app_data["anomaly_detector"] = None
        app_data["model_metadata"] = None
        app_data["best_model_name"] = "None Loaded"
        return None, None, None, None, None


def _try_load_single_model(model_name_to_load):
    """Internal helper to attempt loading a single model file (joblib preferred).
    Can accept a model name prefix or a full path to a .joblib file.
    """
    logger.info(f"Attempting to load model artifact(s) for: {model_name_to_load}")

    latest_model_path = None
    if os.path.isfile(model_name_to_load) and model_name_to_load.endswith(".joblib"):
        # If a direct path to a joblib file is provided
        latest_model_path = model_name_to_load
        logger.info(f"Direct path provided: {latest_model_path}")
    else:
        # Use the raw name for globbing if it contains parentheses or special chars expected in filename
        glob_pattern_name = model_name_to_load 
        logger.info(f"Looking for model file matching pattern prefix: {glob_pattern_name}")

        # Find the latest model file matching the pattern
        # --- Prioritize .joblib extension --- #
        model_pattern_joblib = os.path.join(MODELS_DIR, f"{glob_pattern_name}*.joblib")
        absolute_search_path = os.path.abspath(model_pattern_joblib)
        logger.info(f"Searching for model using absolute path pattern: {absolute_search_path}")
        latest_model_path = find_latest_file(model_pattern_joblib)

    if not latest_model_path:
        # --- Keep error message focused on .joblib --- #
        logger.warning(f"No model file found matching identifier/pattern: {model_name_to_load} in {MODELS_DIR}")
        return None, None, None, None, None

    # --- Try loading the found model file --- #
    try:
        logger.info(f"Loading model artifacts from: {latest_model_path}")
        model_data = joblib.load(latest_model_path)

        # --- Validate Structure --- #
        # Expect 'training_metadata' and 'class_mapping' from pipeline save step
        required_keys = ['model', 'scaler', 'feature_names', 'training_metadata', 'class_mapping']
        anomaly_detector = None # Initialize
        if 'anomaly_detector' in model_data:
            anomaly_detector = model_data['anomaly_detector']
            # Validate loaded anomaly detector object further
            if anomaly_detector is not None and not hasattr(anomaly_detector, 'detect_anomalies'):
                 logger.warning(f"Loaded anomaly_detector object from {latest_model_path} does not have 'detect_anomalies' method. Disabling anomaly detection.")
                 anomaly_detector = None
        else:
            logger.warning(f"'anomaly_detector' key missing in model file {latest_model_path}. Anomaly detection will be disabled.")

        # Check remaining required keys
        if not all(key in model_data for key in required_keys):
            missing_keys = [key for key in required_keys if key not in model_data]
            logger.error(f"Required keys missing in model file {latest_model_path}: {missing_keys}")
            return None, None, None, None, None

        # --- Load components --- #
        model_to_use = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        model_metadata = model_data['training_metadata']
        # Class mapping is loaded but not returned by this helper, handled in startup_data_load

        # --- Load class mapping and age bin info specifically for the app --- #
        # This ensures app_data has the info even if _try_load_single_model is called standalone
        app_data["class_mapping_index_to_label"] = {int(k): str(v) for k, v in model_data.get('class_mapping', {}).items()}
        # Try loading age bin info if present in metadata (might be added later)
        training_meta = model_data.get('training_metadata', {})
        # Only update from artifact if keys exist and are not None
        age_bin_edges_artifact = training_meta.get("age_bin_edges")
        age_bin_labels_artifact = training_meta.get("age_bin_labels")
        if age_bin_edges_artifact is not None:
            app_data["age_bin_edges"] = age_bin_edges_artifact
            logger.debug(f"_try_load: Loaded age_bin_edges from artifact: {app_data['age_bin_edges']}")
        else:
            logger.debug(f"_try_load: age_bin_edges not found in model artifact's training_metadata. Existing app_data value retained: {app_data.get('age_bin_edges')}")
        if age_bin_labels_artifact is not None:
            app_data["age_bin_labels"] = age_bin_labels_artifact
            logger.debug(f"_try_load: Loaded age_bin_labels from artifact: {app_data['age_bin_labels']}")
        else:
            logger.debug(f"_try_load: age_bin_labels not found in model artifact's training_metadata. Existing app_data value retained: {app_data.get('age_bin_labels')}")

        # Log loaded age bin info for debugging
        logger.debug(f"_try_load: Final app_data age_bin_edges: {app_data['age_bin_edges']}") # Changed log to show final
        logger.debug(f"_try_load: Final app_data age_bin_labels: {app_data['age_bin_labels']}") # Changed log to show final

        return model_to_use, scaler, anomaly_detector, feature_names, model_metadata

    except FileNotFoundError:
        logger.error(f"Model file not found (should not happen after find_latest_file): {latest_model_path}")
        return None, None, None, None, None
    except Exception as e:
        logger.error(f"Error loading model artifacts from {latest_model_path}: {e}", exc_info=True)
        return None, None, None, None, None


def perform_prediction(input_features_dict):
    """Performs prediction using loaded model, scaler, and features."""
    # Check essential components (Model, Scaler, Features, Mapping are primary)
    # Anomaly detector is optional but checked
    required_components = ["model", "scaler", "feature_names", "class_mapping_index_to_label"]
    # Check age bin info availability
    age_info_available = app_data.get("age_bin_edges") is not None and app_data.get("age_bin_labels") is not None
    if not age_info_available:
         logger.warning("perform_prediction: Age bin edges/labels not found in app_data. Age bin feature will default.")

    if not all(app_data.get(k) is not None for k in required_components):
        missing = [k for k in required_components if app_data.get(k) is None]
        logger.error(f"Prediction components not ready. Missing: {missing}")
        return {"error": f"Prediction components not ready. Missing: {missing}"}

    # Get the FULL feature list loaded WITH the model
    required_model_features = app_data["feature_names"]
    logger.debug(f"Dashboard model expects {len(required_model_features)} features: {required_model_features[:5]}...{required_model_features[-5:]}")

    # Initialize DataFrame with ALL the required columns (including OHE)
    features_df = pd.DataFrame(columns=required_model_features, index=[0], dtype=float).fillna(0.0)

    # --- Feature Engineering from Raw Inputs (Core Numeric + Categorical Prep) ---
    try:
        # Get raw values safely
        raw_funding = input_features_dict.get('funding_amount')
        raw_employees = input_features_dict.get('employees')
        raw_industry = input_features_dict.get('industry')
        raw_months_since = input_features_dict.get('months_since_first_funding')
        raw_prev_rounds = input_features_dict.get('previous_rounds')

        # Convert to numeric, default to NaN
        funding_amount = pd.to_numeric(raw_funding, errors='coerce')
        employees = pd.to_numeric(raw_employees, errors='coerce')
        months_since = pd.to_numeric(raw_months_since, errors='coerce')
        prev_rounds = pd.to_numeric(raw_prev_rounds, errors='coerce')

        # Default prev_rounds if missing
        prev_rounds = prev_rounds if pd.notna(prev_rounds) else 0.0

        # --- Calculate Core Numeric Features (Handle NaNs/Infs later) ---
        # Apply log1p transform (handle zero/negative/NaN later)
        features_df.loc[0, 'funding_amount_log'] = np.log1p(funding_amount) if pd.notna(funding_amount) else 0.0

        features_df.loc[0, 'employees'] = employees if pd.notna(employees) else 0.0
        features_df.loc[0, 'months_since_first_funding'] = months_since if pd.notna(months_since) else 0.0
        features_df.loc[0, 'previous_rounds'] = prev_rounds # Already defaulted

        # Derived features (handle division by zero/NaN)
        emp = features_df.loc[0, 'employees']
        months = features_df.loc[0, 'months_since_first_funding']
        fa_log = features_df.loc[0, 'funding_amount_log']
        rounds = features_df.loc[0, 'previous_rounds']

        features_df.loc[0, 'employee_efficiency'] = (funding_amount / max(emp, 1.0)) if pd.notna(funding_amount) and pd.notna(emp) else 0.0
        features_df.loc[0, 'funding_velocity'] = (funding_amount / max(months, 1.0)) if pd.notna(funding_amount) and pd.notna(months) else 0.0

        current_date = datetime.now()
        features_df.loc[0, 'funding_year'] = current_date.year # Use current year
        features_df.loc[0, 'funding_month'] = current_date.month # Use current month
        month_angle = 2. * np.pi * (current_date.month - 1) / 12 # Month 0-11
        features_df.loc[0, 'month_sin'] = np.sin(month_angle)
        features_df.loc[0, 'month_cos'] = np.cos(month_angle)

        # Interaction features
        features_df.loc[0, 'funding_amount_x_age'] = fa_log * months if pd.notna(fa_log) and pd.notna(months) else 0.0
        features_df.loc[0, 'employees_x_rounds'] = emp * rounds if pd.notna(emp) and pd.notna(rounds) else 0.0
        vel = features_df.loc[0, 'funding_velocity']
        features_df.loc[0, 'velocity_x_rounds'] = vel * rounds if pd.notna(vel) and pd.notna(rounds) else 0.0
        features_df.loc[0, 'age_x_employees'] = months * emp if pd.notna(months) and pd.notna(emp) else 0.0

        # --- ADDED Pipeline Features (Default to 0 if calculation not possible) --- #
        # These require historical context not available in single prediction, so default them.
        # In a real-world scenario, these might be looked up or estimated differently.
        features_df.loc[0, 'time_since_last_funding'] = 0.0
        features_df.loc[0, 'funding_amount_ratio_vs_prev'] = 1.0 # Default ratio is 1
        features_df.loc[0, 'funding_vs_industry_median'] = 1.0 # Default ratio is 1
        logger.warning("App Prediction: Features 'time_since_last_funding', 'funding_amount_ratio_vs_prev', 'funding_vs_industry_median' defaulted as historical context is unavailable.")
        # --- End Added Features --- #

        # --- Handle Categorical Features (Industry, Age Bin, Employee Bin) ---

        # 1. Industry Category
        industry_col_prefix = 'industry_category_'
        final_industry = str(raw_industry) if raw_industry else 'Unknown' # Use Unknown if None

        # --- Apply Consolidation Mapping (Mimic Pipeline) ---
        # This is a simplification. Ideally, the mapping used in training
        # should be loaded and applied here. For now, assume common mapping rules.
        potential_other_industries = ['Retail', 'Transport & Logistics', 'Research', 'Quantum Computing'] # Example industries often mapped to Other
        if final_industry in potential_other_industries:
             # Check if 'industry_category_Other' exists in the required features
             other_col_name = f'{industry_col_prefix}Other'
             if other_col_name in features_df.columns:
                 final_industry = 'Other' # Map to 'Other' if the column exists
                 logger.info(f"Mapping input industry '{raw_industry}' to '{final_industry}' for OHE.")
             else:
                 logger.warning(f"Input industry '{raw_industry}' would map to 'Other', but '{other_col_name}' column doesn't exist in model features. Using 'Unknown'.")
                 final_industry = 'Unknown'

        # Construct the expected column name using the final industry string
        industry_column_name = f'{industry_col_prefix}{final_industry}'
        if industry_column_name in features_df.columns:
            features_df.loc[0, industry_column_name] = 1.0
            logger.debug(f"Set industry column '{industry_column_name}' to 1.")
        else:
            # Handle case where the final industry (after potential mapping) still isn't a known column
            unknown_industry_col = f'{industry_col_prefix}Unknown'
            if unknown_industry_col in features_df.columns:
                 features_df.loc[0, unknown_industry_col] = 1.0
                 logger.warning(f"Industry column '{industry_column_name}' not found. Setting fallback '{unknown_industry_col}'.")
            else:
                 logger.error(f"Neither industry column '{industry_column_name}' nor fallback '{unknown_industry_col}' found in model features. Industry info lost.")

        # 2. Company Age Bin
        age_bin_col_prefix = 'company_age_bin_'
        age_column_name_to_set = None
        bin_edges = app_data.get("age_bin_edges")
        bin_labels = app_data.get("age_bin_labels")

        if bin_edges and bin_labels and pd.notna(months):
            try:
                if len(bin_labels) == len(bin_edges) - 1:
                    if months < bin_edges[0]:
                        age_bin_label_calculated = bin_labels[0]
                    elif months >= bin_edges[-1]:
                        age_bin_label_calculated = bin_labels[-1]
                    else:
                        cut_result = pd.cut([months], bins=bin_edges, labels=bin_labels, right=True, include_lowest=True)
                        age_bin_label_calculated = cut_result[0]
                    age_column_name_to_set = f'{age_bin_col_prefix}{age_bin_label_calculated}'
                else:
                    logger.error(f"Age bin label/edge mismatch.")
            except Exception as cut_err:
                logger.error(f"Error using pd.cut for age bin: {cut_err}")

        if age_column_name_to_set and age_column_name_to_set in features_df.columns:
             features_df.loc[0, age_column_name_to_set] = 1.0
             logger.debug(f"Set company age bin column '{age_column_name_to_set}' to 1.")
        else:
             unknown_age_col = f'{age_bin_col_prefix}Unknown_Age' # Match pipeline default
             if unknown_age_col in features_df.columns:
                  features_df.loc[0, unknown_age_col] = 1.0
                  logger.warning(f"Could not set age bin column '{age_column_name_to_set}', using fallback '{unknown_age_col}' if available.")
             else:
                  logger.error(f"Could not set age bin column '{age_column_name_to_set}' and fallback '{unknown_age_col}' not found.")

        # 3. Employee Bin
        # --- ADD Employee Binning (Mimic Pipeline) --- #
        emp_bin_col_prefix = 'employees_bin_'
        emp_column_name_to_set = None
        # Use the same fixed bins as the pipeline
        emp_bins = [-np.inf, 10, 50, 200, 1000, np.inf]
        emp_labels = ['1-10', '11-50', '51-200', '201-1000', '1001+']

        if pd.notna(emp):
             try:
                 if months < emp_bins[0]: # Check lower bound explicitly
                      emp_bin_label_calculated = emp_labels[0]
                 elif months >= emp_bins[-1]: # Check upper bound
                      emp_bin_label_calculated = emp_labels[-1]
                 else:
                      emp_cut_result = pd.cut([emp], bins=emp_bins, labels=emp_labels, right=True, include_lowest=True)
                      emp_bin_label_calculated = emp_cut_result[0]
                 emp_column_name_to_set = f'{emp_bin_col_prefix}{emp_bin_label_calculated}'
             except Exception as emp_bin_err:
                  logger.error(f"Error using pd.cut for employee bin: {emp_bin_err}")

        if emp_column_name_to_set and emp_column_name_to_set in features_df.columns:
             features_df.loc[0, emp_column_name_to_set] = 1.0
             logger.debug(f"Set employees bin column '{emp_column_name_to_set}' to 1.")
        else:
             unknown_emp_col = f'{emp_bin_col_prefix}Unknown_Emp' # Match pipeline default
             if unknown_emp_col in features_df.columns:
                  features_df.loc[0, unknown_emp_col] = 1.0
                  logger.warning(f"Could not set employee bin column '{emp_column_name_to_set}', using fallback '{unknown_emp_col}' if available.")
             else:
                  logger.error(f"Could not set employee bin column '{emp_column_name_to_set}' and fallback '{unknown_emp_col}' not found.")
        # --- End Employee Binning --- #

        # --- Final Check & Fill Missing/Infinite --- #
        # Ensure columns match exactly what was loaded
        features_df = features_df.reindex(columns=required_model_features, fill_value=0.0)
        # Replace NaNs/Infinities that might have occurred during calculation
        features_df.fillna(0.0, inplace=True)
        features_df.replace([np.inf, -np.inf], 0.0, inplace=True)

        # Log subset of final features before scaling
        log_cols_subset = required_model_features[:5] + required_model_features[-5:]
        log_features = {k: f"{v:.4f}" for k, v in features_df.loc[0, log_cols_subset].to_dict().items()}
        logger.info(f"Final {len(required_model_features)} engineered features (subset logged): {log_features}")

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}", exc_info=True)
        return {"error": f"Feature engineering failed: {e}"}

    # --- Scaling ---
    try:
        scaler = app_data["scaler"]
        # Ensure features_df has the exact columns in the exact order the scaler expects
        features_df_ordered = features_df[required_model_features] # Reorder based on loaded list
        features_scaled = scaler.transform(features_df_ordered)
        logger.debug(f"Features scaled successfully, shape: {features_scaled.shape}")
    except Exception as e:
        logger.error(f"Error during feature scaling: {e}", exc_info=True)
        return {"error": f"Feature scaling failed: {e}"}

    # --- Prediction ---
    try:
        model = app_data["model"]
        probabilities = model.predict_proba(features_scaled)[0]
        prediction_idx = np.argmax(probabilities)
        confidence = probabilities[prediction_idx]
    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        return {"error": f"Model prediction failed: {e}"}

    # --- Map prediction index to label ---
    try:
        predicted_label = app_data["class_mapping_index_to_label"].get(prediction_idx, f"Unknown Index ({prediction_idx})")
        logger.info(f"Raw prediction index: {prediction_idx}, Mapped Label: {predicted_label}, Confidence: {confidence:.4f}")

        # Map probabilities to STRING LABELS for the chart
        proba_dict = {app_data["class_mapping_index_to_label"].get(i, f"Class_{i}"): float(p) for i, p in enumerate(probabilities)}

    except Exception as e:
        logger.error(f"Error mapping prediction index or probabilities: {e}", exc_info=True)
        return {"error": f"Prediction mapping failed: {e}"}

        # --- Anomaly Detection ---
    anomaly_info = {"is_anomaly": False, "score": 0.0, "reasons": []}
    final_confidence = confidence # Start with raw confidence
    try:
        anomaly_detector = app_data["anomaly_detector"]
        # Check if detector exists and has the method before calling
        if anomaly_detector and hasattr(anomaly_detector, 'detect_anomalies'):
            # Anomaly detector expects the scaled features
            anomaly_result = anomaly_detector.detect_anomalies(features_scaled)
            logger.info(f"Anomaly detection result: {anomaly_result}")
            anomaly_info.update(anomaly_result)

            # Adjust confidence if anomaly detected
            if anomaly_info.get("is_anomaly"):
                 original_confidence = confidence
                 # More aggressive penalty: scale by (1 - anomaly score)
                 penalty_factor = max(0.0, 1.0 - anomaly_info.get('score', 0.0))
                 final_confidence = confidence * penalty_factor
                 logger.info(f"Anomaly detected. Confidence adjusted from {original_confidence:.4f} to {final_confidence:.4f} (penalty factor {penalty_factor:.2f})")
        else:
             logger.warning("Anomaly detector not available or invalid, skipping detection.")

    except AttributeError as ae:
         logger.error(f"Anomaly check error: {ae}. Method might be missing or detector not loaded correctly.", exc_info=True)
         anomaly_info["reasons"].append(f"Error: {ae}")
    except Exception as e:
        logger.error(f"Error during anomaly detection: {e}", exc_info=True)
        anomaly_info["reasons"].append(f"Error: {e}")
        # Don't return error, just log and proceed without anomaly info

    # --- Prepare Output ---
    # --- Apply Post-Prediction Rules --- #
    raw_output = {
         "prediction_index": int(prediction_idx),
         "predicted_label": predicted_label,
         "confidence": float(final_confidence), # Pass confidence *after* anomaly adjustment
         "probabilities": {k: float(v) for k,v in proba_dict.items()},
         "anomaly_info": anomaly_info
    }

    # --- Rule Application Block Commented Out --- #
    # if apply_post_prediction_rules:
    #      logger.info("Applying post-prediction rules...")
    #      adjusted_output = apply_post_prediction_rules(raw_output, input_features_dict)
    #      logger.info(f"Rules applied. Original Label: {raw_output['predicted_label']}, New Label: {adjusted_output['predicted_label']}")
    # else:
    #      logger.warning("Post-prediction rule function not imported. Skipping rule application.")
    # adjusted_output = raw_output # Use raw output if rules cannot be applied
    # Use the raw output directly when rules are disabled
    adjusted_output = raw_output

    # Round final confidence for display
    adjusted_output["confidence"] = round(adjusted_output.get("confidence", 0.0), 3)

    return adjusted_output


def startup_data_load():
    """Loads all necessary data when the app starts."""
    global app_data
    logger.info("--- Starting Application Data Load ---")

    # 1. Load Summary
    latest_summary_path = find_latest_file(SUMMARY_GLOB)
    if not latest_summary_path:
        logger.critical("CRITICAL: No summary file found. Dashboard cannot function.")
        return
    summary_data = load_pipeline_summary(latest_summary_path)
    if not summary_data:
        logger.critical("CRITICAL: Failed to load summary file. Dashboard cannot function.")
        return
    app_data["pipeline_summary"] = summary_data

    # Extract necessary info from summary
    class_mapping_from_json = summary_data.get('class_mapping', {})
    # Create the correct index -> label map, ensuring keys are integers
    app_data["class_mapping_index_to_label"] = {int(k): str(v) for k, v in class_mapping_from_json.items()}
    logger.debug(f"Mapping loaded from summary and converted for app: {app_data['class_mapping_index_to_label']}")
    app_data["benchmarks"] = summary_data.get('benchmarks', {})
    logger.info(f"Benchmarks loaded with string keys: {list(app_data.get('benchmarks', {}).keys())[:10]}...")

    # --- Explicitly load age bin info from summary --- #
    app_data["age_bin_edges"] = summary_data.get("age_bin_edges")
    app_data["age_bin_labels"] = summary_data.get("age_bin_labels")
    if app_data["age_bin_edges"] is None or app_data["age_bin_labels"] is None:
         logger.warning("Age bin edges/labels not found in the loaded summary JSON.")
    else:
         logger.info("Successfully loaded age bin edges and labels from summary.")
    # --- End age bin loading --- #

    # Safely get feature importance
    feature_importance_data = summary_data.get('feature_importance') # Get the whole FI block
    if feature_importance_data and isinstance(feature_importance_data, dict): # Check if it exists and is a dict
        app_data["feature_importance"] = feature_importance_data.get('importance', {}) # Get the inner importance dict
    else:
        app_data["feature_importance"] = {} # Default to empty dict if FI block is missing or not a dict
        logger.warning("Feature importance block missing or invalid in summary file.")

    app_data["best_model_name"] = summary_data.get('best_model_by_accuracy', 'Unknown')

    # 2. Load Model, Scaler, Anomaly Detector based on summary or fallback
    model, scaler, anomaly_detector, feature_names, model_metadata = load_best_model_and_scaler(summary_data)
    if not model or not scaler or not feature_names:
         logger.critical("CRITICAL: Failed to load model/scaler/features. Dashboard cannot function.")
         # Reset relevant app_data fields to prevent partial state
         app_data["model"] = None
         app_data["scaler"] = None
         app_data["feature_names"] = []
         app_data["anomaly_detector"] = None
         app_data["model_metadata"] = None
         return # Stop loading if critical components failed

    # Note: app_data fields (model, scaler, etc.) are set inside load_best_model_and_scaler

    # 3. Find Plot Files
    app_data["latest_forecast_html"] = find_latest_file(FORECAST_HTML_GLOB)
    logger.info(f"Forecast HTML path found: {app_data['latest_forecast_html']}") # Log the path
    app_data["latest_calibration_plot"] = find_latest_file(CALIBRATION_PLOT_GLOB)
    logger.info(f"Calibration plot path found: {app_data['latest_calibration_plot']}") # Log the path

    logger.info("--- Application Data Load Complete ---")
    logger.info(f"Features expected by model: {app_data.get('feature_names', 'N/A')}") # Use .get()
    logger.info(f"Class Mapping (Index -> Label): {app_data.get('class_mapping_index_to_label', 'N/A')}")


# --- Load data on application start ---
startup_data_load()


# --- Dash App Initialization ---
# Use Dash Bootstrap Components for better styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN], suppress_callback_exceptions=True)
app.title = "Startup Funding Stage Predictor"


# --- App Layout ---
# Define input features based on loaded feature names, excluding target/metadata
# Let's define the core input fields we expect based on the user request
input_feature_keys = [
    'funding_amount', 'employees', 'industry',
    'months_since_first_funding', 'previous_rounds'
    # Add 'location' if it's consistently a feature and we have categories
]

# Determine Industry categories (example - ideally load from summary/config)
# Placeholder - this should be dynamically loaded if possible
industry_options = sorted([
    'AI & ML', 'IT & Software', 'Healthcare', 'Biotech', 'FinTech',
    'EdTech', 'Retail', 'Energy', 'Food & Agriculture',
    'Transport & Logistics', 'Real Estate', 'Unknown'
])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Startup Funding Stage Predictor", className="text-center my-4"))),

    dbc.Row([
        # == Left Column: Prediction Engine ==
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Core Prediction Engine")),
                dbc.CardBody([
                    html.P("Enter startup features to predict the likely funding stage."),
                    # --- Input Form ---
                    dbc.Row([
                        dbc.Col(dbc.Label("Funding Amount (USD)"), width=6),
                        dbc.Col(dcc.Input(id='input-funding-amount', type='number', placeholder='e.g., 5000000', step=100000), width=6),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("Number of Employees"), width=6),
                        dbc.Col(dcc.Input(id='input-employees', type='number', placeholder='e.g., 25', step=1), width=6),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("Industry"), width=6),
                        dbc.Col(dcc.Dropdown(id='input-industry', options=industry_options, placeholder='Select Industry'), width=6),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("Months Since First Funding"), width=6),
                        dbc.Col(dcc.Input(id='input-months-since-first', type='number', placeholder='e.g., 18', step=1), width=6),
                    ], className="mb-2"),
                     dbc.Row([
                        dbc.Col(dbc.Label("Previous Funding Rounds"), width=6),
                        dbc.Col(dcc.Input(id='input-previous-rounds', type='number', placeholder='e.g., 1', step=1), width=6),
                    ], className="mb-3"),
                    # Add other inputs similarly if needed (e.g., location)

                    dbc.Button("Predict Funding Stage", id='predict-button', color="primary", n_clicks=0, className="w-100"),
                    html.Hr(),
                    # --- Prediction Output ---
                    html.Div(id='prediction-output-area', children=[
                        dbc.Alert("Prediction results will appear here.", color="secondary")
                    ])
                ])
            ], className="shadow-sm mb-4")
        ], md=6),

        # == Right Column: Explanations & Insights ==
        dbc.Col([
            # -- Section II: Prediction Explanation --
            dbc.Card([
                dbc.CardBody(id='explanation-context-area', children=[
                    dbc.Alert("Detailed explanation will appear here after prediction.", color="info") # Placeholder
                ])
            ], className="shadow-sm mb-4"),

            # -- Section III: Market & Model Insights --
            dbc.Card([
                dbc.CardBody([
                    # Time Series Plot
                    html.H5("Funding Trend Forecast (Interactive)"),
                    html.Iframe(
                        id='forecast-plot-iframe',
                        # Use os.path.basename to get just the filename for assets URL
                        src=app.get_asset_url(os.path.basename(app_data.get("latest_forecast_html", ""))) if app_data.get("latest_forecast_html") and os.path.exists(os.path.join("assets", os.path.basename(app_data.get("latest_forecast_html", "")))) else "",
                        style={"border": "none", "width": "100%", "height": "550px"} # Increased height
                    ) if app_data.get("latest_forecast_html") and os.path.exists(os.path.join("assets", os.path.basename(app_data.get("latest_forecast_html", "")))) else html.P("Forecast plot not available or not found in assets folder."), # Updated condition
                    html.Hr(),
                    # Calibration Plot
                    html.H5("Model Calibration"),
                    html.Img(
                        id='calibration-plot-img',
                        # Use os.path.basename for assets URL
                        src=app.get_asset_url(os.path.basename(app_data.get("latest_calibration_plot", ""))) if app_data.get("latest_calibration_plot") and os.path.exists(os.path.join("assets", os.path.basename(app_data.get("latest_calibration_plot", "")))) else "",
                        style={"max-width": "100%", "height": "auto"}
                    ) if app_data.get("latest_calibration_plot") and os.path.exists(os.path.join("assets", os.path.basename(app_data.get("latest_calibration_plot", "")))) else html.P("Calibration plot not available or not found in assets folder."), # Updated condition
                     html.P("Calibration Check: Shows how well the model's predicted probabilities match actual outcomes. Points closer to the dashed diagonal line indicate better calibration.", className="small text-muted mt-2")
                ])
            ], className="shadow-sm mb-4"),

        ], md=6)
    ])
], fluid=True)

# --- App Callbacks ---

@app.callback(
    [Output('prediction-output-area', 'children'),
     Output('explanation-context-area', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('input-funding-amount', 'value'),
     State('input-employees', 'value'),
     State('input-industry', 'value'), # Pass the raw industry string
     State('input-months-since-first', 'value'),
     State('input-previous-rounds', 'value')]
     # Add State for other inputs if needed
)
def update_prediction_and_context(n_clicks, funding_amount, employees, industry, months_since_first, prev_rounds):
    if n_clicks == 0:
        # Prevent update on initial load
        raise exceptions.PreventUpdate

    if not app_data.get("model") or not app_data.get("feature_names"): # Check both model and features
         pred_alert = dbc.Alert("ERROR: Model or features not loaded. Cannot make predictions.", color="danger")
         explanation_alert = dbc.Alert("Explanation unavailable (model/features not loaded).", color="warning")
         return pred_alert, explanation_alert

    # --- 1. Prepare Input Features ---
    # Pass only the raw input values needed for feature engineering
    input_dict_for_prediction = {
        'funding_amount': funding_amount,
        'employees': employees,
        'industry': industry, # Pass the raw industry string
        'months_since_first_funding': months_since_first,
        'previous_rounds': prev_rounds,
    }
    logger.info(f"Raw inputs for prediction: {input_dict_for_prediction}")

    # --- 2. Perform Prediction ---
    pred_result = perform_prediction(input_dict_for_prediction)

    # --- 3. Generate Output Components ---
    prediction_content = []
    explanation_content = []

    if "error" in pred_result:
        prediction_content.append(dbc.Alert(f"Prediction Error: {pred_result['error']}", color="danger"))
        explanation_content.append(dbc.Alert("Explanation unavailable due to prediction error.", color="warning"))
    else:
        # Prediction Area
        pred_label = pred_result['predicted_label'] # USE THE MAPPED LABEL
        pred_confidence = pred_result['confidence']
        anomaly_info = pred_result['anomaly_info']

        alert_color = "success"
        anomaly_title = ""
        anomaly_details_list = [] # List to hold detailed reasons

        # --- Improved Anomaly Display --- #
        if anomaly_info.get('is_anomaly'):
             alert_color = "warning" # Use warning color for anomalies
             anomaly_score_pct = anomaly_info.get('score', 0.0) * 100
             anomaly_title = html.Strong(f"Anomaly Warning (Score: {anomaly_score_pct:.1f}%) ", className="text-danger")
             # Provide clearer context for the score and reasons
             if anomaly_info.get('reasons'):
                 for reason in anomaly_info['reasons']:
                     anomaly_details_list.append(html.Li(reason, className="small"))
             else:
                 anomaly_details_list.append(html.Li("Anomaly detected, specific reasons unavailable.", className="small"))
        # --- End Improved Anomaly Display --- #


        prediction_content.append(html.Div([
            html.H5("Predicted Funding Stage:", className="mb-0"),
            html.H3(f"{pred_label}", className=f"text-{alert_color} fw-bold"), # Display mapped label
            html.Strong(f"Confidence: {pred_confidence:.1%}"),
            # Display anomaly title and details if present
            html.Div([
                anomaly_title, html.Ul(anomaly_details_list, className="list-unstyled mt-1")
                ], className="mt-2") if anomaly_title else None # Use list-unstyled for cleaner look
        ]))

        # Optional: Probability Distribution Chart (Top 5)
        try:
             # Use the already correctly mapped proba_dict
             top_probs = sorted(pred_result['probabilities'].items(), key=lambda item: item[1], reverse=True)[:5]

             prob_fig = go.Figure(go.Bar(
                 x=[p[1] for p in top_probs],
                 y=[p[0] for p in top_probs], # Use the mapped labels on y-axis
                 orientation='h',
                 marker_color='rgba(50, 171, 96, 0.6)' # Example color
             ))
             prob_fig.update_layout(
                 title="Top Predicted Stage Probabilities",
                 xaxis_title="Probability", yaxis={'categoryorder':'total ascending'},
                 height=250, margin=dict(l=10, r=10, t=30, b=10),
                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                 font=dict(size=10)
             )
             prediction_content.append(dcc.Graph(figure=prob_fig, config={'displayModeBar': False}))
        except Exception as e:
             logger.error(f"Could not generate probability chart: {e}")
             prediction_content.append(html.P("Could not display probability distribution.", className="small text-danger"))


        # Explanation Area
        explanation_content.append(html.H4("II. Prediction Explanation & Context")) # Add section header

        # a) Feature Importance (General)
        feat_importance = app_data.get("feature_importance", {})
        if feat_importance:
            # Convert to list, sort by importance, take top 10
            sorted_importance = sorted(feat_importance.items(), key=lambda item: item[1], reverse=True)
            imp_items = [html.Li(f"{feat}: {imp:.3f}") for feat, imp in sorted_importance[:10]]
            explanation_content.append(html.Div([
                html.H6("General Feature Importance (Top 10 Overall):", className="mt-3"),
                html.Ul(imp_items, className="small list-unstyled")
            ]))
        else:
             explanation_content.append(html.P("General feature importance data not available.", className="small text-muted mt-3"))

        explanation_content.append(html.Hr())

        # b) Benchmark Comparison
        benchmarks = app_data.get("benchmarks", {})
        # Use the predicted STRING label to lookup benchmarks
        pred_label_for_benchmark = pred_result["predicted_label"]
        stage_benchmarks = benchmarks.get(pred_label_for_benchmark)

        if stage_benchmarks:
             comparison_text = [html.H6(f"Comparison to Typical Ranges for Predicted Stage ({pred_label_for_benchmark}):", className="mt-3")]
             # Funding Amount Comparison
             funding_med = stage_benchmarks.get('funding_amount_median')
             funding_q1 = stage_benchmarks.get('funding_amount_q1')
             funding_q3 = stage_benchmarks.get('funding_amount_q3')
             funding_amount_numeric = pd.to_numeric(funding_amount, errors='coerce')

             if pd.notna(funding_amount_numeric) and all(v is not None for v in [funding_med, funding_q1, funding_q3]):
                 comp = "within typical range"
                 color = "text-success"
                 if funding_amount_numeric < funding_q1: comp = "below typical range"; color="text-warning"
                 elif funding_amount_numeric > funding_q3: comp = "above typical range"; color="text-info"
                 comparison_text.append(html.P([
                     html.Strong(" - Funding Amount: "),
                     f"${funding_amount_numeric:,.0f} is ",
                     html.Span(comp, className=color),
                     f" (Median: ${funding_med:,.0f}, Range: ${funding_q1:,.0f}-${funding_q3:,.0f})"
                     ], className="small mb-1"))
             else:
                  comparison_text.append(html.P(html.Strong(" - Funding Amount:"), f" Comparison unavailable (Input: {funding_amount_numeric}, Benchmark Median: {funding_med})", className="small mb-1 text-muted"))

             # Employee Comparison
             emp_med = stage_benchmarks.get('employees_median')
             emp_q1 = stage_benchmarks.get('employees_q1')
             emp_q3 = stage_benchmarks.get('employees_q3')
             employees_numeric = pd.to_numeric(employees, errors='coerce')

             if pd.notna(employees_numeric) and all(v is not None for v in [emp_med, emp_q1, emp_q3]):
                 comp = "within typical range"
                 color = "text-success"
                 if employees_numeric < emp_q1: comp = "below typical range"; color="text-warning"
                 elif employees_numeric > emp_q3: comp = "above typical range"; color="text-info"
                 comparison_text.append(html.P([
                     html.Strong(" - Employees: "),
                     f"{employees_numeric:.0f} is ",
                     html.Span(comp, className=color),
                     f" (Median: {emp_med:.0f}, Range: {emp_q1:.0f}-{emp_q3:.0f})"
                     ], className="small mb-1"))
             else:
                  comparison_text.append(html.P(html.Strong(" - Employees:"), f" Comparison unavailable (Input: {employees_numeric}, Benchmark Median: {emp_med})", className="small mb-1 text-muted"))

             explanation_content.append(html.Div(comparison_text))
        else:
            # Use mapped label if lookup failed
            explanation_content.append(html.P(f"Benchmark data not available for predicted stage '{pred_label_for_benchmark}'.", className="small text-muted mt-3"))

    return prediction_content, explanation_content


# --- Asset Handling ---
# Create assets folder if it doesn't exist
if not os.path.exists("assets"):
    os.makedirs("assets")
    logger.info("Created 'assets' directory.")

# Helper function to copy assets if source exists
def copy_asset(src_path_key, asset_type):
    src_path = app_data.get(src_path_key)
    if src_path and os.path.exists(src_path):
        dest_filename = os.path.basename(src_path)
        dest_path = os.path.join("assets", dest_filename)
        try:
            import shutil
            # Check if destination exists and is older than source
            if not os.path.exists(dest_path) or os.path.getmtime(src_path) > os.path.getmtime(dest_path):
                 shutil.copy2(src_path, dest_path)
                 logger.info(f"Copied/Updated latest {asset_type} to {dest_path}")
            else:
                 logger.info(f"Existing {asset_type} in assets folder is already up-to-date.")
        except Exception as e:
            logger.error(f"Failed to copy {asset_type} to assets folder: {e}")
    else:
         logger.warning(f"{asset_type.capitalize()} source file not found or path not set ('{src_path}'). Ensure the file exists in the expected pipeline output location.")

# Try copying the latest plot files to assets on startup
copy_asset("latest_calibration_plot", "calibration plot")
copy_asset("latest_forecast_html", "forecast HTML")


# --- Run the App ---
if __name__ == '__main__':
    # Optional: reload if running as main script, helps ensure latest assets are checked
    # startup_data_load() # Might be redundant if already called globally
    if not app_data.get("model"): # Use .get()
         logger.critical("Model failed to load on startup. Dashboard will not work.")
         print("\nCRITICAL ERROR: Model failed to load. Please check logs. Dashboard cannot start.\n")
    else:
         logger.info("Starting Dash server...")
         # Ensure assets folder is served - Dash does this by default if 'assets' exists
         app.run(debug=True, port=8054) 