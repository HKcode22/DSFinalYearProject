import joblib
import json
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging
# --- Import the rule function --- #
# from MLPredictiveAnalysis.funding_stage_prediction import apply_post_prediction_rules # <<< RE-COMMENTED

# --- Define AnomalyDetector Class (copied from pipeline) --- #
# This is needed for joblib/pickle to load the saved model artifacts
class AnomalyDetector:
    """Detects anomalies and potential manipulation in startup data"""

    def __init__(self, contamination=0.05):
        """Initialize detector with contamination parameter (expected outlier ratio)"""
        # Need to import IsolationForest here if not globally imported
        try:
             from sklearn.ensemble import IsolationForest
        except ImportError:
             logger.error("Scikit-learn not installed? Cannot use AnomalyDetector.")
             IsolationForest = None
        self.isolation_forest = None
        self.contamination = contamination
        self.feature_ranges = {}
        self.startup_data_cache = {}
        self.known_companies = set()

    def fit(self, X, startup_names=None):
        """Train anomaly detection model on startup data

        Args:
            X: Feature matrix for startups
            startup_names: Optional list of company names
        """
        # Ensure IsolationForest was imported
        if IsolationForest is None:
             logger.error("IsolationForest not available, cannot fit.")
             return False
        try:
            # Train isolation forest for outlier detection
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            )
            self.isolation_forest.fit(X)

            # Store feature ranges for basic sanity checks
            self.feature_ranges = {
                'min': np.min(X, axis=0),
                'max': np.max(X, axis=0),
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0),
                'q1': np.percentile(X, 25, axis=0),
                'q3': np.percentile(X, 75, axis=0)
            }

            # Cache startup data if names provided
            if startup_names is not None:
                for i, name in enumerate(startup_names):
                    if i < len(X):
                        self.startup_data_cache[name] = X[i]
                        self.known_companies.add(name)

            logger.info(f"Fitted anomaly detector with {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"Error fitting anomaly detector: {str(e)}")
            return False

    def detect_anomalies(self, X, company_name=None, threshold=-0.5):
        """
        Detect anomalies in startup data

        Args:
            X: Feature matrix or single sample
            company_name: Optional company name for additional checks
            threshold: Decision threshold (lower = more strict)

        Returns:
            Dictionary with anomaly flags and scores
        """
        if self.isolation_forest is None:
            return {'is_anomaly': False, 'score': 0.0, 'reasons': ['Detector not fitted']}
        try:
            # Ensure X is 2D
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            # Initialize anomaly result
            anomalies = {
                'is_anomaly': False,
                'score': 0.0, # Score is now based SOLELY on Isolation Forest
                'reasons': []
            }

            # Apply isolation forest to get anomaly score
            if self.isolation_forest is not None:
                scores = self.isolation_forest.decision_function(X)
                predictions = self.isolation_forest.predict(X)

                # Lower scores = more anomalous. Convert score to be 0 (normal) to 1 (very anomalous).
                # The default decision_function score range isn't strictly fixed, but often centers around 0.
                # We can normalize or just use the prediction flag.
                # Let's base the flag on the prediction (-1 is outlier) and report the raw score.
                raw_score = scores[0]
                is_outlier = predictions[0] == -1

                if is_outlier:
                    anomalies['is_anomaly'] = True
                    # Map the raw score (which is negative for outliers) to a 0-1 range approximately
                    # A simple approach: scale based on the threshold used for prediction (-1)
                    # This is heuristic - might need refinement based on typical score distributions
                    anomaly_score_mapped = min(1.0, max(0.0, 1.0 - (raw_score / -0.2))) # Example mapping
                    anomalies['score'] = anomaly_score_mapped # Use mapped score
                    anomalies['reasons'].append(f"Isolation Forest outlier (Raw Score: {raw_score:.3f})")
                else:
                    # Still report score even if not flagged as outlier
                    anomalies['score'] = min(1.0, max(0.0, 1.0 - (raw_score / -0.2))) # Report mapped score
                    # anomalies['reasons'].append(f"Isolation Forest normal (Raw Score: {raw_score:.3f})") # Optional: log normal score
            else:
                anomalies['reasons'].append("Isolation Forest not fitted.")

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {
                'is_anomaly': True,
                'reason': f'detection_error: {str(e)}',
                'score': 1.0}
# --- End AnomalyDetector Class --- 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants (match app.py) ---
OUTPUT_DIR = './MainOutput'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
SUMMARY_GLOB = os.path.join(OUTPUT_DIR, "summary_*.json")

# --- Prioritize the Dashboard Decision Tree Calibrated model --- #
DEFAULT_MODEL_PATTERN = os.path.join(MODELS_DIR, "Dashboard_Model_Decision_Tree_Calibrated_v*.joblib")
FALLBACK_MODEL_PATTERN = os.path.join(MODELS_DIR, "*.joblib") # Generic fallback

# --- Helper Functions (match app.py) ---
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

def load_summary(file_path):
    """Loads summary data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            summary_data = json.load(f)
        logger.info(f"Successfully loaded summary from {file_path}")
        return summary_data
    except FileNotFoundError:
        logger.error(f"Summary file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from summary file: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading summary file {file_path}: {e}", exc_info=True)
        return None

def load_latest_model_and_summary():
    """Loads the latest model (prioritizing Dashboard_Model_Decision_Tree), scaler, features, mapping, age bins, and anomaly detector."""
    # Find latest summary
    latest_summary_path = find_latest_file(SUMMARY_GLOB)
    if not latest_summary_path:
        logger.error("No summary JSON found.")
        return None, None, None, None, None, None, None
    
    summary_data = load_summary(latest_summary_path)
    if not summary_data:
        return None, None, None, None, None, None, None

    # --- Load Age Bin Info from Summary --- #
    age_bin_edges = summary_data.get("age_bin_edges")
    age_bin_labels = summary_data.get("age_bin_labels")
    class_mapping = summary_data.get('class_mapping', {})
    # Convert keys to integers if they are strings
    if class_mapping and all(isinstance(k, str) for k in class_mapping.keys()):
         try:
             class_mapping = {int(k): v for k, v in class_mapping.items()}
         except ValueError:
             logger.error("Failed to convert class mapping keys to integers.")
             class_mapping = {} # Reset on error

    if age_bin_edges is None or age_bin_labels is None:
        logger.warning("Age bin edges/labels missing from summary.")

    # --- Find the LATEST Model (Prioritize Dashboard_Model_Decision_Tree) --- #
    model_path = None
    model_type = "Unknown"
    model_to_load = None
    scaler = None
    feature_names = None
    anomaly_detector = None

    # 1. Try loading the specific Dashboard_Model_Decision_Tree prefix FIRST
    dashboard_dt_pattern = "Dashboard_Model_Decision_Tree_Calibrated*.joblib"
    latest_dashboard_dt = find_latest_file(os.path.join(MODELS_DIR, dashboard_dt_pattern))

    if latest_dashboard_dt and os.path.exists(latest_dashboard_dt):
        logger.info(f"Prioritizing latest Decision Tree dashboard model: {latest_dashboard_dt}")
        model_path = latest_dashboard_dt
        model_type = "Decision Tree (Dashboard)"
    else:
        logger.warning(f"No specific Decision Tree Dashboard model found matching '{dashboard_dt_pattern}'.")
        # 2. Fallback: Try loading the absolute latest *.joblib file
        logger.warning("Falling back to loading absolute latest model file...")
        fallback_pattern = "*.joblib"
        latest_fallback_model = find_latest_file(os.path.join(MODELS_DIR, fallback_pattern))
        if latest_fallback_model and os.path.exists(latest_fallback_model):
             logger.info(f"Loading latest fallback model file: {latest_fallback_model}")
             model_path = latest_fallback_model
             model_type = f"Latest Fallback ({os.path.basename(latest_fallback_model)})"
        else:
             logger.error("Could not find specific Decision Tree model or any fallback model.")
             return None, None, None, None, None, None, None

    # --- Load the selected model --- #
    if model_path:
        try:
            logger.info(f"Loading artifact from: {model_path}")
            model_data = joblib.load(model_path)
            required_keys = ['model', 'scaler', 'feature_names', 'class_mapping'] # Removed training_metadata as less critical here
            if all(key in model_data for key in required_keys):
                model_to_load = model_data['model']
                scaler = model_data['scaler']
                feature_names = model_data['feature_names']
                # Use mapping from loaded model if summary one was missing?
                if not class_mapping:
                     loaded_map = model_data.get('class_mapping', {})
                     if loaded_map and all(isinstance(k, str) for k in loaded_map.keys()):
                         try:
                              class_mapping = {int(k): v for k, v in loaded_map.items()}
                              logger.info("Used class mapping loaded from model artifact.")
                         except ValueError:
                              logger.error("Failed to convert class mapping keys from model artifact.")
                # Load anomaly detector if exists
                anomaly_detector = model_data.get('anomaly_detector')
                if anomaly_detector and not hasattr(anomaly_detector, 'detect_anomalies'):
                     logger.warning(f"Anomaly detector in {model_path} is invalid. Disabling.")
                     anomaly_detector = None
                elif anomaly_detector:
                     logger.info(f"Successfully loaded anomaly detector from artifact {model_path}.")
                else:
                     logger.warning(f"Anomaly detector not found in artifact {model_path}.")

                logger.info(f"Successfully loaded {model_type} model, scaler, {len(feature_names)} features.")
            else:
                missing = [k for k in required_keys if k not in model_data]
                logger.error(f"Model artifact {model_path} missing required keys: {missing}")
                return None, None, None, None, None, None, None
        except Exception as e:
            logger.error(f"Error loading model artifact from {model_path}: {e}", exc_info=True)
            return None, None, None, None, None, None, None
    else:
         # This case should ideally not be reached if fallback works, but added for safety
         logger.error("Model path was not determined after checks.")
         return None, None, None, None, None, None, None

    return model_to_load, scaler, feature_names, class_mapping, age_bin_edges, age_bin_labels, anomaly_detector

def test_prediction(input_features_dict, model, scaler, required_features, class_mapping, age_bin_edges, age_bin_labels, anomaly_detector):
    """Performs prediction using loaded components, mimicking app.py's feature engineering."""
    if not model or not scaler or not required_features or not class_mapping:
        logger.error("Essential prediction components not available.")
        return {"error": "Components not loaded."}

    # anomaly_detector can be None, handle calls appropriately
    
    logger.debug(f"Test Script - Model expects {len(required_features)} features.")

    # Initialize DataFrame with ALL the required columns (including OHE)
    features_df = pd.DataFrame(columns=required_features, index=[0], dtype=float).fillna(0.0)

    # --- Feature Engineering (Mirror app.py logic) --- 
    try:
        # Extract raw inputs
        raw_funding_amount = input_features_dict.get('funding_amount')
        raw_employees = input_features_dict.get('employees')
        raw_industry = input_features_dict.get('industry') 
        raw_months_since = input_features_dict.get('months_since_first_funding')
        raw_prev_rounds = input_features_dict.get('previous_rounds')

        # Basic Type Conversion & Handling Missing Raw Inputs
        funding_amount = pd.to_numeric(raw_funding_amount, errors='coerce')
        employees = pd.to_numeric(raw_employees, errors='coerce')
        industry = str(raw_industry) if raw_industry is not None else 'Unknown' # Default missing industry
        months_since = pd.to_numeric(raw_months_since, errors='coerce')
        prev_rounds = pd.to_numeric(raw_prev_rounds, errors='coerce')

        # Default prev_rounds if missing
        prev_rounds = prev_rounds if pd.notna(prev_rounds) else 0.0

        # --- Calculate Core Numeric Features --- #
        # Corrected: Use np.log1p to match training feature engineering
        features_df.loc[0, 'funding_amount_log'] = np.log1p(funding_amount) if funding_amount is not None and pd.notna(funding_amount) else 0.0 # Use log1p and handle NaN
        features_df.loc[0, 'employees'] = employees if employees is not None and pd.notna(employees) else 0.0
        features_df.loc[0, 'months_since_first_funding'] = months_since if months_since is not None and pd.notna(months_since) else 0.0
        features_df.loc[0, 'previous_rounds'] = prev_rounds # Already defaulted

        # --- Calculate Derived Features --- # (Ensure inputs are not NaN before calculation)
        fa_log = features_df.loc[0, 'funding_amount_log']
        emp = features_df.loc[0, 'employees']
        months = features_df.loc[0, 'months_since_first_funding']
        rounds = features_df.loc[0, 'previous_rounds']

        # Employee Efficiency
        features_df.loc[0, 'employee_efficiency'] = (funding_amount / max(emp, 1.0)) if pd.notna(funding_amount) and pd.notna(emp) else 0.0
        # Funding Velocity
        features_df.loc[0, 'funding_velocity'] = (funding_amount / max(months, 1.0)) if pd.notna(funding_amount) and pd.notna(months) else 0.0
        # Funding Amount * Age
        features_df.loc[0, 'funding_amount_x_age'] = fa_log * months if pd.notna(fa_log) and pd.notna(months) else 0.0
        # Employees * Rounds
        features_df.loc[0, 'employees_x_rounds'] = emp * rounds if pd.notna(emp) and pd.notna(rounds) else 0.0
        # Velocity * Rounds
        vel = features_df.loc[0, 'funding_velocity']
        features_df.loc[0, 'velocity_x_rounds'] = vel * rounds if pd.notna(vel) and pd.notna(rounds) else 0.0
        # Age * Employees
        features_df.loc[0, 'age_x_employees'] = months * emp if pd.notna(months) and pd.notna(emp) else 0.0

        # --- Cyclical Features (Assuming current year/month for simplicity in test) ---
        now = datetime.now()
        features_df.loc[0, 'funding_year'] = now.year
        features_df.loc[0, 'funding_month'] = now.month
        month_angle = 2. * np.pi * (now.month - 1) / 12 # Adjust month to be 0-11
        features_df.loc[0, 'month_sin'] = np.sin(month_angle)
        features_df.loc[0, 'month_cos'] = np.cos(month_angle)

        # --- ADD Pipeline Features (Default to 0 like in app.py) --- #
        # These require historical context not available in single prediction
        features_df.loc[0, 'time_since_last_funding'] = 0.0
        features_df.loc[0, 'funding_amount_ratio_vs_prev'] = 1.0 # Default ratio is 1
        features_df.loc[0, 'funding_vs_industry_median'] = 1.0 # Default ratio is 1
        logger.info("Test Script: Defaulted historical features.")
        # --- End Added Features ---

        # --- Categorical Features: Industry --- #
        industry_col_prefix = 'industry_category_'
        # Directly use the input industry to form the column name
        logger.info(f"Test Script - Input industry: '{industry}'")
        industry_column_name = f"{industry_col_prefix}{industry}" # Use original input industry
        if industry_column_name in features_df.columns:
            features_df.loc[0, industry_column_name] = 1.0
        else:
            # Handle unseen industry - set the 'Unknown' column if it exists
            unknown_industry_col = f"{industry_col_prefix}Unknown"
            if unknown_industry_col in features_df.columns:
                features_df.loc[0, unknown_industry_col] = 1.0
            # logger.warning(f"Industry '{industry}' not seen during training, using Unknown.")

        # --- Categorical Features: Company Age Bin --- #
        # Use the loaded edges and labels
        age_bin_col_prefix = 'company_age_bin_' # Ensure correct prefix is used
        age_column_name_to_set = None

        if age_bin_edges is not None and age_bin_labels is not None and months_since is not None and pd.notna(months_since):
            try:
                 # Ensure correct number of labels for bins
                if len(age_bin_labels) == len(age_bin_edges) - 1:
                     # Determine the bin label using pd.cut
                     if months_since < age_bin_edges[0]:
                         age_bin_label_calculated = age_bin_labels[0]
                     elif months_since >= age_bin_edges[-1]:
                         age_bin_label_calculated = age_bin_labels[-1]
                     else:
                         # closed='right' means intervals are (a, b]
                         cut_result = pd.cut([months_since], bins=age_bin_edges, labels=age_bin_labels, right=True, include_lowest=True)
                         age_bin_label_calculated = cut_result[0]

                     age_column_name_to_set = f"{age_bin_col_prefix}{age_bin_label_calculated}"
                     # logger.debug(f"Age {months_since} falls into bin: {age_bin_label_calculated}")
                else:
                    logger.warning(f"Mismatch between age bin edges ({len(age_bin_edges)}) and labels ({len(age_bin_labels)}). Cannot determine age bin.")

            except Exception as age_bin_err:
                 logger.error(f"Error calculating age bin for {months_since}: {age_bin_err}")

        # Set the determined age bin column, or fallback to an 'Unknown' if defined/needed
        if age_column_name_to_set and age_column_name_to_set in features_df.columns:
             features_df.loc[0, age_column_name_to_set] = 1.0
        else:
             # Handle case where calculated bin doesn't exist (shouldn't happen if features match)
             # Or if age info was missing initially
             # Optionally set an 'Unknown' age bin if one exists in columns
             unknown_age_bin_col = f"{age_bin_col_prefix}Unknown"
             if unknown_age_bin_col in features_df.columns:
                   features_df.loc[0, unknown_age_bin_col] = 1.0
             # logger.warning(f"Could not set age bin column '{age_column_name_to_set}' or age info missing.")

        # --- ADD Employee Binning (Mirror app.py) --- #
        emp_bin_col_prefix = 'employees_bin_'
        emp_column_name_to_set = None
        # Use the same fixed bins as the pipeline
        emp_bins = [-np.inf, 10, 50, 200, 1000, np.inf] 
        emp_labels = ['1-10', '11-50', '51-200', '201-1000', '1001+']

        if pd.notna(emp):
            try:
                if emp < emp_bins[0]: # Check lower bound explicitly
                    emp_bin_label_calculated = emp_labels[0]
                elif emp >= emp_bins[-1]: # Check upper bound
                    emp_bin_label_calculated = emp_labels[-1]
                else:
                    emp_cut_result = pd.cut([emp], bins=emp_bins, labels=emp_labels, right=True, include_lowest=True)
                    emp_bin_label_calculated = emp_cut_result[0]
                emp_column_name_to_set = f'{emp_bin_col_prefix}{emp_bin_label_calculated}'
            except Exception as emp_bin_err:
                logger.error(f"Error using pd.cut for employee bin in test: {emp_bin_err}")

        if emp_column_name_to_set and emp_column_name_to_set in features_df.columns:
            features_df.loc[0, emp_column_name_to_set] = 1.0
            logger.debug(f"Test Script - Set employees bin column '{emp_column_name_to_set}' to 1.")
        else:
            unknown_emp_col = f'{emp_bin_col_prefix}Unknown_Emp' # Match pipeline default
            if unknown_emp_col in features_df.columns:
                features_df.loc[0, unknown_emp_col] = 1.0
                logger.warning(f"Test Script - Could not set employee bin column '{emp_column_name_to_set}', using fallback '{unknown_emp_col}' if available.")
            else:
                logger.error(f"Test Script - Could not set employee bin column '{emp_column_name_to_set}' and fallback '{unknown_emp_col}' not found.")
        # --- End Employee Binning ---

        # --- Final Check & Fill Missing/Infinite --- 
        features_df = features_df.reindex(columns=required_features, fill_value=0.0)
        features_df.fillna(0.0, inplace=True)
        features_df.replace([np.inf, -np.inf], 0.0, inplace=True)

        # Log subset of final features 
        log_cols_subset = required_features[:5] + required_features[-5:]
        log_features = {k: f"{v:.4f}" for k, v in features_df.loc[0, log_cols_subset].to_dict().items()}
        logger.info(f"Test Script - Final {len(required_features)} features (subset logged): {log_features}")

    except Exception as e:
        logger.error(f"Error during feature engineering in test: {e}", exc_info=True)
        return {"error": "Feature engineering failed."}

    # --- Scaling --- 
    try:
        features_df_ordered = features_df[required_features]
        features_scaled = scaler.transform(features_df_ordered)
        logger.info(f"Test Script - Scaled Features (shape {features_scaled.shape}): {np.round(features_scaled[:, :5], 4)}...{np.round(features_scaled[:, -5:], 4)}") # Log subset
    except Exception as e:
         logger.error(f"Error during scaling in test: {e}", exc_info=True)
         return {"error": "Scaling failed."}

    # --- Prediction --- 
    try:
        prediction_index = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        predicted_label = class_mapping.get(prediction_index, "Unknown Prediction Index")
        confidence = probabilities[prediction_index]

        # Map probabilities to labels
        prob_dict = {class_mapping.get(i, f"Unknown_{i}"): prob for i, prob in enumerate(probabilities)}

    except Exception as e:
         logger.error(f"Error during prediction in test: {e}", exc_info=True)
         return {"error": "Prediction failed."}

    # --- Anomaly Detection --- 
    anomaly_info = {
        'is_anomaly': False,
        'score': 0.0,
        'reasons': []
    }
    final_confidence = confidence # Start with original confidence

    if anomaly_detector is not None:
        try:
            # Use the detect_anomalies method of the loaded detector
            # Pass the SCALED features
            anomaly_result = anomaly_detector.detect_anomalies(features_scaled)
            anomaly_info.update(anomaly_result)
            if anomaly_info.get("is_anomaly"): 
                # Adjust confidence if anomaly detected
                original_confidence = confidence
                # Simplified penalty adjustment (matches app.py)
                penalty_factor = max(0.0, 1.0 - anomaly_info.get('score', 0.0)) # Higher score = bigger penalty
                final_confidence = confidence * penalty_factor
                logger.info(f"Anomaly detected. Confidence adjusted from {original_confidence:.4f} to {final_confidence:.4f} (penalty factor {penalty_factor:.2f})")
        except Exception as e:
            logger.error(f"Error during test anomaly detection: {e}", exc_info=True)
            anomaly_info["reasons"].append(f"Error: {e}")
    else:
        logger.warning("Test script: Anomaly detector not available or invalid, skipping detection.")

    # --- Return the result BEFORE applying post-prediction rules --- #
    # (Rule application is commented out for this test)
    raw_prediction_result = {
         "prediction_index": int(prediction_index),
        "predicted_label": predicted_label,
         "confidence": float(final_confidence), # Confidence *after* anomaly adjustment
         "probabilities": {k: float(v) for k,v in prob_dict.items()},
        "anomaly_info": anomaly_info
    }
    # # Apply the imported rules function (COMMENTED OUT)
    # # Prepare the initial result dictionary to pass to the rule function
    # initial_prediction_result = raw_prediction_result.copy()
    # # Call the rule function, passing the initial result and the original input features
    # adjusted_prediction_result = apply_post_prediction_rules(initial_prediction_result, input_features_dict)
    # # Return the potentially adjusted result
    # return adjusted_prediction_result

    # Return the raw result (no rules applied)
    return raw_prediction_result

# --- Test Cases ---
test_cases = [
    # Early Stage
    {"case_id": "early_1_seed", "inputs": {"funding_amount": 500000, "employees": 5, "industry": "AI & ML", "months_since_first_funding": 6, "previous_rounds": 0}, "expected_stage_range": ["Pre-Seed", "Seed"]},
    {"case_id": "early_2_seed_health", "inputs": {"funding_amount": 1200000, "employees": 10, "industry": "Healthcare", "months_since_first_funding": 9, "previous_rounds": 0}, "expected_stage_range": ["Seed", "Series A"]},
    # Series A/B
    {"case_id": "mid_1_series_a", "inputs": {"funding_amount": 5000000, "employees": 25, "industry": "FinTech", "months_since_first_funding": 18, "previous_rounds": 1}, "expected_stage_range": ["Seed", "Series A", "Series B"]},
    {"case_id": "mid_2_series_b", "inputs": {"funding_amount": 15000000, "employees": 60, "industry": "IT & Software", "months_since_first_funding": 30, "previous_rounds": 2}, "expected_stage_range": ["Series A", "Series B", "Series C"]},
    {"case_id": "mid_3_series_b_biotech", "inputs": {"funding_amount": 25000000, "employees": 45, "industry": "Biotech", "months_since_first_funding": 36, "previous_rounds": 2}, "expected_stage_range": ["Series B", "Series C"]}, # Biotech often higher funding
    # Later Stage
    {"case_id": "late_1_series_c", "inputs": {"funding_amount": 50000000, "employees": 150, "industry": "Retail", "months_since_first_funding": 48, "previous_rounds": 3}, "expected_stage_range": ["Series B", "Series C", "Series D"]},
    {"case_id": "late_2_series_d", "inputs": {"funding_amount": 120000000, "employees": 300, "industry": "Energy", "months_since_first_funding": 60, "previous_rounds": 4}, "expected_stage_range": ["Series C", "Series D", "Series E"]},
    {"case_id": "late_3_ipo_candidate", "inputs": {"funding_amount": 250000000, "employees": 800, "industry": "AI & ML", "months_since_first_funding": 72, "previous_rounds": 5}, "expected_stage_range": ["Series D", "Series E", "Private Equity", "Post-IPO"]}, # Added Post-IPO
    # Edge Cases / Unknowns
    {"case_id": "edge_1_unknown_industry", "inputs": {"funding_amount": 2000000, "employees": 15, "industry": "Unknown", "months_since_first_funding": 12, "previous_rounds": 1}, "expected_stage_range": ["Seed", "Series A", "venture - series unknown"]},
    {"case_id": "edge_2_low_employees_high_funding", "inputs": {"funding_amount": 80000000, "employees": 30, "industry": "Biotech", "months_since_first_funding": 40, "previous_rounds": 3}, "expected_stage_range": ["Series B", "Series C", "Series D"]}, # High funding, low emp (e.g., IP heavy)
    {"case_id": "edge_3_old_company_small_funding", "inputs": {"funding_amount": 1000000, "employees": 50, "industry": "IT & Software", "months_since_first_funding": 90, "previous_rounds": 2}, "expected_stage_range": ["Seed", "Series A", "Series B", "venture - series unknown"]}, # Old company, maybe pivoted or slow growth
    {"case_id": "edge_4_high_rounds_low_total", "inputs": {"funding_amount": 8000000, "employees": 70, "industry": "EdTech", "months_since_first_funding": 60, "previous_rounds": 5}, "expected_stage_range": ["Series A", "Series B", "Series C", "venture - series unknown"]}, # Many rounds but maybe smaller amounts
]

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting Test Prediction Script ---")

    # Find the specific model artifact to test
    model_path = find_latest_file(DEFAULT_MODEL_PATTERN)
    if not model_path:
        logger.warning(f"No model found for pattern '{DEFAULT_MODEL_PATTERN}'. Trying generic fallback.")
        model_path = find_latest_file(FALLBACK_MODEL_PATTERN)
        if not model_path:
            logger.critical("CRITICAL: No model artifacts found in models directory. Exiting.")
            exit()
        else:
            logger.info(f"Using fallback model: {model_path}")
    else:
        logger.info(f"Using model: {model_path}")


    # Load the artifact
    model_data = load_latest_model_and_summary()
    if not model_data:
        logger.critical("Failed to load model artifact. Exiting.")
        exit()

    results = []
    correct_predictions = 0
    total_predictions = 0

    for case in test_cases:
        case_id = case["case_id"]
        inputs = case["inputs"]
        expected_range = case["expected_stage_range"]
        logger.info(f"\n--- Running Test Case: {case_id} ---")
        logger.info(f"Inputs: {inputs}")
        logger.info(f"Expected Range: {expected_range}")

        prediction_output = test_prediction(inputs, *model_data)

        if "error" in prediction_output:
            logger.error(f"Prediction failed for {case_id}: {prediction_output['error']}")
            results.append({**case, "predicted_label": "ERROR", "confidence": 0.0, "is_correct": False, "anomaly": None})
        else:
            predicted_label = prediction_output["predicted_label"]
            confidence = prediction_output["confidence"]
            anomaly_info = prediction_output["anomaly_info"]
            is_correct = predicted_label in expected_range

            logger.info(f"Predicted Label: {predicted_label}")
            logger.info(f"Confidence: {confidence:.3f}") # Log confidence without anomaly mention
            logger.info(f"Prediction Correct (within range {expected_range}): {is_correct}")

            results.append({
                **case,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "is_correct": is_correct,
                "anomaly": anomaly_info
            })
            total_predictions += 1
            if is_correct:
                correct_predictions += 1

    # --- Summary Report ---
    logger.info("\n--- Test Summary ---")
    logger.info(f"Model Tested: {os.path.basename(model_path)}")
    for res in results:
        status = "PASS" if res["is_correct"] else "FAIL"
        logger.info(
            f"{res['case_id']:<30} | Expected Range: {str(res['expected_stage_range']):<45} | Predicted: {res['predicted_label']:<25} | Conf: {res['confidence']:.3f} | Status: {status}"
        )

    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    logger.info(f"\nOverall Accuracy (Predicted label within expected range): {correct_predictions}/{total_predictions} ({accuracy:.2f}%)")

    logger.info("--- Test Prediction Script Finished ---")