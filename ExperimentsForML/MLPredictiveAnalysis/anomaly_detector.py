import numpy as np
from sklearn.ensemble import IsolationForest
import logging
import pandas as pd # Added for isnull check if needed later

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Detects anomalies and potential manipulation in startup data"""

    def __init__(self, contamination=0.05):
        """Initialize detector with contamination parameter (expected outlier ratio)"""
        self.isolation_forest = None
        self.contamination = contamination
        self.feature_ranges = {}
        self.startup_data_cache = {}
        self.known_companies = set()
        self.feature_names = None # Added to store feature names used during fit

    def fit(self, X, startup_names=None, feature_names=None):
        """Train anomaly detection model on startup data

        Args:
            X: Feature matrix for startups
            startup_names: Optional list of company names
            feature_names: Optional list of feature names corresponding to columns in X
        """
        try:
            # Store feature names
            if feature_names:
                 self.feature_names = list(feature_names)
                 if len(self.feature_names) != X.shape[1]:
                      logger.error(f"Feature name count ({len(self.feature_names)}) mismatch with data columns ({X.shape[1]})")
                      self.feature_names = None # Invalidate if mismatch
                      return False
            else:
                 self.feature_names = None # No names provided
                 logger.warning("Fitting AnomalyDetector without feature names.")


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

            logger.info(f"Fitted anomaly detector with {len(X)} samples.")
            return True
        except Exception as e:
            logger.error(f"Error fitting anomaly detector: {str(e)}", exc_info=True)
            return False

    def detect_anomalies(self, X, company_name=None, threshold=-0.5):
        """
        Detect anomalies in startup data

        Args:
            X: Feature matrix or single sample (numpy array expected)
            company_name: Optional company name for additional checks
            threshold: Decision threshold for Isolation Forest score (lower = more strict)

        Returns:
            Dictionary with anomaly flags and scores
        """
        try:
            if self.isolation_forest is None or not self.feature_ranges:
                 logger.warning("Anomaly detector not fitted. Cannot detect anomalies.")
                 return {'is_anomaly': False, 'score': 0.0, 'reasons': ["Detector not fitted"]}

            # Ensure X is 2D numpy array
            if isinstance(X, pd.DataFrame):
                X = X.values
            elif isinstance(X, pd.Series):
                 X = X.values.reshape(1, -1)
            elif len(X.shape) == 1:
                X = X.reshape(1, -1)

            # Validate input dimensions match what model was trained on
            expected_dims = len(self.feature_ranges['min'])
            if X.shape[1] != expected_dims:
                logger.error(f"Anomaly Detection Feature dimension mismatch: expected {expected_dims}, got {X.shape[1]}")
                return {'is_anomaly': True, 'score': 1.0, 'reasons': [f'Dimension mismatch ({X.shape[1]} vs {expected_dims})']}

            # Run standard anomaly checks
            anomalies = {
                'is_anomaly': False,
                'score': 0.0, # Use combined score, not just forest score
                'reasons': []
            }
            max_violation_score = 0.0 # Track severity based on checks

            # Get feature names if available for logging
            f_names = self.feature_names if self.feature_names else [f'Feat_{i}' for i in range(X.shape[1])]

            # Check against known ranges (allowing some margin)
            range_violations_details = []
            for i, val in enumerate(X[0]):
                 min_val = self.feature_ranges['min'][i]
                 max_val = self.feature_ranges['max'][i]
                 # Allow small margin (e.g., 10% or absolute value for small ranges)
                 margin = max(abs(min_val * 0.1), abs(max_val * 0.1), 1e-6)
                 if val < (min_val - margin) or val > (max_val + margin):
                      range_violations_details.append(f"{f_names[i]}={val:.2f} (Range: [{min_val:.2f},{max_val:.2f}])")
            if range_violations_details:
                 anomalies['reasons'].append(f"Range violations ({len(range_violations_details)}): {'; '.join(range_violations_details[:3])}...")
                 max_violation_score = max(max_violation_score, 0.7)

            # Check extreme feature values using IQR
            iqr_violations_details = []
            for i, val in enumerate(X[0]):
                 q1 = self.feature_ranges['q1'][i]
                 q3 = self.feature_ranges['q3'][i]
                 iqr = q3 - q1
                 # Handle cases where IQR is zero or very small
                 if iqr < 1e-9: iqr = 1e-9
                 lower_bound = q1 - 1.5 * iqr
                 upper_bound = q3 + 1.5 * iqr
                 if val < lower_bound or val > upper_bound:
                      iqr_violations_details.append(f"{f_names[i]}={val:.2f} (IQR Range: [{lower_bound:.2f},{upper_bound:.2f}])")
            if iqr_violations_details:
                 anomalies['reasons'].append(f"IQR violations ({len(iqr_violations_details)}): {'; '.join(iqr_violations_details[:3])}...")
                 max_violation_score = max(max_violation_score, 0.6)

            # Check if company data has suddenly changed drastically
            if company_name and company_name in self.startup_data_cache:
                cached_data = self.startup_data_cache[company_name]
                if len(cached_data) == X.shape[1]: # Ensure dimensions match before comparison
                    # Calculate percent change, handle division by zero
                    diff = np.abs(X[0] - cached_data)
                    denom = np.maximum(np.abs(cached_data), 1e-9) # Avoid division by zero
                    pct_change = diff / denom
                    if np.any(pct_change > 0.5): # 50% change in any feature is suspicious
                        max_change_idx = np.argmax(pct_change)
                        anomalies['reasons'].append(f"Company data change >50% (Max: {pct_change[max_change_idx]*100:.1f}% in {f_names[max_change_idx]})")
                        max_violation_score = max(max_violation_score, 0.8)
                else:
                     logger.warning(f"Dimension mismatch between cached ({len(cached_data)}) and current ({X.shape[1]}) data for {company_name}. Cannot check change.")

            # Apply isolation forest to get anomaly score
            if self.isolation_forest is not None:
                scores = self.isolation_forest.decision_function(X)
                predictions = self.isolation_forest.predict(X)

                # Lower scores = more anomalous. Scale score to be 0 (normal) to 1 (very anomalous)
                # Offset_ is the threshold separating inliers from outliers.
                # Score below offset_ is outlier. Max score is typically around 0.1-0.2 for inliers.
                # We can normalize: score closer to -1 is more anomalous.
                iso_score = scores[0]
                # Simple scaling (adjust as needed based on score distribution)
                # Maps approx [-0.2, 0.2] range to [1, 0]
                scaled_iso_score = np.clip((0.2 - iso_score) / 0.4, 0, 1) # Higher value means more anomalous

                if predictions[0] == -1: # Predicted as outlier
                    anomalies['reasons'].append(f"Isolation Forest outlier (Score: {iso_score:.3f})")
                    max_violation_score = max(max_violation_score, scaled_iso_score)

            # Check for potential manipulation patterns (example)
            # This requires knowing feature indices/names
            # Assuming funding_amount_log might be at index 0 (example)
            funding_log_idx = f_names.index('funding_amount_log') if self.feature_names and 'funding_amount_log' in f_names else -1
            if funding_log_idx != -1 and self._check_funding_manipulation(X[0], funding_log_idx):
                anomalies['reasons'].append("Suspicious funding amount pattern detected")
                max_violation_score = max(max_violation_score, 0.9)

            # Final anomaly decision and score
            if max_violation_score > 0:
                 anomalies['is_anomaly'] = True
                 anomalies['score'] = round(max_violation_score, 3)

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}", exc_info=True)
            return {'is_anomaly': True, 'score': 1.0, 'reasons': [f'Detection error: {str(e)}']}

    def _check_funding_manipulation(self, features, funding_idx):
        """Check for suspicious patterns in funding amount feature (example)."""
        try:
            # Example: Check if log funding amount is suspiciously round (e.g., exactly log1p(100M))
            funding_log_val = features[funding_idx]
            # Check if log value corresponds *exactly* to a round million/billion figure
            # This is a very basic check and likely needs refinement
            round_log_vals = [np.log1p(v) for v in [1e6, 5e6, 10e6, 50e6, 100e6, 500e6, 1e9]]
            if any(abs(funding_log_val - r_val) < 1e-6 for r_val in round_log_vals):
                # logger.debug(f"Suspiciously round log funding value detected: {funding_log_val}")
                return True
            return False
        except Exception:
            return False

    # --- Levenshtein distance function is not used by AnomalyDetector, removed --- # 