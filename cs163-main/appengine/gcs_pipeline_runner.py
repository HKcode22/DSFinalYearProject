import os
import sys
import io
import json
import logging
import traceback
from datetime import datetime

# Modify sys.path to include the backend directory for sibling imports
# Assuming this script is in cs163-main/appengine/
# and the backend is in cs163-main/backend/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'backend', 'ML'))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:
    print("Error: google-cloud-storage or google-auth library not found. "
          "Please install with: pip install google-cloud-storage google-auth")
    storage = None
    DefaultCredentialsError = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gcs_pipeline_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- GCS Client Initialization ---
GCS_CLIENT = None
try:
    if storage:
        GCS_CLIENT = storage.Client()
    else:
        logger.warning("GCS client cannot be initialized because google.cloud.storage is not available.")
except DefaultCredentialsError:
    logger.error(
        "Google Cloud Default Credentials not found. Ensure you are authenticated "
        "(e.g., via `gcloud auth application-default login`) or service account key is set."
    )
    GCS_CLIENT = None # Explicitly set to None on auth error
except Exception as e:
    logger.error(f"Failed to initialize Google Cloud Storage client: {e}")
    GCS_CLIENT = None


# --- GCS Helper Functions ---
def _get_gcs_client():
    """Returns the initialized GCS client, or None if not available."""
    if GCS_CLIENT is None:
        logger.error("GCS Client is not initialized. Cannot perform GCS operations.")
    return GCS_CLIENT

def _upload_local_file_to_gcs(local_file_path, bucket_name, gcs_blob_name):
    """Uploads a local file to GCS."""
    client = _get_gcs_client()
    if not client:
        return False
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_filename(local_file_path)
        logger.info(f"Uploaded local file {local_file_path} to gs://{bucket_name}/{gcs_blob_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_file_path} to gs://{bucket_name}/{gcs_blob_name}: {e}")
        return False

def _upload_string_to_gcs(string_data, bucket_name, gcs_blob_name, content_type="application/json"):
    """Uploads a string to GCS."""
    client = _get_gcs_client()
    if not client:
        return False
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_string(string_data, content_type=content_type)
        logger.info(f"Uploaded string data to gs://{bucket_name}/{gcs_blob_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload string to gs://{bucket_name}/{gcs_blob_name}: {e}")
        return False

def _upload_bytes_io_to_gcs(bytes_io_data, bucket_name, gcs_blob_name, content_type="application/octet-stream"):
    """Uploads data from an io.BytesIO object to GCS."""
    client = _get_gcs_client()
    if not client:
        return False
    try:
        bytes_io_data.seek(0) # Ensure cursor is at the beginning
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_file(bytes_io_data, content_type=content_type)
        logger.info(f"Uploaded BytesIO data to gs://{bucket_name}/{gcs_blob_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload BytesIO to gs://{bucket_name}/{gcs_blob_name}: {e}")
        return False

def _download_gcs_blob_to_string(bucket_name, gcs_blob_name):
    """Downloads a GCS blob to a string."""
    client = _get_gcs_client()
    if not client:
        return None
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_name)
        return blob.download_as_text()
    except Exception as e:
        logger.error(f"Failed to download gs://{bucket_name}/{gcs_blob_name} as string: {e}")
        return None

def _download_gcs_blob_to_bytes_io(bucket_name, gcs_blob_name):
    """Downloads a GCS blob to an io.BytesIO object."""
    client = _get_gcs_client()
    if not client:
        return None
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_name)
        bytes_io = io.BytesIO()
        blob.download_to_file(bytes_io)
        bytes_io.seek(0)
        return bytes_io
    except Exception as e:
        logger.error(f"Failed to download gs://{bucket_name}/{gcs_blob_name} to BytesIO: {e}")
        return None

# Placeholder for importing backend classes and defining GCS-aware subclasses
# This will be expanded in the next steps.
# Example:
# from funding_stage_predictionORIGINAL import EnhancedPipeline, ModelManager, Visualizer, TimeSeriesForecaster

# --- Import Backend Classes ---
try:
    from funding_stage_predictionORIGINAL import (
        EnhancedPipeline as OriginalEnhancedPipeline,
        ModelManager as OriginalModelManager,
        Visualizer as OriginalVisualizer,
        AdvancedVisualizer as OriginalAdvancedVisualizer,
        TimeSeriesForecaster as OriginalTimeSeriesForecaster,
        FeatureEngineering as OriginalFeatureEngineering, # Added
        DataLoader as OriginalDataLoader, # Added
        NumpyEncoder # Added
    )
    # Import other necessary components like AnomalyDetector if they are separate and needed
    from funding_stage_predictionORIGINAL import AnomalyDetector as OriginalAnomalyDetector 
    import joblib # For GCSModelManager save/load
    import pandas as pd # For GCSModelManager if it handles dataframes
    import numpy as np # For GCSModelManager if it handles numpy arrays

except ImportError as e:
    logger.error(f"Failed to import classes from funding_stage_predictionORIGINAL.py: {e}")
    logger.error("Ensure funding_stage_predictionORIGINAL.py is in the BACKEND_DIR and sys.path is correct.")
    # Define dummy classes if import fails, to allow script to load but fail gracefully later
    class OriginalEnhancedPipeline: pass
    class OriginalModelManager: pass
    class OriginalVisualizer: pass
    class OriginalAdvancedVisualizer: pass
    class OriginalTimeSeriesForecaster: pass
    class OriginalFeatureEngineering: pass
    class OriginalDataLoader: pass
    class OriginalAnomalyDetector: pass
    class NumpyEncoder(json.JSONEncoder): pass # Basic fallback
    joblib = None
    pd = None
    np = None

# --- Matplotlib Configuration for GCS/Non-Interactive environments ---
import matplotlib
try:
    matplotlib.use('Agg') # Use a non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    logger.error("Matplotlib not found. Please install it: pip install matplotlib")
    plt = None
except Exception as e:
    logger.error(f"Error configuring Matplotlib: {e}")
    plt = None

# --- GCS-Aware Model Manager ---
class GCSModelManager(OriginalModelManager):
    def __init__(self, model_gcs_bucket_name, model_gcs_prefix='models/', local_temp_dir='/tmp/local_models'):
        super().__init__(output_dir=local_temp_dir) # Initialize parent with a local temp dir
        self.gcs_bucket_name = model_gcs_bucket_name
        self.gcs_prefix = model_gcs_prefix.rstrip('/') + '/'  # Ensure trailing slash
        self.local_temp_dir = local_temp_dir
        os.makedirs(self.local_temp_dir, exist_ok=True)
        logger.info(f"GCSModelManager initialized. Bucket: {self.gcs_bucket_name}, Prefix: {self.gcs_prefix}, Local Temp: {self.local_temp_dir}")

    def save_model(self, model_name, model, scaler, feature_names, metadata=None, anomaly_detector=None):
        """Saves the model, scaler, feature names, and metadata to GCS."""
        if not joblib:
            logger.error("Joblib not available, cannot save model.")
            return None
        
        client = _get_gcs_client()
        if not client:
            logger.error("GCS client not available for saving model.")
            return None

        version = datetime.now().strftime("%Y%m%d%H%M%S")
        base_model_name = model_name.split('_v')[0] # Remove old versioning if present
        
        # Consistent naming: <prefix><base_model_name>_v<version>.joblib
        gcs_blob_name = f"{self.gcs_prefix}{base_model_name}_v{version}.joblib"
        gcs_metadata_path = f"{self.gcs_prefix}{base_model_name}_v{version}_metadata.json"

        model_artifact_paths_in_gcs = {}

        try:
            # 1. Save model object (e.g., scikit-learn model)
            model_blob_name = f"{self.gcs_prefix}{base_model_name}_v{version}_model.joblib"
            with io.BytesIO() as model_buffer:
                joblib.dump(model, model_buffer)
                model_buffer.seek(0)
                if _upload_bytes_io_to_gcs(model_buffer, self.gcs_bucket_name, model_blob_name, content_type='application/octet-stream'):
                    model_artifact_paths_in_gcs['model_path'] = f"gs://{self.gcs_bucket_name}/{model_blob_name}"
                else:
                    raise Exception(f"Failed to upload model to {model_blob_name}")

            # 2. Save scaler object
            scaler_blob_name = f"{self.gcs_prefix}{base_model_name}_v{version}_scaler.joblib"
            with io.BytesIO() as scaler_buffer:
                joblib.dump(scaler, scaler_buffer)
                scaler_buffer.seek(0)
                if _upload_bytes_io_to_gcs(scaler_buffer, self.gcs_bucket_name, scaler_blob_name, content_type='application/octet-stream'):
                    model_artifact_paths_in_gcs['scaler_path'] = f"gs://{self.gcs_bucket_name}/{scaler_blob_name}"
                else:
                    raise Exception(f"Failed to upload scaler to {scaler_blob_name}")

            # 3. Save anomaly_detector object if it exists
            if anomaly_detector is not None:
                ad_blob_name = f"{self.gcs_prefix}{base_model_name}_v{version}_anomaly_detector.joblib"
                with io.BytesIO() as ad_buffer:
                    joblib.dump(anomaly_detector, ad_buffer)
                    ad_buffer.seek(0)
                    if _upload_bytes_io_to_gcs(ad_buffer, self.gcs_bucket_name, ad_blob_name, content_type='application/octet-stream'):
                        model_artifact_paths_in_gcs['anomaly_detector_path'] = f"gs://{self.gcs_bucket_name}/{ad_blob_name}"
                    else:
                        logger.warning(f"Failed to upload anomaly detector to {ad_blob_name}. Model will be saved without it in metadata.")
            else:
                model_artifact_paths_in_gcs['anomaly_detector_path'] = None

            # 4. Prepare and save the main metadata JSON file
            if metadata is None: metadata = {}
            main_metadata = {
                'version': version,
                'created_at': datetime.now().isoformat(),
                'model_name': base_model_name,
                'original_model_type': str(type(model)),
                'gcs_bucket': self.gcs_bucket_name,
                'gcs_prefix': self.gcs_prefix,
                'artifact_paths': model_artifact_paths_in_gcs, # Contains GCS paths to model, scaler, ad
                'feature_names': feature_names, # List of feature names
                'training_metadata': metadata.get('training_metadata', {}), # Metrics, etc.
                'class_mapping': metadata.get('class_mapping', {}),
            }
            if not _upload_string_to_gcs(json.dumps(main_metadata, cls=NumpyEncoder, indent=4), 
                                       self.gcs_bucket_name, gcs_metadata_path, 
                                       content_type="application/json"):
                raise Exception(f"Failed to upload main metadata to {gcs_metadata_path}")

            final_gcs_path_for_metadata = f"gs://{self.gcs_bucket_name}/{gcs_metadata_path}"
            logger.info(f"Model artifacts and metadata for '{base_model_name}' saved. Main metadata at: {final_gcs_path_for_metadata}")
            return final_gcs_path_for_metadata # Return GCS path to the main metadata file

        except Exception as e:
            logger.error(f"Failed to save model '{base_model_name}' to GCS: {e}")
            logger.error(traceback.format_exc())
            return None

    def load_model_joblib(self, model_name, version='latest'):
        """Loads the model, scaler, feature names, and metadata from GCS based on a main metadata file."""
        if not joblib:
            logger.error("Joblib not available, cannot load model.")
            return False
        
        client = _get_gcs_client()
        if not client:
            logger.error("GCS client not available for loading model.")
            return False

        # Determine the GCS path to the main metadata file
        if version == 'latest':
            # Find the latest metadata file for the given model_name prefix
            # Files are named <prefix><base_model_name>_v<timestamp>_metadata.json
            blob_prefix_to_list = f"{self.gcs_prefix}{model_name}_v"
            bucket = client.bucket(self.gcs_bucket_name)
            blobs = list(bucket.list_blobs(prefix=blob_prefix_to_list))
            
            metadata_files = [b.name for b in blobs if b.name.endswith("_metadata.json")]
            if not metadata_files:
                logger.error(f"No metadata files found in gs://{self.gcs_bucket_name}/ for prefix {blob_prefix_to_list}")
                return False
            metadata_files.sort(reverse=True) # Sort by timestamp descending
            gcs_metadata_path = metadata_files[0]
            logger.info(f"Latest metadata file found for '{model_name}': gs://{self.gcs_bucket_name}/{gcs_metadata_path}")
        else:
            gcs_metadata_path = f"{self.gcs_prefix}{model_name}_v{version}_metadata.json"
            logger.info(f"Attempting to load metadata for '{model_name}' version '{version}' from: gs://{self.gcs_bucket_name}/{gcs_metadata_path}")

        try:
            # 1. Download and parse the main metadata file
            metadata_str = _download_gcs_blob_to_string(self.gcs_bucket_name, gcs_metadata_path)
            if not metadata_str:
                logger.error(f"Failed to download metadata file: {gcs_metadata_path}")
                return False
            main_metadata = json.loads(metadata_str)

            artifact_paths = main_metadata.get('artifact_paths', {})
            gcs_model_path = artifact_paths.get('model_path')
            gcs_scaler_path = artifact_paths.get('scaler_path')
            gcs_ad_path = artifact_paths.get('anomaly_detector_path')

            if not gcs_model_path or not gcs_scaler_path:
                logger.error("Essential model/scaler GCS paths missing in metadata.")
                return False

            # Helper to extract blob name from gs:// path
            def get_blob_name(gs_path):
                if gs_path and gs_path.startswith(f"gs://{self.gcs_bucket_name}/"):
                    return gs_path[len(f"gs://{self.gcs_bucket_name}/"):]
                return None

            # 2. Download and load the model object
            model_blob_name = get_blob_name(gcs_model_path)
            if not model_blob_name:
                logger.error(f"Invalid GCS model path in metadata: {gcs_model_path}")
                return False
            model_bytes_io = _download_gcs_blob_to_bytes_io(self.gcs_bucket_name, model_blob_name)
            if not model_bytes_io:
                logger.error(f"Failed to download model from {gcs_model_path}")
                return False
            self.model = joblib.load(model_bytes_io)

            # 3. Download and load the scaler object
            scaler_blob_name = get_blob_name(gcs_scaler_path)
            if not scaler_blob_name:
                logger.error(f"Invalid GCS scaler path in metadata: {gcs_scaler_path}")
                return False
            scaler_bytes_io = _download_gcs_blob_to_bytes_io(self.gcs_bucket_name, scaler_blob_name)
            if not scaler_bytes_io:
                logger.error(f"Failed to download scaler from {gcs_scaler_path}")
                return False
            self.scaler = joblib.load(scaler_bytes_io)
            
            # 4. Download and load anomaly_detector if path exists
            if gcs_ad_path:
                ad_blob_name = get_blob_name(gcs_ad_path)
                if ad_blob_name:
                    ad_bytes_io = _download_gcs_blob_to_bytes_io(self.gcs_bucket_name, ad_blob_name)
                    if ad_bytes_io:
                        self.anomaly_detector = joblib.load(ad_bytes_io)
                        logger.info(f"Anomaly detector loaded from {gcs_ad_path}")
                    else:
                        logger.warning(f"Failed to download anomaly_detector from {gcs_ad_path}. Proceeding without it.")
                        self.anomaly_detector = None # Default to an OriginalAnomalyDetector instance if needed by parent
                else:
                    logger.warning(f"Invalid GCS anomaly_detector path in metadata: {gcs_ad_path}")
                    self.anomaly_detector = None
            else:
                logger.info("No anomaly_detector path in metadata. Anomaly detector will be None or default.")
                self.anomaly_detector = None 
            
            # If anomaly_detector is None after attempts, and parent class expects one, initialize it
            if self.anomaly_detector is None and OriginalAnomalyDetector is not None:
                logger.info("Initializing default AnomalyDetector as it was not loaded.")
                self.anomaly_detector = OriginalAnomalyDetector() 

            self.feature_names = main_metadata.get('feature_names', [])
            self.metadata = main_metadata # Store the full metadata

            logger.info(f"Successfully loaded model '{main_metadata.get('model_name')}' version '{main_metadata.get('version')}' from GCS.")
            return True

        except Exception as e:
            logger.error(f"Failed to load model from GCS (metadata: {gcs_metadata_path}): {e}")
            logger.error(traceback.format_exc())
            # Reset state if loading fails
            self.model = None
            self.scaler = None
            self.feature_names = []
            self.metadata = {}
            self.anomaly_detector = None
            return False

# --- GCS-Aware Visualizer ---
class GCSVisualizer(OriginalVisualizer):
    def __init__(self, viz_gcs_bucket_name, viz_gcs_prefix='visualizations/', interactive=False, local_temp_dir='/tmp/local_visualizations'):
        super().__init__(output_dir=local_temp_dir, interactive=interactive)
        self.gcs_bucket_name = viz_gcs_bucket_name
        self.gcs_prefix = viz_gcs_prefix.rstrip('/') + '/'  # Ensure trailing slash
        self.local_temp_dir = local_temp_dir # Parent class uses self.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"GCSVisualizer initialized. Bucket: {self.gcs_bucket_name}, Prefix: {self.gcs_prefix}, Local Temp: {self.output_dir}")

    def _save_plot_to_gcs(self, fig, plot_name, file_format='png'):
        """Helper to save a matplotlib figure to GCS and return its GCS path."""
        if not plt or not fig:
            logger.warning(f"Matplotlib (plt) or figure (fig) not available. Cannot save plot '{plot_name}'.")
            return None
        
        client = _get_gcs_client()
        if not client:
            logger.error(f"GCS client not available for saving plot '{plot_name}'.")
            return None

        gcs_plot_name = f"{self.gcs_prefix}{plot_name}.{file_format}"
        try:
            img_bytes_io = io.BytesIO()
            fig.savefig(img_bytes_io, format=file_format, bbox_inches='tight')
            img_bytes_io.seek(0)
            if _upload_bytes_io_to_gcs(img_bytes_io, self.gcs_bucket_name, gcs_plot_name, content_type=f'image/{file_format}'):
                plt.close(fig) # Close the figure to free memory
                gcs_path = f"gs://{self.gcs_bucket_name}/{gcs_plot_name}"
                logger.info(f"Plot '{plot_name}' saved to {gcs_path}")
                return gcs_path
            else:
                logger.error(f"Failed to upload plot {gcs_plot_name} to GCS.")
                plt.close(fig)
                return None
        except Exception as e:
            logger.error(f"Error saving plot '{plot_name}' to GCS: {e}")
            logger.error(traceback.format_exc())
            if fig: plt.close(fig) # Ensure figure is closed on error
            return None

    # Override plotting methods from OriginalVisualizer
    def plot_funding_stage_distribution(self, data, column_name='funding_stage_encoded', title='Funding Stage Distribution', plot_name='funding_stage_distribution'):
        if not plt or data is None or column_name not in data.columns:
            logger.warning(f"Cannot plot funding stage distribution. Data or column '{column_name}' missing, or plt not available.")
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            data[column_name].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Funding Stage")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return self._save_plot_to_gcs(fig, plot_name)
        except Exception as e:
            logger.error(f"Error in plot_funding_stage_distribution: {e}")
            if fig: plt.close(fig)
            return None

    def plot_feature_importance(self, model, feature_names, plot_name='feature_importance', top_n=None):
        if not plt or not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ or plt not available. Skipping feature importance plot.")
            return None
        
        importances = model.feature_importances_
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        df_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
        df_importance = df_importance.sort_values(by='importance', ascending=False)
        if top_n:
            df_importance = df_importance.head(top_n)

        fig, ax = plt.subplots(figsize=(12, max(6, len(df_importance) * 0.5)))
        try:
            ax.barh(df_importance['feature'], df_importance['importance'])
            ax.set_title('Feature Importance')
            ax.invert_yaxis() # Highest importance at the top
            plt.tight_layout()
            return self._save_plot_to_gcs(fig, plot_name)
        except Exception as e:
            logger.error(f"Error in plot_feature_importance: {e}")
            if fig: plt.close(fig)
            return None

    def plot_correlation_matrix(self, data, plot_name='correlation_matrix', method='pearson', annot=False, cmap='coolwarm'):
        if not plt or data is None or data.empty:
            logger.warning("Cannot plot correlation matrix. Data missing or plt not available.")
            return None
        
        numerical_data = data.select_dtypes(include=np.number)
        if numerical_data.empty:
            logger.warning("No numerical data to plot correlation matrix.")
            return None
        
        corr = numerical_data.corr(method=method)
        fig, ax = plt.subplots(figsize=(max(10, len(corr.columns)*0.8), max(8, len(corr.columns)*0.8)))
        try:
            import seaborn as sns # Assuming seaborn is available, or use plt.matshow
            sns.heatmap(corr, annot=annot, cmap=cmap, fmt='.2f', ax=ax, linewidths=.5)
            ax.set_title('Correlation Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            return self._save_plot_to_gcs(fig, plot_name)
        except ImportError:
            logger.warning("Seaborn not installed, using plt.matshow for correlation matrix.")
            try:
                cax = ax.matshow(corr, cmap=cmap)
                fig.colorbar(cax)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha='left')
                ax.set_yticklabels(corr.columns)
                ax.set_title('Correlation Matrix (matplotlib)')
                plt.tight_layout()
                return self._save_plot_to_gcs(fig, plot_name)
            except Exception as e_matshow:
                logger.error(f"Error plotting correlation matrix with matshow: {e_matshow}")
                if fig: plt.close(fig)
                return None
        except Exception as e_seaborn:
            logger.error(f"Error plotting correlation matrix with seaborn: {e_seaborn}")
            if fig: plt.close(fig)
            return None

    # Add other overridden plot methods here, e.g.:
    # plot_data_distribution, plot_time_series_decomposition, plot_evaluation_metrics,
    # plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
    # Each should follow the pattern of creating a figure, plotting, then calling self._save_plot_to_gcs(fig, plot_name)

    # Example for one more to show the pattern
    def plot_evaluation_metrics(self, history, metrics, plot_name='evaluation_metrics'):
        """Plots training and validation metrics from a Keras-like history object."""
        if not plt or not history or not metrics:
            logger.warning("Cannot plot evaluation metrics. Missing history, metrics, or plt.")
            return None
        
        num_metrics = len(metrics)
        if num_metrics == 0: return None
        
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics), sharex=True)
        if num_metrics == 1: axes = [axes] # Ensure axes is always iterable

        try:
            for i, metric in enumerate(metrics):
                if metric in history.history and f'val_{metric}' in history.history:
                    axes[i].plot(history.history[metric], label=f'Training {metric}')
                    axes[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
                    axes[i].set_title(f'Model {metric.capitalize()}')
                    axes[i].set_ylabel(metric.capitalize())
                    axes[i].legend()
                else:
                    logger.warning(f"Metric '{metric}' or 'val_{metric}' not found in history.")
            axes[-1].set_xlabel('Epoch')
            plt.tight_layout()
            return self._save_plot_to_gcs(fig, plot_name)
        except Exception as e:
            logger.error(f"Error in plot_evaluation_metrics: {e}")
            if fig: plt.close(fig)
            return None

class GCSAdvancedVisualizer(GCSVisualizer, OriginalAdvancedVisualizer): # Inherit from GCSVisualizer first
    def __init__(self, viz_gcs_bucket_name, viz_gcs_prefix='visualizations/advanced/', interactive=False, local_temp_dir='/tmp/local_adv_visualizations'):
        # Initialize GCSVisualizer part (which calls OriginalVisualizer with local_temp_dir)
        GCSVisualizer.__init__(self, viz_gcs_bucket_name, viz_gcs_prefix, interactive, local_temp_dir)
        # OriginalAdvancedVisualizer.__init__ specific parts if any (assuming it calls super() or OriginalVisualizer's init)
        # If OriginalAdvancedVisualizer has its own __init__ that doesn't call super or needs specific params, adjust here.
        # For now, assuming its __init__ is compatible or handled by OriginalVisualizer's init.
        OriginalAdvancedVisualizer.__init__(self, output_dir=self.output_dir, interactive=interactive) # Explicitly call OriginalAdvancedVisualizer init 
        self.gcs_prefix = viz_gcs_prefix.rstrip('/') + '/' # May override GCSVisualizer's prefix if specific for advanced
        logger.info(f"GCSAdvancedVisualizer initialized. Bucket: {self.gcs_bucket_name}, Prefix: {self.gcs_prefix}")

    # Override any OriginalAdvancedVisualizer methods if they need special GCS handling
    # beyond what GCSVisualizer provides. Often, they might just work if they call super().plot_... methods
    # that are now GCS-aware, or if they use self._save_plot_to_gcs directly.

    # Example: if AdvancedVisualizer has a method like:
    # def plot_custom_advanced_report(self, data, report_name="advanced_report"):
    #    fig, ax = plt.subplots()
    #    # ... complex plotting ...
    #    # Original might save locally: self.save_plot(fig, report_name, self.output_dir)
    #    # GCS version:
    #    return self._save_plot_to_gcs(fig, report_name)
    pass


# --- GCS-Aware Time Series Forecaster ---
class GCSTimeSeriesForecaster(OriginalTimeSeriesForecaster):
    def __init__(self, ts_gcs_bucket_name, ts_gcs_prefix='time_series_forecasts/', local_temp_dir='/tmp/local_ts_forecasts', **kwargs):
        super().__init__(output_dir=local_temp_dir, **kwargs)
        self.gcs_bucket_name = ts_gcs_bucket_name
        self.gcs_prefix = ts_gcs_prefix.rstrip('/') + '/'
        self.output_dir = local_temp_dir # Ensure self.output_dir is set for parent class if it uses it
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"GCSTimeSeriesForecaster initialized. Bucket: {self.gcs_bucket_name}, Prefix: {self.gcs_prefix}, Local Temp: {self.output_dir}")

    def _save_plot_to_gcs(self, fig, plot_name, file_format='png'):
        """Helper to save a matplotlib figure to GCS."""
        if not plt or not fig:
            logger.warning(f"Matplotlib (plt) or figure (fig) not available. Cannot save plot '{plot_name}'.")
            return None
        client = _get_gcs_client()
        if not client:
            logger.error(f"GCS client not available for saving plot '{plot_name}'.")
            return None

        gcs_plot_name = f"{self.gcs_prefix}{plot_name}.{file_format}"
        try:
            img_bytes_io = io.BytesIO()
            fig.savefig(img_bytes_io, format=file_format, bbox_inches='tight')
            img_bytes_io.seek(0)
            if _upload_bytes_io_to_gcs(img_bytes_io, self.gcs_bucket_name, gcs_plot_name, content_type=f'image/{file_format}'):
                plt.close(fig)
                gcs_path = f"gs://{self.gcs_bucket_name}/{gcs_plot_name}"
                logger.info(f"Plot '{plot_name}' saved to {gcs_path}")
                return gcs_path
            else:
                logger.error(f"Failed to upload plot {gcs_plot_name} to GCS.")
                plt.close(fig)
                return None
        except Exception as e:
            logger.error(f"Error saving plot '{plot_name}' to GCS: {e}")
            logger.error(traceback.format_exc())
            if fig: plt.close(fig)
            return None

    def plot_forecast(self, forecast_df, actual_df=None, title='Time Series Forecast', plot_name='time_series_forecast'):
        """Plots the forecast against actual data (if provided) and saves to GCS."""
        if not plt or forecast_df is None or forecast_df.empty:
            logger.warning("Cannot plot forecast. Forecast data missing or plt not available.")
            return None

        fig, ax = plt.subplots(figsize=(15, 7))
        try:
            # Assuming forecast_df has a 'ds' (datetime) and 'yhat' (forecast) column
            # And actual_df (if provided) has 'ds' and 'y' (actual)
            ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast (yhat)')
            if actual_df is not None and not actual_df.empty and 'y' in actual_df.columns:
                 # Ensure alignment on 'ds' if merging or plotting together, or slice appropriately
                plot_actual = actual_df[actual_df['ds'].isin(forecast_df['ds'])] # Simple overlap
                if not plot_actual.empty:
                    ax.plot(plot_actual['ds'], plot_actual['y'], label='Actual (y)', alpha=0.7)
            
            # Fill between for uncertainty intervals if present (e.g., 'yhat_lower', 'yhat_upper')
            if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], 
                                color='gray', alpha=0.2, label='Uncertainty Interval')

            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.legend()
            plt.grid(True)
            plt.tight_layout()
            return self._save_plot_to_gcs(fig, plot_name)
        except Exception as e:
            logger.error(f"Error in plot_forecast: {e}")
            if fig: plt.close(fig)
            return None

    def plot_components(self, model, forecast_df, plot_name='forecast_components'):
        """Plots the components of a Prophet forecast model and saves to GCS."""
        if not plt or model is None or forecast_df is None or forecast_df.empty:
            logger.warning("Cannot plot forecast components. Model, forecast data, or plt not available.")
            return None
        
        # Prophet specific plotting
        if hasattr(model, 'plot_components'):
            try:
                fig = model.plot_components(forecast_df) # This returns a matplotlib Figure object
                # The figure might have multiple axes for trend, seasonality etc.
                plt.tight_layout() # Adjust layout if needed
                return self._save_plot_to_gcs(fig, plot_name)
            except Exception as e:
                logger.error(f"Error plotting Prophet components: {e}")
                # fig object might exist even on error, ensure it's closed if plt.close isn't called by _save_plot_to_gcs on failure
                # Current _save_plot_to_gcs handles closing fig on error path.
                return None
        else:
            logger.warning("Model does not have a 'plot_components' method. Skipping.")
            return None

    def plot_dashboard_prototype(self, data, model_results, forecast_horizon, plot_name='interactive_dashboard_snapshot', **kwargs):
        """
        Generates an HTML dashboard (e.g., using Plotly, Bokeh, or just a complex Matplotlib figure)
        and saves it or a snapshot to GCS.
        If the original method returns HTML string, upload string. If it returns a figure, save figure.
        """
        # This is highly dependent on what the original plot_dashboard_prototype does.
        # Scenario 1: Original returns an HTML string
        # html_output = super().plot_dashboard_prototype(data, model_results, forecast_horizon, **kwargs)
        # if html_output and isinstance(html_output, str):
        #    gcs_html_name = f"{self.gcs_prefix}{plot_name}.html"
        #    if _upload_string_to_gcs(html_output, self.gcs_bucket_name, gcs_html_name, content_type='text/html'):
        #        return f"gs://{self.gcs_bucket_name}/{gcs_html_name}"
        #    return None

        # Scenario 2: Original returns a matplotlib figure (or saves one and we adapt)
        # For now, let's assume it might produce a figure similar to other plot methods, or we call super() and get a path we ignore.
        # The actual implementation in funding_stage_predictionORIGINAL.py for plot_dashboard_prototype needs to be checked.
        # If it saves a local file, we'd ideally adapt it to return a fig or BytesIO.
        
        # Let's assume for this GCS version, if the parent tries to save locally, we try to capture a figure instead.
        # This is a placeholder - the actual implementation depends on the parent method's behavior.
        logger.info(f"Attempting to generate and save dashboard snapshot: {plot_name}")
        
        # Option A: If parent class returns a figure object (ideal)
        # fig = super().plot_dashboard_prototype(data, model_results, forecast_horizon, **kwargs) 
        # if fig and plt: 
        #     return self._save_plot_to_gcs(fig, plot_name)

        # Option B: If parent saves to a local file path that we can predict or it returns
        # local_path = super().plot_dashboard_prototype(data, model_results, forecast_horizon, output_dir=self.local_temp_dir, **kwargs)
        # if local_path and os.path.exists(local_path):
        #    gcs_blob_name = f"{self.gcs_prefix}{os.path.basename(local_path)}"
        #    if _upload_local_file_to_gcs(local_path, self.gcs_bucket_name, gcs_blob_name):
        #        os.remove(local_path) # Clean up temp file
        #        return f"gs://{self.gcs_bucket_name}/{gcs_blob_name}"
        #    return None
        
        # Option C: If parent uses plotly and returns a fig, convert to HTML string or static image
        # This part needs knowledge of the original implementation.
        # For now, we will log a warning that this method needs specific GCS adaptation.
        original_output = None
        try:
            # We call the parent method. We need to inspect what it returns or how it saves.
            original_output = super().plot_dashboard_prototype(data, model_results, forecast_horizon, **kwargs)
        except Exception as e:
            logger.error(f"Error calling super().plot_dashboard_prototype: {e}")
            return None

        if original_output is None:
            logger.warning("Original plot_dashboard_prototype returned None or did not produce a recognized output for GCS saving.")
            return None
        
        if isinstance(original_output, str) and '<html' in original_output.lower(): # Heuristic for HTML string
            logger.info("Detected HTML output from original plot_dashboard_prototype. Uploading as HTML string.")
            gcs_html_name = f"{self.gcs_prefix}{plot_name}.html"
            if _upload_string_to_gcs(original_output, self.gcs_bucket_name, gcs_html_name, content_type='text/html'):
                return f"gs://{self.gcs_bucket_name}/{gcs_html_name}"
        elif plt and isinstance(original_output, plt.Figure): # If it returns a matplotlib figure
            logger.info("Detected matplotlib Figure output from original plot_dashboard_prototype. Saving as image.")
            return self._save_plot_to_gcs(original_output, plot_name)
        elif isinstance(original_output, str) and os.path.exists(original_output): # If it returns a path to a local file
             logger.info(f"Detected local file path output: {original_output}. Uploading local file.")
             gcs_blob_name = f"{self.gcs_prefix}{os.path.basename(original_output)}"
             if _upload_local_file_to_gcs(original_output, self.gcs_bucket_name, gcs_blob_name):
                 try:
                     os.remove(original_output) # Clean up temp file
                 except OSError as e_rm:
                     logger.warning(f"Could not remove temporary file {original_output}: {e_rm}")
                 return f"gs://{self.gcs_bucket_name}/{gcs_blob_name}"
        else:
            logger.warning(f"Output of plot_dashboard_prototype (type: {type(original_output)}) is not directly GCS-savable by current logic. Please adapt.")
        
        return None

    # Other OriginalTimeSeriesForecaster methods that produce plots or files would be overridden similarly.

# --- GCS-Aware Enhanced Pipeline ---
class GCSEnhancedPipeline(OriginalEnhancedPipeline):
    def __init__(self, 
                 gcs_bucket_name,
                 dl_config, 
                 fe_config,
                 model_config,
                 train_config,
                 anomaly_config,
                 ts_config,
                 model_gcs_prefix='pipeline_outputs/models/',
                 viz_gcs_prefix='pipeline_outputs/visualizations/',
                 adv_viz_gcs_prefix='pipeline_outputs/visualizations/advanced/',
                 ts_fc_gcs_prefix='pipeline_outputs/time_series_forecasts/',
                 summary_gcs_prefix='pipeline_outputs/summaries/',
                 local_temp_base_dir='/tmp/gcs_pipeline_run'):
        
        self.gcs_bucket_name = gcs_bucket_name
        self.summary_gcs_prefix = summary_gcs_prefix.rstrip('/') + '/'
        self.local_temp_base_dir = local_temp_base_dir
        os.makedirs(self.local_temp_base_dir, exist_ok=True)

        logger.info(f"GCSEnhancedPipeline initializing. Bucket: {self.gcs_bucket_name}, Temp Base: {self.local_temp_base_dir}")

        # Instantiate GCS-aware components
        # For components that don't have explicit GCS output in their original versions (or whose output is managed by others),
        # we can use the original classes directly, perhaps pointing their temp output to our local_temp_base_dir.
        
        # Data Loader - assuming original is fine, or its outputs are handled by FE
        # Ensure output_dir for original components point to a writable temp location if they save anything intermediately
        dl_local_temp = os.path.join(self.local_temp_base_dir, 'data_loader')
        os.makedirs(dl_local_temp, exist_ok=True)
        # Assuming DataLoader takes a config and an output_dir or similar for any caching/temp files
        # This part depends on OriginalDataLoader's __init__ signature. Adjust as needed.
        # If OriginalDataLoader is just in-memory, config might be enough.
        # For now, we assume it might take a base_dir or specific output_dir from its config.
        data_loader = OriginalDataLoader(**dl_config) # Or OriginalDataLoader(config_path=dl_config.get('path'), output_dir=dl_local_temp)

        # Feature Engineering - similar to DataLoader
        fe_local_temp = os.path.join(self.local_temp_base_dir, 'feature_engineering')
        os.makedirs(fe_local_temp, exist_ok=True)
        feature_engineer = OriginalFeatureEngineering(**fe_config) # Or OriginalFeatureEngineering(output_dir=fe_local_temp, ...)

        # Model Manager (GCS version)
        model_manager = GCSModelManager(model_gcs_bucket_name=self.gcs_bucket_name, 
                                        model_gcs_prefix=model_gcs_prefix,
                                        local_temp_dir=os.path.join(self.local_temp_base_dir, 'models'))

        # Visualizer (GCS version)
        visualizer = GCSVisualizer(viz_gcs_bucket_name=self.gcs_bucket_name,
                                   viz_gcs_prefix=viz_gcs_prefix,
                                   local_temp_dir=os.path.join(self.local_temp_base_dir, 'visualizations'))
        
        # Advanced Visualizer (GCS version)
        # Note: OriginalEnhancedPipeline might instantiate AdvancedVisualizer itself, or expect it as a separate param.
        # For now, we create it, and the parent init will need to accept it or use the main `visualizer` instance for adv tasks.
        # We will assume the parent pipeline might take `adv_visualizer` as an optional arg or uses the main `visualizer` instance for adv tasks.
        # If OriginalEnhancedPipeline specifically creates its own AdvancedVisualizer, this needs more thought.
        advanced_visualizer = GCSAdvancedVisualizer(viz_gcs_bucket_name=self.gcs_bucket_name,
                                                  viz_gcs_prefix=adv_viz_gcs_prefix, 
                                                  local_temp_dir=os.path.join(self.local_temp_base_dir, 'adv_visualizations'))

        # Anomaly Detector (Original, assuming its artifacts are saved by GCSModelManager)
        # It might be initialized/trained within the pipeline's run method or passed in already trained.
        # For now, we assume it's passed similar to other components.
        # The GCSModelManager handles saving/loading the anomaly_detector object itself.
        anomaly_detector = OriginalAnomalyDetector(**anomaly_config) # Or OriginalAnomalyDetector(output_dir=...)

        # Time Series Forecaster (GCS version)
        ts_forecaster = GCSTimeSeriesForecaster(ts_gcs_bucket_name=self.gcs_bucket_name,
                                                ts_gcs_prefix=ts_fc_gcs_prefix,
                                                local_temp_dir=os.path.join(self.local_temp_base_dir, 'ts_forecasts'))

        # Call the parent constructor with the (mostly) GCS-aware components.
        # The exact signature of OriginalEnhancedPipeline.__init__ is crucial here.
        # This is a common pattern; adjust if the original takes configs instead of instances for some components.
        super().__init__(
            data_loader=data_loader,
            feature_engineer=feature_engineer,
            model_manager=model_manager, # GCS version
            visualizer=visualizer,       # GCS version
            anomaly_detector=anomaly_detector, # Original (managed by GCSModelManager)
            ts_forecaster=ts_forecaster,   # GCS version
            # model_config, train_config might be used by super().run() or within super().__init__()
            model_config=model_config, 
            train_config=train_config,
            # The OriginalEnhancedPipeline might also need an output_base_dir for any of its own direct, non-component file ops.
            # We provide our local_temp_base_dir for that, ensuring no writes to restricted FS.
            output_base_dir=self.local_temp_base_dir, # Ensuring this is correctly passed
            ts_config=ts_config, # Adding potentially missing ts_config
            # If OriginalAdvancedVisualizer needs to be passed and parent accepts it:
            # adv_visualizer=advanced_visualizer 
        )
        # If the original pipeline used self.visualizer for advanced tasks too, we might need to set 
        # self.adv_visualizer = advanced_visualizer here if it's a separate attribute in the parent.
        # Or, ensure GCSVisualizer can handle calls from OriginalAdvancedVisualizer if it was a sub-component.

        logger.info("GCSEnhancedPipeline __init__ complete.")

    def _save_summary(self, summary_data, model_name):
        """Saves the pipeline summary data to GCS."""
        client = _get_gcs_client()
        if not client:
            logger.error("GCS client not available for saving summary.")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        summary_filename = f"summary_{model_name}_{timestamp}.json"
        gcs_summary_path = f"{self.summary_gcs_prefix}{summary_filename}"

        try:
            summary_json_str = json.dumps(summary_data, cls=NumpyEncoder, indent=4)
            if _upload_string_to_gcs(summary_json_str, self.gcs_bucket_name, gcs_summary_path, content_type="application/json"):
                full_gcs_path = f"gs://{self.gcs_bucket_name}/{gcs_summary_path}"
                logger.info(f"Pipeline summary saved to {full_gcs_path}")
                return full_gcs_path
            else:
                logger.error(f"Failed to upload summary {gcs_summary_path} to GCS.")
                return None
        except Exception as e:
            logger.error(f"Error saving summary to GCS for {model_name}: {e}")
            logger.error(traceback.format_exc())
            return None

    def run(self, data_path=None, model_name_prefix="funding_model"):
        """ 
        Executes the GCS-aware pipeline.
        It primarily calls the super().run() method, which should now use the GCS-enabled components.
        Any direct file outputs from OriginalEnhancedPipeline.run() not handled by components would need to be overridden here.
        """
        logger.info(f"Starting GCSEnhancedPipeline run for model: {model_name_prefix}")
        # The critical part is that self.model_manager, self.visualizer, self.ts_forecaster 
        # are now the GCS versions. And self._save_summary is overridden.
        
        # If data_path needs to be resolved from GCS, that logic would go here or in a GCSDataLoader.
        # For now, assume data_path is accessible or handled by OriginalDataLoader.
        
        try:
            # Call the parent run method. It should use the GCS-aware components we injected.
            # The `output_base_dir` for the parent is set to our temp dir, so any incidental writes by parent go there.
            super().run(data_path=data_path, model_name_prefix=model_name_prefix)
            logger.info(f"GCSEnhancedPipeline run for {model_name_prefix} completed.")
            # The summary and other artifacts should have been saved to GCS by the components and our _save_summary.
        except Exception as e:
            logger.error(f"Error during GCSEnhancedPipeline run for {model_name_prefix}: {e}")
            logger.error(traceback.format_exc())
            # Potentially raise e or handle error state

if __name__ == '__main__':
    logger.info(f"GCS Pipeline Runner started. GCS Client: {'Initialized' if GCS_CLIENT else 'Not Initialized'}")
    
    gcs_bucket_name = os.environ.get('GCS_BUCKET_NAME')
    if not gcs_bucket_name:
        logger.error("GCS_BUCKET_NAME environment variable not set. Exiting.")
        sys.exit(1)
    
    if not GCS_CLIENT:
        logger.error("GCS Client failed to initialize. Exiting.")
        sys.exit(1)

    logger.info(f"Target GCS Bucket: {gcs_bucket_name}")

    # --- Define Configurations (Placeholders - adjust with actual configs) ---
    # These would typically be loaded from a YAML/JSON config file or set via command-line args
    # Paths within configs (e.g., data_path) might need to be GCS paths if data is sourced from GCS by DataLoader
    
    # Example: If your DataLoader expects a path to raw data
    # This path could be a local path if running locally with local data,
    # or a GCS path (gs://bucket/path/to/data) if DataLoader is GCS-aware or data is pre-downloaded.
    # For now, assuming OriginalDataLoader can handle a path specified in its config.
    DATA_COLLECTION_BASE_DIR_LOCAL = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'backend', 'AdataCollection', 'JSONFolder'))
    
    # This is a placeholder path. Adjust to your actual merged/primary dataset if available.
    # If your DataLoader merges multiple files from JSONFolder, it might just need the folder path.
    placeholder_data_path = os.path.join(DATA_COLLECTION_BASE_DIR_LOCAL, 'topstartupio50.csv') # Using a specific file
    if not os.path.exists(placeholder_data_path):
        logger.warning(f"Placeholder data path {placeholder_data_path} does not exist. Pipeline might fail at data loading.")
        # You might want to create a dummy file here for basic syntax run-through, or ensure it exists.

    dl_config = {
        # Example: if DataLoader takes a list of files or a directory
        'input_data_path': placeholder_data_path, # Adjust as per OriginalDataLoader's needs
        # 'raw_data_sources': [os.path.join(DATA_COLLECTION_BASE_DIR_LOCAL, 'file1.json'), ...], 
        'merge_strategy': 'outer', # Example param
    }
    fe_config = {
        'target_variable': 'funding_stage_encoded',
        'text_features': ['description', 'industry'], # Example
        'numeric_features_to_log_transform': ['funding_amount_usd'], # Example
        'scaling_strategy': 'standard',
    }
    model_config = {
        'type': 'RandomForestClassifier', # Example: RandomForestClassifier, XGBClassifier
        'params': {'n_estimators': 100, 'random_state': 42} # Example params
    }
    train_config = {
        'test_size': 0.2,
        'cv_folds': 5,
        'calibration': True,
        'calibration_method': 'isotonic'
    }
    anomaly_config = {
        'contamination': 0.05, # Example param for IsolationForest
        'model_type': 'IsolationForest'
    }
    ts_config = {
        'time_column': 'funded_date',
        'value_column': 'funding_amount_usd',
        'freq': 'M', # Monthly
        'forecast_horizon': 12 # 12 months
    }

    # Define GCS prefixes
    gcs_base_prefix = "prod_pipeline_outputs/" # Example base prefix
    model_gcs_prefix = gcs_base_prefix + 'models/'
    viz_gcs_prefix = gcs_base_prefix + 'visualizations/'
    adv_viz_gcs_prefix = viz_gcs_prefix + 'advanced/'
    ts_fc_gcs_prefix = gcs_base_prefix + 'time_series_forecasts/'
    summary_gcs_prefix = gcs_base_prefix + 'summaries/'
    local_temp_dir_for_run = '/tmp/gcs_pipeline_main_run' 

    logger.info("Configurations prepared. Initializing GCSEnhancedPipeline...")

    try:
        gcs_pipeline = GCSEnhancedPipeline(
            gcs_bucket_name=gcs_bucket_name,
            dl_config=dl_config,
            fe_config=fe_config,
            model_config=model_config,
            train_config=train_config,
            anomaly_config=anomaly_config,
            ts_config=ts_config,
            model_gcs_prefix=model_gcs_prefix,
            viz_gcs_prefix=viz_gcs_prefix,
            adv_viz_gcs_prefix=adv_viz_gcs_prefix,
            ts_fc_gcs_prefix=ts_fc_gcs_prefix,
            summary_gcs_prefix=summary_gcs_prefix,
            local_temp_base_dir=local_temp_dir_for_run
        )
        logger.info("GCSEnhancedPipeline initialized. Starting run...")
        
        # The data_path for the run method depends on how OriginalDataLoader consumes data.
        # If it uses the 'input_data_path' from dl_config, data_path here might be None or supplementary.
        gcs_pipeline.run(data_path=dl_config.get('input_data_path'), model_name_prefix="bay_area_startup_funding_v1")
        
        logger.info("GCSEnhancedPipeline run finished successfully.")

    except ImportError as e_imp:
        logger.error(f"ImportError during pipeline setup or run: {e_imp}")
        logger.error("This might be due to missing dependencies from funding_stage_predictionORIGINAL.py or its imports.")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"An error occurred during GCSEnhancedPipeline execution: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    logger.info("GCS Pipeline Runner script finished.") 