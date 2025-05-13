from funding_stage_predictionORIGINAL import EnhancedPipeline
from google.cloud import storage
import os
import glob
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gcs_upload_orchestrator.log"), # Log to a file
        logging.StreamHandler() # Also log to console
    ]
)

GCS_BUCKET_NAME = "staging.oval-sunset-450610-h4.appspot.com"
# This is the directory where EnhancedPipeline saves its outputs.
# It's defined relative to the workspace root.
LOCAL_OUTPUT_BASE_DIR = "cs163-main/backend/MainOutput"
# This will be the root folder in your GCS bucket for these outputs.
GCS_DESTINATION_PREFIX = "MainOutput"

def upload_directory_to_gcs(local_path, bucket_name, gcs_path_prefix):
    """
    Recursively uploads files from a local directory to Google Cloud Storage.
    Retains the local directory structure under the gcs_path_prefix in the bucket.
    """
    try:
        storage_client = storage.Client()
        bucket_obj = storage_client.bucket(bucket_name)

        logging.info(f"Scanning directory: {local_path} for files to upload.")
        
        # Ensure the local_path exists
        if not os.path.isdir(local_path):
            logging.error(f"Local directory {local_path} not found. Cannot upload.")
            return

        # Walk through the directory
        for root, _, files in os.walk(local_path):
            for filename in files:
                local_file_path = os.path.join(root, filename)
                
                # Construct the destination blob name
                # Get the path relative to the base local_path
                relative_path = os.path.relpath(local_file_path, local_path)
                # Join with the GCS prefix
                destination_blob_name = os.path.join(gcs_path_prefix, relative_path)
                # Ensure GCS uses forward slashes for paths
                destination_blob_name = destination_blob_name.replace("\\\\", "/")

                blob = bucket_obj.blob(destination_blob_name)
                
                try:
                    blob.upload_from_filename(local_file_path)
                    logging.info(f"Successfully uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")
                except Exception as e:
                    logging.error(f"Failed to upload {local_file_path} to gs://{bucket_name}/{destination_blob_name}. Error: {e}")
        logging.info(f"Finished uploading contents of {local_path}.")

    except Exception as e:
        logging.error(f"Error during GCS upload process for directory {local_path}: {e}")
        logging.error("Please ensure you have authenticated with Google Cloud (e.g., via `gcloud auth application-default login`) and the bucket exists.")

def run_main_pipeline_and_upload():
    """
    Runs the main ML pipeline and then uploads its output to GCS.
    """
    logging.info("Starting pipeline execution and GCS upload orchestrator...")

    # --- Step 1: Run the original ML pipeline ---
    # The EnhancedPipeline's __init__ hardcodes its output_dir relative to the project root.
    # The base_dir for DataLoader is also specified relative to the project root.
    logging.info(f"Attempting to run the main ML pipeline. Outputs will be saved to '{LOCAL_OUTPUT_BASE_DIR}' locally.")
    
    pipeline_success = False
    try:
        # Paths for EnhancedPipeline constructor are relative to the workspace root.
        # The `EnhancedPipeline` class itself correctly sets its output_dir
        # to './cs163-main/backend/MainOutput'.
        # The `base_dir` is for the DataLoader component.
        pipeline = EnhancedPipeline(base_dir='cs163-main/backend/AdataCollection/JSONFolder')
        pipeline_success = pipeline.run()
    except Exception as e:
        logging.error(f"An error occurred while initializing or running the EnhancedPipeline: {e}", exc_info=True)
        pipeline_success = False

    if not pipeline_success:
        logging.error("The main ML pipeline did not complete successfully. Skipping GCS upload.")
        return

    logging.info("Main ML pipeline completed its local execution.")

    # --- Step 2: Upload the contents of LOCAL_OUTPUT_BASE_DIR to GCS ---
    logging.info(f"Starting upload of '{LOCAL_OUTPUT_BASE_DIR}' to GCS bucket '{GCS_BUCKET_NAME}' under prefix '{GCS_DESTINATION_PREFIX}'.")
    upload_directory_to_gcs(LOCAL_OUTPUT_BASE_DIR, GCS_BUCKET_NAME, GCS_DESTINATION_PREFIX)
    logging.info("GCS upload orchestration process completed.")

if __name__ == "__main__":
    run_main_pipeline_and_upload() 