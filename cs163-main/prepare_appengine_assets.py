import os
import glob
import shutil
from pathlib import Path

def find_latest_file_in_dir(directory, pattern):
    """Finds the latest file in a directory matching a glob pattern."""
    try:
        # Ensure directory is a Path object
        dir_path = Path(directory)
        search_pattern = pattern # Pattern is already relative to the directory in glob
        list_of_files = list(dir_path.glob(search_pattern))
        if not list_of_files:
            print(f"[Asset Prep] No files found matching: {pattern} in {directory}")
            return None
        latest_file = max(list_of_files, key=lambda f: f.stat().st_mtime)
        print(f"[Asset Prep] Found latest for {pattern} in {directory}: {latest_file.name}")
        return latest_file
    except Exception as e:
        print(f"[Asset Prep] Error finding latest file for {pattern} in {directory}: {e}")
        return None

def main():
    """Copies the latest required plot files from backend output to app assets."""
    print("[Asset Prep] Starting asset preparation for App Engine...")

    # Define paths relative to this script (assuming it's in cs163-main/)
    script_dir = Path(__file__).parent.resolve()
    
    appengine_assets_dir = script_dir / 'appengine' / 'assets'
    visualizations_source_dir = script_dir / 'backend' / 'MainOutput' / 'visualizations'
    prototype_source_dir = script_dir / 'backend' / 'MainOutput' / 'prototype_dashboard'

    # Ensure the target appengine/assets directory exists
    appengine_assets_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Asset Prep] Ensured App Engine assets directory exists: {appengine_assets_dir}")

    assets_to_sync = [
        # (Source Directory, Source Glob Pattern, Asset Type Name)
        (prototype_source_dir, 'bay_area_funding_trend_interactive_*.html', 'Forecast HTML'),
        (visualizations_source_dir, 'calibration_plot_Random_Forest_Tuned_*.png', 'RF Calibration Plot'),
        (visualizations_source_dir, 'model_comparison_accuracy_rmse_*.png', 'Model Comparison Plot'),
        (visualizations_source_dir, 'funding_stage_dist_*.png', 'Funding Dist Plot'),
        (visualizations_source_dir, 'feature_importance_RandomForestClassifier_*.png', 'RF Feature Importance Plot'),
        # Add other dynamic assets if needed. Static assets like logo.png should ideally already be in appengine/assets.
    ]

    copied_files_count = 0
    for src_dir_path, pattern, name in assets_to_sync:
        latest_src_file_path = find_latest_file_in_dir(src_dir_path, pattern)
        
        if latest_src_file_path:
            dest_filename = latest_src_file_path.name
            dest_path = appengine_assets_dir / dest_filename
            try:
                # Copy if dest doesn't exist or src is newer
                if not dest_path.exists() or latest_src_file_path.stat().st_mtime > dest_path.stat().st_mtime:
                    shutil.copy2(latest_src_file_path, dest_path)
                    print(f"[Asset Prep] Copied/Updated '{name}' to App Engine assets: {dest_filename}")
                    copied_files_count += 1
                else:
                    print(f"[Asset Prep] '{name}' ({dest_filename}) already up-to-date in App Engine assets.")
            except Exception as e:
                print(f"[Asset Prep] Failed to copy '{name}' from {latest_src_file_path} to {dest_path}: {e}")
        else:
            print(f"[Asset Prep] Warning: Latest source file for '{name}' (pattern: {pattern}) not found in {src_dir_path}.")

    # Note: This script does not handle the 'data-flow' or 'model-architecture' images specifically mentioned as
    # get_asset_url_path('data-flow') in app163.py, as they were not in the original assets_to_sync list.
    # If these are dynamic, add them to assets_to_sync. If static, ensure they are in appengine/assets.
    # The logo.png is also assumed to be directly in appengine/assets.

    print(f"[Asset Prep] Asset preparation finished. Copied/Updated {copied_files_count} files to {appengine_assets_dir}.")
    if copied_files_count < len(assets_to_sync):
        print("[Asset Prep] Warning: Not all expected assets were found or copied. Please check output above.")

if __name__ == '__main__':
    main() 