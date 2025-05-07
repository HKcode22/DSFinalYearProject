#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run script for Improved Funding Stage Prediction Dashboards
This script runs the improved dashboard implementation which fixes issues
and better organizes output directories.
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"improved_dashboard_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the improved dashboard implementation
try:
    from MLPredictiveAnalysis.improved_dashboard_implementation import ImprovedDashboardImplementation
except ImportError as e:
    logger.error(f"Error importing improved dashboard implementation: {str(e)}")
    sys.exit(1)

def run_improved_dashboards(base_dir=None, output_dir=None):
    """
    Run the improved dashboard implementation.
    
    Args:
        base_dir (str): Base directory for data
        output_dir (str): Output directory for dashboards
    """
    try:
        # Set default directories if not provided
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to get to the root project directory
            base_dir = os.path.dirname(base_dir)
            logger.info(f"Using base directory: {base_dir}")
        
        if output_dir is None:
            # Use a single organized output directory
            output_dir = os.path.join(base_dir, "FundingStageOutput")
            logger.info(f"Using output directory: {output_dir}")
        
        # Initialize dashboard implementation
        logger.info(f"Initializing improved dashboard implementation...")
        dashboard_impl = ImprovedDashboardImplementation(base_dir, output_dir)
        
        # Generate dashboards
        logger.info("Generating dashboards with better organization...")
        dashboard_paths = dashboard_impl.generate_dashboards()
        
        # Create additional calibration plots
        logger.info("Creating additional calibration plots...")
        processed_data, (X, y) = dashboard_impl.load_and_process_data()
        if processed_data is not None:
            classification_results = dashboard_impl.run_classification_models(X, y)
            dashboard_impl.create_calibration_plots(classification_results)
        
        if dashboard_paths:
            logger.info("Dashboard generation complete!")
            logger.info(f"All outputs saved to: {output_dir}")
            
            # Print summary of created files
            for root, dirs, files in os.walk(output_dir):
                rel_path = os.path.relpath(root, output_dir)
                if rel_path == '.':
                    logger.info(f"Main output directory: {len(files)} files, {len(dirs)} directories")
                else:
                    png_files = [f for f in files if f.endswith('.png')]
                    csv_files = [f for f in files if f.endswith('.csv')]
                    if png_files or csv_files:
                        logger.info(f"- {rel_path}: {len(png_files)} visualizations, {len(csv_files)} data files")
            
            return True
        else:
            logger.error("Failed to generate dashboards.")
            return False
    
    except Exception as e:
        logger.error(f"Error running improved dashboards: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Get command line arguments if provided
    base_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run dashboards
    success = run_improved_dashboards(base_dir, output_dir)
    
    # Print summary message
    if success:
        print("\n===============================================")
        print("✅ DASHBOARD GENERATION COMPLETED SUCCESSFULLY")
        print("===============================================")
        print(f"Output Directory: {output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'FundingStageOutput')}")
        print("Organized Structure:")
        print("  - /dashboards - All visualizations")
        print("    - /classification_dashboards - Model performance")
        print("      - /calibration_curves - Model calibration plots")
        print("      - /confusion_matrices - Confusion matrices")
        print("      - /model_comparison - Model comparison charts")
        print("    - /timeseries_dashboards - Time series forecasts") 
        print("      - /historical_vs_predicted - Historical vs. predicted comparisons")
        print("  - /data - Processed data files")
        print("  - /models - Trained models and metrics")
        print("  - /time_series_forecasts - Time series forecast data")
        print("===============================================")
    else:
        print("\n===============================================")
        print("❌ DASHBOARD GENERATION FAILED")
        print("===============================================")
        print("Please check the logs for details.")
        print("===============================================")
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 