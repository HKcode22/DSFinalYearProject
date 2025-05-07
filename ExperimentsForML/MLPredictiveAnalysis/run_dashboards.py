#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run script for Funding Stage Prediction Dashboards
This script runs the dashboard implementation to generate visualizations
for funding stage prediction data.
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
        logging.FileHandler(f"dashboard_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the dashboard implementation
try:
    from MLPredictiveAnalysis.dashboard_implementation import DashboardImplementation
except ImportError as e:
    logger.error(f"Error importing dashboard implementation: {str(e)}")
    sys.exit(1)

def run_dashboards(base_dir=None, output_dir=None):
    """
    Run the dashboard implementation.
    
    Args:
        base_dir (str): Base directory for data
        output_dir (str): Output directory for dashboards
    """
    try:
        # Set default directories if not provided
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if output_dir is None:
            # Create timestamped output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(base_dir, f"dashboard_output_{timestamp}")
        
        # Initialize dashboard implementation
        logger.info(f"Initializing dashboard implementation with base_dir={base_dir}, output_dir={output_dir}")
        dashboard_impl = DashboardImplementation(base_dir, output_dir)
        
        # Generate dashboards
        logger.info("Generating dashboards...")
        dashboard_paths = dashboard_impl.generate_dashboards()
        
        if dashboard_paths:
            logger.info("Dashboard generation complete!")
            logger.info(f"Dashboards saved to: {output_dir}")
            return True
        else:
            logger.error("Failed to generate dashboards.")
            return False
    
    except Exception as e:
        logger.error(f"Error running dashboards: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Get command line arguments if provided
    base_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run dashboards
    success = run_dashboards(base_dir, output_dir)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 