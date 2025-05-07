import os
import logging
from MLPredictiveAnalysis.funding_continuation import FundingContinuationAnalysis

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('funding_continuation.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set up output directory
    output_dir = './outputContinuation'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize and run analysis
        analyzer = FundingContinuationAnalysis(output_dir=output_dir)
        analyzer.run_analysis()
        logging.info("Analysis script completed successfully")
        
    except Exception as e:
        logging.error(f"Error in analysis script: {str(e)}")
        logging.error("Traceback:", exc_info=True)

if __name__ == "__main__":
    main() 