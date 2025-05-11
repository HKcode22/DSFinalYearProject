import os
import logging
from MLPredictiveAnalysis.funding_stage_prediction9 import EnhancedPipeline

def test_pipeline():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("test_pipeline.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize pipeline with test directories
        base_dir = os.path.abspath("JSONFolder")
        output_dir = os.path.abspath("test_output")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize and run pipeline
        pipeline = EnhancedPipeline(base_dir=base_dir, output_dir=output_dir)
        success = pipeline.run()
        
        # Validate outputs
        if success:
            # Check if visualization files were created
            viz_dir = os.path.join(output_dir, "visualizations")
            expected_plots = [
                "funding_stage_dist_",
                "feature_importance_",
                "model_comparison_",
                "confusion_matrices_",
                "funding_vs_employees_",
                "feature_matrix_",
                "correlation_heatmap_",
                "temporal_trends_",
                "industry_analysis_",
                "advanced_correlations_",
                "feature_distributions_",
                "funding_patterns_",
                "pairwise_features_",
                "full_correlation_heatmap_",
                "violin_funding_by_stage_"
            ]
            
            missing_plots = []
            for plot in expected_plots:
                found = False
                for file in os.listdir(viz_dir):
                    if plot in file:
                        found = True
                        break
                if not found:
                    missing_plots.append(plot)
            
            if missing_plots:
                logger.warning(f"Missing visualizations: {missing_plots}")
            else:
                logger.info("All expected visualizations were created successfully")
            
            # Check if models were saved
            models_dir = os.path.join(output_dir, "models")
            if not os.path.exists(models_dir) or not os.listdir(models_dir):
                logger.warning("No models were saved in the output directory")
            else:
                logger.info(f"Models saved successfully in {models_dir}")
            
            return True
        else:
            logger.error("Pipeline execution failed")
            return False
            
    except Exception as e:
        logger.error(f"Error testing pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    test_pipeline() 