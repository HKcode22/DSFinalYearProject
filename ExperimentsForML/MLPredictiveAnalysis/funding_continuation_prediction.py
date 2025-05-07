import os
import json
from datetime import datetime
from numpy_encoder import NumpyEncoder

class FundingContinuationPrediction:
    def _track_experiment(self, results, metadata):
        """Track experiment results and model versions"""
        experiment_dir = os.path.join(self.output_dir, 'experiments')
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create experiment record
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_data = {
            'timestamp': timestamp,
            'cox_metrics': results['cox'],
            'rsf_metrics': results['rsf'],
            'metadata': metadata,
            'feature_importance': {
                'cox': results['cox'].get('feature_importance', {}),
                'rsf': results['rsf'].get('feature_importance', {})
            }
        }
        
        # Save experiment results
        experiment_path = os.path.join(experiment_dir, f'experiment_{timestamp}.json')
        with open(experiment_path, 'w') as f:
            json.dump(experiment_data, f, indent=4, cls=NumpyEncoder)
        
        # Update experiment index
        index_path = os.path.join(experiment_dir, 'experiment_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = []
        
        index.append({
            'timestamp': timestamp,
            'cox_c_index': float(results['cox'].get('c_index', 0)),
            'rsf_c_index': float(results['rsf'].get('c_index', 0)),
            'data_size': metadata.get('data_size', 0)
        })
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=4, cls=NumpyEncoder)
        
        logger.info(f"Experiment results saved to {experiment_path}")
        
        # Track model versions
        self._update_model_registry(results, timestamp) 