import torch
from collections import OrderedDict

class FederatedDiseasePredictor:
    def __init__(self):
        self.model_mapping = {}
        
    def federated_average(self, model_paths, output_path="global_disease_predictor.pth"):
        """Combine regional disease predictors into global model"""
        # Verify all models have same architecture
        sample_model = torch.load(model_paths[0])
        global_state = OrderedDict()
        
        # Initialize global model structure
        for key in sample_model.keys():
            global_state[key] = torch.zeros_like(sample_model[key])
        
        # Federated averaging
        for i, path in enumerate(model_paths):
            regional_model = torch.load(path)
            for key in regional_model.keys():
                global_state[key] += regional_model[key] / len(model_paths)
            
            # Track regional contributions
            self.model_mapping[f"region_{i}"] = {
                'path': path,
                'region': path.split('/')[0].upper(),  # Extract CA/CO/TX/MA
                'weights_shape': {k: v.shape for k,v in regional_model.items()}
            }
        
        # Save global model
        torch.save(global_state, output_path)
        return global_state

if __name__ == "__main__":
    aggregator = FederatedDiseasePredictor()
    
    # Your regional disease predictors
    regional_models = [
        "ca_data/ca_binary_disease_predictor.pth",
        "co_data/co_binary_disease_predictor.pth",
        "tx_data/tx_binary_disease_predictor.pth",
        "ma_data/ma_binary_disease_predictor.pth"
    ]
    
    # Create global model
    global_model = aggregator.federated_average(regional_models)
    
    # Print federation details
    print("Created global disease predictor from:")
    for region, details in aggregator.model_mapping.items():
        print(f"• {details['region']} model: {details['path']}")
        for param, shape in details['weights_shape'].items():
            print(f"  └ {param}: {shape}")
    
    print("\nGlobal model saved to 'global_disease_predictor.pth'")