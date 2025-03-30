import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import requests
import pickle
import os

# Assuming the model and dataset are defined
from trainModel import ConditionPredictorGAT  # Ensure this matches your model definition
from trainModel import PatientGraphDataset  # Ensure this matches your dataset definition

def load_model(model_path, num_node_features, num_conditions, device):
    """Load the trained model from the given path."""
    model = ConditionPredictorGAT(num_node_features, num_conditions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_one_patient(model, loader, device):
    """Generate predictions for a single patient."""
    model.eval()
    with torch.no_grad():
        for graph_batch, _, enc_indices, _ in loader:
            graph_batch = graph_batch.to(device)
            enc_indices = enc_indices.to(device)
            preds = model(graph_batch, enc_indices)
            preds = torch.sigmoid(preds)
            predicted_labels = (preds > 0.731).int()  # Threshold adjusted for better specificity
            return predicted_labels[0].cpu()  # Return only the first patient's predictions

def send_model_to_server(model, server_url):
    """Send the model weights to the central server."""
    model_weights = model.state_dict()
    # Serialize the model weights
    serialized_weights = pickle.dumps(model_weights)

    # Send the serialized model weights to the server
    response = requests.post(f"{server_url}/receive_weights", data=serialized_weights)
    if response.status_code == 200:
        print("Successfully sent model weights to the server.")
    else:
        print(f"Failed to send model weights. Server responded with: {response.status_code}")

def receive_federated_weights_from_server(server_url):
    """Receive federated weights from the central server."""
    response = requests.get(f"{server_url}/get_federated_model")
    if response.status_code == 200:
        # Deserialize the federated weights
        federated_weights = pickle.loads(response.content)
        return federated_weights
    else:
        print(f"Failed to receive federated weights. Server responded with: {response.status_code}")
        return None

def update_model_with_weights(model, federated_weights):
    """Update the model with the federated weights."""
    model.load_state_dict(federated_weights)
    print("Updated model with federated weights.")
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'best_model.pt'  # Path to your model weights
    dataset_path = 'patient_graphs.pt'  # Path to your dataset
    conditions_csv = 'md_data/conditions.csv'  # Path to conditions list
    batch_size = 16  # Number of patients to process per batch
    server_url = "http://central-server:5001"  # URL of the central server for federated learning
    
    print("Loading dataset...")
    # Assuming PatientGraphDataset is defined to read your dataset
    dataset = PatientGraphDataset(dataset_path, conditions_csv)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    
    print("Loading model...")
    # Determine the number of node features based on your dataset
    num_node_features = dataset[0][0].x.shape[1]
    model = load_model(model_path, num_node_features, dataset.num_conditions, device)
    
    print("Generating prediction for one patient...")
    prediction = predict_one_patient(model, loader, device)
    print("Predicted labels for one patient:")
    print(prediction)
    
    # Send model weights to the server
    send_model_to_server(model, server_url)
    
    # Wait for the federated weights from the server
    print("Waiting for federated weights from the central server...")
    federated_weights = receive_federated_weights_from_server(server_url)
    
    if federated_weights:
        # Update the model with the federated weights
        update_model_with_weights(model, federated_weights)
        testvar = ""
        # Re-predict after receiving federated model weights
        print("Generating new prediction after federated update...")
        new_prediction = predict_one_patient(model, loader, device)
        print("New predicted labels for one patient after federated update:")
        print(new_prediction)

if __name__ == "__main__":
    main()
