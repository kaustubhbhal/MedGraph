import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from trainGAT import ConditionPredictorGAT  # Ensure this matches your model definition
from trainGAT import PatientGraphDataset  # Ensure this matches your dataset definition

def load_model(model_path, num_node_features, num_conditions, device):
    model = ConditionPredictorGAT(num_node_features, num_conditions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for graph_batch, _, enc_indices, _ in loader:
            graph_batch = graph_batch.to(device)
            enc_indices = enc_indices.to(device)
            preds = model(graph_batch, enc_indices)
            preds = torch.sigmoid(preds)
            predicted_labels = (preds > 0.5).int()
            predictions.append(predicted_labels.cpu())
    return torch.cat(predictions, dim=0)

def predict_one_patient(model, loader, device):
    model.eval()
    with torch.no_grad():
        for graph_batch, _, enc_indices, _ in loader:
            graph_batch = graph_batch.to(device)
            enc_indices = enc_indices.to(device)
            preds = model(graph_batch, enc_indices)
            preds = torch.sigmoid(preds)
            predicted_labels = (preds > 0.731).int()
            return predicted_labels[0].cpu()  # Return only the first patient's predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'binary_disease_predictor.pth'
    dataset_path = 'patient_graphs.pt'
    conditions_csv = 'md_data/conditions.csv'
    batch_size = 16
    
    print("Loading dataset...")
    dataset = PatientGraphDataset(dataset_path, conditions_csv)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    
    print("Loading model...")
    num_node_features = dataset[0][0].x.shape[1]
    model = load_model(model_path, num_node_features, dataset.num_conditions, device)
    
    print("Generating prediction for one patient...")
    prediction = predict_one_patient(model, loader, device)
    
    print("Predicted labels for one patient:")
    print(prediction)

    
if __name__ == "__main__":
    main()
