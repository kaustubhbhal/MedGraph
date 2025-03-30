import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
ca_data = "ca_data/ca_data_patient_graphs.pt"
GRAPH_FILE = ca_data 
# "patient_graphs.pt"
CONDITION_SCHEMA_FILE = "condition_schema.json"
LAST_ENCOUNTER_JSON = "last_encounter_conditions.json"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def truncate_code(code):
    """Truncate condition code to the first 5 digits"""
    code_str = str(code)
    return code_str[:5] if len(code_str) > 5 else code_str

def get_node_type(index, data):
    num_timeline = data.num_timeline.item()
    num_medications = data.num_medications.item()
    return ("Encounter" if index < num_timeline else
            "Medication" if index < num_timeline + num_medications else
            "Condition")

def create_binary_targets(graphs, schema):
    """Create binary target vectors using truncated codes"""
    results = {}
    
    for graph_idx, data in enumerate(graphs):
        last_enc_idx = data.num_timeline.item() - 1
        condition_codes = []
        
        # Find conditions connected to last encounter
        for src, dst in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()):
            if dst == last_enc_idx and get_node_type(src, data) == "Condition":
                full_code = int(data.x[src][0].item())
                truncated_code = truncate_code(full_code)
                condition_codes.append(truncated_code)
        
        # Create binary vector using schema
        target_vector = torch.zeros(len(schema), dtype=torch.float)
        for code in condition_codes:
            if code in schema:
                target_vector[schema[code]] = 1.0
                
        results[str(graph_idx)] = {
            "last_encounter_index": last_enc_idx,
            "condition_codes": condition_codes,
            "target_vector": target_vector.tolist()
        }
    
    with open(LAST_ENCOUNTER_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

class DiseasePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index = data.x.to(DEVICE), data.edge_index.to(DEVICE)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)

        # Ensure `ptr` exists for batched data
        if hasattr(data, "ptr"):
            last_enc_indices = []
            for i in range(data.num_graphs):
                start_idx = data.ptr[i]
                timeline_size = data.num_timeline[i].item()
                last_enc_indices.append(start_idx + timeline_size - 1)

            last_enc_features = x[torch.tensor(last_enc_indices, device=DEVICE)]
        else:
            # For single-graph case
            last_enc_indices = data.num_timeline.item() - 1
            last_enc_features = x[last_enc_indices].unsqueeze(0)  # Ensure it's 2D

        return self.classifier(last_enc_features)

class GraphDataset(Dataset):
    def __init__(self, graphs, targets):
        self.graphs = graphs
        self.targets = targets
        
    def __len__(self):
        return len(self.graphs)
        
    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]

def train_binary_model(model, graphs, targets, epochs=100, lr=0.001):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()  # Fixed loss function for binary classification

    dataset = GraphDataset(graphs, targets)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_preds = []
        all_targets = []

        for batch_data, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(batch_data)  # Forward pass

            batch_targets = batch_targets.to(DEVICE).float()

            # Ensure shape match
            if outputs.shape != batch_targets.shape:
                batch_targets = batch_targets.view_as(outputs)

            loss = criterion(outputs, batch_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Calculate accuracy
            predictions = (outputs >= 0.5).float()
            correct_predictions += (predictions == batch_targets).sum().item()
            total_predictions += batch_targets.numel()

            total_loss += loss.item()
            
            # Collect predictions and targets for evaluation
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = (correct_predictions / total_predictions) * 100
        
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()
        
        precision = precision_score(all_targets, all_preds, average='binary', zero_division=1)
        recall = recall_score(all_targets, all_preds, average='binary', zero_division=1)
        f1 = f1_score(all_targets, all_preds, average='binary', zero_division=1)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")


if __name__ == "__main__":
    if not os.path.exists(CONDITION_SCHEMA_FILE) or os.path.getsize(CONDITION_SCHEMA_FILE) == 0:
        df = pd.read_csv("md_data/conditions_cleaned.csv")
        truncated_codes = {truncate_code(code) for code in df['CODE']}
        schema = {code: idx for idx, code in enumerate(sorted(truncated_codes))}
        with open(CONDITION_SCHEMA_FILE, 'w') as f:
            json.dump(schema, f, indent=2)
    else:
        with open(CONDITION_SCHEMA_FILE, 'r') as f:
            schema = json.load(f)
    
    output_dim = len(schema)
    
    graphs = torch.load(GRAPH_FILE, weights_only=False)
    target_data = create_binary_targets(graphs, schema)
    targets = torch.tensor([target_data[str(i)]['target_vector'] for i in range(len(graphs))])

    model = DiseasePredictor(
        input_dim=graphs[0].x.shape[1],
        hidden_dim=256,
        output_dim=output_dim
    ).to(DEVICE)

    train_binary_model(model, graphs, targets, epochs=100)
    torch.save(model.state_dict(), "temp_binary_disease_predictor.pth")
    print("Training complete with all fixes.")
