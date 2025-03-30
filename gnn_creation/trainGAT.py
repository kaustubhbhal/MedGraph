import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PatientGraphDataset(Dataset):
    def __init__(self, pt_file_path, conditions_csv):
        super().__init__()
        # Add safe globals before loading
        from torch_geometric.data.data import DataEdgeAttr
        torch.serialization.add_safe_globals([DataEdgeAttr])
        
        # Load with weights_only=False since we trust our data source
        self.patient_graphs = torch.load(pt_file_path, weights_only=False)
        
        self.condition_codes = pd.read_csv(conditions_csv)['CODE'].unique()
        self.code_to_idx = {code: i for i, code in enumerate(sorted(self.condition_codes))}
        self.idx_to_code = {i: code for i, code in enumerate(sorted(self.condition_codes))}
        self.num_conditions = len(self.condition_codes)
        
    def __len__(self):
        return len(self.patient_graphs)
    
    def __getitem__(self, idx):
        graph = self.patient_graphs[idx]
        
        # Get last encounter node (they're already ordered chronologically)
        num_timeline = graph.num_timeline.item()
        print("num_timeline:", num_timeline)
        last_encounter = num_timeline - 1  # Last encounter is at num_timeline-1
        print("last encounter:", last_encounter)
        
        # Create current condition vector (binary) and edge weights
        current_condition_vec, edge_weights = self._get_condition_vector_and_weights(graph, last_encounter)
        
        return graph, current_condition_vec, last_encounter, edge_weights
        
    def _get_condition_vector_and_weights(self, graph, encounter_idx):
        vec = torch.zeros(self.num_conditions, dtype=torch.float)
        weights = torch.zeros(self.num_conditions, dtype=torch.float)
        
        num_timeline = graph.num_timeline.item()
        num_medications = graph.num_medications.item()
        
        for edge_idx in range(graph.edge_index.shape[1]):
            src, dst = graph.edge_index[:, edge_idx]
            
            # Get edge type from edge_attr
            edge_type = graph.edge_attr[edge_idx].item() if hasattr(graph, 'edge_attr') else 0
            
            # Handle condition connections to our target encounter
            if src == encounter_idx and dst >= (num_timeline + num_medications):
                condition_code = int(graph.x[dst, 0].item())
                condition_idx = self.code_to_idx.get(str(condition_code), -1)
                
                if condition_idx == -1:
                    continue  # Skip conditions not in our CSV
                
                if edge_type == 1.0:  # START
                    vec[condition_idx] = 1
                    weights[condition_idx] = 1
                elif edge_type == 2.0:  # END
                    vec[condition_idx] = 0
                    weights[condition_idx] = 2
                elif edge_type == 0:  # ACTIVE
                    # Check time range for active conditions
                    encounter_time = graph.x[encounter_idx, 0].item()
                    condition_start = graph.x[dst, 1].item()
                    condition_end = graph.x[dst, 2].item()
                    if condition_start <= encounter_time <= condition_end:
                        vec[condition_idx] = 1
                        weights[condition_idx] = 0
                    
            elif dst == encounter_idx and src >= (num_timeline + num_medications):
                condition_code = int(graph.x[src, 0].item())
                condition_idx = self.code_to_idx.get(str(condition_code), -1)
                
                if condition_idx == -1:
                    continue
                
                if edge_type == 2.0:  # END
                    vec[condition_idx] = 0
                    weights[condition_idx] = 2
        
        return vec, weights

class ConditionPredictorGAT(nn.Module):
    def __init__(self, num_node_features, num_conditions, hidden_dim=128):
        super().__init__()
        self.num_conditions = num_conditions
        
        # GAT layers
        self.gat1 = GATConv(num_node_features, hidden_dim, heads=4, edge_dim=1)
        self.gat2 = GATConv(hidden_dim*4, hidden_dim, heads=2, edge_dim=1)
        self.gat3 = GATConv(hidden_dim*2, hidden_dim, heads=1, edge_dim=1)
        
        # Prediction heads
        self.start_probs_head = nn.Sequential(
            nn.Linear(hidden_dim + num_conditions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_conditions),
            nn.Sigmoid())
            
        self.end_probs_head = nn.Sequential(
            nn.Linear(hidden_dim + num_conditions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_conditions),
            nn.Sigmoid())

    def forward(self, data, last_encounter_idx, edge_weights):
        x, edge_index = data.x, data.edge_index
        
        # Initialize edge attributes
        edge_attr = torch.ones(edge_index.size(1), device=x.device)  # Ensure non-zero edge weights
        
        # Get node counts (ensure scalar values)
        num_timeline = data.num_timeline[0].item() if isinstance(data.num_timeline, torch.Tensor) else data.num_timeline
        num_medications = data.num_medications[0].item() if isinstance(data.num_medications, torch.Tensor) else data.num_medications
        
        # Process condition edges
        src, dst = edge_index
        condition_mask = (dst >= (num_timeline + num_medications))
        
        for edge_idx in torch.where(condition_mask)[0]:
            condition_code = int(data.x[dst[edge_idx], 0].item())
            condition_idx = self.code_to_idx.get(str(condition_code), -1)
            if condition_idx != -1:
                edge_attr[edge_idx] = edge_weights[condition_idx]
        
        # Graph encoding
        x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr.unsqueeze(1)))
        x = F.elu(self.gat2(x, edge_index, edge_attr=edge_attr.unsqueeze(1)))
        x = F.elu(self.gat3(x, edge_index, edge_attr=edge_attr.unsqueeze(1)))
        
        # Get last encounter embedding
        last_enc_embed = x[last_encounter_idx]
        
        # Ensure proper dimensions for concatenation
        if last_enc_embed.dim() == 1:
            last_enc_embed = last_enc_embed.unsqueeze(0)  # [1, hidden_dim]
        
        # Make edge_weights match dimension
        if edge_weights.dim() == 1:
            edge_weights = edge_weights.unsqueeze(0)  # [1, num_conditions]
        
        # Concatenate features
        combined = torch.cat([last_enc_embed, edge_weights], dim=1)  # [1, hidden_dim + num_conditions]
        
        # Calculate probabilities
        start_probs = self.start_probs_head(combined).squeeze()
        end_probs = self.end_probs_head(combined).squeeze()
        
        # Create outputs
        connected_mask = (edge_weights > 0).float().squeeze()
        prob_vector = torch.where(
            connected_mask.bool(),
            1 - end_probs,
            start_probs
        )
        
        prediction_vector = torch.round(prob_vector)
        change_vector = prediction_vector - edge_weights.squeeze()
        
        return torch.stack([prediction_vector, change_vector])

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        graph, current_vec, last_enc, edge_weights = batch
        graph = graph.to(device)
        current_vec = current_vec.to(device)
        edge_weights = edge_weights.to(device)
        
        # Convert last_enc to tensor if it isn't already
        if not isinstance(last_enc, torch.Tensor):
            last_enc = torch.tensor(last_enc, device=device)
        else:
            last_enc = last_enc.to(device)
        
        # Forward pass
        output = model(graph, last_enc, edge_weights)
        pred_vec, change_vec = output[0], output[1]
        
        # Calculate loss
        pred_loss = F.binary_cross_entropy(pred_vec, current_vec)
        change_loss = F.mse_loss(change_vec, torch.zeros_like(change_vec))
        change_weight = (current_vec.sum() + 1e-10) / (current_vec.numel() + 1e-10)  # Adaptive weight
        loss = pred_loss + (change_weight * change_loss)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_data, device):
    model.eval()
    change_metrics = {
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0
    }
    
    dataset_instance = test_data.dataset  # Get the dataset instance
    
    with torch.no_grad():
        for graph, current_vec, last_enc, edge_weights in test_data:
            graph = graph.to(device)
            current_vec = current_vec.to(device)
            edge_weights = edge_weights.to(device)
            
            if not isinstance(last_enc, torch.Tensor):
                last_enc = torch.tensor(last_enc, device=device)
                print(last_enc)
                print(last_enc[0].item())
            else:
                last_enc = last_enc.to(device)
                print(last_enc)
                print(last_enc[0].item())
                print(last_enc.shape)  


            
            num_timeline = graph.num_timeline[0].item()
            encounter_nodes = list(range(num_timeline))
            
            if len(encounter_nodes) < 2:
                continue  # Skip patients with insufficient encounters
                
            # Get ground truth for next encounter
            next_encounter = last_enc[0].item() + 1 if (last_enc[0].item() + 1) < num_timeline else last_enc[0].item()
            
            # **FIXED:** Call _get_condition_vector_and_weights from the dataset instance
            true_next, _ = dataset_instance.dataset._get_condition_vector_and_weights(graph, next_encounter)
            true_next = true_next.to(device)
            true_change = true_next - current_vec

            print(f"Non-zero elements in true change vector: {(true_change != 0).sum().item()}")

            
            # Make prediction
            output = model(graph, last_enc, edge_weights)
            pred_change = output[1].cpu()
            
            eval_mask = (true_change != 0)
            pred_changes = pred_change[eval_mask]
            true_changes = true_change[eval_mask]
            print(f"Predicted Change Vector: {pred_change}")
            print(f"True Change Vector: {true_change}")

            
            change_metrics['true_positives'] += ((pred_changes == true_changes) & (true_changes != 0)).sum().item()
            change_metrics['false_positives'] += ((pred_changes != 0) & (true_changes == 0)).sum().item()
            change_metrics['true_negatives'] += ((pred_changes == 0) & (true_changes == 0)).sum().item()
            change_metrics['false_negatives'] += ((pred_changes == 0) & (true_changes != 0)).sum().item()
    
    precision = change_metrics['true_positives'] / (change_metrics['true_positives'] + change_metrics['false_positives'] + 1e-10)
    recall = change_metrics['true_positives'] / (change_metrics['true_positives'] + change_metrics['false_negatives'] + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return {
        'change_precision': precision,
        'change_recall': recall,
        'change_f1': f1,
        'support': change_metrics['true_positives'] + change_metrics['false_negatives']
    }


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pt_file = 'patient_graphs.pt'
    conditions_csv = 'md_data/conditions.csv'
    batch_size = 32
    lr = 0.001
    epochs = 50
    
    # Load data
    print("Loading dataset...")
    dataset = PatientGraphDataset(pt_file, conditions_csv)
    train_data, val_data = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    
    train_loader = DataLoader(
        Subset(dataset, train_data),  # Use Subset instead of list comprehension
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        Subset(dataset, val_data),  # Use Subset instead of list comprehension
        batch_size=batch_size
    )
    
    # Initialize model
    print("Initializing model...")
    num_node_features = dataset[0][0].x.shape[1]
    model = ConditionPredictorGAT(
        num_node_features=num_node_features,
        num_conditions=dataset.num_conditions
    ).to(device)
    
    # Set code mappings
    model.code_to_idx = dataset.code_to_idx
    model.idx_to_code = dataset.idx_to_code
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    
    # Training loop
    print("Starting training...")
    best_val_f1 = 0
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step(val_metrics['change_f1'])
        
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Metrics:')
        print(f"  Precision: {val_metrics['change_precision']:.4f}")
        print(f"  Recall:    {val_metrics['change_recall']:.4f}")
        print(f"  F1:        {val_metrics['change_f1']:.4f}")
        print(f"  Support:   {val_metrics['support']}")
        
        if val_metrics['change_f1'] > best_val_f1:
            best_val_f1 = val_metrics['change_f1']
            torch.save(model.state_dict(), 'best_gat_model.pt')
            print("Saved new best model")
    
    print("\nTraining complete. Best validation F1: {:.4f}".format(best_val_f1))

if __name__ == "__main__":
    main()