import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.utils.data import Dataset, Subset
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

class PatientGraphDataset(Dataset):
    def __init__(self, pt_file_path, conditions_csv):
        super().__init__()
        self.patient_graphs = torch.load(pt_file_path)
        condition_df = pd.read_csv(conditions_csv)
        
        # Store codes as sorted integers for range searching
        self.condition_codes = sorted([int(code) for code in condition_df['CODE'].unique()])
        self.code_ranges = [(code-50, code+50) for code in self.condition_codes]
        
        # Create mapping from original codes to indices
        self.idx_to_code = {i: str(code) for i, code in enumerate(sorted(self.condition_codes))}
        self.code_to_idx = {str(code): i for i, code in enumerate(self.condition_codes)}
        self.num_conditions = len(self.condition_codes)
        
        #print(f"Loaded {self.num_conditions} condition codes")
        #print(f"Code range: {self.condition_codes[0]} to {self.condition_codes[-1]}")
    
    def find_nearest_code(self, query_code):
        """Find codes within ±50 range using binary search"""
        query = int(query_code)
        matches = []
        
        # Binary search for approximate matches
        left, right = 0, len(self.condition_codes)
        while left < right:
            mid = (left + right) // 2
            if self.condition_codes[mid] - 50 <= query <= self.condition_codes[mid] + 50:
                # Found potential match, expand to find all in range
                matches.append(self.condition_codes[mid])
                # Check left
                i = mid - 1
                while i >= 0 and self.condition_codes[i] >= query - 50:
                    if query - 50 <= self.condition_codes[i] <= query + 50:
                        matches.append(self.condition_codes[i])
                    i -= 1
                # Check right
                i = mid + 1
                while i < len(self.condition_codes) and self.condition_codes[i] <= query + 50:
                    if query - 50 <= self.condition_codes[i] <= query + 50:
                        matches.append(self.condition_codes[i])
                    i += 1
                break
            elif self.condition_codes[mid] < query - 50:
                left = mid + 1
            else:
                right = mid
        
        return matches
    # Add to your dataset class
    def analyze_labels(self):
        pos_counts = []
        for i in range(len(self)):
            _, vec, _, _ = self[i]
            pos_counts.append(vec.sum().item())
        
        self.pos_count = sum(pos_counts)
        self.total_labels = len(pos_counts) * self.num_conditions
        
        print("\n=== LABEL ANALYSIS ===")
        print(f"Total patients: {len(pos_counts)}")
        print(f"Total condition labels: {self.total_labels}")
        print(f"Positive labels: {self.pos_count} ({self.pos_count/self.total_labels:.2%})")
        print(f"Patients with ≥1 condition: {sum(c > 0 for c in pos_counts)}")

    def _find_code_collisions(self):
        """Identify which codes collide when truncated"""
        from collections import defaultdict
        code_groups = defaultdict(list)
        for code in self.condition_codes:
            code_groups[str(code)[:6]].append(str(code))
        
        print("\nCode collisions (same first 5 digits):")
        for short_code, group in code_groups.items():
            if len(group) > 1:
                print(f"{short_code}: {group}")
        
    def __len__(self):
        return len(self.patient_graphs)
    
    def __getitem__(self, idx):
        graph = self.patient_graphs[idx]
        
        # Get last encounter index
        num_timeline = graph.num_timeline.item()
        last_encounter = num_timeline - 1
        
        # Get condition vector and weights
        current_condition_vec, edge_weights = self._get_condition_vector_and_weights(graph, last_encounter)
        
        return graph, current_condition_vec, last_encounter, edge_weights
        
    def _get_condition_vector_and_weights(self, graph, encounter_idx):
        vec = torch.zeros(self.num_conditions, dtype=torch.float32)
        weights = torch.zeros(self.num_conditions, dtype=torch.float32)
        
        condition_offset = graph.num_timeline + graph.num_medications
        
        for i in range(graph.edge_index.size(1)):
            src, dst = graph.edge_index[:, i]
            edge_type = graph.edge_attr[i].item() if hasattr(graph, 'edge_attr') else 0
            
            if (src == encounter_idx and dst >= condition_offset) or \
            (dst == encounter_idx and src >= condition_offset):
                
                condition_node = dst if src == encounter_idx else src
                raw_code = int(graph.x[condition_node, 0].item())
                
                # Find all matching codes within ±50
                matched_codes = self.find_nearest_code(raw_code)
                
                if not matched_codes:
                    print(f"No match for code {raw_code} (min: {self.condition_codes[0]}, max: {self.condition_codes[-1]})")
                    continue
                    
                # Mark all matches as active
                for code in matched_codes:
                    condition_idx = self.code_to_idx[str(code)]
                    vec[condition_idx] = 1
                    weights[condition_idx] = 1 if edge_type == 1.0 else 2 if edge_type == 2.0 else 0
        
        active_count = vec.sum().item()
        #if active_count > 0:
            #print(f"Found {active_count} active conditions for encounter {encounter_idx}")
        
        return vec, weights
    
    def collate_fn(batch):
        """Proper collate function that handles Data objects correctly"""
        from torch_geometric.data import Batch
        
        # Unpack the batch components
        graphs = [item[0] for item in batch]
        current_vecs = torch.stack([item[1] for item in batch])
        last_encs = torch.tensor([item[2] for item in batch])
        edge_weights = torch.stack([item[3] for item in batch])
        
        # Batch the graphs using PyG's Batch
        try:
            graph_batch = Batch.from_data_list(graphs)
        except Exception as e:
            print(f"Batching error: {e}")
            raise RuntimeError("Failed to batch graphs - check your data types")
        
        return graph_batch, current_vecs, last_encs, edge_weights

class ConditionPredictorGAT(nn.Module):
    def __init__(self, num_node_features, num_conditions, hidden_dim=128):
        super().__init__()
        self.num_conditions = num_conditions
        
        # Graph attention layers
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=3, edge_dim=1)
        self.conv2 = GATConv(hidden_dim*3, hidden_dim, edge_dim=1)
        
        # Condition prediction head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_conditions)
        )
        
    def forward(self, data, encounter_indices):
        x, edge_index = data.x, data.edge_index
        edge_attr = torch.ones(edge_index.size(1), 1, device=x.device)
        
        # Graph processing
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        
        # Get encounter node embeddings
        encounter_embeddings = []
        batch = getattr(data, 'batch', torch.zeros(x.size(0), device=x.device))
        
        for i, enc_idx in enumerate(encounter_indices):
            # Get nodes belonging to this graph
            graph_nodes = (batch == i).nonzero().squeeze()
            encounter_embeddings.append(x[graph_nodes[enc_idx.item()]])
        
        x = torch.stack(encounter_embeddings)
        
        # Condition prediction
        return torch.sigmoid(self.classifier(x))

def collate_fn(batch):
    """Proper collate function that handles Data objects correctly"""
    from torch_geometric.data import Batch
    
    # Unpack the batch components
    graphs = [item[0] for item in batch]
    current_vecs = torch.stack([item[1] for item in batch])
    last_encs = torch.tensor([item[2] for item in batch])
    edge_weights = torch.stack([item[3] for item in batch])
    
    # Batch the graphs using PyG's Batch
    try:
        graph_batch = Batch.from_data_list(graphs)
    except Exception as e:
        print(f"Batching error: {e}")
        raise RuntimeError("Failed to batch graphs - check your data types")
    
    return graph_batch, current_vecs, last_encs, edge_weights

def get_condition_index(self, raw_code):
    """Robust code lookup with multiple fallbacks"""
    code_variants = [
        str(int(raw_code)),
        str(float(raw_code)),
        str(int(round(raw_code))),
        str(raw_code).split('.')[0]  # Integer part only
    ]
    
    for code in code_variants:
        if code in self.code_to_idx:
            return self.code_to_idx[code]
    
    # Try all mapping variants if default fails
    for mapping_name, mapping in self.code_mappings.items():
        for code in code_variants:
            if code in mapping:
                print(f"Found in {mapping_name} mapping: {raw_code} -> {code}")
                return mapping[code]
    
    return None

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    # First calculate class weights
    pos_count = 0
    total = 0
    for _, labels, _, _ in train_loader:
        pos_count += labels.sum().item()
        total += labels.numel()
    pos_weight = torch.tensor([total/(pos_count+1e-7)]).to(device)
    
    for graph_batch, labels, enc_indices, _ in tqdm(train_loader, desc="Training"):
        graph_batch = graph_batch.to(device)
        labels = labels.to(device)
        enc_indices = enc_indices.to(device)
        
        optimizer.zero_grad()
        preds = model(graph_batch, enc_indices)
        
        # Focal loss for class imbalance
        loss = focal_loss(preds, labels, pos_weight=pos_weight)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"\nTraining Stats:")
    print(f"Positive samples: {pos_count}/{total} ({pos_count/total:.2%})")
    return total_loss / len(train_loader)

def focal_loss(preds, targets, alpha=0.25, gamma=2, pos_weight=500.0):
    bce_loss = F.binary_cross_entropy_with_logits(
        preds, targets, reduction='none', pos_weight=pos_weight
    )
    pt = torch.exp(-bce_loss)
    focal_loss = (alpha * (1-pt)**gamma * bce_loss).mean()
    return focal_loss


def evaluate(model, loader, device):
    model.eval()
    metrics = {
        'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
        'total': 0, 'positives': 0,
        'loss': 0  # Added loss tracking
    }
    
    pos_weight = torch.tensor([100.0]).to(device)  # Adjust based on your data
    
    with torch.no_grad():
        for graph_batch, labels, enc_indices, _ in loader:
            graph_batch = graph_batch.to(device)
            labels = labels.to(device)
            enc_indices = enc_indices.to(device)
            
            preds = model(graph_batch, enc_indices)
            
            # Calculate loss
            loss = F.binary_cross_entropy_with_logits(
                preds, labels, pos_weight=pos_weight
            )
            metrics['loss'] += loss.item()
            
            # Dynamic threshold
            threshold = 0.1  # Start low for imbalanced data
            pred_labels = (torch.sigmoid(preds) > threshold).float()
            
            # Update metrics
            metrics['tp'] += ((pred_labels == 1) & (labels == 1)).sum().item()
            metrics['fp'] += ((pred_labels == 1) & (labels == 0)).sum().item()
            metrics['tn'] += ((pred_labels == 0) & (labels == 0)).sum().item()
            metrics['fn'] += ((pred_labels == 0) & (labels == 1)).sum().item()
            metrics['total'] += labels.numel()
            metrics['positives'] += labels.sum().item()
    
    # Calculate rates
    precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-10)
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    pos_rate = metrics['positives'] / metrics['total']
    
    return {
        'loss': metrics['loss'] / len(loader),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pos_rate': pos_rate,
        'support': metrics['tp'] + metrics['fn']
    }

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pt_file = 'patient_graphs.pt'
    conditions_csv = 'md_data/conditions.csv'
    batch_size = 16  # Reduced for memory
    lr = 0.001
    epochs = 30
    
    # Load data
    print("Loading dataset...")
    dataset = PatientGraphDataset(pt_file, conditions_csv)
    
    dataset.analyze_labels()
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("Initializing model...")
    num_node_features = dataset[0][0].x.shape[1]
    #print("ANALYSIS", analyze_graph(dataset[0][0]))
    model = ConditionPredictorGAT(num_node_features, dataset.num_conditions).to(device)
    
    # Set code mappings
    model.code_to_idx = dataset.code_to_idx
    model.idx_to_code = dataset.idx_to_code
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    best_f1 = 0
    patience = 10
    patience_counter = 0
    # Training loop
    print("Starting training...")
    best_f1 = 0
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        # In your main training loop, replace the print statement with:
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        val_metrics = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Metrics:")
        print(f"- Precision: {val_metrics['precision']:.4f}")
        print(f"- Recall:    {val_metrics['recall']:.4f}")
        print(f"- F1:        {val_metrics['f1']:.4f}")
        print(f"- Pos Rate:  {val_metrics['pos_rate']:.4f}")
        print(f"- Support:   {val_metrics['support']}")
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        scheduler.step(val_metrics['f1'])
    
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")

def analyze_graph(graph):
    print("\nGraph Analysis:")
    print(f"Total nodes: {graph.x.size(0)}")
    print(f"Timeline nodes: {graph.num_timeline.item()}")
    print(f"Medication nodes: {graph.num_medications.item()}")
    print(f"Condition nodes: {graph.num_conditions.item()}")
    
    # Check edge types
    if hasattr(graph, 'edge_attr'):
        unique_edge_types = torch.unique(graph.edge_attr)
        print(f"Edge types present: {unique_edge_types.tolist()}")
    
    # Check condition nodes
    condition_offset = graph.num_timeline.item() + graph.num_medications.item()
    condition_codes = graph.x[condition_offset:, 0].unique()
    print(f"Unique condition codes: {condition_codes.tolist()[:6]}...")



if __name__ == "__main__":
    main()