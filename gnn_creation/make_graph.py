import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch_geometric.data import Data

# --------------------------
# Data Loading & Preparation
# --------------------------

def load_data():
    """Load all CSV files into DataFrames without using procedures."""
    patients = pd.read_csv("md_data/patients.csv")
    observations = pd.read_csv("md_data/observations.csv")
    encounters = pd.read_csv("md_data/encounters.csv")
    conditions = pd.read_csv("md_data/conditions.csv")
    medications = pd.read_csv("md_data/medications.csv")

    # Parse datetime fields for files with time info (encounters, conditions, medications)
    for df in [encounters, conditions, medications]:
        df["START_dt"] = df["START"].apply(lambda x: datetime.fromisoformat(x.replace("Z", "")))
        if "STOP" in df.columns:
            df["STOP_dt"] = df["STOP"].apply(lambda x: datetime.fromisoformat(x.replace("Z", "")) if pd.notnull(x) else None)

    return patients, observations, encounters, conditions, medications

def create_observation_mapping(observations_df):
    """Create mapping from observation codes to feature indices"""
    observation_codes = observations_df["CODE"].unique()
    return {code: i for i, code in enumerate(observation_codes)}, len(observation_codes)

# --------------------------
# Graph Construction
# --------------------------

def build_patient_graph(patient_id, patients_df, observations_df, encounters_df,
                        conditions_df, medications_df, code_to_idx, num_codes):
    """Build a PyG graph for a single patient using encounters only.
    
       Timeline events (nodes) are created based solely on the patient’s encounters,
       sorted by time. Temporal edges are added between successive timeline events.
    
       Medications are attached to timeline events based on the original encounter connection.
       Conditions are first attached using generic temporal overlaps. Then we add special 
       START and END edges for conditions.
    """
    # Get patient-specific data (assumes patients_df has a column "Id")
    pat_data = patients_df[patients_df["Id"] == patient_id]
    if pat_data.empty:
        return None
    pat_data = pat_data.iloc[0]

    # Filter data for the patient
    pat_encounters = encounters_df[encounters_df["PATIENT"] == patient_id].sort_values("START_dt")
    pat_observations = observations_df[observations_df["PATIENT"] == patient_id]
    pat_medications = medications_df[medications_df["PATIENT"] == patient_id]
    pat_conditions  = conditions_df[conditions_df["PATIENT"] == patient_id]

    # Skip patients with no encounters
    if pat_encounters.empty:
        return None

    # --------------------------
    # 1. Build Features for Timeline Events (Encounters Only)
    # --------------------------
    enc_ids = pat_encounters["Id"].tolist()
    # Initialize observation matrix and mask for encounters
    obs_matrix = np.full((len(enc_ids), num_codes), np.nan)
    mask = np.zeros((len(enc_ids), num_codes), dtype=np.float32)

    for idx, enc_id in enumerate(enc_ids):
        enc_obs = pat_observations[pat_observations["ENCOUNTER"] == enc_id]
        for _, row in enc_obs.iterrows():
            code_idx = code_to_idx[row["CODE"]]
            obs_matrix[idx, code_idx] = row["VALUE"]
            mask[idx, code_idx] = 1.0
    obs_matrix = np.nan_to_num(obs_matrix, 0.0)
    encounter_features = np.concatenate([obs_matrix, mask], axis=1)  # shape: (num_encounters, feat_dim)

    # --------------------------
    # 2. Build the Unified Timeline (Encounters Only)
    # --------------------------
    # Each timeline event is a tuple: (timestamp, event_type, feature_vector, external_id)
    timeline_events = []
    for i, (_, row) in enumerate(pat_encounters.iterrows()):
        timestamp = row["START_dt"]
        timeline_events.append((timestamp, "Encounter", encounter_features[i], row["Id"]))

    # Sort timeline events by timestamp
    timeline_events.sort(key=lambda x: x[0])

    timeline_features = []
    # Build a mapping from original encounter ID to new timeline index.
    timeline_encounter_mapping = {}
    for idx, (ts, typ, feat, ext_id) in enumerate(timeline_events):
        timeline_features.append(feat)
        if typ == "Encounter":
            timeline_encounter_mapping[ext_id] = idx
    timeline_features = torch.tensor(np.vstack(timeline_features), dtype=torch.float32)
    num_timeline = timeline_features.shape[0]

    # --------------------------
    # 3. Build Features for Medication & Condition Nodes
    # --------------------------
    med_features = []
    for _, med in pat_medications.iterrows():
        feature = np.zeros(timeline_features.shape[1], dtype=np.float32)
        feature[0] = float(med["CODE"])
        med_features.append(feature)
    medication_nodes = (torch.tensor(med_features, dtype=torch.float32)
                        if med_features else torch.empty((0, timeline_features.shape[1]), dtype=torch.float32))

    cond_features = []
    for _, cond in pat_conditions.iterrows():
        feature = np.zeros(timeline_features.shape[1], dtype=np.float32)
        feature[0] = float(cond["CODE"])
        cond_features.append(feature)
    condition_nodes = (torch.tensor(cond_features, dtype=torch.float32)
                       if cond_features else torch.empty((0, timeline_features.shape[1]), dtype=torch.float32))

    # --------------------------
    # 4. Enhanced Edge Construction
    # --------------------------
    edges = []
    edge_attrs = []
    med_offset = num_timeline  # medication nodes come after timeline nodes
    cond_offset = num_timeline + len(pat_medications)

    # (a) Temporal edges for timeline events:
    for i in range(num_timeline - 1):
        ts_current = timeline_events[i][0]
        ts_next    = timeline_events[i+1][0]
        t_diff = (ts_next - ts_current).total_seconds() / (3600 * 24)  # in days (can be fractional)
        edges.append((i, i+1))
        edge_attrs.append([t_diff])

    # (b) Medication -> Timeline edges:
    for med_idx, (_, med) in enumerate(pat_medications.iterrows()):
        encounter_id = med["ENCOUNTER"]
        if encounter_id in timeline_encounter_mapping:
            timeline_idx = timeline_encounter_mapping[encounter_id]
            edges.append((med_offset + med_idx, timeline_idx))
            edge_attrs.append([0.0])

    # (c) Generic Condition -> Timeline edges (temporal overlaps)
    connected_conditions = set()
    for cond_idx, (_, cond) in enumerate(pat_conditions.iterrows()):
        cond_start = cond["START_dt"]
        cond_stop = cond["STOP_dt"] if pd.notnull(cond["STOP_dt"]) else cond["START_dt"]
        connected = False
        for t_idx, (ts, _, _, _) in enumerate(timeline_events):
            if cond_start <= ts <= cond_stop:
                edges.append((cond_offset + cond_idx, t_idx))
                edge_attrs.append([0.0])
                connected = True
        if connected:
            connected_conditions.add(cond_idx)

    # For conditions with no generic connection, connect to nearest post-encounter
    for cond_idx in range(len(pat_conditions)):
        if cond_idx not in connected_conditions:
            cond = pat_conditions.iloc[cond_idx]
            cond_stop = cond["STOP_dt"] if pd.notnull(cond["STOP_dt"]) else cond["START_dt"]
            post_encounters = [(i, ts) for i, (ts, _, _, _) in enumerate(timeline_events) if ts > cond_stop]
            if post_encounters:
                post_encounters.sort(key=lambda x: x[1])
                t_idx = post_encounters[0][0]
            else:
                t_idx = len(timeline_events) - 1
            if t_idx >= 0:
                edges.append((cond_offset + cond_idx, t_idx))
                edge_attrs.append([0.0])

    # (d) Special START and END edges for each condition
    for cond_idx, (_, cond) in enumerate(pat_conditions.iterrows()):
        cond_start = cond["START_dt"]
        cond_stop = cond["STOP_dt"] if pd.notnull(cond["STOP_dt"]) else cond["START_dt"]
        
        # Find all timeline events within the valid window
        valid_indices = [t_idx for t_idx, (ts, _, _, _) in enumerate(timeline_events)
                        if cond_start <= ts <= cond_stop]

        # If none, choose the earliest encounter after cond_stop (or last encounter)
        if not valid_indices:
            post_encounters = [(i, ts) for i, (ts, _, _, _) in enumerate(timeline_events) if ts > cond_stop]
            if post_encounters:
                valid_indices = [min(post_encounters, key=lambda x: x[1])[0]]
            else:
                valid_indices = [num_timeline - 1]

        # Determine start and end indices (if only one, start and end are equal)
        start_idx = min(valid_indices)
        end_idx = max(valid_indices)

        # Remove ONLY the generic edges that are being replaced
        edges_to_remove = set((cond_offset + cond_idx, t_idx) for t_idx in [start_idx, end_idx])
        filtered_edges = []
        filtered_edge_attrs = []

        for edge, attr in zip(edges, edge_attrs):
            if edge not in edges_to_remove:  # Keep all other generic edges
                filtered_edges.append(edge)
                filtered_edge_attrs.append(attr)

        edges = filtered_edges
        edge_attrs = filtered_edge_attrs

        # Add START edge
        edges.append((cond_offset + cond_idx, start_idx))
        edge_attrs.append([1.0])  # Flag for START edge

        # Add END edge
        edges.append((cond_offset + cond_idx, end_idx))
        edge_attrs.append([2.0])  # Flag for END edge


    # (e) New edges: Medication-Medication temporal overlaps
    for i in range(len(pat_medications)):
        med_i = pat_medications.iloc[i]
        start_i = med_i["START_dt"]
        stop_i = med_i["STOP_dt"] or start_i
        for j in range(i+1, len(pat_medications)):
            med_j = pat_medications.iloc[j]
            start_j = med_j["START_dt"]
            stop_j = med_j["STOP_dt"] or start_j
            if (start_i <= stop_j) and (start_j <= stop_i):
                edges.append((med_offset + i, med_offset + j))
                edges.append((med_offset + j, med_offset + i))
                edge_attrs.extend([[0.0], [0.0]])

    # (f) New edges: Condition-Condition temporal overlaps
    for i in range(len(pat_conditions)):
        cond_i = pat_conditions.iloc[i]
        start_i = cond_i["START_dt"]
        stop_i = cond_i["STOP_dt"] or start_i
        for j in range(i+1, len(pat_conditions)):
            cond_j = pat_conditions.iloc[j]
            start_j = cond_j["START_dt"]
            stop_j = cond_j["STOP_dt"] or start_j
            if (start_i <= stop_j) and (start_j <= stop_i):
                edges.append((cond_offset + i, cond_offset + j))
                edges.append((cond_offset + j, cond_offset + i))
                edge_attrs.extend([[0.0], [0.0]])

    # (g) New edges: Condition-Medication temporal overlaps
    for cond_idx in range(len(pat_conditions)):
        cond = pat_conditions.iloc[cond_idx]
        c_start = cond["START_dt"]
        c_stop = cond["STOP_dt"] or c_start
        for med_idx in range(len(pat_medications)):
            med = pat_medications.iloc[med_idx]
            m_start = med["START_dt"]
            m_stop = med["STOP_dt"] or m_start
            if (c_start <= m_stop) and (m_start <= c_stop):
                edges.append((cond_offset + cond_idx, med_offset + med_idx))
                edges.append((med_offset + med_idx, cond_offset + cond_idx))
                edge_attrs.extend([[0.0], [0.0]])

    # --------------------------
    # 5. Combine All Components and Create PyG Data Object
    # --------------------------
    node_list = [timeline_features]
    if medication_nodes.shape[0] > 0:
        node_list.append(medication_nodes)
    if condition_nodes.shape[0] > 0:
        node_list.append(condition_nodes)
    x = torch.cat(node_list, dim=0)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = None

    num_encounters = sum(1 for (_, typ, _, _) in timeline_events if typ == "Encounter")
    num_medications = medication_nodes.shape[0]
    num_conditions  = condition_nodes.shape[0]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_timeline=torch.tensor([num_encounters], dtype=torch.long),
        num_encounters=torch.tensor([num_encounters], dtype=torch.long),
        num_medications=torch.tensor([num_medications], dtype=torch.long),
        num_conditions=torch.tensor([num_conditions], dtype=torch.long)
    )
    return data


# --------------------------
# Main Execution
# --------------------------

if __name__ == "__main__":
    # Load data and create observation mapping
    patients, observations, encounters, conditions, medications = load_data()
    code_to_idx, num_codes = create_observation_mapping(observations)

    patient_graphs = []
    for _, patient in patients.iterrows():
        graph = build_patient_graph(
            patient["Id"],
            patients,
            observations,
            encounters,
            conditions,
            medications,
            code_to_idx,
            num_codes
        )
        if graph is not None:
            patient_graphs.append(graph)
            print(f"Processed patient {patient['Id']} with {graph.x.size(0)} nodes")

    torch.save(patient_graphs, 'patient_graphs.pt')
    print(f"\nSuccessfully processed {len(patient_graphs)} patient graphs!")
