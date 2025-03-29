import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import os
from collections import defaultdict

# Configuration
DATA_DIR = Path("Maryland_Cleaned")
OUTPUT_DIR = Path("encounter_centric_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load all cleaned CSV files into DataFrames with error handling"""
    print("Loading data...")
    data_files = {
        'patients': "patients_cleaned.csv",
        'conditions': "conditions_cleaned.csv",
        'medications': "medications_cleaned.csv",
        'observations': "observations_cleaned.csv",
        'procedures': "procedures_cleaned.csv",
        'encounters': "encounters_cleaned.csv"
    }
    
    data = {}
    for name, file in data_files.items():
        try:
            data[name] = pd.read_csv(DATA_DIR / file)
            print(f"✓ {name:12} | {len(data[name]):6} records")
        except Exception as e:
            print(f"✕ Error loading {file}: {str(e)}")
            data[name] = pd.DataFrame()
    return data

def create_encounter_centric_graph(data):
    """Create graph where all medical entities connect through encounters"""
    print("\nBuilding encounter-centric graph...")
    G = nx.Graph()
    
    # Add patients
    for _, patient in data['patients'].iterrows():
        G.add_node(f"Patient_{patient['Id']}", 
                  type="patient",
                  label=f"{patient['FIRST']} {patient['LAST']}",
                  gender=patient['GENDER'],
                  color='#ff6b6b',
                  size=300)
    
    # Dictionary to store encounter connections
    encounter_map = defaultdict(list)
    
    # Process all medical entities
    entity_types = {
        'conditions': ('Condition', 'CODE'),
        'medications': ('Medication', 'CODE'),
        'observations': ('Observation', 'CODE'),
        'procedures': ('Procedure', 'CODE')
    }
    
    for entity_key, (entity_name, code_col) in entity_types.items():
        for _, row in data[entity_key].iterrows():
            if pd.isna(row.get('ENCOUNTER')) or pd.isna(row.get(code_col)):
                continue
                
            entity_id = f"{entity_name}_{row[code_col]}"
            encounter_id = f"Encounter_{row['ENCOUNTER']}"
            
            # Add entity node
            G.add_node(entity_id, 
                      type=entity_name.lower(),
                      color=get_entity_color(entity_name),
                      size=150)
            
            # Connect to encounter
            G.add_edge(entity_id, encounter_id, 
                      relationship=f"recorded_in_{entity_name.lower()}")
            
            # Track patient-encounter relationships
            encounter_map[encounter_id].append(f"Patient_{row['PATIENT']}")
    
    # Add encounters and connect to patients
    for _, encounter in data['encounters'].iterrows():
        encounter_id = f"Encounter_{encounter['Id']}"
        patient_id = f"Patient_{encounter['PATIENT']}"
        
        G.add_node(encounter_id,
                  type="encounter",
                  color='#f7cac9',
                  size=200,
                  date=encounter.get('DATE', ''),
                  code=encounter.get('CODE', ''))
        
        # Connect encounter to patient
        G.add_edge(patient_id, encounter_id,
                 relationship="had_encounter")
    
    print(f"✓ Graph contains {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

def get_entity_color(entity_type):
    """Return color codes for different entity types"""
    colors = {
        'Condition': '#4ecdc4',
        'Medication': '#45b7d1',
        'Procedure': '#ffa07a',
        'Observation': '#98d8c8'
    }
    return colors.get(entity_type, '#777777')

def visualize_graph(G, output_path):
    """Generate visualization of the encounter-centric graph"""
    print("\nGenerating visualization...")
    plt.figure(figsize=(30, 30))
    
    # Get positions using bipartite layout
    patients = [n for n in G.nodes() if G.nodes[n]['type'] == 'patient']
    encounters = [n for n in G.nodes() if G.nodes[n]['type'] == 'encounter']
    medical = [n for n in G.nodes() if n not in patients and n not in encounters]
    
    pos = nx.spring_layout(G, k=0.7, iterations=100, seed=42)
    
    # Draw nodes by type
    nx.draw_networkx_nodes(G, pos, nodelist=patients,
                         node_color=[G.nodes[n]['color'] for n in patients],
                         node_size=[G.nodes[n]['size'] for n in patients])
    
    nx.draw_networkx_nodes(G, pos, nodelist=encounters,
                         node_color=[G.nodes[n]['color'] for n in encounters],
                         node_size=[G.nodes[n]['size'] for n in encounters])
    
    nx.draw_networkx_nodes(G, pos, nodelist=medical,
                         node_color=[G.nodes[n]['color'] for n in medical],
                         node_size=[G.nodes[n]['size'] for n in medical])
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.5, width=0.8)
    
    # Add labels only for patients and encounters
    labels = {n: G.nodes[n]['label'] for n in patients if 'label' in G.nodes[n]}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Patients',
                  markerfacecolor='#ff6b6b', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Encounters',
                  markerfacecolor='#f7cac9', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Conditions',
                  markerfacecolor='#4ecdc4', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Medications',
                  markerfacecolor='#45b7d1', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Procedures',
                  markerfacecolor='#ffa07a', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Observations',
                  markerfacecolor='#98d8c8', markersize=15)
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=18)
    plt.title("Encounter-Centric Medical Knowledge Graph", fontsize=24)
    plt.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")

if __name__ == "__main__":
    print("=== Encounter-Centric Medical Graph Generator ===")
    data = load_data()
    
    # Create the encounter-centric graph
    med_graph = create_encounter_centric_graph(data)
    
    # Visualize the complete graph
    visualize_graph(med_graph, OUTPUT_DIR / "full_encounter_graph.png")
    
    # Generate individual patient graphs
    print("\nGenerating individual patient graphs...")
    patients = [n for n in med_graph.nodes() if med_graph.nodes[n]['type'] == 'patient']
    
    for patient in patients[:5]:  # Just show first 5 for demo
        # Get all encounters for this patient
        encounters = [n for n in med_graph.neighbors(patient)]
        # Get all connected medical entities
        medical_entities = []
        for enc in encounters:
            medical_entities.extend([n for n in med_graph.neighbors(enc) 
                                   if n != patient])
        
        # Create subgraph
        subgraph = med_graph.subgraph([patient] + encounters + medical_entities)
        
        # Visualize
        output_path = OUTPUT_DIR / f"{patient}_graph.png"
        visualize_graph(subgraph, output_path)