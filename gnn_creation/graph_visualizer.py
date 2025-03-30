import torch
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

def load_graphs(file_path):
    """Load saved PyG graphs from a .pt file."""
    try:
        graphs = torch.load(file_path, weights_only=False)
        print(f"Loaded {len(graphs)} graphs from {file_path}")
        return graphs
    except Exception as e:
        print(f"Error loading graphs: {str(e)}")
        return None

def visualize_graph(graph, graph_index=0):
    """
    Visualize a single graph with improved layout and labeling.
    Procedures have been removed, so only Encounter, Medication, and Condition nodes are used.
    """
    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes with their types
    node_types = []
    offset = 0

    # Encounter nodes
    for i in range(int(graph.num_encounters)):
        G.add_node(i, type='Encounter')
        node_types.append('Encounter')
    offset += int(graph.num_encounters)

    # Medication nodes
    for i in range(int(graph.num_medications)):
        G.add_node(offset + i, type='Medication')
        node_types.append('Medication')
    offset += int(graph.num_medications)

    # Condition nodes
    for i in range(int(graph.num_conditions)):
        G.add_node(offset + i, type='Condition')
        node_types.append('Condition')
    offset += int(graph.num_conditions)

    # Add edges with time differences (if provided)
    edge_index = graph.edge_index.t().cpu().numpy()
    for i, (src, dst) in enumerate(edge_index):
        if graph.edge_attr is not None:
            time_diff = float(graph.edge_attr[i].item())
            G.add_edge(int(src), int(dst), time_diff=time_diff)
        else:
            G.add_edge(int(src), int(dst))
    
    # Create a structured hierarchical layout based on node type
    pos = hierarchical_layout(G, node_types)
    
    # Set up the figure
    plt.figure(figsize=(14, 10))
    
    # Draw nodes by type with different colors
    type_colors = {
        'Encounter': '#FF9999',
        'Medication': '#99FF99',
        'Condition': '#9999FF'
    }
    
    for node_type, color in type_colors.items():
        nodes = [n for n in G.nodes if G.nodes[n]['type'] == node_type]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=color,
            node_size=800,
            label=node_type
        )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=1.5,
        alpha=0.6,
        edge_color='gray'
    )
    
    # Add edge labels for time differences
    edge_labels = nx.get_edge_attributes(G, 'time_diff')
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.5
    )
    
    # Add node labels (using the first letter of node type + index)
    labels = {}
    for node in G.nodes:
        node_type = G.nodes[node]['type']
        labels[node] = f"{node_type[0]}{node+1}"
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    # Add legend and title
    plt.legend(scatterpoints=1, frameon=False, fontsize=10)
    plt.title(
        f"Patient Graph #{graph_index}\n"
        f"Nodes: {graph.x.size(0)} | Edges: {graph.edge_index.size(1)}",
        fontsize=14
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("my_plot.png")
    plt.show()

def hierarchical_layout(G, node_types):
    """Create a structured hierarchical layout based on node types."""
    pos = {}
    vertical_spacing = 1.5

    # Group nodes by type
    type_groups = defaultdict(list)
    for i, node_type in enumerate(node_types):
        type_groups[node_type].append(i)
    
    # Position nodes in horizontal layers by type
    y_pos = 0
    for node_type, nodes in type_groups.items():
        x_pos = 0
        for node in nodes:
            pos[node] = (x_pos, y_pos)
            x_pos += 1.5
        y_pos -= vertical_spacing
    
    return pos

def main():
    # Load graphs
    graphs = load_graphs("patient_graphs.pt")
    if not graphs:
        return
    
    # Interactive visualization loop
    while True:
        print(f"\nAvailable graphs: 0 - {len(graphs)-1}")
        choice = input("Enter graph index (q to quit): ").strip()
        
        if choice.lower() == 'q':
            break
        
        try:
            index = int(choice)
            if 0 <= index < len(graphs):
                visualize_graph(graphs[index], index)
            else:
                print("Invalid index. Try again.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

if __name__ == "__main__":
    main()