#!/usr/bin/env python3
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import argparse
from typing import Dict, Tuple
import re

def extract_layer_number(node_name: str) -> int:
    """Extract layer number from node name."""
    match = re.search(r'(\d+)', node_name)
    if match:
        return int(match.group(1))
    return 0

def create_network_graph(df: pd.DataFrame, 
                        significance_col: str = 'q-value',
                        max_edges_per_node: int = None,
                        significance_threshold: float = 0.05) -> Tuple[nx.Graph, Dict]:
    """
    Create network graph from motif data.
    
    Args:
        df: DataFrame with columns ['Gene', 'Node', significance_col, 'Query_consensus']
        significance_col: Column to use for edge coloring ('q-value' or 'E-value')
        max_edges_per_node: Maximum number of edges per node (None for all)
        significance_threshold: Filter edges above this significance value
        
    Returns:
        networkx Graph and positions dictionary
    """
    # Filter by significance
    df = df[df[significance_col] <= significance_threshold].copy()
    
    # If max_edges_per_node is specified, keep only the most significant edges
    if max_edges_per_node:
        df = df.sort_values(significance_col, ascending=True)
        df = df.groupby('Node').head(max_edges_per_node)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    nodes_layer = {}
    for node in df['Node'].unique():
        layer = extract_layer_number(node)
        nodes_layer[node] = layer
        G.add_node(node, node_type='neural', layer=layer)
    
    for motif in df['Gene'].unique():
        G.add_node(motif, node_type='motif')
    
    # Add edges with significance values
    for _, row in df.iterrows():
        G.add_edge(row['Node'], row['Gene'], 
                  significance=row[significance_col],
                  consensus=row['Query_consensus'])
    
    # Calculate layout
    # Position neural network nodes in columns by layer
    pos = {}
    layers = sorted(set(nodes_layer.values()))
    
    # Position neural network nodes
    for layer in layers:
        layer_nodes = [node for node, l in nodes_layer.items() if l == layer]
        n_nodes = len(layer_nodes)
        for i, node in enumerate(layer_nodes):
            pos[node] = np.array([layer, i - n_nodes/2])
    
    # Position motif nodes on the right
    motif_nodes = [node for node in G.nodes() if G.nodes[node]['node_type'] == 'motif']
    n_motifs = len(motif_nodes)
    for i, node in enumerate(motif_nodes):
        pos[node] = np.array([max(layers) + 1, i - n_motifs/2])
    
    return G, pos

def plot_network(G: nx.Graph, pos: Dict, 
                title: str = "Neural Network Nodes to Motifs Network",
                output_file: str = None):
    """
    Create interactive network plot using plotly.
    
    Args:
        G: networkx Graph
        pos: Node positions dictionary
        title: Plot title
        output_file: Path to save HTML file (if None, displays plot)
    """
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_colors = []
    edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        significance = edge[2]['significance']
        edge_colors.extend([significance] * 3)
        edge_text.extend([f"Significance: {significance:.2e}<br>Consensus: {edge[2]['consensus']}"] * 3)
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5),
        hoverinfo='text',
        text=edge_text,
        mode='lines',
        line_color='grey',#edge_colors,
        #showscale=True,
        #colorscale='RdBu',
        #reversescale=True
    )
    
    # Create node traces
    neural_nodes = [(node, attr) for node, attr in G.nodes(data=True) 
                    if attr['node_type'] == 'neural']
    motif_nodes = [(node, attr) for node, attr in G.nodes(data=True) 
                   if attr['node_type'] == 'motif']
    
    # Neural network nodes
    neural_trace = go.Scatter(
        x=[pos[node][0] for node, _ in neural_nodes],
        y=[pos[node][1] for node, _ in neural_nodes],
        text=[f"Node: {node}<br>Layer: {attr['layer']}" 
              for node, attr in neural_nodes],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(color='black', width=1)
        ),
        textposition="top center",
        name='Neural Nodes'
    )
    
    # Motif nodes
    motif_trace = go.Scatter(
        x=[pos[node][0] for node, _ in motif_nodes],
        y=[pos[node][1] for node, _ in motif_nodes],
        text=[node for node, _ in motif_nodes],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=20,
            color='lightgreen',
            line=dict(color='black', width=1)
        ),
        textposition="top center",
        name='Motifs'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, neural_trace, motif_trace])
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        height=800,
        width=1200,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    if output_file:
        fig.write_html(output_file)
    else:
        fig.show()

def main():
    parser = argparse.ArgumentParser(description='Create network graph from motif data')
    parser.add_argument('input_file', help='Input CSV/TSV file path')
    parser.add_argument('-o', '--output', help='Output HTML file path', default='motif_network.html')
    parser.add_argument('--sep', help='File separator (comma or tab)', default='\t')
    parser.add_argument('--significance', help='Column to use for edge significance', 
                       choices=['q-value', 'E-value'], default='q-value')
    parser.add_argument('--max-edges', type=int, help='Maximum edges per node', default=None)
    parser.add_argument('--threshold', type=float, help='Significance threshold', default=0.05)
    
    args = parser.parse_args()
    
    # Read data
    sep = ',' if args.sep == 'comma' else '\t'
    df = pd.read_csv(args.input_file, sep=sep)
    
    # Create and plot network
    G, pos = create_network_graph(df, args.significance, args.max_edges, args.threshold)
    plot_network(G, pos, output_file=args.output)
    print(f"Network graph saved to {args.output}")

if __name__ == '__main__':
    main()