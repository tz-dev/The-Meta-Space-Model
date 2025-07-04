import matplotlib.pyplot as plt
import networkx as nx

# Define nodes (reversed order for CP, EP, and MP)
cp_nodes = [f'CP{i}' for i in range(8, 0, -1)]  # Reversed: CP8 to CP1
ep_nodes = [f'EP{i}' for i in range(14, 0, -1)]  # Reversed: EP14 to EP1
mp_nodes = [f'P{i}' for i in range(6, 0, -1)]    # Reversed: P6 to P1

# Define dependencies (CP -> EP) - adjusted for reversed order
cp_to_ep = {
    'CP8': ['EP8', 'EP14'],
    'CP7': ['EP5', 'EP6', 'EP8', 'EP11', 'EP14'],
    'CP6': ['EP3', 'EP4', 'EP7', 'EP9', 'EP10', 'EP11', 'EP13'],
    'CP5': ['EP1', 'EP5', 'EP6', 'EP8', 'EP12', 'EP14'],
    'CP4': ['EP2', 'EP10', 'EP12'],
    'CP3': ['EP2', 'EP3', 'EP4', 'EP7', 'EP9', 'EP11', 'EP13'],
    'CP2': ['EP1', 'EP2', 'EP3', 'EP5', 'EP8', 'EP10', 'EP12', 'EP14'],
    'CP1': ['EP1', 'EP13']
}

# Define EP to Meta-Projection mapping (from 6.6.3 table) - no change needed
ep_to_mp = {
    'EP1': 'P1', 'EP2': 'P1', 'EP5': 'P1',
    'EP3': 'P2', 'EP4': 'P2',
    'EP7': 'P3', 'EP13': 'P3',
    'EP9': 'P4', 'EP11': 'P4',
    'EP10': 'P5', 'EP12': 'P5',
    'EP6': 'P6', 'EP8': 'P6', 'EP14': 'P6'
}

# Colors for CP nodes and their edges
cp_colors = {
    'CP8': 'magenta', 'CP7': 'teal', 'CP6': 'brown', 'CP5': 'purple',
    'CP4': 'orange', 'CP3': 'red', 'CP2': 'green', 'CP1': 'blue'
}

# Pastellfarben für Meta-Projections
mp_colors = {
    'P1': '#FFB6C1',  # Hellrosa
    'P2': '#ADD8E6',  # Hellblau
    'P3': '#90EE90',  # Hellgrün
    'P4': '#D7B9A8',  # Hellbraun
    'P5': '#DDA0DD',  # Helllila
    'P6': '#FFA07A'   # Hellorange
}

# Create directed graph
G = nx.DiGraph()
G.add_nodes_from(cp_nodes + ep_nodes + mp_nodes)

# Add edges: CP -> EP
for cp, eps in cp_to_ep.items():
    for ep in eps:
        G.add_edge(cp, ep)

# Add edges: EP -> Meta-Projection
for ep, mp in ep_to_mp.items():
    G.add_edge(ep, mp)

# Define fixed positions: CPs left, EPs middle, Meta-Projections right, vertically centered
# Center vertically based on max number of nodes (14 EPs)
center_y = 7.0  # Mitte bei 14/2 + 3
pos = {}
for i, cp in enumerate(cp_nodes):
    pos[cp] = (-2.5, center_y - (len(cp_nodes) - 1) / 2 + i)
for i, ep in enumerate(ep_nodes):
    pos[ep] = (0, i)  # Reversed order, i goes from 0 to 13 (EP14 to EP1)
for i, mp in enumerate(mp_nodes):
    pos[mp] = (2.5, center_y - (len(mp_nodes) - 1) / 2 + i)

# Draw figure
plt.figure(figsize=(18, 12))

# Draw CP nodes with white labels
for cp in cp_nodes:
    nx.draw_networkx_nodes(G, pos, nodelist=[cp], node_color=cp_colors[cp], node_size=1000)
    nx.draw_networkx_labels(G, pos, labels={cp: cp}, font_size=10, font_color='white')

# Draw EP nodes
nx.draw_networkx_nodes(G, pos, nodelist=ep_nodes, node_color='lightgray', node_size=800)
nx.draw_networkx_labels(G, pos, labels={ep: ep for ep in ep_nodes}, font_size=10)

# Draw Meta-Projection nodes with pastellfarben
for mp in mp_nodes:
    nx.draw_networkx_nodes(G, pos, nodelist=[mp], node_color=mp_colors[mp], node_size=1200)
    nx.draw_networkx_labels(G, pos, labels={mp: mp}, font_size=12, font_color='black')

# Draw edges with colors based on CP (CP -> EP)
for cp, eps in cp_to_ep.items():
    edges = [(cp, ep) for ep in eps]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=cp_colors[cp])

# Draw edges from EP to Meta-Projection with colors based on Meta-Projection
ep_to_mp_edges = [(ep, mp) for ep, mp in ep_to_mp.items()]
edge_colors = [mp_colors[mp] for ep, mp in ep_to_mp.items()]
nx.draw_networkx_edges(G, pos, edgelist=ep_to_mp_edges, edge_color=edge_colors)

plt.axis('off')
plt.tight_layout()
plt.show()