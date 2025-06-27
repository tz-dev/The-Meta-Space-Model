import matplotlib.pyplot as plt
import networkx as nx

# Define CP and EP nodes
cp_nodes = [f'CP{i}' for i in range(1, 9)]
ep_nodes = [f'EP{i}' for i in range(1, 15)]

# Define dependencies (edges: CP -> EP)
dependencies = {
    'CP1': ['EP1', 'EP13'],
    'CP2': ['EP1', 'EP2', 'EP3', 'EP5', 'EP8', 'EP10', 'EP12', 'EP14'],
    'CP3': ['EP2', 'EP3', 'EP4', 'EP7', 'EP9', 'EP11', 'EP13'],
    'CP4': ['EP2', 'EP10', 'EP12'],
    'CP5': ['EP1', 'EP5', 'EP6', 'EP8', 'EP12', 'EP14'],
    'CP6': ['EP3', 'EP4', 'EP7', 'EP9', 'EP10', 'EP11', 'EP13'],
    'CP7': ['EP5', 'EP6', 'EP8', 'EP11', 'EP14'],
    'CP8': ['EP8', 'EP14']
}

# Colors for CP nodes and their edges
cp_colors = {
    'CP1': 'blue',
    'CP2': 'green',
    'CP3': 'red',
    'CP4': 'orange',
    'CP5': 'purple',
    'CP6': 'brown',
    'CP7': 'teal',
    'CP8': 'magenta'
}

# Create directed graph
G = nx.DiGraph()
G.add_nodes_from(cp_nodes + ep_nodes)
for cp, eps in dependencies.items():
    for ep in eps:
        G.add_edge(cp, ep)

# Define fixed positions: CPs vertically left, EPs vertically right
pos = {}
cp_y = list(range(len(cp_nodes)))
ep_y = list(range(len(ep_nodes)))
for i, cp in enumerate(cp_nodes):
    pos[cp] = (-1.5, cp_y[i] + 3)
for i, ep in enumerate(ep_nodes):
    pos[ep] = (1.5, ep_y[i])

# Draw figure
plt.figure(figsize=(16, 10))

# Draw CP nodes with white labels
for cp in cp_nodes:
    nx.draw_networkx_nodes(G, pos, nodelist=[cp], node_color=cp_colors[cp], node_size=1000)
    nx.draw_networkx_labels(G, pos, labels={cp: cp}, font_size=10, font_color='white')

# Draw EP nodes
nx.draw_networkx_nodes(G, pos, nodelist=ep_nodes, node_color='lightgray', node_size=800)
nx.draw_networkx_labels(G, pos, labels={ep: ep for ep in ep_nodes}, font_size=10)

# Draw edges
for cp, eps in dependencies.items():
    edges = [(cp, ep) for ep in eps]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=cp_colors[cp], arrows=True)

plt.axis('off')
plt.tight_layout()
plt.show()
