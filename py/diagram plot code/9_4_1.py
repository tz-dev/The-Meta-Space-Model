import networkx as nx
import matplotlib.pyplot as plt

# Create directed graph
G = nx.DiGraph()

# Add nodes for classical and MSM paths
nodes = ['Classical_Start', 'Classical_Evolution', 'Classical_End',
         'MSM_Input', 'MSM_Filter', 'MSM_Output']
G.add_nodes_from(nodes)

# Add edges
edges = [('Classical_Start', 'Classical_Evolution', {'label': 'Time Evolution'}),
         ('Classical_Evolution', 'Classical_End', {'label': 'Time Evolution'}),
         ('MSM_Input', 'MSM_Filter', {'label': 'Projection'}),
         ('MSM_Filter', 'MSM_Output', {'label': 'Projection'})]
G.add_edges_from([(u, v) for u, v, _ in edges])

# Define node labels
labels = {
    'Classical_Start': 'Initial State',
    'Classical_Evolution': 'Time Evolution\n(Differential Equations)',
    'Classical_End': 'Final State\n(Infinite States)',
    'MSM_Input': 'Input: Theory Space\n($\\mathcal{F}$, Infinite)',
    'MSM_Filter': 'Filter: CP1–CP8\n(Projection)',
    'MSM_Output': 'Output: Projectable Space\n($\\mathcal{F}_{\\text{proj}}$, Finite)'
}

# Define layout (vertical, two parallel paths)
pos = {
    'Classical_Start': (0, 2), 'Classical_Evolution': (0, 1), 'Classical_End': (0, 0),
    'MSM_Input': (1, 2), 'MSM_Filter': (1, 1), 'MSM_Output': (1, 0)
}

# Set up figure
fig, ax = plt.subplots(figsize=(8, 6))

# Draw graph
nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='black', node_size=2000, arrowsize=20, ax=ax)
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black', verticalalignment='center')

# Draw edge labels
edge_labels = {(u, v): d['label'] for u, v, d in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, label_pos=0.5)

# Remove axes
ax.set_axis_off()

plt.tight_layout()
plt.show()