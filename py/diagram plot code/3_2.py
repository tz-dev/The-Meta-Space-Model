import matplotlib.pyplot as plt
import networkx as nx

# Node definitions
nodes = {
    "meta": r"Entropic Manifold" + "\n" + r"$\mathcal{M}_{\mathrm{meta}}$",
    "post": "8 Core Postulates\n(CP1–CP8)",
    "filter": "Projection Filters\n(Simulation Admissibility)",
    "fields": r"Filtered Field Space" + "\n" + r"$\mathcal{F}_{\mathrm{proj}}$",
    "reality": r"Projected Reality" + "\n" + r"$\mathcal{M}_4$"
}

# Create directed graph
G = nx.DiGraph()
G.add_nodes_from(nodes.keys())
edges = [("meta", "post"), ("post", "filter"), ("filter", "fields"), ("fields", "reality")]
G.add_edges_from(edges)

# Layout
pos = {
    "meta": (0, 4),
    "post": (0, 3),
    "filter": (0, 2),
    "fields": (0, 1),
    "reality": (0, 0)
}

# Set up figure
fig, ax = plt.subplots(figsize=(6, 8))
node_colors = ["lightblue", "cornflowerblue", "lightsalmon", "khaki", "palegreen"]

# Draw elements
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, edgecolors='black', ax=ax)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', width=2, ax=ax)
nx.draw_networkx_labels(G, pos, labels=nodes, font_size=9, font_color='black', verticalalignment='center')

# Finalize
ax.set_axis_off()
plt.tight_layout()
plt.savefig("3_1_2.png", dpi=300)
