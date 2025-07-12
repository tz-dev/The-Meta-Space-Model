import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define axes limits and labels
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Computational Tractability')
ax.set_ylabel('Semantic Depth')

# State space rectangle
state_space = Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(state_space)

# Add label for State Space
ax.text(0.11, 0.88, 'State Space', ha='left', va='top', fontsize=10, backgroundcolor='white')

# W_comp oval (computable subspace)
w_comp = Ellipse((0.5, 0.5), 0.6, 0.4, edgecolor='blue', facecolor='none', linestyle='--', linewidth=2)
ax.add_patch(w_comp)

# Add label for W_comp
ax.text(0.5, 0.5, r'$\mathcal{W}_{\text{comp}}$', ha='center', va='center', fontsize=16, backgroundcolor='white')

# Annotation for uncertainty condition
ax.text(0.875, 0.85, r'$\Delta x \cdot \Delta \lambda \gtrsim \hbar_{\text{eff}}(\tau)$', ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
ax.text(0.5, 0.2, 'Entropic quantization threshold', ha='center', va='bottom', fontsize=10)

# Grid
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('5_1_6.png', dpi=300, bbox_inches='tight')