import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Ensure mathtext for LaTeX rendering
plt.rc('text', usetex=False)  # Use mathtext (default), no external LaTeX required

# Set up figure
fig, ax = plt.subplots(figsize=(8, 6))

# Define computability window W_comp
w_comp = Rectangle((10, 5), 40, 25, edgecolor='blue', facecolor='lightblue', alpha=0.5, label=r'Computability Window $\mathcal{W}_{\text{comp}}$')
ax.add_patch(w_comp)

# Shade non-computable regions
ax.fill_between([0, 100], 30, 50, color='gray', alpha=0.3, label='Non-Computable (High Depth)')
ax.fill_between([50, 100], 0, 50, color='gray', alpha=0.3)
ax.fill_between([0, 10], 0, 50, color='gray', alpha=0.3, label='Non-Computable (Low/High Complexity)')

# Set labels and axis limits
ax.set_xlabel('Complexity (n, spectral mode number)')
ax.set_ylabel('Depth (D, entropic measure)')
ax.set_xlim([0, 100])
ax.set_ylim([0, 50])

# Add annotations

ax.text(30, 15, r'Computability Window $\mathcal{W}_{\text{comp}}$', fontsize=12, color='blue', ha='center')
ax.text(75, 42, 'Incomputable\n(High Depth)', fontsize=10, color='black', ha='center')
ax.text(75, 12, 'Incomputable\n(High Complexity)', fontsize=10, color='black', ha='center')
ax.text(15, 42, 'Incomputable\n(Low Complexity)', fontsize=10, ha='center')
ax.text(30, 45, 'CP6: Simulation Consistency', fontsize=10, color='red')

# Add legend below the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, title='Legend')

# Add grid
ax.grid(True, ls='--')

plt.tight_layout()
plt.savefig('10_4_3.png', dpi=300, bbox_inches='tight')