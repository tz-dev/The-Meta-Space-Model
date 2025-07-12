import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Set up figure
fig, ax = plt.subplots(figsize=(8, 6))

# Generate discrete points (symbolic valid configurations)
np.random.seed(42)  # For reproducibility
n_points = 20
n = np.random.uniform(10, 40, n_points)  # Spectral mode number
S = np.random.uniform(0.2, 0.8, n_points)  # Entropic measure

# Plot scatter
ax.scatter(n, S, color='blue', s=50, label='Valid Configurations')

# Highlight finite solution span
solution_span = Rectangle((10, 0.2), 30, 0.6, edgecolor='red', facecolor='red', alpha=0.1, label='Finite Solution Span')
ax.add_patch(solution_span)

# Set labels and title
ax.set_xlabel('Spectral Mode Number ($n$)')
ax.set_ylabel('Entropic Measure ($S$)')

# Set axis limits
ax.set_xlim([0, 50])
ax.set_ylim([0, 1])

# Add legend
ax.legend(loc='upper right')

# Add grid
ax.grid(True, ls='--')

# Add annotation for numerical relevance
ax.text(0.05, 0.95, 'Small but finite set of configurations (CP3, CP6)', transform=ax.transAxes, fontsize=10, color='red')

plt.tight_layout()
plt.savefig('10_2_4.png', dpi=300, bbox_inches='tight')