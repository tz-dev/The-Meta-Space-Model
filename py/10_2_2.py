import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Set up figure
fig, ax = plt.subplots(figsize=(8, 6))

# Define ellipses for field spaces
total_field_space = Ellipse(xy=(50, 500), width=80, height=800, edgecolor='blue', fc='lightblue', lw=2, label='Total Field Space')
proj_field_space = Ellipse(xy=(50, 500), width=20, height=200, edgecolor='red', fc='lightcoral', lw=2, label='Projectable Field Space')
ax.add_patch(total_field_space)
ax.add_patch(proj_field_space)

# Add filter arrows
ax.annotate('', xy=(50, 700), xytext=(50, 900),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(55, 800, 'Projection \( \pi \)', fontsize=10, color='black')
ax.annotate('', xy=(50, 600), xytext=(50, 400),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(55, 500, 'Filter (CP3, CP6)', fontsize=10, color='black')

# Set labels and title
ax.set_xlabel('Complexity (n, spectral mode number)')
ax.set_ylabel('Number of Field Configurations')

# Set axis limits
ax.set_xlim([0, 100])
ax.set_ylim([0, 1200])

# Add legend
ax.legend()

# Add grid
ax.grid(True, ls='--')

plt.tight_layout()
plt.show()