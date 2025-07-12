import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Set up figure
fig, ax = plt.subplots(figsize=(8, 8))

# Generate galaxy positions (before lensing)
np.random.seed(42)
n_galaxies = 100
x = np.random.uniform(-10, 10, n_galaxies)
y = np.random.uniform(-10, 10, n_galaxies)

# Apply simple lensing distortion (e.g., shear)
kappa = 0.2  # Shear parameter
x_distorted = x + kappa * y
y_distorted = y + kappa * x

# Plot distorted galaxy field
ax.scatter(x_distorted, y_distorted, s=20, color='blue', label='Distorted Galaxies')

# Add saturation zones (entropic limits)
saturation_zones = [Circle((0, 0), 3, edgecolor='red', facecolor='red', alpha=0.2, label='Holographic Saturation Zone'),
                   Circle((5, -5), 2, edgecolor='red', facecolor='red', alpha=0.2)]
for zone in saturation_zones:
    ax.add_patch(zone)

# Set labels and title
ax.set_xlabel('Sky Coordinate ($\\alpha$)')
ax.set_ylabel('Sky Coordinate ($\\delta$)')

# Set axis limits
ax.set_xlim([-12, 12])
ax.set_ylim([-12, 12])

# Add legend
ax.legend(loc='upper right')

# Add grid
ax.grid(True, ls='--')

# Add annotation for JWST correlation
ax.text(0.05, 0.95, 'Entropic limits correlate with JWST observations (P6)', transform=ax.transAxes, fontsize=10, color='red')

plt.tight_layout()
plt.savefig('11_4_3.png', dpi=300, bbox_inches='tight')