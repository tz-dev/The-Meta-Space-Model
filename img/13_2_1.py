import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Set up figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Parameters for plots
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)

# Classical boundary: Geometric horizon (circle)
ax1.add_patch(Circle((0, 0), 1, edgecolor='blue', facecolor='lightblue', alpha=0.3, label='Geometric Horizon'))
ax1.set_xlim([-1.5, 1.5])
ax1.set_ylim([-1.5, 1.5])
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Classical Cosmos: Geometric Boundary')
ax1.legend(loc='upper right')
ax1.grid(True, ls='--')
ax1.set_aspect('equal')

# MSM boundary: Entropic boundary (irregular contour)
entropy = np.exp(-0.5 * (X**2 + Y**2)) + 0.5 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
im2 = ax2.contour(X, Y, entropy, levels=[0.8], colors='red', linestyles='solid', label='Entropic Boundary')
ax2.clabel(im2, inline=True, fontsize=8, fmt='Entropic Boundary')
ax2.set_xlim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True, ls='--')
ax2.set_aspect('equal')

# Add annotation for comparison
fig.text(0.5, 0.015, 'Transition: Geometric to Projectional Horizons (CP2, CP3, CP4)', ha='center', fontsize=10, color='red')

plt.tight_layout()
plt.show()