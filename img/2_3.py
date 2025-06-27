import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up 3D figure and axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define funnel with exponential shrinkage (parametric form)
theta = np.linspace(0, 2 * np.pi, 100)
z = np.linspace(0, 3, 50)  # z-axis: from M_4 (bottom) to M_meta (top)
r_0 = 1.5  # Initial radius at z=0
k = 0.7    # Exponential decay constant
r = r_0 * np.exp(-k * (3 - z))  # Radius decreases exponentially from r_0 at z=0 to near 0 at z=3
X = r * np.outer(np.cos(theta), np.ones_like(z))
Y = r * np.outer(np.sin(theta), np.ones_like(z))
Z = np.outer(np.ones_like(theta), z)

# Plot the funnel surface
ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.3, edgecolor='black', linewidth=0.5)

# Plot particles in M_meta (top of funnel)
np.random.seed(42)
n_particles = 100
meta_x = np.random.uniform(-1.5, 1.5, n_particles)
meta_y = np.random.uniform(-1.5, 1.5, n_particles)
meta_z = np.random.uniform(2.5, 3.0, n_particles)
ax.scatter(meta_x, meta_y, meta_z, c='blue', s=20, label=r'$S(x, y, \tau)$ in $\mathcal{M}_{\text{meta}}$')

# Plot filtered particles at pi (middle of funnel)
filter_z = np.random.uniform(1.0, 1.5, n_particles // 4)
filter_r = r_0 * np.exp(-k * (3 - filter_z))
filter_x = filter_r * np.cos(2 * np.pi * np.random.rand(n_particles // 4))
filter_y = filter_r * np.sin(2 * np.pi * np.random.rand(n_particles // 4))
ax.scatter(filter_x, filter_y, filter_z, c='green', s=20, label=r'Filtered by $\pi$ (CP1–CP8)')

# Plot realized entities in M_4 (bottom of funnel)
m4_z = np.random.uniform(0, 0.2, n_particles // 10)
m4_r = r_0 * np.exp(-k * (3 - m4_z))
m4_x = m4_r * np.cos(2 * np.pi * np.random.rand(n_particles // 10))
m4_y = m4_r * np.sin(2 * np.pi * np.random.rand(n_particles // 10))
ax.scatter(m4_x, m4_y, m4_z, c='red', s=20, label=r'Reality = $\text{Im}(\pi)$ in $\mathcal{M}_4$')

# Adjusted annotations
ax.text(0, 0, 4, r'$\mathcal{M}_{\text{meta}}$', fontsize=14, ha='center')  # Moved above cone
ax.text(2.95, 0, 2.15, r'$\pi: \mathcal{D} \subset \mathcal{M}_{\text{meta}} \to \mathcal{M}_4$', fontsize=13, ha='left')  # Moved right of cone
ax.text(-4.5, 0, 0.35, 'CP1–CP8 Constraints', fontsize=13, ha='right', color='darkblue')  # Moved left of cone
ax.text(0, 0, -0.5, r'$\mathcal{M}_4$', fontsize=13, ha='center')  # Moved further down

# Set plot properties
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-0.5, 3.5)  # Adjusted z-limit to accommodate M_4 position
ax.set_xlabel('X (Projection Space)')
ax.set_ylabel('Y (Projection Space)')
ax.set_zlabel('Z (Entropic Projection Axis, $\tau$)')
ax.set_title('Entropic Projection Funnel in the Meta-Space Model')
ax.legend(loc='upper right')
ax.grid(True)

# Adjust view angle for better visibility
ax.view_init(elev=20, azim=45)

# Save and show
plt.savefig('projection_funnel_3d_exponential_adjusted.png', dpi=300, bbox_inches='tight')
plt.show()