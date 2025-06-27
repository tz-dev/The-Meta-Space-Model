import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

# Set up figure
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# Parameters for S^3 (projected as 2D sphere)
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# Spherical coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Spherical harmonic (example: Y_{2,1})
l, m = 2, 1
Y = sph_harm(m, l, phi, theta).real
surface = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(Y / np.max(np.abs(Y))), rstride=1, cstride=1)

# Add colorbar for spherical harmonic
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(Y)
fig.colorbar(mappable, ax=ax, label='Spherical Harmonic Y_{2,1}', shrink=0.6)

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.tight_layout()
plt.show()