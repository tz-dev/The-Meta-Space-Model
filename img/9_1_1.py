import numpy as np
import matplotlib.pyplot as plt

# Set up figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

# Parameters for 2D grid
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

# G_{\mu\nu}: Symbolic curvature (e.g., scalar curvature R)
R = np.exp(-(X**2 + Y**2))

# I_{\mu\nu}: Entropic curvature (e.g., \nabla_\mu \nabla_\nu S)
I = np.cos(np.pi * X) * np.cos(np.pi * Y)

# Plot G_{\mu\nu} (left)
im1 = ax1.imshow(R, cmap='viridis', extent=[-1, 1, -1, 1], origin='lower')
fig.colorbar(im1, ax=ax1, label='Scalar Curvature $R$', shrink=0.6)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Classical Gravitation: $G_{\\mu\\nu}$ (Curvature)')

# Plot I_{\mu\nu} (right)
im2 = ax2.imshow(I, cmap='plasma', extent=[-1, 1, -1, 1], origin='lower')
fig.colorbar(im2, ax=ax2, label='Entropic Curvature $I_{\\mu\\nu}$', shrink=0.6)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('MSM Tensor: $I_{\\mu\\nu}$ (Entropic)')

plt.tight_layout()
plt.show()