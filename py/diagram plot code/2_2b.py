import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up figure with 1 row, 2 columns
fig = plt.figure(figsize=(12, 6))

# Left: Entropic time axis
ax1 = fig.add_subplot(121)
tau = np.linspace(0, 1, 100)
S_tau = tau  # Entropy increases linearly
ax1.plot(tau, S_tau, label=r'$S(\tau)$', color='blue')
ax1.set_xlabel(r'$\tau$ (Entropic Time)')
ax1.set_ylabel(r'$S(\tau)$ (Entropy)')
ax1.set_title(r'$\mathbb{R}_\tau$: Entropic Time Axis')
ax1.legend()

# Projection arrow
ax1.annotate('', xy=(0.5, 0.8), xytext=(0.5, 0.2),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.text(0.55, 0.5, r'$\pi: \mathcal{M}_{\text{meta}} \to \mathcal{M}_4$', color='red')

# Right: CY_3 torus approximation
ax2 = fig.add_subplot(122, projection='3d')
u = np.linspace(0, 2*np.pi, 20)
v = np.linspace(0, 2*np.pi, 20)
u, v = np.meshgrid(u, v)
R, r = 1, 0.3
x = (R + r * np.cos(v)) * np.cos(u)
y = (R + r * np.cos(v)) * np.sin(u)
z = r * np.sin(v)

ax2.plot_surface(x, y, z, color='purple', alpha=0.5)
ax2.set_title(r'$CY_3$ (Torus Approximation)')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

plt.tight_layout()
plt.show()
