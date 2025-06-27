import numpy as np
import matplotlib.pyplot as plt

# Define spatial uncertainty range (Delta x)
delta_x = np.logspace(-35, -10, 400)  # from Planck to atomic scale [m]

# Effective Planck constant (normalized)
hbar_eff = 1

# Corresponding spectral resolution (Delta lambda)
delta_lambda = hbar_eff / delta_x  # satisfying: Delta x * Delta lambda = hbar_eff

# Plot setup
fig, ax = plt.subplots(figsize=(8, 6))

# Plot uncertainty boundary
ax.plot(delta_x, delta_lambda, label=r'$\Delta x \cdot \Delta \lambda = \hbar_{\mathrm{eff}}$', color='blue')

# Shade allowed (computable) region
ax.fill_between(delta_x, delta_lambda, 1e35, color='lightblue', alpha=0.3, label='Admissible Projection Domain')

# Log-log scale
ax.set_xscale('log')
ax.set_yscale('log')

# Labels and title
ax.set_xlabel(r'Spatial Resolution $\Delta x$ [m]')
ax.set_ylabel(r'Spectral Resolution $\Delta \lambda$')
ax.set_title('Projectional Uncertainty Relation in MSM')

# Limits
ax.set_xlim([1e-35, 1e-10])
ax.set_ylim([1e-10, 1e35])

# Add annotation
ax.text(2e-28, 1e10, r'Minimal Action $\hbar_{\mathrm{eff}}(\tau)$', fontsize=10, color='red')

# Legend and grid
ax.legend()
ax.grid(True, ls='--')

plt.tight_layout()
plt.show()
