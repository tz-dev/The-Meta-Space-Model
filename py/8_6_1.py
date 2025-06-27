import numpy as np
import matplotlib.pyplot as plt

# Set up figure
fig, ax = plt.subplots(figsize=(8, 6))

# Parameters
hbar_eff = 1  # Effective Planck constant (normalized units)
delta_x = np.logspace(-35, -10, 100)  # Spatial uncertainty, from Planck length to atomic scales
delta_lambda = hbar_eff / delta_x  # Projective uncertainty, satisfying Delta x * Delta lambda = hbar_eff

# Plot the boundary
ax.plot(delta_x, delta_lambda, label=r'$\Delta x \cdot \Delta \lambda = \hbar_{\text{eff}}$', color='blue')

# Shade the allowed region (Delta x * Delta lambda >= hbar_eff)
ax.fill_between(delta_x, delta_lambda, 1e35, color='lightblue', alpha=0.3, label='Allowed Region')

# Set log-log scale
ax.set_xscale('log')
ax.set_yscale('log')

# Set labels and title
ax.set_xlabel(r'Spatial Uncertainty ($\Delta x$) [m]')
ax.set_ylabel(r'Projective Uncertainty ($\Delta \lambda$)')

# Set axis limits
ax.set_xlim([1e-35, 1e-10])
ax.set_ylim([1e-10, 1e35])

# Add legend and grid
ax.legend()
ax.grid(True, which="both", ls="--")

# Annotate hbar_eff
ax.text(1e-22, 1e10, r'$\hbar_{\text{eff}} \approx 1$ (normalized)', color='red')

plt.tight_layout()
plt.show()