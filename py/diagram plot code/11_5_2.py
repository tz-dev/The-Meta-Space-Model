import numpy as np
import matplotlib.pyplot as plt

# Set up figure
fig, ax = plt.subplots(figsize=(8, 7))

# Parameters for tau (entropic time)
tau = np.linspace(0, 2, 100)

# MSM coupling: Sigmoid function with locking plateau
g_msm = 0.5 * (1 + np.tanh(2 * (tau - 1)))

# Classical RG-flow: Continuous variation
g_classical = 1 / (1 + tau)

# Plot MSM and classical flows
ax.plot(tau, g_msm, color='blue', label='MSM: Projectional Locking')
ax.plot(tau, g_classical, color='red', linestyle='--', label='Classical RG-Flow')

# Highlight locking plateau
ax.axhline(y=0.5, xmin=0.5, xmax=1, color='green', linestyle=':', label='Locking Plateau')

# Set labels and title
ax.set_xlabel('Entropic Time ($\\tau$)')
ax.set_ylabel('Coupling Constant ($g$)')

# Set axis limits
ax.set_xlim([0, 2])
ax.set_ylim([0, 1])

# Add legend
ax.legend(loc='lower right')

# Add grid
ax.grid(True, ls='--')

# Add annotation for EP7
ax.text(0.05, 0.95, 'Projective Fixpoints vs. Classical RG-Flow (EP7, CP3)', transform=ax.transAxes, fontsize=10, color='red')

plt.tight_layout()
plt.savefig('11_5_2.png', dpi=300, bbox_inches='tight')