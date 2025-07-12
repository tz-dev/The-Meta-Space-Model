import numpy as np
import matplotlib.pyplot as plt

# Entropic time τ
tau = np.linspace(0.1, 2.5, 300)

# Spectral gap behavior (compression scenario)
delta_lambda = np.exp(-1.5 * tau) + 0.01  # avoids division by zero

# Entropy coherence factor (monotonically increasing)
entropy_gradient = 1 - np.exp(-2 * tau)

# Effective Planck constant (normalized)
hbar_eff = 1.0

# Emergent alpha coupling (projectional condition)
alpha_eff = delta_lambda / (entropy_gradient * hbar_eff)

# Plot setup
fig, ax = plt.subplots(figsize=(8, 6))

# Coupling vs. entropy coherence
sc = ax.scatter(entropy_gradient, alpha_eff, c=tau, cmap='plasma', label=r'$\alpha_{\text{eff}}(\tau)$')

# Labels and title
ax.set_xlabel('Entropy Gradient Coherence $\\nabla_\\tau S$')
ax.set_ylabel('Effective Coupling $\\alpha_{\mathrm{eff}}$')

# Reference line for physical α
ax.axhline(1/137, color='green', linestyle='--', linewidth=1, label=r'CODATA $\alpha \approx 1/137$')

# Colorbar for τ
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r'Entropic Time $\tau$')

# Grid, legend, tight layout
ax.grid(True, ls='--', alpha=0.5)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('14_9_1.png', dpi=300, bbox_inches='tight')