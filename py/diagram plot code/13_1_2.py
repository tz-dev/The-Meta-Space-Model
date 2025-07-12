import numpy as np
import matplotlib.pyplot as plt

# Set up figure
fig, ax = plt.subplots(figsize=(8, 6))

# Entropic time axis
tau = np.linspace(0, 2, 200)

# Define each consistency component as smooth but distinct functions of τ
grad_S_term = 0.3 * np.exp(-5 * (tau - 0.4)**2)                   # CP2 term: gradient peak
redundancy_term = 0.2 + 0.05 * np.cos(3 * np.pi * tau)           # CP5 term: oscillatory structure
proj_stability_term = 0.3 * (1 - np.exp(-3 * tau))               # CP3 term: saturating growth
hbar_mismatch_term = 0.15 * np.exp(-10 * (tau - 1.5)**2)         # CP7 term: Gaussian bump near τ = 1.5

# Total functional
C_total = grad_S_term + redundancy_term + proj_stability_term + hbar_mismatch_term

# Plot each component
ax.plot(tau, grad_S_term, label=r'$w_1 \cdot \|\nabla_\tau S\|^2$', linestyle='-', color='blue')
ax.plot(tau, redundancy_term, label=r'$w_2 \cdot R[\pi]$', linestyle='--', color='orange')
ax.plot(tau, proj_stability_term, label=r'$w_3 \cdot \delta S_{\mathrm{proj}}$', linestyle='-.', color='green')
ax.plot(tau, hbar_mismatch_term, label=r'$w_4 \cdot |\hbar_{\mathrm{eff}} - \hbar|^2$', linestyle=':', color='red')

# Plot total C[ψ]
ax.plot(tau, C_total, label=r'$C[\psi]$', linewidth=2.5, color='black')

# Labels and title
ax.set_xlabel('Entropic Time ($\\tau$)')
ax.set_ylabel('Consistency Functional $C[\\psi]$')

# Legend and grid
ax.legend(loc='upper right')
ax.grid(True, ls='--')

plt.tight_layout()
plt.savefig('13_1_2.png', dpi=300, bbox_inches='tight')
