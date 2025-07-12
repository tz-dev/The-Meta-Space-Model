import numpy as np
import matplotlib.pyplot as plt

# Set up figure
fig, ax = plt.subplots(figsize=(8, 6))

# Parameters
n = np.logspace(0, 2, 100)  # Complexity (spectral mode number), from 1 to 100
alpha = 3  # Power-law exponent for field reduction
N_valid = 100 / n**alpha  # Number of valid fields, N_valid ~ n^(-alpha)

# Log-log plot
ax.plot(n, N_valid, label=r'$N_{\text{valid}} \sim n^{-3}$', color='blue')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Complexity ($n$, spectral mode number)')
ax.set_ylabel(r'Number of valid fields ($N_{\text{valid}}$)')
ax.legend()
ax.grid(True, which="both", ls="--")

# Set axis limits
ax.set_xlim([1, 100])
ax.set_ylim([1e-6, 100])

plt.tight_layout()
plt.savefig('8_3_2.png', dpi=300, bbox_inches='tight')