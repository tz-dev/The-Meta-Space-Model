import numpy as np
import matplotlib.pyplot as plt

# Define tau and entropy function
tau = np.linspace(-5, 5, 200)
S = 0.5 * tau**2 + 0.1 * np.sin(0.5 * tau) + 2  # Monotonic increase with slight oscillation

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot entropy curve
ax.plot(tau, S, 'b-', label=r'$S(\tau)$')
ax.fill_between(tau, S, where=(np.gradient(S, tau) > 0), color='lightgreen', alpha=0.5, label=r'$\nabla_\tau S > 0$ (Stable projection)')
ax.fill_between(tau, S, where=(np.gradient(S, tau) == 0), color='gray', alpha=0.3, label=r'$\nabla_\tau S = 0$ (No projection)')
ax.fill_between(tau, S, where=(np.gradient(S, tau) < 0), color='red', alpha=0.3, label=r'$\nabla_\tau S < 0$ (Unstable projection)')

# Annotations
ax.axvline(x=0, color='k', linestyle='--', label=r'$\tau = 0$ (Reference)')
ax.text(-3, 3, r'$\pi$ fails, no time', ha='left', va='bottom', fontsize=10, color='gray')
ax.text(1, 4, r'$\pi$ stabilizes, time emerges', ha='left', va='bottom', fontsize=10, color='green')
ax.text(-1, 1, r'$\pi$ collapses', ha='left', va='bottom', fontsize=10, color='red')

# Labels and title
ax.set_xlabel(r'$\tau \in \mathbb{R}_\tau$ (Ordering Parameter)')
ax.set_ylabel(r'$S(\tau)$ (Entropy)')
ax.set_title('Time as Emergent Ordering via Entropy Gradient')
ax.legend()
ax.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('4_2.png', dpi=300, bbox_inches='tight')