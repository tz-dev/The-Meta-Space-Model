import numpy as np
import matplotlib.pyplot as plt

# RG beta-function coefficients (1-loop, Standard Model normalization)
b1 = -41 / 6   # U(1)_Y
b2 = 19 / 6    # SU(2)_L
b3 = 7         # SU(3)_C

# Initial values at tau = 0 (corresponds to μ ~ 91 GeV)
alpha1_0 = 0.0169
alpha2_0 = 0.0338
alpha3_0 = 0.118

# Entropic time grid
tau = np.linspace(0, 40, 500)

# Integration using analytical 1-loop solution:
def alpha_1loop(alpha0, b, tau):
    return alpha0 / (1 + (b / (2 * np.pi)) * alpha0 * tau)

# Compute RG trajectories
a1 = alpha_1loop(alpha1_0, b1, tau)
a2 = alpha_1loop(alpha2_0, b2, tau)
a3 = alpha_1loop(alpha3_0, b3, tau)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(tau, a1, label=r'$\alpha_1$ (U(1)$_Y$)', color='blue')
plt.plot(tau, a2, label=r'$\alpha_2$ (SU(2)$_L$)', color='green')
plt.plot(tau, a3, label=r'$\alpha_3$ (SU(3)$_C$)', color='red')
plt.axvline(33.1, color='gray', linestyle='--', label=r'GUT scale: $\tau^* \approx 33.1$')
plt.xlabel(r'Entropic Time $\tau$')
plt.ylabel(r'$\alpha_i(\tau)$')
plt.title('RG Flow of Coupling Constants in Entropic Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rg_flow_alpha_tau.png", dpi=300)
plt.show()
