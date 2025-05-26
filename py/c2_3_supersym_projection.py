# ========================================================
# File: c2_3_supersym_projection.py
# Purpose: Simulate entropy-based mass splitting via SUSY projection
# Parameters:
#   - α_S = 0.015 (entropic scaling)
#   - ω = π (projection frequency)
#   - θ = π/4 (phase offset)
#   - Base mass = electron mass
# Output:
#   - Mass histograms and projection states
# ========================================================

import numpy as np
import matplotlib.pyplot as plt

S_mean = 2.74309
S_sigma = 0.05894
k_B = 1.380649e-23
c = 2.99792458e8
alpha_S = 0.015
omega = 3.14159
theta = np.pi / 4
m_base = 9.10938356e-31

N = 10000
np.random.seed(42)
S = np.random.normal(S_mean, S_sigma, N)

projection_mask = np.sign(np.cos(omega * S + theta))

dS = np.random.normal(0, S_sigma, N)
delta_E = alpha_S * np.abs(dS) * k_B
delta_m = delta_E / c**2

m_SM = m_base + delta_m
m_SUSY = m_base - delta_m
visible = np.where(projection_mask > 0, m_SM, m_SUSY)
hidden = np.where(projection_mask < 0, m_SM, m_SUSY)

print("=== Supersymmetric Projection Results ===")
print(f"Mean mass (visible): {np.mean(visible):.5e} kg")
print(f"Mean mass (hidden):  {np.mean(hidden):.5e} kg")
print(f"Mean Δm (split):      {np.mean(delta_m):.5e} kg")

plt.figure(figsize=(10, 6))
plt.hist(m_SM, bins=60, alpha=0.6, label='m_SM', color='blue')
plt.hist(m_SUSY, bins=60, alpha=0.6, label='m_SUSY', color='orange')
plt.title('Entropic SUSY Mass Splitting')
plt.xlabel('Mass (kg)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('img/c2_3_supersym_mass_histogram.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(S[:300], projection_mask[:300], 'k.', alpha=0.7)
plt.title('Projection State vs. Entropy Sample')
plt.xlabel('Sample Index')
plt.ylabel('Projection State (±1)')
plt.grid(True)
plt.savefig('img/c2_3_supersym_projection_mask.png')
plt.close()
