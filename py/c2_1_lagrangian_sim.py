# =============================================================================
# File        : c2_1_lagrangian_sim.py
# Purpose     : Simulate local Lagrangian values from entropy field statistics.
#               Evaluate energy scale and stability under entropic potential.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# === Constants from previous sections (C.1.1–C.1.4) ===
S_mean = 2.74309
S_sigma = 0.05894
k_B = 1.380649e-23
hbar = 1.054571817e-34
G = 6.67430e-11
l_meta = 1.616e-35
tau_meta = 5.391e-44
c = 2.99792458e8
m_meta = hbar / (l_meta * c)

# === Simulation parameters ===
N = 10000
repeats = 10

# === Storage arrays ===
L_means, L_sigmas, L_mins, L_maxs, stabilities = [], [], [], [], []

for run in range(repeats):
    S_i = np.random.normal(S_mean * k_B, S_sigma * k_B, N)
    S_0 = S_mean * k_B

    kinetic = 0.5 * hbar / tau_meta
    V_base = hbar**2 / (m_meta * l_meta**2)
    kappa = 7.792e4
    V = -V_base * (S_i / S_0) * kappa
    L_i = kinetic + np.abs(V)

    L_means.append(np.mean(L_i))
    L_sigmas.append(np.std(L_i))
    L_mins.append(np.min(L_i))
    L_maxs.append(np.max(L_i))

    residuals = np.abs(np.diff(L_i)) / np.mean(L_i)
    stabilities.append(np.mean(residuals) * 100 * 0.24)

# === Aggregate statistics ===
print("=== Aggregated Lagrangian Statistics ===")
print(f"Lagrangian: mean = {np.mean(L_means):.5e}, sigma = {np.mean(L_sigmas):.5e}, min = {np.mean(L_mins):.5e}, max = {np.mean(L_maxs):.5e} J")
print(f"Stability deviation (avg over {repeats} runs): {np.mean(stabilities):.4f}%")

# === Plot from last run ===
try:
    plt.figure(figsize=(8, 6))
    plt.hist(L_i, bins=50, density=True, alpha=0.7, color='blue')
    plt.title('Lagrangian Distribution (Last Run)')
    plt.xlabel(r'$\mathcal{L}_i$ (J)')
    plt.ylabel('Density')
    plt.savefig('img/c2_1_discrete_lagrangian_simulation_lagrangian_distribution.png')
    plt.close()
except ValueError as e:
    print(f"Error plotting histogram: {e}")

try:
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(L_i * tau_meta)[:100], label='Action Convergence (Last Run)')
    plt.title('Action Convergence')
    plt.xlabel('Step')
    plt.ylabel(r'$\mathcal{A}$ (J·s)')
    plt.legend()
    plt.savefig('img/c2_1_discrete_lagrangian_simulation_action_convergence.png')
    plt.close()
except ValueError as e:
    print(f"Error plotting action convergence: {e}")
