# =============================================================================
# C.1.3 – Stability Analysis of Temporal Curvature H_33
# File: c1_3_hessian_h33_stability_analysis.py
# This script performs a statistical analysis of the second derivative
# in time direction (H_33) across the entire entropy field volume.
# It plots a histogram to visualize the distribution and variation of H_33.
# Input: "entropy_field_long.npy"
# Output: Histogram "h33_distribution.png"
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
filename = "entropy_field_long.npy"
dx = dy = dz = 0.1
dtau = 0.001

# --- Load data ---
S = np.load(filename)
Nx, Ny, Nz, Ntau = S.shape
print(f"Loaded field shape: {S.shape}")

# --- Select safe time slice (avoid boundary) ---
tau_index = Ntau - 2
samples = []

# --- Compute H_33 = ∂²S/∂τ² ---
for x in range(1, Nx-1):
    for y in range(1, Ny-1):
        for z in range(1, Nz-1):
            try:
                h33 = (S[x, y, z, tau_index + 1] - 2 * S[x, y, z, tau_index] + S[x, y, z, tau_index - 1]) / (dtau ** 2)
                samples.append(h33)
            except:
                continue

samples = np.array(samples)
mean_h33 = np.mean(samples)
std_h33 = np.std(samples)
median_h33 = np.median(samples)

# --- Output statistics ---
print("=== Stability Analysis of H_33 (C.1.3) ===")
print(f"Samples analyzed: {len(samples)}")
print(f"Mean H_33:    {mean_h33:.6e}")
print(f"Std Dev H_33: {std_h33:.6e}")
print(f"Median H_33:  {median_h33:.6e}")

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.hist(samples, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(mean_h33, color='red', linestyle='--', label=f"Mean: {mean_h33:.2e}")
plt.axvline(median_h33, color='green', linestyle='--', label=f"Median: {median_h33:.2e}")
plt.xlabel("H_33 value")
plt.ylabel("Count")
plt.title("Distribution of H_33 (Temporal Curvature) in Entropy Field")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("h33_distribution.png", dpi=300)
plt.show()
