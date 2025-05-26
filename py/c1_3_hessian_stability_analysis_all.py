# =============================================================================
# File        : c1_3_hessian_stability_analysis_all.py
# Purpose     : Compute second-order spatial/temporal derivatives of a 4D entropy field
#               and analyze stability via Hessian component distributions.
# Input       : "entropy_field_long.npy" (shape: Nx x Ny x Nz x Ntau)
# Grid size   : dx = dy = dz = 0.1 ; dtau = 0.001
# Output      : Summary statistics + histogram plots of H_ii components
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
filename = "entropy_field_long.npy"
dx = dy = dz = 0.1
dtau = 0.001

# === Load the entropy field ===
S = np.load(filename)
Nx, Ny, Nz, Ntau = S.shape
print(f"Loaded field shape: {S.shape}")

# === Choose a time slice (avoid last index to ensure +1 access) ===
tau_index = Ntau - 2

# === Initialize containers for diagonal Hessian entries ===
H11_list, H22_list, H33_list, H44_list = [], [], [], []

# === Loop through valid interior grid points ===
for x in range(1, Nx - 1):
    for y in range(1, Ny - 1):
        for z in range(1, Nz - 1):
            try:
                H11 = (S[x+1, y, z, tau_index] - 2*S[x, y, z, tau_index] + S[x-1, y, z, tau_index]) / dx**2
                H22 = (S[x, y+1, z, tau_index] - 2*S[x, y, z, tau_index] + S[x, y-1, z, tau_index]) / dy**2
                H33 = (S[x, y, z+1, tau_index] - 2*S[x, y, z, tau_index] + S[x, y, z-1, tau_index]) / dz**2
                H44 = (S[x, y, z, tau_index+1] - 2*S[x, y, z, tau_index] + S[x, y, z, tau_index-1]) / dtau**2
                H11_list.append(H11)
                H22_list.append(H22)
                H33_list.append(H33)
                H44_list.append(H44)
            except:
                continue

# === Summary statistics function ===
def summary(name, arr):
    arr = np.array(arr)
    print(f"{name}:")
    print(f"  Mean   = {np.mean(arr):.6e}")
    print(f"  Std    = {np.std(arr):.6e}")
    print(f"  Median = {np.median(arr):.6e}")
    return arr

# === Print summaries ===
print("=== Hessian Stability Analysis (C.1.3) ===")
print(f"Samples per component: {len(H11_list)}")

A11 = summary("H_11 (∂²/∂x²)", H11_list)
A22 = summary("H_22 (∂²/∂y²)", H22_list)
A33 = summary("H_33 (∂²/∂z²)", H33_list)
A44 = summary("H_44 (∂²/∂τ²)", H44_list)

# === Plot histograms ===
components = [("H_11", A11), ("H_22", A22), ("H_33", A33), ("H_44", A44)]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (name, data) in zip(axes.ravel(), components):
    ax.hist(data, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(data), color='red', linestyle='--', label='Mean')
    ax.axvline(np.median(data), color='green', linestyle='--', label='Median')
    ax.set_title(f"{name} Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig("hessian_component_distributions.png", dpi=300)
plt.show()
