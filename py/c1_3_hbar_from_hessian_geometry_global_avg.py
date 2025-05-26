# =============================================================================
# C.1.3 – ℏ from Global Time Curvature Average
# File: c1_3_hbar_from_hessian_geometry_global_avg.py
# This script computes the average temporal second derivative (H_33) of the
# entropy field across the entire 3D space (with safety margin from boundaries).
# It estimates ℏ by projecting the result via hbar ≈ ⟨|H_33|⟩ * kB * dtau.
# Input: "entropy_field_long.npy"
# Output: Printed estimate of ℏ and relative deviation from official value.
# =============================================================================

import numpy as np

# --- Parameters ---
dx = dy = dz = 0.1
dtau = 0.001
kB = 1.380649e-23  # J/K

# --- Load data ---
S = np.load("entropy_field_long.npy")
Nx, Ny, Nz, Ntau = S.shape
tau_idx = Ntau - 2  # second-to-last time index

# --- 2nd derivative in time direction (H_33) ---
def second_derivative_tau(arr, i, j, k, l, dtau):
    return (arr[i,j,k,l+1] - 2*arr[i,j,k,l] + arr[i,j,k,l-1]) / dtau**2

# --- Global average over inner volume (margin to avoid boundaries) ---
margin = 5
sum_H33 = 0.0
count = 0

for i in range(margin, Nx - margin):
    for j in range(margin, Ny - margin):
        for k in range(margin, Nz - margin):
            try:
                H33 = second_derivative_tau(S, i, j, k, tau_idx, dtau)
                sum_H33 += abs(H33)
                count += 1
            except:
                continue

# --- Mean and ℏ projection ---
if count == 0:
    raise RuntimeError("No valid samples found.")

mean_H33 = sum_H33 / count
hbar_proj = mean_H33 * kB * dtau
hbar_official = 1.054571817e-34
rel_dev = (hbar_proj - hbar_official) / hbar_official * 100

# --- Output ---
print("=== ℏ from Global Time Curvature Average (C.1.3) ===")
print(f"Samples averaged: {count}")
print(f"<|H_33|>: {mean_H33:.6e}")
print(f"Projected ℏ = {hbar_proj:.8e} J·s")
print(f"Official  ℏ = {hbar_official:.8e} J·s")
print(f"Relative deviation: {rel_dev:.5f}%")
