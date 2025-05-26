# =============================================================================
# C.1.3 – ℏ from Local Temporal Curvature (Improved)
# File: c1_3_hbar_from_hessian_geometry_improved.py
# This script computes a local estimate of the second derivative in time
# direction (H_33) at the center of the 3D entropy field, averaged over
# a 3×3×3 neighborhood. ℏ is projected using geometric curvature scaling.
# Input: "entropy_field_long.npy"
# =============================================================================

import numpy as np

# --- Parameters ---
dx = dy = dz = 0.1
dtau = 0.001  # small step from long run
kB = 1.380649e-23  # J/K

# --- Load entropy data ---
S = np.load("entropy_field_long.npy")
Nx, Ny, Nz, Ntau = S.shape
x0, y0, z0 = Nx//2, Ny//2, Nz//2
tau_idx = Ntau - 2

# --- Compute local H_33 in 3x3x3 region ---
def second_derivative_4d(arr, axis, i, j, k, l, d):
    if axis == 3:
        return (arr[i,j,k,l+1] - 2*arr[i,j,k,l] + arr[i,j,k,l-1]) / d**2
    return 0.0  # only time direction relevant

sum_H33 = 0.0
count = 0

for dx_i in [-1, 0, 1]:
    for dy_i in [-1, 0, 1]:
        for dz_i in [-1, 0, 1]:
            i = x0 + dx_i
            j = y0 + dy_i
            k = z0 + dz_i
            if 1 <= i < Nx-1 and 1 <= j < Ny-1 and 1 <= k < Nz-1:
                H_33 = second_derivative_4d(S, 3, i, j, k, tau_idx, dtau)
                sum_H33 += abs(H_33)
                count += 1

if count == 0:
    raise RuntimeError("No valid Hessian points in volume")

H33_mean = sum_H33 / count
hbar_proj = H33_mean * kB * dtau
hbar_official = 1.054571817e-34
rel_dev = (hbar_proj - hbar_official) / hbar_official * 100

# --- Output ---
print("=== ℏ from Local Time Curvature (C.1.3) ===")
print(f"Samples averaged: {count}")
print(f"<|H_33|>: {H33_mean:.6e}")
print(f"Projected ℏ = {hbar_proj:.8e} J·s")
print(f"Official  ℏ = {hbar_official:.8e} J·s")
print(f"Relative deviation: {rel_dev:.5f}%")
