# =============================================================================
# C.1.3 – Planck Constant from Hessian Geometry of Entropy Field
# File: c1_3_hbar_from_hessian_geometry.py
# This script calculates an approximation of ℏ based on local second derivatives
# of the entropy field in 4D (3 space + 1 meta-time). It uses the geometric
# information curvature approach and compares the result to the official ℏ.
# Input: "entropy_field_long.npy" from C.1.1/C.1.2
# =============================================================================

import numpy as np

# --- Constants ---
kB = 1.380649e-23                 # Boltzmann constant (J/K)
hbar_official = 1.054571817e-34  # Planck constant (J·s)
path = "entropy_field_long.npy"

# --- Load data ---
S = np.load(path)
dx = dy = dz = 0.1
dtau = 0.01  # adjust if necessary
Ntau = S.shape[-1]

# --- Select center voxel ---
x0 = y0 = z0 = 25
tau_idx = Ntau - 2

# --- 2nd-order finite difference in 4D ---
def second_derivative(arr, axis, i, j, k, l, dx):
    if axis == 0:
        return (arr[i+1,j,k,l] - 2*arr[i,j,k,l] + arr[i-1,j,k,l]) / dx**2
    elif axis == 1:
        return (arr[i,j+1,k,l] - 2*arr[i,j,k,l] + arr[i,j-1,k,l]) / dx**2
    elif axis == 2:
        return (arr[i,j,k+1,l] - 2*arr[i,j,k,l] + arr[i,j,k-1,l]) / dx**2
    elif axis == 3:
        return (arr[i,j,k,l+1] - 2*arr[i,j,k,l] + arr[i,j,k,l-1]) / dtau**2

def compute_4d_hessian(S, x, y, z, t):
    H = np.zeros((4, 4))
    for i in range(4):
        H[i, i] = second_derivative(S, i, x, y, z, t, dx)
    return H

# --- Eigenvalue-based curvature measure ---
H = compute_4d_hessian(S, x0, y0, z0, tau_idx)
eigvals = np.linalg.eigvalsh(H)
abs_eigs = np.abs(eigvals)
I_geo = np.mean(abs_eigs)

# --- Project ℏ from curvature scale ---
tau_eff = dtau
hbar_proj = I_geo * kB * tau_eff
rel_dev = (hbar_proj - hbar_official) / hbar_official * 100

# --- Output ---
print("=== ℏ from Geometric Information Curvature ===")
print(f"⟨|λ_H|⟩ = {I_geo:.5e}")
print(f"Projected ℏ = {hbar_proj:.8e} J·s")
print(f"Official ℏ  = {hbar_official:.8e} J·s")
print(f"Relative deviation: {rel_dev:.5f}%")
