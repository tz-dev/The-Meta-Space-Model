# ========================================================
# File: c7_1_montecarlo_entropic_lensing.py
# Purpose: Simulate entropic lensing effect via Monte Carlo sampling of 2D entropy gradient field
# Method:
#   - Generate random 2D points in [-1,1]^2
#   - Define entropy field S(r) = exp(-r^2) centered at origin
#   - Compute numerical gradient of entropy field at sample points
#   - Normalize gradient vectors as approximate light ray directions
#   - Calculate lensing angles and statistics
# Output:
#   - Print mean and standard deviation of deviation angles in degrees
#   - Save quiver plot of entropic lensing rays
# ========================================================

import numpy as np
import matplotlib.pyplot as plt

# === Monte Carlo sampling ===
np.random.seed(42)
N = 1_000_000
x = np.random.uniform(-1, 1, N)
y = np.random.uniform(-1, 1, N)

# === Entropy field (spherical peak at center) ===
r = np.sqrt(x**2 + y**2)
S = np.exp(-r**2)

# === Numerical gradients ===
dS_dx = -2 * x * np.exp(-r**2)
dS_dy = -2 * y * np.exp(-r**2)

# === Normalize gradient vectors for ray directions ===
norm = np.sqrt(dS_dx**2 + dS_dy**2) + 1e-10  # avoid division by zero
dx_ray = dS_dx / norm
dy_ray = dS_dy / norm

# === Calculate deviation angles (radians and degrees) ===
theta = np.arctan2(dy_ray, dx_ray)
theta_deg = np.degrees(theta)

# === Statistics ===
theta_mean = np.mean(theta_deg)
theta_std = np.std(theta_deg)

# === Output ===
print("=== Monte Carlo Entropic Lensing Prediction ===")
print(f"Samples: {N}")
print(f"Mean deviation angle: {theta_mean:.4f}°")
print(f"Standard deviation:   {theta_std:.4f}°")

# === Plot (optional) ===
plt.figure(figsize=(6, 6))
plt.quiver(x, y, dx_ray, dy_ray, color='blue', alpha=0.5, scale=20)
plt.title("Monte Carlo Entropic Lensing Rays")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.savefig("img/c7_1_entropic_lensing_rays.png")
plt.close()
