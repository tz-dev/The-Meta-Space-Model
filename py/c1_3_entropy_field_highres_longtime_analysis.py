# =============================================================================
# File: c1_3_entropy_field_highres_longtime_analysis.py
# High-resolution 3D entropy field simulation over long meta-time
# Assumptions: dx=0.1, dtau=0.001, Nx=Ny=Nz=60, Ntau=1000
# Simulation involves nonlinear diffusion with gradient-dependent source
# Periodic snapshots saved every 100 τ; final slice visualized
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
Nx, Ny, Nz = 60, 60, 60
Ntau = 1000
dx = dy = dz = 0.1
dtau = 0.001

# --- Entropy Field Initialization ---
S = np.zeros((Nx, Ny, Nz, Ntau))
S[..., 0] = 0.5 + 0.15 * np.random.randn(Nx, Ny, Nz)

D = 0.02
threshold = 0.04
save_interval = 100

# --- Simulation ---
for t in range(1, Ntau):
    laplacian_S = (
        np.roll(S[..., t-1], 1, axis=0) + np.roll(S[..., t-1], -1, axis=0) +
        np.roll(S[..., t-1], 1, axis=1) + np.roll(S[..., t-1], -1, axis=1) +
        np.roll(S[..., t-1], 1, axis=2) + np.roll(S[..., t-1], -1, axis=2) -
        6 * S[..., t-1]
    ) / (dx**2)

    grad_x, grad_y, grad_z = np.gradient(S[..., t-1], dx, edge_order=2)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    source = 0.5 * np.tanh(20 * (grad_magnitude - threshold)) * (grad_magnitude > threshold)
    S[..., t] = S[..., t-1] + dtau * (D * laplacian_S + source)

    if t % save_interval == 0 or t == Ntau - 1:
        snapshot = S[..., t]
        avg = np.mean(snapshot)
        std = np.std(snapshot)
        s_min = np.min(snapshot)
        s_max = np.max(snapshot)
        print(f"τ = {t*dtau:.3f} | mean = {avg:.5f}, std = {std:.5f}, min = {s_min:.5f}, max = {s_max:.5f}")
        np.save(f"entropy_snapshot_{t}.npy", snapshot)

# --- Save full field ---
np.save("entropy_field_long.npy", S)

# --- Visualize mid-plane slice ---
z_mid = Nz // 2
plt.imshow(S[:, :, z_mid, -1], origin='lower', cmap='inferno')
plt.colorbar(label='Entropy S')
plt.title(f'Final mid-plane entropy slice at τ={Ntau*dtau:.3f}')
plt.show()
