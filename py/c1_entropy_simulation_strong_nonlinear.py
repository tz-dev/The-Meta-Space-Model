import numpy as np
import matplotlib.pyplot as plt

# Grid & time setup
Nx, Ny, Nz = 50, 50, 50
Ntau = 600
dx = dy = dz = 0.1
dtau = 0.01

# Initialize entropy field with stronger fluctuations
S = np.zeros((Nx, Ny, Nz, Ntau))
S[..., 0] = 0.5 + 0.15 * np.random.randn(Nx, Ny, Nz)

D = 0.02  # reduced diffusion
threshold = 0.04  # activation threshold for source

for t in range(1, Ntau):
    laplacian_S = (
        np.roll(S[..., t-1], 1, axis=0) + np.roll(S[..., t-1], -1, axis=0) +
        np.roll(S[..., t-1], 1, axis=1) + np.roll(S[..., t-1], -1, axis=1) +
        np.roll(S[..., t-1], 1, axis=2) + np.roll(S[..., t-1], -1, axis=2) -
        6 * S[..., t-1]
    ) / (dx**2)

    grad_x, grad_y, grad_z = np.gradient(S[..., t-1], dx, edge_order=2)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    # Strong nonlinear source: tanh behavior, sharp onset
    source = 0.5 * np.tanh(20 * (grad_magnitude - threshold)) * (grad_magnitude > threshold)

    S[..., t] = S[..., t-1] + dtau * (D * laplacian_S + source)

# Statistics at selected τ
time_steps = [0, Ntau//4, Ntau//2, 3*Ntau//4, Ntau-1]
print("Statistical Summary at Selected τ:")
for t in time_steps:
    snapshot = S[:, :, :, t]
    avg = np.mean(snapshot)
    std = np.std(snapshot)
    s_min = np.min(snapshot)
    s_max = np.max(snapshot)
    print(f"  τ = {t*dtau:.2f} → ⟨S⟩ = {avg:.5f}, σ = {std:.5f}, min = {s_min:.5f}, max = {s_max:.5f}")

# Max position
final_slice = S[:, :, :, -1]
max_val = np.max(final_slice)
max_pos = np.unravel_index(np.argmax(final_slice), final_slice.shape)
print(f"\nGlobal max at τ = {Ntau*dtau:.2f}: S = {max_val:.5f} at position (x, y, z) = {max_pos}")

# Mid-plane visualization
z_mid = Nz // 2
fig, axes = plt.subplots(1, len(time_steps), figsize=(16, 3))
for i, t in enumerate(time_steps):
    im = axes[i].imshow(S[:, :, z_mid, t], origin='lower', cmap='inferno',
                        vmin=np.min(S[:, :, z_mid, t]), vmax=np.max(S[:, :, z_mid, t]))
    axes[i].set_title(f'Meta-time τ={t*dtau:.2f}')
    axes[i].axis('off')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Entropy S')
plt.suptitle('2D Mid-Plane Entropy Slices (Aggressive Nonlinear Dynamics)', fontsize=14)
fig, axes = plt.subplots(1, len(time_steps), figsize=(16, 3), constrained_layout=True)
plt.show()

# Mean entropy over time
mean_entropy = np.mean(S, axis=(0, 1, 2))
plt.figure(figsize=(8, 4))
plt.plot(np.arange(Ntau) * dtau, mean_entropy, color='blue')
plt.xlabel('Meta-Time τ')
plt.ylabel('Average Entropy ⟨S⟩')
plt.title('Mean Entropy Evolution with Enhanced Nonlinearity')
plt.grid(True)
plt.show()
np.save("entropy_field.npy", S)