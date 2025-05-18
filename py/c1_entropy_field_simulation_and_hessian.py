import numpy as np
import matplotlib.pyplot as plt

# --- Parameter ---
Nx, Ny, Nz = 50, 50, 50
Ntau = 600
dx = dy = dz = 0.1
dtau = 0.01

# --- Simulation: Entropiefeld ---
S = np.zeros((Nx, Ny, Nz, Ntau))
S[..., 0] = 0.5 + 0.15 * np.random.randn(Nx, Ny, Nz)

D = 0.02
threshold = 0.04

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

# --- Statistik ---
time_steps = [0, Ntau//4, Ntau//2, 3*Ntau//4, Ntau-1]
print("Statistical Summary at Selected τ:")
for t in time_steps:
    snapshot = S[..., t]
    avg = np.mean(snapshot)
    std = np.std(snapshot)
    s_min = np.min(snapshot)
    s_max = np.max(snapshot)
    print(f"  τ = {t*dtau:.2f} → ⟨S⟩ = {avg:.5f}, σ = {std:.5f}, min = {s_min:.5f}, max = {s_max:.5f}")

final_slice = S[..., -1]
max_val = np.max(final_slice)
max_pos = np.unravel_index(np.argmax(final_slice), final_slice.shape)
print(f"\nGlobal max at τ = {Ntau*dtau:.2f}: S = {max_val:.5f} at position (x, y, z) = {max_pos}")

# --- Visualisierung ---
z_mid = Nz // 2
fig, axes = plt.subplots(1, len(time_steps), figsize=(16, 3))
for i, t in enumerate(time_steps):
    im = axes[i].imshow(S[:, :, z_mid, t], origin='lower', cmap='inferno',
                        vmin=np.min(S[:, :, z_mid, t]), vmax=np.max(S[:, :, z_mid, t]))
    axes[i].set_title(f'Meta-time τ={t*dtau:.2f}')
    axes[i].axis('off')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Entropy S')
plt.suptitle('2D Mid-Plane Entropy Slices (Aggressive Nonlinear Dynamics)', fontsize=14)
plt.show()

mean_entropy = np.mean(S, axis=(0, 1, 2))
plt.figure(figsize=(8, 4))
plt.plot(np.arange(Ntau) * dtau, mean_entropy, color='blue')
plt.xlabel('Meta-Time τ')
plt.ylabel('Average Entropy ⟨S⟩')
plt.title('Mean Entropy Evolution with Enhanced Nonlinearity')
plt.grid(True)
plt.show()

# --- 3D Hessian am globalen Maximum (mit Randcheck) ---
x0, y0, z0 = max_pos
if not (1 <= x0 < Nx-1 and 1 <= y0 < Ny-1 and 1 <= z0 < Nz-1):
    print(f"Warning: global max position {max_pos} too close to boundary, using center point (25,25,25).")
    x0, y0, z0 = 25, 25, 25

def compute_hessian_3d(S_3d, dx, x, y, z):
    def second_derivative(arr, axis, i, j, k):
        if axis == 0:
            return (arr[i+1, j, k] - 2*arr[i, j, k] + arr[i-1, j, k]) / dx**2
        elif axis == 1:
            return (arr[i, j+1, k] - 2*arr[i, j, k] + arr[i, j-1, k]) / dx**2
        elif axis == 2:
            return (arr[i, j, k+1] - 2*arr[i, j, k] + arr[i, j, k-1]) / dx**2

    def mixed_derivative(arr, ax1, ax2, i, j, k):
        offsets = {
            (0,1): arr[i+1, j+1, k] - arr[i+1, j-1, k] - arr[i-1, j+1, k] + arr[i-1, j-1, k],
            (0,2): arr[i+1, j, k+1] - arr[i+1, j, k-1] - arr[i-1, j, k+1] + arr[i-1, j, k-1],
            (1,2): arr[i, j+1, k+1] - arr[i, j+1, k-1] - arr[i, j-1, k+1] + arr[i, j-1, k-1],
        }
        return offsets[(ax1, ax2)] / (4*dx**2)

    H = np.zeros((3,3))
    H[0,0] = second_derivative(S_3d, 0, x, y, z)
    H[1,1] = second_derivative(S_3d, 1, x, y, z)
    H[2,2] = second_derivative(S_3d, 2, x, y, z)
    H[0,1] = H[1,0] = mixed_derivative(S_3d, 0, 1, x, y, z)
    H[0,2] = H[2,0] = mixed_derivative(S_3d, 0, 2, x, y, z)
    H[1,2] = H[2,1] = mixed_derivative(S_3d, 1, 2, x, y, z)
    return H

S_final = S[..., -1]
H3d = compute_hessian_3d(S_final, dx, x0, y0, z0)
eigvals3d = np.linalg.eigvalsh(H3d)
print(f"\n3D Hessian I_μν at (x={x0}, y={y0}, z={z0}):\n{np.round(H3d,6)}")
print(f"Eigenvalues (3D metric signature):\n{np.round(eigvals3d,6)}")

# --- 4D Hessian mit Zeit (Meta-Time) ---
tau0 = Ntau-1

def second_derivative_4d(arr, axis, i, j, k, l, dx):
    if axis == 0:
        return (arr[i+1,j,k,l] - 2*arr[i,j,k,l] + arr[i-1,j,k,l]) / dx**2
    elif axis == 1:
        return (arr[i,j+1,k,l] - 2*arr[i,j,k,l] + arr[i,j-1,k,l]) / dx**2
    elif axis == 2:
        return (arr[i,j,k+1,l] - 2*arr[i,j,k,l] + arr[i,j,k-1,l]) / dx**2
    elif axis == 3:
        return (arr[i,j,k,l+1] - 2*arr[i,j,k,l] + arr[i,j,k,l-1]) / dtau**2

def mixed_derivative_4d(arr, ax1, ax2, i, j, k, l, dx):
    offsets = {
        (0,1): arr[i+1,j+1,k,l] - arr[i+1,j-1,k,l] - arr[i-1,j+1,k,l] + arr[i-1,j-1,k,l],
        (0,2): arr[i+1,j,k+1,l] - arr[i+1,j,k-1,l] - arr[i-1,j,k+1,l] + arr[i-1,j,k-1,l],
        (0,3): arr[i+1,j,k,l+1] - arr[i+1,j,k,l-1] - arr[i-1,j,k,l+1] + arr[i-1,j,k,l-1],
        (1,2): arr[i,j+1,k+1,l] - arr[i,j+1,k-1,l] - arr[i,j-1,k+1,l] + arr[i,j-1,k-1,l],
        (1,3): arr[i,j+1,k,l+1] - arr[i,j+1,k,l-1] - arr[i,j-1,k,l+1] + arr[i,j-1,k,l-1],
        (2,3): arr[i,j,k+1,l+1] - arr[i,j,k+1,l-1] - arr[i,j,k-1,l+1] + arr[i,j,k-1,l-1],
    }
    return offsets[(ax1, ax2)] / (4*dx**2)

def compute_hessian_4d(S, dx, dtau, x, y, z, tau):
    Nx, Ny, Nz, Ntau = S.shape
    if not (1 <= x < Nx-1 and 1 <= y < Ny-1 and 1 <= z < Nz-1 and 1 <= tau < Ntau-1):
        raise ValueError("Point too close to boundary for central differences")

    H = np.zeros((4,4))
    H[0,0] = second_derivative_4d(S, 0, x, y, z, tau, dx)
    H[1,1] = second_derivative_4d(S, 1, x, y, z, tau, dx)
    H[2,2] = second_derivative_4d(S, 2, x, y, z, tau, dx)
    H[3,3] = second_derivative_4d(S, 3, x, y, z, tau, dtau)

    axes = [0,1,2,3]
    for i in range(4):
        for j in range(i+1,4):
            val = mixed_derivative_4d(S, axes[i], axes[j], x, y, z, tau, dx)
            H[i,j] = val
            H[j,i] = val
    return H

tau0 = Ntau - 2  # Zweitletzter Zeitschritt, um Randprobleme zu vermeiden

# Prüfe ob x0,y0,z0 auch innerhalb Rand sind
if not (1 <= x0 < Nx-1 and 1 <= y0 < Ny-1 and 1 <= z0 < Nz-1):
    print(f"Warning: spatial point {(x0, y0, z0)} too close to boundary, using center (25,25,25).")
    x0, y0, z0 = 25, 25, 25

H4d = compute_hessian_4d(S, dx, dtau, x0, y0, z0, tau0)
eigvals4d = np.linalg.eigvalsh(H4d)

print(f"\n4D Hessian I_μν at (x={x0}, y={y0}, z={z0}, τ={tau0}):")
print(np.round(H4d, 6))
print("Eigenvalues (4D metric signature):")
print(np.round(eigvals4d, 6))