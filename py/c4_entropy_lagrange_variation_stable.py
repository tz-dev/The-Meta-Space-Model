import numpy as np
import matplotlib.pyplot as plt

Nx, Ny, Nz = 30, 30, 30
Ntau = 200
dx = 0.1
dtau = 0.001  # kleinerer Schritt für Stabilität

alpha = 1.0
beta = 0.5

S = np.ones((Nx, Ny, Nz, Ntau)) * 0.5
S[..., 0] += 0.05 * np.random.randn(Nx, Ny, Nz)

def discrete_laplacian(field, dx):
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
        np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
        6 * field
    ) / (dx**2)

def potential_derivative(S_field):
    # Stabilere Berechnung (Vermeidung von Overflows)
    return 2 * S_field * (1 - S_field) * (1 - 2 * S_field)

for t in range(1, Ntau):
    S_prev = S[..., t-1]
    lap = discrete_laplacian(S_prev, dx)
    V_prime = potential_derivative(S_prev)

    S_new = S_prev + dtau * (alpha * lap - beta * V_prime)
    S_new = np.clip(S_new, 0, 1)  # Werte begrenzen

    if np.any(np.isnan(S_new)) or np.any(np.isinf(S_new)):
        print(f"Numerical instability detected at step {t}, aborting simulation.")
        break

    S[..., t] = S_new

# Ausgabe einiger numerischer Werte
time_steps = [0, Ntau//4, Ntau//2, 3*Ntau//4, Ntau-1]
print("Statistical Summary at Selected τ:")
for t in time_steps:
    snapshot = S[..., t]
    print(f"τ = {t*dtau:.3f}: mean={np.mean(snapshot):.5f}, std={np.std(snapshot):.5f}, min={np.min(snapshot):.5f}, max={np.max(snapshot):.5f}")

# Visualisierung (wie vorher)
z_mid = Nz // 2
fig, axes = plt.subplots(1, len(time_steps), figsize=(16, 3))
for i, t in enumerate(time_steps):
    im = axes[i].imshow(S[:, :, z_mid, t], origin='lower', cmap='viridis',
                        vmin=0, vmax=1)
    axes[i].set_title(f'τ = {t*dtau:.3f}')
    axes[i].axis('off')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='S Field')
plt.suptitle('Discrete Euler-Lagrange Evolution of Entropy Field (stabilized)')
plt.show()
