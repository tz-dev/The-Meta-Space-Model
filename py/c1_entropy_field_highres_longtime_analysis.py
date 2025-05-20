import numpy as np
import matplotlib.pyplot as plt

# --- Parameter ---
Nx, Ny, Nz = 60, 60, 60      # feineres Gitter
Ntau = 1000                  # längere Meta-Zeit
dx = dy = dz = 0.1
dtau = 0.001                 # feinerer Zeitschritt

# --- Initialisierung Entropiefeld ---
S = np.zeros((Nx, Ny, Nz, Ntau))
S[..., 0] = 0.5 + 0.15 * np.random.randn(Nx, Ny, Nz)

D = 0.02
threshold = 0.04

# Zwischenspeicherung alle 100 Schritte
save_interval = 100

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

    # Ausgabe der Statistik alle 100 Schritte
    if t % save_interval == 0 or t == Ntau-1:
        snapshot = S[..., t]
        avg = np.mean(snapshot)
        std = np.std(snapshot)
        s_min = np.min(snapshot)
        s_max = np.max(snapshot)
        print(f"τ = {t*dtau:.3f} | mean = {avg:.5f}, std = {std:.5f}, min = {s_min:.5f}, max = {s_max:.5f}")
        np.save(f"entropy_snapshot_{t}.npy", snapshot)

# Am Ende Gesamtfeld speichern (kann sehr groß sein!)
np.save("entropy_field_long.npy", S)

# Beispiel: Visualisierung des letzten Mittelschnitts
z_mid = Nz // 2
plt.imshow(S[:, :, z_mid, -1], origin='lower', cmap='inferno')
plt.colorbar(label='Entropy S')
plt.title(f'Final mid-plane entropy slice at τ={Ntau*dtau:.3f}')
plt.show()
