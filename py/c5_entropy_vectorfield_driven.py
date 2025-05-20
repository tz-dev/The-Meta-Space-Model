import numpy as np
import matplotlib.pyplot as plt

Nx, Ny, Nz, Ntau = 20, 20, 20, 100
dx = dtau = 0.1

# Initialisiere Vektorpotential A_mu
A = np.zeros((4, Nx, Ny, Nz, Ntau))

# Gitterdefinition
x = np.linspace(0, Nx*dx, Nx)
y = np.linspace(0, Ny*dx, Ny)
z = np.linspace(0, Nz*dx, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Statisches Entropiefeld mit glattem Peak
S_static = np.exp(-((X - Nx*dx/2)**2 + (Y - Ny*dx/2)**2 + (Z - Nz*dx/2)**2) / 2.0)

# Quelle J_mu als Gradient von S_static
J = np.zeros((4, Nx, Ny, Nz))
J[1], J[2], J[3] = np.gradient(S_static, dx, edge_order=2)

# Zeitentwicklung des Vektorpotentials A
for t in range(1, Ntau):
    A[:, :, :, :, t] = A[:, :, :, :, t-1] + dtau * J[:, :, :, :]

    # Numerische Ausgabe alle 10 Zeitschritte
    if t % 10 == 0 or t == Ntau-1:
        mean_A = np.mean(A[:, :, :, :, t], axis=(1, 2, 3))
        max_A = np.max(A[:, :, :, :, t], axis=(1, 2, 3))
        print(f"t = {t*dtau:.2f} | Mean A_mu: {mean_A} | Max A_mu: {max_A}")

# Visualisierung der A_x-Komponente am letzten Zeitschritt im Mittelschnitt z=Nz//2
plt.imshow(A[1, :, :, Nz//2, -1], origin='lower', cmap='seismic')
plt.colorbar(label='A_x at z mid, final time')
plt.title('Vector potential A_x slice at final time step')
plt.show()
