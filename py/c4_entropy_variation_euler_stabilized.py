import numpy as np
import matplotlib.pyplot as plt

# Parameter
Nx, Ny, Nz = 50, 50, 50
Ntau = 200
dx = 0.1
dtau = 0.005
alpha = 0.1  # Diffusionskoeffizient
beta = 0.5   # Stärke des Potentialterms

# Initialisierung des Feldes: kleine Zufallsstörung um 0.5
S = np.zeros((Nx, Ny, Nz, Ntau))
S[..., 0] = 0.5 + 0.01 * (np.random.rand(Nx, Ny, Nz) - 0.5)

def laplacian(field):
    # Diskrete Laplace-Operator mit periodischen Randbedingungen
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
        np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
        6 * field
    ) / dx**2

def potential_derivative(s):
    # Beispielhaftes Potenzial: V(s) = s^2 (1-s)^2 → V'(s) = 2 s (1-s)(1-2 s)
    return 2 * s * (1 - s) * (1 - 2 * s)

# Zeitschleife: Euler Integration
for t in range(1, Ntau):
    S_prev = S[..., t-1]
    lap = laplacian(S_prev)
    V_prime = potential_derivative(S_prev)
    S_new = S_prev + dtau * (alpha * lap - beta * V_prime)
    # Grenzen auf [0,1] beschränken (optional)
    S_new = np.clip(S_new, 0, 1)
    S[..., t] = S_new

    if t % 20 == 0:
        mean_val = np.mean(S_new)
        std_val = np.std(S_new)
        min_val = np.min(S_new)
        max_val = np.max(S_new)
        print(f"τ = {t*dtau:.3f} | mean = {mean_val:.5f}, std = {std_val:.5f}, min = {min_val:.5f}, max = {max_val:.5f}")

# Visualisierung: Querschnitt in der Mitte bei Start und Ende
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("S(x,y,z=Nz/2, τ=0)")
plt.imshow(S[:, :, Nz//2, 0], cmap='viridis', origin='lower')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title(f"S(x,y,z=Nz/2, τ={Ntau*dtau:.3f})")
plt.imshow(S[:, :, Nz//2, -1], cmap='viridis', origin='lower')
plt.colorbar()
plt.show()
