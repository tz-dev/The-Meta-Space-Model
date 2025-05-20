import numpy as np
import matplotlib.pyplot as plt

# Parameter
Nx, Ny, Nz = 20, 20, 20
Ntau = 100
dx = 0.1
dtau = 0.01
g = 1.0

# Levi-Civita für SU(2)
def levi_civita(i, j, k):
    perms = [(0,1,2),(1,2,0),(2,0,1)]
    if (i,j,k) in perms:
        return 1
    elif (k,j,i) in perms:
        return -1
    else:
        return 0

# Zentraler Differenzenquotient mit periodischen Randbedingungen
def partial_derivative(arr, axis, dx):
    return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*dx)

def compute_field_strength(A_t):
    F = np.zeros((3,4,4,Nx,Ny,Nz))
    axis_map = {1: 0, 2: 1, 3: 2}
    for a in range(3):
        for mu in range(4):
            for nu in range(4):
                if mu == 0 or nu == 0:
                    dA_mu = 0
                    dA_nu = 0
                else:
                    axis_mu = axis_map[mu]
                    axis_nu = axis_map[nu]
                    dA_nu = partial_derivative(A_t[a, nu], axis=axis_mu, dx=dx)
                    dA_mu = partial_derivative(A_t[a, mu], axis=axis_nu, dx=dx)
                F[a, mu, nu] = dA_mu - dA_nu
        for b in range(3):
            for c in range(3):
                eps = levi_civita(a, b, c)
                if eps != 0:
                    for mu in range(4):
                        for nu in range(4):
                            F[a, mu, nu] += g * eps * A_t[b, mu] * A_t[c, nu]
    return F

def compute_energy_density(F):
    # Summiere über alle Indizes a, mu, nu und quadriere das Feld
    return np.sum(F**2, axis=(0,1,2))

# Gitterkoordinaten & Initialisierung
x = np.linspace(0, (Nx-1)*dx, Nx)
y = np.linspace(0, (Ny-1)*dx, Ny)
z = np.linspace(0, (Nz-1)*dx, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Statisches Entropiefeld (Quelle)
S_static = np.exp(-((X - Nx*dx/2)**2 + (Y - Ny*dx/2)**2 + (Z - Nz*dx/2)**2) / 2.0)

# Feldgrößen A_mu^a
A = np.zeros((3,4,Nx,Ny,Nz,Ntau))

# Quellterm: Gradient von S_static (nur Raumkomponenten mu=1..3)
grad_S = np.gradient(S_static, dx, edge_order=2)
J = np.zeros((3,4,Nx,Ny,Nz))
for a in range(3):
    J[a,1] = grad_S[0]
    J[a,2] = grad_S[1]
    J[a,3] = grad_S[2]

# Zeitentwicklung und Speicherung Energiedichte
energy_means = []
energy_maxs = []

for t in range(1, Ntau):
    A_t_minus = A[..., t-1]
    F = compute_field_strength(A_t_minus)
    J_expanded = np.repeat(J[:, :, np.newaxis, :, :, :], 4, axis=2)
    
    F_reduced = np.sum(F, axis=2)      # von (3,4,4,x,y,z) zu (3,4,x,y,z)
    J_reduced = np.sum(J_expanded, axis=2)
    
    A[..., t] = A_t_minus + dtau * (F_reduced + J_reduced)
    
    energy = compute_energy_density(F)
    mean_e = np.mean(energy)
    max_e = np.max(energy)
    energy_means.append(mean_e)
    energy_maxs.append(max_e)
    
    if t % 10 == 0 or t == 1:
        print(f"t = {t*dtau:.2f} | Mean Energy Density = {mean_e:.6e} | Max Energy Density = {max_e:.6e}")


print("Simulation finished.")

plt.plot(np.arange(1, Ntau)*dtau, energy_means, label='Mean Energy Density')
plt.plot(np.arange(1, Ntau)*dtau, energy_maxs, label='Max Energy Density')
plt.xlabel('Meta-Time τ')
plt.ylabel('Energy Density')
plt.legend()
plt.title('Energy Density Evolution of Yang-Mills Field')
plt.grid()
plt.show()
