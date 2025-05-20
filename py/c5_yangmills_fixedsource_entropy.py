import numpy as np

# --- Parameter ---
Nx, Ny, Nz = 20, 20, 20
Ntau = 100
dx = 0.1
dtau = 0.01
g = 1.0  # Kopplungskonstante

# --- Gitterkoordinaten ---
x = np.linspace(0, (Nx-1)*dx, Nx)
y = np.linspace(0, (Ny-1)*dx, Ny)
z = np.linspace(0, (Nz-1)*dx, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- Statisches Entropiefeld (als Quelle für A) ---
S_static = np.exp(-((X - Nx*dx/2)**2 + (Y - Ny*dx/2)**2 + (Z - Nz*dx/2)**2) / 2.0)

# --- Feldgrößen: A_mu^a (a=0..2 Lie-Algebra-Index, mu=0..3 Lorentz-Index) ---
# Form: (Lie-Algebra, Lorentz, x, y, z, tau)
A = np.zeros((3, 4, Nx, Ny, Nz, Ntau))

# --- Hilfsfunktionen ---

def levi_civita(i, j, k):
    perms = [(0,1,2), (1,2,0), (2,0,1)]
    if (i,j,k) in perms:
        return 1
    elif (k,j,i) in perms:
        return -1
    else:
        return 0

def partial_derivative(arr, axis, dx):
    # Zentraler Differenzenquotient mit periodischen Randbedingungen
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
        
        # Nichtabelsche Beiträge
        for b in range(3):
            for c in range(3):
                eps = levi_civita(a, b, c)
                if eps != 0:
                    for mu in range(4):
                        for nu in range(4):
                            F[a, mu, nu] += g * eps * A_t[b, mu] * A_t[c, nu]
    return F

# --- Quellterm: Gradient des Entropiefeldes (nur Raumkomponenten mu=1..3) ---
grad_S = np.gradient(S_static, dx, edge_order=2)  # (3, Nx, Ny, Nz)

J = np.zeros((3,4,Nx,Ny,Nz))  # Quellterm J_a^mu
for a in range(3):
    J[a,1] = grad_S[0]
    J[a,2] = grad_S[1]
    J[a,3] = grad_S[2]

# --- Zeitentwicklung ---
for t in range(1, Ntau):
    A_t_minus = A[..., t-1]
    F = compute_field_strength(A_t_minus)
    
    # Addition der Quelle J zum Feld A (ohne nu-Index)
    # Summe von F über nu, um Form anzupassen
    F_sum = np.sum(F, axis=2)  # ergibt Form (3,4,Nx,Ny,Nz)
    
    A[..., t] = A_t_minus + dtau * (F_sum + J)

    # Ausgabe einiger Werte zur Kontrolle
    if t % 10 == 0 or t == 1:
        mean_A = np.mean(A[..., t], axis=(2,3,4))
        max_A = np.max(A[..., t], axis=(2,3,4))
        print(f"t={t*dtau:.2f} | Mean A_mu^a: {mean_A} | Max A_mu^a: {max_A}")

print("Simulation finished.")
