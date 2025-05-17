import numpy as np

# Parameter
Nx, Ny, Nz = 20, 20, 20
Ntau = 100
dx = 0.1
dtau = 0.01
D = 0.1           # Diffusionskoeffizient Entropiefeld
alpha = 1.0       # Potentialkoeffizient
beta = 0.05       # Kopplung A -> S
gamma = 0.1       # Kopplung S -> A
g = 1.0           # Yang-Mills Kopplungskonstante

# Initialisierung Gitter
x = np.linspace(0, (Nx-1)*dx, Nx)
y = np.linspace(0, (Ny-1)*dx, Ny)
z = np.linspace(0, (Nz-1)*dx, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Entropiefeld initial
S = np.zeros((Nx, Ny, Nz, Ntau))
S[..., 0] = 0.5 + 0.01 * np.random.randn(Nx, Ny, Nz)  # kleines Rauschen

# Yang-Mills-Feld initial (3 Lie-Indizes, 4 Lorentz-Indizes)
A = np.zeros((3, 4, Nx, Ny, Nz, Ntau))

def laplacian(field):
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
        np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
        6 * field
    ) / dx**2

def V_prime(S):
    # Ableitung des Potentials V(S) = S^2 (1 - S)^2 (Beispiel)
    return 2 * S * (1 - S) * (1 - 2 * S)

def partial_derivative(arr, axis):
    return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2 * dx)

def levi_civita(i, j, k):
    perms = [(0,1,2),(1,2,0),(2,0,1)]
    if (i,j,k) in perms:
        return 1
    elif (k,j,i) in perms:
        return -1
    else:
        return 0

def compute_field_strength(A_t):
    F = np.zeros_like(A_t)
    axis_map = {1: 0, 2:1, 3:2}
    for a in range(3):
        for mu in range(4):
            for nu in range(4):
                if mu == 0 or nu == 0:
                    dA_mu = 0
                    dA_nu = 0
                else:
                    axis_mu = axis_map[mu]
                    axis_nu = axis_map[nu]
                    dA_nu = partial_derivative(A_t[a, nu], axis_mu)
                    dA_mu = partial_derivative(A_t[a, mu], axis_nu)
                F[a, mu] += dA_mu - dA_nu
        # Nichtabelsche Anteile
        for b in range(3):
            for c in range(3):
                eps = levi_civita(a,b,c)
                if eps != 0:
                    F[a] += g * eps * A_t[b] * A_t[c]
    return F

for t in range(1, Ntau):
    S_prev = S[..., t-1]
    A_prev = A[..., t-1]

    lap = laplacian(S_prev)
    Vp = V_prime(S_prev)

    # Kopplung A -> S (Norm von A)
    J_coupling = gamma * np.sqrt(np.sum(A_prev**2, axis=(0,1)))

    # Update Entropiefeld mit diskreter Variation
    S[..., t] = S_prev + dtau * (D * lap - alpha * Vp + J_coupling)

    # Yang-Mills Feldstärke
    F = compute_field_strength(A_prev)

    # Rückkopplung S -> A über Gradienten von S
    grad_S = np.array(np.gradient(S_prev, dx))

    J_A = np.zeros_like(A_prev)
    for a in range(3):
        for mu in range(1,4):
            J_A[a, mu] = beta * grad_S[mu-1]

    # Update A
    A[..., t] = A_prev + dtau * (F + J_A)

    # Ausgabe für Monitoring
    if t % 10 == 0 or t == 1:
        print(f"tau={t*dtau:.3f} | <S>={np.mean(S[..., t]):.5f} | max S={np.max(S[..., t]):.5f} | max A={np.max(A[..., t]):.5f}")

print("Diskrete Variation Simulation abgeschlossen.")
