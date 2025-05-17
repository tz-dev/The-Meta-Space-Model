import numpy as np

Nx, Ny, Nz, Ntau = 20, 20, 20, 500
dx = 0.1
dtau = 0.001
g = 1.0

A = np.zeros((3, 4, Nx, Ny, Nz))
x = np.linspace(0, Nx*dx, Nx)
y = np.linspace(0, Ny*dx, Ny)
z = np.linspace(0, Nz*dx, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

S_static = np.exp(-((X - Nx*dx/2)**2 + (Y - Ny*dx/2)**2 + (Z - Nz*dx/2)**2)/2.0)
grad = np.gradient(S_static, dx, edge_order=2)

J = np.zeros((3, 4, Nx, Ny, Nz))
for a in range(3):
    J[a, 1] = grad[0]
    J[a, 2] = grad[1]
    J[a, 3] = grad[2]

def levi_civita(i,j,k):
    perm = [(0,1,2),(1,2,0),(2,0,1)]
    if (i,j,k) in perm:
        return 1
    elif (k,j,i) in perm:
        return -1
    else:
        return 0

def partial_derivative(arr, axis, dx):
    return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*dx)

def lorentz_to_spatial_axis(mu):
    return mu - 1

def compute_field_strength(A_t):
    F = np.zeros((3,4,4,Nx,Ny,Nz))
    for a in range(3):
        for mu in range(1,4):
            for nu in range(1,4):
                axis_mu = lorentz_to_spatial_axis(mu)
                axis_nu = lorentz_to_spatial_axis(nu)
                dA_nu = partial_derivative(A_t[a, nu], axis_mu, dx)
                dA_mu = partial_derivative(A_t[a, mu], axis_nu, dx)
                F[a, mu, nu] = dA_mu - dA_nu
        for b in range(3):
            for c in range(3):
                eps = levi_civita(a,b,c)
                if eps != 0:
                    for mu in range(4):
                        for nu in range(4):
                            F[a, mu, nu] += g * eps * A_t[b, mu] * A_t[c, nu]
    return F

def dA_dtau(A_t):
    F = compute_field_strength(A_t)
    dA = np.sum(F, axis=2) + J
    return dA

def rk4_step(A_t, dt):
    k1 = dA_dtau(A_t)
    k2 = dA_dtau(A_t + 0.5*dt*k1)
    k3 = dA_dtau(A_t + 0.5*dt*k2)
    k4 = dA_dtau(A_t + dt*k3)
    return A_t + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

max_norm = 1e2  # Maximalwertbegrenzung

for t in range(Ntau):
    A = rk4_step(A, dtau)
    # Clipping zur Stabilisierung
    norm = np.linalg.norm(A, axis=(0,1))
    clip_mask = norm > max_norm
    if np.any(clip_mask):
        A[:,:,clip_mask] *= (max_norm / norm[clip_mask])

    if t % 50 == 0 or t == Ntau-1:
        mean_vals = np.mean(A, axis=(2,3,4))
        max_vals = np.max(A, axis=(2,3,4))
        print(f"t = {t*dtau:.3f} | Mean A_mu^a: {mean_vals} | Max A_mu^a: {max_vals}")

print("RK4 Simulation done.")
