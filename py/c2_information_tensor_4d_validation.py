import numpy as np
from scipy.ndimage import gaussian_filter

# Parameters
Nx, Ny, Nz, Ntau = 50, 50, 50, 600

dx = 0.1       # spatial resolution
dtau = 0.01    # meta-time step
sigma = 1.0    # Gaussian smoothing for stability

# Load or generate example 4D entropy field S(x,y,z,tau)
# Replace with np.load("entropy_field.npy") if needed
np.random.seed(42)
S = np.random.rand(Nx, Ny, Nz, Ntau)
S = gaussian_filter(S, sigma=1.5)  # smooth for realistic gradients

# Central point for analysis
x0, y0, z0, t0 = Nx // 2, Ny // 2, Nz // 2, Ntau - 2

# Helper: compute second derivative

def second_derivative(arr, axis, dx_or_dtau):
    return (np.roll(arr, -1, axis=axis) - 2 * arr + np.roll(arr, 1, axis=axis)) / dx_or_dtau**2

# Helper: mixed partial derivatives

def mixed_second_derivative(arr, axis1, axis2, dx1, dx2):
    f_pp = np.roll(np.roll(arr, -1, axis=axis1), -1, axis=axis2)
    f_pn = np.roll(np.roll(arr, -1, axis=axis1), 1, axis=axis2)
    f_np = np.roll(np.roll(arr, 1, axis=axis1), -1, axis=axis2)
    f_nn = np.roll(np.roll(arr, 1, axis=axis1), 1, axis=axis2)
    return (f_pp - f_pn - f_np + f_nn) / (4 * dx1 * dx2)

# Apply Gaussian smoothing before differentiation
S_smooth = gaussian_filter(S, sigma=sigma)

# Build 4D Hessian at point (x0, y0, z0, t0)
H = np.zeros((4, 4))
deltas = [dx, dx, dx, dtau]
labels = ['x', 'y', 'z', 'tau']

for mu in range(4):
    for nu in range(4):
        if mu == nu:
            H[mu, nu] = second_derivative(S_smooth, mu, deltas[mu])[x0, y0, z0, t0]
        else:
            H[mu, nu] = mixed_second_derivative(S_smooth, mu, nu, deltas[mu], deltas[nu])[x0, y0, z0, t0]

# Symmetrize Hessian
H = (H + H.T) / 2

# Eigenvalue analysis
eigvals, eigvecs = np.linalg.eigh(H)

# Output results
print("4D Information Tensor (Hessian) at point (x0, y0, z0, t0):")
print(H)
print("\nEigenvalues (metric signature test):")
print(eigvals)

# Optional: classify signature (Lorentzian-like if one eigenvalue positive and rest negative, or vice versa)
num_positive = np.sum(eigvals > 0)
num_negative = np.sum(eigvals < 0)
print(f"\nSignature: {num_positive} positive, {num_negative} negative eigenvalues")
