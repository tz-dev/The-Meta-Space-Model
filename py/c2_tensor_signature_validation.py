import numpy as np

def generate_entropy_field(shape=(50, 50, 50, 600), seed=42):
    np.random.seed(seed)
    x, y, z, t = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing='ij')
    S = np.sin(4 * np.pi * t) * np.exp(-((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2) * 10)
    S += 0.1 * np.random.randn(*shape)
    return S

def compute_4d_hessian(S, dx, dtau, x, y, z, t):
    delta = [dx, dx, dx, dtau]
    H = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            shift_mu = [0]*4
            shift_nu = [0]*4

            shift_mu[mu] = 1
            shift_nu[nu] = 1

            try:
                idx_pp = (x+shift_mu[0]+shift_nu[0], y+shift_mu[1]+shift_nu[1],
                          z+shift_mu[2]+shift_nu[2], t+shift_mu[3]+shift_nu[3])
                idx_pm = (x+shift_mu[0]-shift_nu[0], y+shift_mu[1]-shift_nu[1],
                          z+shift_mu[2]-shift_nu[2], t+shift_mu[3]-shift_nu[3])
                idx_mp = (x-shift_mu[0]+shift_nu[0], y-shift_mu[1]+shift_nu[1],
                          z-shift_mu[2]+shift_nu[2], t-shift_mu[3]+shift_nu[3])
                idx_mm = (x-shift_mu[0]-shift_nu[0], y-shift_mu[1]-shift_nu[1],
                          z-shift_mu[2]-shift_nu[2], t-shift_mu[3]-shift_nu[3])

                H[mu, nu] = (S[idx_pp] - S[idx_pm] - S[idx_mp] + S[idx_mm]) / (4 * delta[mu] * delta[nu])
            except IndexError:
                H[mu, nu] = 0.0
    return H

def analyze_signature(S, dx=0.1, dtau=0.01, sample_points=10):
    shape = S.shape
    results = []
    for _ in range(sample_points):
        x = np.random.randint(2, shape[0]-2)
        y = np.random.randint(2, shape[1]-2)
        z = np.random.randint(2, shape[2]-2)
        t = np.random.randint(2, shape[3]-2)

        H = compute_4d_hessian(S, dx, dtau, x, y, z, t)
        eigvals = np.linalg.eigvalsh(H)
        pos = np.sum(eigvals > 0)
        neg = np.sum(eigvals < 0)
        results.append((x, y, z, t, list(np.round(eigvals, 4)), (pos, neg)))
        print(f"Point ({x},{y},{z},{t}): Signature {pos}+, {neg}- | Eigenvalues: {np.round(eigvals, 4)}")
    return results

if __name__ == "__main__":
    print("--- Generating synthetic entropy field ---")
    S = generate_entropy_field()
    print("--- Analyzing 4D Hessian signatures ---")
    signature_results = analyze_signature(S)