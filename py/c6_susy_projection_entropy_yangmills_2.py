import numpy as np

def generate_entropy_field(shape=(50, 50, 50, 100), seed=0):
    np.random.seed(seed)
    x, y, z, t = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing='ij')
    S = np.sin(2 * np.pi * t) * np.exp(-((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2) * 10)
    return S

def gamma0():
    return np.diag([1, 1, -1, -1])

def projection_operator():
    return 0.5 * (np.eye(4) + gamma0())

def gradient_entropy(S, dx=0.1, dtau=0.01):
    dS = np.gradient(S, dx, dx, dx, dtau)
    return np.stack(dS, axis=0)  # shape: (4, x, y, z, t)

def apply_projection(P, dS):
    shape = dS.shape[1:]
    norm_projected = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    vec = dS[:, i, j, k, l]
                    proj = P @ vec
                    norm_projected[i, j, k, l] = np.linalg.norm(proj)
    return norm_projected

def symbolic_anticommutator():
    return 2 * np.ones(4)  # placeholder result

if __name__ == "__main__":
    print("--- Generating entropy field for SUSY projection test ---")
    S = generate_entropy_field()
    print("--- Computing gradient ---")
    dS = gradient_entropy(S)
    print("--- Applying projection operator ---")
    P = projection_operator()
    projected_norms = apply_projection(P, dS)
    print("Mean |P * dS| =", np.mean(projected_norms))
    print("Max  |P * dS| =", np.max(projected_norms))
    print("--- Symbolic anticommutator ---")
    print("SUSY Anticommutator:", symbolic_anticommutator())