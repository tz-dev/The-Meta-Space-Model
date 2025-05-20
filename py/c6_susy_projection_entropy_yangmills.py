import numpy as np

# --- Supersymmetrie-Projektionsalgebra (symbolisch + feldgebunden) ---

# Definition einfacher SUSY-Generatoren Q_alpha (symbolisch, 2-Komponenten Majorana)
# Hier repräsentiert als reine Antikommutatorstruktur

def susy_anticommutator(Qa, Qb):
    # Symbolischer SUSY-Antikommutator (vereinfachte Struktur, mit Minkowski-Metrik eta)
    # {Q_alpha, Q_beta} = 2 * gamma^mu_{alpha beta} * P_mu
    # Für symbolische Zwecke: geben wir den Tensor-Index (mu) direkt als Outputstruktur zurück
    gamma = np.identity(4)  # Vereinfachte gamma^mu-Matrizen (DiagBasis)
    P_mu = np.ones(4)       # Impulsoperator symbolisch = 1 in allen Komponenten
    return 2 * np.tensordot(gamma, P_mu, axes=1)  # ergibt ein 4er-Vektor

# --- Projektionsbedingung (z.B. BPS-artig): Q * state = 0 ---
# In unserem Kontext: Anforderungen an Felder, dass bestimmte SUSY-Projektionen sie annihilieren

# Beispielhafte Projektionsbedingung auf ein skalares Entropiefeld S(x,y,z,tau)
# sowie auf ein Yang-Mills Feld A^a_mu(x,y,z,tau)

Nx, Ny, Nz = 10, 10, 10
Ntau = 20
dx = 0.1
dtau = 0.01

# Initialisierte Felder S und A (einfach gehalten)
S = 0.5 + 0.01 * np.random.randn(Nx, Ny, Nz, Ntau)
A = np.zeros((3, 4, Nx, Ny, Nz, Ntau))  # a = 0..2, mu = 0..3

# Projektionsoperator P = 1/2 (1 + Gamma)
# Beispielhaft: Gamma als Zeit-Reflexion (hier numerisch: -1 im Zeitindex)
P_matrix = np.diag([1, 1, 1, -1])  # Projektionsmatrix (1 + gamma^0) 

# Prüfe auf S(x,y,z,tau), ob P * grad S = 0 erfüllt ist (symbolisch = Projektion)
def check_entropy_projection(S):
    grad = np.gradient(S, dx, edge_order=2)
    projected = np.zeros_like(grad[0])
    for mu in range(4):
        weight = P_matrix[mu, mu]
        if mu == 0:
            # Zeitrichtung (tau)
            dS = np.gradient(S, dtau, axis=3)
        else:
            dS = grad[mu-1]  # spatial
        projected += weight * dS
    return projected

# Anwendung auf S
projection_S = check_entropy_projection(S)

print("\nSUSY-Projection Norm (Entropy Field):")
print("Mean |P * dS| =", np.mean(np.abs(projection_S)))
print("Max  |P * dS| =", np.max(np.abs(projection_S)))

# Anwendung auf Yang-Mills Feld A^a_mu: einfache Projektionsprüfung (nur auf mu)
def check_vector_projection(A):
    result = 0
    for mu in range(4):
        weight = P_matrix[mu, mu]
        result += weight * np.mean(np.abs(A[:, mu]))
    return result

proj_A = check_vector_projection(A)

print("\nSUSY-Projection Norm (Yang-Mills Field):")
print("Mean weighted A_mu^a =", proj_A)

# Symbolische SUSY-Struktur als Antikommutator-Test (vereinfachte Struktur):
Q1, Q2 = np.array([1, 0]), np.array([0, 1])
ac = susy_anticommutator(Q1, Q2)
print("\nSymbolic SUSY Anticommutator Result (tensor form):")
print(ac)

# Ende
