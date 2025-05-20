import numpy as np
import matplotlib.pyplot as plt

# Parameter tau (RG-Skala)
tau = np.linspace(0, 40, 1000)
dt = tau[1] - tau[0]

# Anfangswerte Yukawa-Matrix-Elemente (z.B. Top-Bottom Mixing)
y11 = np.zeros_like(tau)
y12 = np.zeros_like(tau)
y21 = np.zeros_like(tau)
y22 = np.zeros_like(tau)

y11[0] = 0.5   # Startwerte
y12[0] = 0.1
y21[0] = 0.1
y22[0] = 0.3

# Vereinfachte Kopplungen (als Konstanten für Beispiel)
alpha1 = 0.0169
alpha2 = 0.0338
alpha3 = 0.118

def beta_y(y_ij, y_matrix):
    y_t = y_ij
    y_sq_sum = np.sum(y_matrix**2)
    return (y_t / (16*np.pi**2)) * ((9/2) * y_sq_sum - ((17/12)*alpha1 + (9/4)*alpha2 + 8*alpha3))

# RG-Integration
for i in range(1, len(tau)):
    Y = np.array([[y11[i-1], y12[i-1]], [y21[i-1], y22[i-1]]])
    
    y11[i] = y11[i-1] + beta_y(y11[i-1], Y) * dt
    y12[i] = y12[i-1] + beta_y(y12[i-1], Y) * dt
    y21[i] = y21[i-1] + beta_y(y21[i-1], Y) * dt
    y22[i] = y22[i-1] + beta_y(y22[i-1], Y) * dt
    
    if np.any(np.isnan([y11[i], y12[i], y21[i], y22[i]])) or np.any(np.abs([y11[i], y12[i], y21[i], y22[i]]) > 10):
        y11[i:] = np.nan
        y12[i:] = np.nan
        y21[i:] = np.nan
        y22[i:] = np.nan
        break

# Eigenwerte der Yukawa-Matrix (physikalische Massen) berechnen
eigvals = np.zeros((len(tau), 2))
for i in range(len(tau)):
    Y = np.array([[y11[i], y12[i]], [y21[i], y22[i]]])
    eigvals[i] = np.linalg.eigvals(Y)

# Numerische Werte an ausgewählten Punkten ausgeben
check_points = [0, 100, 300, 500, 700, 900, 999]  # Indizes für tau ~ 0,4,12,20,28,36,40
print("Selected RG values for Yukawa Mass Eigenvalues:")
for idx in check_points:
    print(f"τ = {tau[idx]:.1f}  →  Mass eigenvalue 1 = {eigvals[idx,0]:.5f},  Mass eigenvalue 2 = {eigvals[idx,1]:.5f}")

# Plot
plt.figure(figsize=(8,6))
plt.plot(tau, eigvals[:,0], label="Mass eigenvalue 1")
plt.plot(tau, eigvals[:,1], label="Mass eigenvalue 2")
plt.xlabel("τ (RG scale)")
plt.ylabel("Yukawa Mass Eigenvalues")
plt.title("RG Flow of Yukawa Mass Matrix Eigenvalues")
plt.grid(True)
plt.legend()
plt.savefig("c15_mass_coupling_matrix.png")
plt.show()
