import numpy as np
import matplotlib.pyplot as plt
import datetime

# Offizieller Boltzmann-Wert in eV/K
kB_official = 8.617333262e-5  # eV/K

# Modellparameter
tau = 0.027  # Projektionszeitpunkt
ln_Omega = np.log(np.exp(tau))  # Entropische Mikrostate
T = 1.0  # Kelvin

def compute_kB(E_proj, ln_Omega=ln_Omega, T=T, beta_k=1.0):
    """Berechne kB aus entropischer Energieprojektion."""
    return (E_proj / (ln_Omega * T)) * beta_k  # in eV/K

def optimize_Eproj(kB_target=kB_official, ln_Omega=ln_Omega, T=T, tol=1e-6, max_iter=100):
    """Finde optimalen E_proj für minimale Abweichung zu kB_official."""
    E_min = 1e-6  # eV
    E_max = 1e-1  # eV

    for i in range(max_iter):
        E_mid = (E_min + E_max) / 2
        kB_model = compute_kB(E_mid)
        dev = (kB_model - kB_target) / kB_target * 100
        if abs(dev) < tol:
            return E_mid, kB_model, dev
        if kB_model > kB_target:
            E_max = E_mid
        else:
            E_min = E_mid
    # fallback
    return E_mid, kB_model, dev

# Optimierung ausführen
E_opt, kB_calc, deviance = optimize_Eproj()

# Ergebnisse anzeigen
print("Meta-Space Model Boltzmann Constant Calculation (Entropy-Based)")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Optimal tau: {tau:.8f}")
print(f"Optimal projected energy E_proj: {E_opt:.10e} eV")
print(f"Calculated kB: {kB_calc:.10e} eV/K")
print(f"Official kB:   {kB_official:.10e} eV/K")
print(f"Relative deviance: {deviance:.4f}%")

# Plot Sensitivität
E_vals = np.linspace(E_opt * 0.8, E_opt * 1.2, 100)
kB_vals = [compute_kB(e) for e in E_vals]

plt.figure(figsize=(8, 6))
plt.plot(E_vals, kB_vals, label="kB(E_proj)")
plt.axhline(kB_official, color='r', linestyle='--', label="Official kB")
plt.xlabel("Projected Energy E_proj [eV]")
plt.ylabel("Boltzmann Constant kB [eV/K]")
plt.title("Sensitivity of kB to E_proj")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("kB_projection_sensitivity.png")
plt.close()
