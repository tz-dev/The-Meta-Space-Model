import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
hbar_official = 1.054571817e-34  # J·s
hbar_official_evs = 6.582119569e-16  # eV·s
c = 2.99792458e8  # m/s
gev_to_joule = 1.602176634e-10  # GeV to J
joule_to_ev = 1 / 1.602176634e-19  # J to eV

def entropy_function(tau):
    """Entropy function S(tau) from Meta-Space Model (Section 12.14)."""
    return np.exp(np.float64(tau)) - 1

def compute_hbar(tau, delta_S=1.0, E_scale=4.8e15, k=0.988, alpha_ent=1.07, beta_cal=1.03):
    """Compute hbar: S = k * hbar * ln(Omega)."""
    dS_dtau = np.exp(np.float64(tau))
    ln_omega = delta_S / dS_dtau
    hbar_model = delta_S / (k * ln_omega * alpha_ent * beta_cal)
    hbar_model_js = hbar_model * (gev_to_joule / E_scale) / c  # J·s
    hbar_model_evs = hbar_model_js * joule_to_ev  # eV·s
    return hbar_model_js, hbar_model_evs

# Parameters
tau = 0.027
delta_S = 1.0
E_scale = 4.8e15
k = 0.988
alpha_ent = 1.07
beta_cal = 1.03

# Compute hbar
hbar_model_js, hbar_model_evs = compute_hbar(tau, delta_S, E_scale, k, alpha_ent, beta_cal)

# Compute deviance
deviance_js = (hbar_model_js - hbar_official) / hbar_official * 100
deviance_evs = (hbar_model_evs - hbar_official_evs) / hbar_official_evs * 100

# Print results
print(f"Meta-Space Model Planck Constant Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Calculated hbar: {hbar_model_js:.8e} J·s")
print(f"Official hbar: {hbar_official:.8e} J·s")
print(f"Deviance (J·s): {deviance_js:.4f}%")
print(f"Calculated hbar: {hbar_model_evs:.8e} eV·s")
print(f"Official hbar: {hbar_official_evs:.8e} eV·s")
print(f"Deviance (eV·s): {deviance_evs:.4f}%")
print(f"Notes: Tau ({tau}), E_scale ({E_scale} GeV), k ({k}), alpha_ent ({alpha_ent}), beta_cal ({beta_cal}).")

# Save results
with open("hbar_results.txt", "w") as f:
    f.write(f"Meta-Space Model Planck Constant Calculation\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Calculated hbar: {hbar_model_js:.8e} J·s\n")
    f.write(f"Official hbar: {hbar_official:.8e} J·s\n")
    f.write(f"Deviance (J·s): {deviance_js:.4f}%\n")
    f.write(f"Calculated hbar: {hbar_model_evs:.8e} eV·s\n")
    f.write(f"Official hbar: {hbar_official_evs:.8e} eV·s\n")
    f.write(f"Deviance (eV·s): {deviance_evs:.4f}%\n")

# Sensitivity analysis
tau_range = np.linspace(0.025, 0.03, 100)
hbar_js_values = [compute_hbar(t)[0] for t in tau_range]
plt.figure(figsize=(8, 6))
plt.plot(tau_range, hbar_js_values, label="hbar(tau)")
plt.axhline(hbar_official, color='r', linestyle='--', label="Official hbar")
plt.xlabel("Tau")
plt.ylabel("hbar (J·s)")
plt.title("hbar Sensitivity to Tau")
plt.grid(True)
plt.legend()
plt.savefig("hbar_sensitivity.png")
plt.close()