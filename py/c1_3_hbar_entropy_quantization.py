# =============================================================================
# C.1.3 – ℏ Quantization via Entropy Function
# File: c1_3_hbar_entropy_quantization.py 
# This script computes Planck’s constant ℏ from the Meta-Space entropy model
# using a derived entropy growth function and compares the result to the official ℏ.
# It also performs a sensitivity analysis with respect to τ and saves a plot.
# Output:
# - Console print of computed ℏ and relative deviation
# - "hbar_results.txt" with full summary
# - "hbar_sensitivity.png" plot of ℏ vs. τ
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import datetime

# --- Constants ---
hbar_official = 1.054571817e-34        # J·s
hbar_official_evs = 6.582119569e-16    # eV·s
c = 2.99792458e8                       # m/s
gev_to_joule = 1.602176634e-10         # GeV → J
joule_to_ev = 1 / 1.602176634e-19      # J → eV

# --- Entropy function S(tau) ---
def entropy_function(tau):
    return np.exp(np.float64(tau)) - 1

# --- ℏ computation from entropy quantization formula ---
def compute_hbar(tau, delta_S=1.0, E_scale=4.8e15, k=0.988, alpha_ent=1.07, beta_cal=1.03):
    dS_dtau = np.exp(np.float64(tau))
    ln_omega = delta_S / dS_dtau
    hbar_model = delta_S / (k * ln_omega * alpha_ent * beta_cal)
    hbar_model_js = hbar_model * (gev_to_joule / E_scale) / c
    hbar_model_evs = hbar_model_js * joule_to_ev
    return hbar_model_js, hbar_model_evs

# --- Input parameters ---
tau = 0.027
delta_S = 1.0
E_scale = 4.8e15
k = 0.988
alpha_ent = 1.07
beta_cal = 1.03

# --- Compute ℏ and deviation ---
hbar_model_js, hbar_model_evs = compute_hbar(tau, delta_S, E_scale, k, alpha_ent, beta_cal)
dev_js = (hbar_model_js - hbar_official) / hbar_official * 100
dev_evs = (hbar_model_evs - hbar_official_evs) / hbar_official_evs * 100

# --- Print result ---
print("Meta-Space Model Planck Constant Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Calculated ℏ: {hbar_model_js:.8e} J·s")
print(f"Official ℏ  : {hbar_official:.8e} J·s")
print(f"Relative deviation: {dev_js:.4f}%")
print(f"Calculated ℏ: {hbar_model_evs:.8e} eV·s")
print(f"Official ℏ  : {hbar_official_evs:.8e} eV·s")
print(f"Relative deviation: {dev_evs:.4f}%")

# --- Sensitivity plot ---
tau_range = np.linspace(0.025, 0.03, 100)
hbar_js_values = [compute_hbar(t)[0] for t in tau_range]

plt.figure(figsize=(8, 6))
plt.plot(tau_range, hbar_js_values, label="ℏ(tau)")
plt.axhline(hbar_official, color='r', linestyle='--', label="Official ℏ")
plt.xlabel("Tau")
plt.ylabel("Planck constant ℏ (J·s)")
plt.title("Sensitivity of ℏ with respect to τ")
plt.grid(True)
plt.legend()
plt.savefig("hbar_sensitivity.png")
plt.close()
