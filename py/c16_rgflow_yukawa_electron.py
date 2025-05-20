import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
hbar_official = 1.054571817e-34  # Official Planck constant in J·s
hbar_official_evs = 6.582119569e-16  # Official Planck constant in eV·s
c = 2.99792458e8  # Speed of light in m/s
gev_to_joule = 1.602176634e-10  # Conversion from GeV to Joule
ev_to_gev = 1e-9  # Conversion from eV to GeV

# Entropy function and quantization
def entropy_function(tau, tau_0=1.0):
    """
    Entropy function S(tau) from Meta-Space Model (Section 12.14).
    tau: entropic time parameter
    tau_0: reference scale (tuned for minimal projection unit)
    """
    return np.exp(tau / tau_0) - 1

def compute_hbar(tau, delta_S=1.0, E_scale=1e16):
    """
    Compute hbar using entropic quantization: S = hbar * ln(Omega).
    delta_S: minimal entropy unit (set to 1 per roadmap)
    E_scale: energy scale in GeV (GUT scale, C.10)
    """
    # Compute ln(Omega) from entropy gradient
    dS_dtau = np.exp(tau)  # Derivative of S = e^tau - 1
    ln_omega = delta_S / dS_dtau  # ln(Omega) = Delta S / (dS/dtau)
    
    # Scale to physical units
    # hbar = Delta S / ln(Omega) in model units, convert to J·s
    hbar_model = delta_S / ln_omega  # In entropic units
    # Convert to physical units using E_scale (GeV) and c
    hbar_model_js = hbar_model * (gev_to_joule / E_scale) / c  # J·s
    hbar_model_evs = hbar_model_js / 1.602176634e-19 * ev_to_gev  # eV·s
    
    return hbar_model_js, hbar_model_evs

# Parameters
tau = 0.1  # Entropic time near minimal projection (tuned)
delta_S = 1.0  # Minimal entropy unit (A.7, B.7)
E_scale = 1e16  # GUT scale in GeV (C.10)

# Compute hbar
hbar_model_js, hbar_model_evs = compute_hbar(tau, delta_S, E_scale)

# Compute deviance
deviance_js = (hbar_model_js - hbar_official) / hbar_official * 100  # in %
deviance_evs = (hbar_model_evs - hbar_official_evs) / hbar_official_evs * 100  # in %

# Print results
print(f"Meta-Space Model Planck Constant Calculation")
print(f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"----------------------------------------")
print(f"Calculated Planck constant (hbar): {hbar_model_js:.8e} J·s")
print(f"Official Planck constant: {hbar_official:.8e} J·s")
print(f"Relative deviance (J·s): {deviance_js:.4f}%")
print(f"Calculated Planck constant (hbar): {hbar_model_evs:.8e} eV·s")
print(f"Official Planck constant: {hbar_official_evs:.8e} eV·s")
print(f"Relative deviance (eV·s): {deviance_evs:.4f}%")
print(f"----------------------------------------")
print(f"Notes for approximation:")
print(f"- Tau ({tau}) tuned for minimal projection unit.")
print(f"- E_scale ({E_scale} GeV) based on GUT scale (C.10).")
print(f"- Adjust tau or E_scale to reduce deviance.")

# Save results to a file
with open("hbar_results.txt", "w") as f:
    f.write(f"Meta-Space Model Planck Constant Calculation\n")
    f.write(f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"----------------------------------------\n")
    f.write(f"Calculated Planck constant (hbar): {hbar_model_js:.8e} J·s\n")
    f.write(f"Official Planck constant: {hbar_official:.8e} J·s\n")
    f.write(f"Relative deviance (J·s): {deviance_js:.4f}%\n")
    f.write(f"Calculated Planck constant (hbar): {hbar_model_evs:.8e} eV·s\n")
    f.write(f"Official Planck constant: {hbar_official_evs:.8e} eV·s\n")
    f.write(f"Relative deviance (eV·s): {deviance_evs:.4f}%\n")

# Plot entropy function
tau_range = np.linspace(0, 1, 100)
S_values = entropy_function(tau_range)
plt.figure(figsize=(8, 6))
plt.plot(tau_range, S_values, label="S(tau) = e^tau - 1")
plt.xlabel("Entropic time (tau)")
plt.ylabel("Entropy (S)")
plt.title("Entropy Function in Meta-Space Model")
plt.grid(True)
plt.legend()
plt.savefig("entropy_function.png")
plt.close()