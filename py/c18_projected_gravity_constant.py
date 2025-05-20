import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
G_official = 6.67430e-11  # m^3 kg^-1 s^-2
hbar = 1.05050097e-34  # J·s
c = 2.99792458e8  # m/s
gev_to_kg = 1.78266192e-27  # GeV/c^2 to kg

def compute_G(M_scale, alpha_G=1.0):
    """Compute G: G = alpha_G * hbar * c / M_scale^2."""
    M_scale_kg = M_scale * gev_to_kg
    G_model = alpha_G * hbar * c / (M_scale_kg ** 2)
    return G_model

# Parameters
M_scale = 1.2235e19  # GeV/c^2
alpha_G = 1.0085

# Compute G
G_model = compute_G(M_scale, alpha_G)

# Compute deviance
deviance = (G_model - G_official) / G_official * 100

# Print results
print(f"Meta-Space Model Gravitational Constant Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Calculated G: {G_model:.8e} m^3 kg^-1 s^-2")
print(f"Official G: {G_official:.8e} m^3 kg^-1 s^-2")
print(f"Deviance: {deviance:.4f}%")
print(f"Notes: M_scale ({M_scale} GeV/c^2), alpha_G ({alpha_G}).")

# Save results
with open("G_results.txt", "w") as f:
    f.write(f"Meta-Space Model Gravitational Constant Calculation\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Calculated G: {G_model:.8e} m^3 kg^-1 s^-2\n")
    f.write(f"Official G: {G_official:.8e} m^3 kg^-1 s^-2\n")
    f.write(f"Deviance: {deviance:.4f}%\n")

# Sensitivity analysis
M_scale_range = np.linspace(1.22e19, 1.225e19, 100)
G_values = [compute_G(M, alpha_G) for M in M_scale_range]
plt.figure(figsize=(8, 6))
plt.plot(M_scale_range, G_values, label="G(M_scale)")
plt.axhline(G_official, color='r', linestyle='--', label="Official G")
plt.xlabel("M_scale (GeV/c^2)")
plt.ylabel("G (m^3 kg^-1 s^-2)")
plt.title("G Sensitivity")
plt.grid(True)
plt.legend()
plt.savefig("G_sensitivity.png")
plt.close()