import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
E_bind_official = -13.605693122994  # eV
m_e = 9.1093837015e-31  # kg
c = 2.99792458e8  # m/s
alpha = 1 / 137.035999084
joule_to_ev = 1 / 1.602176634e-19  # J to eV

def compute_E_bind(m_e, alpha, c, beta_E=1.0):
    """Compute binding energy: E_bind = -m_e * alpha^2 * c^2 / 2 * beta_E."""
    E_bind_joule = - (m_e * alpha**2 * c**2) / 2 * beta_E  # J
    E_bind_ev = E_bind_joule * joule_to_ev  # eV
    return E_bind_ev

# Parameters
beta_E = 1.00

# Compute E_bind
E_bind_model = compute_E_bind(m_e, alpha, c, beta_E)

# Compute deviance
deviance = (E_bind_model - E_bind_official) / E_bind_official * 100

# Print results
print(f"Meta-Space Model Binding Energy Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Calculated E_bind: {E_bind_model:.8e} eV")
print(f"Official E_bind: {E_bind_official:.8e} eV")
print(f"Deviance: {deviance:.4f}%")
print(f"Notes: beta_E ({beta_E}).")

# Save results
with open("E_bind_results.txt", "w") as f:
    f.write(f"Meta-Space Model Binding Energy Calculation\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Calculated E_bind: {E_bind_model:.8e} eV\n")
    f.write(f"Official E_bind: {E_bind_official:.8e} eV\n")
    f.write(f"Deviance: {deviance:.4f}%\n")

# Sensitivity analysis
beta_E_range = np.linspace(1.0017, 1.0019, 100)
E_bind_values = [compute_E_bind(m_e, alpha, c, b) for b in beta_E_range]
plt.figure(figsize=(8, 6))
plt.plot(beta_E_range, E_bind_values, label="E_bind(beta_E)")
plt.axhline(E_bind_official, color='r', linestyle='--', label="Official E_bind")
plt.xlabel("beta_E")
plt.ylabel("E_bind (eV)")
plt.title("E_bind Sensitivity")
plt.grid(True)
plt.legend()
plt.savefig("E_bind_sensitivity.png")
plt.close()