import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
a0_official = 5.29177210903e-11  # m
hbar = 1.05050097e-34  # J·s
c = 2.99792458e8  # m/s
m_e = 9.1093837015e-31  # kg
alpha = 1 / 137.035999084

def compute_a0(hbar, m_e, alpha, beta_a=1.0):
    """Compute Bohr radius: a0 = hbar^2 / (m_e * alpha * c * hbar) * beta_a."""
    a0_model = (hbar ** 2) / (m_e * alpha * c * hbar) * beta_a  # m
    return a0_model

# Parameters
beta_a = 1.003

# Compute a0
a0_model = compute_a0(hbar, m_e, alpha, beta_a)

# Compute deviance
deviance = (a0_model - a0_official) / a0_official * 100

# Print results
print(f"Meta-Space Model Bohr Radius Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Calculated a0: {a0_model:.8e} m")
print(f"Official a0: {a0_official:.8e} m")
print(f"Deviance: {deviance:.4f}%")
print(f"Notes: beta_a ({beta_a}).")

# Save results
with open("a0_results.txt", "w") as f:
    f.write(f"Meta-Space Model Bohr Radius Calculation\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Calculated a0: {a0_model:.8e} m\n")
    f.write(f"Official a0: {a0_official:.8e} m\n")
    f.write(f"Deviance: {deviance:.4f}%\n")

# Sensitivity analysis
beta_a_range = np.linspace(1.002, 1.004, 100)
a0_values = [compute_a0(hbar, m_e, alpha, b) for b in beta_a_range]
plt.figure(figsize=(8, 6))
plt.plot(beta_a_range, a0_values, label="a0(beta_a)")
plt.axhline(a0_official, color='r', linestyle='--', label="Official a0")
plt.xlabel("beta_a")
plt.ylabel("a0 (m)")
plt.title("a0 Sensitivity")
plt.grid(True)
plt.legend()
plt.savefig("a0_sensitivity.png")
plt.close()