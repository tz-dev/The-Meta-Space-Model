import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
sigma_T_official = np.float64(6.652458734e-29)  # Official Thomson cross section in m^2
alpha = np.float64(1 / 137.035999084)  # Fine-structure constant
me = np.float64(9.1093837015e-31)  # Electron mass in kg
c = np.float64(2.99792458e8)  # Speed of light in m/s
hbar = np.float64(1.054571817e-34)  # Reduced Planck constant in J·s

def compute_sigma_T(alpha, me, c, hbar, beta_T=1.0):
    """Compute Thomson cross section: sigma_T = (8pi/3) * (alpha * hbar * c / (me * c^2))^2 * beta_T."""
    r_e = (alpha * hbar * c) / (me * c**2)  # Classical electron radius in m
    sigma_T_model = (8 * np.pi / 3) * r_e**2 * beta_T  # m^2
    return sigma_T_model

def optimize_beta_T(alpha, me, c, hbar, target_sigma_T, tolerance=0.1, max_iter=100):
    """Iteratively find beta_T to achieve deviance < tolerance using binary search."""
    beta_min = np.float64(0.99)
    beta_max = np.float64(1.01)
    iteration_log = []
    for i in range(max_iter):
        beta_T = (beta_min + beta_max) / 2
        sigma_T_model = compute_sigma_T(alpha, me, c, hbar, beta_T)
        deviance = (sigma_T_model - target_sigma_T) / target_sigma_T * 100
        iteration_log.append(f"Iteration {i+1}: beta_T={beta_T:.8f}, sigma_T={sigma_T_model:.8e}, Deviance={deviance:.4f}%")
        if abs(deviance) < tolerance:
            return beta_T, sigma_T_model, deviance, iteration_log
        if sigma_T_model > target_sigma_T:
            beta_max = beta_T
        else:
            beta_min = beta_T
    # Fallback
    beta_T = np.float64(1.0)
    sigma_T_model = compute_sigma_T(alpha, me, c, hbar, beta_T)
    deviance = (sigma_T_model - target_sigma_T) / target_sigma_T * 100
    iteration_log.append(f"Fallback: beta_T={beta_T:.8f}, sigma_T={sigma_T_model:.8e}, Deviance={deviance:.4f}%")
    return beta_T, sigma_T_model, deviance, iteration_log

# Optimize beta_T
beta_T, sigma_T_model, deviance, iteration_log = optimize_beta_T(alpha, me, c, hbar, sigma_T_official)

# Print results
print(f"Meta-Space Model Thomson Cross Section Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Calculated sigma_T: {sigma_T_model:.8e} m^2")
print(f"Official sigma_T: {sigma_T_official:.8e} m^2")
print(f"Deviance: {deviance:.4f}%")
print(f"Notes: beta_T ({beta_T:.8f}).")

# Save results
with open("sigma_T_results.txt", "w") as f:
    f.write(f"Meta-Space Model Thomson Cross Section Calculation\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Calculated sigma_T: {sigma_T_model:.8e} m^2\n")
    f.write(f"Official sigma_T: {sigma_T_official:.8e} m^2\n")
    f.write(f"Deviance: {deviance:.4f}%\n")
    f.write(f"Notes: beta_T ({beta_T:.8f}).\n")
    f.write("Iteration Log:\n")
    for log in iteration_log:
        f.write(log + "\n")

# Sensitivity analysis
beta_T_range = np.linspace(beta_T - 0.005, beta_T + 0.005, 100)
sigma_T_values = [compute_sigma_T(alpha, me, c, hbar, b) for b in beta_T_range]
plt.figure(figsize=(8, 6))
plt.plot(beta_T_range, sigma_T_values, label="sigma_T(beta_T)")
plt.axhline(sigma_T_official, color='r', linestyle='--', label="Official sigma_T")
plt.xlabel("beta_T")
plt.ylabel("sigma_T (m^2)")
plt.title("sigma_T Sensitivity")
plt.grid(True)
plt.legend()
plt.savefig("sigma_T_sensitivity.png")
plt.close()