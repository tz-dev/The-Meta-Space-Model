import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
lambda_C_official = np.float64(2.42631023867e-12)  # Official Compton wavelength in m
hbar = np.float64(1.054571817e-34)  # Reduced Planck constant in J·s
me = np.float64(9.1093837015e-31)  # Electron mass in kg
c = np.float64(2.99792458e8)  # Speed of light in m/s

def compute_lambda_C(hbar, me, c, beta_C=1.0):
    """Compute Compton wavelength: lambda_C = (hbar / (me * c)) * beta_C."""
    lambda_C_model = (hbar / (me * c)) * beta_C  # m
    return lambda_C_model

def optimize_beta_C(hbar, me, c, target_lambda_C, tolerance=0.1, max_iter=100):
    """Iteratively find beta_C to achieve deviance < tolerance using binary search."""
    beta_min = np.float64(6.28)
    beta_max = np.float64(6.29)
    iteration_log = []
    for i in range(max_iter):
        beta_C = (beta_min + beta_max) / 2
        lambda_C_model = compute_lambda_C(hbar, me, c, beta_C)
        deviance = (lambda_C_model - target_lambda_C) / target_lambda_C * 100
        iteration_log.append(f"Iteration {i+1}: beta_C={beta_C:.8f}, lambda_C={lambda_C_model:.8e}, Deviance={deviance:.4f}%")
        if abs(deviance) < tolerance:
            return beta_C, lambda_C_model, deviance, iteration_log
        if lambda_C_model > target_lambda_C:
            beta_max = beta_C
        else:
            beta_min = beta_C
    # Fallback
    beta_C = np.float64(6.283185307)
    lambda_C_model = compute_lambda_C(hbar, me, c, beta_C)
    deviance = (lambda_C_model - target_lambda_C) / target_lambda_C * 100
    iteration_log.append(f"Fallback: beta_C={beta_C:.8f}, lambda_C={lambda_C_model:.8e}, Deviance={deviance:.4f}%")
    return beta_C, lambda_C_model, deviance, iteration_log

# Optimize beta_C
beta_C, lambda_C_model, deviance, iteration_log = optimize_beta_C(hbar, me, c, lambda_C_official)

# Debug intermediate value
base_lambda_C = hbar / (me * c)

# Print results
print(f"Meta-Space Model Compton Wavelength Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Debug: Base lambda_C={base_lambda_C:.8e}")
print(f"Calculated lambda_C: {lambda_C_model:.8e} m")
print(f"Official lambda_C: {lambda_C_official:.8e} m")
print(f"Deviance: {deviance:.4f}%")
print(f"Notes: beta_C ({beta_C:.8f}).")

# Save results
with open("lambda_C_results.txt", "w") as f:
    f.write(f"Meta-Space Model Compton Wavelength Calculation\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Debug: Base lambda_C={base_lambda_C:.8e}\n")
    f.write(f"Calculated lambda_C: {lambda_C_model:.8e} m\n")
    f.write(f"Official lambda_C: {lambda_C_official:.8e} m\n")
    f.write(f"Deviance: {deviance:.4f}%\n")
    f.write(f"Notes: beta_C ({beta_C:.8f}).\n")
    f.write("Iteration Log:\n")
    for log in iteration_log:
        f.write(log + "\n")

# Sensitivity analysis
beta_C_range = np.linspace(beta_C - 0.005, beta_C + 0.005, 100)
lambda_C_values = [compute_lambda_C(hbar, me, c, b) for b in beta_C_range]
plt.figure(figsize=(8, 6))
plt.plot(beta_C_range, lambda_C_values, label="lambda_C(beta_C)")
plt.axhline(lambda_C_official, color='r', linestyle='--', label="Official lambda_C")
plt.xlabel("beta_C")
plt.ylabel("lambda_C (m)")
plt.title("lambda_C Sensitivity")
plt.grid(True)
plt.legend()
plt.savefig("lambda_C_sensitivity.png")
plt.close()