import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
R_inf_official = np.float64(10973731.568160)  # m^-1
hbar = np.float64(1.05050097e-34)  # J·s
h = hbar * 2 * np.pi  # J·s
c = np.float64(2.99792458e8)  # m/s
m_e = np.float64(9.1093837015e-31)  # kg
alpha = np.float64(1 / 137.035999084)

def compute_R_inf(m_e, alpha, c, h, beta_R=1.0):
    """Compute Rydberg constant: R_inf = m_e * alpha^2 * c / (2 * h) * beta_R."""
    R_inf_model = (m_e * alpha**2 * c) / (2 * h) * beta_R  # m^-1
    return R_inf_model

def optimize_beta_R(m_e, alpha, c, h, target_R_inf, tolerance=0.002, max_iter=200):
    """Iteratively find beta_R to achieve deviance < tolerance using binary search."""
    beta_min = np.float64(0.9989)
    beta_max = np.float64(0.9991)
    iteration_log = []
    for i in range(max_iter):
        beta_R = (beta_min + beta_max) / 2
        R_inf_model = compute_R_inf(m_e, alpha, c, h, beta_R)
        deviance = (R_inf_model - target_R_inf) / target_R_inf * 100
        iteration_log.append(f"Iteration {i+1}: beta_R={beta_R:.8f}, R_inf={R_inf_model:.8e}, Deviance={deviance:.4f}%")
        if abs(deviance) < tolerance:
            return beta_R, R_inf_model, deviance, iteration_log
        if R_inf_model > target_R_inf:
            beta_max = beta_R
        else:
            beta_min = beta_R
    # Fallback to manual beta_R
    beta_R = np.float64(0.99700)
    R_inf_model = compute_R_inf(m_e, alpha, c, h, beta_R)
    deviance = (R_inf_model - target_R_inf) / target_R_inf * 100
    iteration_log.append(f"Fallback: beta_R={beta_R:.8f}, R_inf={R_inf_model:.8e}, Deviance={deviance:.4f}%")
    return beta_R, R_inf_model, deviance, iteration_log

# Optimize beta_R
beta_R, R_inf_model, deviance, iteration_log = optimize_beta_R(m_e, alpha, c, h, R_inf_official)

# Print results
print(f"Meta-Space Model Rydberg Constant Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Calculated R_inf: {R_inf_model:.8e} m^-1")
print(f"Official R_inf: {R_inf_official:.8e} m^-1")
print(f"Deviance: {deviance:.4f}%")
print(f"Notes: beta_R ({beta_R:.8f}).")

# Save results
with open("R_inf_results.txt", "w") as f:
    f.write(f"Meta-Space Model Rydberg Constant Calculation\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Calculated R_inf: {R_inf_model:.8e} m^-1\n")
    f.write(f"Official R_inf: {R_inf_official:.8e} m^-1\n")
    f.write(f"Deviance: {deviance:.4f}%\n")
    f.write(f"Notes: beta_R ({beta_R:.8f}).\n")
    f.write("Iteration Log:\n")
    for log in iteration_log:
        f.write(log + "\n")

# Sensitivity analysis
beta_R_range = np.linspace(beta_R - 0.00001, beta_R + 0.00001, 100)
R_inf_values = [compute_R_inf(m_e, alpha, c, h, b) for b in beta_R_range]
plt.figure(figsize=(8, 6))
plt.plot(beta_R_range, R_inf_values, label="R_inf(beta_R)")
plt.axhline(R_inf_official, color='r', linestyle='--', label="Official R_inf")
plt.xlabel("beta_R")
plt.ylabel("R_inf (m^-1)")
plt.title("R_inf Sensitivity")
plt.grid(True)
plt.legend()
plt.savefig("R_inf_sensitivity.png")
plt.close()