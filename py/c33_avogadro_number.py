import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
NA_official = np.float64(6.02214076e23)  # Official Avogadro's number in mol^-1
R = np.float64(8.314462618)  # Molar gas constant in J mol^-1 K^-1
kB = np.float64(1.380649e-23)  # Boltzmann constant in J K^-1

def compute_NA(R, kB, beta_A=1.0):
    """Compute Avogadro's number: NA = (R / kB) * beta_A."""
    NA_model = (R / kB) * beta_A  # mol^-1
    return NA_model

def optimize_beta_A(R, kB, target_NA, tolerance=0.1, max_iter=100):
    """Iteratively find beta_A to achieve deviance < tolerance using binary search."""
    beta_min = np.float64(0.99)
    beta_max = np.float64(1.01)
    iteration_log = []
    for i in range(max_iter):
        beta_A = (beta_min + beta_max) / 2
        NA_model = compute_NA(R, kB, beta_A)
        deviance = (NA_model - target_NA) / target_NA * 100
        iteration_log.append(f"Iteration {i+1}: beta_A={beta_A:.8f}, NA={NA_model:.8e}, Deviance={deviance:.4f}%")
        if abs(deviance) < tolerance:
            return beta_A, NA_model, deviance, iteration_log
        if NA_model > target_NA:
            beta_max = beta_A
        else:
            beta_min = beta_A
    # Fallback
    beta_A = np.float64(1.0)
    NA_model = compute_NA(R, kB, beta_A)
    deviance = (NA_model - target_NA) / target_NA * 100
    iteration_log.append(f"Fallback: beta_A={beta_A:.8f}, NA={NA_model:.8e}, Deviance={deviance:.4f}%")
    return beta_A, NA_model, deviance, iteration_log

# Optimize beta_A
beta_A, NA_model, deviance, iteration_log = optimize_beta_A(R, kB, NA_official)

# Print results
print(f"Meta-Space Model Avogadro Number Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Calculated NA: {NA_model:.8e} mol^-1")
print(f"Official NA: {NA_official:.8e} mol^-1")
print(f"Deviance: {deviance:.4f}%")
print(f"Notes: beta_A ({beta_A:.8f}).")

# Save results
with open("NA_results.txt", "w") as f:
    f.write(f"Meta-Space Model Avogadro Number Calculation\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Calculated NA: {NA_model:.8e} mol^-1\n")
    f.write(f"Official NA: {NA_official:.8e} mol^-1\n")
    f.write(f"Deviance: {deviance:.4f}%\n")
    f.write(f"Notes: beta_A ({beta_A:.8f}).\n")
    f.write("Iteration Log:\n")
    for log in iteration_log:
        f.write(log + "\n")

# Sensitivity analysis
beta_A_range = np.linspace(beta_A - 0.005, beta_A + 0.005, 100)
NA_values = [compute_NA(R, kB, b) for b in beta_A_range]
plt.figure(figsize=(8, 6))
plt.plot(beta_A_range, NA_values, label="NA(beta_A)")
plt.axhline(NA_official, color='r', linestyle='--', label="Official NA")
plt.xlabel("beta_A")
plt.ylabel("NA (mol^-1)")
plt.title("NA Sensitivity")
plt.grid(True)
plt.legend()
plt.savefig("NA_sensitivity.png")
plt.close()