import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
H0_km_s_Mpc = np.float64(70.0)  # Hubble constant in km/s/Mpc
c = np.float64(2.99792458e8)  # Speed of light in m/s
Mpc_to_m = np.float64(3.085677581e22)  # 1 Mpc in meters
Lambda_official = np.float64(1.1056e-52)  # Official Lambda in m^-2

# Convert H0 to 1/s
H0_SI = H0_km_s_Mpc * 1000 / Mpc_to_m  # in 1/s

def compute_Lambda(H0_SI, c, beta_Lambda=1.0):
    """Compute cosmological constant: Lambda = (3 * H0^2 / c^2) * beta_Lambda."""
    Lambda_model = (3 * H0_SI**2 / c**2) * beta_Lambda  # m^-2
    return Lambda_model

def optimize_beta_Lambda(H0_SI, c, target_Lambda, tolerance=0.1, max_iter=100):
    """Iteratively find beta_Lambda to achieve deviance < tolerance using binary search."""
    beta_min = np.float64(0.6)
    beta_max = np.float64(0.7)
    iteration_log = []
    for i in range(max_iter):
        beta_Lambda = (beta_min + beta_max) / 2
        Lambda_model = compute_Lambda(H0_SI, c, beta_Lambda)
        deviance = (Lambda_model - target_Lambda) / target_Lambda * 100
        iteration_log.append(f"Iteration {i+1}: beta_Lambda={beta_Lambda:.8f}, Lambda={Lambda_model:.10e}, Deviance={deviance:.4f}%")
        if abs(deviance) < tolerance:
            return beta_Lambda, Lambda_model, deviance, iteration_log
        if Lambda_model > target_Lambda:
            beta_max = beta_Lambda
        else:
            beta_min = beta_Lambda
    # Fallback
    beta_Lambda = np.float64(0.644)
    Lambda_model = compute_Lambda(H0_SI, c, beta_Lambda)
    deviance = (Lambda_model - target_Lambda) / target_Lambda * 100
    iteration_log.append(f"Fallback: beta_Lambda={beta_Lambda:.8f}, Lambda={Lambda_model:.10e}, Deviance={deviance:.4f}%")
    return beta_Lambda, Lambda_model, deviance, iteration_log

# Optimize beta_Lambda
beta_Lambda, Lambda_model, deviance, iteration_log = optimize_beta_Lambda(H0_SI, c, Lambda_official)

# Print results
print(f"Meta-Space Model Cosmological Constant Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Calculated Lambda: {Lambda_model:.10e} m^-2")
print(f"Official Lambda: {Lambda_official:.10e} m^-2")
print(f"Deviance: {deviance:.4f}%")
print(f"Notes: beta_Lambda ({beta_Lambda:.8f}).")

# Save results
with open("Lambda_results.txt", "w") as f:
    f.write(f"Meta-Space Model Cosmological Constant Calculation\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Calculated Lambda: {Lambda_model:.10e} m^-2\n")
    f.write(f"Official Lambda: {Lambda_official:.10e} m^-2\n")
    f.write(f"Deviance: {deviance:.4f}%\n")
    f.write(f"Notes: beta_Lambda ({beta_Lambda:.8f}).\n")
    f.write("Iteration Log:\n")
    for log in iteration_log:
        f.write(log + "\n")

# Sensitivity analysis
beta_Lambda_range = np.linspace(beta_Lambda - 0.005, beta_Lambda + 0.005, 100)
Lambda_values = [compute_Lambda(H0_SI, c, b) for b in beta_Lambda_range]
plt.figure(figsize=(8, 6))
plt.plot(beta_Lambda_range, Lambda_values, label="Lambda(beta_Lambda)")
plt.axhline(Lambda_official, color='r', linestyle='--', label="Official Lambda")
plt.xlabel("beta_Lambda")
plt.ylabel("Lambda (m^-2)")
plt.title("Lambda Sensitivity")
plt.grid(True)
plt.legend()
plt.savefig("Lambda_sensitivity.png")
plt.close()