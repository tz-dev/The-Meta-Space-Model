import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
mtau_official = np.float64(1.776860)  # GeV/c^2
me = np.float64(0.0005109989461)  # GeV/c^2
alpha = np.float64(1 / 137.035999084)

def compute_mtau(me, alpha, beta_tau=1.0):
    """Compute tau mass: mtau = me * (25/alpha) * beta_tau."""
    scale_factor = 25 / alpha
    mtau_model = me * scale_factor * beta_tau  # GeV/c^2
    return mtau_model

def optimize_beta_tau(me, alpha, target_mtau, tolerance=0.1, max_iter=100):
    """Iteratively find beta_tau to achieve deviance < tolerance using binary search."""
    beta_min = np.float64(1.00)
    beta_max = np.float64(1.03)
    iteration_log = []
    for i in range(max_iter):
        beta_tau = (beta_min + beta_max) / 2
        mtau_model = compute_mtau(me, alpha, beta_tau)
        deviance = (mtau_model - target_mtau) / target_mtau * 100
        iteration_log.append(f"Iteration {i+1}: beta_tau={beta_tau:.8f}, mtau={mtau_model:.8e}, Deviance={deviance:.4f}%")
        if abs(deviance) < tolerance:
            return beta_tau, mtau_model, deviance, iteration_log
        if mtau_model > target_mtau:
            beta_max = beta_tau
        else:
            beta_min = beta_tau
    # Fallback to calculated beta_tau
    beta_tau = np.float64(1.0146)
    mtau_model = compute_mtau(me, alpha, beta_tau)
    deviance = (mtau_model - target_mtau) / target_mtau * 100
    iteration_log.append(f"Fallback: beta_tau={beta_tau:.8f}, mtau={mtau_model:.8e}, Deviance={deviance:.4f}%")
    return beta_tau, mtau_model, deviance, iteration_log

# Optimize beta_tau
beta_tau, mtau_model, deviance, iteration_log = optimize_beta_tau(me, alpha, mtau_official)

# Print results
print(f"Meta-Space Model Tau Mass Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Calculated mtau: {mtau_model:.8e} GeV/c^2")
print(f"Official mtau: {mtau_official:.8e} GeV/c^2")
print(f"Deviance: {deviance:.4f}%")
print(f"Notes: beta_tau ({beta_tau:.8f}).")

# Save results
with open("mtau_results.txt", "w") as f:
    f.write(f"Meta-Space Model Tau Mass Calculation\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Calculated mtau: {mtau_model:.8e} GeV/c^2\n")
    f.write(f"Official mtau: {mtau_official:.8e} GeV/c^2\n")
    f.write(f"Deviance: {deviance:.4f}%\n")
    f.write(f"Notes: beta_tau ({beta_tau:.8f}).\n")
    f.write("Iteration Log:\n")
    for log in iteration_log:
        f.write(log + "\n")

# Sensitivity analysis
beta_tau_range = np.linspace(beta_tau - 0.005, beta_tau + 0.005, 100)
mtau_values = [compute_mtau(me, alpha, b) for b in beta_tau_range]
plt.figure(figsize=(8, 6))
plt.plot(beta_tau_range, mtau_values, label="mtau(beta_tau)")
plt.axhline(mtau_official, color='r', linestyle='--', label="Official mtau")
plt.xlabel("beta_tau")
plt.ylabel("mtau (GeV/c^2)")
plt.title("mtau Sensitivity")
plt.grid(True)
plt.legend()
plt.savefig("mtau_sensitivity.png")
plt.close()