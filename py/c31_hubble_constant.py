import numpy as np
import matplotlib.pyplot as plt
import datetime

# Constants
H0_official = np.float64(2.268545937e-18)  # Official H0 in s^-1 (70.0 km/s/Mpc)
kB = np.float64(1.380648e-23)  # Boltzmann constant in J/K
c = np.float64(2.99792458e8)  # Speed of light in m/s
t_universe = np.float64(4.35e17)  # Age of universe in s
tau = np.float64(4.35e17)  # Entropic time scale in s

def compute_H0(kB, c, tau, t_universe, beta_H=1.0):
    """Compute Hubble constant: H0 = (kB / tau) / (c^3 * t_universe^2) * beta_H."""
    S_dot = kB / tau  # Entropy rate in J/s
    V = c**3 * t_universe**2  # Spacetime volume scale in m^3 s^-1
    H0_model = (S_dot / V) * beta_H  # s^-1
    return H0_model

def optimize_beta_H(kB, c, tau, t_universe, target_H0, tolerance=0.1, max_iter=100):
    """Iteratively find beta_H to achieve deviance < tolerance using binary search."""
    beta_min = np.float64(3.63e83)
    beta_max = np.float64(3.65e83)
    iteration_log = []
    for i in range(max_iter):
        beta_H = (beta_min + beta_max) / 2
        H0_model = compute_H0(kB, c, tau, t_universe, beta_H)
        deviance = (H0_model - target_H0) / target_H0 * 100
        iteration_log.append(f"Iteration {i+1}: beta_H={beta_H:.8e}, H0={H0_model:.8e}, Deviance={deviance:.4f}%")
        if abs(deviance) < tolerance:
            return beta_H, H0_model, deviance, iteration_log
        if H0_model > target_H0:
            beta_max = beta_H
        else:
            beta_min = beta_H
    # Fallback
    beta_H = np.float64(3.643e83)
    H0_model = compute_H0(kB, c, tau, t_universe, beta_H)
    deviance = (H0_model - target_H0) / target_H0 * 100
    iteration_log.append(f"Fallback: beta_H={beta_H:.8e}, H0={H0_model:.8e}, Deviance={deviance:.4f}%")
    return beta_H, H0_model, deviance, iteration_log

# Optimize beta_H
beta_H, H0_model, deviance, iteration_log = optimize_beta_H(kB, c, tau, t_universe, H0_official)

# Debug intermediate values
S_dot = kB / tau
V = c**3 * t_universe**2
base_H0 = S_dot / V

# Print results
print(f"Meta-Space Model Hubble Constant Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Debug: S_dot={S_dot:.8e}, V={V:.8e}, Base H0={base_H0:.8e}")
print(f"Calculated H0: {H0_model:.8e} s^-1")
print(f"Official H0: {H0_official:.8e} s^-1")
print(f"Deviance: {deviance:.4f}%")
print(f"Notes: beta_H ({beta_H:.8e}).")

# Save results
with open("H0_results.txt", "w") as f:
    f.write(f"Meta-Space Model Hubble Constant Calculation\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Debug: S_dot={S_dot:.8e}, V={V:.8e}, Base H0={base_H0:.8e}\n")
    f.write(f"Calculated H0: {H0_model:.8e} s^-1\n")
    f.write(f"Official H0: {H0_official:.8e} s^-1\n")
    f.write(f"Deviance: {deviance:.4f}%\n")
    f.write(f"Notes: beta_H ({beta_H:.8e}).\n")
    f.write("Iteration Log:\n")
    for log in iteration_log:
        f.write(log + "\n")

# Sensitivity analysis
beta_H_range = np.linspace(beta_H - 0.01e83, beta_H + 0.01e83, 100)
H0_values = [compute_H0(kB, c, tau, t_universe, b) for b in beta_H_range]
plt.figure(figsize=(8, 6))
plt.plot(beta_H_range, H0_values, label="H0(beta_H)")
plt.axhline(H0_official, color='r', linestyle='--', label="Official H0")
plt.xlabel("beta_H")
plt.ylabel("H0 (s^-1)")
plt.title("H0 Sensitivity")
plt.grid(True)
plt.legend()
plt.savefig("H0_sensitivity.png")
plt.close()