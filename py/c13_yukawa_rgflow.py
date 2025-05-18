import numpy as np
import matplotlib.pyplot as plt

# Parameters
tau = np.linspace(0, 40, 1000)
y = np.zeros_like(tau)
y[0] = 0.5  # Startwert

# 1-loop gauge couplings (vereinfachte RG-Evolution)
alpha1 = 0.0169 * (1 + tau/40)**(-1)
alpha2 = 0.0338 * (1 + tau/40)**(-1)
alpha3 = 0.118  * (1 + tau/40)**(-1)

# RG evolution
for i in range(1, len(tau)):
    dt = tau[i] - tau[i-1]
    y_t = y[i-1]
    beta_y = (y_t / (16*np.pi**2)) * (
        (9/2) * y_t**2 - ((17/12)*alpha1[i-1] + (9/4)*alpha2[i-1] + 8*alpha3[i-1])
    )
    y[i] = y[i-1] + beta_y * dt

    # Stop if runaway
    if y[i] > 10 or np.isnan(y[i]):
        y[i:] = np.nan
        break

# Plot
plt.figure(figsize=(8,5))
plt.plot(tau, y, label="Top Yukawa yₜ(τ)", color="darkred")
plt.xlabel("τ")
plt.ylabel("yₜ")
plt.title("1-loop RG Flow of Top Yukawa Coupling")
plt.grid(True)
plt.legend()
plt.xlim(0, 40)
plt.ylim(0.50, 0.58)
plt.tight_layout()
plt.savefig("yukawa_rgflow_top.png")
plt.show()

# Numerical output at select τ values
print("\nSelected RG values for yₜ(τ):")
for t_val in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
    idx = np.argmin(np.abs(tau - t_val))
    print(f"τ = {tau[idx]:.1f}  →  yₜ = {y[idx]:.5f}")
