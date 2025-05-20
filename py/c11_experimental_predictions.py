import numpy as np
import matplotlib.pyplot as plt

# 1. Define constants and parameters
G = 6.67430e-11       # gravitational constant
c = 299792458         # speed of light
M = 1.989e30          # solar mass in kg
S0 = 1.0              # reference entropy scale
sigma_S = 0.05        # entropic fluctuation stddev
theta = np.linspace(1e-6, 1e-4, 100)  # angular positions in radians

# 2. Monte Carlo sampling
N = 10000
dS_samples = np.random.normal(0, sigma_S, size=(N,))
# Function f(dS)
f = dS_samples[:, None] / S0

# 3. Compute ensemble of deflection angles
alpha0 = 4 * G * M / (c**2 * theta)       # classical deflection
alphas = alpha0[None, :] * (1 + f)        # corrected by entropic term

# 4. Calculate mean and std deviation
alpha_mean = alphas.mean(axis=0)
alpha_std  = alphas.std(axis=0)

# 5. Plot result
plt.figure()
plt.plot(theta, alpha_mean, label='Mean deflection')
plt.fill_between(theta,
                 alpha_mean - alpha_std,
                 alpha_mean + alpha_std,
                 alpha=0.3, label=r'$\pm1\sigma$ band')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\theta$ [rad]')
plt.ylabel(r'$\hat\alpha(\theta)$ [rad]')
plt.title('Monte-Carlo Entropic Correction to Gravitational Lensing')
plt.legend()
plt.tight_layout()
plt.show()

# 6. Print sample values at selected θ
for idx in [0, 50, 99]:
    print(f"θ={theta[idx]:.1e} rad: α_mean={alpha_mean[idx]:.3e} ±{alpha_std[idx]:.1e}")
