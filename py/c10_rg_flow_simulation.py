import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 1. Define entropic RG flow
def dalpha_dt(alpha, tau, k):
    # alpha: [alpha1, alpha2, alpha3]
    dalpha = np.zeros(3)
    for i in range(3):
        dalpha[i] = -alpha[i]**2 * (-k[i])  # since d/dtau ln(e^{-k tau}) = -k
    return dalpha

# 2. Parameters and initial conditions
k_values = [0.05, 0.04, 0.03]          # example constants for each coupling
alpha0 = [0.0169, 0.0338, 0.118]       # approx SM values at tau0
tau = np.linspace(1, 50, 500)          # entropic time range

# 3. Integrate RG equations
alphas = odeint(dalpha_dt, alpha0, tau, args=(k_values,))

# 4. Plot results
plt.plot(tau, alphas[:,0], label=r'$\alpha_1$')
plt.plot(tau, alphas[:,1], label=r'$\alpha_2$')
plt.plot(tau, alphas[:,2], label=r'$\alpha_3$')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\alpha_i(\tau)$')
plt.legend()
plt.title('Entropic RG Flow of Gauge Couplings')
plt.show()
