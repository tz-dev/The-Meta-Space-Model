# c28_vacuum_permeability.py
import numpy as np
from datetime import datetime

print("Meta-Space Model Vacuum Permeability Calculation")
print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Constants (Model-derived or from previous calculations)
alpha = 7.2973525693e-3           # Fine-structure constant (CODATA or from model)
hbar = 1.05050097e-34             # Planck constant from c17_entropy_quantization_hbar.py [J·s]
e = 1.602176634e-19               # Elementary charge [C]
c = 299792458                    # Speed of light [m/s], temporarily used as fixed

# Compute epsilon_0 from definition of alpha
# α = e² / (4π ε₀ ħ c) → ε₀ = e² / (4π α ħ c)
epsilon_0 = e**2 / (4 * np.pi * alpha * hbar * c)

# Compute mu_0 from: μ₀ = 1 / (ε₀ c²)
mu_0 = 1 / (epsilon_0 * c**2)

# Official CODATA value
mu_0_official = 4 * np.pi * 1e-7  # H/m

# Deviance
deviance = (mu_0 - mu_0_official) / mu_0_official * 100

print(f"Calculated μ₀: {mu_0:.10e} H/m")
print(f"Official μ₀:   {mu_0_official:.10e} H/m")
print(f"Relative deviance: {deviance:+.4f}%")

print("Notes: hbar and alpha from prior model calculations (C.17, B.1)")
