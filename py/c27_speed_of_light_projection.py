# c27_speed_of_light_projection.py

import numpy as np
from datetime import datetime

# Modellbasierte Eingabewerte (aus vorherigen Modulen):
hbar = 1.05050097e-34  # J·s, aus c17
a0 = 5.28715890e-11     # m, aus c19
alpha = 7.2973525693e-3 # Feinstrukturkonstante, aus c10
me_GeV = 5.11060012e-4  # GeV/c^2, aus c26

# Umrechnung m_e von GeV/c² in kg: 1 GeV/c² ≈ 1.78266192e-27 kg
me = me_GeV * 1.78266192e-27  # kg

# Formel: c = hbar / (a0 * me * alpha)
c_model = hbar / (a0 * me * alpha)

# Vergleichswert (CODATA)
c_official = 299_792_458  # m/s
deviation = 100 * (c_model - c_official) / c_official

# Ausgabe
print("Meta-Space Model Speed of Light Calculation")
print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("--------------------------------------------")
print(f"Derived speed of light (model): {c_model:.6e} m/s")
print(f"Official CODATA speed of light: {c_official:.0f} m/s")
print(f"Relative deviation: {deviation:+.4f}%")
