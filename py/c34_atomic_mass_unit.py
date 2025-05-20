import datetime

# Modellbasierte Massen in GeV/c^2
m_p = 0.938809801  # aus c22_proton_mass.py
m_n = 0.939570022  # aus c23_neutron_mass.py
m_e = 0.000511060012  # aus c26_electron_mass.py

# Anzahl an Nukleonen und Elektronen in 12C
Z = 6   # Protonen
N = 6   # Neutronen
e = 6   # Elektronen

# Atomare Masseneinheit aus Modell
mass_C12_model = Z * m_p + N * m_n + e * m_e  # GeV/c^2
u_model = mass_C12_model / 12

# Offizieller Wert (CODATA)
u_official = 0.93149410242  # GeV/c^2

# Abweichung
deviance = (u_model - u_official) / u_official * 100

# Ausgabe
print("Meta-Space Model Atomic Mass Unit Calculation")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model-derived u: {u_model:.11f} GeV/c^2")
print(f"Official u:      {u_official:.11f} GeV/c^2")
print(f"Relative deviance: {deviance:.4f}%")
print(f"Notes: masses from c22 (p), c23 (n), c26 (e)")

# Optionale Justierung
beta_u = u_official / u_model
print(f"Suggested beta_u scaling factor: {beta_u:.8f}")
