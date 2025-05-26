# ========================================================
# File: c5_2_speed_of_light_projection.py
# Purpose: Derive speed of light from entropic projection of meta length and time scales
# Method:
#   - Use Planck-scale inspired length and time meta-scales
#   - Calculate projected speed as length / time
#   - Compare with official speed of light value
# Output:
#   - Print projected speed of light in m/s
#   - Print official speed of light
#   - Print relative deviation in percent
# ========================================================

import numpy as np

# Meta-scale constants (Planck-scale inspired)
length_meta = 1.616e-35  # meters
time_meta = 5.391e-44    # seconds

# Projected speed of light from meta length/time ratio
c_projected = length_meta / time_meta

# Official CODATA speed of light
c_official = 2.99792458e8  # m/s

# Calculate relative deviation (%)
relative_deviation = (c_projected - c_official) / c_official * 100

# Output results
print("=== Entropic Projection of Speed of Light ===")
print(f"Projected c: {c_projected:.8e} m/s")
print(f"Official  c: {c_official:.8e} m/s")
print(f"Relative deviation: {relative_deviation:.5f} %")
