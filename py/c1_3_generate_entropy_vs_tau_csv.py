# =============================================================================
# C.1.3 – Export Entropy vs. Meta-Time to CSV
# File: c1_3_generate_entropy_vs_tau_csv.py
# This script reads simulation snapshots (e.g., "entropy_snapshot_100.npy"),
# extracts the average entropy value for each time point (tau),
# and writes the results to a CSV file for further analysis.
# Input:  Files named like "entropy_snapshot_*.npy"
# Output: CSV file "entropy_vs_tau.csv" with columns: tau [s], mean entropy
# =============================================================================

import numpy as np
import glob
import re
import os

# --- Parameters ---
dtau = 0.001  # Meta-time step (must match simulation setting)
snapshot_pattern = "entropy_snapshot_*.npy"
output_csv = "entropy_vs_tau.csv"

# --- Helper function to extract time step from file name ---
def extract_step(filename):
    match = re.search(r"entropy_snapshot_(\d+)\.npy", filename)
    return int(match.group(1)) if match else None

# --- Collect and sort data entries ---
files = glob.glob(snapshot_pattern)
entries = []
for fn in files:
    step = extract_step(os.path.basename(fn))
    if step is None:
        continue
    tau = step * dtau
    S = np.load(fn)
    mean_S = np.mean(S)
    entries.append((tau, mean_S))

entries.sort(key=lambda x: x[0])  # sort by tau

# --- Write CSV output ---
with open(output_csv, "w") as f:
    f.write("# tau [s], mean_entropy\n")
    for tau, mean_S in entries:
        f.write(f"{tau:.6f},{mean_S:.8f}\n")

print(f"Wrote {len(entries)} entries to {output_csv}")
