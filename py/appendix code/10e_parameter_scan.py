# Script: 10e_parameter_scan.py
# Description: Scans neutrino oscillation parameter space (Δm², θ) by computing the
#              projection-weighted std(P_ee) across redshift-based baselines (z̄ → L).
#              Results are saved class-wise (GALAXY, QSO, etc.).
# Author: MSM Enhancement
# Date: 2025-07-08
# Version: 1.2
# Inputs: z_sky_mean_<class>.csv (e.g., z_sky_mean_galaxy.csv)
# Outputs:
#     - img/10e_oscillation_scan_heatmap_<class>.png
#     - results.csv: Appends "oscillation_scan_min_<CLASS>" metric
# Purpose:
#     - Support EP9, EP12 analysis by constraining oscillation parameters from spatial coherence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import sys

def P_ee(L, E, dm2, theta):
    delta = 1.267 * dm2 * L / E
    return 1 - np.sin(2 * theta)**2 * np.sin(delta)**2

def main(input_csv="z_sky_mean.csv"):
    if not os.path.exists(input_csv):
        print(f"[10e] Input file {input_csv} not found. Skipping.")
        return

    class_name = os.path.basename(input_csv).replace("z_sky_mean_", "").replace(".csv", "").upper()
    output_plot = f"img/10e_oscillation_scan_heatmap_{class_name.lower()}.png"

    df = pd.read_csv(input_csv).dropna(subset=["mean_z"])
    if len(df) < 30:
        print(f"[10e] Warning: Only {len(df)} valid sky bins for {class_name}. Skipping.")
        return

    print(f"[10e] Loaded {len(df)} bins from {input_csv}. Running parameter scan for {class_name}...")
    H0 = 70  # km/s/Mpc
    c = 3e5  # km/s
    df["L_km"] = (c / H0) * df["mean_z"] * 1e6 * 3.086e13

    E = 5.0  # MeV
    theta_vals = np.linspace(0.1, 1.0, 40)
    dm2_vals = np.logspace(-5.5, -4.0, 40)
    metric_grid = np.zeros((len(dm2_vals), len(theta_vals)))

    for i, dm2 in enumerate(dm2_vals):
        for j, theta in enumerate(theta_vals):
            P = P_ee(df["L_km"], E, dm2, theta)
            metric_grid[i, j] = np.std(P)

    # Plot
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        metric_grid,
        extent=[theta_vals[0], theta_vals[-1], dm2_vals[0], dm2_vals[-1]],
        aspect="auto",
        cmap="plasma",
        interpolation="none",
        origin="lower"
    )
    plt.colorbar(im, label="Oscillation Std Dev")
    plt.xlabel("Mixing Angle θ [rad]")
    plt.ylabel("Δm² [eV²]")
    plt.title(f"Parameter Scan: Neutrino Oscillation (CLASS = {class_name})")
    plt.tight_layout()

    os.makedirs("img", exist_ok=True)
    plt.savefig(output_plot)
    plt.close()
    print(f"[10e] Saved parameter scan heatmap to {output_plot}")

    # Best-Fit
    std_min = np.min(metric_grid)
    idx = np.unravel_index(np.argmin(metric_grid), metric_grid.shape)
    best_dm2 = dm2_vals[idx[0]]
    best_theta = theta_vals[idx[1]]
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Write to results.csv
    script_id = "10e_parameter_scan.py"
    param_name = f"oscillation_scan_min_{class_name}"
    results_path = "results.csv"
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    try:
        new_row = [
            script_id,
            param_name,
            f"{std_min:.7f}",
            f"Δm²={best_dm2:.2e}, θ={best_theta:.3f}",
            "",
            timestamp
        ]

        with open(results_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Only write header if file is empty
            if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
                header = ["script", "parameter", "value", "target", "deviation", "timestamp"]
                writer.writerow(header)
            writer.writerows([new_row])
        print(f"[10e] Appended 1 new row to {results_path}.")
    except Exception as e:
        print(f"[10e] Error writing to results.csv: {e}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "z_sky_mean.csv")
