# Script: 10d_entropy_map.py
# Description: Generates a 2D entropy-weighted map across RA×DEC sky bins.
#              Entropy weights are computed from the deviation of mean redshift (z̄)
#              in each bin relative to the global z̄ distribution. Visualizes spatial coherence
#              in the cosmological signal via projection-informed entropy structure.
# Author: MSM Enhancement
# Date: 2025-07-07
# Version: 1.0
# Inputs:
#     - z_sky_mean.csv: Contains RA/DEC bin boundaries and mean redshift per bin.
# Outputs:
#     - img/10d_z_entropy_weight_map.png: RA×DEC entropy heatmap for MSM spatial coherence analysis.
# Dependencies:
#     - pandas: Data ingestion.
#     - numpy: Statistical computation.
#     - matplotlib: Visualization.
#     - os: Filesystem I/O.
# Purpose:
#     - Evaluate projection-consistent entropy deviations across the sky map.
#     - Provide an intuitive tool for detecting large-scale anisotropies or systematic distortions.
#     - Support 10b/10e validation pipeline by offering complementary spatial diagnostics.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def P_ee(L, E, dm2, theta):
    delta = 1.267 * dm2 * L / E
    return 1 - np.sin(2 * theta)**2 * np.sin(delta)**2

def main():
    df = pd.read_csv("z_sky_mean.csv").dropna(subset=["mean_z"])
    H0 = 70
    c = 3e5
    df["L_km"] = (c / H0) * df["mean_z"] * 1e6 * 3.086e13

    E = 5.0  # MeV
    theta_vals = np.linspace(0.1, 1.0, 40)
    dm2_vals = np.logspace(-5.5, -4.0, 40)
    metric_grid = np.zeros((len(dm2_vals), len(theta_vals)))

    for i, dm2 in enumerate(dm2_vals):
        for j, theta in enumerate(theta_vals):
            P = P_ee(df["L_km"], E, dm2, theta)
            metric_grid[i, j] = np.std(P)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        metric_grid,
        extent=[theta_vals[0], theta_vals[-1], dm2_vals[0], dm2_vals[-1]],
        aspect="auto",
        cmap="plasma",
        interpolation="none",
        origin="lower",
        vmin=-80,
        vmax=80
    )
    plt.colorbar(im, label="Oscillation Std Dev")
    plt.xlabel("Mixing Angle θ [rad]")
    plt.ylabel("Δm² [eV²]")
    plt.title("Parameter Scan: Neutrino Oscillation Std Dev")
    plt.tight_layout()

    os.makedirs("img", exist_ok=True)
    plt.savefig("img/10e_oscillation_scan_heatmap.png")
    print("Saved parameter scan heatmap to img/10e_oscillation_scan_heatmap.png")

    os.makedirs("img", exist_ok=True)
    plt.savefig("img/10e_oscillation_scan_heatmap.png")

    # Save best result to results.csv
    std_min = np.min(metric_grid)
    idx = np.unravel_index(np.argmin(metric_grid), metric_grid.shape)
    best_dm2 = dm2_vals[idx[0]]
    best_theta = theta_vals[idx[1]]
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    with open("results.csv", "a", newline='', encoding='utf-8') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow([
            "10e_parameter_scan.py",
            "oscillation_scan_min",
            std_min,
            f"Δm²={best_dm2:.2e}, θ={best_theta:.3f}",
            "",
            timestamp
        ])

    print("Saved parameter scan heatmap to img/10e_oscillation_scan_heatmap.png")

if __name__ == "__main__":
    main()
