# Script: 10d_entropy_map.py
# Description: Computes and visualizes an entropy-weighted RA×DEC sky map based on 
#              deviation of mean redshift (z̄) from its global distribution. The resulting 
#              entropy weights encode projection-consistent spatial anisotropies, which 
#              may signal large-scale structure or observational bias.
# Author: MSM Enhancement
# Date: 2025-07-07
# Version: 1.0
# Inputs:
#     - z_sky_mean.csv: Binned sky coordinates and mean redshift values.
# Outputs:
#     - img/10d_z_entropy_weight_map.png: Entropy heatmap over RA×DEC sky.
#     - results.csv: Adds "entropy_weight_std" metric for model consistency.
# Dependencies:
#     - numpy: Numerical computation.
#     - pandas: CSV ingestion and transformation.
#     - matplotlib: Heatmap visualization.
#     - os, csv, datetime: File and I/O operations.
# Purpose:
#     - Detect spatial coherence or anisotropy in z̄ distributions.
#     - Support downstream analyses (e.g. 10b neutrino metric, 10e parameter scans).
#     - Serve as diagnostic layer for MSM spatial projection consistency.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime

def main():
    df = pd.read_csv("z_sky_mean.csv").dropna(subset=["mean_z"])
    ra_vals = sorted(set(df["ra_min"]))
    dec_vals = sorted(set(df["dec_min"]))
    ra_bins = len(ra_vals)
    dec_bins = len(dec_vals)

    ra_min = min(ra_vals)
    ra_max = max(ra_vals)
    dec_min = min(dec_vals)
    dec_max = max(dec_vals)

    # Entropiegewichtung berechnen
    z_mean = df["mean_z"].mean()
    z_std = df["mean_z"].std()
    df["entropy_weight"] = np.exp(- (df["mean_z"] - z_mean)**2 / (2 * z_std**2))

    # 2D Map erzeugen
    entropy_map = np.full((dec_bins, ra_bins), np.nan)
    for _, row in df.iterrows():
        i = dec_vals.index(row["dec_min"])
        j = ra_vals.index(row["ra_min"])
        entropy_map[i, j] = row["entropy_weight"]

    # Plot
    plt.figure(figsize=(12, 6))
    cmap = plt.cm.inferno
    cmap.set_bad("lightgrey")

    im = plt.imshow(
        np.flipud(entropy_map),
        extent=[ra_min, ra_max, dec_min, dec_max],
        aspect="auto",
        cmap="viridis",
        interpolation="none",
        vmin=-80,
        vmax=80
    )
    plt.colorbar(im, label='Entropy Weight')
    plt.xlabel('Right Ascension (RA) [°]')
    plt.ylabel('Declination (DEC) [°]')
    plt.title('Entropy Weighting across Sky Bins')
    plt.tight_layout()

    os.makedirs("img", exist_ok=True)
    plt.savefig("img/10d_z_entropy_weight_map.png")
    print("Saved entropy map to img/10d_z_entropy_weight_map.png")

    # Ergebnis in results.csv speichern
    script_id = "10d_entropy_map.py"
    parameter = "entropy_weight_std"
    value = df["entropy_weight"].std()
    target = "N/A"
    deviation = "N/A"
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    results_path = "results.csv"
    try:
        rows = []
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            header, *data_rows = rows
            data_rows = [row for row in data_rows if row[0] != script_id]
        else:
            header = ["script", "parameter", "value", "target", "deviation", "timestamp"]
            data_rows = []

        new_row = [script_id, parameter, value, target, deviation, timestamp]

        with open(results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows + [new_row])

        print(f"[10d] Appended results to {results_path}.")
    except Exception as e:
        print(f"[10d] Error writing to results.csv: {e}")

if __name__ == "__main__":
    main()
