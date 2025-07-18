"""
Script: 10a_plot_z_sky_mean.py
Description:
    This script visualizes the mean redshift values from sky regions as a sky map.
    It reads a CSV file (`z_sky_mean.csv`) containing pre-processed mean redshift values
    for RA/DEC bins, then creates a heatmap of the sky and performs an isotropy check
    by comparing redshift means between the northern and southern celestial hemispheres.

    The results include:
    - Global mean, min, and max redshift statistics
    - North vs. South hemisphere mean redshift comparison
    - Welch’s t-test for isotropy check
    - Visualization of Δz̄ (North - South) by RA
    - Summary outputs written to image and CSV files

Author: Generated by Grok 3 (xAI), modified by MSM
Date: 2025-07-07
Version: 1.1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import csv

def main():
    input_file = "z_sky_mean.csv"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    class_tag = ""
    if "galaxy" in input_file.lower():
        class_tag = "_galaxy"
    elif "qso" in input_file.lower():
        class_tag = "_qso"

    if not os.path.exists(input_file):
        print(f"[10a] Input file {input_file} not found. Aborting.")
        return

    df = pd.read_csv(input_file)

    ra_vals = sorted(set(df["ra_min"]))
    dec_vals = sorted(set(df["dec_min"]))
    ra_bins = len(ra_vals)
    dec_bins = len(dec_vals)

    z_mean_map = np.full((dec_bins, ra_bins), np.nan)
    for _, row in df.iterrows():
        i = dec_vals.index(row["dec_min"])
        j = ra_vals.index(row["ra_min"])
        z_mean_map[i, j] = row["mean_z"]

    flat_z = z_mean_map[~np.isnan(z_mean_map)]
    z_min, z_max = np.min(flat_z), np.max(flat_z)
    z_mean = np.mean(flat_z)
    z_std = np.std(flat_z)

    # Hemisphere-Isotropy analysis
    north_mask = df["dec_min"] > 0
    south_mask = df["dec_min"] < 0
    z_north = df[north_mask]["mean_z"].dropna()
    z_south = df[south_mask]["mean_z"].dropna()
    z_mean_north = np.mean(z_north) if len(z_north) > 0 else np.nan
    z_std_north = np.std(z_north) if len(z_north) > 0 else np.nan
    z_mean_south = np.mean(z_south) if len(z_south) > 0 else np.nan
    z_std_south = np.std(z_south) if len(z_south) > 0 else np.nan

    # Welch-Test north vs south
    from scipy.stats import ttest_ind
    ttest_pval = np.nan
    z_mean_delta = np.nan
    if len(z_north) > 1 and len(z_south) > 1:
        ttest_res = ttest_ind(z_north, z_south, equal_var=False, nan_policy='omit')
        ttest_pval = ttest_res.pvalue
        z_mean_delta = z_mean_north - z_mean_south
        print(f"  Welch-Test Δμ = {z_mean_delta:.6f}, p = {ttest_pval:.3e}")

    # visualization: Δz̄(N–S) over RA
    delta_z_by_ra = []
    for ra_bin in ra_vals:
        z_n = df[(df["ra_min"] == ra_bin) & (df["dec_min"] > 0)]["mean_z"].dropna()
        z_s = df[(df["ra_min"] == ra_bin) & (df["dec_min"] < 0)]["mean_z"].dropna()
        if len(z_n) > 0 and len(z_s) > 0:
            delta = np.mean(z_n) - np.mean(z_s)
            delta_z_by_ra.append((ra_bin, delta))

    if delta_z_by_ra:
        ras, deltas = zip(*delta_z_by_ra)
        plt.figure(figsize=(10, 4))
        plt.plot(ras, deltas, marker='o')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel('Right Ascension [°]')
        plt.ylabel('Δz̄ (North – South)')
        plt.title('Hemisphere Redshift Asymmetry by RA')
        plt.tight_layout()
        plt.savefig(f"img/10a_z_delta_ns_by_ra{class_tag}.png")

    print(f"Isotropy check:")
    print(f"  z̄ min  = {z_min:.3f}")
    print(f"  z̄ max  = {z_max:.3f}")
    print(f"  z̄ mean = {z_mean:.3f}")
    print(f"  z̄ std  = {z_std:.3f}")
    print(f"  North mean = {z_mean_north:.3f}, std = {z_std_north:.3f}")
    print(f"  South mean = {z_mean_south:.3f}, std = {z_std_south:.3f}")

    ra_min = min(ra_vals)
    ra_max = max(df["ra_max"])
    dec_min = min(dec_vals)
    dec_max = max(df["dec_max"])

    plt.figure(figsize=(12, 6))
    cmap = plt.cm.plasma
    cmap.set_bad("lightgrey")

    im = plt.imshow(
        np.flipud(z_mean_map),
        extent=[ra_min, ra_max, dec_min, dec_max],
        aspect='auto',
        cmap=cmap,
        interpolation='none'
    )
    plt.colorbar(im, label='Mean Redshift $\\bar{z}$')
    plt.xlabel('Right Ascension (RA) [°]')
    plt.ylabel('Declination (DEC) [°]')
    plt.title('Mean Redshift by Sky Regions')
    plt.grid(False)
    plt.tight_layout()

    try:
        os.makedirs("img", exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create 'img' directory: {e}")

    plt.savefig(f"img/10a_z_sky_mean_map{class_tag}.png")

    # Update results.csv
    results_path = "results.csv"
    script_id = "10a_plot_z_sky_mean.py"
    try:
        # Create new entries
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        suffix = class_tag.upper().replace("_", "")  # "GALAXY" or "QSO"

        new_rows = [
            [script_id, f"z_mean_min_{suffix}", z_min, "", "", timestamp],
            [script_id, f"z_mean_max_{suffix}", z_max, "", "", timestamp],
            [script_id, f"z_mean_avg_{suffix}", z_mean, "N/A", "N/A", timestamp],
            [script_id, f"z_mean_std_{suffix}", z_std, "ideal ≈ 0 (isotrop)", "N/A", timestamp],
            [script_id, f"z_mean_north_{suffix}", z_mean_north, "", "", timestamp],
            [script_id, f"z_mean_south_{suffix}", z_mean_south, "", "", timestamp],
            [script_id, f"z_std_north_{suffix}", z_std_north, "", "", timestamp],
            [script_id, f"z_std_south_{suffix}", z_std_south, "", "", timestamp],
            [script_id, f"z_mean_delta_ns_{suffix}", z_mean_delta, "0", abs(z_mean_delta), timestamp],
            [script_id, f"z_mean_ttest_pval_{suffix}", ttest_pval, "", "", timestamp],
        ]

        # Write results
        with open(results_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
        print(f"[10a] Appended {len(new_rows)} new rows to {results_path}.")
    except Exception as e:
        print(f"[10a] Error updating results.csv: {e}")

    # Summary log
    with open("z_sky_isotropy_summary.txt", "w", encoding="utf-8") as f:
        f.write("Isotropy check:\n")
        f.write(f"z̄ min  = {z_min:.6f}\n")
        f.write(f"z̄ max  = {z_max:.6f}\n")
        f.write(f"z̄ mean = {z_mean:.6f}\n")
        f.write(f"z̄ std  = {z_std:.6f}\n")
        f.write(f"North mean = {z_mean_north:.6f}, std = {z_std_north:.6f}\n")
        f.write(f"South mean = {z_mean_south:.6f}, std = {z_std_south:.6f}\n")

if __name__ == "__main__":
    main()