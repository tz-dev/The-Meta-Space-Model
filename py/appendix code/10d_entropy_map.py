# Script: 10d_entropy_map.py
# Description: Computes and visualizes an entropy-weighted RA×DEC sky map based on 
#              deviation of mean redshift (z̄) from its global distribution. Includes
#              normalized Shannon entropy, hemispheric analysis, and correlation metrics.
# Author: MSM Enhancement
# Date: 2025-07-08
# Version: 1.1
# Inputs: z_sky_mean_<class>.csv (e.g., z_sky_mean_galaxy.csv, z_sky_mean_qso.csv)
# Outputs:
#     - img/10d_z_entropy_weight_map_<class>.png: Entropy heatmap over RA×DEC sky.
#     - results.csv: Metrics (entropy_weight_std, normalized_entropy, etc.) with class suffix.
# Dependencies: numpy, pandas, matplotlib, os, csv, datetime, sys, json
# Purpose:
#     - Detect spatial coherence or anisotropy in z̄ distributions.
#     - Support downstream analyses (e.g., 10b neutrino metric, 10e parameter scans).
#     - Serve as diagnostic layer for MSM spatial projection consistency (EP6, EP12).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import sys
import json

def load_config(path="config_external.json"):
    """Load external configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main(input_csv="z_sky_mean.csv"):
    config = load_config()

    # Extrahiere Klassennamen aus dem Dateinamen
    class_name = os.path.basename(input_csv).replace("z_sky_mean_", "").replace(".csv", "").upper()
    output_plot = f"img/10d_z_entropy_weight_map_{class_name.lower()}.png"

    print(f"[10d] Starting entropy map analysis for {class_name}...")
    if not os.path.exists(input_csv):
        print(f"[10d] Input file {input_csv} not found. Aborting.")
        return

    df = pd.read_csv(input_csv).dropna(subset=["mean_z"])
    min_valid_bins = 30
    if len(df) < min_valid_bins:
        print(f"[10d] Warning: Only {len(df)} valid sky bins (required: ≥{min_valid_bins}). Skipping entropy analysis.")
        
        # Ergebnis in results.csv speichern
        script_id = "10d_entropy_map.py"
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        results_path = "results.csv"

        try:
            if len(df) < min_valid_bins:
                print(f"[10d] Warning: Only {len(df)} valid sky bins (required: ≥{min_valid_bins}). Skipping entropy analysis.")
                new_rows = [
                    [script_id, f"status_entropy_analysis_{class_name}", "insufficient_bins", "≥30", "", timestamp],
                    [script_id, f"valid_bin_count_{class_name}", len(df), "≥30", "", timestamp]
                ]
            else:
                new_rows = [
                    [script_id, f"entropy_weight_std_{class_name}", entropy_weight_std, "N/A", "", timestamp],
                    [script_id, f"normalized_entropy_{class_name}", S_rho, "N/A", "", timestamp],
                    [script_id, f"entropy_weight_std_north_{class_name}", north_std, "N/A", "", timestamp],
                    [script_id, f"entropy_weight_std_south_{class_name}", south_std, "N/A", "", timestamp],
                    [script_id, f"entropy_z_correlation_{class_name}", correlation, "N/A", "", timestamp]
                ]

            with open(results_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Schreibe Header nur, wenn die Datei leer ist
                if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
                    header = ["script", "parameter", "value", "target", "deviation", "timestamp"]
                    writer.writerow(header)
                writer.writerows(new_rows)
            print(f"[10d] Appended {len(new_rows)} new rows to {results_path}.")
        except Exception as e:
            print(f"[10d] Error writing to results.csv: {e}")
        
        return  # Analyse abbrechen

    print(f"[10d] Loaded {len(df)} valid sky bins from {input_csv}.")

    # Robuste Entropiegewichtung mittels Median und MAD
    z_median = df["mean_z"].median()
    mad = np.median(np.abs(df["mean_z"] - z_median))
    mad_scaled = mad * 1.4826  # Approximation zur Standardabweichung für normalverteilte Daten

    if mad_scaled == 0:  # Fallback auf Standardabweichung, falls MAD=0 (z. B. bei wenigen Bins)
        z_mean = df["mean_z"].mean()
        z_std = df["mean_z"].std()
        scale = z_std if z_std > 0 else 1e-6
        df["entropy_weight"] = np.exp(- (df["mean_z"] - z_mean)**2 / (2 * scale**2))
    else:
        df["entropy_weight"] = np.exp(- (df["mean_z"] - z_median)**2 / (2 * mad_scaled**2))

    # Normierte Shannon-Entropie (S_ρ)
    w = df["entropy_weight"] / df["entropy_weight"].sum()  # Normalisierte Gewichte
    S_rho = -np.sum(w * np.log(w + 1e-10)) / np.log(len(w)) if len(w) > 0 else np.nan

    # Hemisphären-Statistik
    north_mask = df["dec_min"] >= 0
    south_mask = df["dec_min"] < 0
    entropy_weight_std = df["entropy_weight"].std()
    north_std = df[north_mask]["entropy_weight"].std() if north_mask.any() else np.nan
    south_std = df[south_mask]["entropy_weight"].std() if south_mask.any() else np.nan

    # Korrelation zwischen entropy_weight und mean_z
    correlation = df["entropy_weight"].corr(df["mean_z"]) if len(df) > 1 else np.nan

    # Threshold-Warnsystem
    # Dynamisches Thresholding: adaptiv auf Nord/Süd-Streuung
    threshold = 1.25 * np.median([north_std, south_std]) if np.isfinite(north_std) and np.isfinite(south_std) else 0.2
    status = "PASS" if entropy_weight_std < threshold else "FAIL"

    # 2D Map erzeugen
    ra_vals = sorted(set(df["ra_min"]))
    dec_vals = sorted(set(df["dec_min"]))
    ra_bins = len(ra_vals)
    dec_bins = len(dec_vals)
    ra_min, ra_max = min(ra_vals), max(ra_vals)
    dec_min, dec_max = min(dec_vals), max(dec_vals)

    entropy_map = np.full((dec_bins, ra_bins), np.nan)
    for _, row in df.iterrows():
        i = dec_vals.index(row["dec_min"])
        j = ra_vals.index(row["ra_min"])
        entropy_map[i, j] = row["entropy_weight"]

    # Plot
    plt.figure(figsize=(12, 6))
    cmap = plt.cm.viridis  # Änderung zu viridis für Konsistenz mit 10b
    cmap.set_bad("lightgrey")

    im = plt.imshow(
        np.flipud(entropy_map),
        extent=[ra_min, ra_max, dec_min, dec_max],
        aspect="auto",
        cmap=cmap,
        interpolation="none",
        vmin=df["entropy_weight"].min(),
        vmax=df["entropy_weight"].max()
    )
    plt.colorbar(im, label="Entropy Weight")
    plt.xlabel("Right Ascension (RA) [°]")
    plt.ylabel("Declination (DEC) [°]")
    plt.title(f"Entropy Weighting across Sky Bins ({class_name})")
    plt.tight_layout()

    os.makedirs("img", exist_ok=True)
    plt.savefig(output_plot)
    plt.close()
    print(f"[10d] Saved entropy map to {output_plot}")

    # Ergebnis in results.csv speichern
    script_id = "10d_entropy_map.py"
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

        new_rows = [
            [script_id, f"entropy_weight_std_{class_name}", entropy_weight_std, "N/A", "", timestamp],
            [script_id, f"normalized_entropy_{class_name}", S_rho, "N/A", "", timestamp],
            [script_id, f"entropy_weight_std_north_{class_name}", north_std, "N/A", "", timestamp],
            [script_id, f"entropy_weight_std_south_{class_name}", south_std, "N/A", "", timestamp],
            [script_id, f"entropy_z_correlation_{class_name}", correlation, "N/A", "", timestamp]
        ]

        with open(results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows + new_rows)

        print(f"[10d] Appended results to {results_path}.")
    except Exception as e:
        print(f"[10d] Error writing to results.csv: {e}")

    # Summary-Ausgabe
    print("\n=====================================")
    print(f"     Meta-Space Model: Entropy Map Summary ({class_name})")
    print("=====================================")
    print("Script: 10d_entropy_map.py")
    print(f"Description: Entropy-weighted sky map analysis for {class_name}")
    print("Postulates: EP6, EP12")
    print(f"Entropy Weight Std: {entropy_weight_std:.6e}")
    print(f"Normalized Shannon Entropy (S_ρ): {S_rho:.6f}")
    print(f"North Hemisphere Entropy Std: {north_std:.6e}")
    print(f"South Hemisphere Entropy Std: {south_std:.6e}")
    print(f"Correlation (entropy_weight, mean_z): {correlation:.6f}")
    print(f"Adaptive Metric Threshold: {threshold:.6f}")
    print(f"Status: {status}")
    print("=====================================\n")

if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "z_sky_mean.csv"
    main(input_csv)
