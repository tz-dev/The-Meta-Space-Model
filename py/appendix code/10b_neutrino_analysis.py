# Script: 10b_neutrino_analysis.py
# Description: Performs neutrino oscillation analysis from redshift map with MSM-consistent projection weighting.
# Author: MSM Enhancement
# Version: 2.0 (2025-07-07)

import numpy as np
import pandas as pd
from datetime import datetime
import csv
import os
import platform
import json

def load_config(path="config_external.json"):
    """Load external configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def P_ee(L, E, dm2, theta):
    """Standard neutrino survival probability."""
    delta = 1.267 * dm2 * L / E
    return 1 - np.sin(2 * theta)**2 * np.sin(delta)**2

def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def analyze_neutrino_oscillation(input_csv="z_sky_mean.csv", output_summary="neutrino_analysis_summary.txt"):
    config = load_config()

    clear_screen()
    print("========================================================")
    print("          Meta-Space Model: Neutrino Analysis           ")  
    print("========================================================")

    print("\n[10b] Starting neutrino oscillation analysis...")
    if not os.path.exists(input_csv):
        print(f"[10b] Input file {input_csv} not found. Aborting.")
        return

    df = pd.read_csv(input_csv).dropna(subset=["mean_z"])
    print(f"[10b] Loaded {len(df)} valid sky bins from {input_csv}.")

    # Cosmology conversion
    H0 = 70  # km/s/Mpc
    c = 3e5  # km/s
    df["L_km"] = (c / H0) * df["mean_z"] * 1e6 * 3.086e13

    # Load parameters
    DM2 = config.get("neutrino_dm2", 7.53e-5)
    theta = config.get("neutrino_theta", 0.5843)
    E_MeV = config.get("neutrino_energy_mev", 5.0)

    # Compute raw oscillation
    df["P_ee"] = P_ee(df["L_km"], E_MeV, DM2, theta)

    # z̄ statistics
    z_mean = df["mean_z"].mean()
    z_std = df["mean_z"].std()

    # Projectional coherence metric (entropy-modulated weight)
    df["entropy_weight"] = np.exp(- (df["mean_z"] - z_mean)**2 / (2 * z_std**2))
    df["P_ee_proj"] = df["P_ee"] * df["entropy_weight"]
    proj_metric = df["P_ee_proj"].std()

    # Aggregate metrics
    mean_P = df["P_ee"].mean()
    std_P = df["P_ee"].std()

    threshold = config.get("thresholds", {}).get("oscillation_metric", 0.05)

    print("[10b] Converted redshift to baseline distances (km).")
    print("[10b] Computed P_ee and projection-weighted metrics.")

    # Output summary
    print("\n=====================================")
    print("     Meta-Space Model: Summary")
    print("=====================================")
    print("Script: 10b_neutrino_analysis.py")
    print("Description: Neutrino oscillation analysis from z-binned sky map")
    print("Postulates: EP6, EP9, EP12, CP7")
    print(f"Computed ⟨P_ee⟩: {mean_P:.6f}")
    print(f"Oscillation Metric (std): {std_P:.6e} (target 0.0, Δ={std_P:.6e})")
    print(f"Entropy Projection Metric: {proj_metric:.6e}")
    print(f"Parameters: E = {E_MeV:.2f} MeV, Δm² = {DM2:.2e} eV², θ = {theta:.4f} rad")
    print(f"Status: {'PASS' if std_P < threshold else 'FAIL'}")
    print("=====================================\n")

    # Write to results.csv
    results_path = "results.csv"
    script_id = "10b_neutrino_analysis.py"
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

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
            [script_id, "oscillation_metric", std_P, "0.0", std_P, timestamp],
            [script_id, "P_ee_mean", mean_P, "N/A", "", timestamp],
            [script_id, "entropy_projection_metric", proj_metric, "N/A", "", timestamp],
        ]

        with open(results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows + new_rows)
        print(f"[10b] Appended results to {results_path}.")
    except Exception as e:
        print(f"[10b] Error writing to results.csv: {e}")

    # Summary log
    with open(output_summary, "w", encoding="utf-8") as f:
        f.write("Neutrino Oscillation Analysis Summary\n")
        f.write(f"Mean P_ee                 = {mean_P:.6f}\n")
        f.write(f"Std. deviation (P_ee)     = {std_P:.6e}\n")
        f.write(f"Projection Coherence (P') = {proj_metric:.6e}\n")
        f.write(f"Energy (MeV)              = {E_MeV:.2f}\n")
        f.write(f"Delta m^2                 = {DM2:.2e} eV^2\n")
        f.write(f"Theta                     = {theta:.4f} rad\n")
        f.write(f"Metric Target             = {threshold:.6f}\n")
        f.write(f"Status                    = {'PASS' if std_P < threshold else 'FAIL'}\n")

    print(f"[10b] Summary written to {output_summary}.")
    print("[10b] Neutrino oscillation analysis complete.")

if __name__ == "__main__":
    analyze_neutrino_oscillation()
