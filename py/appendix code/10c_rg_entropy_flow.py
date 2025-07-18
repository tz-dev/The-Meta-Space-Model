# Script: 10c_rg_entropy_flow.py
# Description: Extracts a renormalization group inspired coupling flow \( \alpha_s(\tau) \)
#              from the sky-binned redshift distribution in z_sky_mean.csv.
#              Converts redshift to scale \( \tau \sim 1 / \log(1 + z) \),
#              computes an effective coupling using 1-loop QCD flow,
#              and compares against the expected low-energy limit \( \alpha_s(1~\mathrm{GeV}^{-1}) \approx 0.30 \).
# Author: MSM Enhancement
# Date: 2025-07-07
# Version: 1.3
# Inputs:
#     - z_sky_mean.csv: CSV file containing binned mean redshift or density per sky region.
#     - config_external.json: External configuration with RG parameters.
# Outputs:
#     - results.csv: Appended with alpha_s_tau_rg and deviation from target.
#     - img/10c_alpha_s_rg_flow.png: RG flow plot of alpha_s vs. tau.
#     - img/10c_alpha_s_hist.png: Histogram of alpha_s values for quality control.
#     - rg_flow_summary.txt: Summary of derived alpha_s at tau = 1 GeV^-1 and comparison.
# Dependencies:
#     - numpy, pandas, matplotlib, os, csv, datetime, platform, json
# Purpose:
#     - Provide a scale-dependent validation layer for RG consistency of emergent couplings.
#     - Compare derived flow \( \alpha_s(\tau) \) with known QCD behaviour at low-energy scales.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import os
import platform
import json

def load_config(path="config_external.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_alpha_s(tau, config):
    Lambda = config.get("rg_lambda", 0.2)
    nf = config.get("rg_nf", 5)
    beta0 = 11 - 2/3 * nf
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 4 * np.pi / (beta0 * np.log(tau**2 / Lambda**2))
        result[~np.isfinite(result)] = np.nan
    return result

def load_empirical_alpha_s_tau():
    try:
        with open("results.csv", "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row[1].strip() in ("alpha_s_tau_1gev", "alpha_s_tau_1gev_validation"):
                    return float(row[2])
    except Exception as e:
        print(f"[10c] Warning: Could not load empirical α_s(τ=1GeV⁻¹): {e}")
    return None

def analyze_rg_flow(input_csv="z_sky_mean.csv", output_summary="rg_flow_summary.txt"):
    print("========================================================")
    print("         Meta-Space Model: RG Entropy Flow Analysis     ")
    print("========================================================")

    print("\n[10c] Starting RG flow reconstruction from z-sky distribution...")
    if not os.path.exists(input_csv):
        print(f"[10c] Input file {input_csv} not found. Aborting.")
        return

    class_name = os.path.basename(input_csv).replace("z_sky_mean_", "").replace(".csv", "").upper()
    param_name = f"alpha_s_tau_rg_{class_name}"
    output_summary = f"rg_flow_summary_{class_name.lower()}.txt"
    flow_plot_path = f"img/10c_alpha_s_rg_flow_{class_name.lower()}.png"
    hist_plot_path = f"img/10c_alpha_s_hist_{class_name.lower()}.png"

    config = load_config()
    df = pd.read_csv(input_csv)

    if "mean_density" not in df.columns:
        print("[10c] Error: Column 'mean_density' not found in input. Aborting.")
        return

    df = df.dropna(subset=["mean_density"])
    df["tau"] = 1 / np.log1p(df["mean_density"])
    df = df[df["tau"].between(0.01, 1000)]

    df["alpha_s"] = compute_alpha_s(df["tau"], config)
    df = df[df["alpha_s"].between(0, 1.5)]

    print(f"[10c] Loaded {len(df)} filtered sky bins with valid tau range.")

    tau_target = 1.0
    alpha_s_target = 0.30
    idx_closest = (df["tau"] - tau_target).abs().idxmin()
    
    alpha_s_at_tau = df.loc[idx_closest, "alpha_s"]
    deviation = abs(alpha_s_at_tau - alpha_s_target)
    alpha_s_empirical = load_empirical_alpha_s_tau()

    print("\n=====================================")
    print("     Meta-Space Model: Summary")
    print("=====================================")
    print("Script: 10c_rg_entropy_flow.py")
    print("Description: RG-inspired coupling flow from z-map entropy structure")
    print("Postulates: EP12, EP13, CP5")
    print(f"Computed α_s(τ=1) = {alpha_s_at_tau:.6f} (target {alpha_s_target:.6f}, Δ={deviation:.6f})")
    print(f"Min/Max α_s = {df['alpha_s'].min():.6f} / {df['alpha_s'].max():.6f}")
    print(f"Valid τ range = {df['tau'].min():.3f} to {df['tau'].max():.3f} GeV^-1")
    print(f"Status: {'PASS' if deviation < 0.05 else 'FAIL'}")
    print(f"Plot: 10c_alpha_s_rg_flow.png")
    print("=====================================\n")

    # Save to csv
    results_path = "results.csv"
    script_id = "10c_rg_entropy_flow.py"
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    try:
        new_row = [script_id, param_name, alpha_s_at_tau, alpha_s_target, deviation, timestamp]

        with open(results_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Only write header if file is empty
            if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
                header = ["script", "parameter", "value", "target", "deviation", "timestamp"]
                writer.writerow(header)
            writer.writerows([new_row])
        print(f"[10c] Appended 1 new row to {results_path}.")
    except Exception as e:
        print(f"[10c] Error writing to results.csv: {e}")

    os.makedirs("img", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df["tau"], df["alpha_s"], 'o', alpha=0.6, label=r"$\alpha_s(\tau)$ from $z$-bins")
    plt.axhline(alpha_s_target, color='r', linestyle='--', label=f"Target αₛ(1) = {alpha_s_target:.2f}")
    if alpha_s_empirical:
        plt.axhline(alpha_s_empirical, color='g', linestyle=':', label=f"Empirical αₛ(1) = {alpha_s_empirical:.2f}")
    plt.xlabel(r"$\tau$ (1/ln(1+ρ)) [GeV$^{-1}$]")
    plt.ylabel(r"$\alpha_s(\tau)$")
    plt.title(r"RG Coupling Flow $\alpha_s(\tau)$")
    plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(flow_plot_path)
    print(f"[10c] Flow plot saved -> {flow_plot_path}")

    plt.figure(figsize=(8, 5))
    plt.hist(df["alpha_s"].dropna(), bins=50, color="navy", alpha=0.7)
    plt.xlabel(r"$\alpha_s(\tau)$")
    plt.ylabel("Count")
    plt.title("Distribution of RG Coupling Values")
    plt.tight_layout()
    plt.savefig(hist_plot_path)
    print(f"[10c] Histogram saved -> {hist_plot_path}")

    with open(output_summary, "w", encoding="utf-8") as f:
        f.write("Renormalization Group Flow Summary\n")
        f.write(f"alpha_s(tau=1) = {alpha_s_at_tau:.6f}\n")
        f.write(f"Target         = {alpha_s_target:.6f}\n")
        f.write(f"Deviation      = {deviation:.6f}\n")
        f.write(f"Tau range      = {df['tau'].min():.6f} to {df['tau'].max():.6f} GeV^-1\n")
        f.write(f"Min/Max alpha  = {df['alpha_s'].min():.6f} / {df['alpha_s'].max():.6f}\n")
        f.write(f"Status         = {'PASS' if deviation < 0.05 else 'FAIL'}\n")
    print(f"[10c] Summary written to {output_summary}.")
    print("[10c] RG flow analysis complete.")

if __name__ == "__main__":
    import sys
    input_arg = sys.argv[1] if len(sys.argv) > 1 else "z_sky_mean.csv"
    analyze_rg_flow(input_arg)