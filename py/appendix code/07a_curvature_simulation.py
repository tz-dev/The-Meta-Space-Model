# Script: 07a_curvature_simulation.py
# Description: Estimates the curvature trace I_{μν} ≈ ⟨|∇²S|⟩ from the entropic field S(x, y, τ)
#   on the meta-space manifold M_meta = S^3 × CY_3 × ℝ_τ. This script serves as a purely geometric
#   consistency check of the MSM framework regarding spatial flatness (Ω_k ≈ 0), based on the
#   Laplacian of S as curvature indicator.
# Formulas & Methods:
#   - Laplacian: ∇²S := ∂²S/∂τ² + ∂²S/∂x² + ∂²S/∂y² (applied only on existing axes)
#   - Curvature estimator: I_{μν} := ⟨|∇²S|⟩ (scalar trace)
#   - Validation: I_{μν} compared against empirical flatness via threshold (e.g. ≤ 1.0)
# Postulates:
#   - CP1: Meta-space geometry (S^3 × CY_3 × ℝ_τ defines the entropic manifold)
#   - CP3: Geometric emergence (observable quantities emerge from metric properties)
#   - CP6: Simulation consistency (cross-checks between field and curvature layers)
#   - EP8: Extended quantum gravity (I_{μν} used as emergent curvature indicator)
# Inputs:
#   - img/s_field.npy: Entropic field S(x, y, τ), generated in Script 02.
#   - config_empirical.json: JSON file specifying I_{μν} target and threshold.
# Outputs:
#   - results.csv: Appends I_{μν} value, target, deviation and timestamp for validation.
# Notes:
#   - No artificial scaling applied; ∇²S computed directly from field gradients.
#   - Robust to 2D and 3D field inputs; skips missing axes gracefully.
# Dependencies: numpy, os, json, csv, datetime, logging

import numpy as np
import os
import json
import csv
from datetime import datetime
import logging

logging.basicConfig(filename='errors.log', level=logging.INFO,
    format='%(asctime)s [07a_curvature_simulation.py] %(levelname)s: %(message)s')

def load_config():
    try:
        with open("config_empirical.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        target = cfg.get("targets", {}).get("I_mu_nu", 0.0)
        threshold = cfg.get("thresholds", {}).get("I_mu_nu", 1.0)
        return target, threshold
    except Exception as e:
        logging.warning(f"No config loaded, using defaults: {e}")
        return 0.0, 1.0

def compute_laplacian(S):
    ndim = S.ndim
    lap = np.zeros_like(S)
    for ax in range(ndim):
        try:
            d2 = np.gradient(np.gradient(S, axis=ax), axis=ax)
            lap += d2
        except Exception as e:
            logging.warning(f"Skipping axis {ax} in ∇²S: {e}")
    return lap

def main():

    try:
        with open('results.csv', 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        rows = [row for row in rows if row and row[0] != '07a_curvature_simulation.py']
        with open('results.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    except FileNotFoundError:
        pass

    print("=======================================================")
    print("   Meta-Space Model: Curvature Estimation (I_{μν})   ")
    print("=======================================================")

    path = "img/s_field.npy"
    if not os.path.exists(path):
        print("Missing s_field.npy – run Script 02 first.")
        return

    S = np.load(path)
    lap_S = compute_laplacian(S)
    I_mu_nu = float(np.mean(np.abs(lap_S)))

    target, threshold = load_config()
    deviation = abs(I_mu_nu - target)
    status = "PASS" if deviation <= threshold else "FAIL"

    # Write to results.csv
    os.makedirs("img", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open("results.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["07a_curvature_simulation.py", "I_mu_nu", I_mu_nu, target, deviation, ts])

    print(f"Computed I_mu_nu = {I_mu_nu:.3e}, Δ = {deviation:.3e}, Status = {status}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Execution failed: {e}")
        raise
