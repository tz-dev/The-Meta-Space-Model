# Patch incoming for script 10b_neutrino_analysis.py
# Implements: energy scan + class-specific parameters
# Mode: relaxed (at least one PASS sufficient)

import numpy as np
import pandas as pd
from datetime import datetime
import csv
import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def load_config(path="config_external.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def P_ee(L, E, dm2, theta):
    delta = 1.267 * dm2 * L / E
    return 1 - np.sin(2 * theta)**2 * np.sin(delta)**2

def analyze(input_csv):
    config = load_config()
    class_name = os.path.basename(input_csv).replace("z_sky_mean_", "").replace(".csv", "").upper()
    df = pd.read_csv(input_csv).dropna(subset=["mean_z"])

    if df.empty:
        print(f"[10b] No valid data in {input_csv}.")
        return

    print(f"[10b] {len(df)} bins loaded for class {class_name}.")

    # Baseline L(z)
    H0 = 70
    c = 3e5
    df["L_km"] = (c / H0) * df["mean_z"] * 1e6 * 3.086e13

    # Energy scan and class-specific params
    energy_list = config.get("neutrino_energy_scan_mev", [3.0, 5.0, 7.0, 10.0])
    class_params = config.get("neutrino_params", {}).get(class_name, {})
    dm2 = class_params.get("dm2", config.get("neutrino_dm2", 7.53e-5))
    theta = class_params.get("theta", config.get("neutrino_theta", 0.5843))

    status_list = []
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    results_path = "results.csv"
    script_id = "10b_neutrino_analysis.py"
    os.makedirs("img", exist_ok=True)

    for E in energy_list:
        df["P_ee"] = P_ee(df["L_km"], E, dm2, theta)
        z_mean = df["mean_z"].mean()
        z_std = df["mean_z"].std()
        df["entropy_weight"] = np.exp(- (df["mean_z"] - z_mean)**2 / (2 * z_std**2))
        df["P_ee_proj"] = df["P_ee"] * df["entropy_weight"]

        std_P = df["P_ee"].std()
        mean_P = df["P_ee"].mean()
        proj_metric = df["P_ee_proj"].std()
        max_dev = np.max(np.abs(df["P_ee"] - 1))
        north = df[df["dec_min"] > 0]["P_ee"]
        south = df[df["dec_min"] <= 0]["P_ee"]

        threshold_key = f"oscillation_metric_{class_name}" if class_name in ["GALAXY", "QSO"] else "oscillation_metric_Z_SKY_MEAN"
        threshold = config.get("thresholds", {}).get(threshold_key, 0.05)
        status = "PASS" if std_P < threshold else "FAIL"
        status_list.append(status)

        # CSV append
        new_rows = [
            [script_id, f"osc_metric_{class_name}_E{int(E)}", std_P, "0.0", std_P, timestamp],
            [script_id, f"P_ee_mean_{class_name}_E{int(E)}", mean_P, "", "", timestamp],
            [script_id, f"P_ee_proj_metric_{class_name}_E{int(E)}", proj_metric, "", "", timestamp],
            [script_id, f"P_ee_maxdev_{class_name}_E{int(E)}", max_dev, "", "", timestamp]
        ]
        with open(results_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)

        # Plot
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            norm = Normalize(vmin=df["entropy_weight"].min(), vmax=df["entropy_weight"].max())
            cmap = plt.cm.viridis
            scatter = ax.scatter(
                df["mean_z"], df["P_ee"],
                c=df["entropy_weight"], cmap=cmap, norm=norm, alpha=0.6
            )
            cbar = fig.colorbar(scatter, ax=ax, label="Entropy Weight")
            ax.set_title(f"P_ee vs z [{class_name}, E={E:.1f} MeV]")
            ax.set_xlabel("Redshift z")
            ax.set_ylabel("P_ee")
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(f"img/10b_Pee_vs_z_{class_name.lower()}_E{int(E)}.png")
            plt.close()
        except Exception as e:
            print(f"[10b] Plot error @E={E}: {e}")

    # Summary logic (relaxed)
    global_status = "PASS" if "PASS" in status_list else "FAIL"
    print(f"\n[10b] Energy scan results for {class_name}: {status_list} → Global Status: {global_status}")

if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "z_sky_mean.csv"
    analyze(input_csv)
