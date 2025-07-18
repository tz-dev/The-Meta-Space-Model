# Script: 04_empirical_validator.py
# Description: Validates Meta-Space Model (MSM) simulation outputs (alpha_s, m_H, Omega_DM, Y_lm_norm, holonomy_norm,
#   stability_metric, scaling_metric, mass_drift_metric, oscillation_metric) against empirical targets on M_meta = S^3 x CY_3 x R_tau,
#   producing deviation metrics and visualizations to ensure empirical consistency.
# Formulas & Methods:
#   - Deviation: Δ = |value - target| for point targets (e.g., alpha_s, m_H, Omega_DM).
#   - Range validation: For parameters like Y_lm_norm, holonomy_norm, check if value in [min, max].
#   - Stability validation: For stability_metric, scaling_metric, check if max(value) >= target.
#   - Visualization: Bar plot of deviations, heatmaps of s_field (Script 02) and psi_alpha (Scripts 03/06a).
#   - RG Validation: α_s(τ≈1GeV⁻¹) checked against QCD running coupling expectation (≈ 0.30).
# Postulates:
#   - CP5: Entropy-coherent stability (deviations within thresholds).
#   - CP6: Simulation consistency via validation of prior results.
#   - CP8: Topological protection (Y_lm_norm, holonomy_norm in valid range).
#   - EP1: Empirical QCD coupling (alpha_s ≈ 0.118).
#   - EP5: Thermodynamic stability (mass_drift_metric within threshold).
#   - EP6: Dark matter projection (Omega_DM ≈ 0.268).
#   - EP7: Empirical consistency of spectral fields.
#   - EP8: Extended quantum gravity (stability_metric for I_mu_nu).
#   - EP11: Empirical Higgs mass (m_H ≈ 125.0 GeV).
#   - EP12: Neutrino oscillations (oscillation_metric within threshold).
#   - EP13: Renormalization group consistency (α_s(τ) flows match empirical QCD at low energies).
# Inputs:
#   - config_empirical*.json: Configuration file with targets and thresholds.
#   - results.csv: Results from Scripts 01-09 (alpha_s, m_H, Omega_DM, etc.).
#   - α_s(τ≈1GeV⁻¹) from Script 02, stored in results.csv.
#   - img/s_field.npy: Field data from Script 02.
#   - img/psi_alpha.npy: Field data from Scripts 03/06a.
# Outputs:
#   - results.csv: Appended validation summary (parameter, value, target, deviation, timestamp).
#   - img/04_validation_bar_plot.png: Bar plot of deviations with thresholds.
#   - img/04_validation_s_field_heatmap.png: Heatmap of s_field.
#   - img/04_validation_psi_alpha_heatmap.png: Heatmap of psi_alpha.
#   - errors.log: Logs errors and validation info.
# Dependencies: numpy, matplotlib, csv, json, glob, logging, tqdm

import csv
import json
import os
import glob
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import platform
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Logging setup
logging.basicConfig(
    filename='errors.log',
    level=logging.INFO,
    format='%(asctime)s [04_empirical_validator.py] %(levelname)s: %(message)s'
)

def clear_screen():
    """Clear the console screen based on the operating system."""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def load_config():
    """Load fixed JSON configuration file for empirical validation."""
    config_path = 'config_empirical.json'
    if not os.path.exists(config_path):
        logging.error(f"Missing fixed config file: {config_path}")
        raise FileNotFoundError(f"Missing {config_path}")
    with open(config_path, 'r', encoding='utf-8') as infile:
        cfg = json.load(infile)
    print(f"[04_empirical_validator.py] Loaded targets: {cfg['targets']}")
    print(f"[04_empirical_validator.py] Loaded thresholds: {cfg['thresholds']}")
    return cfg

def load_results():
    """
    Read results.csv and parse entries, including range references.
    Returns:
        list: List of dictionaries with script, parameter, value, reference, deviation, timestamp.
    """
    print(f"[04_empirical_validator.py] Loading results from results.csv")
    entries = []
    try:
        with open('results.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            seen = {}  # Track latest entry per parameter across all scripts
            for row in reader:
                if len(row) != 6:
                    continue
                script, param, val, ref, dev, ts = row
                try:
                    val_f = float(val)
                except ValueError:
                    continue
                # Handle range references (e.g., "[1e3, 1e6]")
                ref_f = None
                if ref.startswith('[') and ref.endswith(']'):
                    try:
                        ref_f = [float(x) for x in ref.strip('[]').split(',')]
                    except ValueError:
                        ref_f = None
                else:
                    try:
                        ref_f = float(ref)
                    except ValueError:
                        ref_f = None
                dev_f = float(dev) if dev.replace('.', '', 1).isdigit() else None
                key = param  # Use parameter as key to aggregate across scripts
                if key not in seen or (script != '04_empirical_validator.py' and datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S') > datetime.strptime(seen[key]['timestamp'], '%Y-%m-%dT%H:%M:%S')):
                    seen[key] = {
                        'script': script,
                        'parameter': param,
                        'value': val_f,
                        'reference': ref_f,
                        'deviation': dev_f,
                        'timestamp': ts
                    }
            entries = list(seen.values())
        print(f"[04_empirical_validator.py] Loaded {len(entries)} unique entries from results.csv")
        return entries
    except Exception as e:
        logging.error(f"Failed to load results.csv: {e}")
        raise

def validate(entries, cfg):
    """
    Validate entries against empirical targets, ranges, or stability thresholds per CP5, CP6, CP8, EP1, EP5, EP6, EP7, EP8, EP11, EP12.
    Args:
        entries (list): List of result entries.
        cfg (dict): Configuration with targets and thresholds.
    Returns:
        list: Validated entries with parameter, value, target, deviation, threshold, status.
    """
    print(f"[04_empirical_validator.py] Validating results against empirical targets")
    validated = []
    # Aggregate stability_metric and scaling_metric
    stability_values = {'stability_metric': [], 'scaling_metric': []}
    filtered_entries = []
    
    # Collect stability metrics and filter other entries
    for e in entries:
        if e['parameter'] in stability_values:
            stability_values[e['parameter']].append(e['value'])
        else:
            filtered_entries.append(e)
    
    # Add aggregated stability metrics
    for param in stability_values:
        if stability_values[param]:
            value = max(stability_values[param])  # Use maximum value
            filtered_entries.append({
                'script': 'aggregated',
                'parameter': param,
                'value': value,
                'reference': cfg['targets'].get(param, 0.5),
                'deviation': None,
                'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            })

    # Filter our non-dimensional values
    filtered_entries = [
        e for e in filtered_entries
        if isinstance(e['value'], (int, float)) and np.isfinite(e['value'])
    ]

    # Deduplicate by parameter, keeping the first valid entry
    seen_params = set()
    for e in tqdm(filtered_entries, desc="Validating entries", unit="entry"):
        param = 'm_h' if e['parameter'] == 'm_H' else e['parameter']
        
        # --- Class specific neutrino_metric matching ---
        if param.startswith("oscillation_metric_"):
            base = "oscillation_metric"
            if base not in cfg['targets']:
                continue
            target = cfg['targets'][base]
            threshold = cfg['thresholds'].get(param, cfg['thresholds'].get(base, 0.05))

        elif param.startswith("entropy_weight_std_"):
            base = "entropy_weight_std"
            target = cfg['targets'].get(base, 0.2)

            # Calculate dynamic threshold from results.csv
            class_suffix = param.split('_')[-1]
            try:
                values = [
                    float(x['value']) for x in entries
                    if x['parameter'] in [f"entropy_weight_std_north_{class_suffix}",
                                          f"entropy_weight_std_south_{class_suffix}"]
                    and not np.isnan(x['value'])
                ]
                if values:
                    threshold = 1.25 * np.median(values)
                    print(f"[04] Using adaptive entropy threshold for {param}: {threshold:.6f}")
                else:
                    raise ValueError("No valid hemisphere values found")
            except Exception as e:
                logging.warning(f"[04] Adaptive threshold fallback for {param}: {e}")
                threshold = cfg['thresholds'].get(param, 0.05)

        elif param in cfg['targets']:
            target = cfg['targets'][param]
            threshold = cfg['thresholds'].get(param, 0.0)
        else:
            continue

        seen_params.add(param)
        value = e['value']
        status = None
        deviation = None

        # Point target (also for oscillation_metric_<CLASS>)
        if isinstance(target, (int, float)) and param not in ['stability_metric', 'scaling_metric', 'Y_lm_norm', 'holonomy_norm']:
            deviation = abs(value - target)
            status = 'PASS' if deviation <= threshold else 'FAIL'
            logging.info(f"Validated {param}: value={value:.6f}, target={target:.6f}, "
                         f"Δ={deviation:.6f}, threshold={threshold:.6f} → {status}")
        
        # Point target validation (e.g., alpha_s, m_h, Omega_DM)
        if isinstance(target, (int, float)) and param not in ['stability_metric', 'scaling_metric', 'Y_lm_norm', 'holonomy_norm']:
            deviation = abs(e['value'] - target)
            status = 'PASS' if deviation <= threshold else 'FAIL'
            logging.info(f"Validated {param}: value={e['value']:.6f}, target={target:.6f}, "
                         f"Δ={deviation:.6f}, threshold={threshold:.6f} → {status}")
        
        # Range target validation (e.g., Y_lm_norm, holonomy_norm)
        elif isinstance(target, list) and len(target) == 2:
            min_val, max_val = target
            deviation = min(abs(e['value'] - min_val), abs(e['value'] - max_val)) if not (min_val <= e['value'] <= max_val) else 0.0
            status = 'PASS' if min_val <= e['value'] <= max_val else 'FAIL'
            threshold = 0.1 * (max_val - min_val) if threshold == 0.0 else threshold  # Adjust threshold to 10% of range if 0
            logging.info(f"Validated {param}: value={e['value']:.6f}, range=[{min_val}, {max_val}], "
                         f"Δ={deviation:.6f}, threshold={threshold:.6f} → {status}")
        
        # Stability target validation (e.g., stability_metric, scaling_metric)
        elif isinstance(target, (int, float)) and param in ['stability_metric', 'scaling_metric']:
            deviation = max(0.0, target - e['value'])
            status = 'PASS' if e['value'] >= target else 'FAIL'
            threshold = 0.1 * target if threshold == 0.0 else threshold  # Adjust threshold to 10% of target if 0
            logging.info(f"Validated {param}: value={e['value']:.6f}, min={target:.6f}, "
                         f"Δ={deviation:.6f}, threshold={threshold:.6f} → {status}")
        
        validated.append({
            'parameter': param,
            'value': e['value'],
            'target': target,
            'deviation': deviation,
            'threshold': threshold,
            'status': status
        })
    print(f"[04_empirical_validator.py] Validated {len(validated)} parameters")
    return validated

def plot_deviations(validated):
    """
    Generate bar plot of deviations with threshold line.
    Args:
        validated (list): List of validated entries.
    """
    print(f"[04_empirical_validator.py] Generating deviation bar plot")
    labels = [v['parameter'] for v in validated]
    deviations = [v['deviation'] for v in validated]
    thresholds = [v['threshold'] for v in validated]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, deviations, color='steelblue')
    ax.plot(labels, thresholds, 'r--', label='Threshold')
    ax.set_ylabel('Deviation')
    ax.set_title('Empirical Validation Deviations')
    ax.legend()
    for bar, dev in zip(bars, deviations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e-3,
                f"{dev:.6f}", ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/04_validation_bar_plot.png')
    plt.close()
    print(f"[04_empirical_validator.py] Bar plot saved -> img/04_validation_bar_plot.png")

def plot_heatmaps():
    """Load and plot heatmaps for s_field.npy and psi_alpha.npy if available."""
    print(f"[04_empirical_validator.py] Generating heatmaps for validation")
    
    # s_field heatmap
    sf_path = 'img/s_field.npy'
    if os.path.exists(sf_path):
        s_field = np.load(sf_path)
        plt.imshow(np.abs(s_field), cmap='viridis', origin='lower')
        plt.colorbar(label='|s_field| (Script 02)')
        plt.title('Validation: s_field Heatmap')
        plt.savefig('img/04_validation_s_field_heatmap.png')
        plt.close()
        print(f"[04_empirical_validator.py] s_field heatmap saved -> img/04_validation_s_field_heatmap.png")
    else:
        logging.info("s_field.npy not found; skipping its heatmap")
    
    # psi_alpha heatmap
    psi_path = 'img/psi_alpha.npy'
    if os.path.exists(psi_path):
        psi = np.load(psi_path)
        plt.imshow(np.abs(psi), cmap='plasma', origin='lower')
        plt.colorbar(label='|psi_alpha| (Scripts 03/06a)')
        plt.title('Validation: psi_alpha Heatmap')
        plt.savefig('img/04_validation_psi_alpha_heatmap.png')
        plt.close()
        print(f"[04_empirical_validator.py] psi_alpha heatmap saved -> img/04_validation_psi_alpha_heatmap.png")
    else:
        logging.info("psi_alpha.npy not found; skipping its heatmap")

def append_summary(validated):
    """
    Append validation summary to results.csv, 
    after removing all previous entries written by 04_empirical_validator.py.
    
    Args:
        validated (list): List of validated entries.
    """
    script_name = '04_empirical_validator.py'
    print(f"[{script_name}] Cleaning up previous entries from results.csv")

    # Read existing content and filter out entries from script_name
    try:
        with open('results.csv', 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        rows = [row for row in rows if row and row[0] != script_name]
    except FileNotFoundError:
        rows = []  # If file doesn't exist, start fresh

    # Append new validated entries
    ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    for v in validated:
        rows.append([
            script_name,
            f"{v['parameter']}_validation",
            v['value'],
            str(v['target']),
            v['deviation'],
            ts
        ])

    # Write the updated rows back to file
    with open('results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[{script_name}] Validation summary written: {len(validated)} entries")


def main():
    """Main function to orchestrate empirical validation of MSM outputs."""
    clear_screen()

    try:
        with open('results.csv', 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        rows = [row for row in rows if row and row[0] != '04_empirical_validator.py']
        with open('results.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    except FileNotFoundError:
        pass

    print("==============================================")
    print("    Meta-Space Model: Empirical Validation    ")
    print("==============================================")
    
    # Load configuration
    cfg = load_config()
    
    # Load and validate results with progress bar
    with tqdm(total=3, desc="Processing empirical validation", unit="step") as pbar:
        # Load results from results.csv
        entries = load_results()
        pbar.update(1)
        
        # Validate entries against targets
        validated = validate(entries, cfg)
        if not validated:
            print(f"[04_empirical_validator.py] No parameters validated. Exiting.")
            return
        pbar.update(1)
        
        # Generate visualizations
        plot_deviations(validated)
        plot_heatmaps()
        pbar.update(1)

    # Validate I_mu_nu aufrom script 07
    if any(e['parameter'] == 'I_mu_nu' for e in entries):
        try:
            I_mu_nu_entry = next(e for e in entries if e['parameter'] == 'I_mu_nu')
            I_mu_nu = I_mu_nu_entry['value']
            target = cfg['targets'].get('I_mu_nu', 0.0)
            threshold = cfg['thresholds'].get('I_mu_nu', 1.0)
            deviation = abs(I_mu_nu - target)
            status = 'PASS' if deviation <= threshold else 'FAIL'
            validated.append({
                'parameter': 'I_mu_nu',
                'value': I_mu_nu,
                'target': target,
                'deviation': deviation,
                'threshold': threshold,
                'status': status
            })
            print(f"[04] I_mu_nu = {I_mu_nu:.3e}, Δ = {deviation:.3e}, status = {status}")
        except Exception as e:
            print(f"[04] Warning: Could not validate I_mu_nu: {e}")
    
    # Append validation summary to results.csv
    append_summary(validated)
    
    # Summary output
    print("\n=====================================")
    print("     Meta-Space Model: Summary")
    print("=====================================")
    print(f"Script: 04_empirical_validator.py")
    print(f"Description: Validates MSM outputs against empirical targets")
    print(f"Postulates: CP5, CP6, CP8, EP1, EP5, EP6, EP7, EP8, EP11, EP12")
    print(f"Validated Parameters: {len(validated)}")
    for v in validated:
        target_str = f"[{v['target'][0]}, {v['target'][1]}]" if isinstance(v['target'], list) else f"{v['target']:.6f}"
        print(f"- {v['parameter']}: value={v['value']:.6f}, target={target_str}, "
              f"Δ={v['deviation']:.6f}, threshold={v['threshold']:.6f}, status={v['status']}")

    # Classify parameters as internal or external based on script origin
    internal = [v for v in validated if not any(e['script'].startswith('10') for e in entries if e['parameter'] == v['parameter'])]
    external = [v for v in validated if any(e['script'].startswith('10') for e in entries if e['parameter'] == v['parameter'])]
    internal_status = all(v['status'] == 'PASS' for v in internal)
    external_status = all(v['status'] == 'PASS' for v in external)

    print(f"Internal Model Status: {'PASS' if internal_status else 'FAIL'}")
    print(f"Empirical Data Status: {'PASS' if external_status else 'FAIL'}")
    print(f"Plots: 04_validation_bar_plot.png, 04_validation_s_field_heatmap.png, 04_validation_psi_alpha_heatmap.png")
    print("=====================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
        raise