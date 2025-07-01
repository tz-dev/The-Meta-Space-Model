# Script: 04_empirical_validator.py
# Description: Validates Meta-Space Model (MSM) simulation outputs (α_s, m_H) against empirical targets on 𝓜_meta = S^3 × CY_3 × ℝ_τ,
#   producing deviation metrics and visualizations to ensure empirical consistency.
# Formulas & Methods:
#   - Deviation: Δ = |value - target|, validated against empirical thresholds (e.g., Δα_s ≤ 0.005, Δm_H ≤ 0.5).
#   - Validation (CP5, CP6): Ensures entropy-coherent stability and simulation consistency by comparing results to CODATA/Planck targets.
#   - Visualization: Bar plot of deviations, heatmaps of s_field (Script 02) and ψ_α (Scripts 03/06a).
# Postulates:
#   - CP5: Entropy-coherent stability (deviations within thresholds).
#   - CP6: Simulation consistency via validation of prior results.
#   - EP1: Empirical QCD coupling (α_s ≈ 0.118).
#   - EP7: Empirical consistency of spectral fields.
#   - EP11: Empirical Higgs mass (m_H ≈ 125.0 GeV).
# Inputs:
#   - config_empirical*.json: Configuration file with targets (α_s, m_H) and thresholds.
#   - results.csv: Results from Scripts 01–03, 06a (α_s, m_H, etc.).
#   - img/s_field.npy: Field data from Script 02.
#   - img/psi_alpha.npy: Field data from Scripts 03/06a.
# Outputs:
#   - results.csv: Appended validation summary (parameter, value, target, deviation, timestamp).
#   - img/validation_bar_plot.png: Bar plot of deviations with thresholds.
#   - img/validation_s_field_heatmap.png: Heatmap of s_field.
#   - img/validation_psi_alpha_heatmap.png: Heatmap of ψ_α.
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
    """Load JSON configuration file for empirical validation."""
    config_files = glob.glob('config_empirical*.json')
    if not config_files:
        logging.error("No config files matching 'config_empirical*.json'")
        raise FileNotFoundError("Missing config_empirical.json")
    print("Available configuration files:")
    for i, f in enumerate(config_files, 1):
        print(f"  {i}. {f}")
    while True:
        try:
            choice = int(input("Select config file number: ")) - 1
            if 0 <= choice < len(config_files):
                with open(config_files[choice], 'r', encoding='utf-8') as infile:
                    cfg = json.load(infile)
                print(f"[04_empirical_validator.py] Loaded targets: {cfg['targets']}")
                print(f"[04_empirical_validator.py] Loaded thresholds: {cfg['thresholds']}")
                return cfg
            else:
                print("Invalid selection. Please choose a valid number.")
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            logging.error(f"Config loading failed: {e}")
            raise

def load_results():
    """
    Read results.csv and parse entries, skipping non-numeric references/deviations.
    Returns:
        list: List of dictionaries with script, parameter, value, reference, deviation, timestamp.
    """
    print(f"[04_empirical_validator.py] Loading results from results.csv")
    entries = []
    try:
        with open('results.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 6:
                    continue
                script, param, val, ref, dev, ts = row
                try:
                    val_f = float(val)
                except ValueError:
                    continue
                ref_f = float(ref) if ref.replace('.', '', 1).isdigit() else None
                dev_f = float(dev) if dev.replace('.', '', 1).isdigit() else None
                entries.append({
                    'script': script,
                    'parameter': param,
                    'value': val_f,
                    'reference': ref_f,
                    'deviation': dev_f,
                    'timestamp': ts
                })
        print(f"[04_empirical_validator.py] Loaded {len(entries)} entries from results.csv")
        return entries
    except Exception as e:
        logging.error(f"Failed to load results.csv: {e}")
        raise

def validate(entries, cfg):
    """
    Validate entries against empirical targets and thresholds per CP5, CP6, EP1, EP7, EP11.
    Args:
        entries (list): List of result entries.
        cfg (dict): Configuration with targets and thresholds.
    Returns:
        list: Validated entries with parameter, value, target, deviation, threshold, status.
    """
    print(f"[04_empirical_validator.py] Validating results against empirical targets")
    validated = []
    for e in tqdm(entries, desc="Validating entries", unit="entry"):
        key = 'm_h' if e['parameter'] == 'm_H' else e['parameter']
        if key not in cfg['targets']:
            continue
        target = cfg['targets'][key]
        threshold = cfg['thresholds'].get(key, 0.0)
        deviation = abs(e['value'] - target)
        status = 'PASS' if deviation <= threshold else 'FAIL'
        logging.info(f"Validated {e['parameter']}: value={e['value']:.6f}, target={target:.6f}, "
                     f"Δ={deviation:.6f}, threshold={threshold:.6f} → {status}")
        validated.append({
            'parameter': e['parameter'],
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
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, deviations, color='steelblue')
    ax.plot(labels, thresholds, 'r--', label='Threshold')
    ax.set_ylabel('Deviation')
    ax.set_title('Empirical Validation Deviations')
    ax.legend()
    for bar, dev in zip(bars, deviations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e-3,
                f"{dev:.3f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/validation_bar_plot.png')
    plt.close()
    print(f"[04_empirical_validator.py] Bar plot saved → img/validation_bar_plot.png")

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
        plt.savefig('img/validation_s_field_heatmap.png')
        plt.close()
        print(f"[04_empirical_validator.py] s_field heatmap saved → img/validation_s_field_heatmap.png")
    else:
        logging.info("s_field.npy not found; skipping its heatmap")
    
    # psi_alpha heatmap
    psi_path = 'img/psi_alpha.npy'
    if os.path.exists(psi_path):
        psi = np.load(psi_path)
        plt.imshow(np.abs(psi), cmap='plasma', origin='lower')
        plt.colorbar(label='|ψ_α| (Scripts 03/06a)')
        plt.title('Validation: ψ_α Heatmap')
        plt.savefig('img/validation_psi_alpha_heatmap.png')
        plt.close()
        print(f"[04_empirical_validator.py] ψ_α heatmap saved → img/validation_psi_alpha_heatmap.png")
    else:
        logging.info("psi_alpha.npy not found; skipping its heatmap")

def append_summary(validated):
    """
    Append validation summary to results.csv.
    Args:
        validated (list): List of validated entries.
    """
    print(f"[04_empirical_validator.py] Appending validation summary to results.csv")
    ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    with open('results.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for v in validated:
            writer.writerow([
                '04_empirical_validator.py',
                f"{v['parameter']}_validation",
                v['value'],
                v['target'],
                v['deviation'],
                ts
            ])
    print(f"[04_empirical_validator.py] Validation summary appended: {len(validated)} entries")

def main():
    """Main function to orchestrate empirical validation of MSM outputs."""
    clear_screen()
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
    
    # Append validation summary to results.csv
    append_summary(validated)
    
    # Summary output
    print("\n=====================================")
    print("     Meta-Space Model: Summary")
    print("=====================================")
    print(f"Script: 04_empirical_validator.py")
    print(f"Description: Validates MSM outputs (α_s, m_H) against empirical targets")
    print(f"Postulates: CP5, CP6, EP1, EP7, EP11")
    print(f"Validated Parameters: {len(validated)}")
    for v in validated:
        print(f"- {v['parameter']}: value={v['value']:.6f}, target={v['target']:.6f}, "
              f"Δ={v['deviation']:.6f}, threshold={v['threshold']:.6f}, status={v['status']}")
    print(f"Status: {'PASS' if all(v['status'] == 'PASS' for v in validated) else 'FAIL'}")
    print("=====================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
        raise