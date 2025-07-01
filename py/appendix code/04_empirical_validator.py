# Script: 04_empirical_validator.py
# Description: Validates MSM simulation outputs (α_s, m_H) against empirical targets,
#              producing deviation metrics and visualizations.
# Postulates: CP5 (Entropy-Coherent Stability), CP6 (Simulation Consistency),
#             EP1, EP7, EP11 (empirical tests)
# Inputs: results.csv (from Scripts 01–03, 06a), config_empirical.json
# Outputs:
#   - Bar plot of deviations: img/validation_bar_plot.png
#   - Heatmaps of s_field and ψ_α: img/validation_s_field_heatmap.png, img/validation_psi_alpha_heatmap.png
#   - Appended validation summary in results.csv
# Logging: errors.log

import csv
import json
import os
import glob
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# --- Logging Setup ---
logging.basicConfig(
    filename='errors.log',
    level=logging.INFO,
    format='%(asctime)s [04_empirical_validator.py] %(levelname)s: %(message)s'
)

def load_config():
    """Load empirical validation thresholds and targets."""
    files = glob.glob('config_empirical*.json')
    if not files:
        logging.error("No config_empirical.json found")
        raise FileNotFoundError("Missing config_empirical.json")
    print("Available empirical config files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")
    idx = int(input("Select config file number: ")) - 1
    cfg = json.load(open(files[idx], 'r'))
    print(f"[04_empirical_validator.py] Loaded targets: {cfg['targets']}")
    print(f"[04_empirical_validator.py] Loaded thresholds: {cfg['thresholds']}")
    return cfg

def load_results():
    """Read results.csv into a list of entries, skipping non-numeric refs/devs."""
    entries = []
    with open('results.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 6:
                continue
            script, param, val, ref, dev, ts = row
            # parse value
            try:
                val_f = float(val)
            except ValueError:
                continue
            # parse reference if numeric
            try:
                ref_f = float(ref)
            except ValueError:
                ref_f = None
            # parse deviation if numeric
            try:
                dev_f = float(dev)
            except ValueError:
                dev_f = None
            entries.append({
                'script':    script,
                'parameter': param,
                'value':     val_f,
                'reference': ref_f,
                'deviation': dev_f,
                'timestamp': ts
            })
    return entries

def validate(entries, cfg):
    """Compare each entry against its empirical target and threshold."""
    validated = []
    for e in entries:
        key = 'm_h' if e['parameter'] in ('m_H',) else e['parameter']
        if key not in cfg['targets']:
            continue
        target    = cfg['targets'][key]
        threshold = cfg['thresholds'].get(key, 0.0)
        deviation = abs(e['value'] - target)
        status    = 'PASS' if deviation <= threshold else 'FAIL'
        logging.info(f"Validated {e['parameter']}: value={e['value']}, target={target}, "
                     f"Δ={deviation}, thr={threshold} → {status}")
        validated.append({
            'parameter': e['parameter'],
            'value':     e['value'],
            'target':    target,
            'deviation': deviation,
            'threshold': threshold,
            'status':    status
        })
    return validated

def plot_deviations(validated):
    """Bar plot of deviations with threshold line."""
    labels     = [v['parameter'] for v in validated]
    deviations = [v['deviation']  for v in validated]
    thresholds = [v['threshold']  for v in validated]

    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(labels, deviations, color='steelblue')
    ax.plot(labels, thresholds, 'r--', label='Threshold')
    ax.set_ylabel('Deviation')
    ax.set_title('Empirical Validation Deviations')
    ax.legend()
    for bar, dev in zip(bars, deviations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1e-3,
                f"{dev:.3f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/validation_bar_plot.png')
    plt.close()
    print("[04_empirical_validator.py] Bar plot saved → img/validation_bar_plot.png")

def plot_heatmaps():
    """Load and plot s_field.npy and psi_alpha.npy if present."""
    # s_field heatmap
    sf_path = 'img/s_field.npy'
    if os.path.exists(sf_path):
        s_field = np.load(sf_path)
        plt.imshow(np.abs(s_field), cmap='viridis', origin='lower')
        plt.colorbar(label='s_field (Script 02)')
        plt.title('Validation: s_field Heatmap')
        plt.savefig('img/validation_s_field_heatmap.png')
        plt.close()
        print("[04_empirical_validator.py] s_field heatmap saved → img/validation_s_field_heatmap.png")
    else:
        logging.info("s_field.npy not found; skipping its heatmap")

    # psi_alpha heatmap
    psi_path = 'img/psi_alpha.npy'
    if os.path.exists(psi_path):
        psi = np.load(psi_path)
        plt.imshow(np.abs(psi), cmap='plasma', origin='lower')
        plt.colorbar(label='ψ_α (Script 03/06a)')
        plt.title('Validation: ψ_α Heatmap')
        plt.savefig('img/validation_psi_alpha_heatmap.png')
        plt.close()
        print("[04_empirical_validator.py] ψ_α heatmap saved → img/validation_psi_alpha_heatmap.png")
    else:
        logging.info("psi_alpha.npy not found; skipping its heatmap")

def append_summary(validated):
    """Append validation status back into results.csv."""
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
    print("[04_empirical_validator.py] Validation summary appended to results.csv")

def main():
    cfg       = load_config()
    entries   = load_results()
    validated = validate(entries, cfg)
    if not validated:
        print("[04_empirical_validator.py] No parameters validated.")
        return
    plot_deviations(validated)
    plot_heatmaps()
    append_summary(validated)
    print("[04_empirical_validator.py] Empirical validation complete.")

if __name__ == '__main__':
    main()
