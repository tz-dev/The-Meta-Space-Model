# Script: 09_test_proposal_sim.py
# Description: Simulates empirical tests for MSM-Vorhersagen:
#   1) BEC-Massendrift (EP5: Thermodynamic Stability)
#   2) Neutrino-Oszillationen (EP12: Neutrino Oscillations)
# Postulates: CP6 (Simulation Consistency), EP5, EP12
# Inputs: config_test.json, results.csv (α_s, Y_lm_norm)
# Outputs: mass_drift_metric, oscillation_metric, heatmaps
# Logging: Errors to errors.log
# References: BEC (PhysRevLett.126.173403, 2021), DUNE (PhysRevD.103.112011, 2021)

import json
import glob
import logging
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm
from tqdm import tqdm

# Logging-Setup
logging.basicConfig(
    filename='errors.log',
    level=logging.ERROR,
    format='%(asctime)s [09_test_proposal_sim.py] %(levelname)s: %(message)s'
)

def load_config():
    files = glob.glob('config_test*.json')
    if not files:
        logging.error("Keine config_test.json gefunden")
        raise FileNotFoundError("config_test.json fehlt")
    print("Available configuration files:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")
    choice = int(input("Select config file number: ")) - 1
    return json.load(open(files[choice], 'r'))

def load_alpha_s():
    try:
        rows = []
        with open('results.csv', 'r', encoding='utf-8') as f:
            for r in csv.reader(f):
                if r[1] == 'alpha_s':
                    rows.append((r[5], float(r[2])))
        if not rows:
            raise ValueError("Kein alpha_s in results.csv")
        rows.sort(key=lambda x: x[0])
        return rows[-1][1]
    except Exception as e:
        logging.error(f"load_alpha_s failed: {e}")
        raise

def load_y_lm_norm():
    try:
        with open('results.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == '05_s3_spectral_base.py' and row[1] == 'Y_lm_norm':
                    return float(row[2])
        logging.error("Y_lm_norm not found in results.csv")
        raise ValueError("Y_lm_norm not found in results.csv")
    except Exception as e:
        logging.error(f"Failed to load Y_lm_norm: {e}")
        raise

def simulate_bec_drift(alpha_s, freq, time_steps, y_lm_norm):
    t = np.linspace(0, 1, time_steps)
    S_thermo = np.sin(2 * np.pi * freq * t) * y_lm_norm / 1e4
    grad_S = np.gradient(S_thermo)
    delta_m = alpha_s * grad_S * 0.1
    m = 1.0 + np.cumsum(delta_m)
    metric = np.std(delta_m)
    return m, metric, S_thermo

def simulate_neutrino_osc(config, runs, y_lm_norm):
    ν = config['neutrino']
    L = np.linspace(ν['L_min'], ν['L_max'], ν['num_points'])
    E = ν['energy']
    Δm2 = ν['delta_m2']
    θ0 = ν['theta']
    l_N = config['neutrino']['l_N']
    theta_variation = config['neutrino']['theta_variation']
    patterns = []
    for _ in range(runs):
        θ = θ0 * (1 + theta_variation * np.random.randn())
        P_ee = 1 - (np.sin(2 * θ)**2) * np.sin(1.27 * Δm2 * L / E)**2
        P_ee *= (y_lm_norm / 1e9) * np.exp(-L**2 / l_N**2)
        patterns.append(P_ee)
    patterns = np.array(patterns)
    avg_pattern = patterns.mean(axis=0)
    metric = np.std(avg_pattern)
    return L, avg_pattern, metric

def save_heatmap(data, filename, xlabel, ylabel, title):
    try:
        os.makedirs('img', exist_ok=True)
        plt.figure(figsize=(6, 4))
        if data.ndim == 1:
            plt.plot(data, lw=1.5)
        else:
            plt.imshow(data, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label=ylabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out = os.path.join('img', filename)
        plt.savefig(out)
        plt.close()
        print(f"[09_test_proposal_sim.py] Heatmap saved: {out}")
    except Exception as e:
        logging.error(f"save_heatmap({filename}) failed: {e}")
        raise

def write_results(mass_metric, osc_metric):
    ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    try:
        with open('results.csv', 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['09_test_proposal_sim.py', 'mass_drift_metric', mass_metric, 'N/A', 'N/A', ts])
            w.writerow(['09_test_proposal_sim.py', 'oscillation_metric', osc_metric, 'N/A', 'N/A', ts])
        print(f"[09_test_proposal_sim.py] Results written to results.csv")
    except Exception as e:
        logging.error(f"write_results failed: {e}")
        raise

def main():
    try:
        config = load_config()
        alpha_s = load_alpha_s()
        y_lm_norm = load_y_lm_norm()
        freq = config['bec_frequency']
        T = config['time_steps']
        runs = config.get('runs', 50)
        m_thr = config['mass_drift_threshold']
        o_thr = config['oscillation_threshold']

        print(f"[09_test_proposal_sim.py] Simulating BEC drift (α_s={alpha_s:.3f}, freq={freq}, T={T})")
        bec_mass, mass_metric, S_thermo = simulate_bec_drift(alpha_s, freq, T, y_lm_norm)
        save_heatmap(S_thermo, 'test_heatmap_bec.png',
                     'time step', 'S_thermo', 'BEC Entropic Drift Pattern')
        status_mass = "PASS" if mass_metric <= m_thr else "FAIL"
        print(f"  • mass_drift_metric = {mass_metric:.6f} (threshold {m_thr}) → {status_mass}")

        print(f"[09_test_proposal_sim.py] Simulating Neutrino Oscillations (runs={runs})")
        L, osc_pattern, osc_metric = simulate_neutrino_osc(config, runs, y_lm_norm)
        save_heatmap(osc_pattern, 'test_heatmap_osc.png',
                     'L index', 'P_ee', 'Neutrino Survival Probability')
        status_osc = "PASS" if osc_metric <= o_thr else "FAIL"
        print(f"  • oscillation_metric = {osc_metric:.6f} (threshold {o_thr}) → {status_osc}")

        write_results(mass_metric, osc_metric)

        print("[09_test_proposal_sim.py] Computation complete.")
    except Exception as e:
        logging.error(f"Main failed: {e}")
        raise

if __name__ == '__main__':
    main()