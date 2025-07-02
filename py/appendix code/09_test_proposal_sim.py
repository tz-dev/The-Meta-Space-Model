# Script: 09_test_proposal_sim.py
# Description: Simulates empirical tests for Meta-Space Model (MSM) predictions on 𝓜_meta = S^3 × CY_3 × ℝ_τ,
#   focusing on Bose-Einstein Condensate (BEC) mass drift and neutrino oscillations.
# Formulas & Methods:
#   - BEC mass drift (EP5): m(t) = 1.0 + Σ(α_s * ∇S_thermo * 0.1), where S_thermo = sin(2π * freq * t) * Y_lm_norm / 1e4.
#   - Neutrino oscillations (EP12): P_ee = [1 - sin^2(2θ) * sin^2(1.27 * Δm^2 * L / E)] * (Y_lm_norm / 1e9) * exp(-L^2 / l_N^2).
#   - Metrics: mass_drift_metric = std(Δm), oscillation_metric = std(P_ee).
# Postulates:
#   - CP6: Simulation consistency (consistent use of α_s, Y_lm_norm across scripts).
#   - EP5: Thermodynamic stability (BEC mass drift within threshold).
#   - EP12: Neutrino oscillations (survival probability P_ee consistent with empirical data).
# Inputs:
#   - config_test*.json: Configuration file with bec_frequency, time_steps, runs, mass_drift_threshold,
#     oscillation_threshold, neutrino (L_min, L_max, num_points, energy, delta_m2, theta, l_N, theta_variation).
#   - results.csv: α_s (Script 01), Y_lm_norm (Script 05).
# Outputs:
#   - results.csv: Stores mass_drift_metric, oscillation_metric, timestamp.
#   - img/test_heatmap_bec.png: Plot of S_thermo for BEC drift.
#   - img/test_heatmap_osc.png: Plot of P_ee for neutrino oscillations.
#   - errors.log: Logs errors.
# References:
#   - BEC: PhysRevLett.126.173403 (2021)
#   - DUNE: PhysRevD.103.112011 (2021)
# Dependencies: numpy, matplotlib, scipy.stats, json, glob, csv, logging, tqdm

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
import platform
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Logging setup
logging.basicConfig(
    filename='errors.log',
    level=logging.ERROR,
    format='%(asctime)s [09_test_proposal_sim.py] %(levelname)s: %(message)s'
)

def clear_screen():
    """Clear the console screen based on the operating system."""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def load_config():
    """Load fixed JSON configuration file for empirical test simulations."""
    config_path = 'config_test.json'
    if not os.path.exists(config_path):
        logging.error(f"Missing fixed config file: {config_path}")
        raise FileNotFoundError(f"Missing {config_path}")
    with open(config_path, 'r', encoding='utf-8') as infile:
        cfg = json.load(infile)
    print(f"[09_test_proposal_sim.py] Loaded fixed config: bec_frequency={cfg['bec_frequency']}, "
          f"time_steps={cfg['time_steps']}, runs={cfg.get('runs', 50)}")
    return cfg

def load_alpha_s():
    """
    Load the most recent α_s from results.csv.
    Returns:
        float: Most recent α_s value.
    """
    print(f"[09_test_proposal_sim.py] Loading α_s from results.csv")
    try:
        rows = []
        with open('results.csv', 'r', encoding='utf-8') as f:
            for r in csv.reader(f):
                if r[1] == 'alpha_s':
                    rows.append((r[5], float(r[2])))
        if not rows:
            logging.error("α_s not found in results.csv")
            raise ValueError("α_s not found in results.csv")
        rows.sort(key=lambda x: x[0])
        alpha_s = rows[-1][1]
        print(f"[09_test_proposal_sim.py] Loaded α_s = {alpha_s:.6f}")
        return alpha_s
    except Exception as e:
        logging.error(f"Failed to load α_s: {e}")
        raise

def load_y_lm_norm():
    """
    Load Y_lm_norm from results.csv.
    Returns:
        float: Y_lm_norm value from Script 05.
    """
    print(f"[09_test_proposal_sim.py] Loading Y_lm_norm from results.csv")
    try:
        with open('results.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == '05_s3_spectral_base.py' and row[1] == 'Y_lm_norm':
                    norm = float(row[2])
                    print(f"[09_test_proposal_sim.py] Loaded Y_lm_norm = {norm:.6f}")
                    return norm
        logging.error("Y_lm_norm not found in results.csv")
        raise ValueError("Y_lm_norm not found in results.csv")
    except Exception as e:
        logging.error(f"Failed to load Y_lm_norm: {e}")
        raise

def simulate_bec_drift(alpha_s, freq, time_steps, y_lm_norm):
    """
    Simulate BEC mass drift per EP5.
    Args:
        alpha_s (float): Strong coupling constant.
        freq (float): Frequency for entropic field.
        time_steps (int): Number of time steps.
        y_lm_norm (float): Spectral norm from Script 05.
    Returns:
        tuple: (m, metric, S_thermo) - Mass array, drift metric, entropic field.
    """
    print(f"[09_test_proposal_sim.py] Simulating BEC mass drift")
    t = np.linspace(0, 1, time_steps)
    S_thermo = np.sin(2 * np.pi * freq * t) * y_lm_norm / 1e4
    grad_S = np.gradient(S_thermo)
    delta_m = alpha_s * grad_S * 0.1
    m = 1.0 + np.cumsum(delta_m)
    metric = float(np.std(delta_m))
    return m, metric, S_thermo

def simulate_neutrino_osc(config, runs, y_lm_norm):
    """
    Simulate neutrino oscillations per EP12.
    Args:
        config (dict): Configuration dictionary with neutrino parameters.
        runs (int): Number of simulation runs.
        y_lm_norm (float): Spectral norm from Script 05.
    Returns:
        tuple: (L, avg_pattern, metric) - Distance array, average P_ee, oscillation metric.
    """
    print(f"[09_test_proposal_sim.py] Simulating neutrino oscillations")
    nu = config['neutrino']
    L = np.linspace(nu['L_min'], nu['L_max'], nu['num_points'])
    E = nu['energy']
    delta_m2 = nu['delta_m2']
    theta_0 = nu['theta']
    l_N = nu['l_N']
    theta_variation = nu['theta_variation']
    
    patterns = []
    for _ in tqdm(range(runs), desc="Running neutrino simulations", unit="run"):
        theta = theta_0 * (1 + theta_variation * np.random.randn())
        P_ee = 1 - (np.sin(2 * theta)**2) * np.sin(1.27 * delta_m2 * L / E)**2
        P_ee *= (y_lm_norm / 1e9) * np.exp(-L**2 / l_N**2)
        patterns.append(P_ee)
    
    patterns = np.array(patterns)
    avg_pattern = patterns.mean(axis=0)
    metric = float(np.std(avg_pattern))
    return L, avg_pattern, metric

def save_heatmap(data, filename, xlabel, ylabel, title):
    """
    Generate and save heatmap or plot for simulation data.
    Args:
        data (array): Data to visualize (1D for plot, 2D for heatmap).
        filename (str): Output filename.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis or colorbar label.
        title (str): Plot title.
    """
    print(f"[09_test_proposal_sim.py] Generating visualization for {filename}")
    os.makedirs('img', exist_ok=True)
    plt.figure(figsize=(6, 4))
    if data.ndim == 1:
        plt.plot(data, lw=1.5)
        plt.ylabel(ylabel)
    else:
        plt.imshow(data, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label=ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    out = os.path.join('img', filename)
    plt.savefig(out)
    plt.close()
    print(f"[09_test_proposal_sim.py] Visualization saved: {out}")

def write_results(mass_metric, osc_metric):
    """
    Write simulation results to results.csv.
    Args:
        mass_metric (float): BEC mass drift metric.
        osc_metric (float): Neutrino oscillation metric.
    """
    print(f"[09_test_proposal_sim.py] Writing results to results.csv")
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    with open('results.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['09_test_proposal_sim.py', 'mass_drift_metric', mass_metric, 'N/A', 'N/A', timestamp])
        writer.writerow(['09_test_proposal_sim.py', 'oscillation_metric', osc_metric, 'N/A', 'N/A', timestamp])
    print(f"[09_test_proposal_sim.py] Results written: mass_drift_metric={mass_metric:.6f}, "
          f"oscillation_metric={osc_metric:.6f}")

def main():
    """Main function to orchestrate empirical test simulations."""
    clear_screen()
    print("====================================================")
    print("    Meta-Space Model: Empirical Test Simulations    ")
    print("====================================================")
    
    # Load configuration and input data
    config = load_config()
    alpha_s = load_alpha_s()
    y_lm_norm = load_y_lm_norm()
    freq = config['bec_frequency']
    time_steps = config['time_steps']
    runs = config.get('runs', 50)
    mass_threshold = config['mass_drift_threshold']
    osc_threshold = config['oscillation_threshold']
    
    with tqdm(total=3, desc="Processing simulations", unit="step") as pbar:
        # Simulate BEC mass drift
        print(f"[09_test_proposal_sim.py] Simulating BEC drift (α_s={alpha_s:.3f}, freq={freq}, T={time_steps})")
        bec_mass, mass_metric, S_thermo = simulate_bec_drift(alpha_s, freq, time_steps, y_lm_norm)
        save_heatmap(S_thermo, 'test_heatmap_bec.png', 'Time Step', 'S_thermo', 'BEC Entropic Drift Pattern')
        status_mass = "PASS" if mass_metric <= mass_threshold else "FAIL"
        print(f"[09_test_proposal_sim.py] mass_drift_metric = {mass_metric:.6f} "
              f"(threshold {mass_threshold:.6f}) → {status_mass}")
        pbar.update(1)
        
        # Simulate neutrino oscillations
        print(f"[09_test_proposal_sim.py] Simulating neutrino oscillations (runs={runs})")
        L, osc_pattern, osc_metric = simulate_neutrino_osc(config, runs, y_lm_norm)
        save_heatmap(osc_pattern, 'test_heatmap_osc.png', 'L Index', 'P_ee', 'Neutrino Survival Probability')
        status_osc = "PASS" if osc_metric <= osc_threshold else "FAIL"
        print(f"[09_test_proposal_sim.py] oscillation_metric = {osc_metric:.6f} "
              f"(threshold {osc_threshold:.6f}) → {status_osc}")
        pbar.update(1)
        
        # Write results
        write_results(mass_metric, osc_metric)
        pbar.update(1)
    
    # Summary output
    print("\n=====================================")
    print("     Meta-Space Model: Summary")
    print("=====================================")
    print(f"Script: 09_test_proposal_sim.py")
    print(f"Description: Simulates BEC mass drift and neutrino oscillations")
    print(f"Postulates: CP6, EP5, EP12")
    print(f"References: BEC (PhysRevLett.126.173403, 2021), DUNE (PhysRevD.103.112011, 2021)")
    print(f"BEC Mass Drift Metric: {mass_metric:.6f} (threshold {mass_threshold:.6f}, status: {status_mass})")
    print(f"Neutrino Oscillation Metric: {osc_metric:.6f} (threshold {osc_threshold:.6f}, status: {status_osc})")
    print(f"Status: {'PASS' if status_mass == 'PASS' and status_osc == 'PASS' else 'FAIL'}")
    print("=====================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
        raise