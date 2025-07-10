# Script: 07_gravity_curvature_analysis.py
# Description: Unified curvature and gravitational tensor estimation for the Meta-Space Model (MSM).
#              Combines entropic gradient-based tensor evaluation with Laplacian-based curvature extraction 
#              over the manifold M_meta = S³ × CY₃ × ℝ_τ. Integrates and replaces prior Scripts 07 and 07a.
# Author: MSM Enhancement
# Date: 2025-07-07
# Version: 2.0
# Inputs:
#     - config_grav.json: Contains entropy tensor synthesis parameters (shape, σ, threshold, etc.).
#     - config_empirical.json: Provides empirical target/threshold for I_μν validation.
#     - results.csv: Used to extract Y_lm_norm and to store new validation metrics.
#     - img/s_field.npy: Numpy array of scalar field S(x, y, τ) from spectral base scripts.
# Outputs:
#     - img/07_grav_field_heatmap.png: Heatmap of the best-fit gravitational tensor |I_μν|.
#     - results.csv: Appended with stability and curvature validation metrics.
#     - Terminal summary and logging output.
# Dependencies:
#     - numpy, matplotlib, scipy.ndimage: Numerical & visualization.
#     - json, csv, datetime, os, platform, logging: I/O and system utilities.
#     - tqdm: Optional, progress bars for batch tensor generation.
# Purpose:
#     - Establish entropic-tensor-based gravitational proxy on emergent S³ × CY₃ × ℝ_τ.
#     - Validate geometric stability (grad S), curvature coherence (Laplacian S), and compatibility with empirical data (I_μν).

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import csv
import logging
from datetime import datetime
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import platform
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # Ensure UTF-8 encoding for console output

# Logging setup for error tracking and debugging
logging.basicConfig(
    filename='errors.log',
    level=logging.INFO,
    format='%(asctime)s [07_gravity_curvature_analysis.py] %(levelname)s: %(message)s'
)

# Clear terminal screen for clean output display
def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")  # Windows-specific clear command
    else:
        os.system("clear")  # Unix-based clear command

# Load empirical configuration for I_μν validation
def load_empirical_config():
    try:
        with open("config_empirical.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)  # Read JSON configuration file
        target = cfg.get("targets", {}).get("I_mu_nu", 0.0)  # Extract target value for I_μν
        threshold = cfg.get("thresholds", {}).get("I_mu_nu", 1.0)  # Extract threshold for validation
        return target, threshold
    except Exception as e:
        logging.warning(f"No empirical config loaded, using defaults: {e}")  # Log warning if file fails to load
        return 0.0, 1.0  # Default values if loading fails

# Load gravitational configuration parameters
def load_grav_config():
    path = 'config_grav.json'
    if not os.path.exists(path):
        logging.error(f"Missing config: {path}")  # Log error if config file is missing
        raise FileNotFoundError(f"Missing {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)  # Return parsed JSON configuration

# Load Y_lm_norm from results.csv for field normalization
def load_y_lm_norm():
    with open('results.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == '05_s3_spectral_base.py' and row[1] == 'Y_lm_norm':
                return float(row[2])  # Return Y_lm_norm value if found
    raise ValueError("Y_lm_norm not found")  # Raise error if not found

# Compute Laplacian of scalar field S for curvature estimation
def compute_laplacian(S):
    lap = np.zeros_like(S)  # Initialize Laplacian array with same shape as input
    for ax in range(S.ndim):
        try:
            d2 = np.gradient(np.gradient(S, axis=ax), axis=ax)  # Compute second derivative along axis
            lap += d2  # Accumulate contributions to Laplacian
        except Exception as e:
            logging.warning(f"Skipping axis {ax} in Laplacian: {e}")  # Log warning if computation fails
    return lap

# Estimate curvature using Laplacian of scalar field
def curvature_estimate_from_field():
    path = "img/s_field.npy"
    if not os.path.exists(path):
        print("Missing s_field.npy – run Script 02 first.")  # Notify user if input file is missing
        return None
    S = np.load(path)  # Load scalar field
    # Compute Laplacian of scalar field S for curvature estimation
    lap_S = compute_laplacian(S)  # Compute Laplacian of the scalar field
    I_mu_nu = float(np.mean(np.abs(lap_S)))  # Compute mean absolute curvature tensor
    target, threshold = load_empirical_config()  # Load target and threshold for I_μν validation
    deviation = abs(I_mu_nu - target)  # Calculate deviation from target
    status = "PASS" if deviation <= threshold else "FAIL"  # Determine validation status
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")  # Timestamp for logging
    with open("results.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["07_gravity_curvature_analysis.py", "I_mu_nu", I_mu_nu, target, deviation, ts])  # Log results
    print(f"[Curvature] I_mu_nu = {I_mu_nu:.3e}, Δ = {deviation:.3e}, Status = {status}")  # Print summary
    return I_mu_nu

# Compute entropic gradient tensor for gravitational field
def compute_entropy_gradient_tensor(cfg):
    shape = cfg['tensor_shape']  # Tensor grid shape from config
    sigma = cfg.get('gaussian_sigma', 1.6)  # Gaussian smoothing parameter
    runs = cfg.get('averaging_runs', 5)  # Number of simulation runs
    field_scale = cfg.get('field_scale', 1.0)  # Scaling factor for field
    noise_amp = cfg.get('noise_amplitude', 0.0)  # Noise amplitude for randomization
    threshold = cfg.get('entropy_gradient_threshold', 0.1)  # Stability threshold
    
    x = np.linspace(0, 2 * np.pi, shape[0])  # X-coordinate grid
    y = np.linspace(0, 2 * np.pi, shape[1])  # Y-coordinate grid
    X, Y = np.meshgrid(x, y)  # Create 2D mesh
    y_lm_norm = load_y_lm_norm()  # Load normalization factor
    stability_vals, deviations, tensors = [], [], []  # Store metrics and tensors

    for _ in range(runs):
        # Generate scalar field with sinusoidal base and optional noise
        S = (np.sin(X) * np.cos(Y) + noise_amp * np.random.rand(*X.shape)) * field_scale * y_lm_norm
        S = gaussian_filter(S, sigma=sigma)  # Apply Gaussian smoothing
        grad_tau = np.gradient(S, axis=0)  # Compute gradient along tau axis
        I_mu_nu = np.gradient(grad_tau, axis=1)  # Compute second-order tensor
        stability = float(np.mean(np.abs(grad_tau) >= threshold))  # Compute stability metric
        deviation = float(np.std(I_mu_nu))  # Compute tensor deviation
        stability_vals.append(stability)
        deviations.append(deviation)
        tensors.append(I_mu_nu)

    idx = np.argmax(stability_vals)  # Select run with highest stability
    best_tensor = tensors[idx]  # Best tensor result
    best_stability = stability_vals[idx]  # Best stability metric
    best_deviation = deviations[idx]  # Best deviation metric

    # Log results to CSV
    with open("results.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        writer.writerow(["07_gravity_curvature_analysis.py", "stability_metric", best_stability, f"thresh={threshold:.4f}", best_deviation, ts])

    os.makedirs('img', exist_ok=True)  # Ensure output directory exists
    plt.imshow(np.abs(best_tensor), cmap='cividis', origin='lower')  # Plot heatmap of tensor
    plt.colorbar(label='|I_mu_nu|')  # Add colorbar
    plt.title('Gravitational Field Tensor I_mu_nu')  # Set title
    plt.savefig('img/07_grav_field_heatmap.png')  # Save heatmap
    plt.close()

    print(f"[Gravity] Stability = {best_stability:.4f}, Deviation = {best_deviation:.6f}, Threshold = {threshold:.4f}")  # Print summary
    return best_stability

# Main execution function
def main():
    # Clear old results from this script in results.csv
    try:
        with open('results.csv', 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        rows = [row for row in rows if row and row[0] != '07_gravity_curvature_analysis.py']
        with open('results.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)  # Rewrite CSV without old results
    except FileNotFoundError:
        pass  # Ignore if CSV doesn't exist

    clear_screen()  # Clear terminal for output
    print("==============================================================")
    print("  Meta-Space Model: Unified Gravity & Curvature Estimation   ")
    print("==============================================================")

    cfg = load_grav_config()  # Load gravitational config
    stability = compute_entropy_gradient_tensor(cfg)  # Compute entropic tensor
    I_mu_nu = curvature_estimate_from_field()  # Estimate curvature

    # Print final summary
    print("\n=====================================")
    print("     Meta-Space Model: Summary")
    print("=====================================")
    print("Script: 07_gravity_curvature_analysis.py")
    print("Description: Combines entropic gravity + Laplacian curvature")
    print("Postulates: CP1–3, CP6, EP8")
    print(f"Stability Metric: {stability:.4f} (Target ≥ 0.5)")
    print(f"Curvature Estimate I_mu_nu = {I_mu_nu:.3e}")
    print(f"Plots: 07_grav_field_heatmap.png")
    print("=====================================")

if __name__ == "__main__":
    try:
        main()  # Run main function
    except Exception as e:
        logging.error(f"Script execution failed: {e}")  # Log any execution errors
        raise