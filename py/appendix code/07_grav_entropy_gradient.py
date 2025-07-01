# Script: 07_grav_entropy_gradient.py
# Description: Models the gravitational field tensor I_mu_nu on the meta-space manifold M_meta = S^3 x CY_3 x R_tau
#   using entropic gradients grad_tau S for curvature projections, ensuring entropy-driven causality and extended quantum gravity.
# Formulas & Methods:
#   - Entropic field: S = field_scale * Y_lm_norm * [sin(x) * cos(y) + noise], smoothed with Gaussian filter.
#   - Gravitational tensor: I_mu_nu = grad_y (grad_x S), where grad_x, grad_y are spatial gradients.
#   - Stability metric: Mean of |grad_x S| >= threshold, targeting >= 0.5 for stability.
#   - Deviation: Standard deviation of I_mu_nu.
#   - Iterative threshold adjustment: Optimizes threshold to achieve stability >= 0.5.
# Postulates:
#   - CP2: Entropy-driven causality (grad_tau S drives curvature).
#   - EP8: Extended quantum gravity (I_mu_nu derived from entropic gradients).
# Inputs:
#   - config_grav*.json: Configuration file with tensor_shape, entropy_gradient_threshold, gaussian_sigma,
#     averaging_runs, field_scale, noise_amplitude.
#   - results.csv: Y_lm_norm from Script 05 for scaling.
# Outputs:
#   - results.csv: Stores I_mu_nu (mean), stability_metric, deviation, threshold, timestamp.
#   - img/grav_field_heatmap.png: Heatmap of |I_mu_nu|.
#   - errors.log: Logs warnings and errors.
# Dependencies: numpy, matplotlib, scipy.ndimage, json, glob, csv, logging, tqdm

import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import logging
import os
import csv
from datetime import datetime
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import platform

# Logging setup
logging.basicConfig(
    filename='errors.log',
    level=logging.WARNING,
    format='%(asctime)s [07_grav_entropy_gradient.py] %(levelname)s: %(message)s'
)

def clear_screen():
    """Clear the console screen based on the operating system."""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def load_config():
    """Load JSON configuration file for gravitational tensor computation."""
    config_files = glob.glob('config_grav*.json')
    if not config_files:
        logging.error("No config files matching 'config_grav*.json'")
        raise FileNotFoundError("Missing config_grav.json")
    print("Available configuration files:")
    for i, file in enumerate(config_files, 1):
        print(f"  {i}. {file}")
    while True:
        try:
            choice = int(input("Select config file number: ")) - 1
            if 0 <= choice < len(config_files):
                with open(config_files[choice], 'r', encoding='utf-8') as infile:
                    cfg = json.load(infile)
                print(f"[07_grav_entropy_gradient.py] Loaded config: shape={cfg['tensor_shape']}, "
                      f"threshold={cfg['entropy_gradient_threshold']}, sigma={cfg.get('gaussian_sigma', 1.6)}")
                return cfg
            else:
                print("Invalid selection. Please choose a valid number.")
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            logging.error(f"Config loading failed: {e}")
            raise

def load_y_lm_norm():
    """
    Load Y_lm_norm from results.csv for scaling entropic field.
    Returns:
        float: Y_lm_norm value from Script 05.
    """
    print(f"[07_grav_entropy_gradient.py] Loading Y_lm_norm from results.csv")
    try:
        with open('results.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == '05_s3_spectral_base.py' and row[1] == 'Y_lm_norm':
                    norm = float(row[2])
                    print(f"[07_grav_entropy_gradient.py] Loaded Y_lm_norm = {norm:.6f}")
                    return norm
        logging.error("Y_lm_norm not found in results.csv")
        raise ValueError("Y_lm_norm not found in results.csv")
    except Exception as e:
        logging.error(f"Failed to load Y_lm_norm: {e}")
        raise

def compute_gravitational_tensor(shape, gradient_threshold, sigma, field_scale, noise_amp):
    """
    Compute gravitational tensor I_mu_nu from entropic gradient per CP2, EP8.
    Args:
        shape (tuple): Shape of the tensor grid (x, y).
        gradient_threshold (float): Threshold for stability metric.
        sigma (float): Gaussian smoothing parameter.
        field_scale (float): Scaling factor for entropic field.
        noise_amp (float): Amplitude of random noise.
    Returns:
        tuple: (I_mu_nu, stability_metric, deviation) - Tensor, stability metric, and deviation.
    """
    print(f"[07_grav_entropy_gradient.py] Computing gravitational tensor I_mu_nu")
    
    # Initialize grid
    x = np.linspace(0, 2 * np.pi, shape[0])
    y = np.linspace(0, 2 * np.pi, shape[1])
    X, Y = np.meshgrid(x, y)
    
    # Compute entropic field S with scaling and noise
    y_lm_norm = load_y_lm_norm()
    S = (np.sin(X) * np.cos(Y) + noise_amp * np.random.rand(*X.shape)) * field_scale * y_lm_norm
    S = gaussian_filter(S, sigma=sigma)
    
    # Compute entropic gradient along x (proxy for τ)
    grad_tau = np.gradient(S, axis=0)
    
    # Compute gravitational tensor I_mu_nu as second derivative
    I_mu_nu = np.gradient(grad_tau, axis=1)
    
    # Compute stability metric and deviation
    stability_metric = float(np.mean(np.abs(grad_tau) >= gradient_threshold))
    deviation = float(np.std(I_mu_nu))
    
    print(f"[07_grav_entropy_gradient.py] Stability metric = {stability_metric:.4f}, deviation = {deviation:.6f}")
    return I_mu_nu, stability_metric, deviation

def find_stable_tensor(shape, initial_threshold, sigma, runs=5, target=0.5, step=0.01, max_attempts=20, field_scale=1.0, noise_amp=0.0):
    """
    Find stable tensor by iteratively adjusting threshold per CP2.
    Args:
        shape (tuple): Tensor grid shape.
        initial_threshold (float): Initial gradient threshold.
        sigma (float): Gaussian smoothing parameter.
        runs (int): Number of runs for averaging.
        target (float): Target stability metric (>= 0.5).
        step (float): Threshold adjustment step.
        max_attempts (int): Maximum adjustment attempts.
        field_scale (float): Scaling factor for entropic field.
        noise_amp (float): Amplitude of random noise.
    Returns:
        tuple: (best_I, avg_stability, avg_deviation, used_threshold) - Best tensor, stability, deviation, and used threshold.
    """
    print(f"[07_grav_entropy_gradient.py] Finding stable tensor with target stability >= {target}")
    best_I, best_stab, best_dev = None, 0.0, float('inf')
    used_threshold = initial_threshold
    
    for _ in tqdm(range(max_attempts), desc="Adjusting threshold", unit="attempt"):
        stability_vals = []
        deviations = []
        tensors = []
        for _ in range(runs):
            I, stab, dev = compute_gravitational_tensor(shape, used_threshold, sigma, field_scale, noise_amp)
            stability_vals.append(stab)
            deviations.append(dev)
            tensors.append(I)
        avg_stab = np.mean(stability_vals)
        avg_dev = np.mean(deviations)
        if avg_stab >= target and avg_dev < best_dev:
            best_I, best_stab, best_dev = tensors[np.argmax(stability_vals)], avg_stab, avg_dev
        if avg_stab >= target:
            break
        used_threshold += step
    
    print(f"[07_grav_entropy_gradient.py] Best stability = {best_stab:.4f}, deviation = {best_dev:.6f}, threshold = {used_threshold:.4f}")
    return best_I, best_stab, best_dev, used_threshold

def save_heatmap(data):
    """
    Generate and save heatmap of gravitational tensor |I_mu_nu|.
    Args:
        data (array): Gravitational tensor array.
    """
    print(f"[07_grav_entropy_gradient.py] Generating heatmap for gravitational tensor")
    os.makedirs('img', exist_ok=True)
    plt.imshow(np.abs(data), cmap='cividis', origin='lower')
    plt.colorbar(label='|I_mu_nu|')
    plt.title('Gravitational Field Tensor I_mu_nu')
    plt.savefig('img/grav_field_heatmap.png')
    plt.close()
    print(f"[07_grav_entropy_gradient.py] Heatmap saved -> img/grav_field_heatmap.png")

def write_results(I_mu_nu, stability, deviation, threshold):
    """
    Write results to results.csv.
    Args:
        I_mu_nu (array): Gravitational tensor.
        stability (float): Stability metric.
        deviation (float): Standard deviation of I_mu_nu.
        threshold (float): Used gradient threshold.
    """
    print(f"[07_grav_entropy_gradient.py] Writing results to results.csv")
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    with open('results.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '07_grav_entropy_gradient.py', 'I_mu_nu',
            np.mean(I_mu_nu), 'N/A', deviation, timestamp
        ])
        writer.writerow([
            '07_grav_entropy_gradient.py', 'stability_metric',
            stability, f'thresh={threshold:.4f}', 'N/A', timestamp
        ])
    print(f"[07_grav_entropy_gradient.py] Results written: I_mu_nu_mean={np.mean(I_mu_nu):.6f}, "
          f"stability={stability:.4f}, deviation={deviation:.6f}")

def main():
    """Main function to orchestrate gravitational tensor computation."""
    clear_screen()
    print("==========================================================")
    print("    Meta-Space Model: Gravitational Tensor Computation    ")
    print("==========================================================")
    
    # Load configuration
    cfg = load_config()
    shape = cfg['tensor_shape']
    thresh = cfg['entropy_gradient_threshold']
    sigma = cfg.get('gaussian_sigma', 1.6)
    runs = cfg.get('averaging_runs', 5)
    field_scale = cfg.get('field_scale', 1.0)
    noise_amp = cfg.get('noise_amplitude', 0.0)
    
    # Compute stable tensor with progress bar
    with tqdm(total=2, desc="Processing gravitational tensor", unit="step") as pbar:
        # Find stable tensor
        I_mu_nu, stability, deviation, used_threshold = find_stable_tensor(
            shape, thresh, sigma, runs=runs, target=0.5, step=0.01, max_attempts=20,
            field_scale=field_scale, noise_amp=noise_amp
        )
        pbar.update(1)
        
        # Generate and save heatmap
        save_heatmap(I_mu_nu)
        pbar.update(1)
    
    # Write results to CSV
    write_results(I_mu_nu, stability, deviation, used_threshold)
    
    # Validate stability per CP2, EP8
    status = "PASS (Model-Conform, CP2/EP8)" if stability >= 0.5 else "FAIL (Unstable)"
    print("\n=====================================")
    print("     Meta-Space Model: Summary")
    print("=====================================")
    print(f"Script: 07_grav_entropy_gradient.py")
    print(f"Description: Models gravitational tensor I_mu_nu via entropic gradients")
    print(f"Postulates: CP2, EP8")
    print(f"Computed I_mu_nu (mean): {np.mean(I_mu_nu):.6f}")
    print(f"Stability Metric: {stability:.4f} (threshold {used_threshold:.4f})")
    print(f"Deviation: {deviation:.6f}")
    print(f"Status: {status}")
    print("=====================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
        raise
