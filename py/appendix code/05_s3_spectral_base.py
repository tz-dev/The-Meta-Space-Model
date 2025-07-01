# Script: 05_s3_spectral_base.py
# Description: Computes spectral basis functions (Y_lm) on S^3 for the meta-space manifold 𝓜_meta = S^3 × CY_3 × ℝ_τ,
#   ensuring topological protection via summation of spherical harmonic modes.
# Formulas & Methods:
#   - Spherical harmonics: Y_lm(θ, φ) = Σ sph_harm_y(m, l, φ, θ) for l ∈ [0, l_max], m ∈ [-min(l, m_max), min(l, m_max)].
#   - Spectral norm: norm = Σ |Y_lm|^2, validated against CP8 range [1e3, 1e6].
#   - Visualization: Heatmap of |Y_lm| to inspect spectral distribution.
# Postulates:
#   - CP8: Topological protection (spectral norm within [1e3, 1e6] ensures robust S^3 basis).
# Inputs:
#   - config_s3*.json: Configuration file with spectral_modes (l_max, m_max), resolution.
# Outputs:
#   - results.csv: Stores Y_lm_norm, l_max, m_max, timestamp.
#   - img/s3_spectral_heatmap.png: Heatmap of |Y_lm|.
#   - errors.log: Logs debug and error messages.
# Dependencies: numpy, matplotlib, scipy.special, json, glob, csv, logging, tqdm

import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import logging
import os
import csv
from datetime import datetime
from scipy.special import sph_harm_y
from tqdm import tqdm
import platform

# Logging setup
logging.basicConfig(
    filename='errors.log',
    level=logging.DEBUG,
    format='%(asctime)s [05_s3_spectral_base.py] %(levelname)s: %(message)s'
)

def clear_screen():
    """Clear the console screen based on the operating system."""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def load_config():
    """Load JSON configuration file for S^3 spectral basis computation."""
    config_files = glob.glob('config_s3*.json')
    if not config_files:
        logging.error("No config files matching 'config_s3*.json'")
        raise FileNotFoundError("Missing config_s3.json")
    print("Available configuration files:")
    for i, f in enumerate(config_files, 1):
        print(f"  {i}. {f}")
    while True:
        try:
            choice = int(input("Select config file number: ")) - 1
            if 0 <= choice < len(config_files):
                with open(config_files[choice], 'r', encoding='utf-8') as infile:
                    cfg = json.load(infile)
                print(f"[05_s3_spectral_base.py] Loaded config: l_max={cfg['spectral_modes']['l_max']}, "
                      f"m_max={cfg['spectral_modes']['m_max']}, resolution={cfg.get('resolution', 200)}")
                return cfg
            else:
                print("Invalid selection. Please choose a valid number.")
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            logging.error(f"Config loading failed: {e}")
            raise

def compute_spectral_basis(l_max, m_max, theta, phi):
    """
    Compute spectral basis Y_lm on S^3 by summing spherical harmonic modes per CP8.
    Args:
        l_max (int): Maximum angular momentum quantum number.
        m_max (int): Maximum magnetic quantum number.
        theta (array): Theta grid for S^3.
        phi (array): Phi grid for S^3.
    Returns:
        tuple: (Y, norm) - Spectral basis array and its norm.
    """
    print(f"[05_s3_spectral_base.py] Computing Y_lm on S^3 (l_max={l_max}, m_max={m_max})")
    Y = np.zeros(theta.shape, dtype=np.complex128)
    
    # Sum spherical harmonics over l and m
    for l in tqdm(range(l_max + 1), desc="Summing l modes", unit="l"):
        for m in range(-min(l, m_max), min(l, m_max) + 1):
            Y += sph_harm_y(m, l, phi, theta)
    
    # Compute spectral norm
    norm = float(np.sum(np.abs(Y)**2))
    logging.debug(f"Spectral basis norm = {norm:.6f}")
    
    print(f"[05_s3_spectral_base.py] Computed spectral norm = {norm:.6e}")
    return Y, norm

def save_heatmap(Y, filename):
    """
    Generate and save heatmap of spectral basis |Y_lm|.
    Args:
        Y (array): Spectral basis array.
        filename (str): Output filename for heatmap.
    """
    print(f"[05_s3_spectral_base.py] Generating heatmap for spectral basis")
    os.makedirs('img', exist_ok=True)
    plt.imshow(np.abs(Y), cmap='viridis', origin='lower')
    plt.colorbar(label='|Y_lm|')
    plt.title('Spectral Basis on S^3')
    plt.savefig(f'img/{filename}')
    plt.close()
    print(f"[05_s3_spectral_base.py] Heatmap saved: img/{filename}")

def write_results(norm, l_max, m_max):
    """
    Write spectral norm results to results.csv.
    Args:
        norm (float): Spectral basis norm.
        l_max (int): Maximum angular momentum quantum number.
        m_max (int): Maximum magnetic quantum number.
    """
    print(f"[05_s3_spectral_base.py] Writing results to results.csv")
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    with open('results.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '05_s3_spectral_base.py',
            'Y_lm_norm',
            norm,
            '[1e3, 1e6]',
            'N/A',
            timestamp
        ])
    print(f"[05_s3_spectral_base.py] Results written: Y_lm_norm={norm:.6f}, timestamp={timestamp}")

def main():
    """Main function to orchestrate S^3 spectral basis computation."""
    clear_screen()
    print("========================================================")
    print("    Meta-Space Model: S^3 Spectral Basis Computation    ")
    print("========================================================")
    
    # Load configuration
    cfg = load_config()
    l_max = cfg['spectral_modes']['l_max']
    m_max = cfg['spectral_modes']['m_max']
    resolution = cfg.get('resolution', 200)
    
    # Initialize grid for θ, φ on S^3
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)
    
    # Compute spectral basis with progress bar
    with tqdm(total=2, desc="Processing spectral basis", unit="step") as pbar:
        # Compute Y_lm and norm
        Y, norm = compute_spectral_basis(l_max, m_max, theta, phi)
        pbar.update(1)
        
        # Generate and save heatmap
        save_heatmap(Y, 's3_spectral_heatmap.png')
        pbar.update(1)
    
    # Write results to CSV
    write_results(norm, l_max, m_max)
    
    # Validate norm per CP8
    status = "PASS (Model-Conform, CP8)" if 1e3 <= norm <= 1e6 else "FAIL (Out of Range)"
    print("\n=====================================")
    print("     Meta-Space Model: Summary")
    print("=====================================")
    print(f"Script: 05_s3_spectral_base.py")
    print(f"Description: Computes spectral basis Y_lm on S^3")
    print(f"Postulates: CP8")
    print(f"Computed Y_lm_norm: {norm:.6f} (range [1e3, 1e6])")
    print(f"Status: {status}")
    print("=====================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
        raise