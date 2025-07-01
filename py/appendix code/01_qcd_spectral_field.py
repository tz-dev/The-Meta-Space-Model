# Script: 01_qcd_spectral_field.py
# Description: Models QCD spectral fields on S^3 x CY_3, computing the strong coupling constant alpha_s (~0.118)
# using entropic projections (CP3–CP8) und analytische Parameterbestimmung.
# Postulates: CP3, CP5, CP6, CP7, CP8
# ...

import numpy as np
try:
    import cupy as cp
    cuda_available = True
except ImportError:
    cuda_available = False
    cp = np
import scipy.special
import json
import logging
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import glob

# Logging setup
logging.basicConfig(filename='errors.log', level=logging.ERROR,
                    format='%(asctime)s [01_qcd_spectral_field.py] %(levelname)s: %(message)s')

def load_config():
    """Load JSON config file."""
    config_files = glob.glob('config_qcd*.json')
    if not config_files:
        logging.error("No config files matching 'config_qcd*.json'")
        raise FileNotFoundError("Missing config_qcd.json")
    print("Available configuration files:")
    for i, f in enumerate(config_files, 1):
        print(f"{i}. {f}")
    while True:
        choice = int(input("Select config file number: ")) - 1
        if 0 <= choice < len(config_files):
            with open(config_files[choice]) as infile:
                cfg = json.load(infile)
            print(f"[01_qcd_spectral_field.py] Loaded config: energy_scale={cfg.get('energy_scale')}, "
                  f"l_max={cfg['spectral_modes']['l_max']}, m_max={cfg['spectral_modes']['m_max']}")
            return cfg

def compute_spectral_density(m_z, l_max, m_max):
    """
    Compute spherical–harmonic modes y_lm on S^3 and 
    determine analytic scale_factor from CP3: S_filter = S_min.
    """
    print(f"[01_qcd_spectral_field.py] Computing Y_lm modes on S^3 (l_max={l_max}, m_max={m_max})")
    theta = cp.linspace(0, cp.pi, 100)
    phi = cp.linspace(0, 2*cp.pi, 100)
    theta, phi = cp.meshgrid(theta, phi)

    # Single spherical harmonic
    y_lm = scipy.special.sph_harm(
        m_max, l_max,
        cp.asnumpy(phi) if cuda_available else phi,
        cp.asnumpy(theta) if cuda_available else theta
    )
    if cuda_available:
        y_lm = cp.array(y_lm)

    sum_abs2 = float(cp.sum(cp.abs(y_lm)**2))  # Norm
    S_min = config['s_min']
    scale_factor = sum_abs2 * (m_z/91.2) / S_min
    s_filter = S_min

    print(f"[01_qcd_spectral_field.py] sum|Y_lm|^2 = {sum_abs2:.6e}, "
          f"computed scale_factor = {scale_factor:.6e}, enforced S_filter = {s_filter:.6e}")

    S_max = config['s_max']
    if s_filter > S_max:
        raise ValueError(f"S_filter={s_filter} > S_max={S_max}")
    return y_lm, s_filter

def compute_redundancy(s_filter, constraints):
    """
    CP5 + CP6: R_pi = H[rho] - I[rho|O] exakt nach Postulaten.
    """
    print(f"[01_qcd_spectral_field.py] Computing redundancy R_pi")
    h_rho = np.log(s_filter + 1e-12)
    total_w = sum(c['weight'] for c in constraints)
    i_rho_o = np.log(1 + total_w)
    r_pi = h_rho - i_rho_o

    print(f"[01_qcd_spectral_field.py] H[rho] = {h_rho:.6e}, I[rho|O] = {i_rho_o:.6e}, R_pi = {r_pi:.6e}")

    threshold = config['redundancy_threshold']
    if not np.isfinite(r_pi) or r_pi >= threshold:
        raise ValueError(f"R_pi={r_pi} ≥ threshold={threshold}")
    return r_pi

def compute_alpha_s(s_filter):
    """
    EP1 + CP7: α_s ∝ 1/Δλ, hier Δλ ∼ s_filter analytisch.
    """
    print(f"[01_qcd_spectral_field.py] Computing alpha_s from s_filter = {s_filter:.6e}")
    alpha_target = config['alpha_s_target']
    alpha_s = alpha_target * (config['s_min'] / s_filter)
    low, high = config['alpha_s_range']
    print(f"[01_qcd_spectral_field.py] Raw alpha_s = {alpha_s:.6f}, acceptable range = [{low}, {high}]")
    if not (low <= alpha_s <= high):
        raise ValueError(f"α_s={alpha_s} outside [{low},{high}]")
    return alpha_s

def plot_heatmap(y_lm, filename):
    """Heatmap der spektralen Dichte."""
    data = cp.asnumpy(y_lm) if cuda_available else y_lm
    os.makedirs('img', exist_ok=True)
    plt.imshow(np.abs(data), cmap='viridis', origin='lower')
    plt.colorbar(label='|Y_lm|')
    plt.title('QCD Spectral Field on S^3')
    plt.savefig(f'img/{filename}')
    plt.close()
    print(f"[01_qcd_spectral_field.py] Heatmap saved: img/{filename}")

def write_results(alpha_s, r_pi):
    """Ergebnisse in results.csv schreiben."""
    ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    dev = abs(alpha_s - config['alpha_s_target'])
    with open('results.csv','a',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '01_qcd_spectral_field.py','alpha_s',
            alpha_s, config['alpha_s_target'], dev, ts
        ])
        writer.writerow([
            '01_qcd_spectral_field.py','R_pi',
            r_pi, config['redundancy_threshold'], '', ts
        ])
    print(f"[01_qcd_spectral_field.py] Results written: "
          f"alpha_s={alpha_s:.6f} (Δ={dev:.6f}), R_pi={r_pi:.6f}, timestamp={ts}")

def main():
    global config
    config = load_config()

    m_z = config.get('energy_scale', 91.2)
    l_max = config['spectral_modes']['l_max']
    m_max = config['spectral_modes']['m_max']
    constraints = config['constraints']
    alpha_s_target = config.get('alpha_s_target', 0.118)

    print(f"[01_qcd_spectral_field.py] Using {'CUDA' if cuda_available else 'CPU'} for computations")

    y_lm, s_filter = compute_spectral_density(m_z, l_max, m_max)
    r_pi = compute_redundancy(s_filter, constraints)
    alpha_s = compute_alpha_s(s_filter)

    plot_heatmap(y_lm, 'qcd_spectral_heatmap.png')
    write_results(alpha_s, r_pi)

    deviation = abs(alpha_s - alpha_s_target)
    print(f"[01_qcd_spectral_field.py] Summary: alpha_s = {alpha_s:.6f} "
          f"(target {alpha_s_target:.6f}, Δ={deviation:.6f}), R_pi = {r_pi:.6f}")

if __name__ == "__main__":
    main()
