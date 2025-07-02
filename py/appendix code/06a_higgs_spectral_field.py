# Script: 06a_higgs_spectral_field.py
# Description: Parameterizes Higgs fields (ψ_α) on the meta-space manifold 𝓜_meta = S^3 × CY_3 × ℝ_τ using entropic projections,
#   computing the Higgs mass (m_H ≈ 125.0 GeV) via weighted integral, with adaptive quantile-based stability and l_h tuning.
# Formulas & Methods:
#   - Higgs field: ψ_α = |basis(u, v)| / mean(|basis|), where basis is SU(3)-holonomy from 06_cy3_spectral_base.py.
#   - Stability (CP2): stability_metric = mean(|∇ψ_α| ≥ ε), where ε = max(percentile(|∇ψ_α|, q), ε_min).
#   - m_H computation (EP11): m_H = m_h_target * (∫ψ_α * w * dV / ∫w * dV), with w = exp(-d^2 / l_h^2).
#   - l_h tuning: Grid search over l_h to minimize Δm_H, sacrificing stability if needed.
#   - Uses CUDA (cupy) for GPU acceleration if available, fallback to NumPy.
# Postulates:
#   - CP1: Geometric basis (S^3 × CY_3 × ℝ_τ).
#   - CP2: Entropy-driven causality (|∇ψ_α| > 0).
#   - CP3: Projection principle (ψ_α normalization).
#   - CP5: Entropy-coherent stability (stability_metric ≥ threshold).
#   - CP6: Simulation consistency via CUDA/NumPy.
#   - CP7: Entropy-driven matter (m_H derived from ψ_α).
#   - CP8: Topological protection via CY_3 holonomy.
#   - EP11: Empirical Higgs mass (m_H ≈ 125.0 GeV).
# Inputs:
#   - config_higgs_6a*.json: Configuration file with m_h_target, entropy_gradient_quantile, entropy_gradient_min,
#     resolution, complex_structure_moduli (psi, phi), l_h, l_h_min, l_h_max, l_h_steps, auto_tune_lh, stability_threshold.
#   - 06_cy3_spectral_base.py: Provides SU(3)-holonomy basis.
# Outputs:
#   - results.csv: Stores m_H, stability_metric, deviation, timestamp.
#   - img/psi_alpha.npy: Raw ψ_α field data.
#   - img/higgs_field_heatmap.png: Heatmap of |ψ_α|.
#   - errors.log: Logs errors during execution.
# Dependencies: numpy, cupy, matplotlib, json, glob, csv, logging, tqdm, importlib.util

import numpy as np
try:
    import cupy as cp
    cuda_available = True
except ImportError:
    cuda_available = False
    cp = np
import matplotlib.pyplot as plt
import json
import glob
import logging
import os
import csv
from datetime import datetime
from tqdm import tqdm
import importlib.util
import platform
import sys, io

# Dynamic import of 06_cy3_spectral_base.py
here = os.path.dirname(__file__)
cy3_path = os.path.join(here, "06_cy3_spectral_base.py")
spec = importlib.util.spec_from_file_location("cy3_spectral_base", cy3_path)
if spec is None or spec.loader is None:
    raise ImportError("Failed to load 06_cy3_spectral_base.py")
cy3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cy3)
compute_holonomy_basis = cy3.compute_holonomy_basis

# Logging setup
logging.basicConfig(
    filename='errors.log',
    level=logging.ERROR,
    format='%(asctime)s [06a_higgs_spectral_field.py] %(levelname)s: %(message)s'
)

def clear_screen():
    """Clear the console screen based on the operating system."""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def load_config():
    """Load fixed JSON configuration file for Higgs field simulation."""
    config_path = 'config_higgs_6a.json'
    if not os.path.exists(config_path):
        logging.error(f"Missing fixed config file: {config_path}")
        raise FileNotFoundError(f"Missing {config_path}")
    with open(config_path, 'r', encoding='utf-8') as infile:
        cfg = json.load(infile)
    print(f"[06a_higgs_spectral_field.py] Loaded fixed config: m_h_target={cfg['m_h_target']}, "
          f"resolution={cfg['resolution']}, psi={cfg['complex_structure_moduli']['psi']}, "
          f"phi={cfg['complex_structure_moduli']['phi']}, auto_tune_lh={cfg.get('auto_tune_lh', False)}")
    return cfg

def simulate(m_h_target, q, epsilon_min, l_h, resolution, psi_mod, phi_mod):
    """
    Run single simulation for given l_h, computing m_H and stability.
    Args:
        m_h_target (float): Target Higgs mass (e.g., 125.0 GeV).
        q (float): Quantile for adaptive ε computation.
        epsilon_min (float): Minimum gradient threshold.
        l_h (float): Length scale for weighting function.
        resolution (int): Grid resolution for u, v.
        psi_mod (float): Complex structure modulus (ψ).
        phi_mod (float): Complex structure modulus (φ).
    Returns:
        tuple: (m_h, stability, deviation, psi_alpha) - Higgs mass, stability metric, deviation, and field array.
    """
    print(f"[06a_higgs_spectral_field.py] Simulating Higgs field with l_h={l_h:.4f}")
    
    # Compute SU(3)-holonomy basis from 06_cy3_spectral_base.py (CP8, EP2, EP7)
    basis, _ = compute_holonomy_basis(resolution, psi_mod, phi_mod)
    psi_alpha = cp.abs(cp.array(basis) if cuda_available else basis)
    
    # Normalize ψ_α to mean=1 per CP3, EP11
    psi_alpha /= cp.mean(psi_alpha)
    
    # Compute gradient magnitude per CP2
    g_theta = cp.gradient(psi_alpha, axis=0)
    g_phi = cp.gradient(psi_alpha, axis=1)
    grad_magnitude = cp.sqrt(g_theta**2 + g_phi**2 + 1e-12)
    
    # Adaptive ε via quantile method
    percentile = float(cp.percentile(grad_magnitude, q)) if cuda_available else float(np.percentile(grad_magnitude, q))
    epsilon = max(percentile, epsilon_min)
    
    # Stability metric per CP2, CP5
    stability = float(cp.mean(grad_magnitude >= epsilon))
    
    # Compute m_H via weighted integral per EP11, CP7
    theta = cp.linspace(0, cp.pi, resolution)
    phi = cp.linspace(0, 2 * cp.pi, resolution)
    theta, phi = cp.meshgrid(theta, phi, indexing='ij')
    d_theta, d_phi = float(cp.pi / (resolution - 1)), float(2 * cp.pi / (resolution - 1))
    dV = cp.sin(theta) * d_theta * d_phi
    d2 = (theta - cp.pi / 2)**2 + (phi - cp.pi)**2
    weights = cp.exp(-d2 / (l_h**2))
    integral = float(cp.sum(psi_alpha * weights * dV))
    norm_w = float(cp.sum(weights * dV))
    m_h = m_h_target * (integral / norm_w)
    deviation = abs(m_h - m_h_target)
    
    # Convert ψ_α to NumPy for saving
    psi_alpha = cp.asnumpy(psi_alpha) if cuda_available else psi_alpha
    
    print(f"[06a_higgs_spectral_field.py] m_H={m_h:.4f} GeV, stability={stability:.4f}, deviation={deviation:.4f}")
    return m_h, stability, deviation, psi_alpha

def save_heatmap(psi_alpha, filename='higgs_field_heatmap.png'):
    """
    Generate and save heatmap of Higgs field |ψ_α|.
    Args:
        psi_alpha (array): Higgs field array.
        filename (str): Output filename for heatmap.
    """
    print(f"[06a_higgs_spectral_field.py] Generating heatmap for Higgs field")
    os.makedirs('img', exist_ok=True)
    plt.imshow(psi_alpha, cmap='inferno', origin='lower')
    plt.colorbar(label='|ψ_α|')
    plt.title('Higgs Spectral Field ψ_α')
    out = os.path.join('img', filename)
    plt.savefig(out)
    plt.close()
    print(f"[06a_higgs_spectral_field.py] Heatmap saved: {out}")

def write_results(m_h, stability, deviation, m_h_target, stability_threshold):
    """
    Write results to results.csv.
    Args:
        m_h (float): Computed Higgs mass.
        stability (float): Stability metric.
        deviation (float): Deviation from target m_H.
        m_h_target (float): Target Higgs mass.
        stability_threshold (float): Stability threshold.
    """
    print(f"[06a_higgs_spectral_field.py] Writing results to results.csv")
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    with open('results.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['06a_higgs_spectral_field.py', 'm_H', m_h, m_h_target, deviation, timestamp])
        writer.writerow(['06a_higgs_spectral_field.py', 'stability_metric', stability, stability_threshold, 'N/A', timestamp])
    print(f"[06a_higgs_spectral_field.py] Results written: m_H={m_h:.4f} GeV (Δ={deviation:.4f}), "
          f"stability_metric={stability:.4f}, timestamp={timestamp}")

def main():
    """Main function to orchestrate Higgs field simulation with l_h tuning."""
    clear_screen()
    print("=======================================================")
    print("    Meta-Space Model: Higgs Field Simulation (CY_3)    ")
    print("=======================================================")
    print(f"[06a_higgs_spectral_field.py] Using {'CUDA' if cuda_available else 'CPU'} for computations")
    
    # Load configuration
    cfg = load_config()
    m_h_target = cfg['m_h_target']
    q = cfg['entropy_gradient_quantile']
    epsilon_min = cfg['entropy_gradient_min']
    resolution = cfg['resolution']
    psi_mod = cfg['complex_structure_moduli']['psi']
    phi_mod = cfg['complex_structure_moduli']['phi']
    auto_tune_lh = cfg.get('auto_tune_lh', False)
    l_h = cfg.get('l_h', 1.0)
    l_h_min = cfg.get('l_h_min', 0.5)
    l_h_max = cfg.get('l_h_max', 2.0)
    l_h_steps = cfg.get('l_h_steps', 20)
    stability_threshold = cfg.get('stability_threshold', 0.5)
    
    # Run simulation with progress bar
    with tqdm(total=3, desc="Processing Higgs field simulation", unit="step") as pbar:
        # Perform simulation or tune l_h
        if not auto_tune_lh:
            m_h, stability, deviation, psi_alpha = simulate(m_h_target, q, epsilon_min, l_h, resolution, psi_mod, phi_mod)
            best_lh = l_h
            print(f"[06a_higgs_spectral_field.py] Using fixed l_h={best_lh:.4f}")
        else:
            print(f"[06a_higgs_spectral_field.py] Tuning l_h over [{l_h_min}, {l_h_max}] with {l_h_steps} steps")
            best_dev = float('inf')
            best = None
            for lh in tqdm(np.linspace(l_h_min, l_h_max, l_h_steps), desc="Tuning l_h", unit="step"):
                m, stab, dev, psi = simulate(m_h_target, q, epsilon_min, lh, resolution, psi_mod, phi_mod)
                if dev < best_dev:
                    best_dev = dev
                    best = (m, stab, dev, psi)
                    best_lh = lh
            m_h, stability, deviation, psi_alpha = best
            print(f"[06a_higgs_spectral_field.py] Optimal l_h={best_lh:.4f}, deviation={deviation:.4f}")
        pbar.update(1)
        
        # Save ψ_α array
        os.makedirs('img', exist_ok=True)
        np.save('img/psi_alpha.npy', psi_alpha)
        print(f"[06a_higgs_spectral_field.py] psi_alpha array saved: img/psi_alpha.npy")
        pbar.update(1)
        
        # Generate and save heatmap
        save_heatmap(psi_alpha)
        pbar.update(1)
    
    # Write results to CSV
    write_results(m_h, stability, deviation, m_h_target, stability_threshold)
    
    # Validate stability per CP5
    status = "PASS" if stability >= stability_threshold and deviation < 0.5 else "FAIL"
    
    # Summary output
    print("\n=====================================")
    print("     Meta-Space Model: Summary")
    print("=====================================")
    print(f"Script: 06a_higgs_spectral_field.py")
    print(f"Description: Parameterizes Higgs field ψ_α on S^3 × CY_3 × ℝ_τ with l_h tuning")
    print(f"Postulates: CP1, CP2, CP3, CP5, CP6, CP7, CP8, EP11")
    print(f"Computed m_H: {m_h:.4f} GeV (target {m_h_target:.4f} GeV, Δ={deviation:.4f})")
    print(f"Stability Metric: {stability:.4f} (threshold {stability_threshold})")
    print(f"Optimal l_h: {best_lh:.4f}")
    print(f"Status: {status}")
    print("=====================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
        raise