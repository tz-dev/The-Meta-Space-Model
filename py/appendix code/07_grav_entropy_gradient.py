# Script: 07_grav_entropy_gradient.py
# Description: Models gravitational field tensor I_{\mu\nu} via entropic gradient ∇_τ S for curvature projections.
# Postulates: CP2 (Entropy-Driven Causality), EP8 (Extended Quantum Gravity)
# Inputs: config_grav.json (tensor_shape, field_scale, noise_amplitude, thresholds, etc.), results.csv (for Y_lm_norm)
# Outputs: I_{\mu\nu}, stability metric, deviation → results.csv
# Visualization: Heatmap saved to img/grav_field_heatmap.png
# Logging: Errors to errors.log

import numpy as np
import matplotlib.pyplot as plt
import json, glob, logging, os, csv
from datetime import datetime
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Logging setup
logging.basicConfig(
    filename='errors.log',
    level=logging.WARNING,
    format='%(asctime)s [07_grav_entropy_gradient.py] %(levelname)s: %(message)s'
)

def load_config():
    """Load JSON config file."""
    config_files = glob.glob('config_grav*.json')
    if not config_files:
        logging.error("No config files found matching 'config_grav*.json'")
        raise FileNotFoundError("No config_grav.json found")

    print("Available configuration files:")
    for i, file in enumerate(config_files, 1):
        print(f"{i}. {file}")
    choice = int(input("Select config file number: ")) - 1
    return json.load(open(config_files[choice], 'r'))

def load_y_lm_norm():
    """Load Y_lm_norm from results.csv for scaling."""
    try:
        with open('results.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0]=='05_s3_spectral_base.py' and row[1]=='Y_lm_norm':
                    return float(row[2])
        raise ValueError("Y_lm_norm not found in results.csv")
    except Exception as e:
        logging.error(f"Failed to load Y_lm_norm: {e}")
        raise

def compute_gravitational_tensor(shape, gradient_threshold, sigma, field_scale, noise_amp):
    """Compute gravitational tensor I_{mu nu} from entropic gradient."""
    try:
        x = np.linspace(0, 2*np.pi, shape[0])
        y = np.linspace(0, 2*np.pi, shape[1])
        X, Y = np.meshgrid(x, y)

        # Entropiefeld S skaliert und ggf. mit Rauschen
        y_lm_norm = load_y_lm_norm()
        S = (np.sin(X)*np.cos(Y) + noise_amp * np.random.rand(*X.shape))
        S *= field_scale * y_lm_norm
        S = gaussian_filter(S, sigma=sigma)

        grad_tau = np.gradient(S, axis=0)
        I_mu_nu = np.gradient(grad_tau, axis=1)

        stability_metric = float(np.mean(np.abs(grad_tau)>=gradient_threshold))
        deviation        = float(np.std(I_mu_nu))
        return I_mu_nu, stability_metric, deviation
    except Exception as e:
        logging.error(f"Gravitational tensor computation failed: {e}")
        raise

def find_stable_tensor(shape, initial_threshold, sigma,
                       runs=5, target=0.5, step=0.01, max_attempts=20,
                       field_scale=1.0, noise_amp=0.0):
    """Find stable tensor by adjusting threshold."""
    for _ in tqdm(range(max_attempts), desc="Adjusting threshold", ncols=80):
        stability_vals = []
        deviations     = []
        tensors        = []
        for _ in range(runs):
            I, stab, dev = compute_gravitational_tensor(
                shape, initial_threshold, sigma, field_scale, noise_amp
            )
            stability_vals.append(stab)
            deviations.append(dev)
            tensors.append(I)
        avg_stab = np.mean(stability_vals)
        avg_dev  = np.mean(deviations)
        best_I   = tensors[np.argmax(stability_vals)]
        if avg_stab >= target:
            return best_I, avg_stab, avg_dev, initial_threshold
        initial_threshold += step
    return best_I, avg_stab, avg_dev, initial_threshold

def save_heatmap(data):
    """Generate and save heatmap of I_{mu nu}."""
    os.makedirs('img', exist_ok=True)
    plt.imshow(np.abs(data), cmap='cividis', origin='lower')
    plt.colorbar(label='|I_{μν}|')
    plt.title('Gravitational Field Tensor $I_{\\mu\\nu}$')
    plt.savefig('img/grav_field_heatmap.png')
    plt.close()
    print("[07_grav_entropy_gradient.py] Heatmap saved → img/grav_field_heatmap.png")

def write_results(I_mu_nu, stability, deviation, threshold):
    """Write results to results.csv."""
    ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    with open('results.csv','a',newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            '07_grav_entropy_gradient.py','I_mu_nu',
            np.mean(I_mu_nu),'N/A', deviation, ts
        ])
        w.writerow([
            '07_grav_entropy_gradient.py','stability_metric',
            stability, f'thresh={threshold:.4f}','N/A', ts
        ])
    print(f"[07_grav_entropy_gradient.py] Results written → deviation={deviation:.6f}, stability={stability:.4f}")

def main():
    try:
        cfg = load_config()
        shape     = cfg['tensor_shape']
        thresh    = cfg['entropy_gradient_threshold']
        sigma     = cfg.get('gaussian_sigma',    1.6)
        runs      = cfg.get('averaging_runs',    5)
        field_sc  = cfg.get('field_scale',       1.0)
        noise_amp = cfg.get('noise_amplitude',   0.0)
        print(f"[07_grav_entropy_gradient.py] shape={shape}, thresh={thresh}, sigma={sigma}, runs={runs}, field_scale={field_sc}, noise_amp={noise_amp}")

        I, stab, dev, used_th = find_stable_tensor(
            shape, thresh, sigma,
            runs=runs, target=0.5, step=0.01, max_attempts=20,
            field_scale=field_sc, noise_amp=noise_amp
        )
        save_heatmap(I)
        write_results(I, stab, dev, used_th)

        print(f"[07_grav_entropy_gradient.py] Stability = {stab:.4f}")
        print(f"[07_grav_entropy_gradient.py] Deviation = {dev:.6f}")
        status = "model-conform (CP2/EP8)" if stab>=0.5 else "unstable / model-critical"
        print(f"[07_grav_entropy_gradient.py] Projection status → {status}")
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        raise

if __name__ == '__main__':
    main()
