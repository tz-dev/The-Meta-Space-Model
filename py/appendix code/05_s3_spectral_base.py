# Script: 05_s3_spectral_base.py
# Description: Computes spectral basis functions (Y_{lm}) on S^3
# Postulate: CP8 (Topological Protection)
# Inputs: config_s3.json
# Outputs: Spectral basis Y_{lm}, Norm, Heatmap, Eintrag in results.csv
# Visualization: img/s3_spectral_heatmap.png
# Logging: errors.log

import numpy as np
import matplotlib.pyplot as plt
import json, glob, logging, os, csv
from datetime import datetime
from scipy.special import sph_harm_y  # Nach SciPy 1.15.0

# Logging
logging.basicConfig(
    filename='errors.log',
    level=logging.DEBUG,
    format='%(asctime)s [05_s3_spectral_base.py] %(levelname)s: %(message)s'
)

def load_config():
    files = glob.glob('config_s3*.json')
    if not files:
        logging.error("Keine config_s3.json gefunden")
        raise FileNotFoundError("config_s3.json fehlt")
    print("Available configuration files:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")
    idx = int(input("Select config file number: ")) - 1
    return json.load(open(files[idx], 'r'))

def compute_spectral_basis(l_max, m_max, theta, phi):
    """Berechnet Y_lm auf S^3 durch Summe der Kugelflächenmoden."""
    Y = np.zeros(theta.shape, dtype=np.complex128)
    for l in range(l_max + 1):
        for m in range(-min(l, m_max), min(l, m_max) + 1):
            Y += sph_harm_y(m, l, phi, theta)
    norm = np.sum(np.abs(Y)**2)
    logging.debug(f"Spectral basis norm = {norm:.6f}")
    return Y, norm

def save_heatmap(Y, filename):
    os.makedirs('img', exist_ok=True)
    plt.imshow(np.abs(Y), cmap='viridis', origin='lower')
    plt.colorbar(label='|Y_{lm}|')
    plt.title('Spectral Basis on S^3')
    plt.savefig(f'img/{filename}')
    plt.close()
    print(f"[05_s3_spectral_base.py] Heatmap saved: img/{filename}")

def write_results(norm, l_max, m_max):
    with open('results.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        writer.writerow([
            '05_s3_spectral_base.py',
            'Y_lm_norm',
            norm,
            f'l_max={l_max}, m_max={m_max}',
            'N/A',
            ts
        ])
    print("[05_s3_spectral_base.py] Results written to results.csv")

def main():
    try:
        # 1) Config laden
        cfg = load_config()
        l_max = cfg['spectral_modes']['l_max']
        m_max = cfg['spectral_modes']['m_max']
        print(f"[05_s3_spectral_base.py] Using l_max={l_max}, m_max={m_max}")

        # 2) Gitter auf S^3 (θ, φ)
        res = cfg.get('resolution', 200)
        theta = np.linspace(0, np.pi, res)
        phi   = np.linspace(0, 2*np.pi, res)
        theta, phi = np.meshgrid(theta, phi)

        # 3) Basisberechnung
        Y, norm = compute_spectral_basis(l_max, m_max, theta, phi)

        # 4) Visualisierung
        save_heatmap(Y, 's3_spectral_heatmap.png')

        # 5) Ergebnisse protokollieren
        write_results(norm, l_max, m_max)

        # 6) CP8-Konformitäts-Check
        if norm < 1e4:
            status = "too low (underconstrained)"
        elif norm > 1e6:
            status = "too high (overmodulated)"
        else:
            status = "model-conform (CP8)"
        print(f"[05_s3_spectral_base.py] Spectral norm: {norm:.6f} → {status}")
        print("[05_s3_spectral_base.py] Computation complete.")
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        raise

if __name__ == '__main__':
    main()
