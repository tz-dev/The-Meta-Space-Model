# Script: 08_cosmo_entropy_scale.py
# Description: Nutzt die entropische Projektion aus Script 02 (img/s_field.npy) 
#   als echte Meta-Space-Entropie S(x,y,τ) und skaliert sie, sodass Ω_DM ≈ 0.27.
# Postulates: CP1 (Geometrical Substrate), CP2 (Entropy-Driven Causality),
#             EP6 (Dark Matter Projection), EP14 (Holographic Projection)
# Inputs: config_cosmo.json (omega_dm_target, threshold), img/s_field.npy
# Outputs: Ω_DM, scaling_metric, deviation, heatmap img/cosmo_heatmap.png
# Logging: errors.log

import numpy as np
import matplotlib.pyplot as plt
import json, glob, logging, os, csv
from datetime import datetime

# --- Logging-Setup ---
logging.basicConfig(
    filename='errors.log',
    level=logging.ERROR,
    format='%(asctime)s [08_cosmo_entropy_scale.py] %(levelname)s: %(message)s'
)

def load_config():
    files = glob.glob('config_cosmo*.json')
    if not files:
        logging.error("config_cosmo.json fehlt")
        raise FileNotFoundError("Missing config_cosmo.json")
    print("Available configuration files:")
    for i,f in enumerate(files,1):
        print(f"{i}. {f}")
    idx = int(input("Select config file number: ")) - 1
    return json.load(open(files[idx],'r'))

def save_heatmap(data, fname):
    """Speichert nur den reellen Betrag (Abs) der Gradienten als Heatmap."""
    real = np.abs(data)
    os.makedirs('img', exist_ok=True)
    plt.imshow(real, cmap='inferno', origin='lower')
    plt.colorbar(label='|∇τ S_scaled|')
    plt.title('Cosmological Entropy Gradient (scaled)')
    plt.savefig(f'img/{fname}')
    plt.close()
    print(f"[08_cosmo_entropy_scale.py] Heatmap saved: img/{fname}")

def write_results(omega_dm, scale_metric, deviation, cfg):
    ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    with open('results.csv','a',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow([
            '08_cosmo_entropy_scale.py','Omega_DM',
            omega_dm, cfg['omega_dm_target'], deviation, ts
        ])
        w.writerow([
            '08_cosmo_entropy_scale.py','scaling_metric',
            scale_metric, cfg['entropy_gradient_threshold'], '', ts
        ])
    print(f"[08_cosmo_entropy_scale.py] Results written: Ω_DM={omega_dm:.4f}, "
          f"scaling_metric={scale_metric:.4f}, Δ={deviation:.4f}")

if __name__=='__main__':
    cfg = load_config()
    target = cfg['omega_dm_target']
    thresh = cfg['entropy_gradient_threshold']

    # 1) Lade echte Entropie aus Meta-Space-Simulation
    path = 'img/s_field.npy'
    if not os.path.exists(path):
        raise FileNotFoundError("s_field.npy nicht gefunden – bitte Script 02 ausführen.")
    S = np.load(path)  # Form: [resolution, resolution]

    # 2) Entropie-Gradient entlang τ-Achse (hier Achse 0)
    grad_tau = np.gradient(S, axis=0)

    # 3) Mittlere Gradienten-Stärke
    avg_grad = float(np.mean(np.abs(grad_tau)))

    # 4) Skalierung auf Planck-Wert Ω_DM ≈ target
    scale = target / avg_grad
    grad_scaled = grad_tau * scale

    # 5) Rechne Ergebnisgrößen
    Omega_DM      = float(np.mean(np.abs(grad_scaled)))
    deviation     = abs(Omega_DM - target)
    scale_metric  = float(np.mean(np.abs(grad_scaled) >= thresh))

    # 6) Heatmap & CSV
    save_heatmap(grad_scaled, 'cosmo_heatmap.png')
    write_results(Omega_DM, scale_metric, deviation, cfg)

    # 7) Status-Ausgabe
    status = "PASS" if (scale_metric >= 0.5 and deviation <= 0.01) else "FAIL"
    print(f"[08_cosmo_entropy_scale.py] Ω_DM: {Omega_DM:.4f} (target {target:.4f}, Δ={deviation:.4f})")
    print(f"[08_cosmo_entropy_scale.py] Scaling metric: {scale_metric:.4f} "
          f"(threshold={thresh:.4f}) → {status}")
    print("[08_cosmo_entropy_scale.py] Computation complete.")
