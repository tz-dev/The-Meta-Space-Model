# Script: 06_cy3_spectral_base.py
# Beschreibung: Berechnet SU(3)-Holonomien auf einem Calabi–Yau-Dreifachf ald (CY₃)
#   als Entropie-geprägte Spektralbasis für die Projektion π: 𝓜_meta → 𝓜₄.
# Postulate: 
#   • CP8 (Topological Protection)  
#   • EP2 (Phase-Locked Projection)  
#   • EP7 (Gluon Interaction Projection)
# Inputs: config_cy3.json
# Outputs: 
#   • Holonomie-Basis (complexes Feld)  
#   • Heatmap <img/cy3_holonomy_heatmap.png>  
#   • Eintrag in results.csv: holonomy_norm
# Logging: errors.log

import numpy as np
import json, glob, logging, os, csv
from datetime import datetime
import matplotlib.pyplot as plt

# --- Logging Setup ---
logging.basicConfig(
    filename='errors.log',
    level=logging.DEBUG,
    format='%(asctime)s [06_cy3_spectral_base.py] %(levelname)s: %(message)s'
)

def load_config():
    files = glob.glob('config_cy3*.json')
    if not files:
        logging.error("Keine config_cy3.json gefunden")
        raise FileNotFoundError("config_cy3.json fehlt")
    print("Available configuration files:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")
    idx = int(input("Select config file number: ")) - 1
    return json.load(open(files[idx], 'r'))

def compute_holonomy_basis(resolution, psi, phi):
    """
    Mock-Berechnung einer SU(3)-Holonomiebasis:
    basis(u,v) = sin(u+ψ)*cos(v+φ) + i·cos(u−φ), u,v ∈ [0,2π]
    Norm = ∑ |basis|^2
    """
    try:
        u = np.linspace(0, 2*np.pi, resolution)
        v = np.linspace(0, 2*np.pi, resolution)
        u, v = np.meshgrid(u, v, indexing='ij')
        basis = np.sin(u + psi) * np.cos(v + phi) + 1j * np.cos(u - phi)
        norm  = float(np.sum(np.abs(basis)**2))
        return basis, norm
    except Exception as e:
        logging.error(f"Holonomy-Berechnung fehlgeschlagen: {e}")
        raise

def save_heatmap(basis, filename='cy3_holonomy_heatmap.png'):
    """Speichert eine Heatmap der Holonomiebasis."""
    os.makedirs('img', exist_ok=True)
    plt.imshow(np.abs(basis), cmap='plasma', origin='lower')
    plt.colorbar(label='|Holonomy|')
    plt.title('SU(3)-Holonomy Basis auf CY₃')
    out = os.path.join('img', filename)
    plt.savefig(out)
    plt.close()
    print(f"[06_cy3_spectral_base.py] Heatmap saved: {out}")

def write_results(norm, metric, psi, phi):
    """Schreibt holonomy_norm in results.csv."""
    ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    with open('results.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '06_cy3_spectral_base.py',
            'holonomy_norm',
            norm,
            f"metric={metric}, psi={psi}, phi={phi}",
            'N/A',
            ts
        ])
    print("[06_cy3_spectral_base.py] Results written to results.csv")

def compute_cy3_modes():
    """
    Utility-Funktion für andere Skripte (z.B. 06a_higgs_spectral_field.py):
    Lädt die Konfiguration, berechnet die Holonomie-Basis und gibt sie zurück.
    """
    cfg = load_config()
    basis, _ = compute_holonomy_basis(
        cfg['resolution'],
        cfg['complex_structure_moduli']['psi'],
        cfg['complex_structure_moduli']['phi']
    )
    return basis

def main():
    try:
        cfg = load_config()
        metric    = cfg['cy3_metric']
        resolution= cfg['resolution']
        psi       = cfg['complex_structure_moduli']['psi']
        phi       = cfg['complex_structure_moduli']['phi']

        print(f"[06_cy3_spectral_base.py] Using metric={metric}, resolution={resolution}, psi={psi}, phi={phi}")
        basis, norm = compute_holonomy_basis(resolution, psi, phi)

        # Ausgabe
        save_heatmap(basis)
        write_results(norm, metric, psi, phi)

        # Modell-Konformitätscheck (CP8)
        if norm < 1e4:
            status = "too low (underconstrained)"
        elif norm > 1e6:
            status = "too high (overmodulated)"
        else:
            status = "model-conform (CP8)"
        print(f"[06_cy3_spectral_base.py] Holonomy norm: {norm:.6f} → {status}")
        print("[06_cy3_spectral_base.py] Computation complete.")
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
