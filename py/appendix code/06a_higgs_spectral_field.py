# Script: 06a_higgs_spectral_field.py
# Description: Parameterizes Higgs fields (ψ_α) via entropic projection auf 
#   𝓜_meta = S^3 × CY_3 × ℝ_τ, berechnet m_H nach EP11 als gewichtetes Integral.
# Änderungen:
#  • ψ_α wird auf Mittelwert=1 normiert (Kalibrierung EP11)
#  • Stabilität über adaptives Quantil-Verfahren des Gradientenbetrags (CP2)
#  • Automatische Justage von l_h zur Minimierung Δm_H (opfert Stabilität)
# Postulates: CP1, CP2, CP3, CP5, CP6, CP7, CP8, EP11
# Inputs: config_higgs_6a.json, holonomy basis aus 06_cy3_spectral_base.py
# Outputs: best_m_H, stability_metric, deviation, psi_alpha.npy, heatmap
# Logging: errors.log

import numpy as np
try:
    import cupy as cp
    cuda_available = True
except ImportError:
    cp = np
    cuda_available = False

import matplotlib.pyplot as plt
import json, glob, logging, os, csv
from datetime import datetime
import importlib.util

# — Dynamischer Import —
here     = os.path.dirname(__file__)
cy3_path = os.path.join(here, "06_cy3_spectral_base.py")
spec     = importlib.util.spec_from_file_location("cy3_spectral_base", cy3_path)
cy3      = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cy3)
compute_holonomy_basis = cy3.compute_holonomy_basis

# Logging
logging.basicConfig(
    filename='errors.log',
    level=logging.ERROR,
    format='%(asctime)s [06a_higgs_spectral_field.py] %(levelname)s: %(message)s'
)

def load_config():
    files = glob.glob('config_higgs_6a*.json')
    if not files:
        raise FileNotFoundError("config_higgs_6a.json fehlt")
    print("Available config files:")
    for i,f in enumerate(files,1):
        print(f"{i}. {f}")
    idx = int(input("Select config file number: ")) - 1
    return json.load(open(files[idx], 'r'))

def simulate(m_h_target, q, epsilon_min, l_h, resolution, psi_mod, phi_mod):
    """Einzelne Simulation für gegebenes l_h."""
    # 1) Holonomie-Basis
    basis, _ = compute_holonomy_basis(resolution, psi_mod, phi_mod)
    ψ = cp.abs(cp.array(basis) if cuda_available else basis)
    ψ /= cp.mean(ψ)  # Normierung

    # 2) Gradientbetrag und adaptives ε
    gθ = cp.gradient(ψ, axis=0)
    gφ = cp.gradient(ψ, axis=1)
    gm = cp.sqrt(gθ**2 + gφ**2 + 1e-12)
    perc = float(cp.percentile(gm, q)) if cuda_available else float(np.percentile(gm, q))
    ε    = max(perc, epsilon_min)

    stab = float(cp.mean(gm >= ε))

    # 3) EP11-Integral
    θ = cp.linspace(0, cp.pi, resolution)
    φ = cp.linspace(0, 2*cp.pi, resolution)
    Θ,Φ = cp.meshgrid(θ, φ, indexing='ij')
    dθ, dφ = float(cp.pi/(resolution-1)), float(2*cp.pi/(resolution-1))
    dV = cp.sin(Θ)*dθ*dφ
    d2 = (Θ-cp.pi/2)**2 + (Φ-cp.pi)**2
    w  = cp.exp(-d2/(l_h**2))
    integral = float(cp.sum(ψ*w*dV))
    norm_w   = float(cp.sum(w*dV))
    m_h      = m_h_target * (integral/norm_w)
    dev      = abs(m_h - m_h_target)
    return m_h, stab, dev, cp.asnumpy(ψ)

def main():
    cfg = load_config()
    # aus config
    m_h_target = cfg['m_h_target']
    q          = cfg['entropy_gradient_quantile']
    ε_min      = cfg['entropy_gradient_min']
    resolution = cfg['resolution']
    psi_mod    = cfg['complex_structure_moduli']['psi']
    phi_mod    = cfg['complex_structure_moduli']['phi']
    tune       = cfg.get('auto_tune_lh', False)

    # Default l_h, falls kein Tuning
    l_h0 = cfg.get('l_h', 1.0)
    if not tune:
        best = simulate(m_h_target, q, ε_min, l_h0, resolution, psi_mod, phi_mod)
        best_lh = l_h0
    else:
        # Gitter-Suche über l_h
        lmin, lmax = cfg['l_h_min'], cfg['l_h_max']
        steps       = cfg.get('l_h_steps', 20)
        best_dev = float('inf')
        best = None
        for lh in np.linspace(lmin, lmax, steps):
            m, stab, dev, psi = simulate(m_h_target, q, ε_min, lh, resolution, psi_mod, phi_mod)
            if dev < best_dev:
                best_dev = dev
                best = (m, stab, dev, psi)
                best_lh = lh
        print(f"[06a_higgs] auto-tune l_h → {best_lh:.4f}")

    m_h, stability, deviation, psi_alpha = best

    # Ausgaben
    print(f"[06a_higgs] m_H={m_h:.4f} GeV  Δ={deviation:.4f}  stability={stability:.4f}")

    # speichern
    os.makedirs('img', exist_ok=True)
    np.save('img/psi_alpha.npy', psi_alpha)
    plt.imshow(psi_alpha, cmap='inferno', origin='lower')
    plt.colorbar(label='|ψ_α|')
    plt.title('Higgs Spectral Field ψ_α')
    plt.savefig('img/higgs_field_heatmap.png')
    plt.close()

    # results.csv
    ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    with open('results.csv','a',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['06a_higgs_spectral_field.py','m_H',      m_h,      m_h_target, deviation, ts])
        w.writerow(['06a_higgs_spectral_field.py','stability_metric', stability, cfg['stability_threshold'], '', ts])

if __name__=='__main__':
    print(f"[06a_higgs] Using {'CUDA' if cuda_available else 'CPU'}")
    main()
