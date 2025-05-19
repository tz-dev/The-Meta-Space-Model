import numpy as np
import matplotlib.pyplot as plt

# Define the tau range for RG evolution
tau = np.linspace(0, 40, 1000)

# Initial gauge couplings at tau=0 (elektroschwache Skala, ~100 GeV)
alpha1_0 = 0.0169
alpha2_0 = 0.0338
alpha3_0 = 0.118

# Higgs VEV and Higgs self-coupling
v = 246  # GeV
lambda_h = 0.13  # Higgs self-coupling at elektroschwache Skala

def gauge_couplings(t):
    """
    RG running for gauge couplings using Standard Model beta functions.
    Returns alpha1, alpha2, alpha3 at scale t, with logarithmic energy scaling.
    """
    b1, b2, b3 = 41/10, -19/6, -7
    dt = np.log(1 + t * (10**16 / 100 - 1) / 40)
    alpha1 = alpha1_0 / (1 - b1 * alpha1_0 * dt / (16 * np.pi**2))
    alpha2 = alpha2_0 / (1 - b2 * alpha2_0 * dt / (16 * np.pi**2))
    alpha3 = alpha3_0 / (1 - b3 * alpha3_0 * dt / (16 * np.pi**2))
    return alpha1, alpha2, alpha3

def rg_step(y_t, alpha1, alpha2, alpha3, dt, entropy_damping=False, y_b=0):
    """
    Single RG step for top Yukawa coupling y_t with entropy damping and Zweischleifen corrections.
    """
    c = [17/12, 9/4, 8]
    g = [85/6, 45/2, 80]
    alpha = [alpha1, alpha2, alpha3]
    
    # Einschleifen
    beta_y = (y_t / (16 * np.pi**2)) * (4.5 * y_t**2 - sum(c_i * a_i for c_i, a_i in zip(c, alpha)))
    
    # Zweischleifen
    beta_y += (y_t / (16 * np.pi**2)**2) * (
        13.5 * y_t**4 + 6 * y_t**2 * y_b**2 + 0.125 * y_t**2 * lambda_h -
        sum(g_i * a_i**2 for g_i, a_i in zip(g, alpha)) - 34 * alpha3 * y_t**2
    )
    
    if entropy_damping:
        beta_y *= np.exp(-10.0 * y_t**2)
        
    return y_t + beta_y * dt

def run_rg_flow(yb0, yt0, tau_array, entropy_damping=False):
    """
    Run the RG flow for given initial bottom and top Yukawa couplings.
    """
    y_b = np.full_like(tau_array, yb0)
    y_t = np.full_like(tau_array, yt0)

    for i in range(1, len(tau_array)):
        dt = tau_array[i] - tau_array[i-1]
        a1, a2, a3 = gauge_couplings(tau_array[i-1])
        y_t[i] = rg_step(y_t[i-1], a1, a2, a3, dt, entropy_damping, y_b[i-1])
        y_b[i] = rg_step(y_b[i-1], a1, a2, a3, dt, entropy_damping, y_b[i-1])

        if not np.isfinite(y_t[i]) or y_t[i] > 20:
            return y_b[:i], y_t[:i], tau_array[:i], True
        if not np.isfinite(y_b[i]) or y_b[i] > 20:
            return y_b[:i], y_t[:i], tau_array[:i], True

    return y_b, y_t, tau_array, False

def find_max_stable_yt(yb0, yt_low, yt_high, tau_array, tol=1e-6, max_iter=50, entropy_damping=False):
    """
    Binary search to find the yt0 that produces realistic top quark mass.
    """
    target_mt = 172.76
    target_yt = target_mt * np.sqrt(2) / v

    for _ in range(max_iter):
        yt_mid = (yt_low + yt_high) / 2
        _, y_t, tau_run, runaway = run_rg_flow(yb0, yt_mid, tau_array, entropy_damping)
        mt = y_t[0] * v / np.sqrt(2)
        mt_40 = y_t[-1] * v / np.sqrt(2)

        if abs(mt - target_mt) < 0.01 and abs(mt_40 - 170) < 0.5 and not runaway:
            return yt_mid
        elif mt > target_mt or mt_40 > 170.5 or runaway:
            yt_high = yt_mid
        else:
            yt_low = yt_mid
        if (yt_high - yt_low) < tol:
            break
    return yt_low

if __name__ == "__main__":
    yb0_values = [0.02402, 0.02403, 0.02404]
    yt0_low = 0.99390
    yt0_high = 0.99400

    results = []
    for yb0 in yb0_values:
        max_yt0 = find_max_stable_yt(yb0, yt0_low, yt0_high, tau, entropy_damping=True)
        y_b, y_t, tau_run, runaway = run_rg_flow(yb0, max_yt0, tau, entropy_damping=True)
        
        mt = y_t[0] * v / np.sqrt(2)
        mb = y_b[0] * v / np.sqrt(2)
        results.append((yb0, max_yt0, tau_run[-1], mb, mt))

        print(f"y_b0={yb0:.4f} | max y_t0 stable={max_yt0:.4f} | integration up to τ={tau_run[-1]:.2f}")
        print(f"  Final masses at τ=0: m_b = {mb:.3f} GeV, m_t = {mt:.3f} GeV")
        mt_10 = y_t[int(len(tau_run)/4)] * v / np.sqrt(2)
        mt_40 = y_t[-1] * v / np.sqrt(2)
        print(f"  Scale dependence: m_t(τ=10) = {mt_10:.3f} GeV, m_t(τ=40) = {mt_40:.3f} GeV\n")

    plt.figure(figsize=(10, 5))
    plt.plot(tau_run, y_b, label=f"Bottom Yukawa $y_b(\\tau)$ (init={yb0:.4f})")
    plt.plot(tau_run, y_t, label=f"Top Yukawa $y_t(\\tau)$ (init={max_yt0:.4f})", linestyle='--')
    plt.xlabel("Entropic time $\\tau$")
    plt.ylabel("Yukawa coupling")
    plt.title("RG Flow with Optimized Damping and Zweischleifen Corrections")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rg_flow_updated.png')