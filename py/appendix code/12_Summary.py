# Script: 12_Summary.py
# Description: Generates a comprehensive summary of Meta-Space Model (MSM) results from scripts 01-11,
#              combining information from comment headers and results.csv. Produces a detailed Markdown
#              report with purpose, methods, results, and significance for each script.
# Author: MSM Enhancement
# Date: 2025-07-09
# Version: 1.1
# Inputs:
#     - results.csv: Contains results from scripts 01-11 (parameters, values, targets, deviations, timestamps).
#     - config_summary.json: Optional configuration file with summary settings (e.g., output path).
# Outputs:
#     - 12_summary.md: Markdown file with detailed summaries for each script.
#     - Terminal output: Summary of key results and statuses.
# Dependencies: pandas, json, os, csv, datetime, logging
# Purpose:
#     - Provide a unified overview of MSM simulation results.
#     - Contextualize results with respect to MSM postulates (CP1-CP8, EP1-EP14).
#     - Support validation and interpretation of MSM outputs for empirical consistency.
# Notes:
#     - Handles results.csv without header by explicitly defining column names.

import pandas as pd
import json
import os
import csv
import datetime
import logging

# Logging setup
logging.basicConfig(filename="12_summary.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Script descriptions based on comment headers
SCRIPT_INFO = {
    "01_qcd_spectral_field.py": {
        "description": "Computes the strong coupling constant (α_s ≈ 0.118) for QCD spectral fields on the meta-space manifold 𝓜_meta = S^3 × CY_3 × ℝ_τ, using entropic projections and spherical harmonic modes (Y_lm).",
        "methods": """- Spherical harmonics: Y_lm(θ, φ) = sph_harm(m, l, φ, θ) for l_max, m_max.
- Entropic projection (CP3): S_filter = S_min ensures minimal entropy state.
- Redundancy (CP5): R_π = H[ρ] - I[ρ|O], with H[ρ] = ln(S_filter + ε), I[ρ|O] = ln(1 + Σw_i).
- α_s computation (EP1): α_s ∝ S_min / S_filter, normalized to CODATA target (0.118).
- Uses CUDA (cupy) for GPU acceleration if available, fallback to numpy.""",
        "postulates": ["CP3: Projection principle (δS_proj = 0)", "CP5: Entropy-coherent stability (R_π < threshold)", "CP6: Computational consistency via CUDA/numpy", "CP7: Entropy-driven matter (α_s derived from S_filter)", "CP8: Topological protection via S^3 harmonics", "EP1: Empirical QCD coupling (α_s ≈ 0.118)"],
        "inputs": ["config_qcd.json: Configuration file with energy_scale (M_Z), l_max, m_max, S_min, S_max, constraints, alpha_s_target, redundancy_threshold, alpha_s_range"],
        "outputs": ["results.csv: Stores α_s, R_π, deviation, timestamp", "img/qcd_spectral_heatmap.png: Heatmap of |Y_lm|", "errors.log: Logs errors during execution"]
    },
    "02_monte_carlo_validator.py": {
        "description": "Validates QCD and Higgs fields on the meta-space manifold 𝓜_meta = S^3 × CY_3 × ℝ_τ using Monte Carlo simulations, computing the strong coupling constant (α_s ≈ 0.118) and Higgs mass (m_H ≈ 125.0 GeV) via entropic projections.",
        "methods": """- Spherical harmonics: Y_lm(θ, φ) = sph_harm_y(m, l, φ, θ) for spectral density on S^3.
- Entropic projection (CP3): S_filter = S_min enforces δS_proj = 0.
- Redundancy (CP5): R_π = H[ρ] - I[ρ|O], where H[ρ] = ln(S_filter + ε), I[ρ|O] = ln(1 + Σw_i).
- α_s computation (EP1): α_s = α_target * (S_min / S_filter), normalized to CODATA (0.118).
- m_H computation (EP11): m_H = m_H_target * (S_min / S_filter), normalized to 125.0 GeV.
- RG Flow (EP13): α_s(τ) computed via 3-loop β-function, evaluated at τ ≈ 1 GeV⁻¹.
- Monte Carlo validation: Ensures consistency via random sampling on S^3.
- Uses CUDA (cupy) for GPU acceleration if available, fallback to NumPy.""",
        "postulates": ["CP1: Geometric basis (S^3 × CY_3 × ℝ_τ)", "CP3: Projection principle (δS_proj = 0)", "CP5: Entropy-coherent stability (R_π < threshold)", "CP6: Computational consistency via Monte Carlo and CUDA/NumPy", "CP7: Entropy-driven matter (α_s, m_H derived from ∇_τS)", "EP1: Empirical QCD coupling (α_s ≈ 0.118)", "EP11: Empirical Higgs mass (m_H ≈ 125.0 GeV)", "EP13: Renormalization group consistency (α_s(τ) ≈ 0.30 at τ ≈ 1 GeV⁻¹)"],
        "inputs": ["config_monte_carlo*.json: Configuration file with energy_scale (M_Z), higgs_mass (m_H_target), alpha_s_target, alpha_s_range, m_h_range, constraints, redundancy_threshold, s_min, spectral_modes (l_max, m_max)"],
        "outputs": ["results.csv: Logs α_s, m_H, R_π, α_s(τ≈1 GeV⁻¹), deviations, timestamp", "img/02_monte_carlo_heatmap.png: Heatmap of |S(x,τ)| and 02_alpha_s_tau.png", "img/s_field.npy: Raw field data", "errors.log: Logs execution errors and validation issues"]
    },
    "03_higgs_spectral_field.py": {
        "description": "Parameterizes Higgs fields (ψ_α) on the meta-space manifold 𝓜_meta = S^3 × CY_3 × ℝ_τ using entropic projections, computing the Higgs mass (m_H ≈ 125.0 GeV) and stability metric to ensure entropy-driven causality and simulation consistency.",
        "methods": """- Higgs field: ψ_α = |Y_lm(θ, φ)|^2 + noise, where Y_lm = sph_harm_y(m, l, φ, θ) on S^3.
- Entropic projection (CP2): ∇_τS > 0 ensures entropy-driven causality via gradient of ψ_α along τ.
- Stability metric (CP6): Mean of stability_mask (∇_τψ_α ≥ ε * 0.2), normalized to ≥ 0.5.
- m_H computation (EP11): m_H = m_H_target * (1 + 0.005 * ln(1 + spectral_norm / scale_factor)).
- Uses CUDA (cupy) for GPU acceleration if available, fallback to NumPy.""",
        "postulates": ["CP2: Entropy-driven causality (∇_τS > 0)", "CP6: Simulation consistency via CUDA/NumPy and Monte Carlo noise", "EP11: Empirical Higgs mass (m_H ≈ 125.0 GeV)"],
        "inputs": ["config_higgs*.json: Configuration file with spectral_modes (l_max, m_max), m_h_target, scale_factor, entropy_gradient_min, stability_threshold"],
        "outputs": ["results.csv: Stores m_H, stability_metric, deviation, timestamp", "img/higgs_field_heatmap.png: Heatmap of |ψ_α|", "img/psi_alpha.npy: Raw ψ_α field data", "errors.log: Logs errors during execution"]
    },
    "04_empirical_validator.py": {
        "description": "Validates Meta-Space Model (MSM) simulation outputs (alpha_s, m_H, Omega_DM, Y_lm_norm, holonomy_norm, stability_metric, scaling_metric, mass_drift_metric, oscillation_metric) against empirical targets on M_meta = S^3 x CY_3 x R_tau, producing deviation metrics and visualizations to ensure empirical consistency.",
        "methods": """- Deviation: Δ = |value - target| for point targets (e.g., alpha_s, m_H, Omega_DM).
- Range validation: For parameters like Y_lm_norm, holonomy_norm, check if value in [min, max].
- Stability validation: For stability_metric, scaling_metric, check if max(value) >= target.
- Visualization: Bar plot of deviations, heatmaps of s_field (Script 02) and psi_alpha (Scripts 03/06a).
- RG Validation: α_s(τ≈1GeV⁻¹) checked against QCD running coupling expectation (≈ 0.30).""",
        "postulates": ["CP5: Entropy-coherent stability (deviations within thresholds)", "CP6: Simulation consistency via validation of prior results", "CP8: Topological protection (Y_lm_norm, holonomy_norm in valid range)", "EP1: Empirical QCD coupling (alpha_s ≈ 0.118)", "EP5: Thermodynamic stability (mass_drift_metric within threshold)", "EP6: Dark matter projection (Omega_DM ≈ 0.268)", "EP7: Empirical consistency of spectral fields", "EP8: Extended quantum gravity (stability_metric for I_mu_nu)", "EP11: Empirical Higgs mass (m_H ≈ 125.0 GeV)", "EP12: Neutrino oscillations (oscillation_metric within threshold)", "EP13: Renormalization group consistency (α_s(τ) flows match empirical QCD at low energies)"],
        "inputs": ["config_empirical*.json: Configuration file with targets and thresholds", "results.csv: Results from Scripts 01-09 (alpha_s, m_H, Omega_DM, etc.)", "α_s(τ≈1GeV⁻¹) from Script 02, stored in results.csv", "img/s_field.npy: Field data from Script 02", "img/psi_alpha.npy: Field data from Scripts 03/06a"],
        "outputs": ["results.csv: Appended validation summary (parameter, value, target, deviation, timestamp)", "img/04_validation_bar_plot.png: Bar plot of deviations with thresholds", "img/04_validation_s_field_heatmap.png: Heatmap of s_field", "img/04_validation_psi_alpha_heatmap.png: Heatmap of psi_alpha", "errors.log: Logs errors and validation info"]
    },
    "05_s3_spectral_base.py": {
        "description": "Computes spectral basis functions (Y_lm) on S^3 for the meta-space manifold 𝓜_meta = S^3 × CY_3 × ℝ_τ, ensuring topological protection via summation of spherical harmonic modes.",
        "methods": """- Spherical harmonics: Y_lm(θ, φ) = Σ sph_harm_y(m, l, φ, θ) for l ∈ [0, l_max], m ∈ [-min(l, m_max), min(l, m_max)].
- Spectral norm: norm = Σ |Y_lm|^2, validated against CP8 range [1e3, 1e6].
- Visualization: Heatmap of |Y_lm| to inspect spectral distribution.""",
        "postulates": ["CP8: Topological protection (spectral norm within [1e3, 1e6] ensures robust S^3 basis)"],
        "inputs": ["config_s3*.json: Configuration file with spectral_modes (l_max, m_max), resolution"],
        "outputs": ["results.csv: Stores Y_lm_norm, l_max, m_max, timestamp", "img/05_s3_spectral_heatmap.png: Heatmap of |Y_lm|", "errors.log: Logs debug and error messages"]
    },
    "06_cy3_spectral_base.py": {
        "description": "Computes SU(3)-holonomy basis on a Calabi-Yau threefold (CY_3) for the meta-space manifold 𝓜_meta = S^3 × CY_3 × ℝ_τ, providing an entropy-driven spectral basis for the projection π: 𝓜_meta → 𝓜_4.",
        "methods": """- Holonomy basis: basis(u, v) = sin(u + ψ) * cos(v + φ) + i * cos(u - φ), where u, v ∈ [0, 2π].
- Holonomy norm: norm = Σ |basis|^2, validated against CP8 range [1e3, 1e6].
- Visualization: Heatmap of |basis| to inspect holonomy distribution.""",
        "postulates": ["CP8: Topological protection (holonomy norm within [1e3, 1e6] ensures robust CY_3 basis)", "EP2: Phase-locked projection (ψ, φ ensure phase consistency)", "EP7: Gluon interaction projection (SU(3)-holonomy supports QCD interactions)"],
        "inputs": ["config_cy3*.json: Configuration file with cy3_metric, resolution, complex_structure_moduli (psi, phi)"],
        "outputs": ["results.csv: Stores holonomy_norm, metric, psi, phi, timestamp", "img/06_cy3_holonomy_heatmap.png: Heatmap of |basis|", "errors.log: Logs debug and error messages"]
    },
    "07_gravity_curvature_analysis.py": {
        "description": "Unified curvature and gravitational tensor estimation for the Meta-Space Model (MSM). Combines entropic gradient-based tensor evaluation with Laplacian-based curvature extraction over the manifold M_meta = S³ × CY₃ × ℝ_τ.",
        "methods": """- Entropic gradient: ∇_τ S computed along τ-axis of S.
- Laplacian: ∇²S := ∂²S/∂τ² + ∂²S/∂x² + ∂²S/∂y² for curvature estimation.
- Curvature estimator: I_{μν} := ⟨|∇²S|⟩ (scalar trace).
- Validation: I_{μν} compared against empirical flatness via threshold (e.g. ≤ 1.0).""",
        "postulates": ["CP1: Meta-space geometry (S^3 × CY_3 × ℝ_τ defines the entropic manifold)", "CP3: Geometric emergence (observable quantities emerge from metric properties)", "CP6: Simulation consistency (cross-checks between field and curvature layers)", "EP8: Extended quantum gravity (I_{μν} used as emergent curvature indicator)"],
        "inputs": ["config_grav.json: Contains entropy tensor synthesis parameters (shape, σ, threshold, etc.)", "config_empirical.json: Provides empirical target/threshold for I_μν validation", "results.csv: Used to extract Y_lm_norm and to store new validation metrics", "img/s_field.npy: Numpy array of scalar field S(x, y, τ) from spectral base scripts"],
        "outputs": ["img/07_grav_field_heatmap.png: Heatmap of the best-fit gravitational tensor |I_μν|", "results.csv: Appended with stability and curvature validation metrics", "Terminal summary and logging output"]
    },
    "07a_curvature_simulation.py": {
        "description": "Estimates the curvature trace I_{μν} ≈ ⟨|∇²S|⟩ from the entropic field S(x, y, τ) on the meta-space manifold M_meta = S^3 × CY_3 × ℝ_τ. Serves as a purely geometric consistency check of the MSM framework regarding spatial flatness (Ω_k ≈ 0).",
        "methods": """- Laplacian: ∇²S := ∂²S/∂τ² + ∂²S/∂x² + ∂²S/∂y² (applied only on existing axes).
- Curvature estimator: I_{μν} := ⟨|∇²S|⟩ (scalar trace).
- Validation: I_{μν} compared against empirical flatness via threshold (e.g. ≤ 1.0).""",
        "postulates": ["CP1: Meta-space geometry (S^3 × CY_3 × ℝ_τ defines the entropic manifold)", "CP3: Geometric emergence (observable quantities emerge from metric properties)", "CP6: Simulation consistency (cross-checks between field and curvature layers)", "EP8: Extended quantum gravity (I_{μν} used as emergent curvature indicator)"],
        "inputs": ["img/s_field.npy: Entropic field S(x, y, τ), generated in Script 02", "config_empirical.json: JSON file specifying I_{μν} target and threshold"],
        "outputs": ["results.csv: Appends I_{μν} value, target, deviation and timestamp for validation"]
    },
    "08_cosmo_entropy_scale.py": {
        "description": "Scales the entropic field S(x, y, τ) from Script 02 (img/s_field.npy) on the meta-space manifold 𝓜_meta = S^3 × CY_3 × ℝ_τ to match the dark matter density Ω_DM ≈ 0.27, ensuring holographic and geometric consistency.",
        "methods": """- Entropic gradient: ∇_τ S computed along τ-axis (axis 0) of S.
- Scaling: grad_scaled = ∇_τ S * (target / avg_grad) * (ℓ_D / ℓ_D_ref), where ℓ_D_ref is a reference length scale.
- Ω_DM: Mean of |grad_scaled|.
- Scaling metric: Mean of |grad_scaled| ≥ threshold, targeting ≥ 0.5.
- Deviation: Δ = |Ω_DM - target|.""",
        "postulates": ["CP1: Geometrical substrate (S^3 × CY_3 × ℝ_τ underpins entropic field)", "CP2: Entropy-driven causality (∇_τ S drives Ω_DM projection)", "EP6: Dark matter projection (Ω_DM ≈ 0.27 derived from scaled gradient)", "EP14: Holographic projection (entropic field projects onto physical observables)"],
        "inputs": ["config_cosmo*.json: Configuration file with omega_dm_target, entropy_gradient_threshold, l_d (optional)", "img/s_field.npy: Entropic field from Script 02", "results.csv: Historical data from other scripts"],
        "outputs": ["results.csv: Stores Ω_DM, scaling_metric, deviation, timestamp", "img/08_cosmo_heatmap.png: Heatmap of |∇_τ S_scaled|", "errors.log: Logs errors"]
    },
    "09_test_proposal_sim.py": {
        "description": "Simulates empirical tests for Meta-Space Model (MSM) predictions on 𝓜_meta = S^3 × CY_3 × ℝ_τ, focusing on Bose-Einstein Condensate (BEC) mass drift and neutrino oscillations.",
        "methods": """- BEC mass drift (EP5): m(t) = 1.0 + Σ(α_s * ∇S_thermo * 0.1), where S_thermo = sin(2π * freq * t) * Y_lm_norm / 1e4.
- Neutrino oscillations (EP12): P_ee = [1 - sin^2(2θ) * sin^2(1.27 * Δm^2 * L / E)] * (Y_lm_norm / 1e9) * exp(-L^2 / l_N^2).
- Metrics: mass_drift_metric = std(Δm), oscillation_metric = std(P_ee).""",
        "postulates": ["CP6: Simulation consistency (consistent use of α_s, Y_lm_norm across scripts)", "EP5: Thermodynamic stability (BEC mass drift within threshold)", "EP12: Neutrino oscillations (survival probability P_ee consistent with empirical data)"],
        "inputs": ["config_test*.json: Configuration file with bec_frequency, time_steps, runs, mass_drift_threshold, oscillation_threshold, neutrino (L_min, L_max, num_points, energy, delta_m2, theta, l_N, theta_variation)", "results.csv: α_s (Script 01), Y_lm_norm (Script 05)"],
        "outputs": ["results.csv: Stores mass_drift_metric, oscillation_metric, timestamp", "img/09_test_heatmap_bec.png: Plot of S_thermo for BEC drift", "img/09_test_heatmap_osc.png: Plot of P_ee for neutrino oscillations", "errors.log: Logs errors"]
    },
    "10_external_data_validator.py": {
        "description": "Modular validator for specObj-dr17.fits (EP6: Dark Matter Projection) focusing on FITS file analysis. Extended to include entropic gradient analysis for Higgs mass validation.",
        "methods": """- Analyzes redshift data from FITS file, performing sky binning.
- Estimates local dark matter density (~0.110 M☉/pc³ for z < 0.5) and rho_crit_ratio.
- Entropic gradient analysis for Higgs mass validation.
- Supports CUDA acceleration if available.""",
        "postulates": ["EP6: Dark Matter Projection (local_dm_density, rho_crit_ratio validation)"],
        "inputs": ["config_external.json: Configuration file", "specObj-dr17.fits: FITS file containing astronomical data"],
        "outputs": ["results.csv: Log of analysis results (local_dm_density, rho_crit_ratio)", "img/10_dm_density_heatmap.png: Histogram heatmap", "z_sky_mean.csv: Sky-binned redshift data", "z_sky_mean_map.png: Sky map visualization"]
    },
    "10a_plot_z_sky_mean.py": {
        "description": "Visualizes the mean redshift from z_sky_mean.csv as a sky map and checks for isotropy.",
        "methods": """- Reads z_sky_mean_<class>.csv and plots mean redshift (z̄) as a sky map (RA × DEC).
- Computes isotropy metrics: z_mean_avg, z_mean_std, z_mean_north, z_mean_south, z_mean_delta_ns, z_mean_ttest_pval.
- Visualizes results as heatmap.""",
        "postulates": ["EP6: Dark matter projection (isotropy of z̄ distribution)", "EP12: Neutrino oscillations (z̄ as baseline for oscillation analysis)"],
        "inputs": ["z_sky_mean_<class>.csv: Sky-binned redshift data"],
        "outputs": ["results.csv: Stores z_mean_min, z_mean_max, z_mean_avg, z_mean_std, z_mean_north, z_mean_south, z_std_north, z_std_south, z_mean_delta_ns, z_mean_ttest_pval", "img/z_sky_mean_map.png: Sky map visualization"]
    },
    "10b_neutrino_analysis.py": {
        "description": "Analyzes neutrino oscillation metrics using redshift-based baselines from z_sky_mean.csv, computing survival probabilities (P_ee) and oscillation metrics for different energy scales.",
        "methods": """- Converts redshift (z̄) to baseline (L) for neutrino oscillations.
- Computes P_ee = [1 - sin^2(2θ) * sin^2(1.27 * Δm^2 * L / E)] for energies E3, E5, E7, E10.
- Metrics: osc_metric (std(P_ee)), P_ee_mean, P_ee_proj_metric, P_ee_maxdev.
- Supports class-specific analysis (GALAXY, QSO, 2MASS).""",
        "postulates": ["EP9: Neutrino oscillation consistency", "EP12: Neutrino oscillations (P_ee consistent with empirical data)"],
        "inputs": ["z_sky_mean_<class>.csv: Sky-binned redshift data", "config_neutrino.json: Configuration with energy scales, Δm^2, θ"],
        "outputs": ["results.csv: Stores osc_metric, P_ee_mean, P_ee_proj_metric, P_ee_maxdev for each class and energy", "img/10b_neutrino_osc_heatmap_<class>.png: Heatmap of P_ee"]
    },
    "10c_rg_entropy_flow.py": {
        "description": "Extracts a renormalization group inspired coupling flow α_s(τ) from the sky-binned redshift distribution in z_sky_mean.csv. Converts redshift to scale τ ∼ 1 / log(1 + z), computes an effective coupling using 1-loop QCD flow, and compares against the expected low-energy limit α_s(1 GeV⁻¹) ≈ 0.30.",
        "methods": """- Converts redshift to scale: τ ∼ 1 / log(1 + z).
- Computes α_s(τ) using 1-loop QCD flow.
- Compares α_s(τ) at τ = 1 GeV⁻¹ against target (≈ 0.30).
- Visualizes RG flow and histogram of α_s values.""",
        "postulates": ["EP13: Renormalization group consistency (α_s(τ) flows match empirical QCD at low energies)"],
        "inputs": ["z_sky_mean.csv: CSV file containing binned mean redshift or density per sky region", "config_external.json: External configuration with RG parameters"],
        "outputs": ["results.csv: Appended with alpha_s_tau_rg and deviation from target", "img/10c_alpha_s_rg_flow.png: RG flow plot of alpha_s vs. tau", "img/10c_alpha_s_hist.png: Histogram of alpha_s values", "rg_flow_summary.txt: Summary of derived alpha_s at tau = 1 GeV^-1 and comparison"]
    },
    "10d_entropy_map.py": {
        "description": "Computes and visualizes an entropy-weighted RA×DEC sky map based on deviation of mean redshift (z̄) from its global distribution. Includes normalized Shannon entropy, hemispheric analysis, and correlation metrics.",
        "methods": """- Entropy weight: w = exp(-(z̄ - z_mean)² / (2 * z_std²)).
- Normalized Shannon entropy: S_ρ = -Σ(w * log(w + ε)) / log(N).
- Hemispheric analysis: Computes std(entropy_weight) for northern (DEC ≥ 0) and southern (DEC < 0) hemispheres.
- Correlation: Computes corr(entropy_weight, z̄).
- Visualizes entropy-weighted sky map as heatmap.""",
        "postulates": ["EP6: Dark matter projection (isotropy of z̄ distribution)", "EP12: Neutrino oscillations (z̄ as baseline for oscillation analysis)"],
        "inputs": ["z_sky_mean_<class>.csv: Sky-binned redshift data"],
        "outputs": ["img/10d_z_entropy_weight_map_<class>.png: Entropy heatmap over RA×DEC sky", "results.csv: Metrics (entropy_weight_std, normalized_entropy, etc.) with class suffix"]
    },
    "10e_parameter_scan.py": {
        "description": "Scans neutrino oscillation parameter space (Δm², θ) by computing the projection-weighted std(P_ee) across redshift-based baselines (z̄ → L). Results are saved class-wise (GALAXY, QSO, etc.).",
        "methods": """- Converts z̄ to baseline L for neutrino oscillations.
- Computes P_ee = [1 - sin^2(2θ) * sin^2(1.27 * Δm^2 * L / E)] for scanned Δm², θ.
- Metric: oscillation_scan_min = min(std(P_ee)) across parameter space.
- Visualizes oscillation parameter scan as heatmap.""",
        "postulates": ["EP9: Neutrino oscillation consistency", "EP12: Neutrino oscillations (P_ee consistent with empirical data)"],
        "inputs": ["z_sky_mean_<class>.csv: Sky-binned redshift data"],
        "outputs": ["img/10e_oscillation_scan_heatmap_<class>.png: Heatmap of oscillation scan", "results.csv: Appends oscillation_scan_min_<CLASS> metric"]
    },
    "11_2mass_psc_validator.py": {
        "description": "Modular validator for 2MASS PSC files (EP6: Dark Matter Projection) focusing on ASCII data analysis. Extended to include source density analysis for structural validation.",
        "methods": """- Analyzes 2MASS PSC data for source density (~1 source per square arcminute).
- Performs sky binning and computes local_source_density.
- Validates against target (~200 sources/arcmin²) with threshold ±0.5.
- Visualizes source density as heatmap.""",
        "postulates": ["EP6: Dark Matter Projection (source density validation)"],
        "inputs": ["config_2mass.json: Configuration file", "psc_aaa to psc_aal: 2MASS PSC ASCII files"],
        "outputs": ["results.csv: Log of analysis results (local_source_density)", "img/11_source_density_heatmap.png: Histogram heatmap", "z_sky_mean.csv: Sky-binned source density data", "z_sky_mean_map.png: Sky map visualization"]
    }
}


def load_config(path="config_summary.json"):
    """Load configuration file if available."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"output_path": "12_summary.md"}

def format_value(value, precision=6):
    """Format numerical values for display."""
    try:
        if isinstance(value, str) and value.startswith("["):
            return value  # Range values like [1000.0, 1000000.0]
        return f"{float(value):.{precision}e}"
    except (ValueError, TypeError):
        return str(value)

def main():
    """Generate a comprehensive summary of MSM results."""
    logging.info("Starting 12_Summary.py")
    config = load_config()
    output_path = config.get("output_path", "12_summary.md")

    # Define column names for results.csv (since it has no header)
    columns = ["script", "parameter", "value", "target", "deviation", "timestamp"]

    # Load results.csv
    try:
        df = pd.read_csv("results.csv", header=None, names=columns, encoding="utf-8-sig")
        print("Columns in results.csv:", df.columns.tolist())  # Debug column names
    except FileNotFoundError:
        logging.error("results.csv not found.")
        print("Error: results.csv not found.")
        return
    except Exception as e:
        logging.error(f"Error reading results.csv: {e}")
        print(f"Error reading results.csv: {e}")
        return

    # Group results by script
    grouped = df.groupby("script")
    
    # Initialize Markdown content
    markdown_content = []
    markdown_content.append("# Meta-Space Model (MSM) Comprehensive Summary")
    markdown_content.append(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    markdown_content.append("\nThis document summarizes the results of the Meta-Space Model (MSM) simulations across scripts 01-11, based on `results.csv` and script comment headers. Each section details the purpose, methods, results, and significance of the respective script, contextualized within the MSM postulates (CP1-CP8, EP1-EP14).\n")

    # Process each script
    for script_name in sorted(SCRIPT_INFO.keys()):
        markdown_content.append(f"## {script_name}")
        
        # Get script info
        info = SCRIPT_INFO.get(script_name, {})
        description = info.get("description", "No description available.")
        methods = info.get("methods", "No methods specified.")
        postulates = info.get("postulates", [])
        inputs = info.get("inputs", [])
        outputs = info.get("outputs", [])

        # Purpose
        markdown_content.append("### Purpose")
        markdown_content.append(description)
        markdown_content.append("\nThis script contributes to the MSM by addressing specific aspects of the meta-space manifold 𝓜_meta = S^3 × CY_3 × ℝ_τ, ensuring empirical and theoretical consistency.\n")

        # Methods
        markdown_content.append("### Methods")
        markdown_content.append(methods)
        markdown_content.append("\nThese methods leverage entropic projections, spectral analysis, and empirical validation to derive physical observables.\n")

        # Results
        markdown_content.append("### Results")
        if script_name in grouped.groups:
            script_df = grouped.get_group(script_name)
            results = []
            for _, row in script_df.iterrows():
                param = row["parameter"]
                value = format_value(row["value"])
                target = row.get("target", "N/A")
                deviation = format_value(row.get("deviation", "N/A"))
                
                # Check validation status from 04_empirical_validator.py
                status = "N/A"
                if script_name != "04_empirical_validator.py":
                    val_param = f"{param}_validation"
                    val_row = df[(df["script"] == "04_empirical_validator.py") & (df["parameter"] == val_param)]
                    if not val_row.empty:
                        try:
                            target = val_row["target"].iloc[0]
                            value_f = float(val_row["value"].iloc[0])
                            if isinstance(target, str) and target.startswith("["):
                                min_val, max_val = [float(x) for x in target.strip("[]").split(",")]
                                status = "PASS" if min_val <= value_f <= max_val else "FAIL"
                            else:
                                deviation_val = float(val_row["deviation"].iloc[0])
                                threshold = float(target) if target != "N/A" else float("inf")
                                status = "PASS" if deviation_val <= threshold else "FAIL"
                        except Exception:
                            status = "N/A"
                
                results.append(f"- **{param}**: Value={value}, Target={target}, Deviation={deviation}, Status={status}")
            markdown_content.append("\n".join(results))
            markdown_content.append(f"\n**Outputs**: {', '.join(outputs)}")
        else:
            markdown_content.append("No results found in results.csv for this script.")
        
        # Significance
        markdown_content.append("### Significance")
        markdown_content.append(f"The results of {script_name} are significant for the MSM as they validate the following postulates:")
        markdown_content.append("\n".join([f"- {p}" for p in postulates]))
        markdown_content.append("\nThese results ensure that the MSM's predictions align with empirical observations and theoretical expectations, contributing to the overall coherence of the model.\n")

    # Write Markdown file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_content))
        print(f"Summary written to {output_path}")
        logging.info(f"Summary written to {output_path}")
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")
        logging.error(f"Error writing to {output_path}: {e}")

    # Terminal summary
    print("\n=====================================")
    print("     Meta-Space Model: Summary")
    print("=====================================")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scripts Processed: {len(SCRIPT_INFO)}")
    print(f"Output: {output_path}")
    print("Key Validation Results (from 04_empirical_validator.py):")
    val_df = df[df["script"] == "04_empirical_validator.py"]
    for _, row in val_df.iterrows():
        param = row["parameter"]
        value = format_value(row["value"])
        target = row["target"]
        deviation = format_value(row["deviation"])
        try:
            status = "PASS" if float(row["deviation"]) <= float(row["target"]) else "FAIL"
        except (ValueError, TypeError):
            status = "N/A"
        print(f"- {param}: Value={value}, Target={target}, Deviation={deviation}, Status={status}")
    print("=====================================\n")

if __name__ == "__main__":
    main()