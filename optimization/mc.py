"""Monte Carlo sampling logic."""
import sys
import logging
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, TYPE_CHECKING
import numpy as np
import pandas as pd
from tqdm import tqdm

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.config import SimulationConfig, FitConfig, get_matrix_module
from optimization.fit import simulate_model
from optimization.loss import predict_time_series, loss_function

if TYPE_CHECKING:
    from optimization.config import FittingContext

logger = logging.getLogger(__name__)


def numerical_hessian(
    f: Any,
    x: np.ndarray,
    step: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Numerically approximate the Hessian of a scalar function at x (central differences)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)
    if step is None:
        step = np.full(n, 1e-2, dtype=float)
    else:
        step = np.asarray(step, dtype=float)
    f_x = float(f(x))
    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = step[i]
        H[i, i] = (float(f(x + ei)) - 2.0 * f_x + float(f(x - ei))) / (step[i] ** 2)
    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = step[i]
        for j in range(i + 1, n):
            ej = np.zeros(n, dtype=float)
            ej[j] = step[j]
            f_pp = float(f(x + ei + ej))
            f_pm = float(f(x + ei - ej))
            f_mp = float(f(x - ei + ej))
            f_mm = float(f(x - ei - ej))
            val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step[i] * step[j])
            H[i, j] = val
            H[j, i] = val
    return H

def run_monte_carlo_for_pair(
    pair: Tuple[str, str],
    n_mc_samples: Optional[int],
    context: 'FittingContext'
) -> Optional[Path]:
    compound, isomer = pair
    
    # Use provided n_mc_samples or default to config value
    if n_mc_samples is None:
        n_mc_samples = context.config.n_mc_samples
    
    # Set random seed for reproducibility
    # Use Generator instead of global seed for thread-safety in multiprocessing
    if context.config.mc_random_seed is not None:
        rng = np.random.default_rng(context.config.mc_random_seed)
    else:
        rng = np.random.default_rng()
    
    start_time = time.time()
    logger.info(f"[MONTE CARLO] Starting {compound} {isomer}")
    
    try:
        # Expected failures: missing required files for this compound-isomer pair
        mean_path = context.folder_phase1 / f'fit_{compound}_{isomer}.csv'
        if not mean_path.exists():
            logger.warning(f"[MONTE CARLO] Phase 1 file not found: {mean_path}. Skipping {compound} {isomer}.")
            return None
        
        mean_df = pd.read_csv(mean_path)
        if not all(name in mean_df['Parameter'].values for name in context.config.param_names):
            logger.warning(f"[MONTE CARLO] Phase 1 file missing required parameters. Skipping {compound} {isomer}.")
            return None
        
        # Extract Phase 1 point estimates in correct order (linear space)
        phase1_vals_linear = np.array(
            [mean_df[mean_df['Parameter'] == name]['Value'].iloc[0] for name in context.config.param_names],
            dtype=float,
        )
        ndim = len(context.config.param_names)
        
        # Load Phase 2 results (jackknife fits) - now from CSV (linear space)
        jackknife_path = context.folder_phase2 / f'jackknife_{compound}_{isomer}_LOAO.csv'
        if not jackknife_path.exists():
            logger.warning(f"[MONTE CARLO] Phase 2 file not found: {jackknife_path}. Skipping {compound} {isomer}.")
            return None
        
        jackknife_df = pd.read_csv(jackknife_path)
        if not all(name in jackknife_df.columns for name in context.config.param_names):
            logger.warning(f"[MONTE CARLO] Jackknife file missing required parameters. Skipping {compound} {isomer}.")
            return None
        
        jackknife_fits_linear = jackknife_df[context.config.param_names].values.astype(float)
        if jackknife_fits_linear.shape[1] != ndim:
            logger.warning(f"[MONTE CARLO] Jackknife parameter mismatch for {compound} {isomer}.")
            return None

        # Work in log10-space for strictly positive parameters
        eps_param = 1e-12
        phase1_vals_log = np.log10(np.clip(phase1_vals_linear, eps_param, None)).astype(np.float64)
        jackknife_fits_log = np.log10(np.clip(jackknife_fits_linear, eps_param, None)).astype(np.float64)

        # For diagnostics, compute jackknife medians/means in log10-space
        medians_log = np.median(jackknife_fits_log, axis=0)
        means_log = np.mean(jackknife_fits_log, axis=0)
        jackknife_stds_log = np.std(jackknife_fits_log, axis=0, ddof=1 if jackknife_fits_log.shape[0] > 1 else 0)
        stds_log = np.sqrt(np.maximum(jackknife_stds_log ** 2, 1e-4))
        stds_log = np.nan_to_num(stds_log, nan=0.1, posinf=1.0, neginf=0.1).astype(np.float64)

        # Prefer full covariance sampling (Hessian-based) so parameter correlations
        # (e.g. k_ehc vs k_feces) are respected and prediction bands are not
        # inflated by implausible parameter combinations.
        use_full_cov = False
        try:
            df_pair = context.data_cache.get_pair_data(compound, isomer)
            param_names_fit = context.config.get_param_names(compound, isomer, df_pair)
            fixed_params = context.config.get_fixed_parameters(compound, isomer, df_pair)
            if param_names_fit:
                fit_config = FitConfig(compound=compound, isomer=isomer)

                def loss_wrapper(log_fit_params: np.ndarray) -> float:
                    return float(
                        loss_function(
                            log_fit_params,
                            fit_config=fit_config,
                            use_data=df_pair,
                            context=context,
                            simulate_model_func=lambda p, sim_cfg: simulate_model(
                                p, sim_cfg, context,
                                param_names=param_names_fit,
                                fixed_params=fixed_params,
                            ),
                        )
                    )

                phase1_fit_linear = np.array(
                    [mean_df[mean_df["Parameter"] == n]["Value"].iloc[0] for n in param_names_fit],
                    dtype=float,
                )
                x0 = np.log10(np.clip(phase1_fit_linear, eps_param, None)).astype(np.float64)
                log_bounds = context.config.get_log_bounds(param_names_fit)
                steps = np.array([(hi - lo) * 1e-2 for (lo, hi) in log_bounds], dtype=float)
                H = numerical_hessian(loss_wrapper, x0, step=steps)
                ridge = 1e-6 * max(float(np.max(np.diag(H))), 1e-6)
                cov_log = np.linalg.pinv(H + ridge * np.eye(H.shape[0]))
                cov_log = 0.5 * (cov_log + cov_log.T)
                evals, evecs = np.linalg.eigh(cov_log)
                evals = np.maximum(evals, 1e-8)
                cov_log = evecs @ np.diag(evals) @ evecs.T
                L = np.linalg.cholesky(cov_log)
                n_fit = len(param_names_fit)
                Z = rng.standard_normal((n_mc_samples, n_fit), dtype=np.float64)
                fitted_samples_log = x0 + (Z @ L.T)
                name_to_idx = {n: i for i, n in enumerate(context.config.param_names)}
                fitted_indices = np.array([name_to_idx[n] for n in param_names_fit])
                mc_samples_log = np.tile(phase1_vals_log, (n_mc_samples, 1))
                mc_samples_log[:, fitted_indices] = fitted_samples_log
                use_full_cov = True
                logger.info(f"[MONTE CARLO] {compound} {isomer} - Using Hessian full covariance (log10)")
        except Exception as e:
            logger.warning(f"[MONTE CARLO] {compound} {isomer} - Hessian covariance failed, using diagonal: {e}")

        if not use_full_cov:
            logger.info(f"[MONTE CARLO] {compound} {isomer} - Phase 1 point estimates (linear): {phase1_vals_linear}")
            logger.info(f"[MONTE CARLO] {compound} {isomer} - Phase 1 central values (log10): {phase1_vals_log}")
            logger.info(f"[MONTE CARLO] {compound} {isomer} - Jackknife stds (log10, diagonal): {jackknife_stds_log}")
            logger.info(f"[MONTE CARLO] {compound} {isomer} - Generating {n_mc_samples} samples in log10-space (diagonal covariance)")
            mc_samples_log = phase1_vals_log + rng.standard_normal((n_mc_samples, ndim), dtype=np.float64) * stds_log
        else:
            logger.info(f"[MONTE CARLO] {compound} {isomer} - Phase 1 point estimates (linear): {phase1_vals_linear}")
            logger.info(f"[MONTE CARLO] {compound} {isomer} - Phase 1 central values (log10): {phase1_vals_log}")
            logger.info(f"[MONTE CARLO] {compound} {isomer} - Generating {n_mc_samples} samples in log10-space (full covariance)")
        
        # Transform samples back to linear space (vectorized)
        mc_samples = 10.0 ** mc_samples_log
        
        # Log sampling statistics
        sample_means = np.mean(mc_samples, axis=0)
        sample_stds = np.std(mc_samples, axis=0)
        logger.info(f"[MONTE CARLO] {compound} {isomer} - Sample means: {sample_means}")
        logger.info(f"[MONTE CARLO] {compound} {isomer} - Sample stds: {sample_stds}")
        
        # Get data for this compound-isomer pair (convert to arrays once)
        df_pair = context.data_cache.get_pair_data(compound, isomer)
        
        # Get matrices that actually have data for this compound-isomer pair
        available_matrices = []
        # Include those listed in matrix_weights
        for matrix_name, weight in context.config.matrix_weights.items():
            df_matrix = df_pair[df_pair["Matrix"].str.lower() == matrix_name]
            if not df_matrix.empty:
                available_matrices.append(matrix_name)
        # Explicitly include urine and feces if present in data
        for extra_matrix in ["urine", "feces"]:
            df_extra = df_pair[df_pair["Matrix"].str.lower() == extra_matrix]
            if not df_extra.empty and extra_matrix not in available_matrices:
                available_matrices.append(extra_matrix)
        
        # Collect predictions for compartments that have data
        compartments_with_data = list(set(available_matrices))
        # Always include synthetic elimination stream
        if "elim" not in compartments_with_data:
            compartments_with_data.append("elim")

        logger.info(f"[MONTE CARLO] {compound} {isomer} - Collecting predictions for: {compartments_with_data}")
        all_series_by_comp = {comp: [] for comp in compartments_with_data}

        # Performance note: This nested loop is the main bottleneck for large MC runs.
        # Each simulate_model call is expensive (ODE solving), and we can't easily batch
        # because each animal has different intake functions. Optimizations applied:
        # - Pre-computed milk yield lookups (vectorized)
        # - Float32 casting to reduce memory
        # - Vectorized aggregation after collection
        # - Chunked processing for very large MC runs to reduce memory pressure
        # - Pre-allocated simulation configs
        total_predictions = len(mc_samples) * len(context.config.animals)
        
        # Use chunking for very large MC runs to reduce memory usage
        # Process in chunks to avoid excessive memory accumulation
        chunk_size = min(context.config.mc_chunk_size, n_mc_samples)
        n_chunks = (n_mc_samples + chunk_size - 1) // chunk_size
        
        with tqdm(total=total_predictions, desc=f"MC Predictions {compound} {isomer}", unit="pred", leave=False) as pbar:
            # Pre-allocate simulation configs to avoid repeated creation
            sim_configs = [
                SimulationConfig(compound=compound, isomer=isomer, animal=animal)
                for animal in context.config.animals
            ]
            
            # Process MC samples in chunks to reduce memory usage
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_mc_samples)
                chunk_samples = mc_samples[start_idx:end_idx]
                
                # Process this chunk
                for params in chunk_samples.astype(np.float32):
                    for i, animal in enumerate(context.config.animals):
                        pred, all_params = simulate_model(params, sim_configs[i], context)
                        for comp in compartments_with_data:
                            series = predict_time_series(
                                pred, all_params, comp, animal,
                                context.urine_volume_by_animal,
                                context.feces_mass_by_animal,
                                context.feces_mass_default,
                                context.milk_yield_by_animal,
                                matrix_module=get_matrix_module(context.config),
                            ).astype(np.float32)
                            all_series_by_comp[comp].append(series)
                        pbar.update(1)
                
                # Periodic garbage collection for large runs
                if chunk_idx % 10 == 0 and chunk_idx > 0:
                    gc.collect()

        # Process predictions and create output
        records = []
        for comp in compartments_with_data:
            try:
                # Empirical median and 95% CI across all animal-series over all MC samples
                if len(all_series_by_comp[comp]) == 0:
                    raise ValueError("No series accumulated for component")
                stacked = np.vstack(all_series_by_comp[comp])  # shape: (num_series, T)
                ci_lower = np.nanpercentile(stacked, 2.5, axis=0)
                pred_median = np.nanpercentile(stacked, 50.0, axis=0)
                ci_upper = np.nanpercentile(stacked, 97.5, axis=0)
                # Param uncertainty and animal variation
                n_animals = len(context.config.animals)
                n_samples = len(all_series_by_comp[comp]) // n_animals
                if len(all_series_by_comp[comp]) % n_animals != 0:
                    raise ValueError("Unexpected per-sample/animal counts for MC aggregation")
                
                # Reshape to (n_samples, n_animals, n_timepoints) for vectorized operations
                reshaped = np.array(all_series_by_comp[comp]).reshape(n_samples, n_animals, -1)
                
                # Vectorized computation of per-sample means and animal stds
                per_sample_means = np.nanmean(reshaped, axis=1)  # (n_samples, n_timepoints)
                animal_stds = np.nanstd(reshaped, axis=1)  # (n_samples, n_timepoints)
                
                param_uncertainty_lower = np.nanpercentile(per_sample_means, 2.5, axis=0)
                param_uncertainty_upper = np.nanpercentile(per_sample_means, 97.5, axis=0)
                animal_variation = np.nanmean(animal_stds, axis=0)

                # Observation-level prediction interval and variance decomposition in log-space
                sigma_comp = context.config.get_sigma(comp, compound=compound, isomer=isomer)
                eps = context.config.eps

                # 1) Observation-level CI: add measurement noise (sigma) in log-space to stacked predictions
                log_pred = np.log(np.maximum(stacked, eps))
                noise = rng.normal(0, sigma_comp, size=stacked.shape)
                log_obs = log_pred + noise
                obs_draws = np.exp(log_obs)
                obs_ci_lower = np.nanpercentile(obs_draws, 2.5, axis=0)
                obs_ci_upper = np.nanpercentile(obs_draws, 97.5, axis=0)

                # 2) Exact variance decomposition in log-space using raw MC draws
                #    log_reshaped: (n_samples, n_animals, n_timepoints)
                log_reshaped = np.log(np.maximum(reshaped, eps))
                # Parameter variance: variance across samples of the animal-mean log prediction
                log_means_per_sample = np.nanmean(log_reshaped, axis=1)  # (n_samples, n_timepoints)
                ddof_param = 1 if n_samples > 1 else 0
                var_param_log = np.nanvar(log_means_per_sample, axis=0, ddof=ddof_param)  # (n_timepoints,)
                # Animal variance: mean over samples of within-sample variance across animals
                ddof_animal = 1 if n_animals > 1 else 0
                var_within_animals = np.nanvar(log_reshaped, axis=1, ddof=ddof_animal)  # (n_samples, n_timepoints)
                var_animal_log = np.nanmean(var_within_animals, axis=0)  # (n_timepoints,)
                # Observational variance in log-space (constant over time for this compartment)
                var_obs_log = float(sigma_comp ** 2)
                # Total variance
                var_total_log = var_param_log + var_animal_log + var_obs_log

                # Vectorized record creation
                n_timepoints = len(context.config.time_vector)
                records.extend([
                    {
                        'Compound': compound,
                        'Isomer': isomer,
                        'Compartment': comp,
                        'Time': context.config.time_vector[t_idx],
                        'Pred_Median': float(pred_median[t_idx]),
                        'CI_Lower': float(ci_lower[t_idx]),
                        'CI_Upper': float(ci_upper[t_idx]),
                        'Param_CI_Lower': float(param_uncertainty_lower[t_idx]),
                        'Param_CI_Upper': float(param_uncertainty_upper[t_idx]),
                        'Animal_Variation': float(animal_variation[t_idx]),
                        'CI_Observation_Lower': float(obs_ci_lower[t_idx]),
                        'CI_Observation_Upper': float(obs_ci_upper[t_idx]),
                        'Sigma': float(sigma_comp),
                        'Var_param_log': float(var_param_log[t_idx]),
                        'Var_animal_log': float(var_animal_log[t_idx]),
                        'Var_obs_log': var_obs_log,
                        'Var_total_log': float(var_total_log[t_idx]),
                    }
                    for t_idx in range(n_timepoints)
                ])
            except Exception as e:
                logger.error(f"[MONTE CARLO] Failed to process compartment {comp}: {e}")
                continue
        
        # Expected failure: no valid predictions generated
        if len(records) == 0:
            logger.warning(f"[MONTE CARLO] No valid predictions generated for {compound} {isomer}")
            return None
            
        df = pd.DataFrame(records)
        
        # Add provenance metadata for reproducibility and auditability
        df['MC_Samples'] = n_mc_samples
        df['MC_Seed'] = context.config.mc_random_seed if context.config.mc_random_seed is not None else -1
        df['MC_Date'] = datetime.now().isoformat()
        df['Model_Version'] = 'PB(T)K v1.0'
        
        out_csv = context.folder_phase3 / f'predictions_{compound}_{isomer}_monte_carlo.csv'
        df.to_csv(out_csv, index=False)
        
        elapsed_time = time.time() - start_time
        logger.info(f"[MONTE CARLO] {compound} {isomer} - Saved predictions to {out_csv}")
        logger.info(f"[MONTE CARLO] {compound} {isomer} - Total samples: {len(mc_samples)} parameter draws × {len(context.config.animals)} animals = {len(mc_samples) * len(context.config.animals)} total predictions")
        logger.info(f"[MONTE CARLO] {compound} {isomer} - Completed in {elapsed_time:.1f}s")
        del all_series_by_comp, mc_samples
        gc.collect()
        return out_csv

    except Exception as e:
        # Unexpected error: catch, log, and return None to allow parallel execution to continue
        elapsed_time = time.time() - start_time
        logger.error(f"[MONTE CARLO] {compound} {isomer} - FAILED after {elapsed_time:.1f}s: {e}", exc_info=True)
        return None