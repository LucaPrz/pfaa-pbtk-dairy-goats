import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import logging

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.config import ModelConfig, FitConfig, FittingContext, SimulationConfig, get_matrix_module, get_valid_matrices
from optimization.fit import simulate_model
from optimization.loss import predict_single_observation

logger = logging.getLogger(__name__)


def collect_residuals_for_pair(
    pair: Tuple[str, str],
    context: 'FittingContext'
) -> Dict[str, List[float]]:
    """
    Collect log residuals for a single compound-isomer pair.
    Returns a dictionary mapping matrix names to lists of residuals.
    """
    compound, isomer = pair
    valid_matrices = get_valid_matrices(context.config)
    residuals_by_matrix = {m: [] for m in valid_matrices}

    try:
        # Load fitted parameters
        fit_path = context.folder_phase1 / f'fit_{compound}_{isomer}.csv'
        if not fit_path.exists():
            logger.debug(f"[SIGMA EST] Fit file not found: {fit_path}, skipping")
            return residuals_by_matrix

        fit_df = pd.read_csv(fit_path)
        fit_params_dict = dict(zip(fit_df['Parameter'], fit_df['Value']))

        # Get parameter configuration (respects use_simple_model for E_milk vs k_milk)
        from optimization.fit_variables import get_parameter_config
        data_df = context.data_cache.get_pair_data(compound, isomer)
        param_names, fixed_params = get_parameter_config(compound, isomer, data_df, config=context.config)
        
        if not param_names:
            logger.debug(f"[SIGMA EST] No parameters fitted for {compound} {isomer}, skipping")
            return residuals_by_matrix
        
        # Extract fitted parameter values
        fit_params = [fit_params_dict.get(name, 0.0) for name in param_names]
        
        # Get data
        use_data = data_df.copy()
        
        # Create fit config
        fit_config = FitConfig(compound=compound, isomer=isomer)
        
        # Create simulation function
        def simulate_model_func(params, sim_config):
            return simulate_model(
                params, sim_config, context,
                param_names=param_names,
                fixed_params=fixed_params
            )
        
        # Simulate model with fitted parameters
        animals = use_data['Animal'].unique()
        simulation_cache = {}
        
        for animal in animals:
            sim_config = SimulationConfig(
                compound=compound,
                isomer=isomer,
                animal=animal
            )
            try:
                solution, all_params = simulate_model_func(fit_params, sim_config)
                simulation_cache[animal] = (solution, all_params)
            except Exception:
                simulation_cache[animal] = (None, None)
        
        # Collect residuals for each matrix
        matrix_module = get_matrix_module(context.config)
        for matrix_name in use_data['Matrix'].str.lower().unique():
            if matrix_name not in valid_matrices:
                continue
            
            df_matrix = use_data[use_data['Matrix'].str.lower() == matrix_name]
            if df_matrix.empty:
                continue
            
            loq = context.config.loq_milk if matrix_name == "milk" else context.config.loq
            
            for _, row in df_matrix.iterrows():
                animal = row['Animal']
                t = row['Day']
                y_obs = pd.to_numeric(row['Concentration'], errors='coerce')
                
                # Skip invalid observations
                if pd.isna(y_obs) or y_obs <= 0:
                    continue
                
                # Get prediction
                sim, all_params = simulation_cache.get(animal, (None, None))
                if sim is None or all_params is None:
                    continue
                
                y_pred = predict_single_observation(
                    sim, all_params, matrix_name, t, animal,
                    context.urine_volume_by_animal,
                    context.feces_mass_by_animal,
                    context.feces_mass_default,
                    context.milk_yield_by_animal,
                    matrix_module=matrix_module,
                )
                
                # Only use uncensored observations for sigma estimation
                if pd.isna(y_pred) or y_pred <= 0:
                    continue
                
                if y_obs > loq:  # Uncensored observation
                    log_obs = np.log(y_obs + context.config.eps)
                    log_pred = np.log(y_pred + context.config.eps)
                    residual = log_obs - log_pred
                    residuals_by_matrix[matrix_name].append(residual)
        
    except Exception as e:
        logger.debug(f"[SIGMA EST] Failed to collect residuals for {compound} {isomer}: {e}")
    
    return residuals_by_matrix


# Tiny floor so 1/sigma in likelihood and FIM never blows up; no upper or 0.1 clip.
SIGMA_EPS = 1e-9


def _sigma_from_residuals(
    residuals: List[float],
    default_sigma: float,
    min_n: int = 3,
) -> float:
    """
    Estimate sigma from a list of log-residuals (MAD + STD, median).
    Returns default_sigma if too few residuals. Only floor at SIGMA_EPS to avoid 1/sigma issues.
    """
    if len(residuals) < min_n:
        return default_sigma
    arr = np.array(residuals)
    mad = np.median(np.abs(arr - np.median(arr)))
    sigma_mad = 1.4826 * mad if mad > 0 else np.std(arr, ddof=1)
    sigma_std = np.std(arr, ddof=1)
    sigma_est = np.median([sigma_mad, sigma_std])
    return float(max(SIGMA_EPS, sigma_est))


def estimate_sigma_per_pair(
    pair: Tuple[str, str],
    context: 'FittingContext',
    pooled_sigma: Dict[str, float],
) -> Dict[str, float]:
    """
    Estimate sigma per matrix for a single compound-isomer pair (effective model error for that pair).
    Uses only that pair's residuals; falls back to pooled sigma for matrices with too few residuals.
    """
    compound, isomer = pair
    residuals_by_matrix = collect_residuals_for_pair(pair, context)
    valid_matrices = get_valid_matrices(context.config)
    sigma_per_matrix = {}
    for matrix_name in valid_matrices:
        residuals = residuals_by_matrix[matrix_name]
        default = pooled_sigma.get(matrix_name, context.config.sigma_per_matrix.get(matrix_name.lower(), context.config.sigma_default))
        sigma_per_matrix[matrix_name] = _sigma_from_residuals(residuals, default_sigma=default)
    return sigma_per_matrix


def estimate_pooled_sigma(
    context: 'FittingContext',
    pairs: Optional[List[Tuple[str, str]]] = None
) -> Dict[str, float]:
    """
    Pooled sigma per matrix based on *all* residuals across compounds.

    Instead of first estimating a sigma per (compound, isomer, matrix) and then
    taking the median of those sigmas, we aggregate all log‑residuals across
    compounds for each matrix and estimate sigma directly from that pooled set.

    This avoids the situation where many per‑pair sigmas are just defaults
    (because each pair alone has too few residuals), even though across all
    compounds there is ample data to estimate a robust sigma per matrix.
    """
    if pairs is None:
        pairs = []
        for compound in context.config.compounds:
            data_df = context.data_cache.get_pair_data(compound, "Linear")
            if data_df is not None and not data_df.empty:
                pairs.append((compound, "Linear"))
            data_df = context.data_cache.get_pair_data(compound, "Branched")
            if data_df is not None and not data_df.empty:
                pairs.append((compound, "Branched"))

    logger.info(f"[SIGMA EST] Estimating pooled sigma from {len(pairs)} compound-isomer pairs (pooled residuals per matrix)...")

    valid_matrices = get_valid_matrices(context.config)
    # Default sigma per matrix (for fallback when a matrix has no residuals at all)
    default_sigma = {
        m: context.config.sigma_per_matrix.get(m, context.config.sigma_default)
        for m in valid_matrices
    }

    # Collect all residuals per matrix across all pairs
    residuals_by_matrix_all: Dict[str, List[float]] = {m: [] for m in valid_matrices}
    for pair in pairs:
        compound, isomer = pair
        logger.debug(f"[SIGMA EST] Collecting residuals for {compound} {isomer}...")
        pair_residuals = collect_residuals_for_pair(pair, context)
        for matrix_name, res_list in pair_residuals.items():
            if res_list:
                residuals_by_matrix_all[matrix_name].extend(res_list)

    # Pooled sigma per matrix from *all* residuals
    sigma_per_matrix = {}
    for matrix_name in valid_matrices:
        pooled_residuals = residuals_by_matrix_all[matrix_name]
        if len(pooled_residuals) == 0:
            # Absolutely no residuals for this matrix across any compound
            sigma_per_matrix[matrix_name] = default_sigma[matrix_name]
            logger.debug(f"[SIGMA EST] {matrix_name}: no residuals across any pair, using default σ={default_sigma[matrix_name]:.3f}")
            continue

        # Use the same robust estimator as for per‑pair σ, but on pooled residuals.
        # min_n keeps protection against extremely sparse information, but now
        # counts residuals across all compounds.
        sigma_est = _sigma_from_residuals(
            residuals=pooled_residuals,
            default_sigma=default_sigma[matrix_name],
        )
        sigma_per_matrix[matrix_name] = sigma_est
        logger.info(
            f"[SIGMA EST] {matrix_name}: pooled σ={sigma_est:.3f} "
            f"(from {len(pooled_residuals)} pooled residuals)"
        )

    return sigma_per_matrix


def estimate_and_save_pooled_sigma(
    context: 'FittingContext',
    pairs: Optional[List[Tuple[str, str]]] = None
) -> Optional[Dict[str, float]]:
    try:
        if pairs is None:
            pairs = []
            for compound in context.config.compounds:
                data_df = context.data_cache.get_pair_data(compound, "Linear")
                if data_df is not None and not data_df.empty:
                    pairs.append((compound, "Linear"))
                data_df = context.data_cache.get_pair_data(compound, "Branched")
                if data_df is not None and not data_df.empty:
                    pairs.append((compound, "Branched"))
        # Estimate pooled sigma (by matrix, across all pairs)
        sigma_per_matrix = estimate_pooled_sigma(context, pairs)
        
        # Save pooled sigma estimates in the original optimisation folder
        sigma_dir = context.folder_phase1.parent / "sigma_estimates"
        sigma_dir.mkdir(parents=True, exist_ok=True)
        sigma_path = sigma_dir / 'sigma_pooled.csv'
        
        sigma_df = pd.DataFrame([
            {'Matrix': matrix, 'Sigma': sigma}
            for matrix, sigma in sigma_per_matrix.items()
        ])
        sigma_df.to_csv(sigma_path, index=False)
        logger.info(f"[SIGMA EST] Saved pooled sigma estimates to {sigma_path}")

        # Estimate and save compound-level sigma (per pair, per matrix) for MC propagation
        for pair in pairs:
            compound, isomer = pair
            pair_sigma = estimate_sigma_per_pair(pair, context, sigma_per_matrix)
            pair_path = sigma_dir / f'sigma_{compound}_{isomer}.csv'
            pair_df = pd.DataFrame([
                {'Matrix': m, 'Sigma': s, 'Compound': compound, 'Isomer': isomer}
                for m, s in pair_sigma.items()
            ])
            pair_df.to_csv(pair_path, index=False)
            logger.debug(f"[SIGMA EST] Saved compound-level sigma to {pair_path}")
        
        return sigma_per_matrix
        
    except Exception as e:
        logger.error(f"[SIGMA EST] Failed to estimate pooled sigma: {e}", exc_info=True)
        return None