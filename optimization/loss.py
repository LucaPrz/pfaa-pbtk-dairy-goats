import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any, Callable, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy.stats import norm

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model.diagnose as default_matrix_module
from model.solve import SimulationResult
from optimization.config import ModelConfig, TimeHandler, FitConfig, get_matrix_module

if TYPE_CHECKING:
    from optimization.config import FittingContext

logger = logging.getLogger(__name__)

def get_daily_milk(cumulative: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Convert cumulative milk excretion to daily rates."""
    cumulative = np.asarray(cumulative)
    # Guarantee 1D
    if cumulative.ndim > 1:
        cumulative = cumulative.flatten()
    
    # Handle negative cumulative values (numerical issues)
    cumulative = np.maximum(cumulative, 0)  # Ensure non-negative
    
    daily = np.empty_like(cumulative)
    daily[0] = cumulative[0]
    daily[1:] = cumulative[1:] - cumulative[:-1]
    
    # Ensure daily values are non-negative (milk can't be negative)
    daily = np.maximum(daily, 0)
    
    return daily

def cumulative_to_daily(cumulative_array: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Convert cumulative excretion array to daily excretion rates."""
    cum = np.asarray(cumulative_array, dtype=float).reshape(-1)
    if cum.size == 0:
        return np.array([], dtype=float)
    daily = np.diff(cum, prepend=cum[0])
    daily[0] = cum[0]
    return np.maximum(daily, 0.0)

def predict_excretion_concentration(
    daily_mass: np.ndarray, 
    animal: str, 
    excretion_type: str,
    urine_volume_by_animal: Dict[str, float],
    feces_mass_by_animal: Dict[str, float],
    feces_mass_default: float
) -> np.ndarray:
    if excretion_type == "urine":
        if animal not in urine_volume_by_animal:
            raise KeyError(f"Urine volume not found for animal {animal}")
        volume = urine_volume_by_animal[animal]
        if volume <= 0:
            return np.full_like(daily_mass, np.nan, dtype=float)
        return np.maximum(daily_mass / volume, 0.0)  # µg/L
    elif excretion_type == "feces":
        mass = feces_mass_by_animal.get(animal, feces_mass_default)
        if mass <= 0:
            return np.full_like(daily_mass, np.nan, dtype=float)
        return np.maximum(daily_mass / mass, 0.0)  # µg/kg
    else:
        raise ValueError(f"Unknown excretion type: {excretion_type}")

def predict_milk_concentration(
    daily_milk_mass: np.ndarray, 
    animal: str,
    milk_yield_by_animal: Dict[str, np.ndarray]
) -> np.ndarray:
    # Use pre-computed lookup array for O(1) access instead of DataFrame filtering
    yields = milk_yield_by_animal.get(animal)
    if yields is None:
        # Fallback: create default array if animal not found
        yields = np.ones(len(daily_milk_mass), dtype=np.float32)
    else:
        # Ensure yields array is long enough, pad with 1.0 if needed
        if len(yields) < len(daily_milk_mass):
            yields = np.pad(yields, (0, len(daily_milk_mass) - len(yields)), constant_values=1.0)
        # Take only the days we need
        yields = yields[:len(daily_milk_mass)]
    
    # Vectorized division: daily_milk_mass / yields, ensuring non-negative
    concentrations = np.maximum(daily_milk_mass / yields, 0.0)
    return concentrations.astype(np.float32)

def predict_single_observation(
    simulation_tuple: SimulationResult,
    all_params: Dict[str, Any],
    matrix_name: str,
    t: float,
    animal: str,
    urine_volume_by_animal: Dict[str, float],
    feces_mass_by_animal: Dict[str, float],
    feces_mass_default: float,
    milk_yield_by_animal: Dict[str, np.ndarray],
    matrix_module: Optional[Any] = None,
) -> float:
    # Use the provided matrix module (for simple vs full model), falling
    # back to the default diagnose module.
    m = matrix_module if matrix_module is not None else default_matrix_module
    try:
        t_idx = TimeHandler.observation_to_prediction_time(t)

        if matrix_name == "milk":
            cumulative = simulation_tuple.milk_array
            daily_milk = get_daily_milk(cumulative)
            if t_idx >= len(daily_milk):
                return np.nan
            concentrations = predict_milk_concentration(daily_milk, animal, milk_yield_by_animal)
            return float(np.clip(concentrations[t_idx], 0.0, None))

        if matrix_name == "urine":
            cum = simulation_tuple.urine_array
            daily = cumulative_to_daily(cum)
            if daily.size == 0 or t_idx >= len(daily):
                return np.nan
            concentrations = predict_excretion_concentration(daily, animal, "urine", urine_volume_by_animal, feces_mass_by_animal, feces_mass_default)
            if np.any(np.isnan(concentrations)):
                return np.nan
            return float(np.clip(concentrations[t_idx], 0.0, None))

        if matrix_name == "feces":
            cum = simulation_tuple.feces_array
            daily = cumulative_to_daily(cum)
            if daily.size == 0 or t_idx >= len(daily):
                return np.nan
            concentrations = predict_excretion_concentration(daily, animal, "feces", urine_volume_by_animal, feces_mass_by_animal, feces_mass_default)
            if np.any(np.isnan(concentrations)):
                return np.nan
            return float(np.clip(concentrations[t_idx], 0.0, None))

        # Tissue/plasma: mass -> concentration via volume (use active model's compartments)
        if matrix_name not in m.PBTKModel.compartment_idx:
            return np.nan
        idx = m.PBTKModel.compartment_idx[matrix_name]
        raw_pred = simulation_tuple.mass_matrix[t_idx, idx]
        vol_key = "V_mammary_deep" if matrix_name == "mammary_deep" else f"V_{matrix_name}"
        volume = all_params["physiological"].get(vol_key, 0)
        if volume <= 0:
            return np.nan
        return max(0.0, raw_pred / volume)

    except Exception:
        return np.nan

def predict_time_series(
    simulation_tuple: SimulationResult,
    all_params: Dict[str, Any],
    comp: str,
    animal: str,
    urine_volume_by_animal: Dict[str, float],
    feces_mass_by_animal: Dict[str, float],
    feces_mass_default: float,
    milk_yield_by_animal: Dict[str, np.ndarray],
    matrix_module: Optional[Any] = None,
) -> np.ndarray:
    # Use the provided matrix module (for simple vs full model), falling
    # back to the default diagnose module.
    m = matrix_module if matrix_module is not None else default_matrix_module
    pred_matrix = simulation_tuple.mass_matrix

    if comp == "milk":
        cum = simulation_tuple.milk_array
        daily_milk = get_daily_milk(cum)
        return predict_milk_concentration(daily_milk, animal, milk_yield_by_animal)

    if comp == "urine":
        cum = simulation_tuple.urine_array
        daily = cumulative_to_daily(cum)
        if daily.size == 0:
            return np.array([], dtype=float)
        return predict_excretion_concentration(daily, animal, "urine", urine_volume_by_animal, feces_mass_by_animal, feces_mass_default)

    if comp == "feces":
        cum = simulation_tuple.feces_array
        daily = cumulative_to_daily(cum)
        if daily.size == 0:
            return np.array([], dtype=float)
        return predict_excretion_concentration(daily, animal, "feces", urine_volume_by_animal, feces_mass_by_animal, feces_mass_default)

    if comp == "elim":
        cum = simulation_tuple.elim_array
        daily = cumulative_to_daily(cum)
        return daily  # elim is mass rate, not concentration

    if comp not in m.PBTKModel.compartment_idx:
        return np.array([], dtype=float)
    idx = m.PBTKModel.compartment_idx[comp]
    vol_key = "V_mammary_deep" if comp == "mammary_deep" else f"V_{comp}"
    volume = all_params["physiological"].get(vol_key, 0)
    if volume <= 0:
        return np.full_like(pred_matrix[:, idx], np.nan, dtype=float)
    return np.maximum(pred_matrix[:, idx] / volume, 0.0)

def validate_predictions(y_pred: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Return a boolean mask for valid predictions (>0 and finite)."""
    y_pred = np.asarray(y_pred)
    return np.isfinite(y_pred) & (y_pred > 0)

def compute_censored_likelihood_loss(
    y_obs: Union[np.ndarray, List[float]], 
    y_pred: Union[np.ndarray, List[float]], 
    loq: float,
    sigma: float,
    config: ModelConfig
) -> Tuple[float, int]:
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    loq = float(loq)
    sigma = float(sigma)
    
    # Ensure sigma is positive
    if sigma <= 0:
        sigma = config.sigma_default
    
    valid = validate_predictions(y_pred) & np.isfinite(y_obs)
    if not np.any(valid):
        return 0.0, 0

    yo = y_obs[valid]
    yp = y_pred[valid]
    
    # Convert to log space (add eps to avoid log(0))
    # Note: Adding eps before log can introduce small bias, but necessary for numerical stability
    log_obs = np.log(np.maximum(yo, config.eps))
    log_pred = np.log(np.maximum(yp, config.eps))
    log_loq = np.log(np.maximum(loq, config.eps))

    # Identify uncensored and censored observations
    uncensored = yo > loq
    censored = ~uncensored
    
    # Initialize log-likelihood contributions
    log_likelihood_contributions = np.zeros_like(log_obs, dtype=float)
    
    if np.any(uncensored):
        log_obs_unc = log_obs[uncensored]
        log_pred_unc = log_pred[uncensored]
        residuals_unc = log_obs_unc - log_pred_unc
        
        log_likelihood_contributions[uncensored] = (
            -np.log(yo[uncensored] + config.eps)  # -log(y_obs) term
            - np.log(sigma)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * (residuals_unc ** 2) / (sigma ** 2)
        )
    
    if np.any(censored):
        log_pred_cens = log_pred[censored]
        # Standardized threshold: (log(LOQ) - log(y_pred)) / σ
        z_cens = (log_loq - log_pred_cens) / sigma
        
        # For numerical stability, clip z_cens to reasonable range
        z_cens = np.clip(z_cens, -10.0, 10.0)
        
        # Log of CDF: log(Φ(z))
        log_cdf = norm.logcdf(z_cens)
        
        # Handle numerical issues: if log_cdf is too small, use a small value
        # Using -50.0 as minimum to prevent underflow while maintaining numerical precision
        # This corresponds to a probability of ~1e-22, which is effectively zero
        log_cdf = np.maximum(log_cdf, -50.0)  # Prevent underflow
        
        log_likelihood_contributions[censored] = log_cdf
    
    # Sum of log-likelihoods (negative because we're minimizing)
    total_log_likelihood = np.sum(log_likelihood_contributions)
    neg_log_likelihood = -total_log_likelihood
    
    # Return negative log-likelihood (to be minimized) and number of observations
    return float(neg_log_likelihood), int(valid.sum())

def compute_nrmse_log_loss(
    y_obs: Union[np.ndarray, List[float]], 
    y_pred: Union[np.ndarray, List[float]], 
    loq: float,
    config: ModelConfig
) -> Tuple[float, int]:
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    loq = float(loq)

    valid = validate_predictions(y_pred) & np.isfinite(y_obs)
    if not np.any(valid):
        return 0.0, 0

    yo = y_obs[valid]
    yp = y_pred[valid]
    
    # Convert to log space (add eps to avoid log(0))
    log_obs = np.log(np.maximum(yo, config.eps))
    log_pred = np.log(np.maximum(yp, config.eps))
    log_loq = np.log(np.maximum(loq, config.eps))

    uncensored = yo > loq
    censored = ~uncensored

    residuals_sq = np.zeros_like(log_obs)
    if np.any(uncensored):
        r = log_obs[uncensored] - log_pred[uncensored]
        residuals_sq[uncensored] = r * r
    if np.any(censored):
        # For censored observations: if prediction > LOQ, use (log_LOQ - log_pred) as residual
        # If prediction <= LOQ, residual is 0 (prediction is consistent with censoring)
        over = yp[censored] > loq
        r_c = np.zeros_like(log_pred[censored])
        r_c[over] = (log_loq - log_pred[censored][over])
        residuals_sq[censored] = r_c * r_c

    nrmse_log = float(np.sqrt(np.mean(residuals_sq)))
    return nrmse_log, int(valid.sum())

def loss_function(
    log_fit_params: Union[np.ndarray, List[float]], 
    fit_config: FitConfig,
    use_data: pd.DataFrame,
    context: 'FittingContext',
    simulate_model_func: Callable
) -> float:
    total_loss = 0.0
    simulation_cache = {}
    failed_simulation = False

    animals = use_data['Animal'].unique()

    # Convert from log10-space to linear parameter space
    fit_params = []
    for lp in log_fit_params:
        if np.isnan(lp) or np.isinf(lp):
            return 1e6  # Quick rejection in log-space
        fit_params.append(10.0 ** lp)

    # Early parameter validation in linear space
    for p in fit_params:
        if np.isnan(p) or np.isinf(p) or p <= 0:
            return 1e6  # Quick rejection

    # Simulate model for all animals
    from optimization.config import SimulationConfig
    for animal in animals:
        sim_config = SimulationConfig(
            compound=fit_config.compound,
            isomer=fit_config.isomer,
            animal=animal
        )
        try:
            solution, all_params = simulate_model_func(fit_params, sim_config)
            simulation_cache[animal] = (solution, all_params)
        except Exception as e:
            logger.debug(f"Simulation failed for {sim_config} with params {fit_params}: {e}")
            simulation_cache[animal] = (None, None)
            failed_simulation = True

    # Compute log-space NRMSE loss for each matrix
    # Use context.config for matrix_weights, loq, etc.
    y_pred = None  # Track last y_pred for validation
    for matrix_name, weight in context.config.matrix_weights.items():
        df_matrix = use_data[use_data["Matrix"].str.lower() == matrix_name]
        if df_matrix.empty:
            continue

        t_obs = df_matrix['Day'].values
        y_obs = pd.to_numeric(df_matrix['Concentration'], errors='coerce').values
        animals_obs = df_matrix['Animal'].values

        y_pred = []
        for animal, t in zip(animals_obs, t_obs):
            sim, all_params = simulation_cache.get(animal, (None, None))
            if sim is None or all_params is None:
                y_pred.append(np.nan)
                continue
            # predict_single_observation handles day 0 → day 1 shift internally
            y_pred.append(predict_single_observation(
                sim, all_params, matrix_name, t, animal,
                context.urine_volume_by_animal,
                context.feces_mass_by_animal,
                context.feces_mass_default,
                context.milk_yield_by_animal,
                matrix_module=get_matrix_module(context.config),
            ))

        y_pred = np.array(y_pred)
        loq = context.config.loq_milk if matrix_name == "milk" else context.config.loq
        
        # Choose loss function based on config
        if context.config.use_log_rmse_for_fitting:
            # Use log RMSE for fitting (simpler, no sigma needed)
            comp_loss, n_used = compute_nrmse_log_loss(y_obs, y_pred, loq, context.config)
        else:
            # Use censored likelihood (requires sigma)
            # Get sigma, using estimated values if available for this compound-isomer pair
            sigma = context.config.get_sigma(
                matrix_name, 
                compound=fit_config.compound, 
                isomer=fit_config.isomer
            )
            comp_loss, n_used = compute_censored_likelihood_loss(
                y_obs, y_pred, loq, sigma, context.config
            )
        if n_used == 0:
            logger.debug(f"[LOSS] No valid data for {matrix_name}, skipping.")
            continue
        total_loss += weight * comp_loss

    # Penalty section
    if failed_simulation:
        logger.debug(f"[LOSS][FAILURE] Compound={fit_config.compound}, Isomer={fit_config.isomer} -- failed_simulation=True")
        return 1e6

    # Validate final loss value
    if not np.isfinite(total_loss):
        logger.warning(f"[LOSS] Non-finite loss: {total_loss} for {fit_config.compound} {fit_config.isomer}")
        return 1e6
    logger.debug(f"[LOSS] Total loss: {total_loss} for {fit_config.compound} {fit_config.isomer}")
    return total_loss

