"""Model simulation and fitting functions."""
import sys
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Callable, Union, TYPE_CHECKING
from functools import partial
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tqdm import tqdm

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.solve import SimulationResult  # Named tuple used for type hints
from optimization.config import ModelConfig, SimulationConfig, FitConfig, get_matrix_module, get_parameters_module

if TYPE_CHECKING:
    from optimization.config import FittingContext
    from optimization.loss import loss_function

logger = logging.getLogger(__name__)

# Load body weight time series per animal from interpolated CSV (clean data layout)
BODY_WEIGHTS_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "body_weight_interpolated.csv"
_bw_df = pd.read_csv(BODY_WEIGHTS_CSV)
BODY_WEIGHT_SERIES: Dict[str, np.ndarray] = {
    col: _bw_df[col].to_numpy(dtype=float)
    for col in _bw_df.columns
    if col.lower() != "date"
}

def build_intake_function(
    sim_config: SimulationConfig,
    intake_df: pd.DataFrame
) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    sub = intake_df[
        (intake_df["Animal"] == sim_config.animal) &
        (intake_df["Compound"] == sim_config.compound) &
        (intake_df["Isomer"] == sim_config.isomer)
    ]
    if sub.empty:
        return lambda t: np.zeros_like(np.atleast_1d(t), dtype=float) if np.asarray(t).size > 1 else 0.0
    
    day_to_intake = dict(zip(sub["Day"].astype(int), sub["PFAS_Intake_ug_day"].astype(float)))
    
    def intake_func(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.atleast_1d(t)
        out = np.array([day_to_intake.get(int(day), 0.0) for day in t_arr])
        return out if t_arr.size > 1 else float(out[0])
    
    return intake_func

def simulate_model(
    fit_params: List[float], 
    sim_config: SimulationConfig,
    context: 'FittingContext',
    param_names: Optional[List[str]] = None,
    fixed_params: Optional[Dict[str, float]] = None
) -> Tuple[SimulationResult, Dict[str, Any]]:
    # Validate parameters
    if any(np.isnan(p) or np.isinf(p) or p <= 0 for p in fit_params):
        raise ValueError(f"Invalid fit parameters: {fit_params}")
    
    # Get parameter names (use provided or default from config)
    if param_names is None:
        param_names = context.config.param_names
    
    # Build parameters
    # Create dictionary from fitted parameters
    fit_params_dict = dict(zip(param_names, fit_params))
    
    # Add fixed parameters (these override fitted ones if there's a conflict)
    if fixed_params:
        fit_params_dict.update(fixed_params)

    matrix_mod = get_matrix_module(context.config)
    params_mod = get_parameters_module(context.config)
    
    all_params = params_mod.build_parameters(config=sim_config.to_dict(), fit_params=fit_params_dict)
    
    # Build intake function
    intake_function = build_intake_function(sim_config, context.intake_df)

    # Use full body weight time series from CSV for this animal, if available
    body_weight_array = BODY_WEIGHT_SERIES.get(sim_config.animal)
    
    milk_yield_array = None
    if sim_config.animal in context.milk_yield_by_animal:
        milk_yield_array = context.milk_yield_by_animal[sim_config.animal]
    
    # Use unified physiology_provider with measured body weight and milk yield data
    physiology_provider = params_mod.build_dynamic_physiology_provider(
        time_unit="days",
        body_weight_array=body_weight_array,
        milk_yield_array=milk_yield_array
    )
    
    # Run simulation (full or simple model from config)
    model = matrix_mod.PBTKModel(
        params=all_params,
        intake_function=intake_function,
        physiology_provider=physiology_provider
    )
    A0 = np.zeros(len(model.compartment_idx))
    solution = model.simulate_over_time(A0, t_array=context.config.time_vector)
    
    # Validate solution
    if solution is None or len(solution) == 0:
        raise ValueError("Model simulation returned None or empty solution")
    
    for i, sol in enumerate(solution):
        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            raise ValueError(f"Model solution contains NaN or infinite values at index {i}")
    
    return solution, all_params

def run_global_fit_for_pair(
    pair: Tuple[str, str],
    context: 'FittingContext'
) -> Optional[List[float]]:
    from optimization.loss import loss_function
    
    compound, isomer = pair
    start_time = time.time()
    logger.info(f"[GLOBAL FIT] Starting {compound} {isomer}")
    
    # Expected failure: no data for this compound-isomer pair
    df_pair = context.data_cache.get_pair_data(compound, isomer)
    if df_pair.empty:
        logger.warning(f"[GLOBAL FIT] No usable data for {compound} {isomer}. Skipping...")
        return None
    
    total_points = len(df_pair)
    animals = df_pair['Animal'].unique()
    from optimization.fit_variables import get_matrices_above_loq
    matrices_above_loq = get_matrices_above_loq(df_pair, context.config.loq, context.config.loq_milk)
    logger.info(f"[GLOBAL FIT] {compound} {isomer} - Data: {total_points} points, {len(matrices_above_loq)} matrices above LOQ, {len(animals)} animals")

    try:
        fit_config = FitConfig(compound=compound, isomer=isomer)
        
        # Get dynamic parameters for this compound-isomer pair
        param_names = context.config.get_param_names(compound, isomer, df_pair)
        fixed_params = context.config.get_fixed_parameters(compound, isomer, df_pair)
        
        logger.info(f"[GLOBAL FIT] {compound} {isomer} - Fitting parameters: {param_names}")
        if fixed_params:
            logger.info(f"[GLOBAL FIT] {compound} {isomer} - Fixed parameters: {fixed_params}")
        
        # Handle case where no parameters need to be fitted (no signals detected)
        if not param_names:
            logger.warning(f"[GLOBAL FIT] {compound} {isomer} - No parameters to fit (no signals detected). Saving fixed parameters only.")
            # Save fixed parameters only
            all_params_dict = fixed_params.copy() if fixed_params else {}
            
            # Save all parameters (fixed only)
            if all_params_dict:
                fit_df = pd.DataFrame({
                    'Parameter': list(all_params_dict.keys()),
                    'Value': list(all_params_dict.values()),
                    'Compound': compound,
                    'Isomer': isomer
                })
                fit_csv_path = context.folder_phase1 / f'fit_{compound}_{isomer}.csv'
                fit_df.to_csv(fit_csv_path, index=False)
                logger.info(f"[GLOBAL FIT] {compound} {isomer} - Saved fixed parameters to {fit_csv_path}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"[GLOBAL FIT] {compound} {isomer} - Completed in {elapsed_time:.1f}s (no fit needed)")
            return []
        
        log_bounds = context.config.get_log_bounds(param_names)
        
        # Create loss wrapper with context
        def loss_wrapper(log_fit_params):
            return loss_function(
                log_fit_params,
                fit_config=fit_config,
                use_data=df_pair,
                context=context,
                simulate_model_func=lambda p, sim_cfg: simulate_model(
                    p, sim_cfg, context, 
                    param_names=param_names, 
                    fixed_params=fixed_params
                )
            )
        
        result = differential_evolution(
            loss_wrapper,
            bounds=log_bounds,
            **context.config.de_config
        )
        # Convert log10-parameters to linear for storage/logging
        fit_params = [10.0 ** lp for lp in result.x]
        
        # Create full parameter dictionary including fixed parameters
        all_params_dict = dict(zip(param_names, fit_params))
        if fixed_params:
            all_params_dict.update(fixed_params)
        
        logger.info(f"[GLOBAL FIT] {compound} {isomer} - Success: loss={result.fun:.2f}, iterations={result.nit}")
        param_str = ", ".join([f"{name}={val:.3f}" for name, val in all_params_dict.items()])
        logger.info(f"[GLOBAL FIT] {compound} {isomer} - Parameters: {param_str}")
        
        # Save all parameters (fitted + fixed)
        fit_df = pd.DataFrame({
            'Parameter': list(all_params_dict.keys()),
            'Value': list(all_params_dict.values()),
            'Compound': compound,
            'Isomer': isomer
        })
        fit_csv_path = context.folder_phase1 / f'fit_{compound}_{isomer}.csv'
        fit_df.to_csv(fit_csv_path, index=False)
        logger.info(f"[GLOBAL FIT] {compound} {isomer} - Saved parameters to {fit_csv_path}")
        
        # Note: Sigma estimation is now done globally after all fits complete
        # See estimate_and_save_pooled_sigma() in sigma_estimation.py
        
        # Compute identifiability diagnostics (optional, can be disabled for speed)
        try:
            from Analysis.identifiability import compute_identifiability_diagnostics, save_identifiability_report
            logger.info(f"[GLOBAL FIT] {compound} {isomer} - Computing identifiability diagnostics...")
            diagnostics = compute_identifiability_diagnostics((compound, isomer), context)
            if diagnostics is not None:
                # Save diagnostics
                output_dir = context.folder_phase1.parent / "Identifiability"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f'identifiability_{compound}_{isomer}'
                save_identifiability_report(diagnostics, output_path)
                logger.info(f"[GLOBAL FIT] {compound} {isomer} - Identifiability diagnostics saved")
                
                # Log key findings
                cond_num = diagnostics['condition_number']
                if cond_num > 1e6:
                    logger.warning(f"[GLOBAL FIT] {compound} {isomer} - High condition number ({cond_num:.2e}), parameters may be poorly identifiable")
                if diagnostics['unidentifiable_directions']:
                    for unid in diagnostics['unidentifiable_directions']:
                        params_str = ', '.join(unid['parameters'])
                        logger.warning(f"[GLOBAL FIT] {compound} {isomer} - Unidentifiable combination: {params_str} (eigenvalue: {unid['eigenvalue']:.2e})")
        except Exception as e:
            # Don't fail the fit if identifiability diagnostics fail
            logger.debug(f"[GLOBAL FIT] {compound} {isomer} - Identifiability diagnostics failed: {e}", exc_info=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"[GLOBAL FIT] {compound} {isomer} - Completed in {elapsed_time:.1f}s")
        return fit_params
        
    except Exception as e:
        # Unexpected error: catch, log, and return None to allow parallel execution to continue
        elapsed_time = time.time() - start_time
        logger.error(f"[GLOBAL FIT] {compound} {isomer} - FAILED after {elapsed_time:.1f}s: {e}", exc_info=True)
        return None

def run_jackknife_for_pair(
    pair: Tuple[str, str],
    context: 'FittingContext'
) -> Optional[np.ndarray]:
    from optimization.loss import loss_function
    
    compound, isomer = pair
    start_time = time.time()
    logger.info(f"[JACKKNIFE] Starting {compound} {isomer}")
    
    df_pair = context.data_cache.get_pair_data(compound, isomer)
    animals = df_pair['Animal'].unique()
    fit_results = []
    
    try:
        # Get dynamic parameters for this compound-isomer pair (check once before loop)
        param_names = context.config.get_param_names(compound, isomer, df_pair)
        fixed_params = context.config.get_fixed_parameters(compound, isomer, df_pair)
        
        # Handle case where no parameters need to be fitted (no signals detected)
        if not param_names:
            logger.warning(f"[JACKKNIFE] {compound} {isomer} - No parameters to fit (no signals detected). Skipping jackknife.")
            # Return None to indicate skip
            elapsed_time = time.time() - start_time
            logger.info(f"[JACKKNIFE] {compound} {isomer} - Completed in {elapsed_time:.1f}s (no fit needed)")
            return None
        
        log_bounds = context.config.get_log_bounds(param_names)
        
        with tqdm(total=len(animals), desc=f"Jackknife {compound} {isomer}", unit="animal", leave=False) as pbar:
            for i, animal in enumerate(animals):
                animal_start = time.time()
                logger.info(f"[JACKKNIFE] {compound} {isomer} - Processing animal {animal} ({i+1}/{len(animals)})")
                
                jackknife_sample = df_pair[df_pair['Animal'] != animal]
                fit_config = FitConfig(compound=compound, isomer=isomer)
                
                # Create loss wrapper with context
                def loss_wrapper(log_fit_params):
                    return loss_function(
                        log_fit_params,
                        fit_config=fit_config,
                        use_data=jackknife_sample,
                        context=context,
                        simulate_model_func=lambda p, sim_cfg: simulate_model(
                            p, sim_cfg, context,
                            param_names=param_names,
                            fixed_params=fixed_params
                        )
                    )
                
                result = differential_evolution(
                    loss_wrapper,
                    bounds=log_bounds,
                    **context.config.de_config
                )
                
                # Convert to full parameter set (fitted + fixed)
                fitted_params = [10.0 ** lp for lp in result.x]
                all_params_dict = dict(zip(param_names, fitted_params))
                if fixed_params:
                    all_params_dict.update(fixed_params)
                
                # Store in order of default param_names for consistency
                fit_results.append([all_params_dict.get(p, 0.0) for p in context.config.param_names])
                
                animal_time = time.time() - animal_start
                logger.info(f"[JACKKNIFE] {compound} {isomer} - Animal {animal} completed in {animal_time:.1f}s (loss={result.fun:.2f})")
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{result.fun:.2f}", "Time": f"{animal_time:.1f}s"})
        
        fits = np.array(fit_results)
        jackknife_df = pd.DataFrame(fits, columns=context.config.param_names)
        jackknife_df['Compound'] = compound
        jackknife_df['Isomer'] = isomer
        jackknife_df['Animal_Excluded'] = animals
        cols = ['Compound', 'Isomer', 'Animal_Excluded'] + context.config.param_names
        jackknife_df = jackknife_df[cols]
        jackknife_csv_path = context.folder_phase2 / f'jackknife_{compound}_{isomer}_LOAO.csv'
        jackknife_df.to_csv(jackknife_csv_path, index=False)
        logger.info(f"[JACKKNIFE] {compound} {isomer} - Saved jackknife results to {jackknife_csv_path}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"[JACKKNIFE] {compound} {isomer} - Completed in {elapsed_time:.1f}s")
        return fits
        
    except Exception as e:
        # Unexpected error: catch, log, and return None to allow parallel execution to continue
        elapsed_time = time.time() - start_time
        logger.error(f"[JACKKNIFE] {compound} {isomer} - FAILED after {elapsed_time:.1f}s: {e}", exc_info=True)
        return None