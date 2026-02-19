"""CLI entry point for model fitting and Monte Carlo sampling."""
import sys
import logging
import time
import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Any
from functools import partial
from tqdm import tqdm

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathos.multiprocessing import ProcessPool
import numpy as np

from optimization.config import ModelConfig, FittingContext, setup_context
from optimization.io import get_project_root
from optimization.fit import run_global_fit_for_pair, run_jackknife_for_pair
from optimization.mc import run_monte_carlo_for_pair

logger = logging.getLogger(__name__)

# Constants
TOTAL_PHASES_FULL_RUN = 3
TOTAL_PHASES_MC_ONLY = 1

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130  # Standard Unix exit code for Ctrl+C

def count_successful(results: List[Any]) -> int:
    """Count the number of successful (non-None) results."""
    return sum(1 for r in results if r is not None)

def update_progress(pbar: Optional[tqdm], n: int) -> None:
    """Helper to safely update progress bar."""
    if pbar is not None:
        pbar.update(n)

def main() -> int:
    """
    Main entry point for PBPK model fitting and Monte Carlo sampling pipeline.
    
    This function orchestrates a 3-phase computational workflow:
    
    Phase 1: Global Mean Fits
        - Fits PBPK model parameters using all available data
        - Determines optimal parameter values for each compound-isomer pair
        - Results saved to results/Phase1_GlobalFits/
    
    Phase 2: Jackknife Fits  
        - Performs leave-one-out cross-validation
        - Estimates parameter uncertainty via resampling
        - Results saved to results/Phase2_JackknifeFits/
    
    Phase 3: Monte Carlo Sampling
        - Generates prediction distributions using parameter uncertainty
        - Produces probabilistic forecasts for each compound
        - Results saved to results/Phase3_MonteCarloPredictions/
    
    Command-line Arguments:
        --mc-only: Skip phases 1-2, run only Monte Carlo (requires existing fits)
        --pair COMPOUND ISOMER: Process single compound-isomer pair instead of all
        --max-procs N: Limit parallel processes (default: CPU count)
        --n-mc N: Override Monte Carlo sample count (must be positive)
    
    Returns:
        int: Exit code (0=success, 1=error, 130=user interrupted)
    
    Examples:
        Run full pipeline:
            python run.py
        
        Run only Monte Carlo for one compound:
            python run.py --mc-only --pair PFBA Linear
        
        Override MC samples with limited parallelism:
            python run.py --n-mc 5000 --max-procs 4
    
    Raises:
        SystemExit: On validation errors, with appropriate exit code
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run PBPK model fitting with Monte Carlo sampling')
    parser.add_argument('--mc-only', action='store_true', 
                       help='Run only phase 3 (Monte Carlo sampling), skipping phases 1 and 2')
    parser.add_argument('--pair', nargs=2, metavar=('COMPOUND','ISOMER'),
                       help='Run only a single compound-isomer pair (e.g., PFBA Linear)')
    parser.add_argument('--max-procs', type=int, default=None,
                       help='Max parallel processes for outer pool (default: CPU count or number of pairs)')
    parser.add_argument('--n-mc', type=int, default=None,
                       help='Override number of Monte Carlo samples')
    args = parser.parse_args()
    
    # Setup basic console logging first (before file logging setup)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Get project root and setup full logging including file handler
    project_root = None
    try:
        project_root = get_project_root()
        log_dir = project_root / "results" / "Logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Date-time first so logs sort chronologically by filename (ISO-like: YYYY-MM-DD_HH-MM-SS)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_dir / f"{timestamp}_monte_carlo_run.log"
        
        # Add file handler to root logger
        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
    except Exception as e:
        logging.error(f"Failed to setup file logging: {e}")
        # Continue with console logging only
    
    # Setup context (loads all data, creates directories, etc.)
    try:
        context = setup_context(project_root=project_root)
    except Exception as e:
        logger.error(f"Failed to setup context: {e}", exc_info=True)
        return EXIT_ERROR
    compound_isomer_pairs = context.data_cache.get_all_pairs()
    
    # Print all pairs being processed
    logger.info(f"\nAll compound-isomer pairs to process ({len(compound_isomer_pairs)} total):")
    for compound, isomer in sorted(compound_isomer_pairs):
        logger.info(f"  {compound:12} | {isomer:10}")
    logger.info("")
    
    # Determine pairs to run
    if args.pair:
        selected_pair = tuple(args.pair)
        if selected_pair not in compound_isomer_pairs:
            logger.error(f"Pair {args.pair} not found in available pairs")
            logger.error(f"Available pairs: {sorted(compound_isomer_pairs)}")
            return EXIT_ERROR
        selected_pairs: List[Tuple[str, str]] = [selected_pair]
    else:
        selected_pairs = compound_isomer_pairs
    
    # Validate and determine number of Monte Carlo samples
    n_mc_samples = context.config.n_mc_samples  # Default value
    if args.n_mc is not None:
        if args.n_mc <= 0:
            logger.error(f"--n-mc must be positive, got {args.n_mc}")
            return EXIT_ERROR
        n_mc_samples = int(args.n_mc)
        logger.info(f"Overriding n_mc_samples to {n_mc_samples}")
    
    # Overall progress tracking
    if args.mc_only:
        total_phases = TOTAL_PHASES_MC_ONLY
        current_phase = 0
        logger.info(f"=== STARTING MONTE CARLO-ONLY RUN ===")
        logger.info(f"Total compounds to process: {len(selected_pairs)}")
        logger.info(f"Note: Skipping phases 1 and 2, assuming global fits and jackknife results exist")
    else:
        total_phases = TOTAL_PHASES_FULL_RUN
        current_phase = 0
        logger.info(f"=== STARTING FULL MONTE CARLO RUN ===")
        logger.info(f"Total compounds to process: {len(selected_pairs)}")
    
    overall_start_time = time.time()
    
    # Initialize timing and success tracking variables
    mean_time = 0.0
    jack_time = 0.0
    mc_time = 0.0
    successful_mean = 0
    successful_jack = 0
    successful_mc = 0
    results_mean: List[Optional[List[float]]] = []
    results_jackknife: List[Optional[np.ndarray]] = []
    results_mc: List[Optional[Path]] = []
    
    # Setup multiprocessing pool with proper resource management
    max_nodes = len(selected_pairs)
    if args.max_procs is not None and args.max_procs > 0:
        max_nodes = min(max_nodes, int(args.max_procs))
    else:
        max_nodes = min(max_nodes, multiprocessing.cpu_count())
    
    pool = ProcessPool(nodes=max_nodes)
    overall_pbar: Optional[tqdm] = None
    
    try:
        if not args.mc_only:
            overall_pbar = tqdm(total=len(selected_pairs) * total_phases, desc="Overall Progress", unit="task")
        
            # ------------------------------------------------------------------
            # PHASE 1: Global fits (two-stage)
            #   Stage 1: log-RMSE fits to get reasonable means
            #   Sigma estimation: pooled σ per matrix from stage-1 residuals
            #   Stage 2: censored-likelihood (Tobit) fits using those σ
            # ------------------------------------------------------------------
            logger.info("=== PHASE 1: Running global mean fits (two-stage: log-RMSE then censored likelihood) ===")
            start_time = time.time()
            
            # Create partial function with context
            run_global_partial = partial(run_global_fit_for_pair, context=context)

            # --- Stage 1: log-RMSE global fits ---
            logger.info("[PHASE 1 / Stage 1] Log-RMSE global fits (no sigma needed)")
            # Stage 1 always uses log-RMSE, independent of the global default.
            context.config.use_log_rmse_for_fitting = True
            try:
                _ = pool.map(run_global_partial, selected_pairs)
            except KeyboardInterrupt:
                logger.warning("User interrupted during Stage 1 (log-RMSE) global fit phase")
                raise  # Re-raise to ensure finally block executes
            except Exception as e:
                logger.error(f"Stage 1 (log-RMSE) global fit phase failed: {e}", exc_info=True)
                raise  # Re-raise to ensure finally block executes

            # --- Sigma estimation from stage-1 fits ---
            logger.info("=== Estimating pooled sigma from Stage 1 fits ===")
            try:
                from optimization.sigma_estimation import estimate_and_save_pooled_sigma
                sigma_estimates = estimate_and_save_pooled_sigma(context, pairs=selected_pairs)
                if sigma_estimates:
                    logger.info("Pooled sigma estimates computed successfully")
                    for matrix, sigma in sigma_estimates.items():
                        logger.info(f"  {matrix}: σ={sigma:.3f}")
                else:
                    logger.warning("Pooled sigma estimation returned None")
            except Exception as e:
                logger.error(f"Pooled sigma estimation failed: {e}", exc_info=True)
                # Don't fail the entire run if sigma estimation fails; fall back to defaults

            # --- Stage 2: censored-likelihood (Tobit) global fits ---
            logger.info("[PHASE 1 / Stage 2] Censored-likelihood (Tobit) global fits using estimated sigma")
            # Stage 2 and all subsequent fits (jackknife, Hessian, etc.) use censored likelihood.
            context.config.use_log_rmse_for_fitting = False
            try:
                results_mean = pool.map(run_global_partial, selected_pairs)
            except KeyboardInterrupt:
                logger.warning("User interrupted during Stage 2 (Tobit) global fit phase")
                raise  # Re-raise to ensure finally block executes
            except Exception as e:
                logger.error(f"Stage 2 (Tobit) global fit phase failed: {e}", exc_info=True)
                raise  # Re-raise to ensure finally block executes
            
            update_progress(overall_pbar, len(selected_pairs))
            mean_time = time.time() - start_time
            current_phase += 1
            successful_mean = count_successful(results_mean)
            logger.info(f"=== PHASE {current_phase}/{total_phases} COMPLETED ===")
            logger.info(f"Global mean fits (two-stage): {successful_mean}/{len(selected_pairs)} successful in {mean_time:.1f}s")
            
            # ------------------------------------------------------------------
            # PHASE 2: Jackknife fits (use censored-likelihood by default)
            # ------------------------------------------------------------------
            logger.info("=== PHASE 2: Running jackknife fits ===")
            start_time = time.time()
            
            # Create partial function with context
            run_jackknife_partial = partial(run_jackknife_for_pair, context=context)
            try:
                results_jackknife = pool.map(run_jackknife_partial, selected_pairs)
            except KeyboardInterrupt:
                logger.warning("User interrupted during jackknife fit phase")
                raise  # Re-raise to ensure finally block executes
            except Exception as e:
                logger.error(f"Jackknife fit phase failed: {e}", exc_info=True)
                raise  # Re-raise to ensure finally block executes
            
            update_progress(overall_pbar, len(selected_pairs))
            jack_time = time.time() - start_time
            current_phase += 1
            successful_jack = count_successful(results_jackknife)
            logger.info(f"=== PHASE {current_phase}/{total_phases} COMPLETED ===")
            logger.info(f"Jackknife fits: {successful_jack}/{len(selected_pairs)} successful in {jack_time:.1f}s")

        phase_label = "PHASE 1" if args.mc_only else "PHASE 3"
        logger.info(f"=== {phase_label}: Running Monte Carlo sampling ===")
        start_time = time.time()
        
        # Create partial function with context and n_mc_samples
        run_mc_partial = partial(
            run_monte_carlo_for_pair,
            n_mc_samples=n_mc_samples,
            context=context
        )
        
        try:
            results_mc = pool.map(run_mc_partial, selected_pairs)
        except KeyboardInterrupt:
            logger.warning("User interrupted during Monte Carlo phase")
            raise  # Re-raise to ensure finally block executes
        except Exception as e:
            logger.error(f"Monte Carlo phase failed: {e}", exc_info=True)
            raise  # Re-raise to ensure finally block executes
        
        update_progress(overall_pbar, len(selected_pairs))
        
        mc_time = time.time() - start_time
        current_phase += 1
        successful_mc = count_successful(results_mc)
        logger.info(f"=== PHASE {current_phase}/{total_phases} COMPLETED ===")
        logger.info(f"Monte Carlo sampling: {successful_mc}/{len(selected_pairs)} successful in {mc_time:.1f}s")
    
    finally:
        # Ensure progress bar is closed
        if overall_pbar is not None:
            overall_pbar.close()
        
        # Ensure pool is properly closed and joined
        pool.close()
        pool.join()
    
    elapsed_total = time.time() - overall_start_time
    
    # Final summary
    if args.mc_only:
        logger.info(f"=== MONTE CARLO-ONLY RUN COMPLETED ===")
        logger.info(f"Total execution time: {elapsed_total:.1f}s ({elapsed_total/3600:.1f} hours)")
        logger.info(f"Success rate: Monte Carlo={successful_mc}/{len(selected_pairs)}")
        if successful_mc > 0:
            logger.info(f"Average Monte Carlo time: {mc_time / successful_mc:.1f}s per compound")
    else:
        logger.info(f"=== ALL PHASES COMPLETED ===")
        logger.info(f"Total execution time: {elapsed_total:.1f}s ({elapsed_total/3600:.1f} hours)")
        logger.info(f"Success rates: Global={successful_mean}/{len(selected_pairs)}, Jackknife={successful_jack}/{len(selected_pairs)}, Monte Carlo={successful_mc}/{len(selected_pairs)}")
        if successful_mean > 0:
            logger.info(f"Average global fit time: {mean_time / successful_mean:.1f}s per compound")
        if successful_jack > 0:
            logger.info(f"Average jackknife time: {jack_time / successful_jack:.1f}s per compound")
        if successful_mc > 0:
            logger.info(f"Average Monte Carlo time: {mc_time / successful_mc:.1f}s per compound")
        total_successful = min(successful_mean, successful_jack, successful_mc)
        logger.info(f"Fully successful compounds: {total_successful}/{len(selected_pairs)}")
    
    return EXIT_SUCCESS

if __name__ == "__main__":
    # Module-level exception handler ensures cleanup even if main() fails early
    # Inner handlers in main() re-raise to ensure finally blocks execute properly
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        # Use logging module directly in case logger not yet configured
        logging.warning("Interrupted by user")
        sys.exit(EXIT_INTERRUPTED)
    except Exception as e:
        # Use logging module directly in case logger not yet configured
        logging.exception(f"Unexpected error in main: {e}")
        sys.exit(EXIT_ERROR)