"""Quick diagnostic script to test Hessian stability for all compound–isomer pairs.

This reuses the same loss function and parameter configuration as
`optimization.mc.run_monte_carlo_for_pair`, but *only* computes the numerical
Hessian and its implied covariance in log10-space. No Monte Carlo sampling or
ODE simulations are performed.

Outputs:
    results/analysis/hessian_stability.csv
        Columns:
            - Compound, Isomer
            - n_params_fit
            - success               (True/False)
            - exception             (string message if failed)
            - max_diag_H
            - min_diag_H
            - cond_H                (|λ_max(H)| / max(|λ_min(H)|, eps))
            - min_eig_cov
            - max_eig_cov
            - cond_cov              (max_eig_cov / max(min_eig_cov, eps))
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.config import (
    FittingContext,
    FitConfig,
    setup_context,
)
from optimization.io import get_project_root
from optimization.loss import loss_function
from optimization.fit import simulate_model  # type: ignore
from optimization.mc import numerical_hessian


logger = logging.getLogger(__name__)


def compute_hessian_for_pair(
    compound: str,
    isomer: str,
    context: FittingContext,
) -> dict:
    """Compute numerical Hessian and covariance diagnostics for a single pair.

    Returns a dict with diagnostics and success/failure status.
    """
    record: dict = {
        "Compound": compound,
        "Isomer": isomer,
        "n_params_fit": 0,
        "success": False,
        "exception": "",
        "max_diag_H": np.nan,
        "min_diag_H": np.nan,
        "cond_H": np.nan,
        "min_eig_cov": np.nan,
        "max_eig_cov": np.nan,
        "cond_cov": np.nan,
    }

    try:
        df_pair = context.data_cache.get_pair_data(compound, isomer)
        param_names_fit = context.config.get_param_names(compound, isomer, df_pair)
        fixed_params = context.config.get_fixed_parameters(compound, isomer, df_pair)

        if not param_names_fit:
            record["exception"] = "No parameters selected for fitting"
            return record

        record["n_params_fit"] = len(param_names_fit)

        # Load Phase 1 global fit (same as in optimization.mc.run_monte_carlo_for_pair)
        mean_path = context.folder_phase1 / f"fit_{compound}_{isomer}.csv"
        if not mean_path.exists():
            record["exception"] = f"Phase 1 file not found: {mean_path}"
            return record

        mean_df = pd.read_csv(mean_path)
        mean_df = mean_df.copy()
        # Backward compatibility: k_renal -> k_urine
        mean_df.loc[mean_df["Parameter"] == "k_renal", "Parameter"] = "k_urine"

        # Extract Phase 1 point estimates for the *fitted* parameters
        try:
            phase1_fit_linear = np.array(
                [mean_df[mean_df["Parameter"] == n]["Value"].iloc[0] for n in param_names_fit],
                dtype=float,
            )
        except Exception as e:
            record["exception"] = f"Failed to read Phase 1 parameters: {e}"
            return record

        eps_param = 1e-12
        x0 = np.log10(np.clip(phase1_fit_linear, eps_param, None)).astype(np.float64)

        # Log-space bounds and finite-difference step sizes
        log_bounds = context.config.get_log_bounds(param_names_fit)
        steps = np.array([(hi - lo) * 1e-2 for (lo, hi) in log_bounds], dtype=float)

        fit_config = FitConfig(compound=compound, isomer=isomer)

        def loss_wrapper(log_fit_params: np.ndarray) -> float:
            return float(
                loss_function(
                    log_fit_params,
                    fit_config=fit_config,
                    use_data=df_pair,
                    context=context,
                    simulate_model_func=lambda p, sim_cfg: simulate_model(
                        p,
                        sim_cfg,
                        context,
                        param_names=param_names_fit,
                        fixed_params=fixed_params,
                    ),
                )
            )

        # Numerical Hessian in log10-space
        H = numerical_hessian(loss_wrapper, x0, step=steps)

        # Basic Hessian diagnostics
        diag_H = np.diag(H)
        max_diag = float(np.max(diag_H))
        min_diag = float(np.min(diag_H))

        # Condition number of H using eigenvalues (magnitude-based)
        evals_H = np.linalg.eigvalsh(H)
        max_eig_H = float(np.max(np.abs(evals_H)))
        min_eig_H = float(np.min(np.abs(evals_H)))
        eps = 1e-12
        cond_H = max_eig_H / max(min_eig_H, eps)

        # Ridge-regularized covariance in log10-space (same as MC code)
        ridge = 1e-6 * max(max_diag, 1e-6)
        cov_log = np.linalg.pinv(H + ridge * np.eye(H.shape[0]))
        cov_log = 0.5 * (cov_log + cov_log.T)
        evals_cov, _ = np.linalg.eigh(cov_log)
        evals_cov = np.maximum(evals_cov, 1e-12)
        min_eig_cov = float(np.min(evals_cov))
        max_eig_cov = float(np.max(evals_cov))
        cond_cov = max_eig_cov / max(min_eig_cov, eps)

        record.update(
            {
                "success": True,
                "max_diag_H": max_diag,
                "min_diag_H": min_diag,
                "cond_H": cond_H,
                "min_eig_cov": min_eig_cov,
                "max_eig_cov": max_eig_cov,
                "cond_cov": cond_cov,
            }
        )
        return record

    except Exception as e:
        record["exception"] = str(e)
        return record


def main(pairs: Optional[List[Tuple[str, str]]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    project_root = get_project_root()
    context = setup_context(project_root=project_root)

    if pairs is None:
        pairs = context.data_cache.get_all_pairs()

    logger.info(f"Testing Hessian stability for {len(pairs)} compound–isomer pairs")

    results: List[dict] = []
    for compound, isomer in pairs:
        logger.info(f"Computing Hessian diagnostics for {compound} {isomer}...")
        rec = compute_hessian_for_pair(compound, isomer, context)
        status = "OK" if rec["success"] else f"FAIL ({rec['exception']})"
        logger.info(
            f"  -> {status}; n_params_fit={rec['n_params_fit']}, "
            f"cond_H={rec['cond_H']:.2e} cond_cov={rec['cond_cov']:.2e}"
            if rec["success"]
            else f"  -> {status}"
        )
        results.append(rec)

    df = pd.DataFrame(results)
    out_dir = project_root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hessian_stability.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved Hessian diagnostics to {out_path}")


if __name__ == "__main__":
    main()

