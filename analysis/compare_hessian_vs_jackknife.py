"""
Compare Hessian-based parameter uncertainty to jackknife (LOAO) diagnostics.

For a given compound–isomer pair, this script:
  - loads Phase 1 global fit parameters,
  - loads jackknife LOAO fits (if available),
  - recomputes the numerical Hessian in log10-space for the fitted parameters,
  - derives the Hessian-based covariance matrix in log10-space, and
  - compares per-parameter log10 standard deviations:

      std_log (jackknife)  vs  std_log (Hessian)

Results are written to:

  results/analysis/hessian_vs_jackknife_<COMPOUND>_<ISOMER>.csv

This lets you check whether the Gaussian/Hessian approximation is defensible
relative to the empirical LOAO spread, especially for key parameters such as
`k_elim`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.config import setup_context, FittingContext, FitConfig  # type: ignore
from optimization.io import get_project_root  # type: ignore
from optimization.fit import simulate_model  # type: ignore
from optimization.mc import numerical_hessian  # type: ignore
from optimization.loss import loss_function  # type: ignore


logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Hessian-based parameter uncertainty to jackknife (LOAO) spread."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pair",
        nargs=2,
        metavar=("COMPOUND", "ISOMER"),
        help='Single compound–isomer pair (e.g. "PFOS Linear").',
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run comparison for all compound–isomer pairs with jackknife results.",
    )
    return parser.parse_args(argv)


def load_phase1_fit(context: FittingContext, compound: str, isomer: str) -> pd.DataFrame:
    fit_path = context.folder_phase1 / f"fit_{compound}_{isomer}.csv"
    if not fit_path.exists():
        raise FileNotFoundError(f"Phase 1 fit file not found: {fit_path}")
    df = pd.read_csv(fit_path)
    df = df.copy()
    df.loc[df["Parameter"] == "k_renal", "Parameter"] = "k_urine"
    return df


def load_jackknife(context: FittingContext, compound: str, isomer: str) -> pd.DataFrame:
    jk_path = context.folder_phase2 / f"jackknife_{compound}_{isomer}_LOAO.csv"
    if not jk_path.exists():
        raise FileNotFoundError(f"Jackknife file not found: {jk_path}")
    df = pd.read_csv(jk_path)
    if "k_renal" in df.columns and "k_urine" not in df.columns:
        df = df.rename(columns={"k_renal": "k_urine"})
    return df


def compute_hessian_cov_log(
    context: FittingContext,
    compound: str,
    isomer: str,
    fit_df: pd.DataFrame,
) -> Tuple[List[str], np.ndarray]:
    """Return (param_names_fit, cov_log) for fitted parameters in log10-space."""
    df_pair = context.data_cache.get_pair_data(compound, isomer)
    param_names_fit = context.config.get_param_names(compound, isomer, df_pair)
    fixed_params = context.config.get_fixed_parameters(compound, isomer, df_pair)

    if not param_names_fit:
        raise RuntimeError(f"No fitted parameters for {compound} {isomer}")

    eps_param = 1e-12
    phase1_fit_linear = np.array(
        [fit_df[fit_df["Parameter"] == n]["Value"].iloc[0] for n in param_names_fit],
        dtype=float,
    )
    x0 = np.log10(np.clip(phase1_fit_linear, eps_param, None)).astype(np.float64)
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

    H = numerical_hessian(loss_wrapper, x0, step=steps)
    ridge = 1e-6 * max(float(np.max(np.diag(H))), 1e-6)
    cov_log = np.linalg.pinv(H + ridge * np.eye(H.shape[0]))
    cov_log = 0.5 * (cov_log + cov_log.T)
    return param_names_fit, cov_log


def compare_for_pair(
    context: FittingContext,
    compound: str,
    isomer: str,
) -> pd.DataFrame:
    fit_df = load_phase1_fit(context, compound, isomer)
    jk_df = load_jackknife(context, compound, isomer)

    param_names_fit, cov_log = compute_hessian_cov_log(context, compound, isomer, fit_df)

    # Jackknife: log10-space std per fitted parameter
    jk_vals_linear = jk_df[param_names_fit].values.astype(float)
    eps_param = 1e-12
    jk_vals_log = np.log10(np.clip(jk_vals_linear, eps_param, None))
    jk_std_log = np.std(jk_vals_log, axis=0, ddof=1 if jk_vals_log.shape[0] > 1 else 0)

    # Hessian: log10-space std from covariance diagonal
    var_log = np.diag(cov_log)
    hess_std_log = np.sqrt(np.clip(var_log, 0.0, None))

    rows = []
    for name, s_jk, s_hess in zip(param_names_fit, jk_std_log, hess_std_log):
        ratio = (s_hess / s_jk) if s_jk > 0 else np.nan
        rows.append(
            {
                "Parameter": name,
                "Jackknife_std_log10": float(s_jk),
                "Hessian_std_log10": float(s_hess),
                "Hessian/Jackknife_ratio": float(ratio),
            }
        )

    df = pd.DataFrame(rows)
    df.insert(0, "Compound", compound)
    df.insert(1, "Isomer", isomer)
    return df


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    args = parse_args(argv)

    project_root = get_project_root()
    context = setup_context(project_root=project_root)
    out_dir = project_root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pair:
        compound, isomer = args.pair
        logger.info(
            "[HESSIAN VS JACKKNIFE] Comparing uncertainties for %s %s",
            compound,
            isomer,
        )

        try:
            df = compare_for_pair(context, compound, isomer)
        except FileNotFoundError as e:
            logger.error(
                "Required files missing for %s %s: %s. "
                "Make sure you have run the offline jackknife diagnostic first.",
                compound,
                isomer,
                e,
            )
            return 1
        except Exception as e:
            logger.error(
                "Failed to compare Hessian vs jackknife for %s %s: %s",
                compound,
                isomer,
                e,
                exc_info=True,
            )
            return 1

        out_path = out_dir / f"hessian_vs_jackknife_{compound}_{isomer}.csv"
        df.to_csv(out_path, index=False)

        logger.info("[HESSIAN VS JACKKNIFE] Saved comparison to %s", out_path)
        logger.info("\n%s", df.to_string(index=False))
        return 0

    # --all path: aggregate over all pairs that have jackknife results
    all_pairs: List[Tuple[str, str]] = context.data_cache.get_all_pairs()
    logger.info(
        "[HESSIAN VS JACKKNIFE] Running comparison for all pairs with jackknife files (%d total pairs)",
        len(all_pairs),
    )

    frames: List[pd.DataFrame] = []
    n_ok = 0
    for compound, isomer in all_pairs:
        try:
            df_pair = compare_for_pair(context, compound, isomer)
        except FileNotFoundError:
            # No jackknife file for this pair; skip silently
            continue
        except Exception as e:
            logger.warning(
                "Skipping %s %s due to error: %s",
                compound,
                isomer,
                e,
            )
            continue
        frames.append(df_pair)
        n_ok += 1

    if not frames:
        logger.error(
            "[HESSIAN VS JACKKNIFE] No pairs could be compared. "
            "Ensure jackknife diagnostics have been run."
        )
        return 1

    all_df = pd.concat(frames, ignore_index=True)
    out_all_path = out_dir / "hessian_vs_jackknife_all_pairs.csv"
    all_df.to_csv(out_all_path, index=False)

    logger.info(
        "[HESSIAN VS JACKKNIFE] Saved aggregated comparison for %d pairs to %s",
        n_ok,
        out_all_path,
    )

    # Brief group-wise summary by parameter
    summary = (
        all_df.groupby("Parameter")["Hessian/Jackknife_ratio"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    logger.info("Summary of Hessian/Jackknife std(log10) ratios by parameter:\n%s", summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())

