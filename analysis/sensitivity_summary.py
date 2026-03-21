"""
Summarise local sensitivity of milk PFAS concentrations to fitted parameters.

This script builds on the identifiability diagnostics based on the
Fisher Information Matrix (FIM). For a selected compound–isomer pair,
it converts the FIM into simple local sensitivity indices that rank
parameters by their relative influence on log(milk concentration).

The workflow is:
  1. Use the identifiability module to construct gradients of log‑predictions
     with respect to log‑parameters and compute the FIM.
  2. For each parameter θ_j, define a local importance index proportional to
         S_j = sqrt(F_jj)
     where F_jj is the j‑th diagonal element of the FIM.
  3. Normalize S_j to obtain:
         S_j_norm = S_j / sum_k S_k
  4. Save a tidy CSV with one row per parameter and optional plotting‑ready
     columns.

This is intentionally simple and local (around the fitted solution), and is
meant to provide a clear ranking plot for the paper rather than a full
global variance decomposition.

Outputs:
  - results/analysis/sensitivity_summary_<COMPOUND>_<ISOMER>.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.io import get_project_root  # type: ignore
from optimization.config import (  # type: ignore
    FitConfig,
    FittingContext,
    SimulationConfig,
    setup_context,
)
from optimization.fit import simulate_model  # type: ignore
from optimization.fit_variables import get_parameter_config  # type: ignore
from optimization.loss import predict_single_observation  # type: ignore
from auxiliary.project_paths import get_results_root  # type: ignore

logger = logging.getLogger(__name__)


def _load_fit_and_data(
    context: FittingContext,
    compound: str,
    isomer: str,
) -> Tuple[List[str], List[float], pd.DataFrame, Dict[str, float]]:
    """
    Load Phase 1 fit and data for a compound–isomer pair.
    Returns:
      param_names, fit_params_values (in model order), data_df, config_dict
    """
    fit_path = context.folder_phase1 / f"fit_{compound}_{isomer}.csv"
    if not fit_path.exists():
        raise FileNotFoundError(f"Fit file not found: {fit_path}")

    fit_df = pd.read_csv(fit_path)
    fit_params_dict = dict(zip(fit_df["Parameter"], fit_df["Value"]))

    data_df = context.data_cache.get_pair_data(compound, isomer)
    param_names, fixed_params = get_parameter_config(
        compound, isomer, data_df, config=context.config
    )
    if not param_names:
        raise RuntimeError(
        f"No fitted parameters found for {compound} {isomer}"
        )

    fit_params = [float(fit_params_dict.get(name, 0.0)) for name in param_names]

    config_dict = {
        "compound": compound,
        "isomer": isomer,
        "fixed_params": fixed_params,
    }
    return param_names, fit_params, data_df, config_dict


def _compute_gradients_and_fim(
    param_names: List[str],
    fit_params: List[float],
    data_df: pd.DataFrame,
    config_dict: Dict[str, Any],
    context: FittingContext,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Compute gradients (finite difference) and Fisher Information Matrix
    reusing the same definitions as the identifiability script.
    """
    compound = config_dict["compound"]
    isomer = config_dict["isomer"]
    fixed_params = config_dict["fixed_params"]

    fit_config = FitConfig(compound=compound, isomer=isomer)

    animals = data_df["Animal"].unique()
    simulation_cache: Dict[str, Tuple[Any, Any]] = {}

    # Baseline simulations
    for animal in animals:
        sim_cfg = SimulationConfig(compound=compound, isomer=isomer, animal=animal)
        try:
            solution, all_params = simulate_model(
                fit_params,
                sim_cfg,
                context,
                param_names=param_names,
                fixed_params=fixed_params,
            )
            simulation_cache[animal] = (solution, all_params)
        except Exception:
            simulation_cache[animal] = (None, None)

    baseline_preds: List[float] = []
    for _, row in data_df.iterrows():
        animal = row["Animal"]
        t = row["Day"]
        matrix_name = row["Matrix"].lower()
        sim, all_params = simulation_cache.get(animal, (None, None))
        if sim is None or all_params is None:
            baseline_preds.append(np.nan)
            continue
        pred = predict_single_observation(
            sim,
            all_params,
            matrix_name,
            t,
            animal,
            context.urine_volume_by_animal,
            context.feces_mass_by_animal,
            context.feces_mass_default,
            context.milk_yield_by_animal,
        )
        baseline_preds.append(pred)

    baseline_preds_arr = np.array(baseline_preds, dtype=float)
    valid_mask = np.isfinite(baseline_preds_arr) & (baseline_preds_arr > 0)
    baseline_log_preds = np.log(baseline_preds_arr + context.config.eps)

    n_params = len(param_names)
    n_obs = len(data_df)
    gradients = np.zeros((n_obs, n_params), dtype=float)

    for j in range(n_params):
        perturbed = fit_params.copy()
        step = epsilon * max(1.0, abs(perturbed[j]))
        perturbed[j] += step

        perturbed_cache: Dict[str, Tuple[Any, Any]] = {}
        for animal in animals:
            sim_cfg = SimulationConfig(
                compound=compound,
                isomer=isomer,
                animal=animal,
            )
            try:
                sol, all_params = simulate_model(
                    perturbed,
                    sim_cfg,
                    context,
                    param_names=param_names,
                    fixed_params=fixed_params,
                )
                perturbed_cache[animal] = (sol, all_params)
            except Exception:
                perturbed_cache[animal] = (None, None)

        perturbed_preds: List[float] = []
        for _, row in data_df.iterrows():
            animal = row["Animal"]
            t = row["Day"]
            matrix_name = row["Matrix"].lower()
            sim, all_params = perturbed_cache.get(animal, (None, None))
            if sim is None or all_params is None:
                perturbed_preds.append(np.nan)
                continue
            pred = predict_single_observation(
                sim,
                all_params,
                matrix_name,
                t,
                animal,
                context.urine_volume_by_animal,
                context.feces_mass_by_animal,
                context.feces_mass_default,
                context.milk_yield_by_animal,
            )
            perturbed_preds.append(pred)

        perturbed_arr = np.array(perturbed_preds, dtype=float)
        perturbed_log = np.log(perturbed_arr + context.config.eps)

        if abs(step) > 1e-10 and fit_params[j] > 0:
            linear_gradient = (perturbed_log - baseline_log_preds) / step
            log10_gradient = linear_gradient * fit_params[j] * np.log(10.0)
            gradients[:, j] = log10_gradient
        else:
            gradients[:, j] = 0.0

        gradients[~valid_mask, j] = 0.0

    # Build FIM with pooled sigma per matrix, as in identifiability.py
    fisher = np.zeros((n_params, n_params), dtype=float)
    for matrix_name in data_df["Matrix"].str.lower().unique():
        mask = data_df["Matrix"].str.lower() == matrix_name
        g_mat = gradients[mask.values, :]
        if g_mat.shape[0] == 0:
            continue
        sigma = context.config.get_sigma(matrix_name)
        sigma_sq = sigma ** 2
        fisher += (1.0 / sigma_sq) * np.dot(g_mat.T, g_mat)

    return fisher


def compute_sensitivity_summary_for_pair(
    compound: str,
    isomer: str,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """
    High‑level API: compute local sensitivity indices for a single pair.
    """
    project_root = get_project_root()
    context = setup_context(project_root=project_root)

    param_names, fit_params, data_df, cfg = _load_fit_and_data(
        context=context,
        compound=compound,
        isomer=isomer,
    )

    fisher = _compute_gradients_and_fim(
        param_names=param_names,
        fit_params=fit_params,
        data_df=data_df,
        config_dict=cfg,
        context=context,
        epsilon=epsilon,
    )

    diag = np.diag(fisher)
    diag = np.clip(diag, 0.0, None)
    S = np.sqrt(diag)

    total = float(np.sum(S))
    if total > 0:
        S_norm = S / total
    else:
        S_norm = np.zeros_like(S)

    df = pd.DataFrame(
        {
            "Parameter": param_names,
            "Fisher_Diagonal": diag,
            "Local_Sensitivity_S": S,
            "Local_Sensitivity_S_norm": S_norm,
        }
    )
    df = df.sort_values("Local_Sensitivity_S_norm", ascending=False).reset_index(
        drop=True
    )
    df.insert(0, "Compound", compound)
    df.insert(1, "Isomer", isomer)
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results_root = get_results_root()
    # Store all sensitivity outputs under a dedicated subfolder
    out_dir = results_root / "analysis" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine passing compound–isomer pairs from GOF summary, reusing the same
    # thresholds as other analysis scripts.
    gof_path = (
        results_root
        / "analysis"
        / "goodness_of_fit"
        / "goodness_of_fit_summary_by_compound.csv"
    )
    if not gof_path.exists():
        logger.error(
            "[SENSITIVITY] GOF summary file not found at %s. "
            "Run analysis/goodness_of_fit.py first.",
            gof_path,
        )
        return

    gof_df = pd.read_csv(gof_path)
    required_cols = {"Compound", "Isomer", "R2", "GM_Fold_Error", "Bias_log10"}
    missing = required_cols - set(gof_df.columns)
    if missing:
        logger.error(
            "[SENSITIVITY] GOF summary missing required columns: %s",
            sorted(missing),
        )
        return

    crit_r2 = gof_df["R2"] > 0.7
    crit_gm = gof_df["GM_Fold_Error"] < 3.0
    crit_bias = gof_df["Bias_log10"].abs() < 0.3
    passing = gof_df[crit_r2 & crit_gm & crit_bias][["Compound", "Isomer"]]

    if passing.empty:
        logger.warning(
            "[SENSITIVITY] No compound–isomer pairs passed GOF thresholds; nothing to do."
        )
        return

    pairs: list[tuple[str, str]] = [
        (str(row["Compound"]), str(row["Isomer"]))
        for _, row in passing.iterrows()
    ]

    generated_files: list[Path] = []

    for compound, isomer in pairs:
        out_path = out_dir / f"sensitivity_summary_{compound}_{isomer}.csv"
        logger.info(
            "[SENSITIVITY] Computing local sensitivity summary for %s %s",
            compound,
            isomer,
        )
        try:
            df = compute_sensitivity_summary_for_pair(compound, isomer)
        except FileNotFoundError as exc:
            logger.error(
                "[SENSITIVITY] Required fit/data files missing for %s %s: %s",
                compound,
                isomer,
                exc,
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "[SENSITIVITY] Failed to compute sensitivity summary for %s %s: %s",
                compound,
                isomer,
                exc,
            )
            continue

        df.to_csv(out_path, index=False)
        generated_files.append(out_path)
        logger.info(
            "[SENSITIVITY] Saved sensitivity summary for %s %s to %s",
            compound,
            isomer,
            out_path,
        )

    # Build combined heatmap table across passing compounds for convenience.
    if not generated_files:
        logger.warning(
            "[SENSITIVITY] No sensitivity summaries were generated; skipping heatmap table."
        )
        return

    frames: list[pd.DataFrame] = []
    for path in generated_files:
        try:
            df_pair = pd.read_csv(path)
        except Exception:
            continue
        if "Local_Sensitivity_S_norm" not in df_pair.columns:
            continue
        label = f"{df_pair['Compound'].iloc[0]} {df_pair['Isomer'].iloc[0]}"
        frames.append(
            df_pair[["Parameter", "Local_Sensitivity_S_norm"]].rename(
                columns={"Local_Sensitivity_S_norm": label}
            )
        )

    if not frames:
        logger.warning(
            "[SENSITIVITY] No frames available for heatmap table; nothing written."
        )
        return

    heatmap_df = frames[0]
    for extra in frames[1:]:
        heatmap_df = heatmap_df.merge(extra, on="Parameter", how="outer")

    heatmap_out = out_dir / "sensitivity_heatmap_passing_compounds.csv"
    heatmap_df.to_csv(heatmap_out, index=False)
    logger.info(
        "[SENSITIVITY] Saved combined sensitivity heatmap table to %s",
        heatmap_out,
    )


if __name__ == "__main__":
    main()

