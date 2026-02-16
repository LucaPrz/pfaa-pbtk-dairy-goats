"""
Goodness-of-fit diagnostics for the PFAS PBTK model using Monte Carlo predictions.

This script:
  1. Loads Phase 3 Monte Carlo predictions.
  2. Loads experimental observations.
  3. Matches predictions to observations by compound, isomer, compartment, and time
     (with optional Day-0 → Day-1 shift).
  4. Computes per-compound and per-compartment goodness-of-fit metrics:
     - R² (log10 scale)
     - geometric mean fold error
     - bias in log10 and fold space
     - CI coverage (if prediction intervals are available)
     - relative RMSE (per-compartment)
  5. Writes summary CSV tables under `results/analysis/goodness_of_fit/`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import logging
from scipy.stats import linregress

# Small value to avoid log(0)
EPS = 1e-6
LOQ = 0.5
LOQ_MILK = 0.005

# Ensure project root is on sys.path when executed via absolute path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_data_root, get_results_root

logger = logging.getLogger(__name__)


def load_predictions(predictions_dir: Path) -> pd.DataFrame:
    """Load all prediction files from results/optimization/monte_carlo."""
    predictions: List[pd.DataFrame] = []
    for pred_file in predictions_dir.glob("predictions_*_monte_carlo.csv"):
        df = pd.read_csv(pred_file)
        predictions.append(df)

    if not predictions:
        raise ValueError(f"No prediction files found in {predictions_dir}")

    return pd.concat(predictions, ignore_index=True)


def load_observations(data_path: Path) -> pd.DataFrame:
    """Load observation data from the clean PFAS data CSV."""
    df = pd.read_csv(data_path)
    # Convert Concentration to numeric, coercing errors to NaN
    df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")
    # Filter out NaN and zero concentrations (likely below LOQ)
    df = df[df["Concentration"].notna() & (df["Concentration"] > 0)].copy()
    return df


def match_predictions_observations(
    pred_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    shift_day0_to_day1: bool = True,
) -> pd.DataFrame:
    """
    Match predictions with observations by Compound, Isomer, Matrix/Compartment, and Day/Time.

    If shift_day0_to_day1 is True, day 0 observations are matched with day 1 predictions
    (assuming day 0 observations are taken at the end of the first exposure day).
    """
    # Map observation matrix names to prediction compartment names
    matrix_map = {
        "Plasma": "plasma",
        "Milk": "milk",
        "Liver": "liver",
        "Kidney": "kidney",
        "Muscle": "muscle",
        "Heart": "heart",
        "Brain": "brain",
        "Spleen": "spleen",
        "Lung": "lung",
        "Urine": "urine",
        "Feces": "feces",
    }

    obs_df = obs_df.copy()
    obs_df["Compartment"] = obs_df["Matrix"].map(matrix_map)
    obs_df = obs_df.dropna(subset=["Compartment"])  # Remove unmapped matrices

    # Shift day 0 observations to match day 1 predictions if requested
    if shift_day0_to_day1:
        obs_df["Time_to_match"] = obs_df["Day"].apply(lambda d: 1 if d == 0 else d)
    else:
        obs_df["Time_to_match"] = obs_df["Day"]

    # Merge on Compound, Isomer, Compartment, and Day/Time
    merged = obs_df.merge(
        pred_df,
        left_on=["Compound", "Isomer", "Compartment", "Time_to_match"],
        right_on=["Compound", "Isomer", "Compartment", "Time"],
        how="inner",
        suffixes=("_obs", "_pred"),
    )

    # Keep original Day column for reference
    merged = merged.rename(columns={"Day": "Day_original"})
    merged["Day"] = merged["Day_original"]

    return merged


def compute_log_values(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log10 values for observations and predictions."""
    df = df.copy()
    df["obs_value"] = df["Concentration"]
    df["log_obs"] = np.log10(df["obs_value"] + EPS)

    df["pred_value"] = df["Pred_Median"]
    df["log_pred"] = np.log10(df["pred_value"] + EPS)

    return df


def get_functional_group(compound: str) -> str:
    """Extract functional group from compound name (PFCA or PFSA)."""
    if pd.isna(compound):
        return "Unknown"
    compound_str = str(compound).upper()
    if compound_str.endswith("A") or compound_str.endswith("ECHS"):
        return "PFCA"
    elif compound_str.endswith("S"):
        return "PFSA"
    else:
        return "Unknown"


def calculate_r2_and_rrmse(obs: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """Calculate R² and relative RMSE in log space."""
    valid_mask = np.isfinite(obs) & np.isfinite(pred)
    if valid_mask.sum() < 2:
        return np.nan, np.nan

    obs_valid = obs[valid_mask]
    pred_valid = pred[valid_mask]

    slope, intercept, r_value, p_value, std_err = linregress(obs_valid, pred_valid)
    r2 = float(r_value**2)

    residuals = pred_valid - obs_valid
    rmse = float(np.sqrt(np.mean(residuals**2)))
    obs_range = float(np.max(obs_valid) - np.min(obs_valid))
    if obs_range > 0:
        rrmse = rmse / obs_range
    else:
        rrmse = np.nan

    return r2, rrmse


def calculate_r2_rrmse_and_bias(
    obs: np.ndarray,
    pred: np.ndarray,
    obs_values: np.ndarray | None = None,
    pred_values: np.ndarray | None = None,
) -> Tuple[float, float, float]:
    """Calculate R², relative RMSE, and bias in log space."""
    valid_mask = np.isfinite(obs) & np.isfinite(pred)
    if valid_mask.sum() < 2:
        return np.nan, np.nan, np.nan

    obs_valid = obs[valid_mask]
    pred_valid = pred[valid_mask]

    slope, intercept, r_value, p_value, std_err = linregress(obs_valid, pred_valid)
    r2 = float(r_value**2)

    residuals = pred_valid - obs_valid
    rmse = float(np.sqrt(np.mean(residuals**2)))
    obs_range = float(np.max(obs_valid) - np.min(obs_valid))
    if obs_range > 0:
        rrmse = rmse / obs_range
    else:
        rrmse = np.nan

    if obs_values is not None and pred_values is not None:
        obs_vals = obs_values[valid_mask]
        pred_vals = pred_values[valid_mask]
        ratio = np.clip(pred_vals / obs_vals, EPS, None)
        log_ratio = np.log10(ratio)
        bias_log = float(np.mean(log_ratio))
    else:
        bias_log = float(np.mean(pred_valid - obs_valid))

    return r2, rrmse, bias_log


def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute R², fold error, bias and CI coverage per compound-isomer."""
    rows: List[Dict[str, any]] = []
    for (compound, isomer), g in df.groupby(["Compound", "Isomer"]):
        g = g.copy()
        valid_mask = g["log_obs"].notna() & g["log_pred"].notna()
        if valid_mask.sum() > 1:
            slope, intercept, r_value, p_value, std_err = linregress(
                g.loc[valid_mask, "log_obs"],
                g.loc[valid_mask, "log_pred"],
            )
            r2 = float(r_value**2)
        else:
            r2 = np.nan

        ratio = np.clip(g["pred_value"].values / g["obs_value"].values, EPS, None)
        log_ratio = np.log10(ratio)
        gm_fold_error = float(10.0 ** np.mean(np.abs(log_ratio)))

        bias_log = float(np.mean(log_ratio))
        bias_fold = float(10.0 ** abs(bias_log))

        if "CI_Lower" in g.columns and "CI_Upper" in g.columns:
            ci_low = pd.to_numeric(g["CI_Lower"], errors="coerce")
            ci_high = pd.to_numeric(g["CI_Upper"], errors="coerce")
            obs = g["obs_value"]
            ci_valid = ci_low.notna() & ci_high.notna() & obs.notna()
            coverage = (
                float(
                    (
                        (obs[ci_valid] >= ci_low[ci_valid])
                        & (obs[ci_valid] <= ci_high[ci_valid])
                    ).mean()
                )
                if ci_valid.any()
                else np.nan
            )
        else:
            coverage = np.nan

        rows.append(
            {
                "Compound": compound,
                "Isomer": isomer,
                "N": int(len(g)),
                "R2": r2,
                "GM_Fold_Error": gm_fold_error,
                "Bias_log10": bias_log,
                "Bias_fold": bias_fold,
                "CI_Coverage": coverage,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["Compound", "Isomer"]).reset_index(drop=True)
    return summary_df


def compute_compartment_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute R², fold error, bias and CI coverage per compound–isomer–compartment."""
    rows: List[Dict[str, any]] = []
    for (compound, isomer, compartment), g in df.groupby(
        ["Compound", "Isomer", "Compartment"]
    ):
        g = g.copy()
        valid_mask = g["log_obs"].notna() & g["log_pred"].notna()
        if valid_mask.sum() > 1:
            r2, rrmse, bias_log = calculate_r2_rrmse_and_bias(
                g.loc[valid_mask, "log_obs"].values,
                g.loc[valid_mask, "log_pred"].values,
                g.loc[valid_mask, "obs_value"].values,
                g.loc[valid_mask, "pred_value"].values,
            )
        else:
            r2, rrmse, bias_log = np.nan, np.nan, np.nan

        ratio = np.clip(g["pred_value"].values / g["obs_value"].values, EPS, None)
        log_ratio = np.log10(ratio)
        gm_fold_error = float(10.0 ** np.mean(np.abs(log_ratio)))

        if "CI_Lower" in g.columns and "CI_Upper" in g.columns:
            ci_low = pd.to_numeric(g["CI_Lower"], errors="coerce")
            ci_high = pd.to_numeric(g["CI_Upper"], errors="coerce")
            obs = g["obs_value"]
            ci_valid = ci_low.notna() & ci_high.notna() & obs.notna()
            coverage = (
                float(
                    (
                        (obs[ci_valid] >= ci_low[ci_valid])
                        & (obs[ci_valid] <= ci_high[ci_valid])
                    ).mean()
                )
                if ci_valid.any()
                else np.nan
            )
        else:
            coverage = np.nan

        rows.append(
            {
                "Compound": compound,
                "Isomer": isomer,
                "Compartment": compartment,
                "N": int(len(g)),
                "R2": r2,
                "rRMSE": rrmse,
                "GM_Fold_Error": gm_fold_error,
                "Bias_log10": bias_log,
                "Bias_fold": float(10.0 ** abs(bias_log)) if not np.isnan(bias_log) else np.nan,
                "CI_Coverage": coverage,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        ["Compound", "Isomer", "Compartment"]
    ).reset_index(drop=True)
    return summary_df


def main() -> None:
    """Compute and save goodness-of-fit summary tables."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results_root = get_results_root()
    data_root = get_data_root()

    predictions_dir = results_root / "optimization" / "monte_carlo"
    data_path = data_root / "raw" / "pfas_data_no_e1.csv"
    output_dir = results_root / "analysis" / "goodness_of_fit"

    logger.info(f"[GOF] Loading predictions from {predictions_dir}...")
    pred_df = load_predictions(predictions_dir)

    logger.info(f"[GOF] Loading observations from {data_path}...")
    obs_df = load_observations(data_path)

    logger.info("[GOF] Matching predictions with observations...")
    matched_df = match_predictions_observations(
        pred_df, obs_df, shift_day0_to_day1=True
    )
    logger.info(f"[GOF] Matched {len(matched_df)} observation–prediction pairs.")

    if matched_df.empty:
        logger.warning("[GOF] No matched pairs found. Check matrix/compartment names.")
        return

    logger.info("[GOF] Computing log10 values...")
    log_df = compute_log_values(matched_df)

    # Optional: drop compounds without intake data (example from original script)
    logger.info("[GOF] Excluding PFECHS (no intake data)...")
    n_before = len(log_df)
    log_df = log_df[log_df["Compound"] != "PFECHS"].copy()
    n_after = len(log_df)
    logger.info(f"[GOF] Removed {n_before - n_after} PFECHS data points.")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[GOF] Computing per-compound summary table...")
    summary_df = compute_summary_table(log_df)
    summary_path = output_dir / "goodness_of_fit_summary_by_compound.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"[GOF] Saved per-compound summary to {summary_path}")

    logger.info("[GOF] Computing per-compartment summary table...")
    compartment_summary_df = compute_compartment_summary_table(log_df)
    compartment_summary_path = output_dir / "goodness_of_fit_summary_by_compartment.csv"
    compartment_summary_df.to_csv(compartment_summary_path, index=False)
    logger.info(f"[GOF] Saved per-compartment summary to {compartment_summary_path}")


if __name__ == "__main__":
    main()

