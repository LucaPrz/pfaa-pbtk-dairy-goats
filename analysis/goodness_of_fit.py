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
import matplotlib.pyplot as plt

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
) -> pd.DataFrame:
    """
    Match predictions with observations by Compound, Isomer, Matrix/Compartment, and Day.
    Observation Day d is matched to prediction Time d (Day 0 = baseline).
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

    # Merge on Compound, Isomer, Compartment, and Day
    merged = obs_df.merge(
        pred_df,
        left_on=["Compound", "Isomer", "Compartment", "Day"],
        right_on=["Compound", "Isomer", "Compartment", "Time"],
        how="inner",
        suffixes=("_obs", "_pred"),
    )

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
    # Exclude all baseline observations (Day 0) across compartments from GOF
    # metrics. Day 0 values often reflect pre-exposure background that the
    # model does not explicitly represent and can dominate fold/bias metrics.
    if "Day" in df.columns:
        df = df[df["Day"] > 0].copy()

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
    # Exclude all baseline observations (Day 0) across compartments from GOF.
    if "Day" in df.columns:
        df = df[df["Day"] > 0].copy()

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


def plot_log_pred_vs_obs_for_passing(
    log_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    r2_threshold: float = 0.7,
    gmfe_threshold: float = 2.0,
    bias_abs_threshold: float = 0.25,
) -> None:
    """
    Create a log10(predicted) vs log10(observed) scatter plot including only
    compound–isomer pairs that pass predefined goodness-of-fit criteria.

    Passing criteria (per compound–isomer):
      - R² > r2_threshold
      - GM_Fold_Error < gmfe_threshold
      - |Bias_log10| < bias_abs_threshold

    The plot is saved as:
      <output_dir>/log_pred_vs_log_obs_passing_compounds.png
    """
    if log_df.empty or summary_df.empty:
        logger.warning("[GOF] Cannot plot log_pred vs log_obs: empty input data.")
        return

    # Exclude baseline (Day 0) observations, consistent with summary tables
    if "Day" in log_df.columns:
        log_df = log_df[log_df["Day"] > 0].copy()

    crit_r2 = summary_df["R2"] > r2_threshold
    crit_gm = summary_df["GM_Fold_Error"] < gmfe_threshold
    crit_bias = summary_df["Bias_log10"].abs() < bias_abs_threshold
    passing = summary_df[crit_r2 & crit_gm & crit_bias][["Compound", "Isomer"]]

    if passing.empty:
        logger.warning("[GOF] No compound–isomer pairs pass the GOF thresholds; skipping plot.")
        return

    passing_pairs = set(map(tuple, passing.values))
    mask = log_df[["Compound", "Isomer"]].apply(tuple, axis=1).isin(passing_pairs)
    df_pass = log_df[mask].copy()
    if df_pass.empty:
        logger.warning("[GOF] No matched observations for passing compounds; skipping plot.")
        return
    
    from auxiliary.plot_style import set_paper_plot_style
    set_paper_plot_style()

    # Light grey y=x reference line
    min_val = float(min(df_pass["log_obs"].min(), df_pass["log_pred"].min()))
    max_val = float(max(df_pass["log_obs"].max(), df_pass["log_pred"].max()))
    plt.plot([min_val, max_val], [min_val, max_val], color="lightgray", linestyle="--", label="1:1 line")

    # Scatter points; color by compound–isomer label to distinguish in legend
    df_pass["Label"] = df_pass["Compound"] + " " + df_pass["Isomer"]
    for label, g in df_pass.groupby("Label"):
        plt.scatter(g["log_obs"], g["log_pred"], s=10, alpha=0.7, label=label)

    plt.xlabel("log10(observed concentration)")
    plt.ylabel("log10(predicted concentration)")
    plt.legend(fontsize=6, markerscale=2, loc="best")

    # Overall R² and GM fold error for passing compounds (for annotation)
    from math import isfinite
    obs = df_pass["log_obs"].values
    pred = df_pass["log_pred"].values
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() > 1:
        slope, intercept, r_value, p_value, std_err = linregress(obs[valid], pred[valid])
        r2_overall = float(r_value ** 2)

        obs_val = df_pass["obs_value"].values[valid]
        pred_val = df_pass["pred_value"].values[valid]
        ratio = np.clip(pred_val / obs_val, EPS, None)
        log_ratio = np.log10(ratio)
        gmfe_overall = float(10.0 ** np.mean(np.abs(log_ratio)))

        text = f"R² = {r2_overall:.2f}\nGMFE = {gmfe_overall:.2f}"
        plt.text(
            0.05,
            0.95,
            text,
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
    plt.tight_layout()

    # Save under global figures folder for consistency across analyses
    figures_dir = get_results_root() / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "log_pred_vs_log_obs_passing_compounds.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[GOF] Saved log_pred vs log_obs plot for passing compounds to {out_path}")


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
    matched_df = match_predictions_observations(pred_df, obs_df)
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

    # Optional: log10(predicted) vs log10(observed) plot for passing compounds
    try:
        plot_log_pred_vs_obs_for_passing(log_df, summary_df, output_dir=output_dir)
    except Exception as e:
        logger.warning(f"[GOF] Failed to create log_pred vs log_obs plot: {e}")


if __name__ == "__main__":
    main()

