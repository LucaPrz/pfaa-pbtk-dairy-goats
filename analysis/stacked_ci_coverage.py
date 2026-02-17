"""
Stacked uncertainty decomposition and CI coverage analysis.

Uses Monte Carlo prediction outputs (param-only, param+animal, param+animal+observational CIs)
and matched observations to:

  1. Decompose uncertainty into three stacked levels:
     - Parameter uncertainty only (Param_CI_Lower/Upper)
     - Parameter + animal variation (CI_Lower/Upper)
     - Parameter + animal + observational (CI_Observation_Lower/Upper)

  2. Report empirical coverage of each stacked CI (fraction of observations
     falling inside the interval), overall and by compound-isomer and compartment.

Outputs:
  - results/analysis/uncertainty_decomposition/stacked_ci_coverage.csv
  - results/analysis/uncertainty_decomposition/stacked_ci_coverage_by_compound.csv
  - results/analysis/uncertainty_decomposition/stacked_ci_coverage_by_compartment.csv
  - results/analysis/uncertainty_decomposition/stacked_ci_decomposition_summary.csv (interval widths)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_data_root, get_results_root

# Reuse loading and matching from goodness_of_fit
from analysis.goodness_of_fit import (
    load_observations,
    load_predictions,
    match_predictions_observations,
)

logger = logging.getLogger(__name__)

# Required prediction columns for stacked CIs
PARAM_CI_COLS = ("Param_CI_Lower", "Param_CI_Upper")
PARAM_ANIMAL_CI_COLS = ("CI_Lower", "CI_Upper")
PARAM_ANIMAL_OBS_CI_COLS = ("CI_Observation_Lower", "CI_Observation_Upper")


def _coverage(
    obs: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> Tuple[float, int]:
    """Fraction of observations in [lower, upper]; count of valid pairs."""
    valid = lower.notna() & upper.notna() & obs.notna()
    n = int(valid.sum())
    if n == 0:
        return np.nan, 0
    inside = (obs[valid] >= lower[valid]) & (obs[valid] <= upper[valid])
    return float(inside.mean()), n


def _log_width(lower: pd.Series, upper: pd.Series, eps: float = 1e-9) -> pd.Series:
    """Log10(upper/lower) as a measure of interval width in log-space."""
    return np.log10(np.maximum(upper, eps) / np.maximum(lower, eps))


def compute_stacked_coverage(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each matched row, compute whether the observation falls inside each CI level.
    Returns same dataframe with added columns: In_Param_CI, In_Param_Animal_CI, In_Param_Animal_Obs_CI.
    """
    df = matched_df.copy()
    obs = df["Concentration"]

    for name, (lo, hi) in [
        ("Param", PARAM_CI_COLS),
        ("Param_Animal", PARAM_ANIMAL_CI_COLS),
        ("Param_Animal_Obs", PARAM_ANIMAL_OBS_CI_COLS),
    ]:
        if lo not in df.columns or hi not in df.columns:
            df[f"In_{name}_CI"] = np.nan
            continue
        low = pd.to_numeric(df[lo], errors="coerce")
        high = pd.to_numeric(df[hi], errors="coerce")
        df[f"In_{name}_CI"] = (obs >= low) & (obs <= high)

    return df


def coverage_overall(df: pd.DataFrame) -> pd.DataFrame:
    """One row: overall coverage and N for each CI level."""
    obs = df["Concentration"]
    rows: List[Dict[str, float | int]] = []

    for level, (lo_col, hi_col) in [
        ("Param_only", PARAM_CI_COLS),
        ("Param_plus_Animal", PARAM_ANIMAL_CI_COLS),
        ("Param_plus_Animal_plus_Obs", PARAM_ANIMAL_OBS_CI_COLS),
    ]:
        if lo_col not in df.columns or hi_col not in df.columns:
            rows.append({"CI_Level": level, "Coverage": np.nan, "N": 0})
            continue
        cov, n = _coverage(obs, df[lo_col], df[hi_col])
        rows.append({"CI_Level": level, "Coverage": cov, "N": n})

    return pd.DataFrame(rows)


def coverage_by_compound(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage and N per compound-isomer per CI level."""
    obs = df["Concentration"]
    rows: List[Dict[str, object]] = []

    for (compound, isomer), g in df.groupby(["Compound", "Isomer"]):
        for level, (lo_col, hi_col) in [
            ("Param_only", PARAM_CI_COLS),
            ("Param_plus_Animal", PARAM_ANIMAL_CI_COLS),
            ("Param_plus_Animal_plus_Obs", PARAM_ANIMAL_OBS_CI_COLS),
        ]:
            if lo_col not in g.columns or hi_col not in g.columns:
                rows.append({
                    "Compound": compound,
                    "Isomer": isomer,
                    "CI_Level": level,
                    "Coverage": np.nan,
                    "N": 0,
                })
                continue
            cov, n = _coverage(g["Concentration"], g[lo_col], g[hi_col])
            rows.append({
                "Compound": compound,
                "Isomer": isomer,
                "CI_Level": level,
                "Coverage": cov,
                "N": n,
            })

    return pd.DataFrame(rows)


def coverage_by_compartment(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage and N per compartment per CI level."""
    rows: List[Dict[str, object]] = []

    for (compartment,), g in df.groupby(["Compartment"]):
        for level, (lo_col, hi_col) in [
            ("Param_only", PARAM_CI_COLS),
            ("Param_plus_Animal", PARAM_ANIMAL_CI_COLS),
            ("Param_plus_Animal_plus_Obs", PARAM_ANIMAL_OBS_CI_COLS),
        ]:
            if lo_col not in g.columns or hi_col not in g.columns:
                rows.append({
                    "Compartment": compartment,
                    "CI_Level": level,
                    "Coverage": np.nan,
                    "N": 0,
                })
                continue
            cov, n = _coverage(g["Concentration"], g[lo_col], g[hi_col])
            rows.append({
                "Compartment": compartment,
                "CI_Level": level,
                "Coverage": cov,
                "N": n,
            })

    return pd.DataFrame(rows)


def decomposition_summary(matched_df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """
    Summarise:
      - interval widths (in log-space), and
      - exact variance components in log-space where available
        (Var_param_log, Var_animal_log, Var_obs_log, Var_total_log from MC outputs),
      together with Sigma/Animal_Variation where available.

    One row per compound–isomer–compartment.
    """
    df = matched_df.copy()
    rows: List[Dict[str, object]] = []

    for level, (lo_col, hi_col) in [
        ("Param_only", PARAM_CI_COLS),
        ("Param_plus_Animal", PARAM_ANIMAL_CI_COLS),
        ("Param_plus_Animal_plus_Obs", PARAM_ANIMAL_OBS_CI_COLS),
    ]:
        if lo_col not in df.columns or hi_col not in df.columns:
            continue
        df[f"LogWidth_{level}"] = _log_width(df[lo_col], df[hi_col], eps=eps)

    group_cols = ["Compound", "Isomer", "Compartment"]
    for key, g in df.groupby(group_cols):
        comp, iso, compartment = key
        row: Dict[str, object] = {
            "Compound": comp,
            "Isomer": iso,
            "Compartment": compartment,
            "N": len(g),
        }

        # Mean log-widths (base-10) per CI level (descriptive only)
        for c in [col for col in df.columns if col.startswith("LogWidth_")]:
            level_name = c.replace("LogWidth_", "")
            row[f"Mean_LogWidth_{level_name}"] = g[c].mean()

        # Exact variance components in log-space, if present from MC predictions
        for col in ["Var_param_log", "Var_animal_log", "Var_obs_log", "Var_total_log"]:
            if col in g.columns:
                row[col] = float(np.nanmean(g[col]))

        # Optional fractions of total variance (only if all components present and Var_total_log > 0)
        var_p = row.get("Var_param_log", None)
        var_a = row.get("Var_animal_log", None)
        var_o = row.get("Var_obs_log", None)
        var_tot = row.get("Var_total_log", None)
        if all(isinstance(v, (float, int)) and np.isfinite(v) for v in [var_p, var_a, var_o, var_tot]) and var_tot > 0:
            row["Frac_param_log"] = var_p / var_tot
            row["Frac_animal_log"] = var_a / var_tot
            row["Frac_obs_log"] = var_o / var_tot

        if "Animal_Variation" in g.columns:
            row["Mean_Animal_Variation"] = g["Animal_Variation"].mean()
        if "Sigma" in g.columns:
            row["Sigma"] = g["Sigma"].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)


def run(
    results_root: Optional[Path] = None,
    data_path: Optional[Path] = None,
    predictions_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    pattern: Optional[str] = None,
) -> None:
    if results_root is None:
        results_root = get_results_root()
    if data_path is None:
        data_path = get_data_root() / "raw" / "pfas_data_no_e1.csv"
    if predictions_dir is None:
        predictions_dir = results_root / "optimization" / "monte_carlo"
    if output_dir is None:
        output_dir = results_root / "analysis" / "uncertainty_decomposition"

    pred_df = load_predictions(predictions_dir)
    if pattern:
        if "_" in pattern:
            c, i = pattern.split("_", 1)
            pred_df = pred_df[(pred_df["Compound"] == c) & (pred_df["Isomer"] == i)]
        else:
            pred_df = pred_df[pred_df["Compound"] == pattern]
        if pred_df.empty:
            logger.warning(f"[STACKED_CI] No predictions match pattern '{pattern}'")
            return

    obs_df = load_observations(data_path)
    matched_df = match_predictions_observations(pred_df, obs_df)

    if matched_df.empty:
        logger.warning("[STACKED_CI] No matched observation–prediction pairs.")
        return

    # Exclude PFECHS for consistency with goodness_of_fit
    matched_df = matched_df[matched_df["Compound"] != "PFECHS"].copy()
    if matched_df.empty:
        logger.warning("[STACKED_CI] No data after excluding PFECHS.")
        return

    logger.info(f"[STACKED_CI] Matched {len(matched_df)} pairs.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Coverage tables
    overall = coverage_overall(matched_df)
    out_overall = output_dir / "stacked_ci_coverage.csv"
    overall.to_csv(out_overall, index=False)
    logger.info(f"[STACKED_CI] Saved overall coverage to {out_overall}")

    by_compound = coverage_by_compound(matched_df)
    out_compound = output_dir / "stacked_ci_coverage_by_compound.csv"
    by_compound.to_csv(out_compound, index=False)
    logger.info(f"[STACKED_CI] Saved by-compound coverage to {out_compound}")

    by_compartment = coverage_by_compartment(matched_df)
    out_compartment = output_dir / "stacked_ci_coverage_by_compartment.csv"
    by_compartment.to_csv(out_compartment, index=False)
    logger.info(f"[STACKED_CI] Saved by-compartment coverage to {out_compartment}")

    # Decomposition summary (interval widths)
    decomp = decomposition_summary(matched_df)
    if not decomp.empty:
        out_decomp = output_dir / "stacked_ci_decomposition_summary.csv"
        decomp.to_csv(out_decomp, index=False)
        logger.info(f"[STACKED_CI] Saved decomposition summary to {out_decomp}")

    # Log overall coverage
    for _, r in overall.iterrows():
        logger.info(
            f"[STACKED_CI] {r['CI_Level']}: coverage = {r['Coverage']:.2%} (N = {r['N']})"
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Stacked CI uncertainty decomposition and coverage analysis."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional: restrict to compound_isomer (e.g. PFOS_Linear) or compound.",
    )
    args = parser.parse_args()
    run(pattern=args.pattern)


if __name__ == "__main__":
    main()
