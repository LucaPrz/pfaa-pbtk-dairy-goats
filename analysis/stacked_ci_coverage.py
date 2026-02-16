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
    Summarise interval widths (in log-space) and Sigma/Animal_Variation where available.
    One row per compound-isomer-compartment (or aggregate) with:
      - mean log-width per CI level, and
      - approximate variance components in log10-space:
          Var_param, Var_animal, Var_obs, Var_total,
        assuming approximately normal log10 distributions and central 95% CIs.
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

    agg_cols = [c for c in df.columns if c.startswith("LogWidth_")]
    if not agg_cols:
        return pd.DataFrame()

    group_cols = ["Compound", "Isomer", "Compartment"]
    for key, g in df.groupby(group_cols):
        comp, iso, compartment = key
        row: Dict[str, object] = {
            "Compound": comp,
            "Isomer": iso,
            "Compartment": compartment,
            "N": len(g),
        }

        # Mean log-widths (base-10) per CI level
        for c in agg_cols:
            level_name = c.replace("LogWidth_", "")
            row[f"Mean_LogWidth_{level_name}"] = g[c].mean()

        # Approximate variance decomposition in log10-space using CI widths.
        # For a normal log10(Y) ~ N(mu, s^2), a central 95% CI has width 2 * z * s
        # in log10 units, with z ≈ 1.96. Thus s ≈ width / (2*z), Var = s^2.
        z_95 = 1.96
        denom = 2.0 * z_95

        width_param = row.get("Mean_LogWidth_Param_only", None)
        width_param_animal = row.get("Mean_LogWidth_Param_plus_Animal", None)
        width_total = row.get("Mean_LogWidth_Param_plus_Animal_plus_Obs", None)

        var_param = var_param_animal = var_total = var_animal = var_obs = None

        if isinstance(width_param, (float, int)) and np.isfinite(width_param):
            s_param = width_param / denom
            var_param = float(max(s_param ** 2, 0.0))
            row["Var_param_log10"] = var_param

        if isinstance(width_param_animal, (float, int)) and np.isfinite(width_param_animal):
            s_param_animal = width_param_animal / denom
            var_param_animal = float(max(s_param_animal ** 2, 0.0))
            row["Var_param_plus_animal_log10"] = var_param_animal

        if isinstance(width_total, (float, int)) and np.isfinite(width_total):
            s_total = width_total / denom
            var_total = float(max(s_total ** 2, 0.0))
            row["Var_total_log10"] = var_total

        # Derived components: Var_animal and Var_obs (clipped at zero)
        if var_param is not None and var_param_animal is not None:
            var_animal = max(var_param_animal - var_param, 0.0)
            row["Var_animal_log10"] = var_animal

        if var_param_animal is not None and var_total is not None:
            var_obs = max(var_total - var_param_animal, 0.0)
            row["Var_obs_log10"] = var_obs

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
    shift_day0_to_day1: bool = True,
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
    matched_df = match_predictions_observations(
        pred_df, obs_df, shift_day0_to_day1=shift_day0_to_day1
    )

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
    parser.add_argument(
        "--no-day0-shift",
        action="store_true",
        help="Do not shift day-0 observations to day-1 for matching.",
    )
    args = parser.parse_args()
    run(shift_day0_to_day1=not args.no_day0_shift, pattern=args.pattern)


if __name__ == "__main__":
    main()
