"""
Diagnostics for LOWESS smoothing of daily PFAS intake.

This script:
  1. Loads the daily intake table used by the model.
  2. For each (Animal, Compound, Isomer) intake series, constructs a
     contiguous daily grid (Day 0..max_day, missing days = 0).
  3. Applies LOWESS smoothing with the current default settings
     (frac = 2/3, it = 3) and a small set of alternative frac values.
  4. Computes RMSE between raw and smoothed intake for each series
     and each frac, considering only days with non-zero raw intake.
  5. Writes a summary CSV under
       results/analysis/intake_smoothing/intake_smoothing_rmse.csv
     with per-series and overall metrics so we can judge whether
     the default smoothing is overly aggressive or too weak.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

# Ensure project root is on sys.path when executed via absolute path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.io import get_project_root, load_data
from optimization.config import ModelConfig

logger = logging.getLogger(__name__)


def _compute_rmse_all_days(raw: np.ndarray, smoothed: np.ndarray) -> float:
    """RMSE between raw and smoothed intake over all days."""
    if raw.shape != smoothed.shape:
        raise ValueError("raw and smoothed arrays must have the same shape")
    diff = smoothed - raw
    return float(np.sqrt(np.mean(diff**2)))


def _compute_rmse_nonzero_days(raw: np.ndarray, smoothed: np.ndarray) -> float:
    """RMSE between raw and smoothed intake, restricted to days with non-zero raw intake."""
    if raw.shape != smoothed.shape:
        raise ValueError("raw and smoothed arrays must have the same shape")
    mask = raw > 0
    if not np.any(mask):
        return np.nan
    diff = smoothed[mask] - raw[mask]
    return float(np.sqrt(np.mean(diff**2)))


def diagnose_intake_smoothing(
    frac_values: List[float] | Tuple[float, ...] = (
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        2.0 / 3.0,
        0.7,
        0.8,
        0.9,
    ),
    it: int = 3,
) -> pd.DataFrame:
    """
    Run intake smoothing diagnostics for a set of LOWESS frac values.

    Returns a DataFrame with columns:
      Animal, Compound, Isomer, max_day, frac,
      rmse_all_days, rmse_nonzero_days
    """
    project_root = get_project_root()
    config = ModelConfig()

    intake_df, *_ = load_data(config, project_root=project_root)
    if intake_df.empty:
        raise ValueError("Intake table is empty; cannot run diagnostics.")

    intake_df = intake_df.copy()
    intake_df["Day"] = intake_df["Day"].astype(int)
    intake_df["PFAS_Intake_ug_day"] = intake_df["PFAS_Intake_ug_day"].astype(float)

    results: List[Dict[str, float]] = []

    group_cols = ["Animal", "Compound", "Isomer"]
    for (animal, compound, isomer), g in intake_df.groupby(group_cols):
        g = g.sort_values("Day").drop_duplicates(subset=["Day"], keep="first")
        days = g["Day"].to_numpy(dtype=int)
        values = g["PFAS_Intake_ug_day"].to_numpy(dtype=float)

        if values.size == 0:
            continue

        max_day = int(g["Day"].max())
        full_days = np.arange(0, max_day + 1, dtype=int)

        series = pd.Series(values, index=days).reindex(full_days, fill_value=0.0)
        raw_grid = series.to_numpy(dtype=float)

        for frac in frac_values:
            try:
                smoothed = lowess(
                    endog=raw_grid,
                    exog=full_days,
                    frac=frac,
                    it=it,
                    return_sorted=False,
                )
                smoothed = np.clip(smoothed, 0.0, None)
                rmse_all = _compute_rmse_all_days(raw_grid, smoothed)
                rmse_nonzero = _compute_rmse_nonzero_days(raw_grid, smoothed)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "Failed LOWESS for Animal=%s, Compound=%s, Isomer=%s, frac=%.3f: %s",
                    animal,
                    compound,
                    isomer,
                    frac,
                    e,
                )
                rmse_all = np.nan
                rmse_nonzero = np.nan

            results.append(
                {
                    "Animal": animal,
                    "Compound": compound,
                    "Isomer": isomer,
                    "max_day": max_day,
                    "frac": float(frac),
                    "rmse_all_days": rmse_all,
                    "rmse_nonzero_days": rmse_nonzero,
                }
            )

    return pd.DataFrame(results)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("Running intake smoothing diagnostics (LOWESS)...")
    df = diagnose_intake_smoothing()

    if df.empty:
        logger.warning("No intake series found; nothing to save.")
        return

    project_root = get_project_root()
    out_dir = project_root / "results" / "analysis" / "intake_smoothing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "intake_smoothing_rmse.csv"

    df.to_csv(out_csv, index=False)
    logger.info("Saved intake smoothing RMSE diagnostics to %s", out_csv)

    # Also log simple overall summaries for quick inspection
    summary_all = (
        df.groupby("frac")["rmse_all_days"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    summary_nonzero = (
        df.groupby("frac")["rmse_nonzero_days"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    logger.info("RMSE (all days) summary by frac:\n%s", summary_all.to_string(index=False))
    logger.info(
        "RMSE (non-zero days) summary by frac:\n%s",
        summary_nonzero.to_string(index=False),
    )

    # Report the frac with the lowest median RMSE over all days as a simple "optimum"
    best_all = summary_all.sort_values("median").iloc[0]
    logger.info(
        "Best frac by median RMSE over all days: frac=%.3f (median=%.4f, mean=%.4f, N=%d)",
        best_all["frac"],
        best_all["median"],
        best_all["mean"],
        int(best_all["count"]),
    )


if __name__ == "__main__":
    main()

