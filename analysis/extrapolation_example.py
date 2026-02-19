"""
Create example extrapolation plots with shaded confidence intervals
for a single compound–isomer and compartment, using Phase 3 MC outputs.

The plot shows (y-axis in log scale so CI bands are visible, not squashed to zero):
  - Median prediction over time
  - 95% param-only, param+animal, and obs-level CIs (shaded)
  - Observed data points

Figure is saved under:
  results/figures/extrapolation_example_<compound>_<isomer>_<compartment>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root on sys.path and import helpers
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_results_root, get_data_root
from analysis.goodness_of_fit import load_observations, match_predictions_observations


def plot_extrapolation_example(
    compound: str,
    isomer: str,
    compartment: str,
    results_root: Path | None = None,
) -> Path | None:
    if results_root is None:
        results_root = get_results_root()

    mc_dir = results_root / "optimization" / "monte_carlo"
    csv_path = mc_dir / f"predictions_{compound}_{isomer}_monte_carlo.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MC predictions not found: {csv_path}")

    # Load predictions and subset to chosen compartment
    pred_df = pd.read_csv(csv_path)
    g = pred_df[pred_df["Compartment"] == compartment].copy().sort_values("Time")
    if g.empty:
        print(f"No data for compartment '{compartment}' in {csv_path.name}")
        return None

    t = g["Time"].values
    median = g["Pred_Median"].values
    # Parameter-only CI for mean trajectory
    param_lo = g["Param_CI_Lower"].values
    param_hi = g["Param_CI_Upper"].values
    # Param + animal CI
    pa_lo = g["CI_Lower"].values
    pa_hi = g["CI_Upper"].values
    # Observation-level CI (param + animal + obs)
    obs_lo = g["CI_Observation_Lower"].values
    obs_hi = g["CI_Observation_Upper"].values

    # For log-scale plot: avoid zeros so the CI band is visible (not "on the ground")
    ymin = 1e-4
    param_lo = np.maximum(param_lo, ymin)
    pa_lo = np.maximum(pa_lo, ymin)
    obs_lo = np.maximum(obs_lo, ymin)
    median_plot = np.maximum(median, ymin)

    # Load observations and match to predictions (to get observed points)
    data_path = get_data_root() / "raw" / "pfas_data_no_e1.csv"
    obs_df = load_observations(data_path)
    matched = match_predictions_observations(pred_df, obs_df)
    matched = matched[
        (matched["Compound"] == compound)
        & (matched["Isomer"] == isomer)
        & (matched["Compartment"] == compartment)
    ].copy()

    fig, ax = plt.subplots(figsize=(7, 4))

    # Observation-level band (widest, backmost)
    ax.fill_between(
        t,
        obs_lo,
        obs_hi,
        color="tab:blue",
        alpha=0.2,
        label="95% obs-level CI (param + animal + obs)",
    )
    # Param + animal band
    ax.fill_between(
        t,
        pa_lo,
        pa_hi,
        color="tab:green",
        alpha=0.3,
        label="95% CI (param + animal)",
    )
    # Parameter-only band (narrowest, on top)
    ax.fill_between(
        t,
        param_lo,
        param_hi,
        color="tab:orange",
        alpha=0.4,
        label="95% param-only CI (mean)",
    )
    # Median curve (use clipped values for log scale)
    ax.plot(t, median_plot, color="black", lw=2, label="Median prediction")

    # Observed data points (if any); clip to ymin for log scale
    if not matched.empty:
        obs_conc = matched["Concentration"].values
        obs_conc_plot = np.maximum(obs_conc, ymin)
        ax.scatter(
            matched["Time"],
            obs_conc_plot,
            color="black",
            s=20,
            alpha=0.8,
            marker="o",
            label="Observed data",
        )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(f"{compartment} concentration (µg/kg)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    figures_dir = results_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        figures_dir
        / f"extrapolation_example_{compound}_{isomer}_{compartment}.png"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved extrapolation example to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create extrapolation example plot with shaded CIs."
    )
    parser.add_argument(
        "--compound",
        type=str,
        default="PFOS",
        help="Compound name, e.g. PFOS",
    )
    parser.add_argument(
        "--isomer",
        type=str,
        default="Linear",
        help="Isomer name, e.g. Linear or Branched",
    )
    parser.add_argument(
        "--compartment",
        type=str,
        default="milk",
        help="Compartment name, e.g. milk, plasma.",
    )
    args = parser.parse_args()

    plot_extrapolation_example(
        compound=args.compound,
        isomer=args.isomer,
        compartment=args.compartment,
    )


if __name__ == "__main__":
    main()

