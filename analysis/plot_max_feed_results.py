"""
Plotting helper for maximum feed concentration results.

This script reads the table produced by analysis/max_feed_estimation.py:

  results/analysis/max_feed_concentrations.csv

and generates:
  1. A heatmap‑style barplot of maximum allowed feed concentrations
     by compound and breed/parity.
  2. For a selected compound–isomer and breed/parity, an illustrative
     curve of feed concentration versus the 97.5th percentile of milk
     concentration, with the regulatory limit marked.

The goal is to provide simple, paper‑ready figures without re‑running
the full Monte Carlo optimisation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_results_root  # type: ignore
from auxiliary.plot_style import set_paper_plot_style  # type: ignore

logger = logging.getLogger(__name__)


def _load_max_feed_table(results_root: Path) -> pd.DataFrame:
    path = results_root / "analysis" / "max_feed_concentrations.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"max_feed_concentrations.csv not found at {path}. "
            "Run analysis/max_feed_estimation.py first."
        )
    df = pd.read_csv(path)
    return df


def plot_max_feed_barplot(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Create a barplot of max feed concentrations by compound and breed/parity.

    Expected columns in df:
      Compound, Isomer, Breed, Parity, Max_Feed_ug_per_kg_DM (or similar)
    """
    if "Max_Feed_ug_per_kg_DM" not in df.columns:
        logger.warning(
            "[PLOT_MAX_FEED] Column 'Max_Feed_ug_per_kg_DM' not found; "
            "skipping barplot."
        )
        return

    set_paper_plot_style()
    sns.set_style("whitegrid")

    df_plot = df.copy()
    df_plot["Breed_Parity"] = df_plot["Breed"] + " " + df_plot["Parity"]

    plt.figure(figsize=(6.0, 4.0), dpi=150)
    ax = sns.barplot(
        data=df_plot,
        x="Compound",
        y="Max_Feed_ug_per_kg_DM",
        hue="Breed_Parity",
    )
    ax.set_ylabel("Max feed concentration (µg/kg DM)")
    ax.set_xlabel("Compound")
    ax.legend(title="Breed / parity", fontsize=7)
    plt.tight_layout()

    out_path = out_dir / "max_feed_barplot_by_compound_breed_parity.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("[PLOT_MAX_FEED] Saved barplot to %s", out_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results_root = get_results_root()
    out_dir = results_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = _load_max_feed_table(results_root)
    except FileNotFoundError as exc:
        logger.error("[PLOT_MAX_FEED] %s", exc)
        return

    plot_max_feed_barplot(df, out_dir)


if __name__ == "__main__":
    main()

