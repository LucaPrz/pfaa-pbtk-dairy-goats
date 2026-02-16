"""
Urine volume calculation for dairy goats based on creatinine clearance.

Inputs:
- data/raw/creatinine.csv

Primary output (used as model input):
- data/processed/urine_volume_per_goat.csv

Secondary output:
- results/figures/urine_volume_distribution.png
"""

from pathlib import Path
from typing import Dict
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Run from project root: python mass_balance/urine_volume.py  or  python -m mass_balance.urine_volume
_CLEAN_ROOT = Path(__file__).resolve().parent.parent
if str(_CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLEAN_ROOT))

from auxiliary.project_paths import get_data_root, get_results_root
from auxiliary.plot_style import set_paper_plot_style

# Apply global plot style once for this script
set_paper_plot_style()

def _get_paths() -> Dict[str, Path]:
    """
    Resolve all relevant paths for this module within the project root.
    """
    data_root = get_data_root()

    creatinine_path = data_root / "raw" / "creatinine.csv"
    urine_volume_out = data_root / "processed" / "urine_volume_per_goat.csv"

    return {
        "creatinine_path": creatinine_path,
        "urine_volume_out": urine_volume_out,
    }


def load_body_weight_timeseries() -> pd.DataFrame:
    """
    Load body weight (kg) per goat and date from the interpolated body weight file.

    Returns a long-format DataFrame with columns:
        - date (datetime64[ns], date only)
        - Animal ('E2', 'E3', 'E4')
        - BW_kg
    """
    data_root = get_data_root()
    bw_path = data_root / "raw" / "body_weight_interpolated.csv"
    bw = pd.read_csv(bw_path)

    # Parse date column and normalise to date only
    bw["date"] = pd.to_datetime(bw["date"]).dt.normalize()

    # All columns except 'date' are assumed to be animal IDs (E2, E3, E4, ...)
    value_cols = [c for c in bw.columns if c.lower() != "date"]

    # Wide (date, E2, E3, E4, ...) -> long (date, Animal, BW_kg)
    bw_long = bw.melt(
        id_vars="date",
        value_vars=value_cols,
        var_name="Animal",
        value_name="BW_kg",
    )

    return bw_long


def compute_urine_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute urine volume per sample [L/day] from creatinine concentrations.

    Uses allometric scaling:
        V_urine_L_per_day = 0.424 * BW^0.75 / Crea_mmol_per_L
    """
    df = df.copy()

    # Allometric scaling term
    df["BW075"] = df["BW_kg"] ** 0.75

    # Urine volume per sample [L/day]
    df["V_urine_L_per_day"] = 0.424 * df["BW075"] / df["Crea mmol/L"]

    return df


def main() -> None:
    """
    Main entry point to calculate urine volume per goat.

    Reads creatinine data, computes sample-level urine volumes,
    and saves median daily urine volume per goat as:
        02_data/processed/urine_volume_per_goat.csv
    """
    paths = _get_paths()

    df = pd.read_csv(paths["creatinine_path"], sep=";", decimal=",")

    # Parse measurement date (e.g. '22.05.23') and normalise to date only
    df["date"] = pd.to_datetime(df["Date"], format="%d.%m.%y").dt.normalize()

    # Load body weight time series and merge by Animal + date
    bw_long = load_body_weight_timeseries()
    df_merged = df.merge(bw_long, on=["date", "Animal"], how="left")

    # Optionally guard against missing BW_kg by filling with per-animal mean
    if df_merged["BW_kg"].isna().any():
        bw_means = bw_long.groupby("Animal")["BW_kg"].mean()
        df_merged["BW_kg"] = df_merged.apply(
            lambda row: bw_means[row["Animal"]] if pd.isna(row["BW_kg"]) else row["BW_kg"],
            axis=1,
        )

    df_with_v = compute_urine_volume(df_merged)

    # Urine excretion per goat (median)
    grouped = (
        df_with_v.groupby("Animal")["V_urine_L_per_day"]
        .median()
        .rename("V_urine_median_L_per_day")
        .reset_index()
    )

    paths["urine_volume_out"].parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(paths["urine_volume_out"], index=False)

    print("Wrote median urine excretion per goat to:", paths["urine_volume_out"])
    print(grouped)

    # Plot distribution of urine volumes (right-skewed) for visual check
    plot_urine_volume_distribution(df_with_v)


def plot_urine_volume_distribution(df_with_v: pd.DataFrame) -> None:
    """
    Plot the distribution of daily urine volume per goat and save to results/figures.

    The main axis shows KDE curves per goat; a stacked "rug stripe" for each
    goat is drawn just below the x-axis on the same axis to emphasise the
    sample locations.
    Plots are saved to 08_results/figures/urine_volume_distribution.png
    """
    results_root = get_results_root()
    figures_dir = results_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Two stacked axes: top for KDE, bottom for rug (strong separation)
    fig, (ax_kde, ax_rug) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.02},
        constrained_layout=True,
    )

    # Smooth density estimate (one curve per goat)
    sns.kdeplot(
        data=df_with_v,
        x="V_urine_L_per_day",
        hue="Animal",
        common_norm=False,
        palette="colorblind",
        fill=False,
        linewidth=2,
        ax=ax_kde,
    )

    # Rug plot: one separate horizontal band per goat, more prominent
    animals = sorted(df_with_v["Animal"].unique())
    colors = sns.color_palette("colorblind", n_colors=len(animals))

    for i, (animal, color) in enumerate(zip(animals, colors)):
        sub = df_with_v[df_with_v["Animal"] == animal]
        # Draw relatively tall ticks so each row stands out
        ax_rug.vlines(
            x=sub["V_urine_L_per_day"],
            ymin=i - 0.35,
            ymax=i + 0.35,
            colors=[color],
            linewidth=1.2,
        )

    # Label only the KDE axis; keep rug axis unlabeled for a cleaner look
    ax_kde.set_ylabel("Density")
    ax_rug.set_xlabel("Urine volume (L/day)")
    ax_rug.set_yticks([])
    ax_rug.set_yticklabels([])
    ax_rug.set_ylabel("")
    # Restrict x-axis to physically meaningful range (no negative volumes)
    max_v = df_with_v["V_urine_L_per_day"].max()
    ax_kde.set_xlim(left=0.0, right=max_v * 1.1)
    out_path = figures_dir / "urine_volume_distribution.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved urine volume distribution plot to: {out_path}")


if __name__ == "__main__":
    main()

