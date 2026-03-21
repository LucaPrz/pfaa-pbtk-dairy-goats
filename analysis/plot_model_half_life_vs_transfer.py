"""
Plot **model-based systemic half-life** versus **model-based milk transfer rate**
using the diagnostic helpers implemented in `model.diagnose.PBTKModel`.

This script no longer reconstructs the simulations itself. Instead, it relies on
the model-based summary table produced by `analysis/half_life_vs_milk_transfer.py`,
which uses:

  - `PBTKModel.systemic_half_life` (wrapping `eigenmode_half_lives`) to derive a
    single systemic half-life (days) from the slowest positive eigenmode of the
    transition matrix at a representative time point, and
  - `PBTKModel.steady_state_milk_transfer_rate` to derive a model-based milk
    transfer rate TR_milk at the same time point under a constant intake.

Axes in the plot:

  - x-axis: `t_half_system_days` (systemic half-life, days; eigenmode-based)
  - y-axis: `transfer_rate` × 100 (model-based milk transfer rate, % of intake)

Inputs:
  - results/analysis/toxicokinetics/half_life_vs_milk_transfer_rate_model_based.csv

Outputs:
  - results/figures/model_half_life_vs_milk_transfer.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_results_root  # type: ignore
from auxiliary.plot_style import set_paper_plot_style  # type: ignore


def _short_label(compound: str, isomer: str) -> str:
    """Return compact label with n-/br- nomenclature where applicable."""
    iso = isomer.lower()
    if iso.startswith("linear"):
        prefix = "n-"
    elif iso.startswith("branched"):
        prefix = "br-"
    else:
        prefix = ""
    return f"{prefix}{compound}"


def _make_base_axes(df: pd.DataFrame):
    """Create base scatter plot and return (fig, ax, x_vals_all, y_vals_all)."""
    set_paper_plot_style()

    fig, ax = plt.subplots(figsize=(5.2, 4.5), dpi=600)
    ax.set_facecolor("white")

    # Scatter plot: x = systemic half-life (days), y = TR_milk * 100 % (log scale)
    palette = {"PFCA": "#0d9488", "PFSA": "#d97706"}
    x_vals_all = df["t_half_system_days"].astype(float)
    y_vals_all = (df["transfer_rate"] * 100.0).astype(float)

    sns.scatterplot(
        x=x_vals_all,
        y=y_vals_all,
        hue=df["functional_group"],
        style=df["Isomer"],
        palette=palette,
        s=80,
        alpha=0.9,
        edgecolor="white",
        linewidth=1.0,
        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"Systemic half-life $t_{1/2}$ (days)")
    ax.set_ylabel(r"Milk transfer rate (%)")

    # Regression line and r/p annotation (top-left), similar to GOF stats box
    from scipy import stats

    x_all = x_vals_all.values
    y_all = y_vals_all.values
    valid = np.isfinite(x_all) & np.isfinite(y_all) & (y_all > 0)
    if valid.sum() > 2:
        x_lin = x_all[valid]
        y_log = np.log10(y_all[valid])
        slope, intercept, r, p, se = stats.linregress(x_lin, y_log)
        x_line = np.linspace(x_lin.min(), x_lin.max(), 100)
        y_line = 10.0 ** (intercept + slope * x_line)
        ax.plot(
            x_line,
            y_line,
            color="#374151",
            linewidth=1.5,
            linestyle="--",  # dashed regression line
            zorder=1,
            alpha=0.7,
            label="Semilog-y regression",
        )
        text = rf"$r = {r:.2f}$" + "\n" + rf"$p = {p:.5f}$"
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # Legend, roughly matching GOF-style aesthetics
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen[lab] = h
    legend = ax.legend(
        seen.values(),
        seen.keys(),
        fontsize=7,
        loc="best",
    )
    legend._legend_box.align = "right"

    # Axes limits and ticks (no explicit grid; match GOF style)
    from matplotlib.ticker import LogLocator, FuncFormatter

    x_vals = df["t_half_system_days"].replace(0, np.nan).dropna()
    if len(x_vals) > 0:
        x_min, x_max = float(x_vals.min()), float(x_vals.max())
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))

    y_vals = (df["transfer_rate"] * 100.0).replace(0, np.nan).dropna()
    if len(y_vals) > 0:
        y_min, y_max = float(y_vals.min()), float(y_vals.max())
        ax.set_ylim(max(1e-4, y_min / 1.5), y_max * 1.5)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}"))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")

    fig.tight_layout()
    return fig, ax, x_vals_all, y_vals_all


def main() -> None:
    results_root = get_results_root()

    # Load model-based systemic half-lives and transfer rates derived via
    # `PBTKModel` diagnostics in `analysis/half_life_vs_milk_transfer.py`.
    model_summary_path = (
        results_root
        / "analysis"
        / "toxicokinetics"
        / "half_life_vs_milk_transfer_rate_model_based.csv"
    )
    if not model_summary_path.exists():
        raise FileNotFoundError(
            f"Model-based summary not found at {model_summary_path}. "
            "Run analysis/half_life_vs_milk_transfer.py first."
        )

    df = pd.read_csv(model_summary_path)
    if df.empty:
        raise ValueError("Model-based half-life/transfer summary CSV is empty.")

    figures_dir = results_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) Plot WITHOUT any point annotations
    fig_plain, ax_plain, _, _ = _make_base_axes(df)
    out_plain = figures_dir / "model_half_life_vs_milk_transfer.svg"
    fig_plain.savefig(out_plain, dpi=300, format="svg", facecolor="white")
    plt.close(fig_plain)

    # 2) Plot WITH every point annotated (n-/br- nomenclature)
    fig_annot, ax_annot, x_vals_all, y_vals_all = _make_base_axes(df)

    finite_mask = (
        np.isfinite(x_vals_all)
        & np.isfinite(y_vals_all)
        & (x_vals_all > 0)
        & (y_vals_all > 0)
    )
    if finite_mask.any():
        df_finite = df.loc[finite_mask].copy()
        used_positions = []

        for _, row in df_finite.iterrows():
            x_pt = float(row["t_half_system_days"])
            y_pt = float(row["transfer_rate"] * 100.0)
            label = _short_label(str(row["Compound"]), str(row["Isomer"]))

            # Try several offset positions to reduce overlap
            offsets = [
                (1.03, 1.03, "left", "bottom"),
                (0.97, 1.03, "right", "bottom"),
                (1.03, 0.97, "left", "top"),
                (0.97, 0.97, "right", "top"),
            ]
            for x_fac, y_fac, ha, va in offsets:
                trial_x = x_pt * x_fac
                trial_y = y_pt * y_fac
                if all(
                    abs(trial_x - ux) / max(trial_x, ux, 1e-12) > 0.03
                    or abs(np.log10(trial_y) - np.log10(uy)) > 0.08
                    for ux, uy in used_positions
                ):
                    ax_annot.text(
                        trial_x,
                        trial_y,
                        label,
                        fontsize=7,
                        ha=ha,
                        va=va,
                        color="#111827",
                        alpha=0.9,
                    )
                    used_positions.append((trial_x, trial_y))
                    break

    out_annot = figures_dir / "model_half_life_vs_milk_transfer_annotated.svg"
    fig_annot.savefig(out_annot, dpi=600, format="svg", facecolor="white")
    plt.close(fig_annot)

    print(f"[MODEL_HL] Loaded model-based half-life/transfer table from {model_summary_path}")
    print(f"[MODEL_HL] Saved unannotated plot to {out_plain}")
    print(f"[MODEL_HL] Saved fully annotated plot to {out_annot}")


if __name__ == "__main__":
    main()

