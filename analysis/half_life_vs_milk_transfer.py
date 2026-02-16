"""
Plot half-life (from data) versus milk transfer rate.

Half-life is estimated from plasma concentration decay in the depuration phase
(two-point method, measurements above LOQ only). Milk transfer rate comes from
mass balance.

Inputs:
  - data/raw/pfas_data_no_e1.csv (plasma depuration for half-life)
  - results/analysis/toxicokinetics/milk_transfer_rates_all_compounds.csv
    If missing, derived from results/mass_balance/mass_balance_results.csv

Output:
  - results/analysis/toxicokinetics/half_life_vs_milk_transfer_rate.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_data_root, get_results_root  # type: ignore
from auxiliary.plot_style import set_paper_plot_style  # type: ignore
from mass_balance.experimental_mass_balance import (  # type: ignore
    _get_functional_group,
    _get_chain_length,
)

# Depuration phase: day after which decay is used for half-life (align with partition_coefficients)
_DEPURATION_START_DAY = 55
_PLASMA_LOQ = 0.5  # µg/L; only use measurements above LOQ for decay estimate

# Compound/Isomer labels that are not in the transfer table (exclude from half-life estimation)
_EXCLUDE_ISOMERS = {"Total"}
_EXCLUDE_COMPOUNDS = {"SD PFCA", "SD PFSA", "Summe PFCA", "Summe PFSA", "√((∑SD)^2/Anzahl PFAS)"}



def _get_default_palettes() -> Tuple[Dict[str, str], Dict[str, str]]:
    # PFCA teal, PFSA amber
    func_palette = {
        "PFCA": "#0d9488",  # teal
        "PFSA": "#d97706",  # amber
    }
    isomer_markers = {
        "Linear": "o",
        "Branched": "s",
    }
    return func_palette, isomer_markers


def _get_functional_group_public(compound: str) -> str:
    return _get_functional_group(compound)


def _get_chain_length_public(compound: str) -> int:
    return _get_chain_length(compound)


def _estimate_half_life_from_depuration_data(
    data_path: Path,
    matrix: str = "Plasma",
    depuration_start_day: int = _DEPURATION_START_DAY,
    loq: float = _PLASMA_LOQ,
) -> pd.DataFrame:
    """
    Estimate elimination half-life (days) from concentration decay in the depuration phase.

    Uses only measurements above LOQ. For each (Compound, Isomer), we take the
    earliest and latest depuration day that have at least one concentration > LOQ,
    use geometric mean concentration at each of those days, then:
      k = -ln(C_late / C_early) / (t_late - t_early),  t_half = ln(2) / k.
    Compounds/Isomers in _EXCLUDE_* are skipped. Returns DataFrame: Compound, Isomer, t_half_system_days.
    """
    df = pd.read_csv(data_path)
    df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")
    df = df.dropna(subset=["Concentration", "Day"])
    df = df[
        (df["Matrix"].str.strip().str.lower() == matrix.lower())
        & (df["Day"] > depuration_start_day)
        & (df["Concentration"] > loq)
    ].copy()
    df = df[
        ~df["Compound"].isin(_EXCLUDE_COMPOUNDS)
        & ~df["Isomer"].isin(_EXCLUDE_ISOMERS)
    ]

    rows = []
    for (compound, isomer), group in df.groupby(["Compound", "Isomer"]):
        if group.shape[0] < 2:
            rows.append(
                {"Compound": str(compound), "Isomer": str(isomer), "t_half_system_days": float("nan")}
            )
            continue
        by_day = (
            group.groupby("Day")["Concentration"]
            .apply(lambda s: np.exp(np.log(s).mean()))
            .sort_index()
        )
        if len(by_day) < 2:
            rows.append(
                {"Compound": str(compound), "Isomer": str(isomer), "t_half_system_days": float("nan")}
            )
            continue
        t_early = by_day.index[0]
        t_late = by_day.index[-1]
        if t_late <= t_early:
            rows.append(
                {"Compound": str(compound), "Isomer": str(isomer), "t_half_system_days": float("nan")}
            )
            continue
        c_early = by_day.iloc[0]
        c_late = by_day.iloc[-1]
        if c_late >= c_early or c_early <= 0:
            rows.append(
                {"Compound": str(compound), "Isomer": str(isomer), "t_half_system_days": float("nan")}
            )
            continue
        k = -np.log(c_late / c_early) / (t_late - t_early)
        if not np.isfinite(k) or k <= 0:
            t_half = float("nan")
        else:
            t_half = float(np.log(2.0) / k)
        rows.append(
            {"Compound": str(compound), "Isomer": str(isomer), "t_half_system_days": t_half}
        )

    return pd.DataFrame(rows)


def _ensure_transfer_rates(results_root: Path) -> Path:
    """
    Ensure we have a per‑compound/isomer milk transfer‑rate summary.
    Preferred: results/analysis/toxicokinetics/milk_transfer_rates_all_compounds.csv
    If missing, derive from results/mass_balance/mass_balance_results.csv.
    """
    out_path = (
        results_root
        / "analysis"
        / "toxicokinetics"
        / "milk_transfer_rates_all_compounds.csv"
    )
    if out_path.exists():
        return out_path

    mb_path = results_root / "mass_balance" / "mass_balance_results.csv"
    if not mb_path.exists():
        raise FileNotFoundError(
            f"Cannot build transfer-rate summary: {mb_path} does not exist."
        )

    mb = pd.read_csv(mb_path)
    required = {"Compound", "Isomer", "transfer_rate_milk"}
    missing = required - set(mb.columns)
    if missing:
        raise ValueError(
            f"mass_balance_results.csv is missing required columns: {sorted(missing)}"
        )

    df = mb.copy()
    df = df[df["Isomer"] != "Total"]
    summary = (
        df.groupby(["Compound", "Isomer"], as_index=False)["transfer_rate_milk"]
        .mean()
        .rename(columns={"transfer_rate_milk": "transfer_rate"})
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"[HL] Saved milk transfer-rate summary to {out_path}")
    return out_path


def _build_merged(results_root: Path, data_path: Path) -> pd.DataFrame:
    """Build table: half-life from data (plasma depuration) + milk transfer rate."""
    tr_path = _ensure_transfer_rates(results_root)
    df_tr = pd.read_csv(tr_path)
    df_hl = _estimate_half_life_from_depuration_data(data_path, matrix="Plasma")
    df_hl["functional_group"] = df_hl["Compound"].map(_get_functional_group_public)
    df_hl["chain_length"] = df_hl["Compound"].map(_get_chain_length_public)
    merged = df_hl.merge(df_tr, on=["Compound", "Isomer"], how="inner")
    merged = merged[
        ~((merged["Compound"] == "PFPeA") & (merged["Isomer"] == "Linear"))
    ]
    return merged


def _plot_half_life_vs_transfer(
    merged: pd.DataFrame,
    output_dir: Path,
    filename: str,
) -> Path:
    """Draw half-life vs milk transfer rate and save."""
    from scipy import stats

    if merged.empty:
        raise ValueError("No data to plot for half-life vs milk transfer rate.")

    set_paper_plot_style()
    sns.set_style("white")
    func_palette, isomer_markers = _get_default_palettes()
    fig, ax = plt.subplots(figsize=(5.2, 4.5), dpi=150)
    ax.set_facecolor("white")
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.grid(False)

    # Regression line (drawn first so points sit on top); y in %
    x_all = merged["t_half_system_days"].values.astype(float)
    y_all = 100.0 * merged["transfer_rate"].values.astype(float)  # fraction -> %
    valid = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0) & (y_all > 0)
    if valid.sum() > 2:
        x_log = np.log10(x_all[valid])
        y_log = np.log10(y_all[valid])
        slope_log, intercept_log, r, p, se = stats.linregress(x_log, y_log)
        x_line = np.linspace(x_all[valid].min(), x_all[valid].max(), 100)
        y_line = 10.0 ** (intercept_log + slope_log * np.log10(x_line))
        ax.plot(
            x_line, y_line,
            color="#333333",
            linewidth=2.0,
            linestyle="-",
            zorder=1,
            label=f"Regression log-log (r = {r:.3f}, p = {p:.3f})",
        )

    for (fg, isomer), group in merged.groupby(
        ["functional_group", "Isomer"], dropna=False
    ):
        x = group["t_half_system_days"].values
        y = 100.0 * group["transfer_rate"].values  # fraction -> %
        color = func_palette.get(fg, "#525252")
        marker = isomer_markers.get(isomer, "o")
        ax.scatter(
            x, y,
            facecolor=color,
            edgecolor="white",
            linewidth=1.2,
            s=72,
            marker=marker,
            label=f"{fg} {isomer}",
            zorder=10,
        )

    ax.set_xlabel(r"$t_{1/2}$ (days)")
    ax.set_ylabel("Milk transfer rate (%)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    x_vals = merged["t_half_system_days"].replace(0, np.nan).dropna()
    if len(x_vals) > 0:
        x_min, x_max = float(x_vals.min()), float(x_vals.max())
        ax.set_xlim(max(0.1, x_min / 1.1), x_max * 1.1)

    from matplotlib.ticker import LogLocator
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2g}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccc")

    handles, labels = ax.get_legend_handles_labels()
    unique: Dict[str, plt.Line2D] = {}
    for h, lab in zip(handles, labels):
        if lab not in unique:
            unique[lab] = h
    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=9,
        title_fontsize=9,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
    )
    fig.tight_layout()
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=300, format="png", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    results_root = get_results_root()
    data_root = get_data_root()
    data_path = data_root / "raw" / "pfas_data_no_e1.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    output_dir = results_root / "analysis" / "toxicokinetics"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = _build_merged(results_root, data_path)
    out_path = _plot_half_life_vs_transfer(
        merged,
        output_dir,
        "half_life_vs_milk_transfer_rate.png",
    )
    print(f"[HL] Saved plot to {out_path}")


if __name__ == "__main__":
    main()
