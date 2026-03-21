"""
Plot **model-derived systemic half-life** versus **model-derived milk transfer rate**
using the diagnostic helpers in `model.diagnose.PBTKModel` and the same physiology
submodel used in the optimisation pipeline.

Model-based quantities:

  1. **Systemic half-life (days)** – derived from the eigenvalues of the
     transition matrix (T) at a representative time point (t_rep)
     via `PBTKModel.systemic_half_life`, which wraps
     `PBTKModel.eigenmode_half_lives`:

         t_{1/2,i} = ln(2) / |Re(λ_i)|   for Re(λ_i) < 0.

     We use the slowest positive mode as a single systemic half-life.
  2. **Milk transfer rate TR_milk (fraction of intake)** – computed from
     `PBTKModel.steady_state_milk_transfer_rate` at the same representative
     time point, assuming a constant unit intake.

The script:

  - Reads the set of PFAS compound–isomer pairs from
    `results/analysis/kinetics/kinetic_characteristics.csv`
    (i.e. the GOF‑passing pairs used in the optimisation analyses).
  - For each pair, loads the fitted parameters from
    `results/optimization/global_fit/fit_<COMPOUND>_<ISOMER>.csv`.
  - Builds a `PBTKModel` instance with:
      * parameters from `parameters.build_parameters`,
      * a simple constant-intake function, and
      * a dynamic physiology provider from
        `parameters.build_dynamic_physiology_provider` (same physiology
        submodel as the optimisation pipeline; Alpine, multiparous).
  - Evaluates systemic half-life and steady-state milk transfer rate at a
    representative lactation time and writes a summary CSV plus a log–log
    scatter plot.

Outputs (model-based):
  - results/analysis/toxicokinetics/half_life_vs_milk_transfer_rate_model_based.csv
  - results/analysis/toxicokinetics/half_life_vs_milk_transfer_rate_model_based.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_results_root  # type: ignore
from auxiliary.plot_style import set_paper_plot_style  # type: ignore
from mass_balance.experimental_mass_balance import _get_functional_group, _get_chain_length  # type: ignore
from parameters.parameters import (  # type: ignore
    build_parameters,
    build_dynamic_physiology_provider,
)
from model.diagnose import PBTKModel  # type: ignore

# Reference physiology, intake, and representative time for model-based calculations
REF_BREED = "Alpine"
REF_PARITY = "multiparous"
INTAKE_UG_PER_DAY = 1.0  # arbitrary constant intake; cancels in fraction
# Representative time in days (mid‑lactation scale) at which we evaluate T and physiology
REPRESENTATIVE_TIME_DAYS = 150.0

# Legacy constants still used by the data-based helper functions below
_DEPURATION_START_DAY = 55
_PLASMA_LOQ = 0.5  # µg/L; only use measurements above LOQ for decay estimate
_EXCLUDE_ISOMERS = {"Total"}
_EXCLUDE_COMPOUNDS = {
    "SD PFCA",
    "SD PFSA",
    "Summe PFCA",
    "Summe PFSA",
    "√((∑SD)^2/Anzahl PFAS)",
}


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


def _build_model_merged(results_root: Path) -> pd.DataFrame:
    """
    Build a table of **model-based** systemic half-life and milk transfer rate.

    For each compound–isomer pair in `kinetic_characteristics.csv` for which a
    Phase 1 global fit exists, we:

      - Load fitted parameters from `results/optimization/global_fit/fit_<COMPOUND>_<ISOMER>.csv`.
      - Construct a `PBTKModel` with a constant intake and dynamic physiology
        via `build_dynamic_physiology_provider` (same curves as optimisation).
      - Evaluate:
          * systemic_half_life_days  ←  PBTKModel.systemic_half_life(...)
          * transfer_rate            ←  PBTKModel.steady_state_milk_transfer_rate(...)

    Returns a DataFrame with at least:
      - Compound, Isomer
      - t_half_system_days (slowest eigenmode, days)
      - t_half_fast_days (fastest eigenmode, days; may be NaN)
      - transfer_rate (fraction of intake excreted in milk at steady state)
      - functional_group, chain_length
    """
    # Kinetic characteristics define the set of GOF‑passing pairs to consider
    kinetics_path = (
        results_root
        / "analysis"
        / "kinetics"
        / "kinetic_characteristics.csv"
    )
    if not kinetics_path.exists():
        raise FileNotFoundError(
            f"kinetic_characteristics.csv not found at {kinetics_path}. "
            "Run analysis/kinetic_characteristics.py first."
        )

    df_kin = pd.read_csv(kinetics_path)
    if df_kin.empty:
        raise ValueError("kinetic_characteristics.csv is empty.")

    # Derive functional group and chain length for nicer colouring/annotation
    df_kin["functional_group"] = df_kin["Compound"].map(_get_functional_group_public)
    df_kin["chain_length"] = df_kin["Compound"].map(_get_chain_length_public)

    # ------------------------------------------------------------------
    # Physiology provider: use the SAME milk-yield and body-weight time
    # series as the optimisation workflow, so that steady-state TR_milk
    # is directly comparable to what is used during fitting.
    # ------------------------------------------------------------------
    from optimization.config import setup_context  # type: ignore
    from optimization.io import get_project_root  # type: ignore

    project_root = get_project_root()
    context = setup_context(project_root=project_root)

    # Build a *population-typical* physiology by taking the day-wise median
    # across all animals' body-weight and milk-yield series used in the
    # optimisation workflow. This uses the same measured/smoothed inputs
    # but avoids anchoring to a single reference animal (E2).
    bw_series = []
    my_series = []
    for animal in sorted(context.milk_yield_by_animal.keys()):
        bw = context.body_weight_by_animal.get(animal)
        my = context.milk_yield_by_animal.get(animal)
        if bw is None or my is None:
            continue
        # Ensure same length; load_data uses a common max_day, so this should hold.
        if len(bw) != len(my):
            continue
        bw_series.append(np.asarray(bw, dtype=float))
        my_series.append(np.asarray(my, dtype=float))

    if not bw_series or not my_series:
        raise ValueError(
            "Could not build population-median physiology arrays from optimisation context "
            "(no overlapping body_weight/milk_yield series)."
        )

    bw_stack = np.stack(bw_series, axis=0)
    my_stack = np.stack(my_series, axis=0)
    bw_array = np.median(bw_stack, axis=0)
    my_array = np.median(my_stack, axis=0)

    physiology_provider = build_dynamic_physiology_provider(
        time_unit="days",
        body_weight_array=bw_array,
        milk_yield_array=my_array,
    )

    # Choose a representative time at which the (population-median) milk yield
    # is close to its median positive value rather than using the last day.
    # This reflects typical mid‑lactation conditions while remaining tied to
    # the same measured/smoothed milk-yield curves as the optimisation.
    my_array = np.asarray(my_array, dtype=float)
    positive_mask = my_array > 0.0
    if not positive_mask.any():
        # Fallback: use day 0 if all yields are zero (should not happen in practice).
        t_rep_global = 0.0
    else:
        positive_yields = my_array[positive_mask]
        median_yield = float(np.median(positive_yields))
        # Index (day) whose yield is closest to the median
        idx_candidates = np.where(positive_mask)[0]
        idx_rep = idx_candidates[np.argmin(np.abs(my_array[positive_mask] - median_yield))]
        t_rep_global = float(idx_rep)

    def intake_const(_t: float) -> float:
        return INTAKE_UG_PER_DAY

    rows: List[Dict[str, object]] = []

    for compound, isomer in (
        df_kin[["Compound", "Isomer"]].drop_duplicates().itertuples(index=False)
    ):
        compound_str = str(compound)
        isomer_str = str(isomer)

        fit_path = (
            results_root
            / "optimization"
            / "global_fit"
            / f"fit_{compound_str}_{isomer_str}.csv"
        )
        if not fit_path.exists():
            # No fitted parameters for this pair – keep a row with NaNs so that
            # downstream merging/plotting can decide how to handle it.
            rows.append(
                {
                    "Compound": compound_str,
                    "Isomer": isomer_str,
                    "t_half_system_days": float("nan"),
                    "t_half_fast_days": float("nan"),
                    "transfer_rate": float("nan"),
                }
            )
            continue

        fit_df = pd.read_csv(fit_path)
        if fit_df.empty:
            rows.append(
                {
                    "Compound": compound_str,
                    "Isomer": isomer_str,
                    "t_half_system_days": float("nan"),
                    "t_half_fast_days": float("nan"),
                    "transfer_rate": float("nan"),
                }
            )
            continue

        fit_params = dict(zip(fit_df["Parameter"], fit_df["Value"]))

        # Build full parameter dict (same helper used by optimisation code),
        # using E2 as the reference animal for physiology volumes.
        config = {"animal": "E2", "compound": compound_str, "isomer": isomer_str}
        all_params = build_parameters(config=config, fit_params=fit_params)

        model = PBTKModel(
            params=all_params,
            intake_function=intake_const,
            physiology_provider=physiology_provider,
        )

        # Use clamped representative time so milk_yield is defined and non-zero
        t_rep = t_rep_global

        # Eigenmode-based half-lives
        half_lives = model.eigenmode_half_lives(t=t_rep)
        if half_lives.size > 0:
            t_fast = float(half_lives[0])
            t_slow = float(half_lives[-1])
        else:
            t_fast = float("nan")
            t_slow = float("nan")

        # Single systemic half-life helper (slowest positive eigenmode)
        systemic_half_life = model.systemic_half_life(t=t_rep)
        if systemic_half_life is None:
            systemic_half_life = float("nan")

        # Steady-state milk transfer rate at the same representative time
        tr_milk = model.steady_state_milk_transfer_rate(t=t_rep)
        if tr_milk is None:
            tr_milk = float("nan")

        rows.append(
            {
                "Compound": compound_str,
                "Isomer": isomer_str,
                "t_half_system_days": float(systemic_half_life),
                "t_half_fast_days": float(t_fast),
                "transfer_rate": float(tr_milk),
            }
        )

    df_model = pd.DataFrame(rows)

    # Attach functional group and chain length for plotting/colouring
    df_model = df_model.merge(
        df_kin[["Compound", "Isomer", "functional_group", "chain_length"]],
        on=["Compound", "Isomer"],
        how="left",
    )

    return df_model


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
    # Interpret "transfer_rate" as fraction and plot in %
    y_all = 100.0 * merged["transfer_rate"].values.astype(float)
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
        y = 100.0 * group["transfer_rate"].values
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
    output_dir = results_root / "analysis" / "toxicokinetics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model-based systemic half-life and transfer-rate table using the
    # diagnostic helpers from `model.diagnose.PBTKModel` and the same
    # physiology submodel as the optimisation pipeline.
    merged_model = _build_model_merged(results_root)

    csv_path_model = (
        output_dir / "half_life_vs_milk_transfer_rate_model_based.csv"
    )
    merged_model.to_csv(csv_path_model, index=False)

    out_path_model = _plot_half_life_vs_transfer(
        merged_model,
        output_dir,
        "half_life_vs_milk_transfer_rate_model_based.png",
    )
    print(f"[HL_MODEL] Saved model-based table to {csv_path_model}")
    print(f"[HL_MODEL] Saved model-based plot to {out_path_model}")


if __name__ == "__main__":
    main()
