"""
Plot EFSA‑4 PFAS milk concentration time‑courses for Alpine/Saanen goats
under feed concentrations taken from the bovine PFAS assessment (Table 2).

For each EFSA‑4 compound (PFOS, PFOA, PFNA, PFHxS), we:
  - assume exclusive feeding of grass silage at the LOQ / highest detected
    concentration reported in the bovine assessment,
  - simulate an EXPO_DEP‑style scenario (constant exposure up to
    EXPOSURE_STOP_DAY, then 0 µg/kg DM),
  - derive milk concentrations over time for both breeds (Alpine, Saanen)
    and both parities (primiparous, multiparous),
  - overlay the EU indicative milk levels (REGULATORY_LIMITS).

Output:
  - results/figures/efsa4_feed_based_milk_profiles.svg
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_results_root  # type: ignore
from auxiliary.plot_style import set_paper_plot_style  # type: ignore
from parameters.parameters import (  # type: ignore
    build_parameters,
    build_dynamic_physiology_provider,
)
from model.diagnose import PBTKModel  # type: ignore

# Re‑use configuration and helper from the exposure scenarios script
from analysis import breed_parity_exposure_scenarios as bp  # type: ignore


EFSA4_COMPOUNDS = ("PFOS", "PFOA", "PFNA", "PFHxS")
FEED_TYPE = "grass_silage"

# Consistent colours: EFSA‑4 first, then palette for other compounds
COMPOUND_COLORS: Dict[str, str] = {
    "PFOS": "#1f77b4",
    "PFOA": "#ff7f0e",
    "PFNA": "#2ca02c",
    "PFHxS": "#d62728",
}
# Extra colours for non‑EFSA‑4 compounds with grass_silage data
_EXTRA_COLORS = [
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5",
]


def _get_feed_concentration(compound: str) -> float:
    """
    Return the grass‑silage concentration (µg/kg, fresh weight) from the
    bovine assessment table for an EFSA‑4 compound.
    """
    low, high = bp._get_feed_concentrations_for_compound(  # type: ignore[attr-defined]
        compound=compound,
        feed_type=FEED_TYPE,
    )
    # For EFSA‑4 and a single feed type we treat low/high as identical.
    return float(high)


def _build_depuration_intake_function(
    compound: str,
    feed_type: str,
    physiology_provider,
) -> Tuple[float, callable]:
    """
    Build EXPO_DEP‑style intake function:
      u(t) = C_feed * DMI(t) for t <= EXPOSURE_STOP_DAY
             0                for t > EXPOSURE_STOP_DAY
    """
    feed_conc_low, _ = bp._get_feed_concentrations_for_compound(  # type: ignore[attr-defined]
        compound=compound,
        feed_type=feed_type,
    )
    C_feed = float(feed_conc_low)

    def intake(t: float) -> float:
        phys = physiology_provider(t)
        dmi = float(phys.get("DMI", 0.0))
        if t <= bp.EXPOSURE_STOP_DAY:  # type: ignore[attr-defined]
            return C_feed * dmi
        return 0.0

    return C_feed, intake


def _simulate_milk_profile(
    compound: str,
    isomer: str,
    breed: str,
    parity: str,
    feed_type: str,
    results_root: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic EXPO_DEP simulation returning (t_eval, milk_conc).
    """
    fit_params: Dict[str, float] = bp._load_phase1_fit(  # type: ignore[attr-defined]
        compound=compound,
        isomer=isomer,
        results_root=results_root,
    )

    physiology_provider = build_dynamic_physiology_provider(
        breed=breed,
        parity=parity,
        time_unit="days",
    )

    _, intake_function = _build_depuration_intake_function(
        compound=compound,
        feed_type=feed_type,
        physiology_provider=physiology_provider,
    )

    config = {"animal": "E2", "compound": compound, "isomer": isomer}
    all_params = build_parameters(config=config, fit_params=fit_params)

    model = PBTKModel(
        params=all_params,
        intake_function=intake_function,
        physiology_provider=physiology_provider,
    )

    t_eval = np.arange(0.0, bp.TOTAL_DAYS + 1.0, 1.0)  # type: ignore[attr-defined]
    A0 = np.zeros(model.compartment_number)
    sim_result = model.simulate_over_time(A0, t_eval)

    A_matrix = sim_result.mass_matrix
    pi_plasma = model.projection_vector("plasma")
    amount_plasma = A_matrix @ pi_plasma

    PC = all_params.get("partition_coefficients", {})
    P_milk = PC.get("P_milk", 1.0)

    milk_conc = np.full_like(amount_plasma, np.nan, dtype=float)
    for i, t_i in enumerate(t_eval):
        phys = physiology_provider(float(t_i))
        V_plasma = float(phys.get("V_plasma", 0.0))
        if V_plasma > 0:
            c_plasma = float(amount_plasma[i]) / V_plasma
            milk_conc[i] = P_milk * c_plasma

    return t_eval, milk_conc


def _short_label(breed: str, parity: str) -> str:
    if parity.startswith("primi"):
        p = "1st"
    else:
        p = "multi"
    return f"{breed[0]}-{p}"


def main() -> None:
    results_root = get_results_root()
    figures_root = results_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    set_paper_plot_style()

    # For each EFSA‑4 compound choose the first isomer that has a Phase 1
    # fit, preferring "Linear" when available.
    gof_path = (
        results_root
        / "analysis"
        / "goodness_of_fit"
        / "goodness_of_fit_summary_by_compound.csv"
    )
    import pandas as pd

    df_gof = pd.read_csv(gof_path)

    # For each EFSA‑4 compound choose a preferred isomer based on the GOF
    # summary, preferring Linear where available.
    preferred_isomers: Dict[str, str] = {}
    for compound in EFSA4_COMPOUNDS:
        df_c = df_gof[df_gof["Compound"] == compound]
        if df_c.empty:
            continue
        iso = None
        for cand in ["Linear", "Branched"]:
            sub = df_c[df_c["Isomer"].str.contains(cand, case=False, na=False)]
            if not sub.empty:
                iso = str(sub.iloc[0]["Isomer"])
                break
        if iso is None:
            iso = str(df_c.iloc[0]["Isomer"])
        preferred_isomers[compound] = iso

    if not preferred_isomers:
        raise RuntimeError("No EFSA‑4 compounds with Phase 1 fits found.")

    fig, axes = plt.subplots(
        2, 2, figsize=(6.0, 5.0), sharex=True, sharey=True, dpi=600
    )
    axes = axes.reshape(-1)

    breeds = bp.BREEDS  # type: ignore[attr-defined]
    parities = bp.PARITIES  # type: ignore[attr-defined]

    # Panels are now per compound; within each panel we show curves for
    # all goat profiles (breed × parity).
    goat_profiles = [(breed, parity) for breed in breeds for parity in parities]

    for idx, compound in enumerate(EFSA4_COMPOUNDS):
        if compound not in preferred_isomers:
            continue
        if idx >= len(axes):
            break
        ax = axes[idx]
        isomer = preferred_isomers[compound]

        for breed, parity in goat_profiles:
            try:
                t, milk = _simulate_milk_profile(
                    compound=compound,
                    isomer=isomer,
                    breed=breed,
                    parity=parity,
                    feed_type=FEED_TYPE,
                    results_root=results_root,
                )
            except Exception:
                continue

            label = _short_label(breed, parity)

            ax.plot(
                t,
                milk,
                label=label,
                linewidth=1.4,
                alpha=0.9,
            )

        # Regulatory milk limit line (if defined)
        limit = bp.REGULATORY_LIMITS.get(compound, None)  # type: ignore[attr-defined]
        if limit is not None:
            ax.axhline(
                y=limit,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="EU indicative level" if idx == 0 else None,
            )

        ax.set_title(compound)
        ax.set_yscale("log")

    # Shared axes limits: log y clipped to show all curves and EU limits
    Y_MIN, Y_MAX = 1e-6, 0.1  # µg/kg milk
    for ax in axes:
        ax.set_xlabel("Time (days)")
        ax.set_xlim(0, float(bp.TOTAL_DAYS))  # type: ignore[attr-defined]
        ax.set_ylim(Y_MIN, Y_MAX)

    # Single shared y-axis label
    fig.text(
        0.01,
        0.5,
        r"Milk concentration (µg/kg)",
        va="center",
        rotation="vertical",
    )

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    unique: Dict[str, object] = {}
    for h, lab in zip(handles, labels):
        if lab not in unique:
            unique[lab] = h
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=(0.05, 0.0, 1.0, 0.95))

    out_path = figures_root / "efsa4_feed_based_milk_profiles.svg"
    fig.savefig(out_path, format="svg", dpi=600, facecolor="white")
    plt.close(fig)

    print(f"[EFSA4_FEED] Saved EFSA‑4 feed‑based milk profiles to {out_path}")


if __name__ == "__main__":
    main()

