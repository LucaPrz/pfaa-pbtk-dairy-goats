"""
Summarise core kinetic characteristics of the PFAS PBTK model under a
standardised exposure scenario.

For each compound–isomer that passes the global goodness‑of‑fit criteria,
this script:
  1. Runs a deterministic simulation for a reference breed/parity under a
     constant complete‑feed concentration of 1 µg/kg DM for a fixed
     exposure duration, followed by a PFAS‑free depuration phase.
  2. Derives simple kinetic metrics in milk and plasma:
       - Time to reach 90% and 95% of steady state during exposure
       - Time to 50% and 90% decline after switching to clean feed
       - Approximate steady‑state feed→milk transfer factor
       - Milk:plasma concentration ratio at the end of exposure
  3. Writes a tidy CSV table that can be used directly in the paper.

Outputs:
  - results/analysis/kinetic_characteristics.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on sys.path when executed via absolute path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_results_root  # type: ignore
from parameters.parameters import (  # type: ignore
    build_parameters,
    build_dynamic_physiology_provider,
)
from model.diagnose import PBTKModel  # type: ignore

logger = logging.getLogger(__name__)


# Reference exposure scenario
REF_BREED = "Alpine"
REF_PARITY = "multiparous"
EXPOSURE_DAYS = 200
TOTAL_DAYS = 300
FEED_CONC_STD_UG_PER_KG_DM = 1.0

# Compartments used for kinetic summaries
MILK_COMPARTMENT = "milk"
PLASMA_COMPARTMENT = "plasma"


def _load_passing_compounds(results_root: Path) -> List[Tuple[str, str]]:
    """
    Load compound–isomer pairs that pass global GOF thresholds.

    Expectation: goodness_of_fit_summary_by_compound.csv was produced by
    analysis/goodness_of_fit.py and includes the columns:
      Compound, Isomer, R2, GM_Fold_Error, Bias_log10
    """
    gof_path = (
        results_root
        / "analysis"
        / "goodness_of_fit"
        / "goodness_of_fit_summary_by_compound.csv"
    )
    if not gof_path.exists():
        raise FileNotFoundError(
            f"GOF summary file not found: {gof_path}. "
            "Run analysis/goodness_of_fit.py first."
        )

    df = pd.read_csv(gof_path)
    required_cols = {"Compound", "Isomer", "R2", "GM_Fold_Error", "Bias_log10"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"GOF summary is missing required columns: {sorted(missing)}"
        )

    # Use the same default thresholds as plot_log_pred_vs_obs_for_passing
    crit_r2 = df["R2"] > 0.7
    crit_gm = df["GM_Fold_Error"] < 3.0
    crit_bias = df["Bias_log10"].abs() < 0.3
    passing = df[crit_r2 & crit_gm & crit_bias][["Compound", "Isomer"]]

    pairs: List[Tuple[str, str]] = []
    for _, row in passing.iterrows():
        pairs.append((str(row["Compound"]), str(row["Isomer"])))

    if not pairs:
        logger.warning(
            "[KINETIC] No compound–isomer pairs passed the GOF thresholds; "
            "kinetic characteristics will be empty."
        )
    return pairs


def _load_phase1_fit(
    compound: str,
    isomer: str,
    results_root: Path,
) -> Dict[str, float]:
    """Load Phase 1 fitted parameters as a dict."""
    fit_path = (
        results_root
        / "optimization"
        / "global_fit"
        / f"fit_{compound}_{isomer}.csv"
    )
    if not fit_path.exists():
        raise FileNotFoundError(f"Phase 1 fit file not found: {fit_path}")
    fit_df = pd.read_csv(fit_path)
    return dict(zip(fit_df["Parameter"], fit_df["Value"]))


def _build_intake_function(feed_conc: float, physiology_provider):
    """
    Intake function u(t) = feed_conc * DMI(t) during exposure.
    After EXPOSURE_DAYS, intake is set to zero (clean feed).
    """

    def intake(t: float) -> float:
        if t > EXPOSURE_DAYS:
            return 0.0
        phys = physiology_provider(t)
        dmi = float(phys.get("DMI", 0.0))
        return feed_conc * dmi

    return intake


def _run_deterministic_simulation(
    compound: str,
    isomer: str,
    results_root: Path,
    feed_conc_ug_per_kg_dm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single deterministic simulation for the reference breed/parity.

    Returns:
      time_array (days),
      milk_conc (µg/kg),
      plasma_conc (µg/L or model units)
    """
    fit_params = _load_phase1_fit(compound, isomer, results_root)

    physiology_provider = build_dynamic_physiology_provider(
        breed=REF_BREED,
        parity=REF_PARITY,
        time_unit="days",
    )

    intake_function = _build_intake_function(
        feed_conc=feed_conc_ug_per_kg_dm,
        physiology_provider=physiology_provider,
    )

    config = {"animal": "E2", "compound": compound, "isomer": isomer}
    all_params = build_parameters(config=config, fit_params=fit_params)

    model = PBTKModel(
        params=all_params,
        intake_function=intake_function,
        physiology_provider=physiology_provider,
    )

    # Time grid and initial conditions (zero body burden)
    t_eval = np.arange(0.0, TOTAL_DAYS + 1.0, 1.0)
    A0 = np.zeros(model.compartment_number)
    sim_result = model.simulate_over_time(A0, t_eval)

    # Project plasma amounts and convert to concentrations using V_plasma(t)
    A_matrix = sim_result.mass_matrix  # shape (T, n_compartments)
    pi_plasma = model.projection_vector(PLASMA_COMPARTMENT)
    amount_plasma = A_matrix @ pi_plasma  # shape (T,)

    plasma_conc = np.full_like(amount_plasma, np.nan, dtype=float)
    milk_conc = np.full_like(amount_plasma, np.nan, dtype=float)

    PC = all_params.get("partition_coefficients", {})
    P_milk = PC.get("P_milk", 1.0)

    for i, t_i in enumerate(t_eval):
        phys = physiology_provider(float(t_i))
        V_plasma = float(phys.get("V_plasma", 0.0))
        if V_plasma > 0:
            c_plasma = float(amount_plasma[i]) / V_plasma
            plasma_conc[i] = c_plasma
            milk_conc[i] = P_milk * c_plasma

    return t_eval, milk_conc, plasma_conc


def _time_to_fraction_of_final(
    t: np.ndarray,
    y: np.ndarray,
    start_idx: int,
    end_idx: int,
    fraction: float,
) -> float:
    """
    Time within [start_idx, end_idx] for y to first reach a given fraction
    of its value at end_idx.

    Example:
      - For time to 90% of steady state during exposure:
        start_idx=0, end_idx=index at EXPOSURE_DAYS, fraction=0.9
      - For depuration (decline), y is decreasing; fraction is relative to
        the value at start_idx.
    """
    if end_idx <= start_idx:
        return float("nan")

    y_ref = float(y[end_idx])
    if not np.isfinite(y_ref) or y_ref <= 0:
        return float("nan")

    target = fraction * y_ref
    segment = y[start_idx : end_idx + 1]

    # For increasing trajectory (build‑up), look for y >= target
    # For declining trajectory, caller should reverse indices appropriately.
    if fraction >= 1.0:
        return float("nan")

    if segment.size == 0:
        return float("nan")

    if segment[0] >= target:
        return float(t[start_idx])

    for i in range(1, segment.size):
        if not np.isfinite(segment[i - 1]) or not np.isfinite(segment[i]):
            continue
        if segment[i] >= target:
            t0 = float(t[start_idx + i - 1])
            t1 = float(t[start_idx + i])
            y0 = float(segment[i - 1])
            y1 = float(segment[i])
            if y1 == y0:
                return t1
            frac = (target - y0) / (y1 - y0)
            return t0 + frac * (t1 - t0)

    return float("nan")


def _time_to_fraction_decline(
    t: np.ndarray,
    y: np.ndarray,
    start_idx: int,
    end_idx: int,
    fraction_of_start: float,
) -> float:
    """
    Time after start_idx (within [start_idx, end_idx]) when y first drops
    below fraction_of_start * y[start_idx].
    """
    if end_idx <= start_idx:
        return float("nan")

    y0 = float(y[start_idx])
    if not np.isfinite(y0) or y0 <= 0:
        return float("nan")

    target = fraction_of_start * y0
    segment = y[start_idx : end_idx + 1]

    if segment.size == 0:
        return float("nan")

    if segment[0] <= target:
        return float(t[start_idx])

    for i in range(1, segment.size):
        if not np.isfinite(segment[i - 1]) or not np.isfinite(segment[i]):
            continue
        if segment[i] <= target:
            t0 = float(t[start_idx + i - 1])
            t1 = float(t[start_idx + i])
            y_prev = float(segment[i - 1])
            y_curr = float(segment[i])
            if y_curr == y_prev:
                return t1
            frac = (target - y_prev) / (y_curr - y_prev)
            return t0 + frac * (t1 - t0)

    return float("nan")


def _compute_kinetic_metrics_for_pair(
    compound: str,
    isomer: str,
    results_root: Path,
) -> Dict[str, float]:
    """Run deterministic simulation and derive kinetic metrics for one pair."""
    t, milk, plasma = _run_deterministic_simulation(
        compound=compound,
        isomer=isomer,
        results_root=results_root,
        feed_conc_ug_per_kg_dm=FEED_CONC_STD_UG_PER_KG_DM,
    )

    # Indices for exposure and depuration windows
    exposure_end_day = min(EXPOSURE_DAYS, int(t[-1]))
    idx_exposure_end = int(exposure_end_day)
    idx_dep_start = idx_exposure_end
    idx_dep_end = len(t) - 1

    # Time to 90% and 95% of "steady state" during exposure (relative to value at end of exposure)
    t90_milk = _time_to_fraction_of_final(
        t, milk, start_idx=0, end_idx=idx_exposure_end, fraction=0.9
    )
    t95_milk = _time_to_fraction_of_final(
        t, milk, start_idx=0, end_idx=idx_exposure_end, fraction=0.95
    )
    t90_plasma = _time_to_fraction_of_final(
        t, plasma, start_idx=0, end_idx=idx_exposure_end, fraction=0.9
    )
    t95_plasma = _time_to_fraction_of_final(
        t, plasma, start_idx=0, end_idx=idx_exposure_end, fraction=0.95
    )

    # Time to 50% and 90% decline in milk and plasma after exposure stop
    t50_dep_milk = _time_to_fraction_decline(
        t, milk, start_idx=idx_dep_start, end_idx=idx_dep_end, fraction_of_start=0.5
    )
    t90_dep_milk = _time_to_fraction_decline(
        t, milk, start_idx=idx_dep_start, end_idx=idx_dep_end, fraction_of_start=0.1
    )
    t50_dep_plasma = _time_to_fraction_decline(
        t,
        plasma,
        start_idx=idx_dep_start,
        end_idx=idx_dep_end,
        fraction_of_start=0.5,
    )
    t90_dep_plasma = _time_to_fraction_decline(
        t,
        plasma,
        start_idx=idx_dep_start,
        end_idx=idx_dep_end,
        fraction_of_start=0.1,
    )

    # Approximate steady‑state metrics based on value at end of exposure
    milk_end = float(milk[idx_exposure_end])
    plasma_end = float(plasma[idx_exposure_end])

    # Feed→milk transfer factor: milk concentration per µg/kg DM in feed
    if FEED_CONC_STD_UG_PER_KG_DM > 0.0 and np.isfinite(milk_end):
        transfer_milk = milk_end / FEED_CONC_STD_UG_PER_KG_DM
    else:
        transfer_milk = float("nan")

    # Milk:plasma ratio at the end of exposure
    if plasma_end > 0 and np.isfinite(plasma_end):
        milk_plasma_ratio = milk_end / plasma_end
    else:
        milk_plasma_ratio = float("nan")

    return {
        "Compound": compound,
        "Isomer": isomer,
        "Breed": REF_BREED,
        "Parity": REF_PARITY,
        "Exposure_Days": float(EXPOSURE_DAYS),
        "Total_Days": float(TOTAL_DAYS),
        "Feed_Conc_ug_per_kg_DM": float(FEED_CONC_STD_UG_PER_KG_DM),
        "t90_milk_exposure_days": float(t90_milk),
        "t95_milk_exposure_days": float(t95_milk),
        "t90_plasma_exposure_days": float(t90_plasma),
        "t95_plasma_exposure_days": float(t95_plasma),
        "t50_milk_depuration_days": float(t50_dep_milk),
        "t90_milk_depuration_days": float(t90_dep_milk),
        "t50_plasma_depuration_days": float(t50_dep_plasma),
        "t90_plasma_depuration_days": float(t90_dep_plasma),
        "milk_conc_end_exposure": milk_end,
        "plasma_conc_end_exposure": plasma_end,
        "transfer_factor_milk_per_ug_per_kg_DM": float(transfer_milk),
        "milk_plasma_ratio_end_exposure": float(milk_plasma_ratio),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results_root = get_results_root()

    logger.info("[KINETIC] Loading passing compound–isomer pairs from GOF summary...")
    pairs = _load_passing_compounds(results_root)
    if not pairs:
        logger.warning("[KINETIC] No passing pairs; nothing to do.")
        return

    rows: List[Dict[str, float]] = []
    for compound, isomer in pairs:
        logger.info(
            "[KINETIC] Computing kinetic metrics for %s %s...", compound, isomer
        )
        try:
            metrics = _compute_kinetic_metrics_for_pair(
                compound=compound,
                isomer=isomer,
                results_root=results_root,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "[KINETIC] Failed to compute metrics for %s %s: %s",
                compound,
                isomer,
                exc,
            )
            continue
        rows.append(metrics)

    if not rows:
        logger.warning(
            "[KINETIC] No metrics computed; check logs for individual failures."
        )
        return

    df = pd.DataFrame(rows)
    # Store under a dedicated kinetics subfolder
    out_dir = results_root / "analysis" / "kinetics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kinetic_characteristics.csv"
    df.to_csv(out_path, index=False)
    logger.info("[KINETIC] Saved kinetic characteristics table to %s", out_path)


if __name__ == "__main__":
    main()

