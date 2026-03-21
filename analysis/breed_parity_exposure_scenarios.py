"""
Application 1: Breed × parity × exposure/depuration scenarios.

This script uses the PBTK model to explore how production type and
lactation state influence milk PFAS levels under standardised exposure
patterns.

For each compound–isomer that passes the global goodness‑of‑fit criteria,
and for each (breed, parity) combination, we simulate three scenarios:

  - Scenario A (EXPO_LOW):  constant complete‑feed concentration of
    1 µg/kg DM for the entire simulation horizon.
  - Scenario B (EXPO_HIGH): constant complete‑feed concentration of
    5 µg/kg DM for the entire horizon.
  - Scenario C (EXPO_DEP):  1 µg/kg DM up to EXPOSURE_STOP_DAY, then
    0 µg/kg DM (PFAS‑free feed) for the remainder (depuration).

For each scenario we extract:
  - End‑of‑exposure milk concentration (median and 97.5th percentile
    if Monte Carlo is used, otherwise deterministic value).
For the depuration scenario C we additionally compute:
  - Time from exposure stop until milk concentration falls below the
    regulatory limit (if defined for the compound),
  - Time to 50% and 90% decline relative to the exposure‑stop level.

Results are written to:
  - results/analysis/breed_parity_exposure_scenarios.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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

# Map (compound, isomer) -> whether the Phase 1 fit passed GOF criteria.
GOF_FLAGS: Dict[Tuple[str, str], bool] = {}


# Scenarios (exposure pattern)
SCENARIOS = ("EXPO_LOW", "EXPO_HIGH", "EXPO_DEP")

# Feed types corresponding to the bovine assessment table
FEED_TYPES = ("grass_silage", "maize_silage", "lucerne")

# Default generic complete‑feed concentrations (µg/kg DM) used when no
# compound‑specific information is available or when a value is missing
# in the bovine assessment table.
FEED_CONC_DEFAULT_LOW = 1.0
FEED_CONC_DEFAULT_HIGH = 5.0

# Compound‑ and feed‑specific concentrations (µg/kg, fresh weight) from
# the bovine PFAS assessment (Table 2). Values are taken as LOQs or
# highest detected concentrations. Keys are compound names (upper‑case)
# and feed types as in FEED_TYPES.
FEED_CONC_TABLE: Dict[str, Dict[str, float]] = {
    "PFPEA": {
        "grass_silage": 4.0,
        # maize not analysed
        # lucerne not analysed
    },
    "PFHXA": {
        "grass_silage": 1.5,
        "maize_silage": 1.3,
        "lucerne": 1.5,
    },
    "PFHPA": {
        "grass_silage": 0.15,
        "maize_silage": 0.30,
        "lucerne": 0.10,
    },
    "PFOA": {
        "grass_silage": 0.05,
        "maize_silage": 0.10,
        "lucerne": 0.05,
    },
    "PFNA": {
        "grass_silage": 0.15,
        "maize_silage": 0.30,
        "lucerne": 0.20,
    },
    "PFDA": {
        "grass_silage": 0.50,
        "maize_silage": 0.30,
        "lucerne": 0.20,
    },
    "PFUNDA": {
        "grass_silage": 0.50,
        "maize_silage": 0.30,
        "lucerne": 0.10,
    },
    "PFDODA": {
        "grass_silage": 0.50,
        "maize_silage": 0.10,
        "lucerne": 0.10,
    },
    "PFTRDA": {
        "grass_silage": 0.10,
        "maize_silage": 0.30,
        "lucerne": 0.10,
    },
    "PFTEDA": {
        "grass_silage": 0.05,
        "maize_silage": 0.10,
        "lucerne": 0.20,
    },
    "PFHXDA": {
        "grass_silage": 0.10,
        # maize not analysed
        "lucerne": 0.10,
    },
    "PFBS": {
        "grass_silage": 0.05,
        "maize_silage": 0.20,
        "lucerne": 0.20,
    },
    "PFHXS": {
        "grass_silage": 0.15,
        "maize_silage": 0.10,
        "lucerne": 0.10,
    },
    "PFHPS": {
        "grass_silage": 0.05,
        "maize_silage": 0.10,
        "lucerne": 0.20,
    },
    "PFOS": {
        "grass_silage": 0.15,
        "maize_silage": 0.10,
        # lucerne had two detections at 0.068 and 0.076 µg/kg; we take
        # the higher value here.
        "lucerne": 0.076,
    },
    "PFDS": {
        "grass_silage": 0.20,
        "maize_silage": 0.10,
        "lucerne": 0.20,
    },
    "11CL-PF3OUDS": {
        "grass_silage": 0.50,
        "maize_silage": 0.60,
        "lucerne": 0.50,
    },
    "9CL-PF3ONS": {
        "grass_silage": 1.00,
        "maize_silage": 0.30,
        "lucerne": 1.00,
    },
    "NADONA": {
        "grass_silage": 0.05,
        "maize_silage": 0.20,
        # lucerne not analysed
    },
    "GENX": {
        "grass_silage": 2.00,
        # maize not analysed
        "lucerne": 1.00,
    },
}

BREEDS: List[str] = ["Alpine", "Saanen"]
PARITIES: List[str] = ["primiparous", "multiparous"]

# Time settings (days)
TOTAL_DAYS = 300
EXPOSURE_STOP_DAY = 200  # used in depuration scenario

# Regulatory milk targets (µg/kg fresh weight); align with max_feed_estimation
REGULATORY_LIMITS: Dict[str, float] = {
    "PFOS": 0.02,
    "PFOA": 0.01,
    "PFNA": 0.05,
    "PFHxS": 0.06,
}

MILK_COMPARTMENT = "milk"


def _load_passing_pairs(results_root: Path) -> List[Tuple[str, str]]:
    """
    Load compound–isomer pairs that pass GOF thresholds.
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
    required = {"Compound", "Isomer", "R2", "GM_Fold_Error", "Bias_log10"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"GOF summary missing required columns: {sorted(missing)}"
        )

    crit_r2 = df["R2"] > 0.7
    crit_gm = df["GM_Fold_Error"] < 3.0
    crit_bias = df["Bias_log10"].abs() < 0.3
    mask_pass = crit_r2 & crit_gm & crit_bias

    # Populate global GOF_FLAGS for all compound–isomer pairs
    GOF_FLAGS.clear()
    for idx, row in df.iterrows():
        key = (str(row["Compound"]), str(row["Isomer"]))
        GOF_FLAGS[key] = bool(mask_pass.loc[idx])

    # Return all pairs (including non‑passing); exploratory fits are
    # distinguished via GOF_FLAGS in the output table.
    pairs: List[Tuple[str, str]] = []
    for _, row in df[["Compound", "Isomer"]].iterrows():
        pairs.append((str(row["Compound"]), str(row["Isomer"])))
    if not pairs:
        logger.warning(
            "[BREED_SCENARIOS] No compound–isomer pairs found in GOF summary."
        )
    return pairs


def _load_phase1_fit(
    compound: str,
    isomer: str,
    results_root: Path,
) -> Dict[str, float]:
    fit_path = (
        results_root
        / "optimization"
        / "global_fit"
        / f"fit_{compound}_{isomer}.csv"
    )
    if not fit_path.exists():
        raise FileNotFoundError(f"Phase 1 fit file not found: {fit_path}")
    df = pd.read_csv(fit_path)
    return dict(zip(df["Parameter"], df["Value"]))


def _get_feed_concentrations_for_compound(
    compound: str,
    feed_type: str,
) -> Tuple[float, float]:
    """
    Return (feed_conc_low, feed_conc_high) in µg/kg DM for a compound.

    If compound‑ and feed‑specific values from the bovine assessment
    table are available, those are used (low=high=value). Otherwise the
    generic defaults are returned.
    """
    key = compound.upper()
    ftype = feed_type.lower()
    table = FEED_CONC_TABLE.get(key)
    if table is not None:
        value = table.get(ftype)
        if value is not None:
            # For now we do not distinguish between "low" and "high"
            # within a given feed type; EXPO_LOW and EXPO_HIGH will use
            # the same concentration for that feed.
            return float(value), float(value)
    return FEED_CONC_DEFAULT_LOW, FEED_CONC_DEFAULT_HIGH


def _build_intake_function(
    scenario: str,
    feed_conc_low: float,
    feed_conc_high: float,
    physiology_provider,
):
    """
    Build intake function u(t) = feed_concentration * DMI(t)
    according to the scenario definition.

    - EXPO_LOW:  constant feed_conc_low for all days
    - EXPO_HIGH: constant feed_conc_high for all days
    - EXPO_DEP:  feed_conc_low up to EXPOSURE_STOP_DAY, then 0
    """
    scenario = str(scenario).upper()

    def intake(t: float) -> float:
        phys = physiology_provider(t)
        dmi = float(phys.get("DMI", 0.0))

        if scenario == "EXPO_LOW":
            feed_conc = feed_conc_low
        elif scenario == "EXPO_HIGH":
            feed_conc = feed_conc_high
        elif scenario == "EXPO_DEP":
            feed_conc = feed_conc_low if t <= EXPOSURE_STOP_DAY else 0.0
        else:
            feed_conc = 0.0

        return feed_conc * dmi

    return intake


def _run_deterministic_scenario(
    compound: str,
    isomer: str,
    results_root: Path,
    breed: str,
    parity: str,
    scenario: str,
    feed_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one deterministic simulation for (compound, isomer, breed, parity, scenario).

    Returns:
      t_eval (days), milk_conc (array)
    """
    fit_params = _load_phase1_fit(compound, isomer, results_root)

    physiology_provider = build_dynamic_physiology_provider(
        breed=breed,
        parity=parity,
        time_unit="days",
    )

    feed_conc_low, feed_conc_high = _get_feed_concentrations_for_compound(
        compound=compound,
        feed_type=feed_type,
    )

    intake_function = _build_intake_function(
        scenario=scenario,
        feed_conc_low=feed_conc_low,
        feed_conc_high=feed_conc_high,
        physiology_provider=physiology_provider,
    )

    config = {"animal": "E2", "compound": compound, "isomer": isomer}
    all_params = build_parameters(config=config, fit_params=fit_params)

    model = PBTKModel(
        params=all_params,
        intake_function=intake_function,
        physiology_provider=physiology_provider,
    )

    t_eval = np.arange(0.0, TOTAL_DAYS + 1.0, 1.0)
    A0 = np.zeros(model.compartment_number)
    sim_result = model.simulate_over_time(A0, t_eval)

    # Derive milk "concentration" time course from plasma amounts and partitioning
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


def _time_to_fraction_decline(
    t: np.ndarray,
    y: np.ndarray,
    start_idx: int,
    end_idx: int,
    fraction_of_start: float,
) -> float:
    """
    Time after start_idx when y first drops below fraction_of_start * y[start_idx].
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


def _analyse_depuration_times(
    compound: str,
    t: np.ndarray,
    milk: np.ndarray,
    exposure_stop_day: int,
) -> Dict[str, float]:
    """
    Compute depuration‑related times for the EXPO_DEP scenario.
    """
    idx_start = min(exposure_stop_day, int(t[-1]))
    idx_end = len(t) - 1

    t50 = _time_to_fraction_decline(
        t, milk, start_idx=idx_start, end_idx=idx_end, fraction_of_start=0.5
    )
    t90 = _time_to_fraction_decline(
        t, milk, start_idx=idx_start, end_idx=idx_end, fraction_of_start=0.1
    )

    # Time until milk drops below regulatory limit (if defined)
    limit = REGULATORY_LIMITS.get(compound)
    if limit is None:
        t_below_limit = float("nan")
    else:
        t_below_limit = float("nan")
        for i in range(idx_start, idx_end + 1):
            if not np.isfinite(milk[i]):
                continue
            if milk[i] <= limit:
                t_below_limit = float(t[i])
                break

    return {
        "t50_depuration_days": float(t50),
        "t90_depuration_days": float(t90),
        "t_below_reg_limit_days": float(t_below_limit),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results_root = get_results_root()
    pairs = _load_passing_pairs(results_root)
    if not pairs:
        logger.warning(
            "[BREED_SCENARIOS] No passing pairs; nothing to simulate."
        )
        return

    rows: List[Dict[str, float]] = []

    for compound, isomer in pairs:
        for breed in BREEDS:
            for parity in PARITIES:
                for feed_type in FEED_TYPES:
                    for scenario in SCENARIOS:
                        logger.info(
                            "[BREED_SCENARIOS] Simulating %s %s, breed=%s, parity=%s, feed=%s, scenario=%s",
                            compound,
                            isomer,
                            breed,
                            parity,
                            feed_type,
                            scenario,
                        )
                        try:
                            t, milk = _run_deterministic_scenario(
                                compound=compound,
                                isomer=isomer,
                                results_root=results_root,
                                breed=breed,
                                parity=parity,
                                scenario=scenario,
                                feed_type=feed_type,
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.error(
                                "[BREED_SCENARIOS] Failed for %s %s (%s, %s, %s, %s): %s",
                                compound,
                                isomer,
                                breed,
                                parity,
                                feed_type,
                                scenario,
                                exc,
                            )
                            continue

                        # End‑of‑exposure milk concentration
                        if scenario == "EXPO_DEP":
                            exposure_end_day = min(EXPOSURE_STOP_DAY, int(t[-1]))
                        else:
                            exposure_end_day = int(t[-1])
                        idx_end_expo = min(exposure_end_day, len(t) - 1)
                        milk_end = float(milk[idx_end_expo])

                        row: Dict[str, float] = {
                            "Compound": compound,
                            "Isomer": isomer,
                            "Breed": breed,
                            "Parity": parity,
                            "Feed_Type": feed_type,
                            "Scenario": scenario,
                            "GOF_Passed": float(
                                1.0
                                if GOF_FLAGS.get((compound, isomer), False)
                                else 0.0
                            ),
                            "Exposure_Stop_Day": float(
                                EXPOSURE_STOP_DAY
                                if scenario == "EXPO_DEP"
                                else exposure_end_day
                            ),
                            "Total_Days": float(TOTAL_DAYS),
                            "Milk_Conc_End_Exposure": milk_end,
                        }

                        if scenario == "EXPO_DEP":
                            dep_stats = _analyse_depuration_times(
                                compound=compound,
                                t=t,
                                milk=milk,
                                exposure_stop_day=EXPOSURE_STOP_DAY,
                            )
                            row.update(dep_stats)

                        rows.append(row)

    if not rows:
        logger.warning(
            "[BREED_SCENARIOS] No scenario rows computed; nothing to write."
        )
        return

    df = pd.DataFrame(rows)
    # Store under a dedicated applications subfolder
    out_dir = results_root / "analysis" / "applications"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "breed_parity_exposure_scenarios.csv"
    df.to_csv(out_path, index=False)
    logger.info(
        "[BREED_SCENARIOS] Saved breed/parity exposure scenarios table to %s",
        out_path,
    )


if __name__ == "__main__":
    main()

