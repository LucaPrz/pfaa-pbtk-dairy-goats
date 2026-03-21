"""
Generate a prediction grid plot for all matrices over one lactation cycle.

The user specifies feed concentration (µg/kg DM), compound, isomer, breed,
and parity. The script runs a Monte Carlo PBTK simulation (Hessian-based
parameter sampling, aligned with optimization/mc.py) and produces a grid
of time-series plots (one per matrix: milk, plasma, urine, feces, and
tissue compartments) over the lactation period.

Prediction intervals (default: parameter uncertainty only):
  - param (default): 95% CI from MC over fitted parameters (inverse Hessian)
  - param+animal: add variability across experimental animals (E2, E3, E4)
  - param+animal+obs: add observational (measurement) uncertainty

Usage:
  python analysis/prediction_grid.py --feed 1.0 --compound PFOS --isomer Linear \\
      --breed Alpine --parity primiparous --exposure-start 0 --exposure-end 200
  python analysis/prediction_grid.py -f 5.0 -c PFOA -i Linear -b Saanen -p multiparous \\
      --uncertainty param+animal+obs -o results/figures/pred_grid.png

Output:
  - results/figures/prediction_grid_<compound>_<isomer>.png (or --output path)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_results_root
from model.diagnose import PBTKModel
from model.solve import SimulationResult
from optimization.config import setup_context
from optimization.loss import (
    cumulative_to_daily,
    get_daily_milk,
    predict_excretion_concentration,
    predict_milk_concentration,
)
from optimization.mc import generate_mc_parameter_samples_hessian
from parameters.parameters import build_parameters, build_dynamic_physiology_provider

logger = logging.getLogger(__name__)

# Default lactation cycle length (days)
DEFAULT_DAYS = 300

# Monte Carlo defaults
DEFAULT_MC_SAMPLES = 500
MC_RANDOM_SEED = 42

# Matrices to plot (excretion streams + tissue compartments)
# Order: plasma first (central compartment), then milk (derived from plasma), then excretions
EXCRETION_MATRICES = ["plasma", "milk", "urine", "feces"]
TISSUE_MATRICES = ["liver", "kidney", "muscle", "heart", "brain", "spleen", "lung", "rest"]
ALL_MATRICES = EXCRETION_MATRICES + TISSUE_MATRICES

# Fallback excretion values when processed data is unavailable (representative goat)
DEFAULT_URINE_VOLUME_L_PER_DAY = 2.0
DEFAULT_FECES_MASS_KG_PER_DAY = 0.5

BREEDS = ["Alpine", "Saanen"]
PARITIES = ["primiparous", "multiparous"]

# Uncertainty modes
UNCERTAINTY_MODES = ("param", "param+animal", "param+animal+obs")


def _load_phase1_fit(compound: str, isomer: str, results_root: Path) -> Dict[str, float]:
    """Load Phase 1 fitted parameters."""
    fit_path = (
        results_root / "optimization" / "global_fit" / f"fit_{compound}_{isomer}.csv"
    )
    if not fit_path.exists():
        raise FileNotFoundError(
            f"Phase 1 fit not found: {fit_path}. "
            f"Run optimization for {compound} {isomer} first."
        )
    df = pd.read_csv(fit_path)
    return dict(zip(df["Parameter"], df["Value"]))


def _load_urine_feces_defaults(project_root: Path) -> Tuple[float, float]:
    """Load median urine volume and feces mass from processed data if available."""
    try:
        from optimization.io import _load_urine_and_feces

        urine_by_animal, feces_by_animal, feces_default = _load_urine_and_feces(
            project_root
        )
        urine_vals = [v for v in urine_by_animal.values() if np.isfinite(v)]
        feces_vals = [v for v in feces_by_animal.values() if np.isfinite(v)]
        urine = float(np.median(urine_vals)) if urine_vals else DEFAULT_URINE_VOLUME_L_PER_DAY
        feces = feces_default if np.isfinite(feces_default) else (
            float(np.median(feces_vals)) if feces_vals else DEFAULT_FECES_MASS_KG_PER_DAY
        )
        return urine, feces
    except Exception:
        return DEFAULT_URINE_VOLUME_L_PER_DAY, DEFAULT_FECES_MASS_KG_PER_DAY


def _extend_animal_arrays_to_days(
    milk_by_animal: Dict[str, np.ndarray],
    bw_by_animal: Dict[str, np.ndarray],
    total_days: int,
) -> None:
    """Extend milk yield and body weight arrays to total_days+1 (in place)."""
    for animal in list(milk_by_animal.keys()):
        arr = milk_by_animal[animal]
        if len(arr) < total_days + 1:
            ext = np.zeros(total_days + 1, dtype=float)
            ext[: len(arr)] = arr
            last_val = float(np.max(arr)) if np.any(arr > 0) else 1e-10
            ext[len(arr) :] = last_val
            milk_by_animal[animal] = ext
    for animal in list(bw_by_animal.keys()):
        arr = bw_by_animal[animal]
        if len(arr) < total_days + 1:
            ext = np.full(total_days + 1, np.nan)
            ext[: len(arr)] = arr
            valid = arr[~np.isnan(arr)]
            last_val = float(valid[-1]) if len(valid) > 0 else 60.0
            ext[len(arr) :] = last_val
            bw_by_animal[animal] = ext


def _get_animal_physiology_from_context(context, total_days: int):
    """
    Get milk yield, body weight, urine, feces per animal from FittingContext.
    Extends arrays to total_days+1 if needed (context uses config.time_vector by default).
    """
    milk_by_animal = {k: v.copy() for k, v in context.milk_yield_by_animal.items()}
    bw_by_animal = {k: v.copy() for k, v in context.body_weight_by_animal.items()}
    _extend_animal_arrays_to_days(milk_by_animal, bw_by_animal, total_days)
    return (
        milk_by_animal,
        bw_by_animal,
        context.urine_volume_by_animal,
        context.feces_mass_by_animal,
        context.feces_mass_default,
    )


def _build_intake_function(
    feed_concentration: float,
    physiology_provider,
    exposure_start: float,
    exposure_end: float,
    total_days: int = DEFAULT_DAYS,
) -> callable:
    """Intake = feed_concentration * DMI(t) for t in [exposure_start, exposure_end]."""

    def intake(t: float) -> float:
        if t < 0 or t > total_days:
            return 0.0
        if t < exposure_start or t > exposure_end:
            return 0.0
        phys = physiology_provider(t)
        dmi = float(phys.get("DMI", 0.0))
        return feed_concentration * dmi

    return intake


def _extract_matrix_time_series(
    sim_result: SimulationResult,
    all_params: Dict,
    physiology_provider,
    milk_yield_array: np.ndarray,
    urine_volume: float,
    feces_mass: float,
    animal: str = "ref",
) -> Dict[str, np.ndarray]:
    """
    Extract concentration time series for all matrices from simulation result.

    Returns dict: matrix_name -> array of concentrations (or mass rates for elim).
    """
    t_array = sim_result.time_array
    n = len(t_array)

    # Synthetic animal lookup for milk/urine/feces
    milk_yield_by_animal = {animal: milk_yield_array}
    urine_volume_by_animal = {animal: urine_volume}
    feces_mass_by_animal = {animal: feces_mass}
    feces_mass_default = feces_mass

    result: Dict[str, np.ndarray] = {}

    # Milk
    cum_milk = sim_result.milk_array
    daily_milk = get_daily_milk(cum_milk)
    milk_conc = predict_milk_concentration(
        daily_milk, animal, milk_yield_by_animal
    )
    result["milk"] = np.maximum(milk_conc, 0.0)

    # Plasma (from mass_matrix)
    ci = PBTKModel.compartment_idx
    if "plasma" in ci:
        idx = ci["plasma"]
        phys_dict = physiology_provider(0.0)
        V_plasma = phys_dict.get("V_plasma", 0.0)
        if V_plasma <= 0:
            result["plasma"] = np.full(n, np.nan)
        else:
            amounts = sim_result.mass_matrix[:, idx]
            # V_plasma varies with time; get per-day
            plasma_conc = np.zeros(n)
            for i in range(n):
                phys = physiology_provider(float(t_array[i]))
                v = float(phys.get("V_plasma", 0.0))
                plasma_conc[i] = amounts[i] / v if v > 0 else np.nan
            result["plasma"] = np.maximum(plasma_conc, 0.0)

    # Urine
    cum_urine = sim_result.urine_array
    daily_urine = cumulative_to_daily(cum_urine)
    urine_conc = predict_excretion_concentration(
        daily_urine, animal, "urine",
        urine_volume_by_animal, feces_mass_by_animal, feces_mass_default,
    )
    result["urine"] = np.maximum(urine_conc, 0.0)

    # Feces
    cum_feces = sim_result.feces_array
    daily_feces = cumulative_to_daily(cum_feces)
    feces_conc = predict_excretion_concentration(
        daily_feces, animal, "feces",
        urine_volume_by_animal, feces_mass_by_animal, feces_mass_default,
    )
    result["feces"] = np.maximum(feces_conc, 0.0)

    # Tissue compartments
    vol_keys = {
        "liver": "V_liver",
        "kidney": "V_kidney",
        "muscle": "V_muscle",
        "heart": "V_heart",
        "brain": "V_brain",
        "spleen": "V_spleen",
        "lung": "V_lung",
        "rest": "V_rest",
    }
    for comp, vol_key in vol_keys.items():
        if comp not in ci:
            continue
        idx = ci[comp]
        amounts = sim_result.mass_matrix[:, idx]
        conc = np.zeros(n)
        for i in range(n):
            phys = physiology_provider(float(t_array[i]))
            v = float(phys.get(vol_key, 0.0))
            conc[i] = amounts[i] / v if v > 0 else np.nan
        result[comp] = np.maximum(conc, 0.0)

    return result


def _run_single_simulation(
    fit_params: Dict[str, float],
    compound: str,
    isomer: str,
    physiology_provider,
    intake_function,
    t_eval: np.ndarray,
    milk_yield_array: np.ndarray,
    urine_volume: float,
    feces_mass: float,
    animal: str = "ref",
) -> Dict[str, np.ndarray]:
    """Run one simulation and return matrix time series."""
    config = {"animal": "E2", "compound": compound, "isomer": isomer}
    all_params = build_parameters(config=config, fit_params=fit_params)
    model = PBTKModel(
        params=all_params,
        intake_function=intake_function,
        physiology_provider=physiology_provider,
    )
    A0 = np.zeros(model.compartment_number)
    sim_result = model.simulate_over_time(A0, t_eval)
    return _extract_matrix_time_series(
        sim_result,
        all_params,
        physiology_provider,
        milk_yield_array,
        urine_volume,
        feces_mass,
        animal=animal,
    )


def run_prediction(
    feed_concentration: float,
    compound: str,
    isomer: str,
    breed: str,
    parity: str,
    exposure_start: float = 0.0,
    exposure_end: Optional[float] = None,
    total_days: int = DEFAULT_DAYS,
    results_root: Optional[Path] = None,
    project_root: Optional[Path] = None,
    include_animal: bool = False,
    include_obs: bool = False,
    mc_samples: int = DEFAULT_MC_SAMPLES,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[Dict[str, Dict[str, np.ndarray]]]]:
    """
    Run simulation (deterministic or MC) and return time array and matrix time series.

    Returns:
        t_array: time points (days)
        matrix_series: dict matrix_name -> concentration array (median or deterministic)
        ci_intervals: if MC, dict matrix_name -> {lower, upper} or {lower, upper, lower_obs, upper_obs}
    """
    if results_root is None:
        results_root = get_results_root()
    if project_root is None:
        project_root = PROJECT_ROOT

    fit_params = _load_phase1_fit(compound, isomer, results_root)
    t_eval = np.arange(0.0, total_days + 1.0, 1.0)
    exp_end = exposure_end if exposure_end is not None else float(total_days)

    # MC: parameter uncertainty via inverse Hessian (same as optimization/mc.py)
    context = setup_context(project_root, verbose=False)
    result = generate_mc_parameter_samples_hessian(
        compound, isomer, context,
        n_samples=mc_samples,
        random_seed=MC_RANDOM_SEED,
    )
    if result is None:
        raise RuntimeError(
            f"Hessian-based MC sampling failed for {compound} {isomer}. "
            "Ensure Phase 1 fit exists and the compound–isomer has fittable data."
        )
    mc_params, param_names = result

    all_series: Dict[str, List[np.ndarray]] = {m: [] for m in ALL_MATRICES}

    if include_animal:
        milk_by_animal, bw_by_animal, urine_by_animal, feces_by_animal, feces_default = _get_animal_physiology_from_context(
            context, total_days
        )
        animals = [a for a in milk_by_animal if a in bw_by_animal]
        if not animals:
            animals = ["E2", "E3", "E4"]  # fallback
    else:
        physiology_provider = build_dynamic_physiology_provider(
            breed=breed, parity=parity, time_unit="days"
        )
        milk_yield_array = np.array([
            float(physiology_provider(t).get("milk_yield", 0.0)) for t in t_eval
        ], dtype=float)
        milk_yield_array = np.maximum(milk_yield_array, 1e-10)
        urine_vol, feces_mass = _load_urine_feces_defaults(project_root)
        animals = ["ref"]

    rng = np.random.default_rng(MC_RANDOM_SEED)

    for sample_idx in range(mc_samples):
        mc_fit = fit_params.copy()
        mc_fit.update(dict(zip(param_names, mc_params[sample_idx])))

        for animal in animals:
            if include_animal:
                physiology_provider = build_dynamic_physiology_provider(
                    breed=breed,
                    parity=parity,
                    time_unit="days",
                    body_weight_array=bw_by_animal.get(animal),
                    milk_yield_array=milk_by_animal.get(animal),
                )
                milk_yield_array = milk_by_animal[animal]
                urine_vol = float(urine_by_animal.get(animal, 2.0))
                feces_mass = float(feces_by_animal.get(animal, feces_default))
            # else: physiology_provider, milk_yield_array, urine_vol, feces_mass from outer scope

            intake_function = _build_intake_function(
                feed_concentration, physiology_provider,
                exposure_start=exposure_start, exposure_end=exp_end, total_days=total_days,
            )
            try:
                series = _run_single_simulation(
                    mc_fit, compound, isomer,
                    physiology_provider, intake_function, t_eval,
                    milk_yield_array, urine_vol, feces_mass,
                    animal=animal,
                )
                for m, arr in series.items():
                    all_series[m].append(arr.astype(np.float32))
            except Exception as e:
                logger.warning("MC sample %d animal %s failed: %s", sample_idx, animal, e)

    # Aggregate: median and 95% CI. Compute percentiles in LOG space so the median
    # appears in the visual middle on the log-scale plot (linear-space percentiles
    # can make the median look off-center when displayed on log y-axis).
    matrix_series = {}
    ci_intervals: Dict[str, Dict[str, np.ndarray]] = {}
    eps = 1e-10

    for m in ALL_MATRICES:
        if not all_series[m]:
            continue
        stacked = np.vstack(all_series[m])
        stacked_safe = np.maximum(stacked, eps)
        log_stacked = np.log(stacked_safe)
        # Percentiles in log space -> median visually centered on log plot
        lower_log = np.nanpercentile(log_stacked, 2.5, axis=0)
        median_log = np.nanpercentile(log_stacked, 50.0, axis=0)
        upper_log = np.nanpercentile(log_stacked, 97.5, axis=0)
        lower = np.exp(lower_log)
        median = np.exp(median_log)
        upper = np.exp(upper_log)
        matrix_series[m] = median
        ci_intervals[m] = {"lower": lower, "upper": upper}

        if include_obs:
            sigma = context.config.get_sigma(m, compound=compound, isomer=isomer)
            log_pred = np.log(np.maximum(stacked, eps))
            noise = rng.normal(0, sigma, size=stacked.shape)
            obs_draws = np.exp(log_pred + noise)
            lower_obs = np.nanpercentile(obs_draws, 2.5, axis=0)
            upper_obs = np.nanpercentile(obs_draws, 97.5, axis=0)
            ci_intervals[m]["lower_obs"] = lower_obs
            ci_intervals[m]["upper_obs"] = upper_obs

    return t_eval, matrix_series, ci_intervals


def plot_prediction_grid(
    t_array: np.ndarray,
    matrix_series: Dict[str, np.ndarray],
    compound: str,
    isomer: str,
    feed_concentration: float,
    breed: str,
    parity: str,
    exposure_start: float = 0.0,
    exposure_end: Optional[float] = None,
    ci_intervals: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    uncertainty_mode: str = "param",
    output_path: Path = None,  # type: ignore[assignment]
) -> None:
    """Create a grid of subplots, one per matrix, with optional CI bands."""
    matrices = [m for m in ALL_MATRICES if m in matrix_series]
    n_matrices = len(matrices)
    if n_matrices == 0:
        raise ValueError("No matrix data to plot")

    n_cols = 4
    n_rows = (n_matrices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    exp_end = exposure_end if exposure_end is not None else t_array[-1]
    title = (
        f"{compound} {isomer} | Feed: {feed_concentration} µg/kg DM | "
        f"Exposure: day {exposure_start:.0f}–{exp_end:.0f} | {breed} {parity}"
    )
    if ci_intervals:
        title += f" | 95% CI ({uncertainty_mode}) | median"
    fig.suptitle(title, fontsize=12, y=1.02)

    for idx, matrix in enumerate(matrices):
        ax = axes_flat[idx]
        # Shade exposure window
        if exposure_start < exp_end:
            ax.axvspan(exposure_start, exp_end, alpha=0.15, color="gray", zorder=0)
        y = matrix_series[matrix]

        # CI bands (back to front: obs, param+animal, param)
        if ci_intervals and matrix in ci_intervals:
            ci = ci_intervals[matrix]
            if "lower_obs" in ci and "upper_obs" in ci:
                lo_obs = np.maximum(ci["lower_obs"], 0.0)
                hi_obs = np.maximum(ci["upper_obs"], 0.0)
                ax.fill_between(
                    t_array, lo_obs, hi_obs,
                    alpha=0.2, color="tab:blue", zorder=1,
                    label="95% obs-level" if idx == 0 else None,
                )
            lo = np.maximum(ci["lower"], 0.0)
            hi = np.maximum(ci["upper"], 0.0)
            ax.fill_between(
                t_array, lo, hi,
                alpha=0.35, color="tab:orange", zorder=2,
                label="95% param CI" if idx == 0 else None,
            )

        ax.plot(t_array, np.maximum(y, 0.0), color="black", lw=1.5, zorder=3)
        ax.set_title(matrix.capitalize(), fontsize=10)
        ax.set_ylabel("µg/kg or µg/L")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t_array[-1])

        # For near-zero excretion (no urinary/faecal route), avoid misleading 1e-10
        # offset display: use fixed ylim and show 0 clearly instead of 0.96–1.04×1e-10
        NEAR_ZERO_THRESHOLD = 1e-6
        if np.nanmax(y) < NEAR_ZERO_THRESHOLD:
            ax.set_ylim(0, NEAR_ZERO_THRESHOLD)
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))

    for idx in range(n_matrices, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[-1].set_xlabel("Days in milk")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prediction grid to %s", output_path)


def _list_available_pairs(results_root: Path) -> None:
    """Print available compound–isomer pairs (Phase 1 fits)."""
    fit_dir = results_root / "optimization" / "global_fit"
    if not fit_dir.exists():
        print("No global_fit directory found. Run optimization first.")
        return
    files = sorted(fit_dir.glob("fit_*.csv"))
    if not files:
        print("No Phase 1 fits found.")
        return
    print("Available compound–isomer pairs (Phase 1 fits):")
    for f in files:
        stem = f.stem  # e.g. "fit_PFOS_Linear"
        suffix = stem[4:] if stem.startswith("fit_") else stem  # "PFOS_Linear"
        parts = suffix.rsplit("_", 1)
        compound, isomer = (parts[0], parts[1]) if len(parts) == 2 else (suffix, "?")
        print(f"  {compound} {isomer}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prediction grid plot for all matrices over one lactation cycle."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available compound–isomer pairs and exit",
    )
    parser.add_argument(
        "-f", "--feed",
        type=float,
        help="Feed concentration (µg/kg DM)",
    )
    parser.add_argument(
        "-c", "--compound",
        type=str,
        help="Compound name (e.g. PFOS, PFOA)",
    )
    parser.add_argument(
        "-i", "--isomer",
        type=str,
        help="Isomer (e.g. Linear, Branched)",
    )
    parser.add_argument(
        "-b", "--breed",
        type=str,
        choices=BREEDS,
        help="Breed (Alpine or Saanen)",
    )
    parser.add_argument(
        "-p", "--parity",
        type=str,
        choices=PARITIES,
        help="Parity (primiparous or multiparous)",
    )
    parser.add_argument(
        "-d", "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Lactation cycle length in days (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--exposure-start",
        type=float,
        default=0.0,
        help="Exposure start day (default: 0)",
    )
    parser.add_argument(
        "--exposure-end",
        type=float,
        default=None,
        help="Exposure end day (default: same as --days, i.e. full lactation)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for figure (default: results/figures/prediction_grid_<compound>_<isomer>.png)",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional: save matrix time series to CSV",
    )
    parser.add_argument(
        "--uncertainty",
        type=str,
        choices=UNCERTAINTY_MODES,
        default="param",
        help="Uncertainty for prediction intervals: param (default), param+animal, param+animal+obs",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=DEFAULT_MC_SAMPLES,
        help=f"Number of Monte Carlo samples for parameter uncertainty (default: {DEFAULT_MC_SAMPLES})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results_root = get_results_root()

    if args.list:
        _list_available_pairs(results_root)
        return

    required = ["feed", "compound", "isomer", "breed", "parity"]
    missing = [r for r in required if getattr(args, r) is None]
    if missing:
        parser.error(
            f"The following arguments are required for prediction: {', '.join(missing)}. "
            "Use --list to see available compound–isomer pairs."
        )
    project_root = PROJECT_ROOT

    include_animal = "animal" in args.uncertainty
    include_obs = "obs" in args.uncertainty

    t_array, matrix_series, ci_intervals = run_prediction(
        feed_concentration=args.feed,
        compound=args.compound,
        isomer=args.isomer,
        breed=args.breed,
        parity=args.parity,
        exposure_start=args.exposure_start,
        exposure_end=args.exposure_end,
        total_days=args.days,
        results_root=results_root,
        project_root=project_root,
        include_animal=include_animal,
        include_obs=include_obs,
        mc_samples=args.mc_samples,
    )

    if args.output is None:
        args.output = (
            results_root
            / "figures"
            / f"prediction_grid_{args.compound}_{args.isomer}.png"
        )

    plot_prediction_grid(
        t_array,
        matrix_series,
        compound=args.compound,
        isomer=args.isomer,
        feed_concentration=args.feed,
        breed=args.breed,
        parity=args.parity,
        exposure_start=args.exposure_start,
        exposure_end=args.exposure_end if args.exposure_end is not None else args.days,
        ci_intervals=ci_intervals,
        uncertainty_mode=args.uncertainty,
        output_path=args.output,
    )

    if args.save_csv is not None:
        rows: List[Dict] = []
        for day_idx, t in enumerate(t_array):
            row = {"Day": t}
            for matrix, series in matrix_series.items():
                row[matrix] = series[day_idx] if day_idx < len(series) else np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(args.save_csv, index=False)
        logger.info("Saved time series to %s", args.save_csv)


if __name__ == "__main__":
    main()
