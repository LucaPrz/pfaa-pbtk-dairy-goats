"""
Estimate maximum complete-feed PFAS concentrations that keep the
95% CI of milk concentration below regulatory limits.

This script:
  1. Uses Phase 1 fits + jackknife results to build a multivariate
     parameter distribution per compound–isomer.
  2. Runs Monte Carlo simulations for hypothetical constant complete-feed
     concentrations for different breed/parity combinations.
  3. Finds, by linear scaling, the maximum feed concentration for which
     the 97.5th percentile of milk concentrations never exceeds the
     regulatory limit.

Results are written to:
  - results/analysis/max_feed_concentrations.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import logging

# Ensure project root is on sys.path when executed via absolute path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_results_root
from parameters.parameters import build_parameters, build_dynamic_physiology_provider
from model.diagnose import PBTKModel

logger = logging.getLogger(__name__)


# Regulatory milk targets (µg/kg fresh weight)
REGULATORY_LIMITS: Dict[str, float] = {
    "PFOS": 0.02,
    "PFOA": 0.01,
    "PFNA": 0.05,
    "PFHxS": 0.06,
}

# Compounds to analyze (only Linear isomers for the table)
COMPOUNDS: List[str] = ["PFOS", "PFOA", "PFNA", "PFHxS"]
ISOMER: str = "Linear"

# Breed and parity combinations – dairy goats + simple Holstein profile
BREEDS: List[str] = ["Alpine", "Saanen", "Holstein_cow"]
PARITIES: List[str] = ["primiparous", "multiparous"]

# Simulation settings
DAYS: int = 300
EXPOSURE_PERIOD: int = 200
MC_SAMPLES: int = 500  # Monte Carlo samples per evaluation


def load_phase1_fit(
    compound: str, isomer: str, results_root: Path
) -> Dict[str, float]:
    """Load Phase 1 fitted parameters as a dict."""
    fit_path = results_root / "optimization" / "global_fit" / f"fit_{compound}_{isomer}.csv"
    if not fit_path.exists():
        raise FileNotFoundError(f"Phase 1 fit file not found: {fit_path}")
    fit_df = pd.read_csv(fit_path)
    return dict(zip(fit_df["Parameter"], fit_df["Value"]))


def load_jackknife_results(
    compound: str, isomer: str, results_root: Path
) -> pd.DataFrame:
    """Load Phase 2 jackknife results as a DataFrame."""
    jk_path = (
        results_root
        / "optimization"
        / "jackknife"
        / f"jackknife_{compound}_{isomer}_LOAO.csv"
    )
    if not jk_path.exists():
        raise FileNotFoundError(f"Jackknife file not found: {jk_path}")
    return pd.read_csv(jk_path)


def get_param_names(compound: str, isomer: str) -> List[str]:
    """
    Determine parameter names for the given compound–isomer pair using
    the same logic as the optimisation pipeline.
    """
    from optimization.config import ModelConfig
    from optimization.fit_variables import check_data_signals

    # Load data once here (clean data path)
    data_path = PROJECT_ROOT / "data" / "raw" / "pfas_data_no_e1.csv"
    data_df = pd.read_csv(data_path)

    config = ModelConfig()

    # Reuse ModelConfig.get_param_names, which in turn asks fit_variables
    pair_data = data_df[
        (data_df["Compound"] == compound) & (data_df["Isomer"] == isomer)
    ]
    if pair_data.empty:
        # Fallback: use default param_names
        return list(config.param_names)

    # This call will internally use get_parameter_config and data signals
    return config.get_param_names(compound, isomer, data_df)


def generate_mc_parameter_samples(
    phase1_params: Dict[str, float],
    jackknife_df: pd.DataFrame,
    param_names: List[str],
    n_samples: int,
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate Monte Carlo samples in log10-space using jackknife covariance.

    Returns:
        mc_params: array of shape (n_samples, n_params) in linear space.
    """
    if not param_names:
        raise ValueError("param_names is empty – nothing to sample.")

    # Extract jackknife fits for these parameters
    jk_vals_linear = jackknife_df[param_names].values.astype(float)
    if jk_vals_linear.ndim != 2 or jk_vals_linear.shape[1] != len(param_names):
        raise ValueError("Jackknife parameter mismatch.")

    eps_param = 1e-12
    phase1_vec = np.array(
        [phase1_params.get(name, 1.0) for name in param_names], dtype=float
    )
    phase1_log = np.log10(np.clip(phase1_vec, eps_param, None))
    jk_log = np.log10(np.clip(jk_vals_linear, eps_param, None))

    # Covariance in log10-space from jackknife
    n_jk = jk_log.shape[0]
    if n_jk > 1:
        z_mean = np.mean(jk_log, axis=0)
        z_centered = jk_log - z_mean
        cov = ((n_jk - 1) / n_jk) * np.dot(z_centered.T, z_centered)
        # Regularise if needed
        min_eig = float(np.min(np.linalg.eigvals(cov)))
        if min_eig < 1e-8:
            cov += np.eye(len(param_names)) * 1e-8
    else:
        # Diagonal covariance if only one jackknife sample
        stds_log = np.std(jk_log, axis=0, ddof=0)
        cov = np.diag(np.maximum(stds_log**2, 1e-6))

    rng = np.random.default_rng(random_seed)
    try:
        samples_log = rng.multivariate_normal(phase1_log, cov, size=n_samples)
    except np.linalg.LinAlgError:
        # Fallback: diagonal only
        diag_cov = np.diag(np.diag(cov))
        samples_log = rng.multivariate_normal(phase1_log, diag_cov, size=n_samples)

    # Transform back to linear space
    return 10.0 ** samples_log


def build_intake_function(
    feed_concentration: float,
    physiology_provider,
    exposure_period: int = EXPOSURE_PERIOD,
) -> callable:
    """
    Build an intake function u(t) = feed_concentration * DMI(t) during exposure.

    DMI(t) is obtained from the physiology provider (kg/day).
    """

    def intake(t: float) -> float:
        if t > exposure_period:
            return 0.0
        phys = physiology_provider(t)
        dmi = float(phys.get("DMI", 0.0))
        return feed_concentration * dmi

    return intake


def get_daily_from_cumulative(cumulative: np.ndarray) -> np.ndarray:
    """
    Convert cumulative excretion array to daily excretion.

    Assumes constant daily steps (1 day between entries).
    """
    cumulative = np.asarray(cumulative, dtype=float).reshape(-1)
    if cumulative.size == 0:
        return cumulative
    daily = np.diff(cumulative, prepend=cumulative[0])
    daily[0] = cumulative[0]
    return np.maximum(daily, 0.0)


def calculate_milk_concentration_ci(
    compound: str,
    isomer: str,
    breed: str,
    parity: str,
    feed_concentration: float,
    results_root: Path,
    mc_samples: int = MC_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate milk concentration time series with 95% CI using Monte Carlo.

    Returns:
        time_array: Time points (days)
        median: Median milk concentration (µg/kg)
        ci_upper: 97.5th percentile (upper bound of 95% CI) (µg/kg)
    """
    # Load Phase 1 and jackknife results
    fit_params = load_phase1_fit(compound, isomer, results_root)
    jackknife_df = load_jackknife_results(compound, isomer, results_root)
    param_names = get_param_names(compound, isomer)

    # MC parameter samples in linear space
    mc_params = generate_mc_parameter_samples(
        phase1_params=fit_params,
        jackknife_df=jackknife_df,
        param_names=param_names,
        n_samples=mc_samples,
        random_seed=42,
    )

    # Physiology provider for chosen breed/parity
    physiology_provider = build_dynamic_physiology_provider(
        breed=breed,
        parity=parity,
        time_unit="days",
    )

    # Intake function: feed_concentration * DMI(t) during exposure
    intake_function = build_intake_function(
        feed_concentration=feed_concentration,
        physiology_provider=physiology_provider,
        exposure_period=EXPOSURE_PERIOD,
    )

    # Time grid
    time_array = np.arange(0, DAYS + 1, 1.0)

    all_milk_concentrations: List[List[float]] = []

    for idx in range(mc_samples):
        # Combine Phase 1 params with MC sample
        mc_fit_params = fit_params.copy()
        for j, name in enumerate(param_names):
            mc_fit_params[name] = mc_params[idx, j]

        # Build full parameter dict (mostly for partitioning/AE; physiology is dynamic)
        config = {"animal": "E2", "compound": compound, "isomer": isomer}
        all_params = build_parameters(config=config, fit_params=mc_fit_params)

        # Construct model
        model = PBTKModel(
            params=all_params,
            intake_function=intake_function,
            physiology_provider=physiology_provider,
        )

        A0 = np.zeros(len(model.compartment_idx))
        result = model.simulate_over_time(A0, time_array)

        milk_conc_series: List[float] = []
        for t_idx, t in enumerate(time_array):
            if t_idx > 0:
                daily_milk_cum = result.milk_array[: t_idx + 1]
                daily_milk = get_daily_from_cumulative(daily_milk_cum)[-1]
                phys = physiology_provider(t)
                milk_yield = float(phys.get("milk_yield", 0.0))
                if milk_yield > 0:
                    milk_conc = daily_milk / milk_yield
                else:
                    milk_conc = 0.0
            else:
                milk_conc = 0.0
            milk_conc_series.append(milk_conc)

        all_milk_concentrations.append(milk_conc_series)

    all_milk_concentrations_arr = np.array(all_milk_concentrations, dtype=float)
    median = np.median(all_milk_concentrations_arr, axis=0)
    ci_upper = np.percentile(all_milk_concentrations_arr, 97.5, axis=0)

    return time_array, median, ci_upper


def find_max_feed_concentration(
    compound: str,
    isomer: str,
    breed: str,
    parity: str,
    regulatory_limit: float,
    results_root: Path,
    test_concentration: float = 0.1,
    mc_samples: int = MC_SAMPLES,
) -> float:
    """
    Find maximum feed concentration (µg/kg DMI) such that the 95% CI upper bound
    of milk concentration never exceeds the regulatory limit.

    Assumes linear scaling of milk concentration with feed concentration.
    """
    # CI at test concentration
    _, _, ci_upper = calculate_milk_concentration_ci(
        compound,
        isomer,
        breed,
        parity,
        test_concentration,
        results_root,
        mc_samples=mc_samples,
    )
    max_ci = float(np.max(ci_upper))

    if max_ci <= 0:
        return 0.0

    max_feed = test_concentration * (regulatory_limit / max_ci)

    # Verify and add small safety margin if necessary
    _, _, ci_upper_verify = calculate_milk_concentration_ci(
        compound,
        isomer,
        breed,
        parity,
        max_feed,
        results_root,
        mc_samples=mc_samples,
    )
    max_ci_verify = float(np.max(ci_upper_verify))

    if max_ci_verify > regulatory_limit:
        safety_factor = regulatory_limit / max_ci_verify * 0.99  # 1% margin
        max_feed *= safety_factor

    result = np.floor(max_feed * 100.0) / 100.0
    return max(0.0, float(result))


def main() -> None:
    """Calculate maximum feed concentrations for all compound–breed–parity combinations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results_root = get_results_root()

    print("=" * 70)
    print("CALCULATING MAXIMUM FEED CONCENTRATIONS")
    print("=" * 70)
    print(f"Regulatory limits (µg/kg milk): {REGULATORY_LIMITS}")
    print(f"MC samples per calculation: {MC_SAMPLES}")
    print("=" * 70)

    rows: List[Dict[str, any]] = []

    for compound in COMPOUNDS:
        if compound not in REGULATORY_LIMITS:
            print(f"\nWarning: No regulatory limit for {compound}, skipping...")
            continue

        regulatory_limit = REGULATORY_LIMITS[compound]
        print(f"\n{compound} {ISOMER} (limit: {regulatory_limit} µg/kg milk):")

        for breed in BREEDS:
            for parity in PARITIES:
                print(f"  {breed}, {parity}...", end=" ", flush=True)
                try:
                    max_conc = find_max_feed_concentration(
                        compound=compound,
                        isomer=ISOMER,
                        breed=breed,
                        parity=parity,
                        regulatory_limit=regulatory_limit,
                        results_root=results_root,
                        test_concentration=0.1,
                        mc_samples=MC_SAMPLES,
                    )

                    rows.append(
                        {
                            "compound": compound,
                            "isomer": ISOMER,
                            "breed": breed,
                            "parity": parity,
                            "max_feed_concentration_ug_kg_DMI": max_conc,
                            "regulatory_limit_ug_kg_milk": regulatory_limit,
                        }
                    )
                    print(f"✓ {max_conc:.2f} µg/kg DMI")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    logger.exception(
                        "Error while computing max feed concentration for "
                        f"{compound} {ISOMER}, {breed}, {parity}"
                    )

    df_results = pd.DataFrame(rows)

    analysis_root = results_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    out_path = analysis_root / "max_feed_concentrations.csv"
    df_results.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print(f"Results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

