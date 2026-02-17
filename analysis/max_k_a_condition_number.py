"""
Find the maximum k_a (absorption rate) that keeps the ODE transition matrix T
well-conditioned (cond(T) < ILL_CONDITIONED_THRESHOLD = 1e12).

Useful when fixing k_a for "instant absorption" to reduce k_a–k_ehc correlation:
we need a finite k_a that is high enough to approximate instant absorption but
not so high that the solver hits matrix condition limits.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.structure import PBTKStructure

# Same threshold as in model/structure.py and model/solve.py
ILL_CONDITIONED_THRESHOLD = 1e12


def build_params_for_condition_test(
    animal: str = "E2",
    compound: str = "PFOS",
    isomer: str = "Linear",
    k_a: float = 10.0,
    k_ehc: float = 1.0,
) -> dict:
    """Build full params dict with a given k_a (and optional k_ehc) for matrix test."""
    from parameters.parameters import (
        build_parameters,
        calculate_physiology_from_body_weight,
        load_partition_coefficients,
        DEFAULT_ABSORPTION_EXCRETION,
    )

    config = {"compound": compound, "isomer": isomer, "animal": animal}
    # Use defaults for AE except k_a (and k_ehc) so we control the scale
    absorption_excretion = DEFAULT_ABSORPTION_EXCRETION.copy()
    absorption_excretion["k_a"] = k_a
    absorption_excretion["k_ehc"] = k_ehc

    physiological_dict = calculate_physiology_from_body_weight(47.0)  # E2 ~ 47 kg
    physiological_dict["milk_yield"] = 0.0  # avoid lactation curve dependency
    partition_coefficients = load_partition_coefficients(compound, isomer)

    return {
        "physiological": physiological_dict,
        "partition_coefficients": partition_coefficients,
        "absorption_excretion": absorption_excretion,
    }


def main() -> None:
    # Test with two k_ehc scenarios: moderate (10) and high (e.g. fitted PFOS ~280)
    for k_ehc_test in (10.0, 300.0):
        _run_sweep(k_ehc_test)
    print("\nUse the more conservative (smaller) max k_a if both k_ehc scenarios are relevant.")


def _run_sweep(k_ehc_test: float) -> None:
    # Sweep k_a in log-spaced steps from 10 to 1e7
    k_a_values = np.logspace(1, 7, 61)  # 10, 10^1.1, ..., 10^7

    print("\nSweeping k_a for cond(T) < 1e12 (instant absorption ~ large k_a)...")
    print("Using animal E2, PFOS Linear, k_ehc =", k_ehc_test)

    cond_numbers = []
    for k_a in k_a_values:
        params = build_params_for_condition_test(
            animal="E2", compound="PFOS", isomer="Linear",
            k_a=float(k_a), k_ehc=k_ehc_test,
        )
        structure = PBTKStructure(
            params=params,
            intake_function=lambda t: 0.0,
            physiology_provider=lambda t: params["physiological"],
        )
        T = structure.transition_matrix(0.0)
        cond_num = np.linalg.cond(T)
        cond_numbers.append(cond_num)

    cond_numbers = np.array(cond_numbers)
    # Find largest k_a such that cond(T) < threshold
    below = cond_numbers < ILL_CONDITIONED_THRESHOLD
    if not np.any(below):
        k_a_max_safe = np.nan
        print("No k_a in range keeps cond(T) < 1e12.")
    else:
        idx_last_safe = np.where(below)[0][-1]
        k_a_max_safe = float(k_a_values[idx_last_safe])
        print(f"Maximum k_a with cond(T) < 1e12: {k_a_max_safe:.4g}")

    # Also report first k_a where cond exceeds threshold (if any)
    above = cond_numbers >= ILL_CONDITIONED_THRESHOLD
    if np.any(above):
        idx_first_bad = np.where(above)[0][0]
        k_a_first_bad = float(k_a_values[idx_first_bad])
        print(f"First k_a with cond(T) >= 1e12: {k_a_first_bad:.4g}")

    # Save short summary (one file per k_ehc)
    out_dir = PROJECT_ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        {
            "k_a": float(k_a_values[i]),
            "cond_T": float(cond_numbers[i]),
            "below_threshold": cond_numbers[i] < ILL_CONDITIONED_THRESHOLD,
        }
        for i in range(len(k_a_values))
    ]
    out_path = out_dir / f"max_k_a_condition_sweep_k_ehc_{int(k_ehc_test)}.csv"
    pd.DataFrame(summary).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    # Recommended value: use 1/10 of max safe to stay away from the cliff
    if not np.isnan(k_a_max_safe):
        k_a_recommended = k_a_max_safe / 10.0
        print(f"Recommended fixed k_a (0.1 * max safe): {k_a_recommended:.4g}")
    return k_a_max_safe if not np.isnan(k_a_max_safe) else None


if __name__ == "__main__":
    main()
