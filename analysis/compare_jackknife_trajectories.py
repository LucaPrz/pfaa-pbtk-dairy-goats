"""
Simulate with global-fit and jackknife (LOAO) parameter sets and compare trajectories.

Answers: do the 3 jackknife parameter vectors produce similar predicted curves
to the global fit, or are they wildly different? If similar → parameters
trade off (different params, same predictions). If different → param uncertainty
translates to real prediction spread.

Usage:
  python analysis/compare_jackknife_trajectories.py --compound PFOS --isomer Linear
Outputs:
  results/figures/jackknife_trajectories_PFOS_Linear_plasma.png
  results/figures/jackknife_trajectories_PFOS_Linear_milk.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.config import setup_context, SimulationConfig, get_matrix_module
from optimization.fit import simulate_model
from optimization.loss import predict_time_series
from auxiliary.project_paths import get_results_root


def get_param_vector_from_row(row: pd.Series, param_names: list[str]) -> np.ndarray:
    """Build parameter vector in param_names order from a CSV row."""
    return np.array([float(row[name]) for name in param_names], dtype=float)


def main(compound: str = "PFOS", isomer: str = "Linear") -> None:
    context = setup_context()
    param_names = context.config.param_names
    time_vec = context.config.time_vector
    animals = context.config.animals

    # Load global fit
    global_path = context.folder_phase1 / f"fit_{compound}_{isomer}.csv"
    if not global_path.exists():
        raise FileNotFoundError(f"Global fit not found: {global_path}")
    global_df = pd.read_csv(global_path)
    param_to_val = global_df.set_index("Parameter")["Value"]
    global_params = np.array([float(param_to_val[name]) for name in param_names], dtype=float)

    # Load jackknife (3 rows)
    jack_path = context.folder_phase2 / f"jackknife_{compound}_{isomer}_LOAO.csv"
    if not jack_path.exists():
        raise FileNotFoundError(f"Jackknife not found: {jack_path}")
    jack_df = pd.read_csv(jack_path)
    if jack_df.shape[0] != 3:
        raise ValueError(f"Expected 3 jackknife rows, got {jack_df.shape[0]}")

    # Build list: (label, param_array)
    sets = [("Global fit", global_params)]
    for _, row in jack_df.iterrows():
        label = f"LOAO (excl. {row['Animal_Excluded']})"
        params = get_param_vector_from_row(row, param_names)
        sets.append((label, params))

    # Simulate each set for all animals and compute mean trajectory per compartment
    compartments = ["plasma", "milk"]
    matrix_module = get_matrix_module(context.config)
    series_by_set_comp: dict[str, dict[str, np.ndarray]] = {}

    for label, params in sets:
        series_by_set_comp[label] = {}
        for comp in compartments:
            all_animal_series = []
            for animal in animals:
                sim_config = SimulationConfig(compound=compound, isomer=isomer, animal=animal)
                pred, all_params = simulate_model(
                    params.tolist(),
                    sim_config,
                    context,
                    param_names=param_names,
                    fixed_params=None,
                )
                series = predict_time_series(
                    pred,
                    all_params,
                    comp,
                    animal,
                    context.urine_volume_by_animal,
                    context.feces_mass_by_animal,
                    context.feces_mass_default,
                    context.milk_yield_by_animal,
                    matrix_module=matrix_module,
                )
                all_animal_series.append(series)
            # Mean across animals (same as MC "per_sample_mean" for one param set)
            mean_traj = np.nanmean(np.array(all_animal_series), axis=0)
            series_by_set_comp[label][comp] = mean_traj

    # Plot plasma and milk
    figures_dir = get_results_root() / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    t = time_vec

    for comp in compartments:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["black", "tab:orange", "tab:green", "tab:blue"]
        for (label, _), color in zip(sets, colors):
            y = series_by_set_comp[label][comp]
            y_plot = np.maximum(y, 1e-4)
            ax.plot(t, y_plot, label=label, color=color, lw=2)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(f"{comp} concentration (µg/kg)")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-4)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        out = figures_dir / f"jackknife_trajectories_{compound}_{isomer}_{comp}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")

    # Numeric comparison: at peak and over time
    print("\n--- Mean trajectory at t=56 (peak) ---")
    for comp in compartments:
        vals = [series_by_set_comp[label][comp] for label, _ in sets]
        idx_56 = int(56) if 56 <= t[-1] else len(t) - 1
        concs = [v[idx_56] for v in vals]
        print(f"  {comp}: global={concs[0]:.4f}, LOAO E2={concs[1]:.4f}, LOAO E3={concs[2]:.4f}, LOAO E4={concs[3]:.4f}")
        print(f"        min={min(concs):.4f}, max={max(concs):.4f}, ratio max/min={max(concs)/max(min(concs),1e-10):.2f}")

    # Ratio max/min across the 4 trajectories at each time (plasma)
    vals_plasma = np.array([series_by_set_comp[label]["plasma"] for label, _ in sets])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_plasma = np.nanmax(vals_plasma, axis=0) / np.maximum(np.nanmin(vals_plasma, axis=0), 1e-10)
    print(f"\n  Plasma: ratio max/min across 4 trajectories: max over time = {np.nanmax(ratio_plasma):.2f}, at t={t[np.nanargmax(ratio_plasma)]:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare global vs jackknife trajectories")
    parser.add_argument("--compound", type=str, default="PFOS")
    parser.add_argument("--isomer", type=str, default="Linear")
    args = parser.parse_args()
    main(compound=args.compound, isomer=args.isomer)
