"""
Generate three didactic figures for a single PFAS compound–isomer pair:

1) Phase 1: Observed vs predicted concentrations using Phase 1 global best-fit parameters.
2) Phase 2: Jackknife distribution of a key parameter (e.g. k_elim).
3) Phase 3: Monte Carlo prediction uncertainty bands with illustrative sample trajectories.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

# Ensure the project root (which contains the `optimization` package)
# is on sys.path, even when this file is executed via an absolute path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from auxiliary.plot_style import set_paper_plot_style
from optimization.config import (
    setup_context,
    FittingContext,
    SimulationConfig,
    FitConfig,
)
from optimization.io import get_project_root
from optimization.fit import simulate_model
from optimization.loss import predict_time_series, loss_function
from optimization.mc import numerical_hessian
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger(__name__)


def _load_context() -> FittingContext:
    """Set up the shared fitting context (data cache, config, folders)."""
    project_root = get_project_root()
    context = setup_context(project_root=project_root)
    return context


def load_observed_data(
    context: FittingContext,
    compound: str,
    isomer: str,
    matrix: str = "Plasma",
) -> pd.DataFrame:
    """
    Load observed concentration–time data for a given compound–isomer–matrix.

    Returns a DataFrame with at least columns: Compound, Isomer, Matrix, Animal, Day, Concentration.
    """
    df_pair = context.data_cache.get_pair_data(compound, isomer)
    if df_pair.empty:
        logger.warning(
            "[FIGURES] No data found for %s %s; returning empty DataFrame.",
            compound,
            isomer,
        )
        return df_pair

    df_matrix = df_pair[df_pair["Matrix"] == matrix].copy()
    if df_matrix.empty:
        logger.warning(
            "[FIGURES] No %s observations for %s %s; using all matrices instead.",
            matrix,
            compound,
            isomer,
        )
        df_matrix = df_pair.copy()

    df_matrix = df_matrix[df_matrix["Concentration"].notna()]
    return df_matrix


def load_phase1_parameters(
    context: FittingContext, compound: str, isomer: str
) -> pd.DataFrame:
    """
    Load Phase 1 global fit parameters for a compound–isomer pair.

    Returns the raw fit DataFrame with columns: Parameter, Value, Compound, Isomer.
    """
    fit_path = context.folder_phase1 / f"fit_{compound}_{isomer}.csv"
    if not fit_path.exists():
        raise FileNotFoundError(f"Phase 1 fit file not found: {fit_path}")

    fit_df = pd.read_csv(fit_path)
    # Backwards compatibility: k_renal -> k_urine
    fit_df = fit_df.copy()
    fit_df.loc[fit_df["Parameter"] == "k_renal", "Parameter"] = "k_urine"
    return fit_df


def get_phase1_param_vector(
    context: FittingContext, fit_df: pd.DataFrame
) -> np.ndarray:
    """
    Build parameter vector in the order expected by simulate_model (context.config.param_names).
    """
    params: List[float] = []
    for name in context.config.param_names:
        row = fit_df[fit_df["Parameter"] == name]
        if row.empty:
            raise KeyError(
                f"Required parameter '{name}' missing from Phase 1 fit file."
            )
        params.append(float(row["Value"].iloc[0]))
    return np.asarray(params, dtype=float)


def load_mc_predictions(
    context: FittingContext, compound: str, isomer: str
) -> pd.DataFrame:
    """
    Load Monte Carlo prediction summary for a compound–isomer pair.

    Returns the full prediction table from results/optimization/monte_carlo.
    """
    mc_path = context.folder_phase3 / f"predictions_{compound}_{isomer}_monte_carlo.csv"
    if not mc_path.exists():
        raise FileNotFoundError(f"Monte Carlo predictions file not found: {mc_path}")

    df = pd.read_csv(mc_path)
    return df


def plot_phase1_global_fit(
    context: FittingContext,
    compound: str,
    isomer: str,
    matrix: str = "Plasma",
    compartment: str = "plasma",
    show_prediction: bool = True,
    show_observations: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Phase 1: observed vs predicted using global best-fit parameters.

    - Observations: scatter of Concentration vs Day for the chosen matrix.
    - Prediction: mean over animals of model-predicted time series for the corresponding compartment.
    - Annotation: key parameter values from Phase 1 fit.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    obs_df = load_observed_data(context, compound, isomer, matrix=matrix)
    if obs_df.empty:
        logger.warning(
            "[FIGURES] No observations to plot for Phase 1 (%s %s).", compound, isomer
        )

    # Load best-fit parameters and build vector for simulate_model
    fit_df = load_phase1_parameters(context, compound, isomer)
    param_vec = get_phase1_param_vector(context, fit_df)

    time_vector = context.config.time_vector
    all_series: List[np.ndarray] = []

    mean_pred: Optional[np.ndarray] = None
    if show_prediction:
        for animal in context.config.animals:
            sim_cfg = SimulationConfig(compound=compound, isomer=isomer, animal=animal)
            try:
                solution, all_params = simulate_model(param_vec, sim_cfg, context)
            except Exception as exc:
                logger.warning(
                    "[FIGURES] Simulation failed for Phase 1 %s %s animal %s: %s",
                    compound,
                    isomer,
                    animal,
                    exc,
                )
                continue

            series = predict_time_series(
                solution,
                all_params,
                compartment,
                animal,
                context.urine_volume_by_animal,
                context.feces_mass_by_animal,
                context.feces_mass_default,
                context.milk_yield_by_animal,
            )
            all_series.append(np.asarray(series, dtype=float))

        if not all_series:
            logger.warning(
                "[FIGURES] No successful simulations for Phase 1 (%s %s).",
                compound,
                isomer,
            )
        else:
            stacked = np.vstack(all_series)
            mean_pred = np.nanmean(stacked, axis=0)
            ax.plot(
                time_vector,
                mean_pred,
                color="C0",
                lw=2.0,
                label="Model prediction (global fit)",
            )

    if show_observations and not obs_df.empty:
        x_pts = pd.to_numeric(obs_df["Day"], errors="coerce").to_numpy(dtype=float)
        y_pts = pd.to_numeric(obs_df["Concentration"], errors="coerce").to_numpy(dtype=float)
        ax.scatter(
            x_pts,
            y_pts,
            color="black",
            alpha=0.7,
            s=25,
            label=f"Observed ({matrix})",
        )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(f"{matrix} concentration (µg/L)")

    ax.set_title(f"Phase 1: Global fit for {compound} {isomer}")

    # Annotate key parameters
    param_names_to_show = ["k_ehc", "k_elim", "k_urine", "k_a", "k_feces"]
    lines = []
    for name in param_names_to_show:
        row = fit_df[fit_df["Parameter"] == name]
        if not row.empty:
            val = float(row["Value"].iloc[0])
            lines.append(f"{name} = {val:.3g} 1/day")
    if lines:
        text = "\n".join(lines)
        ax.text(
            0.98,
            0.02,
            text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_phase2_parameter_distribution(
    context: FittingContext,
    compound: str,
    isomer: str,
    param_name: str = "k_elim",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Phase 2: Hessian-based marginal distribution for a single parameter.

    - Uses the numerical Hessian of the censored-likelihood loss in log10-space
      around the Phase 1 optimum to obtain a covariance matrix for the fitted
      parameters.
    - Plots the implied normal density in log10-space for the selected parameter,
      with the Phase 1 point estimate as the mean.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    # Phase 1 point estimate
    fit_df = load_phase1_parameters(context, compound, isomer)
    row = fit_df[fit_df["Parameter"] == param_name]
    best_val = float(row["Value"].iloc[0]) if not row.empty else np.nan

    if not np.isfinite(best_val) or best_val <= 0:
        logger.warning(
            "[FIGURES] Phase 1 estimate for %s %s parameter %s is non-positive or NaN.",
            compound,
            isomer,
            param_name,
        )
        return fig

    # Compute Hessian-based covariance in log10-space for the fitted parameters
    df_pair = context.data_cache.get_pair_data(compound, isomer)
    param_names_fit = context.config.get_param_names(compound, isomer, df_pair)
    fixed_params = context.config.get_fixed_parameters(compound, isomer, df_pair)

    if param_name not in param_names_fit:
        logger.warning(
            "[FIGURES] Parameter %s is not among the fitted parameters for %s %s.",
            param_name,
            compound,
            isomer,
        )
        return fig

    eps_param = 1e-12
    phase1_fit_linear = np.array(
        [fit_df[fit_df["Parameter"] == n]["Value"].iloc[0] for n in param_names_fit],
        dtype=float,
    )
    x0 = np.log10(np.clip(phase1_fit_linear, eps_param, None)).astype(np.float64)
    log_bounds = context.config.get_log_bounds(param_names_fit)
    steps = np.array([(hi - lo) * 1e-2 for (lo, hi) in log_bounds], dtype=float)

    fit_config = FitConfig(compound=compound, isomer=isomer)

    def loss_wrapper(log_fit_params: np.ndarray) -> float:
        return float(
            loss_function(
                log_fit_params,
                fit_config=fit_config,
                use_data=df_pair,
                context=context,
                simulate_model_func=lambda p, sim_cfg: simulate_model(
                    p,
                    sim_cfg,
                    context,
                    param_names=param_names_fit,
                    fixed_params=fixed_params,
                ),
            )
        )

    try:
        H = numerical_hessian(loss_wrapper, x0, step=steps)
    except Exception as exc:
        logger.error(
            "[FIGURES] Failed to compute Hessian for %s %s: %s",
            compound,
            isomer,
            exc,
            exc_info=True,
        )
        return fig

    # Ridge-regularized covariance in log10-space
    ridge = 1e-6 * max(float(np.max(np.diag(H))), 1e-6)
    cov_log = np.linalg.pinv(H + ridge * np.eye(H.shape[0]))
    cov_log = 0.5 * (cov_log + cov_log.T)

    # Extract marginal variance for the requested parameter in log10-space
    idx = param_names_fit.index(param_name)
    var_log = float(cov_log[idx, idx])
    if not np.isfinite(var_log) or var_log <= 0:
        logger.warning(
            "[FIGURES] Non-positive or NaN variance for %s %s parameter %s.",
            compound,
            isomer,
            param_name,
        )
        return fig

    mu = float(np.log10(best_val))
    sigma = float(np.sqrt(var_log))

    # Use a fixed ±4σ window around the mean in log10-space
    x_grid = np.linspace(mu - 4.0 * sigma, mu + 4.0 * sigma, 400)

    pdf_log = (
        1.0
        / (sigma * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((x_grid - mu) ** 2) / (sigma**2))
    )
    ax.fill_between(
        x_grid,
        pdf_log,
        color="C0",
        alpha=0.25,
    )
    ax.plot(
        x_grid,
        pdf_log,
        color="C0",
        lw=2.0,
        label="Assumed normal in log10 space",
    )

    # Mark the Phase 1 estimate in log10-space
    ax.axvline(
        mu,
        color="black",
        linestyle="--",
        lw=2,
        label="Phase 1 estimate",
    )

    # Focus the x-axis around the ±4σ window
    ax.set_xlim(x_grid[0], x_grid[-1])

    # Show ticks in linear k-space while plotting in log10-space
    def _log10_to_linear_label(x: float, _pos: int) -> str:
        val = 10.0 ** x
        if val >= 1.0:
            return f"{val:.1f}"
        else:
            return f"{val:.3f}"

    ax.xaxis.set_major_formatter(FuncFormatter(_log10_to_linear_label))
    ax.set_xlabel(f"{param_name} (1/day)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Phase 2: Hessian-based distribution of {param_name}\n{compound} {isomer}"
    )
    ax.grid(axis="y", alpha=0.2)
    if np.isfinite(best_val):
        ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


def plot_phase3_mc_prediction(
    context: FittingContext,
    compound: str,
    isomer: str,
    matrix: str = "Plasma",
    compartment: str = "plasma",
    n_sample_trajectories: int = 30,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Phase 3: Monte Carlo prediction uncertainty.

    - Thin semi-transparent lines: illustrative sample trajectories reconstructed from
      MC variance (in log-space) around the median prediction.
    - Thick line: median prediction.
    - Shaded band: 95% prediction interval (observation-level if available).
    - Optional scatter: observed data for the same matrix.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    mc_df = load_mc_predictions(context, compound, isomer)
    df_comp = mc_df[mc_df["Compartment"] == compartment].copy()
    if df_comp.empty:
        raise ValueError(
            f"No Monte Carlo predictions for compartment '{compartment}' "
            f"({compound} {isomer})."
        )

    df_comp = df_comp.sort_values("Time")
    t = df_comp["Time"].values.astype(float)
    median = df_comp["Pred_Median"].values.astype(float)

    # Use observation-level CI if available, otherwise param-only CI
    if {"CI_Observation_Lower", "CI_Observation_Upper"}.issubset(df_comp.columns):
        lower = df_comp["CI_Observation_Lower"].values.astype(float)
        upper = df_comp["CI_Observation_Upper"].values.astype(float)
    else:
        lower = df_comp["CI_Lower"].values.astype(float)
        upper = df_comp["CI_Upper"].values.astype(float)

    # Approximate sample trajectories using total log-variance, if available
    if "Var_total_log" in df_comp.columns:
        var_log = df_comp["Var_total_log"].values.astype(float)
        sigma_log = np.sqrt(np.clip(var_log, 0.0, None))
        eps = context.config.eps
        base_log = np.log(np.maximum(median, eps))

        rng = np.random.default_rng(42)
        for _ in range(n_sample_trajectories):
            z = rng.normal(0.0, 1.0)
            log_traj = base_log + z * sigma_log
            traj = np.exp(log_traj)
            ax.plot(
                t,
                traj,
                color="C0",
                alpha=0.08,
                lw=1.0,
            )

    # Median and CI band
    ax.plot(t, median, color="C0", lw=2.0, label="Median prediction")
    ax.fill_between(
        t,
        lower,
        upper,
        color="C0",
        alpha=0.25,
        label="95% prediction interval",
    )

    # Observations for context
    obs_df = load_observed_data(context, compound, isomer, matrix=matrix)
    if not obs_df.empty:
        x_pts = pd.to_numeric(obs_df["Day"], errors="coerce").to_numpy(dtype=float)
        y_pts = pd.to_numeric(obs_df["Concentration"], errors="coerce").to_numpy(dtype=float)
        ax.scatter(
            x_pts,
            y_pts,
            color="black",
            alpha=0.6,
            s=20,
            label=f"Observed ({matrix})",
        )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(f"{matrix} concentration (µg/L)")
    ax.set_title(
        f"Phase 3: Monte Carlo prediction uncertainty\n{compound} {isomer} ({compartment})"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate three-phase optimization figures for a PFAS compound."
    )
    parser.add_argument(
        "--compound",
        type=str,
        default="PFOS",
        help='Compound name (e.g. "PFOS").',
    )
    parser.add_argument(
        "--isomer",
        type=str,
        default="Linear",
        help='Isomer (e.g. "Linear").',
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default="Plasma",
        help='Observation matrix to plot (e.g. "Plasma", "Milk").',
    )
    parser.add_argument(
        "--compartment",
        type=str,
        default="plasma",
        help='Model compartment corresponding to the chosen matrix (e.g. "plasma", "milk").',
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/figures",
        help="Output directory for figures (relative to project root).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    set_paper_plot_style()

    args = parse_args(argv)
    compound = args.compound
    isomer = args.isomer
    matrix = args.matrix
    compartment = args.compartment

    context = _load_context()
    project_root = get_project_root()
    outdir = project_root / Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[FIGURES] Generating optimization workflow figures for %s %s (%s/%s)",
        compound,
        isomer,
        matrix,
        compartment,
    )

    # Phase 1: combined prediction + observations (main figure)
    fig1 = plot_phase1_global_fit(
        context,
        compound=compound,
        isomer=isomer,
        matrix=matrix,
        compartment=compartment,
    )
    fig1_path = outdir / f"phase1_global_fit_{compound}_{isomer}.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    logger.info("[FIGURES] Saved Phase 1 figure to %s", fig1_path)

    # Phase 2
    fig2 = plot_phase2_parameter_distribution(
        context,
        compound=compound,
        isomer=isomer,
        param_name="k_elim",
    )
    fig2_path = outdir / f"phase2_hessian_k_elim_{compound}_{isomer}.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    logger.info("[FIGURES] Saved Phase 2 figure to %s", fig2_path)

    # Phase 3
    fig3 = plot_phase3_mc_prediction(
        context,
        compound=compound,
        isomer=isomer,
        matrix=matrix,
        compartment=compartment,
        n_sample_trajectories=30,
    )
    fig3_path = outdir / f"phase3_mc_prediction_{compound}_{isomer}.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    logger.info("[FIGURES] Saved Phase 3 figure to %s", fig3_path)


if __name__ == "__main__":
    main()

