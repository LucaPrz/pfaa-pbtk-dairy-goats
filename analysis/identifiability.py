"""
Identifiability diagnostics using Fisher Information Matrix.

Computes the Fisher Information Matrix from model predictions and their gradients,
then analyzes eigenvalues and condition number to identify unidentifiable parameter combinations.
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import logging

# Ensure the project root (which contains the `optimization` package)
# is on sys.path, even when this file is executed via an absolute path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.config import ModelConfig, FitConfig, FittingContext, SimulationConfig
from optimization.fit import simulate_model
from optimization.loss import predict_single_observation
from optimization.fit_variables import get_parameter_config
from optimization.io import get_project_root

logger = logging.getLogger(__name__)


def compute_gradients_finite_difference(
    fit_params: List[float],
    param_names: List[str],
    fit_config: FitConfig,
    use_data: pd.DataFrame,
    context: "FittingContext",
    simulate_model_func: Any,
    epsilon: float = 1e-6,
) -> np.ndarray:
    n_params = len(param_names)
    n_obs = len(use_data)

    # Get baseline predictions
    animals = use_data["Animal"].unique()
    simulation_cache: Dict[str, Tuple[Any, Any]] = {}

    for animal in animals:
        sim_config = SimulationConfig(
            compound=fit_config.compound,
            isomer=fit_config.isomer,
            animal=animal,
        )
        try:
            solution, all_params = simulate_model_func(fit_params, sim_config)
            simulation_cache[animal] = (solution, all_params)
        except Exception:
            simulation_cache[animal] = (None, None)

    # Compute baseline predictions
    baseline_preds: List[float] = []
    for _, row in use_data.iterrows():
        animal = row["Animal"]
        t = row["Day"]
        matrix_name = row["Matrix"].lower()

        sim, all_params = simulation_cache.get(animal, (None, None))
        if sim is None or all_params is None:
            baseline_preds.append(np.nan)
            continue

        pred = predict_single_observation(
            sim,
            all_params,
            matrix_name,
            t,
            animal,
            context.urine_volume_by_animal,
            context.feces_mass_by_animal,
            context.feces_mass_default,
            context.milk_yield_by_animal,
        )
        baseline_preds.append(pred)

    baseline_preds_arr = np.array(baseline_preds, dtype=float)
    valid_mask = np.isfinite(baseline_preds_arr) & (baseline_preds_arr > 0)
    baseline_log_preds = np.log(baseline_preds_arr + context.config.eps)

    # Debug: summarize baseline predictions and valid observations
    try:
        n_finite = int(np.isfinite(baseline_preds_arr).sum())
        n_positive = int((baseline_preds_arr > 0).sum())
        logger.info(
            "[IDENTIFIABILITY][DEBUG] Baseline preds summary: "
            "n_obs=%d, n_finite=%d, n_positive=%d",
            n_obs,
            n_finite,
            n_positive,
        )
    except Exception:
        # Debug logging should never break the main computation
        pass

    # Initialize gradient matrix
    gradients = np.zeros((n_obs, n_params), dtype=float)

    # Compute gradients for each parameter
    for j, _param_name in enumerate(param_names):
        # Perturb parameter
        perturbed_params = fit_params.copy()
        # Use relative step size: epsilon * max(1, abs(param))
        step = epsilon * max(1.0, abs(perturbed_params[j]))
        perturbed_params[j] += step

        # Get perturbed predictions
        perturbed_sim_cache: Dict[str, Tuple[Any, Any]] = {}
        for animal in animals:
            sim_config = SimulationConfig(
                compound=fit_config.compound,
                isomer=fit_config.isomer,
                animal=animal,
            )
            try:
                solution, all_params = simulate_model_func(perturbed_params, sim_config)
                perturbed_sim_cache[animal] = (solution, all_params)
            except Exception:
                perturbed_sim_cache[animal] = (None, None)

        perturbed_preds: List[float] = []
        for _, row in use_data.iterrows():
            animal = row["Animal"]
            t = row["Day"]
            matrix_name = row["Matrix"].lower()

            sim, all_params = perturbed_sim_cache.get(animal, (None, None))
            if sim is None or all_params is None:
                perturbed_preds.append(np.nan)
                continue

            pred = predict_single_observation(
                sim,
                all_params,
                matrix_name,
                t,
                animal,
                context.urine_volume_by_animal,
                context.feces_mass_by_animal,
                context.feces_mass_default,
                context.milk_yield_by_animal,
            )
            perturbed_preds.append(pred)

        perturbed_preds_arr = np.array(perturbed_preds, dtype=float)
        perturbed_log_preds = np.log(perturbed_preds_arr + context.config.eps)

        if abs(step) > 1e-10 and fit_params[j] > 0:
            # Convert linear space gradient to log10 space
            linear_gradient = (perturbed_log_preds - baseline_log_preds) / step
            log10_gradient = linear_gradient * fit_params[j] * np.log(10.0)
            gradients[:, j] = log10_gradient
        else:
            gradients[:, j] = 0.0

        # Mask invalid observations
        gradients[~valid_mask, j] = 0.0

        # Debug: summarize gradient column for this parameter
        try:
            col = gradients[:, j]
            n_nonzero = int(np.count_nonzero(col))
            max_abs = float(np.max(np.abs(col))) if col.size > 0 else 0.0
            logger.info(
                "[IDENTIFIABILITY][DEBUG] Gradients for %s: n_nonzero=%d, max_abs=%.3e",
                _param_name,
                n_nonzero,
                max_abs,
            )
        except Exception:
            pass

    return gradients


def compute_fisher_information_matrix(
    gradients: np.ndarray,
    use_data: pd.DataFrame,
    context: "FittingContext",
) -> np.ndarray:
    n_params = gradients.shape[1]
    fisher_matrix = np.zeros((n_params, n_params), dtype=float)

    # Group by matrix to get appropriate sigma
    for matrix_name in use_data["Matrix"].str.lower().unique():
        matrix_mask = use_data["Matrix"].str.lower() == matrix_name
        matrix_gradients = gradients[matrix_mask.values, :]

        if matrix_gradients.shape[0] == 0:
            continue

        # Get pooled sigma for this matrix (shared across all compounds)
        sigma = context.config.get_sigma(matrix_name)
        sigma_sq = sigma ** 2

        # Add contribution: (1/σ²) * Σ gradients
        # F_mn = Σ (1/σ²) * (∂log y/∂θ_m) * (∂log y/∂θ_n)
        fisher_matrix += (1.0 / sigma_sq) * np.dot(matrix_gradients.T, matrix_gradients)

    # Debug: summarize Fisher matrix
    try:
        diag = np.diag(fisher_matrix)
        max_diag = float(np.max(diag)) if diag.size > 0 else 0.0
        n_zero_diag = int(np.sum(diag == 0.0)) if diag.size > 0 else 0
        logger.info(
            "[IDENTIFIABILITY][DEBUG] Fisher matrix summary: max_diag=%.3e, n_zero_diag=%d",
            max_diag,
            n_zero_diag,
        )
    except Exception:
        pass

    return fisher_matrix


def analyze_identifiability(
    fisher_matrix: np.ndarray,
    param_names: List[str],
    threshold: float = 1e-6,
) -> Dict[str, Any]:
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(fisher_matrix)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Condition number (max/min eigenvalue of FIM)
    n_params = len(param_names)
    max_eigenval = float(np.max(eigenvals)) if eigenvals.size > 0 else 0.0

    # Smallest eigenvalue above threshold (if any)
    positive_eigs = eigenvals[eigenvals > threshold]
    if positive_eigs.size == 0:
        min_eigenval = 0.0
        condition_number = float("inf")
    else:
        min_eigenval = float(np.min(positive_eigs))
        condition_number = max_eigenval / min_eigenval if min_eigenval > threshold else float(
            "inf"
        )

    # Per-dimension condition number: n-th root so it's comparable across different
    # parameter counts
    if condition_number == float("inf") or condition_number <= 0:
        condition_number_per_dim = float("inf")
    else:
        condition_number_per_dim = float(condition_number ** (1.0 / n_params))

    # Identify unidentifiable directions (small eigenvalues)
    unidentifiable: List[Dict[str, Any]] = []
    for i, ev in enumerate(eigenvals):
        if ev < threshold:
            # Find which parameters contribute most to this eigenvector
            ev_vec = eigenvecs[:, i]
            # Normalize eigenvector
            ev_vec_norm = ev_vec / (np.linalg.norm(ev_vec) + 1e-10)
            # Find parameters with large contributions
            contributions = np.abs(ev_vec_norm)
            top_params = np.argsort(contributions)[::-1][:2]  # Top 2 parameters
            unidentifiable.append(
                {
                    "eigenvalue": float(ev),
                    "eigenvector": ev_vec_norm,
                    "parameters": [param_names[p] for p in top_params],
                    "contributions": {
                        param_names[p]: float(ev_vec_norm[p]) for p in top_params
                    },
                }
            )

    # Compute correlation matrix from Fisher matrix (inverse covariance approximation)
    try:
        cov_approx = np.linalg.inv(fisher_matrix)
        stds = np.sqrt(np.diag(cov_approx))
        corr_matrix = cov_approx / np.outer(stds, stds)
    except np.linalg.LinAlgError:
        corr_matrix = np.eye(len(param_names))

    return {
        "eigenvalues": eigenvals,
        "eigenvectors": eigenvecs,
        "condition_number": condition_number,
        "condition_number_per_dim": condition_number_per_dim,
        "unidentifiable_directions": unidentifiable,
        "correlation_matrix": corr_matrix,
        "param_names": param_names,
    }


def compute_identifiability_diagnostics(
    pair: Tuple[str, str],
    context: "FittingContext",
    epsilon: float = 1e-6,
) -> Optional[Dict[str, Any]]:
    compound, isomer = pair
    try:
        # Load fitted parameters
        fit_path = context.folder_phase1 / f"fit_{compound}_{isomer}.csv"
        if not fit_path.exists():
            logger.warning(f"[IDENTIFIABILITY] Fit file not found: {fit_path}")
            return None

        fit_df = pd.read_csv(fit_path)
        fit_params_dict = dict(zip(fit_df["Parameter"], fit_df["Value"]))

        # Get parameter configuration (respect model config)
        data_df = context.data_cache.get_pair_data(compound, isomer)
        param_names, fixed_params = get_parameter_config(
            compound, isomer, data_df, config=context.config
        )

        if not param_names:
            logger.warning(f"[IDENTIFIABILITY] No parameters to fit for {compound} {isomer}")
            return None

        # Extract fitted parameter values in correct order
        fit_params = [float(fit_params_dict.get(name, 0.0)) for name in param_names]

        # Get data
        use_data = data_df.copy()

        # Create fit config
        fit_config = FitConfig(compound=compound, isomer=isomer)

        # Create simulation function
        def simulate_model_func(params, sim_config):
            return simulate_model(
                params,
                sim_config,
                context,
                param_names=param_names,
                fixed_params=fixed_params,
            )

        # Compute gradients
        logger.info(f"[IDENTIFIABILITY] Computing gradients for {compound} {isomer}...")
        gradients = compute_gradients_finite_difference(
            fit_params,
            param_names,
            fit_config,
            use_data,
            context,
            simulate_model_func,
            epsilon=epsilon,
        )

        # Compute Fisher Information Matrix
        logger.info(
            f"[IDENTIFIABILITY] Computing Fisher Information Matrix for {compound} {isomer} "
            "(using pooled sigma)..."
        )
        fisher_matrix = compute_fisher_information_matrix(gradients, use_data, context)

        # Analyze identifiability
        logger.info(f"[IDENTIFIABILITY] Analyzing identifiability for {compound} {isomer}...")
        analysis = analyze_identifiability(fisher_matrix, param_names)

        # Add metadata
        analysis["compound"] = compound
        analysis["isomer"] = isomer
        analysis["fisher_matrix"] = fisher_matrix
        analysis["gradients"] = gradients

        return analysis

    except Exception as e:
        logger.error(f"[IDENTIFIABILITY] Failed for {compound} {isomer}: {e}", exc_info=True)
        return None


def save_identifiability_report(
    diagnostics: Dict[str, Any],
    output_path: Path,
) -> None:
    compound = diagnostics["compound"]
    isomer = diagnostics["isomer"]
    param_names = diagnostics["param_names"]

    # Save summary CSV
    fisher_matrix = diagnostics.get("fisher_matrix")
    if fisher_matrix is not None:
        fisher_diagonal = np.diag(fisher_matrix)
    else:
        fisher_diagonal = np.zeros(len(param_names))

    eigenvals = diagnostics.get("eigenvalues", [])
    max_eval = float(eigenvals[0]) if len(eigenvals) > 0 else np.nan
    min_eval = float(eigenvals[-1]) if len(eigenvals) > 0 else np.nan

    summary_df = pd.DataFrame(
        {
            "Parameter": param_names,
            "Fisher_Info_Diagonal": fisher_diagonal,
            "Max_Eigenvalue": max_eval,
            "Min_Eigenvalue": min_eval,
        }
    )
    summary_df["Condition_Number"] = diagnostics["condition_number"]
    summary_df["Condition_Number_Per_Dim"] = diagnostics.get(
        "condition_number_per_dim", np.nan
    )

    csv_path = output_path.with_suffix(".csv")
    summary_df.to_csv(csv_path, index=False)

    # Save text report
    report_path = output_path.with_suffix(".txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Identifiability Diagnostics: {compound} {isomer}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Condition Number: {diagnostics['condition_number']:.2e}\n")
        f.write("(Higher = less identifiable, >1e6 suggests problems)\n")
        cond_per_dim = diagnostics.get("condition_number_per_dim")
        if cond_per_dim is not None and np.isfinite(cond_per_dim):
            n_params = len(param_names)
            f.write(
                f"Condition Number (per dim, ^{n_params} root): {cond_per_dim:.2e}\n"
            )
            f.write(
                "(Comparable across different # of fitted parameters; use for fair comparison)\n"
            )
        f.write("\n")

        f.write("Eigenvalues (sorted, descending):\n")
        for i, ev in enumerate(diagnostics["eigenvalues"]):
            f.write(f"  {i+1}. {float(ev):.2e}\n")
        f.write("\n")

        if diagnostics["unidentifiable_directions"]:
            f.write("Unidentifiable Parameter Combinations:\n")
            for i, unid in enumerate(diagnostics["unidentifiable_directions"]):
                f.write(f"  {i+1}. Eigenvalue: {unid['eigenvalue']:.2e}\n")
                f.write(f"     Parameters: {', '.join(unid['parameters'])}\n")
                f.write(f"     Contributions: {unid['contributions']}\n")
        else:
            f.write("No strongly unidentifiable combinations detected.\n")

        f.write("\nCorrelation Matrix:\n")
        corr_df = pd.DataFrame(
            diagnostics["correlation_matrix"],
            index=param_names,
            columns=param_names,
        )
        f.write(corr_df.to_string())
        f.write("\n")

    logger.info(f"[IDENTIFIABILITY] Saved report to {report_path}")


def save_mean_identifiability_plot(
    mean_log_info: Dict[str, float],
    output_path: Path,
) -> None:
    """Save a ranked horizontal plot of mean identifiability per parameter."""
    if not mean_log_info:
        logger.warning("[IDENTIFIABILITY] No identifiability stats available for mean plot.")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from auxiliary.plot_style import set_paper_plot_style

        set_paper_plot_style()
    except Exception as exc:  # pragma: no cover - plotting optional
        logger.warning("[IDENTIFIABILITY] Matplotlib unavailable, skipping mean plot: %s", exc)
        return

    # Sort parameters by mean identifiability (descending)
    items = sorted(mean_log_info.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    scores = [v for _, v in items]

    # Color by process / route for visual grouping
    color_map = {
        "k_elim": "C0",
        "k_a": "C1",
        "k_urine": "C2",
        "k_feces": "C2",
        "k_ehc": "C2",
    }
    colors = [color_map.get(n, "C0") for n in names]

    y_pos = np.arange(len(names))
    height = max(3.0, 0.4 * len(names))

    fig, ax = plt.subplots(figsize=(8, height))

    # Horizontal lollipop plot: line from 0 to value with a marker at the end
    ax.hlines(y_pos, 0.0, scores, color=colors, linewidth=2.2)
    ax.scatter(scores, y_pos, color=colors, zorder=3, s=40)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # highest score on top
    ax.set_xlabel("Mean log10 Fisher info (F_ii)")

    # Tighten x-limits to data range with small margin
    xmin = min(scores) - 0.2
    xmax = max(scores) + 0.2
    ax.set_xlim(xmin, xmax)

    ax.grid(True, axis="x", alpha=0.2)

    # De-clutter spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"[IDENTIFIABILITY] Saved mean identifiability plot to {output_path}")


def regenerate_all_identifiability_reports(
    project_root: Optional[Path] = None,
    context: Optional["FittingContext"] = None,
) -> None:
    """
    Regenerate all identifiability reports with updated CSV format.

    This function finds all fit files from Phase 1 and regenerates the
    corresponding identifiability reports with the corrected CSV format.
    """
    if project_root is None:
        project_root = get_project_root()

    if context is None:
        from optimization.config import setup_context

        context = setup_context(project_root=project_root)

    # Define the input and output directories
    phase1_dir = project_root / "results" / "optimization" / "global_fit"
    output_dir = project_root / "results" / "analysis" / "identifiability"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional GOF filter: use only compound–isomer pairs that pass GOF thresholds
    gof_summary_path = project_root / "results" / "analysis" / "goodness_of_fit" / "goodness_of_fit_summary_by_compound.csv"
    passing_pairs: Optional[set[tuple[str, str]]] = None
    if gof_summary_path.exists():
        try:
            gof_df = pd.read_csv(gof_summary_path)
            crit_r2 = gof_df["R2"] > 0.7
            crit_gm = gof_df["GM_Fold_Error"] < 2.0
            crit_bias = gof_df["Bias_log10"].abs() < 0.25
            passing = gof_df[crit_r2 & crit_gm & crit_bias][["Compound", "Isomer"]]
            if not passing.empty:
                passing_pairs = set(map(tuple, passing.to_numpy()))
                logger.info(
                    "[IDENTIFIABILITY] Using GOF filter: %d compound–isomer pairs pass thresholds.",
                    len(passing_pairs),
                )
            else:
                logger.warning(
                    "[IDENTIFIABILITY] GOF summary found but no pairs pass thresholds; using all pairs."
                )
        except Exception as exc:
            logger.warning(
                "[IDENTIFIABILITY] Failed to apply GOF filter from %s: %s; using all pairs.",
                gof_summary_path,
                exc,
            )

    # Find all fit files
    fit_files = sorted(phase1_dir.glob("fit_*.csv"))

    if not fit_files:
        logger.warning(f"No fit files found in {phase1_dir}")
        return

    logger.info(
        f"Regenerating identifiability reports for {len(fit_files)} compound-isomer pairs..."
    )

    successful = 0
    failed = 0

    # Aggregate identifiability across pairs: mean log10 Fisher diagonal per parameter
    aggregate_stats: Dict[str, Dict[str, float]] = {}

    for fit_file in fit_files:
        # Extract compound and isomer from filename: fit_COMPOUND_ISOMER.csv
        parts = fit_file.stem.replace("fit_", "").split("_", 1)
        if len(parts) != 2:
            logger.warning(
                f"Could not parse compound-isomer from {fit_file.name}, skipping"
            )
            failed += 1
            continue

        compound = parts[0]
        isomer = parts[1]
        pair = (compound, isomer)

        try:
            logger.info(f"Processing {compound} {isomer}...")
            diagnostics = compute_identifiability_diagnostics(pair, context)

            if diagnostics is not None:
                output_path = output_dir / f"identifiability_{compound}_{isomer}"
                save_identifiability_report(diagnostics, output_path)

                # Update aggregate stats only for pairs that pass GOF criteria (if available)
                use_for_aggregate = (
                    passing_pairs is None or (compound, isomer) in passing_pairs
                )
                if use_for_aggregate:
                    param_names = diagnostics.get("param_names", [])
                    fisher_matrix = diagnostics.get("fisher_matrix")
                    if fisher_matrix is not None and len(param_names) == fisher_matrix.shape[0]:
                        diag_vals = np.diag(fisher_matrix).astype(float)
                        for name, val in zip(param_names, diag_vals):
                            if not np.isfinite(val) or val <= 0:
                                continue
                            stat = aggregate_stats.setdefault(name, {"sum_log": 0.0, "count": 0.0})
                            stat["sum_log"] += float(np.log10(val))
                            stat["count"] += 1.0

                logger.info(f"✓ Regenerated report for {compound} {isomer}")
                successful += 1
            else:
                logger.warning(f"✗ No diagnostics computed for {compound} {isomer}")
                failed += 1
        except Exception as e:
            logger.error(
                f"✗ Failed to regenerate report for {compound} {isomer}: {e}",
                exc_info=True,
            )
            failed += 1

    # Create a single ranked horizontal plot of mean identifiability per parameter
    mean_log_info = {}
    rows = []
    for name, stat in aggregate_stats.items():
        if stat["count"] <= 0:
            continue
        mean_val = stat["sum_log"] / stat["count"]
        mean_log_info[name] = mean_val
        rows.append(
            {
                "Parameter": name,
                "Mean_log10_Fisher_diag": mean_val,
                "N_pairs": stat["count"],
            }
        )

    if mean_log_info:
        figures_dir = project_root / "results" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        mean_plot_path = figures_dir / "identifiability_mean_information.png"
        save_mean_identifiability_plot(mean_log_info, mean_plot_path)

        # Save the underlying numbers for inspection
        try:
            df_mean = pd.DataFrame(rows).sort_values(
                "Mean_log10_Fisher_diag", ascending=False
            )
            csv_path = figures_dir / "identifiability_mean_information.csv"
            df_mean.to_csv(csv_path, index=False)
            logger.info(
                "[IDENTIFIABILITY] Saved mean identifiability table to %s", csv_path
            )
        except Exception as exc:
            logger.warning(
                "[IDENTIFIABILITY] Failed to save mean identifiability CSV: %s", exc
            )

    logger.info(f"Regeneration complete: {successful} successful, {failed} failed")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Regenerate all reports
    regenerate_all_identifiability_reports()

