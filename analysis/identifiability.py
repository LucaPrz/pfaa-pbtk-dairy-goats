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


def _get_functional_group(compound: str) -> str:
    """Classify PFAS functional group from compound name (PFCA vs PFSA)."""
    if compound is None:
        return "Unknown"
    name = str(compound).upper()
    if name.endswith("A") or name.endswith("ECHS"):
        return "PFCA"
    if name.endswith("S"):
        return "PFSA"
    return "Unknown"


def _get_chain_length(compound: str) -> int:
    """Approximate perfluoroalkyl chain length from common PFAS abbreviations."""
    if compound is None:
        return 99
    name = str(compound).upper()

    # Explicit mapping for PFAS in this project (perfluorinated chain length)
    chain_map = {
        # PFCA
        "PFBA": 4,
        "PFPEA": 5,
        "PFHXA": 6,
        "PFHPA": 7,
        "PFOA": 8,
        "PFNA": 9,
        "PFDA": 10,
        "PFUNDA": 11,
        "PFDODA": 12,
        "PFTRDA": 13,
        "PFTEDA": 14,
        # PFSA
        "PFBS": 4,
        "PFPES": 5,
        "PFHXS": 6,
        "PFHPS": 7,
        "PFOS": 8,
        "PFNS": 9,
        "PFDS": 10,
        "PFUNDS": 11,
        "PFDODS": 12,
        "PFTRDS": 13,
        "PFTEDS": 14,
    }

    return chain_map.get(name, 99)


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
    ci_bounds: Dict[str, Tuple[float, float]],
    n_counts: Dict[str, float],
    output_path: Path,
) -> None:
    """Save a ranked horizontal forest-style plot of mean identifiability per parameter.

    Parameters are shown on the y-axis and the mean log10 Fisher information
    (diagonal element) on the x-axis as point estimates with 95% CIs.
    """
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

    # One distinct color per parameter, using the same color cycle as other
    # project figures (driven by seaborn's default palette via set_paper_plot_style).
    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    if prop_cycle is not None:
        base_colors = prop_cycle.by_key().get("color", ["C0"])
    else:
        base_colors = ["C0"]
    colors = [base_colors[i % len(base_colors)] for i in range(len(names))]

    # Extract CI bounds and sample sizes in the same order
    lower = [ci_bounds.get(name, (val, val))[0] for name, val in zip(names, scores)]
    upper = [ci_bounds.get(name, (val, val))[1] for name, val in zip(names, scores)]
    ns = [n_counts.get(name, 0.0) for name in names]

    y_pos = np.arange(len(names))
    height = max(3.0, 0.4 * len(names))

    fig, ax = plt.subplots(figsize=(8, height))

    # Forest-style plot: point estimates with horizontal confidence intervals.
    for i, (m, lo, hi, c) in enumerate(zip(scores, lower, upper, colors)):
        if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
            xerr = np.array([[m - lo], [hi - m]])
        else:
            xerr = None
        ax.errorbar(
            m,
            y_pos[i],
            xerr=xerr,
            fmt="o",
            color=c,
            ecolor=c,
            elinewidth=1.5,
            capsize=3,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # highest score on top
    ax.set_xlabel("Mean log10 Fisher info (F_ii)")

    # Tighten x-limits to cover CIs with small margin
    finite_lows = [lo for lo in lower if np.isfinite(lo)]
    finite_highs = [hi for hi in upper if np.isfinite(hi)]
    if finite_lows and finite_highs:
        xmin = min(finite_lows) - 0.2
        xmax = max(finite_highs) + 0.2
    else:
        xmin = min(scores) - 0.2
        xmax = max(scores) + 0.2
    ax.set_xlim(xmin, xmax)

    # Annotate each point with its sample size (N_pairs)
    x_span = xmax - xmin
    text_offset = 0.02 * x_span
    for i, (hi, n_val) in enumerate(zip(upper, ns)):
        if not np.isfinite(hi):
            continue
        ax.text(
            hi + text_offset,
            y_pos[i],
            f"n={int(n_val)}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.grid(True, axis="x", alpha=0.2)

    # De-clutter spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"[IDENTIFIABILITY] Saved mean identifiability plot to {output_path}")


def save_parameter_correlation_plot(
    corr_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a scatter-style heatmap of parameter–parameter correlations across compound–isomer pairs.

    Rows correspond to compound–isomer labels, columns to parameter pairs, and color encodes
    the correlation coefficient in [-1, 1].
    """
    if corr_table.empty:
        logger.warning("[IDENTIFIABILITY] Empty correlation table, skipping correlation plot.")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from auxiliary.plot_style import set_paper_plot_style

        set_paper_plot_style()
    except Exception as exc:  # pragma: no cover - plotting optional
        logger.warning(
            "[IDENTIFIABILITY] Matplotlib unavailable, skipping correlation plot: %s", exc
        )
        return

    # Ensure deterministic column ordering but preserve the incoming row order
    corr_table = corr_table.copy()
    corr_table = corr_table.reindex(sorted(corr_table.columns), axis=1)

    labels = list(corr_table.index)
    pairs = list(corr_table.columns)

    n_rows = len(labels)
    n_cols = len(pairs)

    y_pos = np.arange(n_rows)
    x_pos = np.arange(n_cols)

    height = max(3.0, 0.4 * n_rows)
    fig, ax = plt.subplots(figsize=(2.5 + 0.6 * n_cols, height))

    cmap = plt.get_cmap("coolwarm")

    # Plot each column as a vertical strip of colored markers
    for j, pair in enumerate(pairs):
        vals = corr_table[pair].to_numpy(dtype=float)
        # Only plot entries where correlation is defined (finite)
        valid_mask = np.isfinite(vals)
        if not np.any(valid_mask):
            continue
        vals_valid = vals[valid_mask]
        y_valid = y_pos[valid_mask]

        # Normalize correlations in [-1, 1] to [0, 1] for colormap
        norm_vals = (np.clip(vals_valid, -1.0, 1.0) + 1.0) / 2.0
        colors = cmap(norm_vals)

        ax.scatter(
            np.full_like(vals_valid, j, dtype=float),
            y_valid,
            color=colors,
            s=80,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_frame_on(False)
    ax.grid(alpha=0.3, axis="x")
    ax.set_axisbelow(True)

    ax.set_xticks(x_pos)

    # LaTeX-style labels for parameter pairs, e.g. "k_urine–k_feces" -> "$k_{urine}$–$k_{feces}$"
    def _latex_param(name: str) -> str:
        if name.startswith("k_"):
            base = name.split("k_", 1)[1]
            return rf"$k_{{{base}}}$"
        return name

    pair_labels = []
    for pair in pairs:
        parts = pair.split("–")
        if len(parts) == 2:
            left = _latex_param(parts[0])
            right = _latex_param(parts[1])
            pair_labels.append(f"{left}–{right}")
        else:
            pair_labels.append(pair)

    ax.set_xticklabels(pair_labels, rotation=45, ha="right")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    ax.tick_params(size=0, colors="0.3")
    ax.set_xlabel("Parameter pair correlation", loc="right")

    # Add a colorbar for correlation scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Correlation coefficient", rotation=90)

    # Adjust layout to avoid cutting off labels and colorbar
    fig.subplots_adjust(left=0.25, bottom=0.25, right=0.85, top=0.95)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[IDENTIFIABILITY] Saved parameter correlation plot to {output_path}")


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

    # Aggregate identifiability across pairs: log10 Fisher diagonal per parameter
    aggregate_stats: Dict[str, Dict[str, float]] = {}
    # Long-format storage of parameter–parameter correlations across compound–isomer pairs
    corr_long_rows: List[Dict[str, Any]] = []

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
                param_names = diagnostics.get("param_names", [])
                fisher_matrix = diagnostics.get("fisher_matrix")
                corr_matrix = diagnostics.get("correlation_matrix")

                if use_for_aggregate:
                    # Aggregate Fisher diagonal only for passing pairs (if GOF filter is used)
                    if fisher_matrix is not None and len(param_names) == fisher_matrix.shape[0]:
                        diag_vals = np.diag(fisher_matrix).astype(float)
                        for name, val in zip(param_names, diag_vals):
                            if not np.isfinite(val) or val <= 0:
                                continue
                            stat = aggregate_stats.setdefault(name, {"values": []})
                            stat["values"].append(float(np.log10(val)))

                # Store parameter–parameter correlations (all pairs, no hardcoding, regardless of GOF)
                if (
                    corr_matrix is not None
                    and isinstance(corr_matrix, np.ndarray)
                    and corr_matrix.shape == (len(param_names), len(param_names))
                ):
                    for i in range(len(param_names)):
                        for j in range(i + 1, len(param_names)):
                            rho = float(corr_matrix[i, j])
                            if not np.isfinite(rho):
                                continue
                            corr_long_rows.append(
                                {
                                    "Compound": compound,
                                    "Isomer": isomer,
                                    "Label": f"{compound} {isomer}",
                                    "Pair": f"{param_names[i]}–{param_names[j]}",
                                    "Correlation": rho,
                                }
                            )

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
    mean_log_info: Dict[str, float] = {}
    ci_bounds: Dict[str, Tuple[float, float]] = {}
    n_counts: Dict[str, float] = {}
    rows = []
    for name, stat in aggregate_stats.items():
        values = np.asarray(stat.get("values", []), dtype=float)
        if values.size == 0:
            continue
        n = float(values.size)
        mean_val = float(values.mean())
        if values.size > 1:
            sd = float(values.std(ddof=1))
            se = sd / np.sqrt(values.size)
            ci_half = 1.96 * se
            ci_low = mean_val - ci_half
            ci_high = mean_val + ci_half
        else:
            ci_low = mean_val
            ci_high = mean_val

        mean_log_info[name] = mean_val
        ci_bounds[name] = (ci_low, ci_high)
        n_counts[name] = n
        rows.append(
            {
                "Parameter": name,
                "Mean_log10_Fisher_diag": mean_val,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "N_pairs": n,
            }
        )

    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if mean_log_info:
        mean_plot_path = figures_dir / "identifiability_mean_information.png"
        save_mean_identifiability_plot(mean_log_info, ci_bounds, n_counts, mean_plot_path)

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

    # Build and save parameter–parameter correlation strip plot (long-format to wide pivot)
    if corr_long_rows:
        try:
            df_corr_long = pd.DataFrame(corr_long_rows)
            # Determine desired compound–isomer ordering:
            # functional group (PFCA/PFSA), branched vs linear, then chain length.
            order_meta = (
                df_corr_long[["Compound", "Isomer", "Label"]]
                .drop_duplicates()
                .copy()
            )
            order_meta["Functional_Group"] = order_meta["Compound"].apply(
                _get_functional_group
            )
            order_meta["Chain_Length"] = order_meta["Compound"].apply(
                _get_chain_length
            )

            fg_order = {"PFCA": 0, "PFSA": 1, "Unknown": 2}
            iso_order = {"Branched": 0, "Linear": 1}

            order_meta["FG_Rank"] = order_meta["Functional_Group"].map(fg_order).fillna(2)
            order_meta["Iso_Rank"] = order_meta["Isomer"].map(iso_order).fillna(1)

            order_meta = order_meta.sort_values(
                ["FG_Rank", "Iso_Rank", "Chain_Length", "Compound", "Isomer"]
            )
            ordered_labels = list(order_meta["Label"])

            # Pivot to wide format: index = compound–isomer label, columns = parameter pair
            corr_table = df_corr_long.pivot_table(
                index="Label", columns="Pair", values="Correlation", aggfunc="mean"
            )
            corr_table = corr_table.reindex(ordered_labels)
            corr_table = corr_table.reindex(sorted(corr_table.columns), axis=1)

            corr_plot_path = figures_dir / "identifiability_parameter_correlations.png"
            save_parameter_correlation_plot(corr_table, corr_plot_path)

            # Also save underlying numbers
            corr_csv_path = figures_dir / "identifiability_parameter_correlations.csv"
            corr_table.to_csv(corr_csv_path)
            logger.info(
                "[IDENTIFIABILITY] Saved parameter correlation table to %s", corr_csv_path
            )
        except Exception as exc:
            logger.warning(
                "[IDENTIFIABILITY] Failed to build/save parameter correlation plot: %s", exc
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

