"""
Standalone script to create a Phase 1 parameter forest plot.

The plot shows, on the y-axis, the fitted kinetic parameters and, on the x-axis,
the mean log10 Phase 1 fitted values (1/day) across all compound–isomer pairs,
with 95% confidence intervals (based on the across-pair variability).

Outputs
-------
- results/figures/phase1_parameter_forest.png  (forest plot)
- results/figures/phase1_parameter_forest.csv  (underlying summary statistics)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import sys

import numpy as np
import pandas as pd

# Ensure the project root (which contains the `optimization` package)
# is on sys.path, even when this file is executed via an absolute path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.io import get_project_root


logger = logging.getLogger(__name__)


def _save_phase1_parameter_forest_plot(
    mean_log_values: Dict[str, float],
    ci_bounds: Dict[str, Tuple[float, float]],
    n_counts: Dict[str, float],
    output_path: Path,
) -> None:
    """
    Save a horizontal forest plot of Phase 1 fitted parameter values.

    Parameters are shown on the y-axis and the mean log10 fitted value
    (across compound–isomer pairs) on the x-axis as point estimates with 95% CIs.
    """
    if not mean_log_values:
        logger.warning(
            "[PHASE1_FOREST] No Phase 1 parameter statistics available for forest plot."
        )
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from auxiliary.plot_style import set_paper_plot_style

        set_paper_plot_style()
    except Exception as exc:  # pragma: no cover - plotting optional
        logger.warning(
            "[PHASE1_FOREST] Matplotlib unavailable, skipping Phase 1 forest plot: %s",
            exc,
        )
        return

    # Sort parameters by mean fitted value (descending)
    items = sorted(mean_log_values.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    scores = [v for _, v in items]

    # Color cycle consistent with other figures
    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    if prop_cycle is not None:
        base_colors = prop_cycle.by_key().get("color", ["C0"])
    else:
        base_colors = ["C0"]
    colors = [base_colors[i % len(base_colors)] for i in range(len(names))]

    lower = [ci_bounds.get(name, (val, val))[0] for name, val in zip(names, scores)]
    upper = [ci_bounds.get(name, (val, val))[1] for name, val in zip(names, scores)]
    ns = [n_counts.get(name, 0.0) for name in names]

    y_pos = np.arange(len(names))
    height = max(3.0, 0.4 * len(names))

    fig, ax = plt.subplots(figsize=(8, height))

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
    ax.invert_yaxis()
    ax.set_xlabel("Mean log10 fitted parameter value (1/day)")

    finite_lows = [lo for lo in lower if np.isfinite(lo)]
    finite_highs = [hi for hi in upper if np.isfinite(hi)]
    if finite_lows and finite_highs:
        xmin = min(finite_lows) - 0.2
        xmax = max(finite_highs) + 0.2
    else:
        xmin = min(scores) - 0.2
        xmax = max(scores) + 0.2
    ax.set_xlim(xmin, xmax)

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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(
        f"[PHASE1_FOREST] Saved Phase 1 parameter forest plot to {output_path}"
    )


def build_phase1_parameter_summary(
    project_root: Optional[Path] = None,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]], Dict[str, float], List[Dict[str, Any]]]:
    """
    Read Phase 1 global fit results and compute summary statistics per parameter.

    Returns
    -------
    mean_log : dict
        Parameter -> mean log10(value).
    ci_bounds : dict
        Parameter -> (CI_low, CI_high) in log10 space (approx. 95% CI).
    counts : dict
        Parameter -> number of contributing compound–isomer pairs.
    rows : list of dict
        Long-form table used for CSV export.
    """
    if project_root is None:
        project_root = get_project_root()

    phase1_dir = project_root / "results" / "optimization" / "global_fit"
    if not phase1_dir.exists():
        logger.warning("[PHASE1_FOREST] Phase 1 directory not found: %s", phase1_dir)
        return {}, {}, {}, []

    phase1_param_values: Dict[str, List[float]] = {}

    fit_files = sorted(phase1_dir.glob("fit_*.csv"))
    if not fit_files:
        logger.warning("[PHASE1_FOREST] No Phase 1 fit files found in %s", phase1_dir)
        return {}, {}, {}, []

    for fit_file in fit_files:
        try:
            df = pd.read_csv(fit_file)
        except Exception as exc:
            logger.warning(
                "[PHASE1_FOREST] Failed to read %s: %s",
                fit_file,
                exc,
            )
            continue

        for _, row in df.iterrows():
            try:
                name = str(row["Parameter"])
                val = float(row["Value"])
            except Exception:
                continue
            if not np.isfinite(val) or val <= 0:
                continue
            phase1_param_values.setdefault(name, []).append(float(np.log10(val)))

    mean_log: Dict[str, float] = {}
    ci_bounds: Dict[str, Tuple[float, float]] = {}
    counts: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []

    for name, values in phase1_param_values.items():
        vals = np.asarray(values, dtype=float)
        if vals.size == 0:
            continue
        n = float(vals.size)
        mean_val = float(vals.mean())
        if vals.size > 1:
            sd = float(vals.std(ddof=1))
            se = sd / np.sqrt(vals.size)
            ci_half = 1.96 * se
            ci_low = mean_val - ci_half
            ci_high = mean_val + ci_half
        else:
            ci_low = mean_val
            ci_high = mean_val

        mean_log[name] = mean_val
        ci_bounds[name] = (ci_low, ci_high)
        counts[name] = n
        rows.append(
            {
                "Parameter": name,
                "Mean_log10_value": mean_val,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "N_pairs": n,
            }
        )

    return mean_log, ci_bounds, counts, rows


def main(project_root: Optional[Path] = None) -> None:
    if project_root is None:
        project_root = get_project_root()

    mean_log, ci_bounds, counts, rows = build_phase1_parameter_summary(project_root)
    if not mean_log:
        logger.warning("[PHASE1_FOREST] No Phase 1 parameter statistics computed; exiting.")
        return

    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_path = figures_dir / "phase1_parameter_forest.png"
    _save_phase1_parameter_forest_plot(mean_log, ci_bounds, counts, plot_path)

    # Save underlying summary table
    try:
        df_phase1 = pd.DataFrame(rows).sort_values(
            "Mean_log10_value", ascending=False
        )
        csv_path = figures_dir / "phase1_parameter_forest.csv"
        df_phase1.to_csv(csv_path, index=False)
        logger.info(
            "[PHASE1_FOREST] Saved Phase 1 parameter summary table to %s",
            csv_path,
        )
    except Exception as exc:
        logger.warning(
            "[PHASE1_FOREST] Failed to save Phase 1 parameter CSV: %s",
            exc,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

