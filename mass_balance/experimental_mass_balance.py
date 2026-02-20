"""
Mass balance calculation of the experimental data

Inputs (clean + original data):
- data/raw/pfas_data_no_e1.csv
- data/processed/pfas_daily_intake.csv
- data/processed/feces_volume_per_day.csv
- data/processed/urine_volume_per_goat.csv

Primary outputs (clean results):
- results/mass_balance/mass_balance_results.csv

Secondary outputs:
- results/figures/mass_balance_plot.png
"""

from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Run from project root: python mass_balance/experimental_mass_balance.py  or  python -m mass_balance.experimental_mass_balance
_CLEAN_ROOT = Path(__file__).resolve().parent.parent
if str(_CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLEAN_ROOT))

from auxiliary.project_paths import get_data_root, get_results_root
from auxiliary.plot_style import set_paper_plot_style
from parameters import parameters
from optimization.fit_variables import check_data_signals


def load_urine_volumes() -> Dict[str, float]:
    """
    Load animal-specific urine volumes from the clean processed file.

    Returns dict mapping Animal -> V_urine_median_L_per_day.
    """
    data_root = get_data_root()
    urine_path = data_root / "processed" / "urine_volume_per_goat.csv"
    if not urine_path.exists():
        raise FileNotFoundError(f"Urine volume file not found: {urine_path}")

    df = pd.read_csv(urine_path)
    return {
        str(row["Animal"]): float(row["V_urine_median_L_per_day"])
        for _, row in df.iterrows()
    }


def load_feces_masses() -> Dict[str, float]:
    """
    Load animal-specific feces masses from the clean processed file.

    Uses the median of `feces_wet_kg_per_d` per animal.
    """
    data_root = get_data_root()
    feces_path = data_root / "processed" / "feces_volume_per_day.csv"
    if not feces_path.exists():
        raise FileNotFoundError(f"Feces volume file not found: {feces_path}")

    df = pd.read_csv(feces_path)

    if "Animal" not in df.columns:
        raise ValueError("Expected column 'Animal' in feces_volume_per_day.csv")

    # Try standard column names from the clean feces_volume output
    feces_col = None
    for col in ["feces_wet_kg_per_d", "Feces_Mass_kg_per_day", "feces_mass"]:
        if col in df.columns:
            feces_col = col
            break

    if feces_col is None:
        raise ValueError(
            "Feces volume file is missing a feces mass column "
            "(expected one of 'feces_wet_kg_per_d', 'Feces_Mass_kg_per_day', 'feces_mass')."
        )

    return {
        animal: float(group[feces_col].median())
        for animal, group in df.groupby("Animal")
    }


def compute_transfer_rates(
    df_group: pd.DataFrame,
    intake_grouped: pd.DataFrame,
    urine_volume_by_animal: Dict[str, float],
    feces_mass_by_animal: Dict[str, float],
) -> pd.DataFrame:
    """
    For a given (Compound, Isomer) group of measurements, compute per-animal:
      - total_intake: from daily intake data (µg)
      - auc_milk_amount: trapezoidal AUC over (Day, amount) where Matrix == "Milk x Milk Amount" (µg)
      - auc_urine_amount: trapezoidal AUC over (Day, amount) where Matrix == "Urine"
                           (concentration in µg/L * volume in L/day) (µg)
      - auc_feces_amount: trapezoidal AUC over (Day, amount) where Matrix == "Feces"
                           (concentration in µg/kg * mass in kg/day) (µg)
      - transfer rates: auc_amount / total_intake for each matrix
      - total_body_burden: sum of tissue amounts on last measurement day
                           (Plasma in µg/L × L, tissues in µg/kg × kg) (µg)
      - mass_balance: (total_body_burden + total_eliminated) / total_intake

    Returns a dataframe with one row per animal.
    """
    results = []

    compound = str(df_group["Compound"].iloc[0]) if not df_group.empty else ""
    isomer = str(df_group["Isomer"].iloc[0]) if not df_group.empty else ""

    intake_filtered = intake_grouped[
        (intake_grouped["Compound"] == compound) & (intake_grouped["Isomer"] == isomer)
    ][["Animal", "total_intake"]].set_index("Animal")["total_intake"]

    animals = sorted(df_group["Animal"].unique())
    for animal in animals:
        dfa = df_group[df_group["Animal"] == animal]

        total_intake = float(intake_filtered.get(animal, np.nan))

        # Animal-specific excretion constants
        URINE_VOLUME_PER_DAY_L = urine_volume_by_animal.get(animal, np.nan)
        FECES_MASS_PER_DAY_KG = feces_mass_by_animal.get(animal, np.nan)

        # Milk amounts at sparse sampling times
        milk_amt_rows = dfa[dfa["Matrix"] == "Milk x Milk Amount"].copy()
        milk_amt_rows["Concentration"] = pd.to_numeric(
            milk_amt_rows["Concentration"], errors="coerce"
        )
        milk_amt_rows["Day"] = pd.to_numeric(milk_amt_rows["Day"], errors="coerce")
        milk_amt_rows = milk_amt_rows.dropna(subset=["Concentration", "Day"])
        milk_amt_rows = milk_amt_rows.sort_values("Day")

        if milk_amt_rows.shape[0] >= 2:
            auc_milk_amount = float(
                np.trapz(
                    milk_amt_rows["Concentration"].to_numpy(),
                    milk_amt_rows["Day"].to_numpy(),
                )
            )
        elif milk_amt_rows.shape[0] == 1:
            auc_milk_amount = 0.0
        else:
            auc_milk_amount = np.nan

        # Urine amounts (concentration * volume)
        urine_rows = dfa[dfa["Matrix"] == "Urine"].copy()
        urine_rows["Concentration"] = pd.to_numeric(
            urine_rows["Concentration"], errors="coerce"
        )
        urine_rows["Day"] = pd.to_numeric(urine_rows["Day"], errors="coerce")
        urine_rows = urine_rows.dropna(subset=["Concentration", "Day"])
        urine_rows = urine_rows.sort_values("Day")

        # Special handling in the original analysis:
        # For E3 PFHxS (Linear), below-LOQ urine values were set to LOQ/2.
        LOQ_URINE = 0.5  # µg/L
        LOQ_URINE_SUB = LOQ_URINE / 2  # 0.25 µg/L
        if animal == "E3" and compound == "PFHxS" and isomer == "Linear":
            urine_rows["Concentration"] = urine_rows["Concentration"].replace(
                0, LOQ_URINE_SUB
            )
            urine_rows["Concentration"] = urine_rows["Concentration"].fillna(
                LOQ_URINE_SUB
            )
            urine_rows.loc[
                urine_rows["Concentration"] < LOQ_URINE, "Concentration"
            ] = LOQ_URINE_SUB

        if urine_rows.shape[0] >= 2 and pd.notna(URINE_VOLUME_PER_DAY_L):
            urine_amounts = (
                urine_rows["Concentration"].to_numpy() * URINE_VOLUME_PER_DAY_L
            )
            auc_urine_amount = float(
                np.trapz(urine_amounts, urine_rows["Day"].to_numpy())
            )
        else:
            auc_urine_amount = 0.0

        # Feces amounts (concentration * mass)
        feces_rows = dfa[dfa["Matrix"] == "Feces"].copy()
        feces_rows["Concentration"] = pd.to_numeric(
            feces_rows["Concentration"], errors="coerce"
        )
        feces_rows["Day"] = pd.to_numeric(feces_rows["Day"], errors="coerce")
        feces_rows = feces_rows.dropna(subset=["Concentration", "Day"])
        feces_rows = feces_rows.sort_values("Day")

        if feces_rows.shape[0] >= 2 and pd.notna(FECES_MASS_PER_DAY_KG):
            feces_amounts = (
                feces_rows["Concentration"].to_numpy() * FECES_MASS_PER_DAY_KG
            )
            auc_feces_amount = float(
                np.trapz(feces_amounts, feces_rows["Day"].to_numpy())
            )
        else:
            auc_feces_amount = 0.0

        # Transfer rates
        transfer_rate_milk = (
            auc_milk_amount / total_intake
            if (pd.notna(auc_milk_amount) and pd.notna(total_intake) and total_intake != 0)
            else np.nan
        )
        transfer_rate_urine = (
            auc_urine_amount / total_intake
            if (pd.notna(auc_urine_amount) and pd.notna(total_intake) and total_intake != 0)
            else np.nan
        )
        transfer_rate_feces = (
            auc_feces_amount / total_intake
            if (pd.notna(auc_feces_amount) and pd.notna(total_intake) and total_intake != 0)
            else np.nan
        )

        # Mass balance: remaining PFAS in body tissues on last measurement day
        last_day = df_group[df_group["Animal"] == animal]["Day"].max()
        if pd.isna(last_day):
            total_body_burden = np.nan
        else:
            last_day_data = df_group[
                (df_group["Animal"] == animal) & (df_group["Day"] == last_day)
            ]

            tissue_columns = [
                "Plasma",
                "Liver",
                "Kidney",
                "Lung",
                "Spleen",
                "Heart",
                "Brain",
                "Muscle",
            ]
            total_body_burden = 0.0

            try:
                config = {"animal": animal, "compound": compound, "isomer": isomer}
                goat_params = parameters.build_parameters(config)
                phys = goat_params["physiological"]

                tissue_masses = {
                    "Plasma": phys["V_plasma"],  # L
                    "Liver": phys["V_liver"],  # kg
                    "Kidney": phys["V_kidney"],  # kg
                    "Lung": phys["V_lung"],  # kg
                    "Spleen": phys["V_spleen"],  # kg
                    "Heart": phys["V_heart"],  # kg
                    "Brain": phys["V_brain"],  # kg
                    "Muscle": phys["V_muscle"],  # kg
                }

                for tissue in tissue_columns:
                    tissue_data = last_day_data[last_day_data["Matrix"] == tissue]
                    if not tissue_data.empty and pd.notna(
                        tissue_data["Concentration"].iloc[0]
                    ):
                        conc = float(tissue_data["Concentration"].iloc[0])
                        mass_or_vol = tissue_masses.get(tissue, 0.0)
                        amount = conc * mass_or_vol
                        total_body_burden += amount
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not get parameters for animal {animal}: {e}")
                total_body_burden = np.nan

        total_eliminated = auc_milk_amount + auc_urine_amount + auc_feces_amount
        if (
            pd.notna(total_body_burden)
            and pd.notna(total_eliminated)
            and pd.notna(total_intake)
            and total_intake > 0
        ):
            mass_balance = (total_body_burden + total_eliminated) / total_intake
            unaccounted = total_intake - (total_body_burden + total_eliminated)
        else:
            mass_balance = np.nan
            unaccounted = np.nan

        results.append(
            {
                "Animal": animal,
                "Compound": compound,
                "Isomer": isomer,
                "total_intake": total_intake,
                "auc_milk_amount": auc_milk_amount,
                "auc_urine_amount": auc_urine_amount,
                "auc_feces_amount": auc_feces_amount,
                "total_body_burden": total_body_burden,
                "total_eliminated": total_eliminated,
                "mass_balance": mass_balance,
                "unaccounted": unaccounted,
                "transfer_rate_milk": transfer_rate_milk,
                "transfer_rate_urine": transfer_rate_urine,
                "transfer_rate_feces": transfer_rate_feces,
            }
        )

    return pd.DataFrame(results)


def _is_summary_compound(compound: str) -> bool:
    """
    Return True if `compound` is an aggregate (e.g. 'Sum PFCA') that should
    be excluded from per‑compound mass balance plots.
    """
    if pd.isna(compound):
        return True

    compound_str = str(compound).strip()

    if compound_str.startswith("SD ") or compound_str.startswith("Sum") or compound_str.startswith("Summe"):
        return True

    if "√" in compound_str or "∑" in compound_str:
        return True

    return False


def _get_functional_group(compound: str) -> str:
    """Return functional group (PFCA / PFSA / Unknown) from compound name."""
    if pd.isna(compound):
        return "Unknown"
    c = str(compound).upper()
    if c.endswith("A") or c.endswith("ECHS"):
        return "PFCA"
    if c.endswith("S"):
        return "PFSA"
    return "Unknown"


def _get_chain_length(compound: str) -> int:
    """Extract perfluorinated chain length from compound name."""
    if pd.isna(compound):
        return 0

    c = str(compound).upper()
    name = c.replace("PF", "")

    if name.endswith("ECHS"):
        name = name[:-4]
    elif name.endswith("A") or name.endswith("S"):
        name = name[:-1]

    chain_map = {
        "TR": 13,
        "TE": 14,
        "UN": 11,
        "DO": 12,
        "PE": 5,
        "HX": 6,
        "HP": 7,
        "B": 4,
        "O": 8,
        "N": 9,
        "D": 10,
    }

    for indicator, length in chain_map.items():
        if name.startswith(indicator):
            return length

    return 0


def calculate_mass_balance() -> None:
    """
    Calculate the mass balance of the experimental data and write a
    per‑animal results table.

    The complete mass balance for each PFAS compound and isomer is calculated by accounting for all intake and elimination pathways.
    """


    data_root = get_data_root()
    results_root = get_results_root()

    # ------------------------------------------------------------------
    # Load clean input tables
    # ------------------------------------------------------------------
    pfas_path = data_root / "raw" / "pfas_data_no_e1.csv"
    intake_path = data_root / "processed" / "pfas_daily_intake.csv"

    if not pfas_path.exists():
        raise FileNotFoundError(f"PFAS measurement table not found: {pfas_path}")
    if not intake_path.exists():
        raise FileNotFoundError(f"Daily intake table not found: {intake_path}")

    df_pfas = pd.read_csv(pfas_path)
    # Ensure numeric concentrations
    df_pfas = df_pfas.copy()
    df_pfas["Concentration"] = pd.to_numeric(df_pfas["Concentration"], errors="coerce")

    # ------------------------------------------------------------------
    # Timepoint-wise LOQ imputation (experimental mass-balance only)
    # ------------------------------------------------------------------
    #
    # For each (Compound, Isomer, Matrix, Day), if at least one animal has
    # a concentration above the matrix-specific LOQ, then all censored
    # values (< LOQ) at that timepoint in that matrix are set to LOQ/2.
    # If no animal has a value above LOQ at that timepoint, censored
    # values are left as-is (typically 0).
    #
    loq_default = 0.5
    loq_milk = 0.005

    if {"Compound", "Isomer", "Matrix", "Day", "Animal", "Concentration"}.issubset(
        df_pfas.columns
    ):
        for (compound, isomer, matrix, day), g in df_pfas.groupby(
            ["Compound", "Isomer", "Matrix", "Day"], dropna=False
        ):
            conc = pd.to_numeric(g["Concentration"], errors="coerce")
            if conc.isna().all():
                continue
            threshold = loq_milk if str(matrix).strip().lower() == "milk" else loq_default
            if (conc > threshold).any():
                mask = g.index[conc < threshold]
                if not mask.empty:
                    df_pfas.loc[mask, "Concentration"] = threshold / 2.0
    df_intake = pd.read_csv(intake_path)

    # Sum daily intake per animal / compound / isomer (µg)
    intake_grouped = (
        df_intake.groupby(["Animal", "Compound", "Isomer"], as_index=False)["PFAS_Intake_ug_day"]
        .sum()
        .rename(columns={"PFAS_Intake_ug_day": "total_intake"})
    )

    # Clean excretion "constants" from other clean scripts
    urine_volume_by_animal = load_urine_volumes()
    feces_mass_by_animal = load_feces_masses()

    # ------------------------------------------------------------------
    # Delegate the actual AUC and body‑burden calculations to the
    # well‑tested helper from Analysis.mass_balance.
    # ------------------------------------------------------------------
    per_animal_results = []
    for (_, _), df_group in df_pfas.groupby(["Compound", "Isomer"], dropna=False):
        if df_group.empty:
            continue
        df_res = compute_transfer_rates(
            df_group,
            intake_grouped,
            urine_volume_by_animal,
            feces_mass_by_animal,
        )
        per_animal_results.append(df_res)

    if per_animal_results:
        df_result = pd.concat(per_animal_results, ignore_index=True)
    else:
        # Fallback empty frame with expected columns
        df_result = pd.DataFrame(
            columns=[
                "Animal",
                "Compound",
                "Isomer",
                "total_intake",
                "auc_milk_amount",
                "auc_urine_amount",
                "auc_feces_amount",
                "total_body_burden",
                "total_eliminated",
                "mass_balance",
                "unaccounted",
                "transfer_rate_milk",
                "transfer_rate_urine",
                "transfer_rate_feces",
            ]
        )

    out_dir = results_root / "mass_balance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mass_balance_results.csv"
    df_result.to_csv(out_path, index=False)

    print(f"Saved experimental mass balance results to: {out_path}")

def plot_mass_balance() -> None:
    """
    Plot the mass balance of the experimental data

    This function reads the per‑animal results created by `calculate_mass_balance`,
    summarises mass balance per compound–isomer combination, and produces a
    bar plot with 95% confidence intervals and individual animal values
    overlaid.
    """
    results_root = get_results_root()
    data_root = get_data_root()

    results_path = results_root / "mass_balance" / "mass_balance_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Mass balance results not found at {results_path}. "
            "Run calculate_mass_balance() first."
        )

    df = pd.read_csv(results_path)
    if df.empty:
        print("No mass balance results available to plot.")
        return

    # Basic cleaning and helper columns
    df = df.copy()
    df["mass_balance"] = pd.to_numeric(df["mass_balance"], errors="coerce")
    df = df[df["mass_balance"].notna()]
    df = df[df["Isomer"] != "Total"]
    df = df[~df["Compound"].apply(_is_summary_compound)]
    # Exclude compound–isomer combinations with zero total intake
    df["total_intake"] = pd.to_numeric(df["total_intake"], errors="coerce")
    df = df[df["total_intake"] > 0]

    if df.empty:
        print("No valid mass balance data after filtering.")
        return

    df["functional_group"] = df["Compound"].apply(_get_functional_group)
    df["chain_length"] = df["Compound"].apply(_get_chain_length)
    df = df[df["chain_length"] > 0]

    if df.empty:
        print("No valid compound–isomer combinations after applying chain‑length filter.")
        return

    # Load raw PFAS measurement data to identify model-relevant signals
    pfas_path = data_root / "raw" / "pfas_data_no_e1.csv"
    if not pfas_path.exists():
        raise FileNotFoundError(f"PFAS measurement table not found: {pfas_path}")
    data_df = pd.read_csv(pfas_path)

    # ------------------------------------------------------------------
    # Summarise per compound–isomer (mean ± 95% CI)
    # ------------------------------------------------------------------
    summary_rows = []
    for (compound, isomer), g in df.groupby(["Compound", "Isomer"], dropna=False):
        # Skip pairs without detectable signals in model matrices/elimination routes
        signals = check_data_signals(str(compound), str(isomer), data_df)
        if not any(
            [
                signals.get("milk_signal", False),
                signals.get("plasma_signal", False),
                signals.get("feces_signal", False),
                signals.get("feces_depuration_signal", False),
                signals.get("urine_signal", False),
            ]
        ):
            continue
        mb_values = pd.to_numeric(g["mass_balance"], errors="coerce").dropna()
        if mb_values.empty:
            continue

        mean_mb = float(mb_values.mean())
        std_mb = float(mb_values.std(ddof=1)) if len(mb_values) > 1 else 0.0
        n = int(len(mb_values))

        if n > 1 and std_mb > 0:
            sem = std_mb / np.sqrt(n)
            t_crit = stats.t.ppf(0.975, n - 1)
            ci_half_width = float(t_crit * sem)
        else:
            ci_half_width = 0.0

        summary_rows.append(
            {
                "Compound": compound,
                "Isomer": isomer,
                "functional_group": g["functional_group"].iloc[0],
                "chain_length": int(g["chain_length"].iloc[0]),
                "mean_mass_balance": mean_mb,
                "ci_half_width": ci_half_width,
                "n_animals": n,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        print("No summary statistics available for plotting.")
        return

    summary_df = summary_df.sort_values(
        ["functional_group", "Isomer", "chain_length"], ascending=[True, True, True]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Build bar plot
    # ------------------------------------------------------------------
    set_paper_plot_style()

    x = np.arange(len(summary_df))
    means = summary_df["mean_mass_balance"].to_numpy()
    ci = summary_df["ci_half_width"].to_numpy()

    colors = summary_df["functional_group"].map(
        {"PFCA": "#2E86AB", "PFSA": "#A23B72"}
    ).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.bar(
        x,
        means,
        yerr=[ci, ci],
        color=colors,
        edgecolor="black",
        alpha=0.8,
        capsize=3,
        linewidth=0.8,
    )

    # Overlay individual animal data as jittered points
    for i, row in enumerate(summary_df.itertuples(index=False)):
        mask = (df["Compound"] == row.Compound) & (df["Isomer"] == row.Isomer)
        mb_vals = df.loc[mask, "mass_balance"].dropna().to_numpy()
        if mb_vals.size == 0:
            continue
        jitter = np.linspace(-0.15, 0.15, mb_vals.size)
        ax.scatter(
            np.full_like(mb_vals, x[i], dtype=float) + jitter,
            mb_vals,
            color="black",
            s=25,
            alpha=0.7,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

    # Cosmetic settings
    ax.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Complete mass balance (1.0)",
    )

    labels = []
    for compound, isomer in zip(summary_df["Compound"], summary_df["Isomer"]):
        prefix = "n-" if str(isomer).lower().startswith("lin") else "br-"
        labels.append(f"{prefix}{compound}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mass balance")
    ax.set_xlabel("Compound (isomer)")

    ylim_max = max(1.3, float(np.nanmax(means + ci) * 1.1))
    ax.set_ylim(0.0, ylim_max)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2E86AB", edgecolor="black", label="PFCA"),
        Patch(facecolor="#A23B72", edgecolor="black", label="PFSA"),
    ]
    ax.legend(handles=legend_elements, frameon=True, loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    fig.tight_layout()

    fig_dir = results_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_fig = fig_dir / "mass_balance_plot.png"
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)

    print(f"Saved experimental mass balance plot to: {out_fig}")


def main() -> None:
    """Convenience entry point when run as a script."""
    calculate_mass_balance()
    plot_mass_balance()


if __name__ == "__main__":
    main()
