"""
Feces volume calculation for dairy goats using digestibility assumptions.

Inputs (in the clean project structure):
- data/raw/hcl_insolvable_ash.csv
- data/processed/pfas_daily_intake.csv

Primary output (used by the model as input):
- data/processed/feces_volume_per_day.csv
"""

from pathlib import Path
from typing import Optional, Dict
import sys

import pandas as pd

# Run from project root: python mass_balance/feces_volume.py  or  python -m mass_balance.feces_volume
_CLEAN_ROOT = Path(__file__).resolve().parent.parent
if str(_CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLEAN_ROOT))

from auxiliary.project_paths import get_data_root

# ---------------------------------------------------------------------------
# Constants and literature digestibility
# ---------------------------------------------------------------------------

# Supplement dry matter intake per animal (kg/day)
SUPP_DMI_KG_PER_DAY: float = 0.5

# Assume feces density ~ 1 kg/L
FECES_DENSITY_KG_PER_L: float = 1.0

# Literature range from Coleman et al. (2003), organic matter digestibility
LITERATURE_DIGESTIBILITY_RANGE = (42.2, 65.7)  # low, high (% of DM)

# Default digestibility value for mixed hay diets (Coleman et al. 2003)
LITERATURE_DIGESTIBILITY_TYPICAL = 58.0  # %

# More detailed OM digestibility by hay type and maturity
# Values taken from Coleman et al. (2003), Table 2 (g/kg), converted to %
LITERATURE_DIGESTIBILITY_BY_HAY: Dict[str, float] = {
    # Alfalfa (legume hays)
    "alfalfa_early_bloom": 657.3 / 10.0,
    "alfalfa_prebloom": 609.9 / 10.0,
    # Bermudagrass (warm-season perennial)
    "bermudagrass_4_weeks": 508.0 / 10.0,
    "bermudagrass_8_weeks": 532.5 / 10.0,
    "bermudagrass_mature": 520.7 / 10.0,
    # Crabgrass (warm-season annual)
    "crabgrass_boot": 578.8 / 10.0,
    "crabgrass_mature": 572.7 / 10.0,
    "crabgrass_vegetative": 577.7 / 10.0,
    # Eastern gamagrass (warm-season perennial)
    "eastern_gamagrass_boot": 545.9 / 10.0,
    "eastern_gamagrass_early_bloom": 586.2 / 10.0,
    "eastern_gamagrass_mature": 546.3 / 10.0,
    # Fescue (cool-season perennial)
    "fescue_early_bloom": 630.9 / 10.0,
    "fescue_mature": 422.3 / 10.0,
    "fescue_soft_dough": 556.2 / 10.0,
    # Caucasian bluestem (warm-season perennial)
    "caucasian_bluestem_early_bloom": 581.1 / 10.0,
    "caucasian_bluestem_late_bloom": 605.9 / 10.0,
    # Plains bluestem (warm-season perennial)
    "plains_bluestem_early_bloom": 625.1 / 10.0,
    "plains_bluestem_late_bloom": 590.0 / 10.0,
    # Wheat (cool-season annual)
    "wheat_dough": 569.5 / 10.0,
    "wheat_milk": 535.9 / 10.0,
}


def get_literature_digestibility(
    description: Optional[str] = None,
) -> float:
    """
    Return an organic matter digestibility percentage from literature.

    If description is provided, perform a simple fuzzy match on the keys of
    LITERATURE_DIGESTIBILITY_BY_HAY. Otherwise, return the typical default.
    """
    if description:
        desc_lower = description.lower()
        # Simple search over keys
        for key, value in LITERATURE_DIGESTIBILITY_BY_HAY.items():
            if desc_lower in key:
                return value

    # Fallback: recommended default for mixed hay diets
    return LITERATURE_DIGESTIBILITY_TYPICAL


def _get_paths() -> Dict[str, Path]:
    """
    Resolve all relevant paths for this module within the project root.

    Returns a dict with:
    - ash_path
    - intake_path
    - feces_volume_primary_out
    """
    data_root = get_data_root()

    ash_path = data_root / "raw" / "hcl_insolvable_ash.csv"
    intake_path = data_root / "processed" / "pfas_daily_intake.csv"

    feces_volume_primary_out = data_root / "processed" / "feces_volume_per_day.csv"

    return {
        "ash_path": ash_path,
        "intake_path": intake_path,
        "feces_volume_primary_out": feces_volume_primary_out,
    }


def compute_daily_feces_volume_from_digestibility(
    literature_description: Optional[str] = "default",
) -> pd.DataFrame:
    """
    Compute daily feces volume using digestibility information.

    Parameters
    ----------
    literature_description:
        Optional description to select a hay type from the literature dict.
    """
    paths = _get_paths()
    ash_path = paths["ash_path"]
    intake_path = paths["intake_path"]

    # Load AIA data for water percentage
    ash = pd.read_csv(ash_path)

    # Create lookup dictionaries for feces water percentage
    feces_water_pct: Dict[str, float] = {}

    for _, row in ash.iterrows():
        animal = row["animal"]
        matrix = row["matrix"]
        if matrix == "Feces":
            feces_water_pct[animal] = row["water_pct"]

    # Load daily intake data
    intake = pd.read_csv(intake_path)
    intake = intake[intake["Animal"].isin(["E2", "E3", "E4"])].copy()

    # Get unique days and animals
    intake_daily = intake[["Day", "Animal", "Feed_Intake_kg_day"]].drop_duplicates()

    # Add feces water percentage
    intake_daily["feces_water_pct"] = intake_daily["Animal"].map(feces_water_pct)

    # Calculate total dry matter intake (hay + supplement)
    intake_daily["hay_dm_intake_kg_per_d"] = intake_daily["Feed_Intake_kg_day"]
    intake_daily["total_dm_intake_kg_per_d"] = (
        intake_daily["hay_dm_intake_kg_per_d"] + SUPP_DMI_KG_PER_DAY
    )

    # Assign digestibility values from literature (single value for all rows)
    lit_value = get_literature_digestibility(literature_description)
    intake_daily["dm_digestibility_pct"] = lit_value

    # Calculate feces dry matter from digestibility
    # Feces DM = Total DM Intake × (1 - Digestibility/100)
    intake_daily["feces_dm_kg_per_d"] = (
        intake_daily["total_dm_intake_kg_per_d"]
        * (1 - intake_daily["dm_digestibility_pct"] / 100.0)
    )

    # Convert DM to wet mass using water percentage
    # DM fraction = 1 - (water_pct / 100)
    intake_daily["feces_dm_frac"] = 1 - (intake_daily["feces_water_pct"] / 100.0)
    intake_daily["feces_wet_kg_per_d"] = (
        intake_daily["feces_dm_kg_per_d"] / intake_daily["feces_dm_frac"]
    )

    # Calculate volume
    intake_daily["feces_volume_l_per_d"] = (
        intake_daily["feces_wet_kg_per_d"] / FECES_DENSITY_KG_PER_L
    )

    # Keep relevant columns
    result = intake_daily[
        [
            "Day",
            "Animal",
            "Feed_Intake_kg_day",
            "total_dm_intake_kg_per_d",
            "dm_digestibility_pct",
            "feces_dm_kg_per_d",
            "feces_wet_kg_per_d",
            "feces_volume_l_per_d",
        ]
    ].sort_values(["Animal", "Day"])

    return result


def main() -> None:
    """
    Main entry point to calculate feces volume.

    Uses literature digestibility values (Coleman et al. 2003),
    and writes the output file
    """
    paths = _get_paths()

    print("=" * 60)
    print("Calculating feces volume using literature digestibility")
    print("(Coleman et al. 2003: 58% default for mixed hay diets)")
    print("=" * 60)

    df_literature = compute_daily_feces_volume_from_digestibility(
        literature_description="default",
    )

    # Ensure target directory exists
    paths["feces_volume_primary_out"].parent.mkdir(parents=True, exist_ok=True)
    df_literature.to_csv(paths["feces_volume_primary_out"], index=False)

    print(f"\nSaved feces volume (literature-based) to: {paths['feces_volume_primary_out']}")
    print("(This is the primary file used by the model)")
    print("\nFirst few rows:")
    print(df_literature.head())
    print(
        f"\nDigestibility used: {df_literature['dm_digestibility_pct'].iloc[0]:.1f}%"
    )
    print(
        "Feces volume range: "
        f"{df_literature['feces_volume_l_per_d'].min():.2f} - "
        f"{df_literature['feces_volume_l_per_d'].max():.2f} L/day"
    )

if __name__ == "__main__":
    main()

