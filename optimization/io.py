"""
Local I/O helpers for the optimisation workflow.

This module provides:
  - `get_project_root()` → the project (repository) root directory
  - `DataCache` → cached access to the PFAS data CSV
  - `load_data(...)` → intake and physiology inputs for fitting / MC
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Return the root of the project (repository root).

    This delegates to the shared helper in `auxiliary/project_paths.py`
    so that all scripts agree on where the clean project root is.
    """
    # Import lazily to avoid circular imports
    from auxiliary.project_paths import get_clean_root

    return get_clean_root()


@dataclass
class DataCache:
    """
    Lightweight cache for the PFAS measurement table used during fitting.

    Parameters
    ----------
    compounds:
        List of compound names to keep.  Other compounds are discarded.
    data_path:
        Path to the cleaned PFAS measurement CSV
        (`data/raw/pfas_data_no_e1.csv` in this repository).
    """

    compounds: Sequence[str]
    data_path: Path

    def __post_init__(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        # Normalise column names we rely on
        required_cols = {"Compound", "Isomer", "Matrix", "Animal", "Day", "Concentration"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"PFAS data table {self.data_path} is missing columns: {sorted(missing)}")

        # Drop aggregate "Total" isomer and restrict to requested compounds
        mask = df["Isomer"] != "Total"
        if self.compounds:
            mask &= df["Compound"].isin(list(self.compounds))
        self._df = df.loc[mask].copy()

    def get_pair_data(self, compound: str, isomer: str) -> pd.DataFrame:
        """Return all rows for a given compound–isomer pair."""
        df_pair = self._df[
            (self._df["Compound"] == compound) & (self._df["Isomer"] == isomer)
        ].copy()
        return df_pair

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """
        Return all distinct (compound, isomer) pairs present in the cache.

        Pairs are sorted alphabetically for deterministic processing order.
        """
        pairs_df = (
            self._df[["Compound", "Isomer"]]
            .drop_duplicates()
            .sort_values(["Compound", "Isomer"])
        )
        return [tuple(x) for x in pairs_df.to_numpy()]  # type: ignore[list-item]


def _load_urine_and_feces(project_root: Path) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Load urine volumes and feces masses from processed mass-balance tables.

    Returns
    -------
    urine_volume_by_animal:
        Animal → urine volume in L/day.
    feces_mass_by_animal:
        Animal → feces mass in kg/day.
    feces_mass_default:
        Fallback feces mass (overall median) used when an animal is missing.
    """
    data_root = project_root / "data"

    urine_path = data_root / "processed" / "urine_volume_per_goat.csv"
    feces_path = data_root / "processed" / "feces_volume_per_day.csv"

    if not urine_path.exists():
        raise FileNotFoundError(f"Urine volume file not found: {urine_path}")
    if not feces_path.exists():
        raise FileNotFoundError(f"Feces volume file not found: {feces_path}")

    urine_df = pd.read_csv(urine_path)
    feces_df = pd.read_csv(feces_path)

    urine_col = "V_urine_median_L_per_day"
    if urine_col not in urine_df.columns:
        raise ValueError(f"Expected column '{urine_col}' in {urine_path}")

    urine_volume_by_animal: Dict[str, float] = {
        str(row["Animal"]): float(row[urine_col])
        for _, row in urine_df.iterrows()
        if pd.notna(row[urine_col])
    }

    if "Animal" not in feces_df.columns:
        raise ValueError(f"Expected column 'Animal' in {feces_path}")

    feces_col_candidates = ["feces_wet_kg_per_d", "Feces_Mass_kg_per_day", "feces_mass"]
    feces_col = next((c for c in feces_col_candidates if c in feces_df.columns), None)
    if feces_col is None:
        raise ValueError(
            f"Feces table {feces_path} is missing a feces mass column "
            f"(expected one of {feces_col_candidates})."
        )

    feces_mass_by_animal: Dict[str, float] = {
        str(animal): float(group[feces_col].median())
        for animal, group in feces_df.groupby("Animal")
        if group[feces_col].notna().any()
    }

    # Overall median across animals as a generic fallback
    all_masses = [v for v in feces_mass_by_animal.values() if np.isfinite(v)]
    feces_mass_default = float(np.median(all_masses)) if all_masses else 0.0

    return urine_volume_by_animal, feces_mass_by_animal, feces_mass_default


def _load_milk_yield_by_animal(project_root: Path, max_day: int = 140) -> Dict[str, np.ndarray]:
    """
    Build per‑animal daily milk‑yield time series from the raw feed table.

    Day 0 has no measurement in the CSV (days run 1..max_day). It is filled with
    the yield on day 1 so that simulation at t=0 does not divide by zero when
    computing milk concentration; this is a reasonable stand‑in for the first
    experimental day.

    Returns
    -------
    dict:
        Animal → np.ndarray of shape (max_day + 1,) containing daily
        milk yield in kg/day.  Missing days (other than 0) are filled with 0.
    """
    data_root = project_root / "data"
    feed_path = data_root / "raw" / "feed_intake_milk_yield.csv"
    if not feed_path.exists():
        raise FileNotFoundError(f"Feed/milk table not found: {feed_path}")

    df = pd.read_csv(feed_path)
    required_cols = {"Day", "Animal", "Milk_Yield_kg_per_day"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{feed_path} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["Day"] = df["Day"].astype(int)
    milk_yield_by_animal: Dict[str, np.ndarray] = {}

    for animal, group in df.groupby("Animal"):
        series = np.zeros(max_day + 1, dtype=float)
        for _, row in group.iterrows():
            day = int(row["Day"])
            if 0 <= day <= max_day and pd.notna(row["Milk_Yield_kg_per_day"]):
                series[day] = float(row["Milk_Yield_kg_per_day"])
        # Day 0: no measurement in CSV; use day 1 yield to avoid divide-by-zero at t=0
        if series[1] > 0:
            series[0] = series[1]
        milk_yield_by_animal[str(animal)] = series

    return milk_yield_by_animal


def load_data(config, project_root: Optional[Path] = None):
    """
    Load all data required for fitting and Monte Carlo.

    Returns
    -------
    intake_df:
        Daily PFAS intake table (`data/processed/pfas_daily_intake.csv`).
    urine_volume_by_animal:
        Mapping Animal → urine volume (L/day).
    feces_mass_by_animal:
        Mapping Animal → feces mass (kg/day).
    milk_yield_by_animal:
        Mapping Animal → milk yield time series (kg/day).
    feces_mass_default:
        Fallback feces mass (kg/day) used when an animal is missing.
    """
    if project_root is None:
        project_root = get_project_root()

    data_root = project_root / "data"
    intake_path = data_root / "processed" / "pfas_daily_intake.csv"

    if not intake_path.exists():
        raise FileNotFoundError(
            f"Daily intake table not found at {intake_path}. "
            "Run `python mass_balance/intake.py` first to generate it."
        )

    intake_df = pd.read_csv(intake_path)

    urine_volume_by_animal, feces_mass_by_animal, feces_mass_default = _load_urine_and_feces(
        project_root
    )
    milk_yield_by_animal = _load_milk_yield_by_animal(project_root, max_day=int(config.time_vector.max()))

    logger.info(
        "Loaded intake and physiological data: %d intake rows, %d animals with urine volumes, "
        "%d animals with feces mass, %d animals with milk yield series",
        len(intake_df),
        len(urine_volume_by_animal),
        len(feces_mass_by_animal),
        len(milk_yield_by_animal),
    )

    return (
        intake_df,
        urine_volume_by_animal,
        feces_mass_by_animal,
        milk_yield_by_animal,
        feces_mass_default,
    )


__all__ = ["get_project_root", "DataCache", "load_data"]

