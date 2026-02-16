"""
Daily PFAS intake calculation for dairy goats.

Inputs (in the clean project structure):
- data/raw/pfas_data_no_e1.csv
- data/raw/feed_intake_milk_yield.csv

Primary output (used as model input):
- data/processed/pfas_daily_intake.csv
"""

from pathlib import Path
from typing import Tuple
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Run from project root: python mass_balance/intake.py  or  python -m mass_balance.intake
_CLEAN_ROOT = Path(__file__).resolve().parent.parent
if str(_CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLEAN_ROOT))

from auxiliary.project_paths import get_data_root, get_results_root
from auxiliary.plot_style import set_paper_plot_style

# Apply global figure/plot style (shared with other plots)
set_paper_plot_style()


# Exposure period (days) - intake is constant during this period, zero after
EXPOSURE_PERIOD_DAYS: int = 56

# Maximum day to generate (end of the experiment)
MAX_DAY: int = 140

# Isolation weeks for feed intake measurements
ISOLATION_WEEKS = [1, 6, 9]

# Animal IDs
ANIMALS = ["E2", "E3", "E4"]


def _get_paths() -> dict:
    """
    Resolve input and output paths within the clean data directory.
    """
    data_root = get_data_root()

    pfas_data_file = data_root / "raw" / "pfas_data_no_e1.csv"
    feed_data_file = data_root / "raw" / "feed_intake_milk_yield.csv"
    output_file = data_root / "processed" / "pfas_daily_intake.csv"

    return {
        "pfas_data_file": pfas_data_file,
        "feed_data_file": feed_data_file,
        "output_file": output_file,
    }


def load_data(file_path_hay: Path, file_path_feed: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load hay PFAS concentrations and feed intake data.
    """
    if not file_path_hay.exists():
        raise FileNotFoundError(f"Data file not found: {file_path_hay}")

    if not file_path_feed.exists():
        raise FileNotFoundError(f"Data file not found: {file_path_feed}")

    data = pd.read_csv(file_path_hay)
    hay_df = data[data["Matrix"] == "Hay"]
    feed_df = pd.read_csv(file_path_feed)

    return hay_df, feed_df


def calculate_hay_average(hay_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average hay concentrations by compound, isomer, and day.
    """
    hay_df = hay_df.copy()
    hay_df["Concentration"] = pd.to_numeric(hay_df["Concentration"], errors="coerce")

    # Filter out NaN concentrations and limit to valid day range
    hay_df = hay_df[
        (hay_df["Concentration"].notna())
        & (hay_df["Day"] >= 0)
        & (hay_df["Day"] <= MAX_DAY)
    ].copy()

    hay_avg = (
        hay_df.groupby(["Compound", "Isomer", "Day"])["Concentration"]
        .mean()
        .reset_index()
    )

    return hay_avg


def calculate_hay_average_exposure(hay_avg: pd.DataFrame) -> pd.DataFrame:
    """
    Average hay concentration over the exposure period only.
    """
    hay_exposure = hay_avg[hay_avg["Day"] <= EXPOSURE_PERIOD_DAYS].copy()

    hay_mean = (
        hay_exposure.groupby(["Compound", "Isomer"])["Concentration"]
        .mean()
        .reset_index()
    )
    hay_mean = hay_mean.rename(columns={"Concentration": "Mean_Concentration"})

    return hay_mean


def combined_intake_dataframe(feed_df: pd.DataFrame, hay_avg: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full daily PFAS intake table from hay and feed data.
    """
    # Average hay concentration during exposure (removes sampling-time confounding)
    hay_mean = calculate_hay_average_exposure(hay_avg)

    # Weekly mean feed intake for weeks 1 and 6 per animal
    weekly_feed_intake = {}  # (animal, week) -> mean feed intake
    for week in [1, 6]:
        week_data = feed_df[
            (feed_df["Week"] == week)
            & (feed_df["Feed_Intake_kg_per_day"].notna())
            & (feed_df["Day"] <= EXPOSURE_PERIOD_DAYS)
        ]
        if not week_data.empty:
            for animal in ANIMALS:
                animal_week_data = week_data[week_data["Animal"] == animal]
                if not animal_week_data.empty:
                    weekly_feed_intake[(animal, week)] = float(
                        animal_week_data["Feed_Intake_kg_per_day"].mean()
                    )

    # Average feed intake over all isolation weeks per animal
    isolation_feed = feed_df[
        (feed_df["Week"].isin(ISOLATION_WEEKS))
        & (feed_df["Feed_Intake_kg_per_day"].notna())
        & (feed_df["Day"] <= EXPOSURE_PERIOD_DAYS)
    ]

    avg_feed_intake_per_animal = {}  # animal -> average feed intake
    for animal in ANIMALS:
        animal_data = isolation_feed[isolation_feed["Animal"] == animal]
        if not animal_data.empty:
            avg_feed_intake_per_animal[animal] = float(
                animal_data["Feed_Intake_kg_per_day"].mean()
            )

    # Create day-to-week mapping from feed data
    day_to_week = {}
    for _, row in feed_df.iterrows():
        day = int(row["Day"])
        week = int(row["Week"])
        day_to_week[day] = week

    # All unique compound-isomer pairs from hay mean data
    compounds_isomers = (
        hay_mean[["Compound", "Isomer"]]
        .drop_duplicates()
        .sort_values(["Compound", "Isomer"])
        .values.tolist()
    )

    # Lookup for mean hay concentrations
    hay_conc_dict = {
        (row["Compound"], row["Isomer"]): float(row["Mean_Concentration"])
        for _, row in hay_mean.iterrows()
    }

    # Calculate daily intake for each day, animal, compound, isomer
    results = []
    for day in range(MAX_DAY + 1):
        in_exposure = day <= EXPOSURE_PERIOD_DAYS
        week = day_to_week.get(day)

        for animal in ANIMALS:
            # Determine feed intake for this animal and day
            # Use weekly mean for weeks 1 and 6, otherwise use average
            if week in [1, 6] and (animal, week) in weekly_feed_intake:
                feed_intake = weekly_feed_intake[(animal, week)]
            else:
                feed_intake = avg_feed_intake_per_animal.get(animal, 0.0)

            for compound, isomer in compounds_isomers:
                # Average hay concentration during exposure period
                if in_exposure:
                    hay_conc = hay_conc_dict.get((compound, isomer), 0.0)
                else:
                    hay_conc = 0.0

                # Intake = concentration × feed intake during exposure, else zero
                if in_exposure and hay_conc > 0 and feed_intake > 0:
                    intake = hay_conc * feed_intake
                else:
                    intake = 0.0

                results.append(
                    {
                        "Animal": animal,
                        "Day": day,
                        "Compound": compound,
                        "Isomer": isomer,
                        "PFAS_Intake_ug_day": float(intake),
                        "Hay_Concentration_ug_kg": hay_conc if in_exposure else None,
                        "Feed_Intake_kg_day": feed_intake,
                    }
                )

    intake_df = pd.DataFrame(results)
    return intake_df


def main() -> None:
    """
    Entry point: compute and save daily PFAS intake table.
    """
    paths = _get_paths()

    hay_df, feed_df = load_data(paths["pfas_data_file"], paths["feed_data_file"])
    hay_avg = calculate_hay_average(hay_df)
    intake_df = combined_intake_dataframe(feed_df, hay_avg)

    # Save data
    paths["output_file"].parent.mkdir(parents=True, exist_ok=True)
    intake_df.to_csv(paths["output_file"], index=False)

    print(f"Intake data saved to {paths['output_file']}")


if __name__ == "__main__":
    main()

