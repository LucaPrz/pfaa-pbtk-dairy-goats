"""
Generate a unified `data/raw/animal_data.csv` from the original
Excel workbook `data/original/Tierdaten_neu.xlsx`.

Output columns (long format):
  - Animal              (E2, E3, E4)
  - Day                 (integer, starting at 1 from first Tierdaten date)
  - Date                (calendar date, ISO format)
  - Body_Weight_kg      (daily, linearly interpolated from recorded weights)
  - Feed_Intake_kg_per_day  (Futteraufnahme, 88% DM, kg/day)
  - Milk_Yield_kg_per_day   (Milchmenge, L/day ≈ kg/day)

This file can be used as a single source of truth for physiology and
exposure (intake + milk yield) per animal per day.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_data_root  # type: ignore


ANIMALS = ["E2", "E3", "E4"]


def load_tierdaten() -> pd.DataFrame:
    data_root = get_data_root()
    xls_path = data_root / "original" / "Tierdaten_neu.xlsx"
    if not xls_path.exists():
        raise FileNotFoundError(f"Tierdaten workbook not found: {xls_path}")
    xls = pd.ExcelFile(xls_path)
    df = pd.read_excel(xls, sheet_name="Tierdaten")
    return df


def parse_tierdaten(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, dict]:
    """
    Split Tierdaten into:
      - data: rows with valid dates, sorted by date
      - mapping dicts for columns:
          weight_cols[animal] -> column name
          feed_cols[animal]   -> column name
          milk_cols[animal]   -> column name
    """
    header = df.iloc[0]
    data = df.iloc[1:].copy()

    date_col = "Unnamed: 2"
    data = data[data[date_col].notna()].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)

    # Identify column blocks by header labels in row 0
    weight_cols: dict[str, str] = {}
    feed_cols: dict[str, str] = {}
    milk_cols: dict[str, str] = {}

    for col in df.columns:
        label = header[col]
        if pd.isna(label):
            continue
        if label in ANIMALS:
            idx = df.columns.get_loc(col)
            # Heuristic: based on observed layout
            if "Gewicht" in col or 15 <= idx <= 17:
                weight_cols[str(label)] = col
            elif "Futteraufnahme" in col or 7 <= idx <= 10:
                feed_cols[str(label)] = col
            elif "Milchmenge" in col or 31 <= idx <= 33:
                milk_cols[str(label)] = col

    return data, weight_cols, feed_cols, milk_cols


def build_animal_data() -> pd.DataFrame:
    df = load_tierdaten()
    data, weight_cols, feed_cols, milk_cols = parse_tierdaten(df)
    date_col = "Unnamed: 2"

    # Daily date range across entire experiment
    all_dates = pd.date_range(data[date_col].min(), data[date_col].max(), freq="D")

    rows: list[dict[str, object]] = []

    for animal in ANIMALS:
        w_col = weight_cols.get(animal)
        f_col = feed_cols.get(animal)
        m_col = milk_cols.get(animal)
        if w_col is None or f_col is None or m_col is None:
            continue

        sub = data[[date_col, w_col, f_col, m_col]].set_index(date_col).astype(float)

        # Reindex to daily and interpolate/ffill/bfill for weight
        bw = sub[w_col].reindex(all_dates)
        bw = bw.interpolate(method="time").ffill().bfill()

        # For feed and milk we keep recorded values and forward-fill within gaps,
        # assuming intake/yield is piecewise constant between measurement days.
        feed = sub[f_col].reindex(all_dates)
        feed = feed.ffill()
        milk = sub[m_col].reindex(all_dates)
        milk = milk.ffill()

        for d in all_dates:
            rows.append(
                {
                    "Animal": animal,
                    "Date": d.date(),
                    "Body_Weight_kg": float(bw.loc[d]),
                    "Feed_Intake_kg_per_day": float(feed.loc[d]) if np.isfinite(feed.loc[d]) else np.nan,
                    "Milk_Yield_kg_per_day": float(milk.loc[d]) if np.isfinite(milk.loc[d]) else np.nan,
                }
            )

    out = pd.DataFrame(rows)

    # Day index: Day 0 = first date (baseline), Day 1 = same first date (first exposure), Day 2, 3, ...
    # So Day 0 and Day 1 are the same calendar date; Day 0 has 0 feed (baseline), Day 1 has actual feed.
    out["Date_dt"] = pd.to_datetime(out["Date"])
    start_date = out["Date_dt"].min()
    out["Day"] = (out["Date_dt"] - start_date).dt.days + 1  # 1, 2, 3, ...
    out = out.drop(columns=["Date_dt"])

    # Insert Day 0 for each animal: same date as Day 1, feed = 0 (baseline)
    day1_rows = out[out["Day"] == 1].copy()
    day1_rows["Day"] = 0
    day1_rows["Feed_Intake_kg_per_day"] = 0.0
    out = pd.concat([day1_rows, out], ignore_index=True)

    # Sort and reorder columns
    out = out.sort_values(["Animal", "Day"])
    out = out[
        [
            "Animal",
            "Day",
            "Date",
            "Body_Weight_kg",
            "Feed_Intake_kg_per_day",
            "Milk_Yield_kg_per_day",
        ]
    ]
    return out


def main() -> None:
    data_root = get_data_root()
    out_df = build_animal_data()
    out_path = data_root / "raw" / "animal_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved animal data to {out_path}")


if __name__ == "__main__":
    main()

