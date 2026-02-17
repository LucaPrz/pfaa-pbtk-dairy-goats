"""
Compare raw vs moving-average PFAA intake to see if smoothing changes intake significantly.

Usage (from project root): python analysis/compare_intake_moving_average.py
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.fit import build_intake_function
from optimization.config import SimulationConfig

def main():
    data_root = Path(__file__).resolve().parent.parent / "data"
    intake_path = data_root / "processed" / "pfas_daily_intake.csv"
    intake_df = pd.read_csv(intake_path)

    # Exposure period: days 1--56
    exposure_days = np.arange(1, 57, dtype=int)
    window = 10  # default in config

    # Sample a few animal/compound/isomer combinations that have non-zero intake
    sub = intake_df[
        (intake_df["Day"] >= 1) & (intake_df["Day"] <= 56) &
        (intake_df["PFAS_Intake_ug_day"] > 0)
    ]
    keys = sub.groupby(["Animal", "Compound", "Isomer"]).agg(
        total_raw=("PFAS_Intake_ug_day", "sum")
    ).reset_index()

    results = []
    for _, row in keys.head(20).iterrows():
        animal, compound, isomer = row["Animal"], row["Compound"], row["Isomer"]
        sim_config = SimulationConfig(compound=compound, isomer=isomer, animal=animal)

        raw_func = build_intake_function(sim_config, intake_df, moving_average_window=None)
        smooth_func = build_intake_function(sim_config, intake_df, moving_average_window=window)

        raw_vals = raw_func(exposure_days)
        smooth_vals = smooth_func(exposure_days)
        raw_vals = np.atleast_1d(raw_vals)
        smooth_vals = np.atleast_1d(smooth_vals)

        total_raw = float(np.sum(raw_vals))
        total_smooth = float(np.sum(smooth_vals))
        if total_raw <= 0:
            continue
        pct_change_total = 100 * (total_smooth - total_raw) / total_raw

        # Per-day relative differences (where raw > 0)
        mask = raw_vals > 0
        if mask.any():
            rel_diff = np.abs(smooth_vals[mask] - raw_vals[mask]) / raw_vals[mask]
            mean_rel_diff_pct = 100 * float(np.mean(rel_diff))
            max_rel_diff_pct = 100 * float(np.max(rel_diff))
        else:
            mean_rel_diff_pct = max_rel_diff_pct = np.nan

        results.append({
            "Animal": animal,
            "Compound": compound,
            "Isomer": isomer,
            "total_raw_ug": total_raw,
            "total_smooth_ug": total_smooth,
            "pct_change_total": pct_change_total,
            "mean_abs_rel_diff_pct": mean_rel_diff_pct,
            "max_abs_rel_diff_pct": max_rel_diff_pct,
        })

    res_df = pd.DataFrame(results)
    print("Moving average window:", window, "days")
    print("\nTotal intake (exposure period): raw vs smoothed")
    print(res_df[["Animal", "Compound", "Isomer", "total_raw_ug", "total_smooth_ug", "pct_change_total"]].to_string())
    print("\nSummary across sampled animal/compound/isomer:")
    print("  Total intake: mean % change = {:.2f}%, max |% change| = {:.2f}%".format(
        res_df["pct_change_total"].mean(),
        res_df["pct_change_total"].abs().max()
    ))
    print("  Per-day (where raw>0): mean |relative diff| = {:.1f}%, max = {:.1f}%".format(
        res_df["mean_abs_rel_diff_pct"].mean(),
        res_df["max_abs_rel_diff_pct"].max()
    ))

if __name__ == "__main__":
    main()
