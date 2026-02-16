from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd

# Ensure project root is on sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auxiliary.project_paths import get_data_root


# Parameters
LOQ = 0.5  # µg/L; replace sub-LOQ and zero values with LOQ/2 = 0.25 (plasma/tissues)
LOQ_sub = LOQ / 2
MILK_LOQ = 0.005  # µg/L; LOQ for milk concentration
DEPURATION_START_DAY = 55  # Day after which depuration/slaughter data is used

# Tissue matrices to analyze
TISSUE_LABELS = ['Liver', 'Kidney', 'Lung', 'Spleen', 'Heart', 'Muscle', 'Brain']

# File paths (use centralised project paths)
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = get_data_root() / 'raw' / 'pfas_data_no_e1.csv'
PARTITIONS_OUT_DIR = get_data_root() / 'processed'


def load_and_filter_data(data_path: Path) -> tuple:
    data = pd.read_csv(data_path)
    
    # Get plasma and tissue entries after exposure (depuration/slaughter period)
    plasma = data[(data['Matrix'] == 'Plasma') & (data['Day'] > DEPURATION_START_DAY)].copy()
    tissues = data[data['Matrix'].isin(TISSUE_LABELS) & (data['Day'] > DEPURATION_START_DAY)].copy()
    
    return data, plasma, tissues


def handle_loq_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Concentration'] = pd.to_numeric(df['Concentration'], errors='coerce')
    df['Concentration'] = df['Concentration'].apply(
        lambda x: LOQ_sub if pd.isna(x) or x < LOQ else x
    )
    return df


def estimate_plasma_at_slaughter(group: pd.DataFrame, target_day: int) -> tuple[float, str]:
    # Check if measurement exists at target day
    match = group[group['Day'] == target_day]
    if not match.empty:
        return match['Concentration'].iloc[0], 'measured'
    
    # Estimate using exponential decay if two or more depuration points available
    if group.shape[0] >= 2:
        late = group.nlargest(2, 'Day').sort_values('Day')
        t1, c1 = late.iloc[0]['Day'], late.iloc[0]['Concentration']
        t2, c2 = late.iloc[1]['Day'], late.iloc[1]['Concentration']
        
        if c1 > 0 and c2 > 0 and (t2 - t1) != 0:
            k = -np.log(c2 / c1) / (t2 - t1)
            days_ahead = target_day - t2
            est = c2 * np.exp(-k * days_ahead)
            # If estimated concentration is below LOQ, substitute LOQ/2
            final_est = LOQ_sub if (pd.isna(est) or est < LOQ) else est
            return final_est, 'estimated'
    
    # Not enough data to estimate
    return np.nan, 'missing'


def calculate_plasma_estimates(plasma: pd.DataFrame, slaughter_days: dict[str, int]) -> pd.DataFrame:
    records = []
    for (animal, compound, isomer), group in plasma.groupby(
        ['Animal', 'Compound', 'Isomer']
    ):
        target_day = slaughter_days.get(animal)
        if target_day is None:
            conc, flag = np.nan, 'missing'
        else:
            conc, flag = estimate_plasma_at_slaughter(group, target_day)
        records.append({
            'Animal': animal,
            'Compound': compound,
            'Isomer': isomer,
            'Plasma_Slaughter': conc,
            'Plasma_Flag': flag,
        })

    return pd.DataFrame.from_records(records)


def calculate_partition_coefficient(
    row: pd.Series, 
    empirical_pcs: Optional[dict] = None
) -> tuple:
    tissue_conc = row['Concentration']
    plasma_conc = row['Plasma_Slaughter']
    tissue_flag = row['Tissue_Flag']
    plasma_flag = row['Plasma_Flag']
    
    # Case 1: Both tissue and plasma are sub-LOQ (fast-eliminating compound)
    if (tissue_flag == 'sub-LOQ' and plasma_flag in ['measured', 'estimated'] 
        and plasma_conc == LOQ_sub):
        tissue = row.get('Matrix')
        if empirical_pcs is not None:
            return empirical_pcs.get(tissue, 1.0), 'empirical_low'
        else:
            # First pass: use placeholder, will be filled later
            return np.nan, 'empirical_low'
    
    # Case 2: Tissue is sub-LOQ but plasma is measurable
    elif tissue_flag == 'sub-LOQ' and plasma_conc > LOQ_sub:
        return tissue_conc / plasma_conc, 'tissue_sub_loq'
    
    # Case 3: Plasma is sub-LOQ but tissue is measurable
    elif (plasma_flag in ['measured', 'estimated'] and plasma_conc == LOQ_sub 
          and tissue_conc > LOQ_sub):
        return tissue_conc / plasma_conc, 'plasma_sub_loq'
    
    # Case 4: Both are measurable
    elif tissue_conc > LOQ_sub and plasma_conc > LOQ_sub:
        return tissue_conc / plasma_conc, 'both_measured'
    
    # Case 5: Missing data
    else:
        return np.nan, 'missing_data'


def compute_partition_coefficients(
    tissues_slaughter: pd.DataFrame,
    plasma_estimates: pd.DataFrame
) -> pd.DataFrame:
    # Merge tissue and plasma data
    merged = pd.merge(
        tissues_slaughter,
        plasma_estimates,
        on=['Animal', 'Compound', 'Isomer'],
        how='left'
    )
    
    # First pass: Calculate direct ratios (cases 2-5), leave Case 1 as NaN
    pc_results = merged.apply(calculate_partition_coefficient, axis=1, empirical_pcs=None)
    merged['Partition_Coefficient'] = [result[0] for result in pc_results]
    merged['PC_Calculation_Type'] = [result[1] for result in pc_results]
    
    # Calculate tissue-specific medians from direct ratio results
    direct_ratio_mask = merged['PC_Calculation_Type'].isin(
        ['tissue_sub_loq', 'plasma_sub_loq', 'both_measured']
    )
    direct_ratio_data = merged[direct_ratio_mask].dropna(subset=['Partition_Coefficient'])
    tissue_medians = (
        direct_ratio_data.groupby('Matrix')['Partition_Coefficient']
        .median()
        .to_dict()
    )
    
    # Second pass: Fill in Case 1 (fast-eliminating) using calculated medians
    empirical_low_mask = merged['PC_Calculation_Type'] == 'empirical_low'
    for idx in merged[empirical_low_mask].index:
        tissue = merged.loc[idx, 'Matrix']
        merged.loc[idx, 'Partition_Coefficient'] = tissue_medians.get(tissue, 1.0)
    
    # Add quality flag
    merged['Quality_Flag'] = merged['PC_Calculation_Type'].map({
        'empirical_low': 'fast_eliminating_empirical',
        'tissue_sub_loq': 'tissue_sub_loq',
        'plasma_sub_loq': 'plasma_sub_loq',
        'both_measured': 'both_measured',
        'missing_data': 'missing_data'
    })
    
    return merged, tissue_medians


def add_empirical_partition_coefficients(
    mean_pc_by_tissue: pd.DataFrame,
    all_compounds: pd.DataFrame,
    tissue_medians: dict
) -> pd.DataFrame:
    # Get compounds that have partition coefficients
    compounds_with_pc = mean_pc_by_tissue[['Compound', 'Isomer']].drop_duplicates()
    
    # Find compounds without partition coefficients
    compounds_without_pc = all_compounds.merge(
        compounds_with_pc,
        on=['Compound', 'Isomer'],
        how='left',
        indicator=True
    )
    compounds_without_pc = compounds_without_pc[
        compounds_without_pc['_merge'] == 'left_only'
    ][['Compound', 'Isomer']]
    
    if compounds_without_pc.empty:
        return mean_pc_by_tissue
    
    # Define empirical partition coefficients using tissue-level medians
    empirical_pcs = {m: tissue_medians.get(m, 1.0) for m in TISSUE_LABELS}
    
    # Create empirical partition coefficient entries
    empirical_entries = []
    for _, row in compounds_without_pc.iterrows():
        compound = row['Compound']
        isomer = row['Isomer']
        
        for matrix, pc_value in empirical_pcs.items():
            empirical_entries.append({
                'Matrix': matrix,
                'Compound': compound,
                'Isomer': isomer,
                'Mean_Partition_Coefficient': pc_value
            })
    
    if empirical_entries:
        empirical_df = pd.DataFrame(empirical_entries)
        mean_pc_by_tissue = pd.concat([mean_pc_by_tissue, empirical_df], ignore_index=True)
    
    return mean_pc_by_tissue


def compute_mammary_partition_coefficients(data: pd.DataFrame) -> pd.DataFrame:
    # Filter to plasma and milk records
    sub = data[
        data["Matrix"].str.lower().isin(["plasma", "milk"])
    ][["Animal", "Day", "Matrix", "Compound", "Isomer", "Concentration"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Matrix", "Compound", "Isomer", "Mean_Partition_Coefficient"])

    # Ensure numeric concentrations
    sub["Concentration"] = pd.to_numeric(sub["Concentration"], errors="coerce")

    entries: list[dict] = []

    # Loop over compound–isomer pairs
    for (compound, isomer), pair_df in sub.groupby(["Compound", "Isomer"]):
        # Pivot to get paired plasma/milk per Animal–Day
        pivot = pair_df.pivot_table(
            index=["Animal", "Day"],
            columns="Matrix",
            values="Concentration",
            aggfunc="mean",
        )
        # Normalise column names to lower case
        cols = {c: c.lower() for c in pivot.columns}
        pivot.rename(columns=cols, inplace=True)

        if not {"plasma", "milk"}.issubset(pivot.columns):
            continue

        detected = pivot[
            (pivot["plasma"] > LOQ) & (pivot["milk"] > MILK_LOQ)
        ].dropna(subset=["plasma", "milk"])
        if detected.empty:
            continue

        x = detected["plasma"].to_numpy(dtype=float)
        y = detected["milk"].to_numpy(dtype=float)
        if x.size < 2:
            continue

        num = float((x * y).sum())
        den = float((x * x).sum())
        if den <= 0:
            continue

        slope = num / den
        if slope <= 0 or not np.isfinite(slope):
            continue

        entries.append({
            "Matrix": "Milk",
            "Compound": compound,
            "Isomer": isomer,
            "Mean_Partition_Coefficient": slope,
        })

    if not entries:
        return pd.DataFrame(columns=["Matrix", "Compound", "Isomer", "Mean_Partition_Coefficient"])

    return pd.DataFrame(entries)

def main() -> None:
    # Load and filter data
    data, plasma, tissues = load_and_filter_data(DATA_PATH)
    
    # Handle LOQ values
    plasma = handle_loq_values(plasma)
    tissues = handle_loq_values(tissues)
    
    # Find slaughter day for each animal (maximum day with tissue measurements)
    slaughter_days = tissues.groupby('Animal')['Day'].max().to_dict()
    
    # Calculate plasma estimates at slaughter day
    plasma_estimates = calculate_plasma_estimates(plasma, slaughter_days)
    
    # Prepare tissue data on slaughter day
    tissues_slaughter = tissues[
        tissues.apply(lambda row: row['Day'] == slaughter_days[row['Animal']], axis=1)
    ].copy()
    
    # Flag tissue values < LOQ
    tissues_slaughter['Tissue_Flag'] = np.where(
        tissues_slaughter['Concentration'] == LOQ_sub, 'sub-LOQ', 'measured'
    )
    
    # Compute partition coefficients
    merged, tissue_medians = compute_partition_coefficients(tissues_slaughter, plasma_estimates)
    
    # Aggregate by tissue, compound, isomer
    mean_pc_by_tissue = (
        merged.groupby(['Matrix', 'Compound', 'Isomer'])['Partition_Coefficient']
        .mean()
        .reset_index()
        .rename(columns={'Partition_Coefficient': 'Mean_Partition_Coefficient'})
    )
    
    # Add empirical partition coefficients for compounds with no tissue data
    print("Adding empirical partition coefficients for compounds with no tissue data...")
    all_compounds = data[['Compound', 'Isomer']].drop_duplicates()
    mean_pc_by_tissue = add_empirical_partition_coefficients(
        mean_pc_by_tissue, all_compounds, tissue_medians
    )

    # Compute mammary (plasma–milk) partition coefficients and append
    print("Computing plasma–milk partition coefficients for mammary/milk space...")
    mammary_pc = compute_mammary_partition_coefficients(data)
    if not mammary_pc.empty:
        mean_pc_by_tissue = pd.concat([mean_pc_by_tissue, mammary_pc], ignore_index=True)
        print(f"Added {len(mammary_pc)} mammary partition coefficient entries.")
    else:
        print("No mammary partition coefficients could be computed from data.")
    
    # Save mean partition coefficients
    PARTITIONS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mean_partitions_path = PARTITIONS_OUT_DIR / 'partitions_mean.csv'
    mean_pc_by_tissue.to_csv(mean_partitions_path, index=False)
    print(f"Saved mean partition coefficients to: {mean_partitions_path}")
    print(f"Total partition coefficient entries: {len(mean_pc_by_tissue)}")

if __name__ == "__main__":
    main()