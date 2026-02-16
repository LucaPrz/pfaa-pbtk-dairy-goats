"""Determine which parameters to fit based on data signals."""
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np

# Full model: mammary compartment with fixed plasma–milk partition (no free k_milk)
ALL_PARAMETERS = ["k_ehc", "k_elim", "k_renal", "k_a", "k_feces"]
# Simple model: perfusion-limited milk from plasma, E_milk
ALL_PARAMETERS_SIMPLE = ["E_milk", "k_ehc", "k_elim", "k_renal", "k_a", "k_feces"]

REQUIRED_PARAMETERS = ["k_a", "k_elim"]  # Same for both models

SIGNAL_DEPENDENT_PARAMETERS = {
    # Milk route is governed by fixed plasma–milk partition and milk yield,
    # so there is no free k_milk parameter in the full model anymore.
    "k_ehc": "feces_depuration_signal",
    "k_feces": "feces_signal",
    "k_renal": "urine_signal",
}
SIGNAL_DEPENDENT_PARAMETERS_SIMPLE = {
    "E_milk": "milk_signal",
    "k_ehc": "feces_depuration_signal",
    "k_feces": "feces_signal",
    "k_renal": "urine_signal",
}

def _check_matrix_signal(pair_data: pd.DataFrame, matrix_name: str, loq: float) -> bool:
    """Helper function to check if a matrix has detectable signal above LOQ."""
    matrix_data = pair_data[pair_data['Matrix'].str.lower() == matrix_name.lower()]
    if matrix_data.empty:
        return False
    conc = pd.to_numeric(matrix_data['Concentration'], errors='coerce').dropna()
    return (conc > loq).any()

def check_data_signals(
    compound: str,
    isomer: str,
    data_df: Optional[pd.DataFrame] = None
) -> Dict[str, bool]:
    # Check for data signals
    signals = {
        "milk_signal": False,
        "plasma_signal": False,
        "feces_signal": False,
        "feces_depuration_signal": False,
        "urine_signal": False,
    }
    
    # If we have raw data, check it directly
    if data_df is not None and not data_df.empty:
        pair_data = data_df[
            (data_df['Compound'] == compound) & 
            (data_df['Isomer'] == isomer)
        ]
        
        if not pair_data.empty:
            # TODO: These should come from config, but keeping hardcoded for now to avoid circular import
            # Consider refactoring to pass config as parameter
            EXPOSURE_PERIOD_DAYS = 56
            LOQ = 0.5
            LOQ_MILK = 0.005
            
            # Check signals for each matrix
            signals["intake_data"] = _check_matrix_signal(pair_data, 'hay', LOQ)
            signals["milk_signal"] = _check_matrix_signal(pair_data, 'milk', LOQ_MILK)
            signals["plasma_signal"] = _check_matrix_signal(pair_data, 'plasma', LOQ)
            signals["feces_signal"] = _check_matrix_signal(pair_data, 'feces', LOQ)
            signals["urine_signal"] = _check_matrix_signal(pair_data, 'urine', LOQ)
            
            # Check depuration phase signals (Day > EXPOSURE_PERIOD_DAYS)
            feces_data = pair_data[pair_data['Matrix'].str.lower() == 'feces']
            feces_depuration = feces_data[feces_data['Day'] > EXPOSURE_PERIOD_DAYS]
            if not feces_depuration.empty:
                feces_dep_conc = pd.to_numeric(feces_depuration['Concentration'], errors='coerce').dropna()
                signals["feces_depuration_signal"] = (feces_dep_conc > LOQ).any()
                    
    return signals

def get_parameter_config(
    compound: str,
    isomer: str,
    data_df: Optional[pd.DataFrame] = None,
    use_default: bool = True,
    config: Optional[Any] = None,
) -> Tuple[List[str], Dict[str, float]]:
    signals = check_data_signals(compound, isomer, data_df)
    any_signal = any([
        signals.get("milk_signal", False),
        signals.get("plasma_signal", False),
        signals.get("feces_signal", False),
        signals.get("feces_depuration_signal", False),
        signals.get("urine_signal", False),
    ])
    if not any_signal:
        return [], {}

    use_simple = getattr(config, "use_simple_model", False) if config is not None else False
    params_to_fit = list(REQUIRED_PARAMETERS)
    fixed = {}

    if signals.get("feces_signal", False):
        params_to_fit.append("k_feces")
    else:
        fixed["k_feces"] = 0.0
    if signals.get("feces_depuration_signal", False):
        params_to_fit.append("k_ehc")
    else:
        fixed["k_ehc"] = 0.0
    if signals.get("urine_signal", False):
        params_to_fit.append("k_renal")
    else:
        fixed["k_renal"] = 0.0

    seen = set()
    unique_params = []
    for p in params_to_fit:
        if p not in seen:
            seen.add(p)
            unique_params.append(p)
    return unique_params, fixed