from pathlib import Path
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, Optional
import warnings

# Import physiological curves (same directory).
# The original code expected a `physiological` module; in this cleaned
# project the curves live in `physiology_submodel`, so we import it
# under the expected name to keep downstream code unchanged.
from . import physiology_submodel as physiological

HEMATOCRIT = 0.27
CARDIAC_OUTPUT = 8.17 * (1 - HEMATOCRIT)  # L/(h·kg) plasma-equivalent for goats

VOLUME_FACTORS = {
    "V_plasma": 0.0549 * (1 - HEMATOCRIT),
    "V_liver": 0.0189,
    "V_stomach": 0.0369,
    "V_intestine": 0.0336,
    "V_spleen": 0.0031,
    "V_kidney": 0.0038,
    "V_muscle": 0.3858,
    "V_heart": 0.0044,
    "V_brain": 0.0032,
    "V_lung": 0.0122,
}


COW_VOLUME_FACTORS = {
    # Blood 4.52% -> plasma fraction using global HEMATOCRIT
    "V_plasma": 0.0452 * (1 - HEMATOCRIT),
    "V_liver": 0.0131,        # 1.31%
    "V_stomach": (0.0019 + 0.0102 + 0.0066 + 0.0029),  # reticulum + rumen + omasum + abomasum
    "V_intestine": (0.0082 + 0.0055),  # small + large intestine
    "V_spleen": 0.0016,       # 0.16%
    "V_kidney": 0.0025,       # 0.25%
    "V_muscle": 0.3610,       # 36.10% (from generic dairy cow table)
    "V_heart": 0.0037,        # 0.37%
    "V_brain": 0.0007,        # 0.07%
    "V_lung": 0.0070,         # 0.70%
}

FLOW_FACTORS = {
    "Q_spleen": 0.0454,
    "Q_stomach": 0.0776,
    "Q_intestine": 0.1923,
    "Q_hepatic": 0.0120,
    "Q_kidney": 0.1298,
    "Q_muscle": 0.1009,
    "Q_heart": 0.0559,
    "Q_brain": 0.0295,
}

# Cow-specific plasma flow fractions (Holstein, percent of cardiac output)
# Kidneys: 7%, Hepatic artery: 10%, Portal vein (intestine drainage): 49%
# Other organs (spleen, stomach, muscle, heart, brain) use goat fractions
COW_FLOW_FACTORS = {
    "Q_spleen": 0.0454,    # approximated from goat (no cow-specific data)
    "Q_stomach": 0.0776,   # approximated from goat
    "Q_intestine": 0.49,   # 49% portal vein
    "Q_hepatic": 0.10,     # 10% hepatic artery
    "Q_kidney": 0.07,      # 7% kidneys
    "Q_muscle": 0.1009,    # approximated from goat
    "Q_heart": 0.0559,     # approximated from goat
    "Q_brain": 0.0295,     # approximated from goat
}

# Default absorption and excretion rates
DEFAULT_ABSORPTION_EXCRETION = {
    "k_sto": (2.53 + (1.22 * 1.0)) - ((2.61 * (1.0 ** 2)) / 6000.0),  # NI=1.0, PCO=1.0
    "k_a": 10,
    "k_feces": 0.05,
    "k_renal": 0,
    "k_ehc": 1.69,
    "k_elim": 0,
}

def calculate_physiology_from_body_weight(
    body_weight: float,
    include_milk_scale: bool = False,
    days: Optional[float] = None,
    parity: Optional[str] = None,
    breed: Optional[str] = None,
) -> Dict[str, float]:
    # Choose species-specific volume and flow factors (default: goats)
    if breed is not None and breed.startswith("Holstein_cow"):
        vol_factors = COW_VOLUME_FACTORS
        flow_factors = COW_FLOW_FACTORS
        # Cow-specific cardiac output and hematocrit (Holstein)
        cow_hematocrit = 0.378
        cow_cardiac_output = 8.7 * (1 - cow_hematocrit)  # L/(h·kg) plasma-equivalent
        plasma_flow_per_hour = body_weight * cow_cardiac_output
    else:
        vol_factors = VOLUME_FACTORS
        flow_factors = FLOW_FACTORS
        plasma_flow_per_hour = body_weight * CARDIAC_OUTPUT
    
    # Calculate volumes (all in L)
    physiological_dict = {}
    for vol_name, factor in vol_factors.items():
        physiological_dict[vol_name] = factor * body_weight
    
    # Calculate flows (all in L/h)
    for flow_name, factor in flow_factors.items():
        physiological_dict[flow_name] = factor * plasma_flow_per_hour
    
    # Q_lung equals total plasma flow
    physiological_dict["Q_lung"] = plasma_flow_per_hour
    
    # Calculate Q_rest and V_rest (residual terms)
    physiological_dict["Q_rest"] = plasma_flow_per_hour - sum([
        physiological_dict["Q_spleen"], physiological_dict["Q_intestine"],
        physiological_dict["Q_hepatic"], physiological_dict["Q_kidney"],
        physiological_dict["Q_muscle"], physiological_dict["Q_heart"],
        physiological_dict["Q_brain"], physiological_dict["Q_stomach"],
    ])
    
    physiological_dict["V_rest"] = body_weight - sum([
        physiological_dict["V_spleen"], physiological_dict["V_intestine"],
        physiological_dict["V_liver"], physiological_dict["V_kidney"],
        physiological_dict["V_muscle"], physiological_dict["V_heart"],
        physiological_dict["V_brain"], physiological_dict["V_plasma"],
        physiological_dict["V_stomach"], physiological_dict["V_lung"]
    ])
    
    # Add milk_yield if requested (actual milk production in kg/day)
    if include_milk_scale:
        if days is None or parity is None:
            raise ValueError("days and parity required when include_milk_scale=True")
        milk_yield = physiological.lactation_curve(np.array([days]), parity)[0]
        # Store actual milk yield (kg/day) for direct use in excretion with
        # plasma–milk partition-based milk route
        physiological_dict["milk_yield"] = milk_yield
    
    return physiological_dict

def build_milk_yield_function(
    animal: str, 
    milk_yield_array: np.ndarray
) -> Callable[[float], float]:
    def milk_yield_func(t: float) -> float:
        day = int(t)
        if 0 <= day < len(milk_yield_array):
            milk_yield = float(milk_yield_array[day])
            if milk_yield <= 0:
                warnings.warn(
                    f"No valid milk yield found for animal {animal} at day {day} (value: {milk_yield}). "
                    f"Using 0.0 for milk excretion.",
                    UserWarning
                )
            return milk_yield
        warnings.warn(
            f"No milk yield data available for animal {animal} at day {day} "
            f"(array length: {len(milk_yield_array)}). Using 0.0 for milk excretion.",
            UserWarning
        )
        return 0.0
    return milk_yield_func

def load_partition_coefficients(compound: str, isomer: str) -> Dict[str, float]:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    partitions_path = project_root / "data" / "processed" / "partitions_mean.csv"
    part_df = pd.read_csv(partitions_path)
    query = (part_df["Compound"] == compound) & (part_df["Isomer"] == isomer)
    filtered = part_df[query]
    if filtered.empty:
        raise ValueError(
            f"No partition coefficients found in {partitions_path} "
            f"for Compound='{compound}', Isomer='{isomer}'."
        )
    
    partition_dict = dict(zip(filtered["Matrix"], filtered["Mean_Partition_Coefficient"]))
    # Ensure required entries exist with sensible fallbacks
    # P_milk: plasma–milk partition (canonical name). CSV may use "Mammary" or "Milk"
    p_milk = partition_dict.get("Mammary", partition_dict.get("Milk", 1.0))
    partition_dict["P_milk"] = p_milk
    partition_dict.setdefault("Rest", 0.01)
    return partition_dict

def build_parameters(config, fit_params=None) -> dict:
    # Get body weight from config
    BODY_WEIGHTS = {"E2": 47.0, "E3": 35.9, "E4": 37.8}
    body_weight = BODY_WEIGHTS[config['animal']]
    
    # Calculate physiological parameters from body weight (goat default)
    physiological_dict = calculate_physiology_from_body_weight(body_weight)
    
    # Load partition coefficients
    compound = config['compound']
    isomer = config['isomer']
    partition_coefficients = load_partition_coefficients(compound, isomer)
    
    # Get absorption/excretion rates (copy to avoid modifying defaults)
    absorption_excretion = DEFAULT_ABSORPTION_EXCRETION.copy()
    
    # Override with fit parameters
    for param_dict in [absorption_excretion, partition_coefficients]:
        for key in list(param_dict.keys()):
            if fit_params and key in fit_params:
                param_dict[key] = fit_params[key]
    
    parameters = {
        "physiological": physiological_dict,
        "partition_coefficients": partition_coefficients,
        "absorption_excretion": absorption_excretion,
    }
    
    return parameters


def build_dynamic_physiology_provider(
    breed: Optional[str] = None,
    parity: Optional[str] = None,
    time_unit: str = "days",
    body_weight: Optional[float] = None,
    body_weight_array: Optional[np.ndarray] = None,
    milk_yield_array: Optional[np.ndarray] = None,
    dmi_array: Optional[np.ndarray] = None,
    dim_offset: float = 0.0,
) -> Callable[[float], Dict[str, float]]:
    # Determine if we need curves (when measured data is not provided)
    use_measured_body_weight = (body_weight is not None) or (body_weight_array is not None)
    use_measured_milk_yield = milk_yield_array is not None
    use_measured_dmi = dmi_array is not None
    
    # If any curves are needed, breed/parity are required
    if not (use_measured_body_weight and use_measured_milk_yield):
        if breed is None or parity is None:
            raise ValueError(
                "breed and parity are required when using curves. "
                "Either provide measured data (body_weight, milk_yield_array) "
                "or provide breed and parity for curve calculation."
            )
    
    # Get body weight curve parameters (for fallback, if needed)
    bw_params = None
    if breed is not None and parity is not None:
        bw_params = physiological.get_params(breed, parity)
    
    
    def physiology_provider(t: float) -> Dict[str, float]:
        # Convert time to days in milk (DIM) if needed, then apply optional offset
        if time_unit == "hours":
            days = t / 24.0
        elif time_unit == "days":
            days = t
        else:
            raise ValueError(f"Unknown time_unit: {time_unit}. Must be 'days' or 'hours'")
        dim = days + dim_offset
        
        # Get body weight (measured array, measured static, or calculated)
        if body_weight_array is not None:
            day_idx = int(days)
            if 0 <= day_idx < len(body_weight_array):
                BW = float(body_weight_array[day_idx])
            else:
                # Fallback: if out of range, use last available value
                BW = float(body_weight_array[-1])
        elif body_weight is not None:
            BW = body_weight
        else:
            if bw_params is None:
                raise ValueError(
                    "Cannot calculate body weight: breed/parity not provided and no "
                    "measured body_weight or body_weight_array supplied"
                )
            # Body weight curve as a function of DIM
            BW = physiological.body_weight_curve(np.array([dim]), bw_params)[0]
        
        # Calculate base physiology from body weight (respect breed for species-specific volumes)
        phys = calculate_physiology_from_body_weight(BW, breed=breed)
        
        # Get milk yield (measured or calculated)
        if use_measured_milk_yield:
            day = int(days)
            if 0 <= day < len(milk_yield_array):
                milk_yield = float(milk_yield_array[day])
                if milk_yield <= 0:
                    milk_yield = 0.0
            else:
                milk_yield = 0.0
        else:
            # Use lactation curve
            if parity is None:
                raise ValueError("Cannot calculate milk yield: parity not provided and milk_yield_array not provided")
            milk_yield = physiological.lactation_curve(np.array([dim]), parity, breed=breed)[0]
        
        phys["milk_yield"] = milk_yield
        
        # Get DMI (measured or calculated) - for reference, not used in physiology calculation
        if use_measured_dmi:
            day = int(days)
            if 0 <= day < len(dmi_array):
                dmi = float(dmi_array[day])
            else:
                dmi = 0.0
        else:
            # Calculate from curves (requires breed/parity, but DMI is optional so we can skip if not available)
            try:
                BW_array = np.array([BW])
                lactation_array = np.array([milk_yield])
                dmi = physiological.dry_matter_intake_curve(
                    np.array([dim]), BW_array, lactation_array
                )[0]
            except Exception:
                # DMI is optional, so we can skip if calculation fails
                dmi = 0.0
        
        phys["DMI"] = dmi
        
        return phys
    
    return physiology_provider