"""Configuration classes for model fitting and Monte Carlo sampling."""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Any
from pathlib import Path
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Excretion streams (model outputs)
VALID_EXCRETION_STREAMS = {"feces", "urine", "milk"}


def get_matrix_module(config: Any) -> Any:
    """
    Return the matrix module containing PBTKModel.

    The core model class `PBTKModel` lives in `model/diagnose.py`.
    """
    from model import diagnose as matrix_module
    return matrix_module


def get_parameters_module(config: Any) -> Any:
    """Return the parameters module (parameters/parameters.py)."""
    from parameters import parameters as params_module
    return params_module


def get_valid_matrices(config: Any) -> Set[str]:
    """Valid matrix names for predictions/residuals (compartments + excretion streams)."""
    matrix_module = get_matrix_module(config)
    compartments = set(matrix_module.PBTKModel.compartment_idx.keys())
    return compartments | VALID_EXCRETION_STREAMS


@dataclass
class ModelConfig:
    """Configuration for model fitting and Monte Carlo sampling."""
    n_mc_samples: int = 10000  
    mc_chunk_size: int = 1000  # Process MC samples in chunks to reduce memory usage
    mc_random_seed: Optional[int] = 42  # Random seed for MC reproducibility

    # Default parameter names (used as fallback or for compounds not in signal data)
    # Mammary with fixed plasma–milk partition (no free k_milk)
    param_names: List[str] = field(default_factory=lambda: ["k_ehc", "k_elim", "k_renal", "k_a", "k_feces"])
    animals: List[str] = field(default_factory=lambda: ["E2", "E3", "E4"])
    compounds: Optional[List[str]] = None  # Will be determined from data if None
    fittable_pairs: Optional[List[Tuple[str, str]]] = None  # (compound, isomer) with hay > 0 and ≥1 matrix above LOQ
    eps: float = 1e-6
    time_vector: np.ndarray = field(default_factory=lambda: np.arange(0, 141, 1))
    matrix_weights: Dict[str, float] = field(default_factory=lambda: {
        "plasma": 1.0, "milk": 1.0, "liver": 0.5, "kidney": 0.5, "muscle": 0.8, 
        "heart": 0.1, "brain": 0.1, "spleen": 0.1, "lung": 0.1,
        "feces": 0.5, "urine": 0.5
    })
    loq: float = 0.5
    loq_milk: float = 0.005
    sigma_default: float = 0.4
    sigma_per_matrix: Dict[str, float] = field(default_factory=lambda: {
        "plasma": 0.4, "milk": 0.4, "liver": 0.5, "kidney": 0.5, "muscle": 0.5,
        "heart": 0.5, "brain": 0.5, "spleen": 0.5, "lung": 0.5,
        "feces": 0.5, "urine": 0.5
    })
    use_dynamic_parameters: bool = True  # Enable dynamic parameter selection based on data signals
    use_log_rmse_for_fitting: bool = True  # If True, fit with log RMSE then estimate sigma; if False, use censored likelihood directly

    def get_sigma(self, matrix_name: str, compound: Optional[str] = None, isomer: Optional[str] = None) -> float:
        try:
            from optimization.io import get_project_root
            project_root = get_project_root()
            # Original project stores pooled and per-pair sigma here
            sigma_dir = project_root / "results" / "optimization" / "sigma_estimates"

            # Prefer compound-level sigma for MC propagation (effective model error for that pair)
            if compound is not None and isomer is not None:
                pair_path = sigma_dir / f"sigma_{compound}_{isomer}.csv"
                if pair_path.exists():
                    sigma_df = pd.read_csv(pair_path)
                    row = sigma_df[sigma_df["Matrix"].str.lower() == matrix_name.lower()]
                    if not row.empty:
                        return float(row["Sigma"].iloc[0])

            # Fall back to pooled sigma (by matrix, across compounds)
            pooled_path = sigma_dir / "sigma_pooled.csv"
            if pooled_path.exists():
                sigma_df = pd.read_csv(pooled_path)
                row = sigma_df[sigma_df["Matrix"].str.lower() == matrix_name.lower()]
                if not row.empty:
                    return float(row["Sigma"].iloc[0])
        except Exception:
            pass

        return self.sigma_per_matrix.get(matrix_name.lower(), self.sigma_default)
    
    def get_compounds_from_data(self, data_path: Optional[Path] = None, data_df: Optional[pd.DataFrame] = None) -> List[str]:
        from optimization.fit_variables import check_pair_fittable

        # Compounds to ignore (summary/aggregate compounds)
        compounds_to_ignore = {
            "Sum", "Sum EFSA-4", "Sum branched", "Sum linear",
            "Summe PFCA", "Summe PFSA"
        }

        # Load data if not provided
        if data_df is None:
            if data_path is None:
                from optimization.io import get_project_root
                project_root = get_project_root()
                data_path = project_root / "data" / "raw" / "pfas_data_no_e1.csv"

            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")

            data_df = pd.read_csv(data_path)

        # Get all unique compound-isomer pairs (excluding "Total" isomer and ignored compounds)
        pairs = (
            data_df[["Compound", "Isomer"]]
            .drop_duplicates()
            .query('Isomer != "Total"')
            .query('Compound not in @compounds_to_ignore')
        )

        # Include a pair only if (a) hay concentration is not 0 and (b) at least one
        # of FIT_RELEVANT_MATRICES has a measurement above LOQ. Log matrices above LOQ.
        pairs_with_signals = []
        compounds_with_signals = set()

        for _, row in pairs.iterrows():
            compound = row["Compound"]
            isomer = row["Isomer"]
            is_fittable, matrices_above_loq = check_pair_fittable(
                compound, isomer, data_df, loq=self.loq, loq_milk=self.loq_milk
            )
            if is_fittable:
                pairs_with_signals.append((compound, isomer))
                compounds_with_signals.add(compound)
                logger.info(
                    f"  {compound} {isomer}: matrices above LOQ: %s",
                    ", ".join(matrices_above_loq),
                )

        self.fittable_pairs = sorted(pairs_with_signals, key=lambda x: (x[0], x[1]))
        compounds_list = sorted(list(compounds_with_signals))
        return compounds_list
    
    def get_param_names(self, compound: Optional[str] = None, isomer: Optional[str] = None, 
                       data_df: Optional[pd.DataFrame] = None) -> List[str]:
        if not self.use_dynamic_parameters or compound is None or isomer is None:
            return self.param_names
        
        try:
            from optimization.fit_variables import get_parameter_config
            params_to_fit, _ = get_parameter_config(compound, isomer, data_df, use_default=True, config=self)
            return params_to_fit
        except Exception:
            return self.param_names
    
    def get_fixed_parameters(self, compound: Optional[str] = None, isomer: Optional[str] = None,
                            data_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        if not self.use_dynamic_parameters or compound is None or isomer is None:
            return {}
        
        try:
            from optimization.fit_variables import get_parameter_config
            _, fixed_params = get_parameter_config(compound, isomer, data_df, config=self)
            return fixed_params
        except Exception:
            return {}
    
    def get_log_bounds(self, param_names: List[str]) -> List[Tuple[float, float]]:
        """
        Bounds per parameter in log10 space.

        Default global bounds correspond to linear 1e-3 to ~316 1/day. For some
        parameters (e.g. k_ehc), we restrict the range slightly to reduce
        extreme, weakly identifiable values that drive very wide tails in the
        extrapolation CIs without improving fit.
        """
        global_low, global_high = -3.0, 2.5
        per_param_bounds = {
            # Enterohepatic recirculation: restrict to 1e-2–1e2 1/day
            "k_ehc": (-3.0, 2.5),
        }
        bounds: List[Tuple[float, float]] = []
        for name in param_names:
            low, high = per_param_bounds.get(name, (global_low, global_high))
            bounds.append((low, high))
        return bounds
    
    de_config: Dict = field(default_factory=lambda: {
        'strategy': 'best1bin',
        'popsize': 15,
        'maxiter': 1000,
        'tol': 0.01,
        'mutation': (0.5, 1.0),
        'recombination': 0.7,
        'polish': True,
        'updating': 'deferred',
        'seed': 42
    })

@dataclass
class FitConfig:
    compound: str
    isomer: str
    animal: Optional[str] = None  # None for global fit, set for animal-specific operations

@dataclass
class SimulationConfig:
    compound: str
    isomer: str
    animal: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format expected by Model.parameters.build_parameters."""
        return {
            'compound': self.compound,
            'isomer': self.isomer,
            'animal': self.animal
        }

# Forward declaration to avoid circular import
# Will be properly imported in context.py or run.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from optimization.io import DataCache

@dataclass
class FittingContext:
    config: ModelConfig
    data_cache: 'DataCache'  # Forward reference to avoid circular import
    intake_df: pd.DataFrame
    urine_volume_by_animal: Dict[str, float]
    feces_mass_by_animal: Dict[str, float]
    feces_mass_default: float
    milk_yield_by_animal: Dict[str, np.ndarray]
    body_weight_by_animal: Dict[str, np.ndarray]
    folder_phase1: Path
    folder_phase2: Path
    folder_phase3: Path

def setup_context(project_root: Optional[Path] = None) -> FittingContext:
    # Import here to avoid circular imports
    from optimization.io import DataCache, load_data, get_project_root

    if project_root is None:
        # Project root is the repository root (pfaa-pbtk-dairy-goats)
        project_root = get_project_root()

    config = ModelConfig()

    # Use original optimisation folder structure under results/optimization/
    # so this clean project remains compatible with existing fit files.
    results_root = project_root / "results" / "optimization"
    folder_phase1 = results_root / "global_fit"
    folder_phase2 = results_root / "jackknife"
    folder_phase3 = results_root / "monte_carlo"
    folder_phase1.mkdir(parents=True, exist_ok=True)
    folder_phase2.mkdir(parents=True, exist_ok=True)
    folder_phase3.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    logger.info("Loading data files...")
    (
        intake_df,
        urine_volume_by_animal,
        feces_mass_by_animal,
        milk_yield_by_animal,
        feces_mass_default,
        body_weight_by_animal,
    ) = load_data(
        config, project_root=project_root
    )
    
    # Determine compounds from data if not explicitly set
    data_path = project_root / "data" / "raw" / "pfas_data_no_e1.csv"
    if config.compounds is None:
        logger.info(
            "Determining compound-isomer pairs: hay concentration not 0 and "
            "at least one of [Brain, Feces, Heart, Kidney, Liver, Lung, Milk, Muscle, Plasma, Spleen, Urine] above LOQ."
        )
        config.compounds = config.get_compounds_from_data(data_path=data_path)
        n_pairs = len(config.fittable_pairs) if config.fittable_pairs else 0
        logger.info(f"Found {len(config.compounds)} compounds ({n_pairs} compound-isomer pairs) meeting criteria")
    else:
        logger.info(f"Using explicitly set compounds: {', '.join(config.compounds)}")
    
    # Create data cache (only fittable pairs are returned by get_all_pairs if set)
    data_cache = DataCache(
        compounds=config.compounds,
        data_path=data_path,
        fittable_pairs=config.fittable_pairs,
    )
    
    # Create and return context
    return FittingContext(
        config=config,
        data_cache=data_cache,
        intake_df=intake_df,
        urine_volume_by_animal=urine_volume_by_animal,
        feces_mass_by_animal=feces_mass_by_animal,
        feces_mass_default=feces_mass_default,
        milk_yield_by_animal=milk_yield_by_animal,
        body_weight_by_animal=body_weight_by_animal,
        folder_phase1=folder_phase1,
        folder_phase2=folder_phase2,
        folder_phase3=folder_phase3
    )