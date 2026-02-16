"""
ODE system definition for the PBTK model
"""

from typing import Dict, Optional, Callable, Tuple, Union, Any

import numpy as np


class PBTKStructure:
    """
    Core structural definition of the PBTK model.

    Responsibilities:
      - define compartments and their indices
      - manage physiological inputs (flows, volumes, partition coefficients)
      - build the transition matrix T(t)
    """

    compartment_idx = {
        "stomach": 0,
        "intestine": 1,
        "spleen": 2,
        "liver": 3,
        "plasma": 4,
        "kidney": 5,
        "muscle": 6,
        "heart": 7,
        "brain": 8,
        "lung": 9,
        "rest": 10,
    }

    # Numerical constants used by solvers/diagnostics
    MAX_INTEGRAL_VALUE = 1e6
    ILL_CONDITIONED_THRESHOLD = 1e12
    MODERATELY_ILL_CONDITIONED_THRESHOLD = 1e6
    LARGE_DYNAMIC_RANGE_THRESHOLD = 1e6
    NEAR_ZERO_THRESHOLD = 1e-15
    MASS_CONSERVATION_TOLERANCE = 1e-10

    def __init__(
        self,
        params: Dict[str, Any],
        intake_function: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        physiology_provider: Optional[Callable[[float], Dict[str, float]]] = None,
        milk_yield_function: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.params = params
        self.intake_function = intake_function
        self.physiology_provider = physiology_provider
        self.milk_yield_function = milk_yield_function
        self.compartment_number = len(self.compartment_idx)

        # Caches filled by solver logic (kept here so subclasses can use them)
        self._transition_cache: Dict[int, np.ndarray] = {}
        self._exp_cache: Dict[Union[int, Tuple[int, int]], np.ndarray] = {}
        self._eye_matrix = np.eye(self.compartment_number)
        self._projection_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Helper accessors
    # ------------------------------------------------------------------
    def projection_vector(self, compartment: str) -> np.ndarray:
        """Row-vector π_i selecting compartment i from the state vector."""
        if compartment not in self._projection_cache:
            pi = np.zeros(self.compartment_number)
            if compartment in self.compartment_idx:
                idx = self.compartment_idx[compartment]
                pi[idx] = 1.0
            self._projection_cache[compartment] = pi
        return self._projection_cache[compartment]

    def _get_milk_yield(self, t: float) -> float:
        """Milk yield (kg/day) at time t."""
        if callable(self.milk_yield_function):
            try:
                return float(self.milk_yield_function(t))
            except Exception:
                pass

        if callable(self.physiology_provider):
            try:
                q_provider = self.physiology_provider(t)
                return float(q_provider.get("milk_yield", 0.0))
            except Exception:
                return 0.0
        return 0.0

    def _get_physiology(self, t: float) -> Dict[str, float]:
        """Return physiological flows/volumes at time t."""
        if callable(self.physiology_provider):
            return self.physiology_provider(t)
        return self.params["physiological"]

    def input_vector(self, t: float) -> np.ndarray:
        """External input vector u(t) with intake into the stomach."""
        u = np.zeros((self.compartment_number, 1))
        u[self.compartment_idx["stomach"], 0] = self.intake_function(t)
        return u

    # ------------------------------------------------------------------
    # Transition-matrix builders
    # ------------------------------------------------------------------
    def _build_gastrointestinal_terms(
        self, T: np.ndarray, ci: Dict[str, int], AE: Dict[str, float]
    ) -> None:
        # Stomach: emptying to intestine
        T[ci["stomach"], ci["stomach"]] = -AE["k_sto"]
        T[ci["intestine"], ci["stomach"]] = AE["k_sto"]

        # Intestine: absorption to liver, fecal excretion, enterohepatic recirculation
        T[ci["intestine"], ci["intestine"]] = -(AE["k_a"] + AE["k_feces"])
        T[ci["liver"], ci["intestine"]] += AE["k_a"]
        T[ci["intestine"], ci["liver"]] += AE["k_ehc"]

    def _build_spleen_terms(
        self,
        T: np.ndarray,
        ci: Dict[str, int],
        Q: Dict[str, float],
        V: Dict[str, float],
        PC: Dict[str, float],
    ) -> None:
        T[ci["spleen"], ci["plasma"]] += Q["Q_spleen"] / V["V_plasma"]
        T[ci["liver"], ci["spleen"]] += Q["Q_spleen"] / (PC["Spleen"] * V["V_spleen"])
        T[ci["spleen"], ci["spleen"]] += -Q["Q_spleen"] / (
            PC["Spleen"] * V["V_spleen"]
        )

    def _build_liver_terms(
        self,
        T: np.ndarray,
        ci: Dict[str, int],
        Q: Dict[str, float],
        V: Dict[str, float],
        PC: Dict[str, float],
        AE: Dict[str, float],
    ) -> None:
        flow_liv = Q["Q_hepatic"] + Q["Q_intestine"]
        T[ci["liver"], ci["liver"]] += -flow_liv / (PC["Liver"] * V["V_liver"]) - AE[
            "k_ehc"
        ]
        T[ci["plasma"], ci["liver"]] += flow_liv / (PC["Liver"] * V["V_liver"])
        T[ci["liver"], ci["plasma"]] += flow_liv / V["V_plasma"]

    def _build_plasma_terms(
        self,
        T: np.ndarray,
        ci: Dict[str, int],
        Q: Dict[str, float],
        V: Dict[str, float],
        AE: Dict[str, float],
        PC: Dict[str, float],
        milk_yield: float,
    ) -> None:
        total_flow = (
            Q["Q_hepatic"]
            + Q["Q_intestine"]
            + Q["Q_spleen"]
            + Q["Q_kidney"]
            + Q["Q_muscle"]
            + Q["Q_heart"]
            + Q["Q_brain"]
            + Q["Q_rest"]
            + Q["Q_lung"]
        )
        T[ci["plasma"], ci["plasma"]] += -total_flow / V["V_plasma"] - AE["k_elim"]

        if milk_yield > 0:
            V_plasma = V.get("V_plasma", 0.0)
            P_milk = PC.get("P_milk", 1.0)
            if V_plasma > 0 and P_milk > 0:
                k_milk_eff = (P_milk * milk_yield) / V_plasma
                T[ci["plasma"], ci["plasma"]] -= k_milk_eff

    def _build_kidney_terms(
        self,
        T: np.ndarray,
        ci: Dict[str, int],
        Q: Dict[str, float],
        V: Dict[str, float],
        PC: Dict[str, float],
        AE: Dict[str, float],
    ) -> None:
        T[ci["kidney"], ci["kidney"]] -= Q["Q_kidney"] / (
            PC["Kidney"] * V["V_kidney"]
        ) + AE["k_renal"]
        T[ci["kidney"], ci["plasma"]] += Q["Q_kidney"] / V["V_plasma"]
        T[ci["plasma"], ci["kidney"]] += Q["Q_kidney"] / (
            PC["Kidney"] * V["V_kidney"]
        )

    def _build_generic_organ_terms(
        self,
        T: np.ndarray,
        ci: Dict[str, int],
        Q: Dict[str, float],
        V: Dict[str, float],
        PC: Dict[str, float],
    ) -> None:
        for name, flow, P, vol in [
            ("muscle", Q["Q_muscle"], PC["Muscle"], V["V_muscle"]),
            ("heart", Q["Q_heart"], PC["Heart"], V["V_heart"]),
            ("brain", Q["Q_brain"], PC["Brain"], V["V_brain"]),
            ("rest", Q["Q_rest"], PC["Rest"], V["V_rest"]),
            ("lung", Q["Q_lung"], PC["Lung"], V["V_lung"]),
        ]:
            i = ci[name]
            T[i, i] += -flow / (P * vol)
            T[i, ci["plasma"]] += flow / V["V_plasma"]
            T[ci["plasma"], i] += flow / (P * vol)

    # ------------------------------------------------------------------
    # Public: transition matrix
    # ------------------------------------------------------------------
    def transition_matrix(self, t: float) -> np.ndarray:
        """Return the transition matrix T(t) for time t."""
        Q = self._get_physiology(t)
        PC = self.params["partition_coefficients"]
        AE = self.params["absorption_excretion"]
        ci = self.compartment_idx
        V = Q

        T = np.zeros((self.compartment_number, self.compartment_number))

        self._build_gastrointestinal_terms(T, ci, AE)
        self._build_spleen_terms(T, ci, Q, V, PC)
        self._build_liver_terms(T, ci, Q, V, PC, AE)
        milk_yield = self._get_milk_yield(t)
        self._build_plasma_terms(T, ci, Q, V, AE, PC, milk_yield)
        self._build_kidney_terms(T, ci, Q, V, PC, AE)
        self._build_generic_organ_terms(T, ci, Q, V, PC)

        return T

