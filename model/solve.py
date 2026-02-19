"""
Analytical solver for the PBTK model
"""

from collections import namedtuple
from typing import Dict, Tuple, Union, Any

import numpy as np
from scipy.linalg import expm

from .structure import PBTKStructure


SimulationResult = namedtuple(
    "SimulationResult",
    ["mass_matrix", "time_array", "feces_array", "urine_array", "milk_array", "elim_array"],
)


class PBTKSolver(PBTKStructure):
    """Numerical solver and simulation utilities for the clean PBTK model."""

    def solve_ode_with_cache(
        self, A: np.ndarray, exp_Tt: np.ndarray, T: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        if np.all(u == 0):
            return exp_Tt @ A
        try:
            steady = -np.linalg.solve(T, u)
            return steady + exp_Tt @ (A - steady)
        except np.linalg.LinAlgError:
            print(
                "[WARN] ODE solve failed (singular matrix), "
                "falling back to state propagation without input"
            )
            return exp_Tt @ A

    def solve_integral_with_cache(
        self, A: np.ndarray, exp_Tt: np.ndarray, T: np.ndarray, u: np.ndarray, dt: float
    ) -> np.ndarray:
        try:
            cond_num = np.linalg.cond(T)
            if cond_num > self.ILL_CONDITIONED_THRESHOLD:
                print(
                    f"[WARN] Ill-conditioned matrix (cond={cond_num:.2e}), "
                    "returning zero integral (no excretion accumulation)"
                )
                return np.zeros_like(A)

            try:
                T_inv = np.linalg.inv(T)

                if np.all(u == 0):
                    integral = (exp_Tt - self._eye_matrix) @ T_inv @ A
                else:
                    steady = -T_inv @ u
                    integral = steady * dt + (exp_Tt - self._eye_matrix) @ T_inv @ (
                        A - steady
                    )

            except np.linalg.LinAlgError:
                print(
                    "[WARN] Matrix inversion failed, using trapezoidal fallback for integral"
                )
                if np.all(u == 0):
                    A_new = exp_Tt @ A
                    integral = (A + A_new) * dt / 2
                else:
                    try:
                        steady = -np.linalg.solve(T, u)
                        A_new = exp_Tt @ A + (exp_Tt - self._eye_matrix) @ steady
                        integral = (A + A_new) * dt / 2
                    except np.linalg.LinAlgError:
                        print(
                            "[WARN] Trapezoidal fallback also failed, "
                            "returning zero integral"
                        )
                        return np.zeros_like(A)

            integral = np.maximum(integral, 0)

            clipped_high = np.any(integral > self.MAX_INTEGRAL_VALUE)
            if clipped_high:
                n_clipped = int(np.sum(integral > self.MAX_INTEGRAL_VALUE))
                max_val = float(np.max(integral))
                print(
                    f"[WARN] Clipping {n_clipped} integral value(s) from "
                    f"max {max_val:.2e} to {self.MAX_INTEGRAL_VALUE:.0e}"
                )

            integral = np.clip(integral, 0, self.MAX_INTEGRAL_VALUE)

            if np.any(~np.isfinite(integral)):
                print(
                    "[WARN] Non-finite values in integral, "
                    "returning zero (no excretion accumulation)"
                )
                return np.zeros_like(A)

            return integral

        except Exception as e:
            print(
                f"[WARN] solve_integral_with_cache failed completely: {e}, "
                "returning zero integral"
            )
            return np.zeros_like(A)

    # ------------------------------------------------------------------
    # Cached transition matrix and matrix exponential
    # ------------------------------------------------------------------
    def _get_cached_transition_matrix(self, t: float) -> np.ndarray:
        t_key = int(t)
        if t_key not in self._transition_cache:
            self._transition_cache[t_key] = self.transition_matrix(t)
        return self._transition_cache[t_key]

    def _get_cached_matrix_exponential(self, T: np.ndarray, dt: float) -> np.ndarray:
        dt_key = int(dt)

        if self.physiology_provider is None:
            if dt_key not in self._exp_cache:
                self._exp_cache[dt_key] = expm(T * dt)
            return self._exp_cache[dt_key]

        return expm(T * dt)

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    def _simulate_step(
        self, A: np.ndarray, t0: float, t1: float, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt = t1 - t0

        try:
            T = self._get_cached_transition_matrix(t0)
            exp_Tt = self._get_cached_matrix_exponential(T, dt)
            u = self.input_vector(t0)

            A_new = self.solve_ode_with_cache(A, exp_Tt, T, u)
            integral = self.solve_integral_with_cache(A, exp_Tt, T, u, dt)

            return A_new, integral

        except Exception as e:
            if verbose:
                print(
                    f"[WARN] Numerical error at t = {t0:.2f}–{t1:.2f} h: {str(e)}"
                )
            A_new = A.copy()
            integral = np.zeros_like(A)
            return A_new, integral

    # ------------------------------------------------------------------
    # Public simulation API
    # ------------------------------------------------------------------
    def simulate_over_time(
        self, A0: np.ndarray, t_array: np.ndarray, verbose: bool = False
    ) -> SimulationResult:
        A = np.zeros((self.compartment_number, 1))
        A[: A0.shape[0], 0] = A0.flatten()
        A_matrix = [A.flatten()]

        milk_total = urine_total = feces_total = elim_total = 0.0
        milk_array = [milk_total]
        urine_array = [urine_total]
        feces_array = [feces_total]
        elim_array = [elim_total]

        AE = self.params["absorption_excretion"]

        for i in range(1, len(t_array)):
            t0 = float(t_array[i - 1])
            t1 = float(t_array[i])

            A_new, integral = self._simulate_step(A, t0, t1, verbose)

            milk_yield_now = self._get_milk_yield(t0)
            if milk_yield_now > 0:
                PC = self.params["partition_coefficients"]
                P_milk = PC.get("P_milk", 1.0)
                if P_milk > 0:
                    pi_plasma = self.projection_vector("plasma")
                    plasma_integral = float(pi_plasma @ integral)
                    phys = self._get_physiology(t0)
                    V_plasma = phys.get("V_plasma", 0.0)
                    if V_plasma > 0:
                        k_milk_eff = (P_milk * milk_yield_now) / V_plasma
                        milk_total += k_milk_eff * plasma_integral

            k_urine = AE.get("k_urine", 0.0)
            if k_urine > 0:
                pi_plasma = self.projection_vector("plasma")
                urine_total += k_urine * float(pi_plasma @ integral)

            pi_intestine = self.projection_vector("intestine")
            feces_total += AE["k_feces"] * float(pi_intestine @ integral)

            k_elim = AE.get("k_elim", 0.0)
            if k_elim > 0:
                pi_plasma = self.projection_vector("plasma")
                elim_total += k_elim * float(pi_plasma @ integral)

            A = A_new
            A_matrix.append(A.flatten())
            milk_array.append(milk_total)
            urine_array.append(urine_total)
            feces_array.append(feces_total)
            elim_array.append(elim_total)

        return SimulationResult(
            mass_matrix=np.array(A_matrix),
            time_array=np.array(t_array),
            feces_array=np.array(feces_array),
            urine_array=np.array(urine_array),
            milk_array=np.array(milk_array),
            elim_array=np.array(elim_array),
        )

    def update_parameters(self, new_params: Dict[str, Any]) -> None:
        """Update model parameters in place."""
        self.params = new_params

