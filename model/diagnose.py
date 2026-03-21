"""
Diagnostic helpers
"""

from typing import Tuple, List, Optional

import numpy as np

from .solve import PBTKSolver, SimulationResult


class PBTKModel(PBTKSolver):
    """
    Provides:
      - structural definition (from `PBTKStructure`)
      - numerical solver (from `PBTKSolver`)
      - diagnostics (defined here)
    """

    # ------------------------------------------------------------------
    # Mass-balance diagnostics
    # ------------------------------------------------------------------
    def check_mass_balance(self, simulation_result: SimulationResult) -> Tuple[float, float]:
        A_matrix = simulation_result.mass_matrix
        time_array = simulation_result.time_array
        feces_array = simulation_result.feces_array
        urine_array = simulation_result.urine_array
        milk_array = simulation_result.milk_array
        elim_array = simulation_result.elim_array

        dt = np.diff(time_array)
        intake_array = np.array([self.intake_function(ti) for ti in time_array[:-1]])
        total_intake = np.sum(intake_array * dt)

        body_burden = A_matrix[-1].sum()
        total_excreted = (
            feces_array[-1] + urine_array[-1] + milk_array[-1] + elim_array[-1]
        )

        balance_error = abs((body_burden + total_excreted) - total_intake)
        relative_error = balance_error / total_intake if total_intake > 0 else 0.0

        print("\nFinal compartment loads:")
        for name, idx in sorted(self.compartment_idx.items(), key=lambda x: x[1]):
            print(f"  - {name.capitalize():<10}: {A_matrix[-1, idx]:.3f} µg")

        print(f"\nTotal intake:        {float(total_intake):.3f} µg")
        print(f"Final body load:     {float(body_burden):.3f} µg")
        print(f"Total excreted:      {float(total_excreted):.3f} µg")
        print(f"  - Feces: {float(feces_array[-1]):.3f} µg")
        print(f"  - Urine: {float(urine_array[-1]):.3f} µg")
        print(f"  - Milk:  {float(milk_array[-1]):.3f} µg")
        print(f"  - Elim:  {float(elim_array[-1]):.3f} µg")
        print(f"Mass balance error:  {float(balance_error):.6f} µg")
        print(f"Relative error:      {float(relative_error):.6%}")

        return balance_error, relative_error

    def check_mass_balance_from_simulation(
        self, A0: np.ndarray, t_array: np.ndarray
    ) -> Tuple[float, float]:
        sim_result = self.simulate_over_time(A0, t_array)
        return self.check_mass_balance(sim_result)

    # ------------------------------------------------------------------
    # Matrix diagnostics
    # ------------------------------------------------------------------
    def check_transition_matrix_conservation(self, t: float = 0.0) -> bool:
        T = self.transition_matrix(t)
        AE = self.params["absorption_excretion"]
        ci = self.compartment_idx

        print("\n=== TRANSITION MATRIX MASS CONSERVATION CHECK ===")

        column_sums = np.sum(T, axis=0)
        unexpected_violations = []

        for name, idx in sorted(ci.items(), key=lambda x: x[1]):
            sum_val = column_sums[idx]

            expected_violation = 0.0
            if name == "intestine":
                expected_violation = -AE["k_feces"]
            elif name == "plasma":
                expected_violation = -(AE.get("k_elim", 0.0) + AE.get("k_urine", 0.0))

            is_expected = abs(sum_val - expected_violation) < self.MASS_CONSERVATION_TOLERANCE

            if is_expected:
                status = "✓" if abs(sum_val) < self.MASS_CONSERVATION_TOLERANCE else "✓ (excretion)"
            else:
                status = "✗"
                if abs(sum_val - expected_violation) > self.MASS_CONSERVATION_TOLERANCE:
                    unexpected_violations.append((name, sum_val, expected_violation))

            print(f"  {status} {name.capitalize():<10}: {sum_val:.2e}")
            if not is_expected and abs(sum_val) > self.MASS_CONSERVATION_TOLERANCE:
                print(f"           Expected: {expected_violation:.2e}")

        if unexpected_violations:
            print("\n[WARNING] Unexpected mass conservation violations detected!")
            for name, actual, expected in unexpected_violations:
                print(f"  - {name}: actual={actual:.2e}, expected={expected:.2e}")
        else:
            print("\n✓ Mass conservation verified (all violations are expected excretion terms)")

        return len(unexpected_violations) == 0

    def diagnose_matrix_issues(self, t: float = 0.0) -> bool:
        T = self.transition_matrix(t)

        issues = []

        cond_num = np.linalg.cond(T)
        if cond_num > self.ILL_CONDITIONED_THRESHOLD:
            issues.append(f"Very ill-conditioned matrix (cond={cond_num:.2e})")
        elif cond_num > self.MODERATELY_ILL_CONDITIONED_THRESHOLD:
            issues.append(f"Moderately ill-conditioned matrix (cond={cond_num:.2e})")

        max_val = float(np.max(np.abs(T)))
        min_nonzero = float(np.min(np.abs(T[T != 0])))
        ratio = max_val / min_nonzero if min_nonzero > 0 else np.inf

        if ratio > self.LARGE_DYNAMIC_RANGE_THRESHOLD:
            issues.append(f"Large dynamic range in matrix ({ratio:.2e})")

        if np.any(np.abs(np.diag(T)) < self.NEAR_ZERO_THRESHOLD):
            issues.append("Near-zero diagonal elements detected")

        if issues:
            print("[WARN] Matrix issues detected:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("Matrix looks numerically stable")

        return len(issues) == 0

    # ------------------------------------------------------------------
    # Eigenmode half-lives and steady-state transfer
    # ------------------------------------------------------------------
    def eigenmode_half_lives(self, t: float = 0.0, tol: float = 1e-8) -> np.ndarray:
        """
        Return eigenmode half-lives (days) for the transition matrix at time t.

        Each eigenmode i with real, negative eigenvalue λ_i contributes a decay
        term exp(λ_i t). The associated half-life is:

            t_{1/2,i} = ln(2) / |Re(λ_i)|

        This function returns all such half-lives, sorted ascending
        (fastest to slowest mode).
        """
        T = self.transition_matrix(t)
        eigvals = np.linalg.eigvals(T)

        # Select modes with sufficiently small imaginary part and negative real part
        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)
        mask = (np.abs(imag_parts) <= tol) & (real_parts < 0.0)
        rates = -real_parts[mask]  # positive decay rates

        if rates.size == 0:
            return np.array([], dtype=float)

        half_lives = np.log(2.0) / rates  # days, since T is in 1/day
        return np.sort(half_lives)

    def systemic_half_life(
        self,
        t: float = 0.0,
        tol: float = 1e-8,
    ) -> Optional[float]:
        """
        Return a single "systemic" half-life (days) based on the slowest
        positive eigenmode of the transition matrix at time t.

        If no suitable eigenmodes are found, returns None.
        """
        half_lives = self.eigenmode_half_lives(t=t, tol=tol)
        if half_lives.size == 0:
            return None
        # Slowest mode = largest half-life
        return float(half_lives[-1])

    def steady_state_milk_transfer_rate(self, t: float = 0.0) -> Optional[float]:
        """
        Compute a model-based steady-state milk transfer rate TR_milk.

        Assumes:
          - Constant, unit intake into the stomach compartment (1 arbitrary
            mass unit per day).
          - Linear, time-invariant transition matrix at time t.

        The steady-state amount vector A_ss satisfies:

            T A_ss + u = 0  →  A_ss = -T^{-1} u

        From A_ss we derive:
          - Steady-state plasma concentration C_plasma_ss,
          - Steady-state milk concentration via P_milk,
          - Milk mass excreted per day = C_milk_ss * milk_yield(t),
          - TR_milk = (milk mass/day) / (intake mass/day).

        Returns:
          TR_milk as a dimensionless fraction of intake, or None if the
          steady-state system cannot be solved.
        """
        # Build transition matrix and a unit intake vector into the stomach
        T = self.transition_matrix(t)
        u = np.zeros((self.compartment_number, 1))
        if "stomach" not in self.compartment_idx:
            return None
        u[self.compartment_idx["stomach"], 0] = 1.0  # 1 unit/day into stomach

        try:
            A_ss = -np.linalg.solve(T, u)
        except np.linalg.LinAlgError:
            return None

        # Steady-state plasma amount and concentration
        pi_plasma = self.projection_vector("plasma")
        plasma_amount = float(pi_plasma @ A_ss)
        phys = self._get_physiology(t)
        V_plasma = float(phys.get("V_plasma", 0.0))
        if V_plasma <= 0.0:
            return None
        C_plasma_ss = plasma_amount / V_plasma

        # Milk concentration via partition coefficient
        PC = self.params.get("partition_coefficients", {})
        P_milk = float(PC.get("P_milk", 1.0))
        C_milk_ss = P_milk * C_plasma_ss

        # Milk yield (kg/day) at time t
        milk_yield = float(self._get_milk_yield(t))

        # Steady-state milk mass per day divided by intake mass per day (1.0)
        TR_milk = C_milk_ss * milk_yield
        return float(TR_milk)

