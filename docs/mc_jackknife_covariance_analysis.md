# Point 7: Jackknife covariance with 3 animals (PFOS Linear)

## What the log reported

- **Covariance matrix condition number:** 4.46e+06  
- **Max correlation:** k_elim vs k_a = **-1.000**  
- **Jackknife samples:** 3 (one per excluded animal: E2, E3, E4)  
- **Parameters:** 5 (k_ehc, k_elim, k_renal, k_a, k_feces)

## Why this happens

1. **Rank of the covariance**  
   The covariance is computed from the **centered** jackknife fits in log10-space:  
   `Σ = (n-1)/n * X'X` with `X` of shape `(3, 5)`.  
   So `X` has at most rank 3, and `X'X` has at most rank 3. With three samples, the centered matrix has rank 2 (one DoF lost to the mean), so the 5×5 covariance has **at most 2 positive eigenvalues**; the other three are zero (numerically ≈ 1e-18 or 1e-19).

2. **Regularization in the code**  
   The code adds `1e-8 * I` when `min_eigenval < 1e-8`. That makes the matrix strictly positive definite so `np.random.multivariate_normal` can run, but:
   - The three “zero” directions get variance ≈ 1e-8 in log10-space (negligible).
   - Condition number = (largest eigenvalue) / (smallest) ≈ 0.045 / 1e-8 ≈ **4.5e6**, matching the log.

3. **Perfect correlation k_elim vs k_a**  
   With only three points in 5D, the centered jackknife points lie in a 2D subspace. In that subspace, the scatter of (k_a, k_elim) is a line (three points → exact line), so the sample correlation is **±1**. The data show: when k_a goes up (e.g. E3), k_elim goes down; when k_a goes down (e.g. E4), k_elim goes up → **-1.000**. So this is a structural consequence of n=3 and p=5, not a bug.

4. **What MC sampling actually does**  
   - The full covariance has 2 “real” eigenvalues (~0.0033 and ~0.045) and 3 at ~1e-8.  
   - So the 10,000 MC draws lie almost in a **2D subspace** of the 5D parameter space.  
   - In the other three directions, variance is ~1e-8; those parameters barely move across samples.  
   - Prediction intervals are therefore driven by only **2 effective degrees of freedom** of parameter uncertainty, and the “full” covariance is **not** a meaningful 5D uncertainty estimate.

## Implications

- **Numerically:** Nothing fails; sampling runs and produces valid (finite) predictions.  
- **Statistically:** The full covariance is **degenerate**: we are estimating a 5×5 matrix from 3 points, so correlations (including k_elim vs k_a = -1) are structural, not informative.  
- **Interpretation:** MC bands reflect a narrow, constrained kind of uncertainty (effectively 2D). Using **diagonal covariance** (jackknife marginal variances only, no correlations) would at least give **marginal** uncertainty for each parameter and avoid implying false precision in the correlation structure.

## Recommendation (implemented in code)

The Monte Carlo step **always** uses **diagonal covariance**: variances from jackknife stds in log10 per parameter, no correlations. The full-covariance path was removed so that:

- Marginal uncertainty per parameter is used consistently.  
- We never sample from a rank-deficient matrix or report misleading condition numbers and ±1 correlations.  
- The code stays simple (no branching on `n_jackknife` vs `ndim`, no regularization, no LinAlgError fallback).

With 3 animals this matches the design; if you ever have more jackknife samples than parameters, diagonal is still a valid, conservative choice (you simply do not use the extra information in the off-diagonals).

---

## With always 3 animals: use diagonal

If you **always** have three animals, you will **always** have 3 jackknife samples and 5 (or more) parameters, so **n_jackknife ≤ ndim** every time. In that situation it is **sensible to rely on diagonal covariance** for Monte Carlo, and the code does that automatically. You are not giving up any real information: the “full” covariance is rank-deficient and its correlations are artifacts of having fewer samples than parameters.

---

## Full (rank-deficient) vs diagonal: exact differences and implications

| Aspect | Full covariance (rank-deficient + 1e-8 regularization) | Diagonal covariance |
|--------|----------------------------------------------------------|---------------------|
| **What is used** | 5×5 matrix from jackknife; 2 eigenvalues “real”, 3 set to ~1e-8. | 5 variances only (jackknife std² in log10 per parameter). No off-diagonals. |
| **Effective dimensions** | **2**: MC draws lie almost in a 2D subspace. Three parameter directions have variance ~1e-8 (no real spread). | **5**: Each parameter varies independently. All five directions have variance from jackknife. |
| **Correlations** | **Structural**: e.g. k_elim vs k_a = −1 from geometry of 3 points in 5D, not from data strength. | **None**: parameters are sampled independently. |
| **Marginal variances** | Same as diagonal (the diagonal of the full cov is the same jackknife variance per parameter). | Same: we use jackknife std² for each parameter. |
| **Joint behaviour** | Parameters move in a fixed 2D pattern (e.g. k_a up ⇒ k_elim down in a fixed ratio). | Parameters can move in any combination within their own ranges. |
| **Condition number** | Very large (~4e6); matrix is barely invertible. | 1 (identity scaling); numerically trivial. |
| **Interpretation** | Suggests a precise joint distribution and strong correlations, but both are artifacts of n=3. | Honest: we only have marginal uncertainty per parameter; no claim about joint structure. |

**Implications for predictions**

- **Full (degenerate):** Prediction intervals are driven by **only 2 effective degrees of freedom**. In 3 directions parameters hardly change across MC samples, so uncertainty is understated in those directions and the shape of the uncertainty region is determined by the arbitrary 2D subspace.
- **Diagonal:** Prediction intervals reflect **marginal uncertainty for all 5 parameters**. Intervals can be wider or different in shape (e.g. more “box-like” in parameter space), but they do not rely on spurious correlations. With only 3 animals we cannot estimate a meaningful 5×5 correlation matrix anyway, so diagonal is the defensible choice.

**Bottom line:** With 3 animals, the full jackknife covariance does **not** add valid information over the diagonal; it only adds numerical issues and misleading ±1 correlations. Using diagonal is the right choice and is what the code does when `n_jackknife ≤ ndim`.
