# Uncertainty decomposition and estimating "missing" uncertainty

## Three-way decomposition

Total variance of a log(observation) is decomposed into:

- **Parameter uncertainty**: from MC/jackknife (uncertainty in fitted kinetic parameters).
- **Animal variation**: between-animal variability (same parameters, different animals).
- **Observational**: residual variance in log-space (Sigma² from the Tobit fit).

Within the current model, these three add up to total variance. "Missing" means variance from sources that are **not** included in the model or MC.

## 1. Estimating missing via assay precision (residual split)

If you have an independent estimate of **assay precision in log-space** (e.g. from QC or repeat measurements), you can split the observational variance:

- **Var_obs = Sigma²** (from the fit).
- **Var_assay = assay_sigma²** (your assay precision).
- **Var_model = max(0, Sigma² − assay_sigma²)** = residual variance not explained by measurement error (model/structural error).

Then **Var_model** is the "missing" part of the residual that we attribute to model error rather than assay noise.

**Usage:**

```bash
python analysis/uncertainty_analysis.py --assay-sigma 0.2 --pattern "PFOS_Linear"
```

Output: `results/analysis/uncertainty_decomposition/missing_uncertainty_assay_split.csv` with columns:

- `Sigma`, `Assay_sigma`, `Var_obs_log`, `Var_assay`, `Var_model`
- `Frac_obs_as_assay`, `Frac_obs_as_model_error` (fraction of observational variance)
- `Var_total_mean`, `Frac_total_as_model_error` (fraction of **total** variance that is model error)

So **Frac_total_as_model_error** answers: “If assay error is X, how much of total uncertainty is missing (model error)?”

## 2. Estimating missing via sensitivity (other sources)

Other sources not in the three components:

- **Partition coefficients**: currently point estimates; uncertainty not in MC.
- **Input uncertainty**: feed intake, body weight, milk yield treated as fixed.
- **Parameter heterogeneity**: one parameter set per draw for all animals; no random effects per animal.

You can estimate how much variance each would add by **sensitivity runs**:

1. Choose a compartment and time (e.g. milk at day 56).
2. Run the model at **baseline** and at **perturbed** values, e.g.:
   - Partition coefficients × 1.1 (or sample from a CV).
   - Feed intake × 1.1.
   - Body weight × 1.05.
3. Compute elasticity: (ΔY / Y) / (ΔX / X).
4. If input X has CV = c, approximate **Var_X ≈ (c × elasticity)²** in log-space and add to total variance.

Then “missing” from that source ≈ Var_X; sum over sources for a rough total. This requires running the model (or optimization/solve) with perturbed inputs; it is not automated in the current script but can be done with small one-off scripts or by hand.

## Summary

| What you have | How to estimate missing |
|---------------|--------------------------|
| Assay precision (log-space sigma) | `--assay-sigma` → model error = Sigma² − assay_sigma² |
| Nothing else | Use residual split only; interpret Sigma² as upper bound on “unexplained” |
| Willing to run sensitivity | Perturb partitions, intake, body weight; compute elasticities and Var from assumed CVs |
