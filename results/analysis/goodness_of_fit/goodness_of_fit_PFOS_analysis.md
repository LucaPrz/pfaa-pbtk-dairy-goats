# Goodness of fit: PFOS (Linear and Branched)

Analysis uses Monte Carlo median predictions matched to observations; metrics in log10 space unless noted.

---

## Overall (all compartments combined)

| Pair            | N   | R²    | GM fold error | Bias (log10) | Bias (fold) | 95% CI coverage |
|-----------------|-----|-------|----------------|--------------|-------------|------------------|
| **PFOS Linear** | 91  | 0.901 | 1.51           | −0.078       | 1.20        | **91.2%**        |
| **PFOS Branched** | 84 | 0.945 | 1.44           | −0.068       | 1.17        | 71.4%            |

- **R²**: Good (Linear 0.90, Branched 0.95); log-predicted vs log-observed are well aligned.
- **GM fold error**: ~1.5×; typical prediction error is about 50% in fold terms.
- **Bias**: Slight under-prediction (negative bias): predictions ~1.2× lower than observations on average.
- **CI coverage**: PFOS Linear 91% is close to nominal 95%; Branched 71% suggests intervals may be narrow for that isomer.

---

## By compartment

### PFOS Linear

| Compartment | N   | R²    | rRMSE | GM fold error | Bias (log10) | CI coverage |
|-------------|-----|-------|-------|----------------|--------------|-------------|
| **plasma**  | 24  | **0.976** | 0.054 | 1.23  | +0.008 (≈ unbiased) | **91.7%** |
| **milk**    | 27  | **0.815** | 0.232 | 1.73  | −0.119 (under)      | 77.8%      |
| **feces**   | 13  | **0.837** | 0.429 | 2.50  | −0.337 (under)      | 100%       |
| **urine**   | 6   | 0.165 | 0.377 | 1.52  | −0.076              | 100%       |
| brain, heart, kidney, liver, lung, muscle, spleen | 3 each | — | — | ~1.1–1.4 | small | 100% |

- **Plasma**: Best fit (R² 0.98, low rRMSE, near-zero bias) and good CI coverage.
- **Milk**: Good R² (0.82), moderate under-prediction (~1.3×) and fold error (1.7×).
- **Feces**: Good R² (0.84) but largest under-prediction (~2.2×) and fold error (2.5×); may reflect k_feces/k_ehc trade-off or timing.
- **Urine**: Low R² (0.17) with only 6 points; bias and fold error moderate.
- **Tissues** (N=3 each): Too few points for meaningful R²; fold errors ~1.1–1.3, coverage 100%.

### PFOS Branched

| Compartment | N   | R²    | rRMSE | GM fold error | Bias (log10) | CI coverage |
|-------------|-----|-------|-------|----------------|--------------|-------------|
| **plasma**  | 24  | **0.959** | 0.109 | 1.51  | −0.028 (slight under) | 62.5%  |
| **milk**    | 27  | **0.842** | 0.149 | 1.59  | −0.161 (under)        | 48.1%  |
| **feces**   | 9   | 0.01  | 0.371 | 1.26  | −0.012                | 100%   |
| **urine**   | 5   | 0.10  | 0.441 | 1.34  | −0.086                | 80%    |
| (tissues)   | 2–3 each | — | — | ~1.1–1.3 | small | 100%   |

- **Plasma**: Very good R² (0.96); CI coverage 62.5% suggests prediction intervals may be too tight.
- **Milk**: Good R² (0.84), under-prediction ~1.4×; coverage 48% again suggests narrow CIs.
- **Feces/urine**: Few points; R² low, fold errors moderate.

---

## Summary

- **PFOS Linear**: Strong overall fit (R² 0.90), good plasma and milk fit, feces under-predicted; 95% CI coverage overall (91%) is good.
- **PFOS Branched**: Slightly higher R² (0.95) overall but lower CI coverage (71%); plasma and milk fits are good, intervals may be conservative.
- **Main limitation**: Feces under-prediction for Linear (and sparse feces/urine for Branched); consistent with identifiable k_feces/k_ehc trade-off. Plasma and milk are well captured for both isomers.
