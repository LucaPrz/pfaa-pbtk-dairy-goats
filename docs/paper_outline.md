# Paper sentence outline

## Introduction

1. **PFAS and PFAAs as a global concern**
   1. Per- and polyfluoroalkyl substances (PFAS) are a large group of synthetic chemicals characterized by extreme environmental persistence and global distribution; perfluoroalkyl acids (PFAAs) are a major subclass of particular concern due to persistence, bioaccumulation potential, and ubiquity in environmental and biological matrices.

2. **Toxicological relevance of PFAAs**
   1. Evidence links selected PFAAs to adverse developmental, immunological, and metabolic effects; structural similarity suggests shared modes of action and supports cumulative risk assessment rather than chemical-by-chemical evaluation.

3. **Food-chain transfer and human exposure**
   1. Diet is a major exposure route; animal-derived foods contribute substantially to human PFAA intake via transfer from contaminated feed. The European Food Safety Authority has set tolerable weekly intakes and maximum concentrations in food and feed to protect consumers.

4. **Relevance of goat milk**
   1. In many arid and semi-arid regions, goat milk is a key source of protein and essential nutrients for local populations.
   2. Goat milk is also frequently used as a substitute for human breast milk, especially for infants and children who cannot tolerate cow’s milk.
   3. Because infants have high milk consumption relative to body weight and may be particularly sensitive to chemical exposures, PFAA transfer into goat milk could represent a critical exposure pathway for this vulnerable population group.

5. **Role of physiologically based toxicokinetic (PBTK) models**
   1. Physiologically based toxicokinetic models provide mechanistic tools that quantitatively link external exposure to internal concentrations in specific tissues and fluids.
   2. By explicitly representing key physiological and chemical-specific processes, PBTK models support extrapolation across exposure scenarios, dose levels, and populations, including different life stages.
   3. For regulatory risk assessment, PBTK models offer advantages such as improved biological realism, explicit treatment of uncertainty and variability, and the ability to test “what-if” scenarios that are difficult or unethical to study experimentally.

6. **Current state of PFAA PBTK modeling in food-producing animals**
   1. Existing PBTK models for PFAAs have been developed for several food-producing species, including cattle, swine, and laying hens.
   2. Direct extrapolation of these species-specific models to goats is limited by differences in physiology, such as body composition, organ sizes, and metabolic rates.
   3. In addition, goats differ from other species in milk production characteristics and lactation dynamics, which are expected to influence the kinetics of PFAA transfer into milk.

7. **Knowledge gap, novelty, and objective of the present study**
   1. No curated PBTK model exists for PFAAs in dairy goats, which hampers feed safety and milk contamination risk assessment. This study is the **first PBTK model for PFAAs in dairy goats**; the **first to link systemic half-life to milk transfer rate across multiple PFAAs** in a food-producing species; and the **first to apply a goat PBTK model for regulatory feed safety** (e.g. maximum allowable feed concentrations and scenario-based risk assessment).
   2. The objective is to develop, parameterize, and evaluate a physiologically based toxicokinetic model for selected PFAAs in dairy goats, with emphasis on milk transfer and regulatory applicability.

## Methods

1. **In vivo feeding study**
   1. The in vivo feeding study was designed with a control and exposure groups, defined housing conditions, and a study duration sufficient to characterize both accumulation and clearance of PFAAs.
   2. During the exposure phase, goats received PFAA-contaminated hay, and cumulative PFAA intake was quantified to provide well-characterized external dose metrics.
   3. A subsequent depuration phase without contaminated feed was conducted to characterize elimination kinetics and to distinguish between distribution and clearance processes.
   4. A structured sampling strategy was implemented, including repeated collection of milk, blood, urine, feces, and selected tissues, to provide rich time-course data for model calibration and evaluation.
   5. Concentrations of individual PFAAs in all biological matrices were quantified using validated LC–MS/MS methods with appropriate quality assurance and quality control procedures.
   6. Data from one animal (E1) were excluded from the analysis based on predefined criteria to avoid bias in model fitting and evaluation.
   7. The primary purpose of this experimental study was to generate high-quality kinetic data that could be used to calibrate and independently evaluate the PBTK model.

2. **PBTK model structure**
   1. The PBTK model comprises eleven physiological compartments and associated excretion pools, representing key tissues and elimination pathways relevant for PFAA kinetics in dairy goats.
   2. Gastrointestinal absorption processes and fecal losses are explicitly described to link feedborne exposure to systemic circulation.
   3. Distribution within the body is represented by a plasma-centered structure that captures exchange with tissue compartments via blood flow and partitioning.
   4. Liver–intestine enterohepatic circulation is included to account for biliary excretion and potential reabsorption of PFAAs in the gut.
   5. Elimination via milk, urine, and feces is modeled explicitly to quantify contributions of each pathway to overall clearance and to predict concentrations in edible products.
   6. An additional unspecified elimination pathway is incorporated to capture residual clearance processes that are not assigned to identified physiological routes, and its inclusion is justified based on model performance and biological plausibility.
   7. The purpose of this section is to provide a conceptual description of the model structure prior to introducing the mathematical formulation.

3. **Model solving**
   1. The model is formulated as a system of linear ordinary differential equations that describe the time evolution of PFAA amounts in each compartment.
   2. An analytical matrix-based solution is derived to solve the ODE system efficiently and accurately for arbitrary exposure scenarios.
   3. A steady-state formulation is provided to derive closed-form expressions for long-term equilibrium concentrations under constant exposure conditions.
   4. Exact integration methods are used to compute cumulative excretion via milk, urine, and feces over specified time periods.
   5. Conditions for numerical stability and accuracy are derived and implemented to ensure reliable simulations across a wide range of parameter values and exposure patterns.

4. **Optimization, uncertainty analysis, and prediction**
   1. The loss function and error model are defined on the logarithmic scale of concentrations to appropriately weight relative errors across several orders of magnitude.
   2. Observations below the limit of quantification are treated using a limit-of-quantification-aware approach that accounts for censoring rather than discarding data or substituting arbitrary values.
   3. Weights are assigned to different compartments and matrices to balance their contributions to the overall loss function in accordance with data density and regulatory relevance.
   4. An initial round of parameter fitting based on root-mean-square error is used to obtain reasonable starting values for subsequent likelihood-based optimization.
   5. A Tobit likelihood formulation is then applied to formally incorporate censored observations into the parameter estimation process.
   6. Global optimization is performed in log-parameter space with biologically informed bounds to ensure positivity and to reflect plausible ranges for kinetic parameters; a compound- and isomer-specific fitting strategy is used to capture differences in kinetics among PFAAs while exploiting shared structure where justified. Differential evolution is used as the global optimizer. *[Algorithm settings, bounds, and reproducibility details (e.g. random seeds) are given in Supplementary Material.]*
   7. Hessian-based covariance estimation is used to approximate the joint uncertainty in fitted parameters.
   8. Monte Carlo uncertainty propagation is performed by drawing multivariate log-normal samples from the covariance structure to generate ensembles of parameter sets and corresponding prediction intervals for model outputs.
   9. Variance decomposition is carried out to separate contributions from parameter uncertainty, biological variability, and measurement error to the total predictive uncertainty; nested prediction intervals are constructed to communicate these components and to support transparent regulatory interpretation.

5. **Parameter identifiability**
   1. A Fisher Information Matrix is constructed based on model sensitivities and the data structure to quantify the amount of information available for estimating each parameter.
   2. Local sensitivities of model outputs with respect to individual parameters are calculated to identify which parameters strongly influence observable quantities.
   3. Eigenvalue analysis of the Fisher Information Matrix is performed to diagnose overall identifiability and to detect parameter combinations that are poorly informed by the data.
   4. The condition number of the Fisher Information Matrix is used as a summary metric of identifiability and numerical stability.
   5. Parameter correlations are evaluated to identify strongly correlated parameters that may be difficult to estimate independently.
   6. Identifiability results are used in the Results section to qualify parameter estimates and to indicate which conclusions are robust to identifiability concerns.

6. **Physiology sub-model for extrapolation**
   1. A physiology sub-model is defined so that key physiological inputs to the PBTK model vary with time (days in milk) and with production scenario (breed × parity). It is coupled to the PBTK model to enable extrapolation beyond the experimental conditions and supports scenario analysis for feed safety and regulatory use.
   2. **Time-varying quantities:** Body weight (kg), milk yield (kg/day), and dry matter intake (kg/day) are functions of day in milk. Organ volumes (plasma, liver, stomach, intestine, spleen, kidney, muscle, heart, brain, lung, rest) and plasma flows are derived from body weight at each time point using species-specific allometric fractions (volume and flow factors from goat literature); cardiac output is 8.17×(1−hematocrit) L/(h·kg) with hematocrit 0.27, so volumes and flows scale with body weight over the lactation.
   3. **Body weight:** BW(d) = BW_min + (BW0 − BW_min)·exp(−a·d) + exp(b·(d − day0)), with parameters (BW_min, BW0, a, b, day0) depending on breed and parity. Ranges in the implemented registry: BW_min 48.8–70.3 kg, BW0 52.1–78.7 kg (Alpine/Saanen × primiparous/multiparous).
   4. **Milk yield:** Lactation curve (kg/day) = potlac × max(shape(d), 0), with bi-exponential shape(d) = A1·exp(−k1·d) − A2·exp(−k2·d). Coefficients (A1, k1, A2, k2) and potential lactation (potlac, kg) are specified per (breed, parity); potlac is 880 (primiparous) or 950 (multiparous) in the current registry.
   5. **Dry matter intake:** DMI (kg/day) is given by a linear empirical model in body weight and milk yield (and optional concentrate), using species-level coefficients for goats (e.g. INRA-based). DMI is used for intake scaling in exposure scenarios; organ volumes and flows used by the PBTK model depend on BW and thus indirectly on the same curves.
   6. **Scenarios:** The sub-model is parameterized for Alpine and Saanen, primiparous and multiparous, so that extrapolation and applications (e.g. breed×parity scenario analysis, maximum feed estimation) can vary physiology over these four combinations. When fitting or when animal-specific data are used, body weight and milk yield can alternatively be supplied as measured time series instead of curves.

## Results

1. **Model performance**
   1. Among 30 PFAAs from the transfer study, 15 compound–isomer combinations met the acceptability criteria: R² > 0.7, geometric mean fold error (GMFE) < 3, and |bias_log10| < 0.3. The passing set includes PFDA, PFDS, PFDoDA, PFDoDS (Branched), PFNS, PFOS, PFTrDA (Linear), PFTrDS (Branched), PFUnDA (Linear), PFUnDS, and others meeting these thresholds; excluded compounds (e.g. PFBA, PFHxA, PFNA, PFOA Linear, PFPeA, PFTeDA, PFTrDS Linear) are listed with the primary reason for exclusion (high GMFE, low R², or strong bias from sparse/censored data or poor identifiability).
   2. Goodness-of-fit is reported per compound (R², GMFE, bias_log10, bias_fold, CI coverage) and per compartment (R², rRMSE, GMFE, bias) in summary tables; a log(predicted) vs log(observed) plot is shown for passing compounds. For passing compound–isomers, GMFE by compartment is directional: plasma typically 1.2–2.1, milk 1.4–2.4, solid tissues (liver, kidney, etc.) ~1.2–1.9; feces 1.2–2.2; urine more variable and often higher (e.g. 1.5–5 or worse where data are sparse).
   3. Main challenges are noisy and censored data from naturally grown contaminated hay
   4. Across compartments, plasma, milk, and solid tissues are well represented; feces and especially urine are harder to model. Reasons are timing of excretion, measurement error and LOQ censoring.

2. **Uncertainty decomposition and prediction-interval coverage**
   1. Stacked prediction intervals are constructed at three levels: parameter uncertainty only (Param_only), parameter plus inter-animal variability (Param_plus_Animal), and parameter plus animal plus observational error (Param_plus_Animal_plus_Obs).
   2. **Display:** Empirical coverage (fraction of observations falling within each interval) is reported in **a single table**: one row per compound–isomer, three columns for coverage at each of the three CI levels, plus sample size N. This table (from `stacked_ci_coverage_by_compound.csv`) is the primary uncertainty result; no separate “interval width” summary is needed in the main text.
   3. A **stacked bar** figure for passing compounds (from `variance_decomposition_passing_compounds.png`) illustrates the relative contribution of parameter, biological, and measurement uncertainty to total predictive uncertainty. The finding is stated in numbers: coverage at the full uncertainty level (Param_plus_Animal_plus_Obs) was ≥90% for 12 of 15 passing compounds (or report the actual count from the coverage table).

3. **Parameter identifiability**
   1. The Fisher Information Matrix (FIM) is used to assess identifiability; the mean log10 diagonal of the FIM by parameter (across passing compounds) is reported (e.g. k_elim ≈ 3.7, k_feces ≈ 3.1, k_ehc ≈ 3.0; k_urine ≈ 2.1; k_a ≈ 1.0, with wide CI for k_a).
   2. **Interpretation:** Elimination and excretion parameters (k_elim, k_feces, k_ehc) are generally well identified; k_urine is informed only when urine data are sufficient (few compounds); absorption (k_a) is often poorly identified (low FIM, high correlation with other parameters). Parameter correlations are summarized; strongly correlated pairs (e.g. trade-offs between elimination pathways) are identified and their implications noted. **What was done:** Poorly identified parameters are retained in the model but reported with appropriate caveats (e.g. wide uncertainty or “informed mainly by prior/regularization”); no parameters are fixed to literature values solely due to identifiability—the sensitivity and identifiability results are used to interpret and qualify the estimates rather than to remove them.
   3. Eigenvalue analysis and condition number of the FIM are reported; compounds or parameter combinations with poor identifiability are explicitly noted so that readers know which estimates to treat with caution.

4. **Toxicokinetic characteristics (half-life vs milk transfer)** — *key scientific finding*
   1. For each passing compound–isomer, the model’s transition matrix is evaluated at a representative mid-lactation time (population-median physiology). Systemic half-life (days) is the slowest positive eigenmode, t_{1/2} = ln(2)/|Re(λ)|; the range across compounds is about 0.85 d to 25 d.
   2. At the same time, the steady-state milk transfer rate (fraction of intake excreted in milk) is computed; the range across compounds is about 0.1% to 13% of intake.
   3. The relationship between model-based systemic half-life and model-based milk transfer rate is presented in a log–log scatter plot; log–log linear regression (r and p) is reported. **Biological interpretation:** Compounds with longer systemic half-life show higher transfer to milk—slower elimination favours body accumulation and thus more mass transferred into milk per unit intake. This has direct implications for risk: long-chain PFAAs that persist in the body also dominate milk contamination under continuous exposure; the same relationship can inform which compounds merit priority for feed controls.

5. **Biological interpretation (determinants of milk transfer; lactation and milk yield)**
   1. **Determinants of milk transfer:** Local sensitivity (sensitivity_summary, heatmap) shows which parameters drive milk concentration (e.g. k_elim vs k_a vs excretion pathways); the half-life–transfer relationship is interpreted as the dominant TK determinant: systemic elimination rate governs both body burden and milk transfer. Brief discussion of absorption and excretion pathway contributions, where identifiable.
   2. **Effect of lactation stage and milk production rate:** Time-course predictions (e.g. feed-based milk profiles) show how concentration varies over the lactation via the physiology sub-model. Breed×parity comparisons (breed_parity_exposure_scenarios.csv) show the effect of milk production rate (and body size) on end-of-exposure milk concentration; the net effect (e.g. higher yield diluting concentration) is summarized so that TK findings are discussed in biological terms.

6. **Applications**
   1. Scenario analysis over a full lactation: milk (and optionally plasma) concentrations are shown for different breeds, parities, and exposure patterns (breed_parity_exposure_scenarios).
   2. Feed-based exposure scenario: predicted milk concentration time courses over a complete lactation (e.g. grass silage / EFSA4-based scenario) are presented; implications for consumer exposure or compliance with guidance values are briefly discussed.
   3. Maximum allowable feed concentrations: for PFAAs with EU indicative levels in milk, the maximum feed concentration (µg/kg DM) that keeps predicted milk at or below the regulatory limit is estimated by breed and parity (max_feed_concentrations table). **Regulatory punchline:** The maximum allowable feed concentrations are compared explicitly to common or typical measured feed concentrations (e.g. from monitoring or the same assessment used for feed-based scenarios). The finding that these maxima fall **below** or **above** typical measured levels is stated clearly—e.g. “maximum allowable feed for PFOS is 0.06–0.07 µg/kg DM (by scenario), below [or above] typical measured concentrations in grass silage (X µg/kg DM),” so that regulators see immediately whether current exposure levels are in a margin of safety or of concern.

## Discussion

1. **Overview**
   1. This study provides the first physiologically based toxicokinetic model for multiple PFAAs in dairy goats, parameterized and evaluated against an in vivo transfer study with explicit treatment of uncertainty and identifiability.
   2. Main findings are summarized: 15 compound–isomer combinations met acceptability criteria; the positive relationship between model-derived systemic half-life and milk transfer rate is a key mechanistic result; and the model is applied to scenario analysis and maximum allowable feed concentration estimates for regulatory use. The discussion interprets these findings, compares them to the goat study data and to other PBTK models, states limitations, and outlines implications.

2. **Comparison with data from the goat transfer study**
   1. Model-based transfer rates and half-lives are compared to data-based estimates from the same study where available (e.g. plasma depuration half-life from concentration decay; milk transfer from mass balance). **Directional claim:** Agreement is generally within a factor of 2 where data are sufficient; larger discrepancies occur for short-chain or poorly identified compounds where censoring is heaviest (or state the actual finding, e.g. “model-based half-lives were within 30% of data-based estimates for X of Y compounds”).
   2. For compounds with sufficient data, the direction and magnitude of any bias (e.g. model over- or under-predicting milk or plasma) are interpreted in light of model structure (e.g. single systemic half-life, excretion pathway assumptions) and data quality (censoring, sampling timing).

3. **Comparison with other PBTK models (cattle, swine, laying hens)**
   1. Model-derived half-lives and milk (or tissue) transfer rates for goats are compared to published values from PFAA PBTK models in cattle, swine, and laying hens, where comparable metrics exist. Species differences in physiology (body size, lactation curve, clearance pathways) are invoked to interpret differences in half-life and transfer.
   2. The goat model is placed in context: it fills a gap for small ruminants and supports the generalisation that longer-chain PFAAs tend to have longer half-lives and higher transfer to milk or eggs in food-producing species, while quantitative differences reflect species-specific physiology and exposure design.

4. **Limitations**
   1. **Data:** The in vivo study used naturally contaminated hay, leading to noisy and censored data (especially below LOQ); only 15 of 30 compound–isomer combinations passed fit criteria. Single study, single herd, and limited animal numbers limit generalisability; exclusion of one animal (E1) is noted.
   2. **Model and identifiability:** Some parameters (e.g. k_a, k_urine for some compounds) are poorly identified; results are interpreted with appropriate caveats. Model structure (e.g. single systemic elimination rate, unspecified elimination pathway) may not capture all kinetic detail. Physiology sub-model is parameterized for Alpine and Saanen under defined conditions; extrapolation to other breeds or production systems carries uncertainty.
   3. **Applications:** Maximum allowable feed concentrations and scenario results depend on the assumed physiology, regulatory limits, and feed concentration inputs; comparison with measured feed levels is essential for regulatory interpretation.

5. **Implications**
   1. **Regulatory:** The model can support feed safety assessment and risk evaluation for goat milk (e.g. comparing maximum allowable feed concentrations to typical or worst-case measured feed concentrations; scenario-based exposure assessment over a full lactation). It provides a tool for prioritising PFAAs and exposure scenarios of concern.
   2. **Scientific:** The half-life–milk transfer relationship offers a mechanistic basis for understanding and predicting milk contamination across PFAAs and may extend to other small ruminants or compounds if validated. Identifiability and uncertainty analysis set a standard for transparency in PBTK applications. Future work might include extrapolation to other types of dairy cattle  (cows and sheep).
