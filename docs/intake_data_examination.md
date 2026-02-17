# Examination of `data/processed/pfas_daily_intake.csv`

## Structure and usage

- **Columns:** `Animal`, `Day`, `Compound`, `Isomer`, `PFAS_Intake_ug_day`, `Hay_Concentration_ug_kg`, `Feed_Intake_kg_day`
- **Used by:** `optimization/io.py` loads it; `optimization/fit.py` uses it via `build_intake_function()` which looks up `(Animal, Compound, Isomer)` and maps `Day` → `PFAS_Intake_ug_day`. Missing days get `0.0`.
- **Formula (upstream):** `PFAS_Intake_ug_day = Hay_Concentration_ug_kg × Feed_Intake_kg_day` (from `mass_balance/intake.py`).

## Checks performed

1. **Coverage**  
   - PFOS Linear: 423 rows (3 animals × 141 days). All days 0–140 present. No missing days for the fit.

2. **Exposure window**  
   - First day with positive intake: **Day 2** (Day 0 and 1 are zero).  
   - Last day with positive intake: **Day 56** (matches `EXPOSURE_PERIOD_DAYS = 56` in `mass_balance/intake.py` and `fit_variables.py`).  
   - No zeros in the middle of exposure (2–56) for PFOS Linear.

3. **Totals (PFOS Linear)**  
   - E2: 7712 µg total over 55 days  
   - E3: 7582 µg total  
   - E4: 7082 µg total  
   - Plausible and similar across animals.

4. **NaNs**  
   - `Hay_Concentration_ug_kg`: 10 710 NaNs. These are Day 0 and days outside the exposure window; the script writes `None` when `in_exposure` is False. **`PFAS_Intake_ug_day` has no NaNs** and is 0 when hay is not in exposure, so the model intake is correct.

5. **Day alignment**  
   - Model time `t` is integer days; intake is taken from the row with `Day == int(t)`. So `t=0` → 0 intake, `t=2` → first positive intake.  
   - Observation time mapping: “observation Day d = model time index d (Day 0 = baseline)” . So the intake file’s “Day” is the same as the model’s integer time.

## Where intake could still cause problems

1. **Wrong exposure start (Day 1 vs Day 2)**  
   - In the intake table, the first positive intake is **Day 2** because `animal_data.csv` has `Feed_Intake_kg_per_day = 0` on Day 1.  
   - If the study protocol defines “Day 1” as first exposure day, the model effectively starts intake one day late (Day 2). That can:
     - Slightly under-predict early concentrations,
     - Be partly “fixed” by the optimizer with different k_a / k_ehc / k_feces, adding to correlation.

2. **Units or scaling**  
   - If hay concentrations or feed intake were scaled/wrong (e.g. per kg DM vs per kg as-fed), total intake would be biased and the fit would shift (e.g. different apparent clearance and EHC).

3. **Animal–day coverage**  
   - `build_intake_function` uses only rows for that (Animal, Compound, Isomer). If an animal had no rows for a compound–isomer, it would get a constant 0 intake. For PFOS Linear all three animals have full 0–140 day coverage.

## Conclusion

- The processed intake file is **internally consistent**: no gaps in days, exposure 2–56, no NaNs in `PFAS_Intake_ug_day`, and totals look plausible.  
- A **possible** source of bias/correlation is the **1-day offset** (first intake on Day 2 instead of Day 1) if the protocol says exposure starts on Day 1; verifying that against the protocol and, if needed, aligning first exposure day in `animal_data` or in the intake script would be the main check.  
- Parameter correlation (k_a, k_ehc, k_feces) is primarily structural (same dynamics, few animals); intake errors would add to that but are unlikely to be the sole cause.
