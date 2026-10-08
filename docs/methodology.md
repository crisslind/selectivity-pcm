# Methodology, calibration, and limitations

This document describes the main methodological choices behind `selectivity-pcm`.

## 1. Prediction

The current default pocket PCM combines:

```text
Morgan fingerprint
+
85-position KLIFS pocket encoding
```

and predicts pIC50 with a random forest.

The model is evaluated with scaffold-grouped cross-validation.

## 2. Scaffold-grouped validation

Bemis-Murcko scaffolds are grouped globally across targets.

This prevents the same scaffold from appearing in both the train and test fold against different targets.

This is stricter than random observation splitting and reduces direct chemical leakage.

## 3. Pseudo-prospective holdout

A separate holdout workflow removes complete compounds from the training data.

Unlike scaffold CV, this holdout intentionally allows related analogues to remain in training.

The purpose is to mimic an active medicinal-chemistry project in which the next compounds may be related to previously measured chemistry.

The two evaluations therefore answer different questions:

```text
scaffold CV:
How well does the model generalize across chemical scaffolds?

compound holdout:
How well does the workflow behave in an analogue-rich project setting?
```

## 4. Calibration

Raw random-forest values are treated as potency predictions, not probabilities.

The workflow therefore uses out-of-fold historical predictions for empirical calibration.

For a new predicted pIC50, nearby historical OOF predictions from the same target are selected and used to estimate quantities such as:

```text
P(pIC50 >= 6)
P(pIC50 >= 7)
P(pIC50 >= 8)
```

This provides a local empirical estimate of how often predictions in that region corresponded to experimentally strong compounds.

## 5. Selectivity calibration

For a primary target and an off-target:

```text
predicted margin =
predicted primary pIC50
-
predicted off-target pIC50
```

Paired historical OOF predictions are used to estimate selectivity risk.

For example:

```text
P(primary - off-target <= 1 log unit)
```

can be interpreted as an empirical probability that the off-target lies within approximately 10-fold potency of the desired target.

## 6. Insufficient data

A selectivity probability is not forced when too few paired calibration examples exist.

The workflow instead returns:

```text
INSUFFICIENT_DATA
```

This is important because some target pairs may have many individual measurements but very little overlapping compound data.

## 7. Critical anti-targets vs profile targets

A project may contain:

```text
PRIMARY_TARGET
CRITICAL_OFFTARGETS
PROFILE_TARGETS
```

Critical anti-targets can affect hard recommendation logic and MPO scoring.

Profile targets provide context and warnings.

For example:

```text
PROFILE_OK
PROFILE_CLOSE
PROFILE_LIABILITY
PROFILE_INSUFFICIENT_DATA
PROFILE_LIABILITY_AND_INSUFFICIENT_DATA
```

This avoids treating every target in a profiling panel as biologically equivalent.

## 8. MPO-style ranking

The example ROCK2 decision layer combines:

- primary potency
- empirical potency probability
- critical anti-target selectivity margin
- calibrated critical anti-target risk
- confidence / calibration support

into a 0-100 score.

The current example weighting is project-specific and should not be treated as universal.

Users should define progression thresholds and target roles based on their own biology and program goals.

## 9. External validation

A blinded external test was performed on 15 structures from public patent literature.

Before prediction:

- the structures were treated as unknowns
- exact structure overlap with the training data was checked
- model outputs were frozen before patent activity bins were inspected

The main observations were:

- useful signal for ROCK2 potency ranking
- underprediction / compression of highly potent compounds
- severe compression of large ROCK2-over-ROCK1 selectivity differences

The patent series therefore exposed a clear domain-generalization limitation.

## 10. Regression toward the mean

Tree ensembles are strong interpolators but limited extrapolators.

For novel chemistry, predicted activities may be pulled toward regions well represented in the training data.

This was visible in the external patent test.

A high-confidence numerical prediction should therefore not automatically be interpreted as evidence that the model can extrapolate to extreme potency or selectivity.

## 11. Data limitations

The public workflow inherits limitations from ChEMBL, including:

- assay heterogeneity
- differences in constructs and assay conditions
- uneven target coverage
- sparse multi-target overlap
- replicate variability
- publication bias

The curation pipeline reduces some of these issues but cannot eliminate them.

## 12. Reproducibility

The workflows use:

- explicit random seeds
- deterministic target configurations
- scaffold-grouped folds
- repository-root-relative paths
- saved OOF predictions for calibration

Generated data and result files are intentionally separated from reusable source code.

## 13. Model benchmarking

Random Forest, LightGBM, XGBoost, and ExtraTrees were evaluated under identical scaffold folds.

Random Forest currently performs best overall.

The benchmark suggests that future improvements are more likely to come from better representations, domain-awareness, or additional data than from swapping between closely related tree ensembles.

## 14. Intended use

`selectivity-pcm` is an experimental modelling toolkit.

It is intended to support:

- exploratory SAR analysis
- model comparison
- selectivity hypothesis generation
- prioritization
- uncertainty-aware review

It is not intended to replace experimental potency or selectivity measurements.

Experimental data should remain the basis for compound progression.
