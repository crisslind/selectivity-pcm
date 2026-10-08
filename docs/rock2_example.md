# ROCK2 example workflow

This document describes the complete ROCK2 example included with `selectivity-pcm`.

## Target panel

```text
Primary target:
ROCK2

Critical anti-target:
ROCK1

Profile targets:
CSNK2A2
AURKA
STK3
```

The distinction between critical anti-targets and profile targets is deliberate:

- critical anti-targets can drive hard decisions
- profile targets provide additional context without automatically vetoing a compound

## 1. Build the public dataset

Run:

```bash
python scripts/build_rock2_panel_dataset.py
```

The ROCK2 workflow uses public ChEMBL data and retains exact quantitative IC50 measurements for regression.

The curated dataset is based on biochemical single-protein measurements and pIC50 values are calculated from IC50 in nM.

## 2. Inspect the observed selectivity landscape

Run:

```bash
python scripts/analyze_rock2_landscape.py
```

This builds a compound-by-target activity matrix and summarizes observed ROCK2 selectivity against available off-target measurements.

The public dataset is sparse across full multi-target profiles, which is one reason the workflow separates hard anti-target decisions from broader profile interpretation.

## 3. Generate KLIFS pocket definitions

Check panel coverage:

```bash
python scripts/check_klifs_panel_coverage.py
```

Generate the selected pockets:

```bash
python scripts/fetch_klifs_pockets.py
```

The resulting example file is:

```text
examples/rock2/kinase_pockets.py
```

Each kinase is represented by the 85 aligned KLIFS pocket positions.

The current example structures are:

```text
ROCK2     4L6Q
ROCK1     3V8S
CSNK2A2   3OFM
AURKA     4DEE
STK3      5DH3
```

## 4. Pocket PCM

Run:

```bash
PYTHONPATH=. python scripts/run_rock2_pocket_pcm.py
```

Each observation is represented as:

```text
Morgan fingerprint (2048)
+
KLIFS pocket one-hot encoding (1785)
=
3833 features
```

The current implementation uses a `RandomForestRegressor`.

Evaluation uses 5-fold scaffold-grouped cross-validation with global Bemis-Murcko scaffold groups.

Current mean results:

```text
RMSE        0.853
MAE         0.606
R²          0.582
Spearman    0.738
```

Per-target mean metrics:

```text
Target    RMSE    MAE     R²      Spearman
AURKA     0.956   0.727   0.526   0.714
CSNK2A2   0.817   0.523   0.693   0.813
ROCK1     0.821   0.628   0.469   0.684
ROCK2     0.837   0.626   0.402   0.639
STK3      0.841   0.610   0.416   0.680
```

## 5. Pseudo-prospective holdout

Run:

```bash
PYTHONPATH=. python scripts/run_prospective_holdout.py
```

This:

1. selects whole compounds for holdout
2. removes every observation for those compounds from training
3. trains the model on the remaining data
4. predicts all configured targets for the held-out compounds
5. calibrates potency probabilities and selectivity risk

The holdout is compound-based rather than scaffold-disjoint because it is intended to mimic a medicinal-chemistry setting in which nearby analogues may already exist in the training set.

Current overall performance:

```text
RMSE   0.747
MAE    0.551
R²     0.472
```

## 6. Calibration

Potency probabilities are estimated empirically from nearby out-of-fold predictions.

Examples include:

```text
P(pIC50 >= 6)
P(pIC50 >= 7)
P(pIC50 >= 8)
```

Selectivity risk is also estimated empirically from paired OOF predictions.

For a 1-log margin:

```text
P(off-target lies within 1 log unit of the primary target)
```

If paired support is too sparse, the workflow reports:

```text
INSUFFICIENT_DATA
```

rather than forcing a probability.

## 7. Final recommendation report

Run:

```bash
PYTHONPATH=. python scripts/build_final_report.py
```

Important fields include:

```text
ROCK2_predicted_pIC50
ROCK1_predicted_pIC50
worst_critical_offtarget
worst_critical_margin
max_critical_offtarget_risk
worst_profile_target
worst_profile_margin
max_profile_target_risk
profile_insufficient_targets
profile_status
profile_comment
MPO_rank
selectivity_MPO
recommendation
recommendation_type
```

The current top-level decisions are:

```text
ADVANCE
WATCH
DEPRIORITIZE
```

## 8. External compounds

Prepare:

```csv
compound_id,smiles
compound_001,CCOc1ccc(...)
compound_002,CN1CCN(...)
```

Run:

```bash
PYTHONPATH=. python scripts/predict_external_compounds.py
PYTHONPATH=. python scripts/build_external_report.py
```

The external workflow trains on all available curated public data and produces predictions without observed labels.

## 9. Blind patent validation

The ROCK2 model was tested on 15 external compounds from public patent literature after verifying zero exact-structure overlap with the training set.

The test showed:

- useful signal for ROCK2 potency ranking
- compression of absolute potency toward the training-set mean
- poor extrapolation of very large ROCK2/ROCK1 selectivity differences

This is an important example of why model prediction, calibration, and progression decisions should be kept as separate layers.
