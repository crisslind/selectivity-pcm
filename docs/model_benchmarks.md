# Model benchmarks

The current pocket-PCM workflow uses a `RandomForestRegressor`.

Several alternative regressors were evaluated using the same:

- ROCK2-panel dataset
- Morgan fingerprint features
- KLIFS pocket features
- 5-fold scaffold-grouped cross-validation
- target definitions
- train/test folds

This makes the comparison primarily a test of the regressor rather than the representation.

## Overall results

```text
Model          RMSE    MAE     R²      Spearman
RandomForest   0.853   0.606   0.582   0.738
LightGBM       0.856   0.633   0.579   0.731
XGBoost        0.865   0.644   0.570   0.725
ExtraTrees     1.110   0.749   0.292   0.613
```

Random forest was retained as the default.

## Random Forest

Current pocket-PCM baseline:

```text
RMSE        0.853
MAE         0.606
R²          0.582
Spearman    0.738
```

ROCK2-specific mean performance:

```text
RMSE        0.837
MAE         0.626
R²          0.402
Spearman    0.639
```

## LightGBM

Overall:

```text
RMSE        0.856
MAE         0.633
R²          0.579
Spearman    0.731
```

ROCK2-specific:

```text
RMSE        0.859
MAE         0.654
R²          0.369
Spearman    0.614
```

LightGBM was competitive overall but did not improve the primary-target metrics.

## XGBoost

Overall:

```text
RMSE        0.865
MAE         0.644
R²          0.570
Spearman    0.725
```

ROCK2-specific:

```text
RMSE        0.859
MAE         0.658
R²          0.370
Spearman    0.616
```

XGBoost was slightly worse than the RF baseline across the main metrics.

## ExtraTrees

Overall:

```text
RMSE        1.110
MAE         0.749
R²          0.292
Spearman    0.613
```

ROCK2-specific:

```text
RMSE        1.092
MAE         0.783
R²         -0.022
Spearman    0.485
```

ExtraTrees was substantially worse and was not retained.

## Interpretation

The benchmark suggests that the current feature representation is already handled effectively by the random forest.

LightGBM and XGBoost remain reasonable alternative baselines, but neither provided a meaningful improvement under identical scaffold CV.

The external patent validation also suggests that the main limitation is not simply the choice of tree ensemble. Large potency/selectivity extrapolations for novel chemistry remain difficult.

Future model work would therefore be more interesting if it changes the representation or modelling paradigm, for example:

- learned molecular representations
- explicit ligand-target interaction models
- graph neural networks
- protein language-model embeddings
- similarity-aware or domain-of-applicability models

rather than continuing to tune closely related tree ensembles.
