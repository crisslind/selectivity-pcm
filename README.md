# selectivity-pcm

`selectivity-pcm` is a Python toolkit for modelling kinase potency and selectivity using ligand fingerprints together with protein descriptors.

The project includes:

- a **CDK benchmark** using CDK1/CDK2/CDK4/CDK6
- a **ROCK2 selectivity workflow** using ROCK2 as the primary target, ROCK1 as a critical anti-target, and additional kinases as profile targets

The reusable code under `src/selectivity_pcm/` is target-agnostic. Target definitions, protein sequences, KLIFS pockets, and project-specific decision logic are kept outside the core package.

```text
compound
    ↓
primary-target potency
    ↓
critical anti-target selectivity
    ↓
profile-target context
    ↓
calibrated probabilities / risk
    ↓
MPO-style ranking
    ↓
ADVANCE / WATCH / DEPRIORITIZE
```

## Features

- ChEMBL IC50 retrieval and curation
- Morgan fingerprint QSAR
- pooled multi-target modelling
- whole-sequence PCM
- KLIFS pocket-based PCM
- scaffold-grouped cross-validation
- empirical probability calibration
- selectivity-risk estimation
- critical anti-target vs profile-target separation
- external compound prediction
- configurable MPO-style ranking

## Installation

```bash
git clone https://github.com/crisslind/selectivity-pcm.git
cd selectivity-pcm
python -m pip install -e .
```

For KLIFS/OpenCADD utilities:

```bash
python -m pip install -e ".[klifs]"
```

For development:

```bash
python -m pip install -e ".[dev]"
```

Python 3.11+ is currently required.

## Repository structure

```text
selectivity-pcm/
├── docs/
│   ├── custom_projects.md
│   ├── methodology.md
│   ├── model_benchmarks.md
│   └── rock2_example.md
├── examples/
│   ├── cdk/
│   └── rock2/
├── scripts/
├── src/selectivity_pcm/
│   ├── analysis/
│   ├── data/
│   ├── features/
│   └── models/
├── tests/
├── pyproject.toml
└── README.md
```

Generated datasets and results are intentionally excluded from version control.

# Quick start: ROCK2 example

The ROCK2 example uses:

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

## 1. Build the dataset

```bash
python scripts/build_rock2_panel_dataset.py
```

## 2. Inspect the observed selectivity landscape

```bash
python scripts/analyze_rock2_landscape.py
```

## 3. Inspect or generate KLIFS pockets

```bash
python scripts/check_klifs_panel_coverage.py
python scripts/fetch_klifs_pockets.py
```

The ROCK2 pocket configuration is stored in:

```text
examples/rock2/kinase_pockets.py
```

## 4. Run the pocket PCM model

```bash
python scripts/run_rock2_pocket_pcm.py
```

The current feature representation combines:

```text
Morgan fingerprint              2048
KLIFS pocket one-hot            1785
------------------------------------
Total                           3833
```

Current 5-fold scaffold-CV benchmark:

```text
RMSE       0.853
MAE        0.606
R²         0.582
Spearman   0.738
```

## 5. Run a pseudo-prospective holdout

```bash
python scripts/run_prospective_holdout.py
```

Current holdout performance:

```text
RMSE   0.747
MAE    0.551
R²     0.472
```

## 6. Build the final report

```bash
python scripts/build_final_report.py
```

The report combines:

- primary-target potency
- critical anti-target margin and risk
- profile-target context
- calibration support
- MPO-style ranking
- recommendation

For the full ROCK2 walkthrough, see [`docs/rock2_example.md`](docs/rock2_example.md).

# Predict external compounds

Prepare a CSV such as:

```csv
compound_id,smiles
compound_001,CCOc1ccc(...)
compound_002,CN1CCN(...)
```

Then run:

```bash
python scripts/predict_external_compounds.py
python scripts/build_external_report.py
```

The external workflow trains on the curated dataset and predicts the configured target panel for unseen compounds.

# Use your own targets

The core package is not restricted to ROCK2 or CDKs.

A custom modelling table should contain at least:

```text
compound_id
target_name
pIC50
canonical_smiles
```

Example ChEMBL panel:

```python
from selectivity_pcm.data.chembl import build_target_dataset

TARGETS = {
    "TARGET_A": "CHEMBL123",
    "TARGET_B": "CHEMBL456",
}

df = build_target_dataset(TARGETS)
```

Example pocket PCM:

```python
from selectivity_pcm.models.pocket_pcm import (
    cross_validate_pocket_pcm_scaffold,
)

overall_metrics, target_metrics, predictions = (
    cross_validate_pocket_pcm_scaffold(
        df,
        pockets=POCKETS,
        n_splits=5,
        random_state=42,
    )
)
```

For a fuller guide to custom panels, pocket definitions, sequence PCM, and project-specific target roles, see [`docs/custom_projects.md`](docs/custom_projects.md).

# Model choice

The current default regressor is `RandomForestRegressor`.

Using identical Morgan + KLIFS features and scaffold folds:

```text
Model          RMSE    MAE     R²      Spearman
RandomForest   0.853   0.606   0.582   0.738
LightGBM       0.856   0.633   0.579   0.731
XGBoost        0.865   0.644   0.570   0.725
ExtraTrees     1.110   0.749   0.292   0.613
```

Random Forest was retained because it performed best overall under the same scaffold-grouped validation scheme.

See [`docs/model_benchmarks.md`](docs/model_benchmarks.md) for details.

# Calibration and decision logic

The workflow separates three concepts:

```text
prediction
calibration
decision
```

The model predicts pIC50.

Out-of-fold historical predictions are then used for empirical potency and selectivity calibration.

The decision layer combines model output, selectivity, uncertainty, and project-specific target priorities.

Critical anti-targets can affect progression decisions directly, while profile targets provide broader context.

See [`docs/methodology.md`](docs/methodology.md).

# External validation

The ROCK2 workflow was tested blindly on a small external series from public patent literature after checking for exact structure overlap with the training data.

The model retained useful signal for ROCK2 potency ranking, but absolute potency and especially large ROCK2/ROCK1 selectivity differences were compressed toward the training-set mean.

This is an important limitation:

> Predictions should not be interpreted as reliable quantitative extrapolations for novel chemotypes without supporting calibration or experimental data.

# Tests

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run:

```bash
pytest -q
ruff check .
```

The current tests cover core feature dimensions, selectivity-margin logic, and insufficient-calibration handling.

# Documentation

- [`docs/rock2_example.md`](docs/rock2_example.md) — complete ROCK2 workflow
- [`docs/custom_projects.md`](docs/custom_projects.md) — use your own targets and data
- [`docs/model_benchmarks.md`](docs/model_benchmarks.md) — model comparison
- [`docs/methodology.md`](docs/methodology.md) — calibration, validation, decision logic, and limitations

# Intended use

`selectivity-pcm` is an experimental modelling toolkit for:

- exploratory SAR analysis
- model comparison
- selectivity hypothesis generation
- compound prioritization
- uncertainty-aware review

It is not intended to replace experimental potency or selectivity measurements.

# License

MIT License. See [`LICENSE`](LICENSE).
