# Using selectivity-pcm with your own project

The core `selectivity_pcm` package is target-agnostic.

A new project needs:

1. compound-target activity data
2. target descriptors
3. project-specific target roles
4. optional calibration and decision rules

## Activity data

The main modelling functions expect at least:

```text
compound_id
target_name
pIC50
canonical_smiles
```

Example:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "compound_id": [
            "cmpd_1",
            "cmpd_1",
            "cmpd_2",
            "cmpd_2",
        ],
        "target_name": [
            "TARGET_A",
            "TARGET_B",
            "TARGET_A",
            "TARGET_B",
        ],
        "pIC50": [
            7.4,
            6.2,
            8.0,
            6.5,
        ],
        "canonical_smiles": [
            "SMILES_A",
            "SMILES_A",
            "SMILES_B",
            "SMILES_B",
        ],
    }
)
```

## Fetch a custom ChEMBL panel

Define a target-to-ChEMBL mapping:

```python
TARGETS = {
    "TARGET_A": "CHEMBL123",
    "TARGET_B": "CHEMBL456",
    "TARGET_C": "CHEMBL789",
}
```

Then:

```python
from selectivity_pcm.data.chembl import build_target_dataset

raw = build_target_dataset(TARGETS)
```

The returned data can then be passed through the curation workflow before modelling.

## Pocket PCM

Define KLIFS-aligned pocket data:

```python
POCKETS = {
    "TARGET_A": {
        "pdb": "XXXX",
        "structure_klifs_id": 1234,
        "pocket_sequence": "85-residue aligned pocket sequence",
    },
    "TARGET_B": {
        "pdb": "YYYY",
        "structure_klifs_id": 5678,
        "pocket_sequence": "85-residue aligned pocket sequence",
    },
}
```

Then run:

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

The model combines ligand Morgan fingerprints with target pocket descriptors.

## Whole-sequence PCM

Define protein sequences:

```python
PROTEINS = {
    "TARGET_A": {
        "uniprot": "P12345",
        "sequence": "MKT...",
    },
    "TARGET_B": {
        "uniprot": "Q67890",
        "sequence": "MPE...",
    },
}
```

Then:

```python
from selectivity_pcm.models.pcm import (
    cross_validate_pcm_scaffold,
)

metrics, target_metrics, predictions = (
    cross_validate_pcm_scaffold(
        df,
        proteins=PROTEINS,
        n_splits=5,
        random_state=42,
    )
)
```

## Ligand-only QSAR

Ligand-only modelling utilities live under:

```text
selectivity_pcm.models.qsar
```

These provide a useful baseline for asking whether protein information adds predictive value.

## Pooled multi-target model

A simpler multi-target baseline is available under:

```text
selectivity_pcm.models.pooled
```

This encodes target identity without using protein sequence or pocket structure.

## Defining target roles

For a discovery project, define a primary target and any hard anti-targets explicitly:

```python
PRIMARY_TARGET = "TARGET_A"

CRITICAL_OFFTARGETS = [
    "TARGET_B",
    "TARGET_C",
]
```

All other modelled targets can be treated as profile targets.

This distinction is useful because not every measured off-target has equal biological importance.

## Selectivity margin

The workflow defines predicted selectivity margin as:

```text
primary pIC50 - off-target pIC50
```

Example:

```text
Primary pIC50      8.0
Off-target pIC50   6.5
Margin             1.5
```

Interpretation:

```text
1 log unit ≈ 10-fold
2 log units ≈ 100-fold
3 log units ≈ 1000-fold
```

## External compound scoring

Prepare:

```csv
compound_id,smiles
candidate_001,SMILES
candidate_002,SMILES
```

The ROCK2 scripts provide a worked example of how to:

1. train on the full curated dataset
2. predict every configured target
3. calibrate potency and selectivity risk
4. build a final recommendation report

For a new project, these scripts can be copied and supplied with a different target/pocket configuration.

## Recommended workflow for a new target family

A practical sequence is:

```text
1. define target panel
2. fetch / curate activities
3. inspect compound-target coverage
4. build ligand-only QSAR baseline
5. build pooled multi-target baseline
6. build sequence or pocket PCM
7. compare models with scaffold CV
8. generate OOF predictions for calibration
9. define critical anti-targets
10. score external candidates
11. inspect model support before making decisions
```

The important principle is that target configuration should live outside the reusable model code.
