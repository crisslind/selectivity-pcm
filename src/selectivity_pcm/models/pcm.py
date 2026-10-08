import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold

from selectivity_pcm.features.protein_features import (
    protein_sequence_features,
)
from selectivity_pcm.models.qsar import (
    get_scaffold,
    smiles_to_array,
)


def prepare_pcm_dataset(
    df: pd.DataFrame,
    proteins: dict,
):
    """
    Prepare ligand + protein features for PCM.

    Each observation:
        Morgan fingerprint
        +
        protein sequence descriptors
        ->
        pIC50
    """

    features = []
    activities = []
    compound_ids = []
    target_names = []
    scaffold_groups = []

    if not proteins:
        raise ValueError(
            "Protein mapping cannot be empty"
        )

    protein_feature_cache = {}

    for target_name, protein in proteins.items():
        protein_feature_cache[target_name] = (
            protein_sequence_features(
                protein["sequence"]
            )
        )

    for _, row in df.iterrows():

        target_name = row["target_name"]

        if target_name not in proteins:
            continue

        ligand_fp = smiles_to_array(
            row["canonical_smiles"]
        )

        if ligand_fp is None:
            continue

        protein_features = (
            protein_feature_cache[
                target_name
            ]
        )

        feature_vector = np.concatenate(
            [
                ligand_fp,
                protein_features,
            ]
        )

        scaffold = get_scaffold(
            row["canonical_smiles"]
        )

        if scaffold is None:
            continue

        features.append(
            feature_vector
        )

        activities.append(
            row["pIC50"]
        )

        compound_ids.append(
            row["compound_id"]
        )

        target_names.append(
            target_name
        )

        scaffold_groups.append(
            scaffold
        )

    return (
        np.asarray(features),
        np.asarray(activities),
        compound_ids,
        target_names,
        np.asarray(scaffold_groups),
    )

def cross_validate_pcm_scaffold(
    df: pd.DataFrame,
    proteins: dict,
    n_splits: int = 5,
    random_state: int = 42,
):
    (
        X,
        y,
        compound_ids,
        target_names,
        scaffold_groups,
    ) = prepare_pcm_dataset(
        df,
        proteins=proteins,
        )

    splitter = GroupKFold(
        n_splits=n_splits,
    )

    fold_metrics = []
    target_metrics = []
    prediction_tables = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(
            X,
            y,
            groups=scaffold_groups,
        ),
        start=1,
    ):

        train_scaffolds = set(
            scaffold_groups[train_idx]
        )

        test_scaffolds = set(
            scaffold_groups[test_idx]
        )

        overlap = (
            train_scaffolds
            & test_scaffolds
        )

        print(
            f"Fold {fold}: "
            f"scaffold overlap = {len(overlap)}"
        )

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=random_state + fold,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        spearman, _ = spearmanr(
            y_test,
            predictions,
        )

        fold_metrics.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),

                "RMSE": (
                    mean_squared_error(
                        y_test,
                        predictions,
                    )
                    ** 0.5
                ),

                "MAE": mean_absolute_error(
                    y_test,
                    predictions,
                ),

                "R2": r2_score(
                    y_test,
                    predictions,
                ),

                "Spearman": spearman,
            }
        )

        prediction_table = pd.DataFrame(
            {
                "compound_id": [
                    compound_ids[i]
                    for i in test_idx
                ],

                "target_name": [
                    target_names[i]
                    for i in test_idx
                ],

                "fold": fold,

                "observed_pIC50": y_test,

                "predicted_pIC50": predictions,
            }
        )

        prediction_tables.append(
            prediction_table
        )

        for target in sorted(
            set(target_names)
        ):
            target_data = prediction_table[
                prediction_table[
                    "target_name"
                ]
                == target
            ]

            if len(target_data) < 2:
                continue

            observed = target_data[
                "observed_pIC50"
            ].to_numpy()

            predicted = target_data[
                "predicted_pIC50"
            ].to_numpy()

            target_spearman, _ = (
                spearmanr(
                    observed,
                    predicted,
                )
            )

            target_metrics.append(
                {
                    "target": target,
                    "fold": fold,
                    "n_test": len(
                        target_data
                    ),

                    "RMSE": (
                        mean_squared_error(
                            observed,
                            predicted,
                        )
                        ** 0.5
                    ),

                    "MAE": (
                        mean_absolute_error(
                            observed,
                            predicted,
                        )
                    ),

                    "R2": r2_score(
                        observed,
                        predicted,
                    ),

                    "Spearman": (
                        target_spearman
                    ),
                }
            )

    return (
        pd.DataFrame(
            fold_metrics
        ),

        pd.DataFrame(
            target_metrics
        ),

        pd.concat(
            prediction_tables,
            ignore_index=True,
        ),
    )