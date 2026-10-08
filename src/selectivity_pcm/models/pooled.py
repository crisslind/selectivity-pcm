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

from selectivity_pcm.models.qsar import (
    get_scaffold,
    smiles_to_array,
)


def prepare_pooled_dataset(
    df: pd.DataFrame,
):
    """
    Prepare a pooled compound-target dataset.

    Features consist of:
        Morgan fingerprint
        +
        one-hot encoded target identity

    Each row represents one compound-target measurement.
    """

    required_columns = {
        "compound_id",
        "canonical_smiles",
        "target_name",
        "pIC50",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}"
        )

    targets = sorted(
        df["target_name"]
        .dropna()
        .unique()
    )

    target_to_index = {
        target: i
        for i, target in enumerate(targets)
    }

    features = []
    activities = []
    compound_ids = []
    target_names = []
    scaffold_groups = []

    for _, row in df.iterrows():
        fp = smiles_to_array(
            row["canonical_smiles"]
        )

        if fp is None:
            continue

        target_vector = np.zeros(
            len(targets),
            dtype=np.uint8,
        )

        target_vector[
            target_to_index[row["target_name"]]
        ] = 1

        feature_vector = np.concatenate(
            [
                fp,
                target_vector,
            ]
        )

        scaffold = get_scaffold(
            row["canonical_smiles"]
        )

        if scaffold is None:
            continue

        features.append(feature_vector)
        activities.append(row["pIC50"])
        compound_ids.append(row["compound_id"])
        target_names.append(row["target_name"])
        scaffold_groups.append(scaffold)

    X = np.asarray(features)
    y = np.asarray(activities)

    return (
        X,
        y,
        compound_ids,
        target_names,
        np.asarray(scaffold_groups),
        targets,
    )


def cross_validate_pooled_scaffold(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Five-fold scaffold-grouped CV for the pooled target-aware model.

    A scaffold is never present in both training and test data
    within the same fold.

    Because scaffold grouping is global, all measurements for
    compounds belonging to a held-out scaffold remain in the test set,
    regardless of target.
    """

    (
        X,
        y,
        compound_ids,
        target_names,
        scaffold_groups,
        targets,
    ) = prepare_pooled_dataset(df)

    splitter = GroupKFold(
        n_splits=n_splits,
    )

    fold_metrics = []
    per_target_metrics = []
    fold_predictions = []

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

        overall_metrics = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),

            "train_mean": y_train.mean(),
            "train_std": y_train.std(),
            "train_min": y_train.min(),
            "train_max": y_train.max(),

            "test_mean": y_test.mean(),
            "test_std": y_test.std(),
            "test_min": y_test.min(),
            "test_max": y_test.max(),

            "RMSE": mean_squared_error(
                y_test,
                predictions,
            ) ** 0.5,

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

        fold_metrics.append(
            overall_metrics
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

        fold_predictions.append(
            prediction_table
        )

        # Evaluate each target separately inside this fold
        for target in targets:
            target_mask = (
                prediction_table["target_name"]
                == target
            )

            target_data = (
                prediction_table[
                    target_mask
                ]
            )

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

            per_target_metrics.append(
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

                    "R2": (
                        r2_score(
                            observed,
                            predicted,
                        )
                    ),

                    "Spearman": (
                        target_spearman
                    ),
                }
            )

    fold_metrics_df = pd.DataFrame(
        fold_metrics
    )

    per_target_metrics_df = (
        pd.DataFrame(
            per_target_metrics
        )
    )

    predictions_df = pd.concat(
        fold_predictions,
        ignore_index=True,
    )

    return (
        fold_metrics_df,
        per_target_metrics_df,
        predictions_df,
    )