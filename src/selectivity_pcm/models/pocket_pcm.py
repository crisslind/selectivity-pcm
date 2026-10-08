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

from selectivity_pcm.features.pocket_features import pocket_one_hot
from selectivity_pcm.models.qsar import (
    get_scaffold,
    smiles_to_array,
)


def prepare_pocket_pcm_dataset(
    df: pd.DataFrame,
    pockets: dict,
):
    """
    Build ligand + KLIFS pocket features for PCM.

    Feature vector:
        Morgan fingerprint
        +
        85-position KLIFS pocket one-hot encoding

    The same target pocket vector is reused for all
    compounds measured against that target.
    """

    required_columns = {
        "compound_id",
        "target_name",
        "pIC50",
        "canonical_smiles",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if not pockets:
        raise ValueError(
            "Pocket mapping cannot be empty"
        )
    pocket_cache = {
        target: pocket_one_hot(
            data["pocket_sequence"]
        )
        for target, data in pockets.items()
    }

    features = []
    activities = []
    compound_ids = []
    target_names = []
    scaffold_groups = []

    skipped = 0

    for row in df.itertuples(index=False):
        target = row.target_name

        if target not in pocket_cache:
            raise ValueError(
                f"No KLIFS pocket defined for target: {target}"
            )

        ligand_fp = smiles_to_array(
            row.canonical_smiles
        )

        if ligand_fp is None:
            skipped += 1
            continue

        pocket_fp = pocket_cache[target]

        combined = np.concatenate(
            [
                ligand_fp,
                pocket_fp,
            ]
        )

        scaffold = get_scaffold(
            row.canonical_smiles
        )

        if scaffold is None:
            skipped += 1
            continue

        features.append(
            combined
        )

        activities.append(
            float(row.pIC50)
        )

        compound_ids.append(
            row.compound_id
        )

        target_names.append(
            target
        )

        scaffold_groups.append(
            scaffold
        )

    if not features:
        raise ValueError(
            "No usable PCM observations were generated."
        )

    X = np.asarray(
        features,
        dtype=np.float32,
    )

    y = np.asarray(
        activities,
        dtype=float,
    )

    compound_ids = np.asarray(
        compound_ids
    )

    target_names = np.asarray(
        target_names
    )

    scaffold_groups = np.asarray(
        scaffold_groups
    )

    print(
        f"Pocket PCM observations: {len(y)}"
    )

    print(
        f"Feature dimension: {X.shape[1]}"
    )

    print(
        f"Skipped observations: {skipped}"
    )

    return (
        X,
        y,
        compound_ids,
        target_names,
        scaffold_groups,
    )


def cross_validate_pocket_pcm_scaffold(
    df: pd.DataFrame,
    pockets: dict,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Evaluate pocket PCM using scaffold-grouped cross-validation.

    Scaffolds are grouped globally across all targets.
    This prevents the same chemical scaffold appearing
    in both train and test folds against different targets.
    """

    (
        X,
        y,
        compound_ids,
        target_names,
        scaffold_groups,
    ) = prepare_pocket_pcm_dataset(
        df,
        pockets=pockets,
    )

    cv = GroupKFold(
        n_splits=n_splits
    )

    overall_metrics = []
    target_metrics = []
    predictions = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
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
            random_state=(
                random_state
                + fold
            ),
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        y_pred = model.predict(
            X_test
        )

        rmse = mean_squared_error(
            y_test,
            y_pred,
        ) ** 0.5

        mae = mean_absolute_error(
            y_test,
            y_pred,
        )

        r2 = r2_score(
            y_test,
            y_pred,
        )

        spearman = spearmanr(
            y_test,
            y_pred,
        ).statistic

        overall_metrics.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Spearman": spearman,
            }
        )

        test_targets = target_names[
            test_idx
        ]

        test_compounds = compound_ids[
            test_idx
        ]

        for target in sorted(
            set(test_targets)
        ):
            mask = (
                test_targets
                == target
            )

            target_y = y_test[mask]
            target_pred = y_pred[mask]

            if len(target_y) < 2:
                continue

            target_rmse = (
                mean_squared_error(
                    target_y,
                    target_pred,
                )
                ** 0.5
            )

            target_mae = (
                mean_absolute_error(
                    target_y,
                    target_pred,
                )
            )

            target_r2 = r2_score(
                target_y,
                target_pred,
            )

            target_spearman = (
                spearmanr(
                    target_y,
                    target_pred,
                ).statistic
            )

            target_metrics.append(
                {
                    "target": target,
                    "fold": fold,
                    "n_test": len(
                        target_y
                    ),
                    "RMSE": target_rmse,
                    "MAE": target_mae,
                    "R2": target_r2,
                    "Spearman": (
                        target_spearman
                    ),
                }
            )

        for compound_id, target, observed, predicted in zip(
            test_compounds,
            test_targets,
            y_test,
            y_pred,
        ):
            predictions.append(
                {
                    "compound_id": (
                        compound_id
                    ),
                    "target_name": (
                        target
                    ),
                    "fold": fold,
                    "observed_pIC50": (
                        observed
                    ),
                    "predicted_pIC50": (
                        predicted
                    ),
                }
            )

    return (
        pd.DataFrame(
            overall_metrics
        ),
        pd.DataFrame(
            target_metrics
        ),
        pd.DataFrame(
            predictions
        ),
    )