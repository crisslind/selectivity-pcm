from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, train_test_split

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,
)


def smiles_to_array(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = morgan_generator.GetFingerprint(mol)

    return np.asarray(fp, dtype=np.uint8)


def prepare_target_dataset(
    df: pd.DataFrame,
    target_name: str,
):
    subset = df[
        df["target_name"] == target_name
    ].copy()

    features = []
    activities = []
    compound_ids = []
    smiles_list = []

    for _, row in subset.iterrows():
        fp = smiles_to_array(row["canonical_smiles"])

        if fp is None:
            continue

        features.append(fp)
        activities.append(row["pIC50"])
        compound_ids.append(row["compound_id"])
        smiles_list.append(row["canonical_smiles"])

    X = np.asarray(features)
    y = np.asarray(activities)

    return X, y, compound_ids, smiles_list


def train_qsar(
    df: pd.DataFrame,
    target_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X, y, compound_ids = prepare_target_dataset(
        df,
        target_name,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _ids_train,
        ids_test,
    ) = train_test_split(
        X,
        y,
        compound_ids,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    spearman, _ = spearmanr(
    y_test,
    predictions,
    )   

    metrics = {
    "target": target_name,
    "split": "scaffold",
    "n_total": len(y),
    "n_train": len(y_train),
    "n_test": len(y_test),
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


    prediction_table = pd.DataFrame(
        {
            "compound_id": ids_test,
            "observed_pIC50": y_test,
            "predicted_pIC50": predictions,
        }
    )

    return model, metrics, prediction_table

def get_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    return Chem.MolToSmiles(
        scaffold,
        isomericSmiles=True,
    )

def scaffold_split(
    smiles_list,
    test_fraction: float = 0.2,
    random_state: int = 42,
):
    """
    Split compounds by Bemis-Murcko scaffold.

    Compounds sharing a scaffold are kept in the same split.
    Scaffold groups are shuffled before assignment.
    """

    scaffold_groups = defaultdict(list)

    for index, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)

        if scaffold is None:
            continue

        scaffold_groups[scaffold].append(index)

    groups = list(scaffold_groups.values())

    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)

    n_total = len(smiles_list)
    n_test_target = int(n_total * test_fraction)

    train_indices = []
    test_indices = []

    for group in groups:
        if len(test_indices) + len(group) <= n_test_target:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    return train_indices, test_indices

def get_scaffold_groups(smiles_list):
    groups = []

    for smiles in smiles_list:
        scaffold = get_scaffold(smiles)

        if scaffold is None:
            scaffold = ""

        groups.append(scaffold)

    return np.asarray(groups)

def cross_validate_qsar_scaffold(
    df: pd.DataFrame,
    target_name: str,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Scaffold-grouped K-fold cross-validation.

    Each Bemis-Murcko scaffold is assigned to exactly one test fold.
    """

    X, y, compound_ids, smiles_list = prepare_target_dataset(
        df,
        target_name,
    )

    groups = get_scaffold_groups(smiles_list)

    splitter = GroupKFold(
        n_splits=n_splits,
    )

    fold_metrics = []
    fold_predictions = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        overlap = train_groups & test_groups

        print(
            f"{target_name} fold {fold}: "
            f"scaffold overlap = {len(overlap)}"
        )

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=random_state + fold,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(X_test)

        spearman, _ = spearmanr(
            y_test,
            predictions,
        )

        metrics = {
            "target": target_name,
            "split": "scaffold_cv",
            "fold": fold,
            "n_train": len(y_train),
            "n_test": len(y_test),

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

        fold_metrics.append(metrics)

        fold_predictions.append(
            pd.DataFrame(
                {
                    "compound_id": [
                        compound_ids[i]
                        for i in test_idx
                    ],
                    "fold": fold,
                    "observed_pIC50": y_test,
                    "predicted_pIC50": predictions,
                }
            )
        )

    metrics_df = pd.DataFrame(
        fold_metrics
    )

    predictions_df = pd.concat(
        fold_predictions,
        ignore_index=True,
    )

    return metrics_df, predictions_df

def train_qsar_scaffold_split(
    df: pd.DataFrame,
    target_name: str,
    test_fraction: float = 0.2,
    random_state: int = 42,
    ):
    X, y, compound_ids, smiles_list = prepare_target_dataset(
        df,
        target_name,
    )

    train_idx, test_idx = scaffold_split(
    smiles_list,
    test_fraction=test_fraction,
    random_state=random_state,
    )

    train_scaffolds = {
    get_scaffold(smiles_list[i])
    for i in train_idx
    }

    test_scaffolds = {
        get_scaffold(smiles_list[i])
        for i in test_idx
    }

    overlap = train_scaffolds & test_scaffolds

    print(
        target_name,
        "scaffold overlap:",
        len(overlap),
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    _ids_train = [
        compound_ids[i]
        for i in train_idx
    ]

    ids_test = [
        compound_ids[i]
        for i in test_idx
    ]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    spearman, _ = spearmanr(
    y_test,
    predictions,
    )

    metrics = {
    "target": target_name,
    "split": "scaffold",
    "n_total": len(y),
    "n_train": len(y_train),
    "n_test": len(y_test),

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

    prediction_table = pd.DataFrame(
        {
            "compound_id": ids_test,
            "observed_pIC50": y_test,
            "predicted_pIC50": predictions,
        }
    )

    return model, metrics, prediction_table

