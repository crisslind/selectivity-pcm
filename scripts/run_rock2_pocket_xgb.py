import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from selectivity_pcm.config import load_python_config
from selectivity_pcm.models.pocket_pcm import prepare_pocket_pcm_dataset
from selectivity_pcm.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

INPUT_FILE = (
    DATA_DIR
    / "rock2_panel_analysis_structures.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "rock2_panel"
)

KINASE_POCKETS = load_python_config(
    PROJECT_ROOT / "examples" / "rock2" / "kinase_pockets.py",
    "KINASE_POCKETS",
)

N_SPLITS = 5
RANDOM_STATE = 42


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded observations: {len(df)}"
    )

    (
        X,
        y,
        _compound_ids,
        target_names,
        scaffold_groups,
    ) = prepare_pocket_pcm_dataset(
        df,
        pockets=KINASE_POCKETS,
    )

    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    overall_metrics = []
    target_metrics = []

    print(
        "\nRunning XGBoost pocket PCM "
        "5-fold scaffold CV..."
    )

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

        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=(
                RANDOM_STATE
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

        rmse = (
            mean_squared_error(
                y_test,
                y_pred,
            )
            ** 0.5
        )

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

        fold_df = pd.DataFrame(
            {
                "target":
                    target_names[test_idx],
                "observed":
                    y_test,
                "predicted":
                    y_pred,
            }
        )

        for target, group in (
            fold_df.groupby("target")
        ):

            if len(group) < 2:
                continue

            target_metrics.append(
                {
                    "fold": fold,
                    "target": target,
                    "n": len(group),
                    "RMSE":
                        mean_squared_error(
                            group["observed"],
                            group["predicted"],
                        )
                        ** 0.5,
                    "MAE":
                        mean_absolute_error(
                            group["observed"],
                            group["predicted"],
                        ),
                    "R2":
                        r2_score(
                            group["observed"],
                            group["predicted"],
                        ),
                    "Spearman":
                        spearmanr(
                            group["observed"],
                            group["predicted"],
                        ).statistic,
                }
            )

    overall_df = pd.DataFrame(
        overall_metrics
    )

    target_df = pd.DataFrame(
        target_metrics
    )

    print(
        "\nOverall XGBoost CV metrics:"
    )

    print(
        overall_df
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nMean overall metrics:"
    )

    print(
        overall_df[
            [
                "RMSE",
                "MAE",
                "R2",
                "Spearman",
            ]
        ]
        .mean()
        .round(3)
        .to_string()
    )

    print(
        "\nMean metrics by target:"
    )

    print(
        target_df
        .groupby("target")[
            [
                "RMSE",
                "MAE",
                "R2",
                "Spearman",
            ]
        ]
        .mean()
        .round(3)
        .to_string()
    )

    overall_df.to_csv(
        OUTPUT_DIR
        / "pocket_xgb_scaffold_cv_metrics.csv",
        index=False,
    )

    target_df.to_csv(
        OUTPUT_DIR
        / "pocket_xgb_scaffold_cv_target_metrics.csv",
        index=False,
    )


if __name__ == "__main__":
    main()