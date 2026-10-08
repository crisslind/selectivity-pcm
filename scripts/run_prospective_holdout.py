
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from selectivity_pcm.analysis.calibration import (
    empirical_selectivity_risk,
    empirical_threshold_probabilities,
)
from selectivity_pcm.config import load_python_config
from selectivity_pcm.features.pocket_features import pocket_one_hot
from selectivity_pcm.models.qsar import smiles_to_array
from selectivity_pcm.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

INPUT_FILE = (
    DATA_DIR
    / "rock2_panel_analysis_structures.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "rock2_panel"
    / "prospective_holdout"
)

CALIBRATION_FILE = (
    RESULTS_DIR
    / "rock2_panel"
    / "pocket_pcm_scaffold_cv_predictions.csv"
)

KINASE_POCKETS = load_python_config(
    PROJECT_ROOT / "examples" / "rock2" / "kinase_pockets.py",
    "KINASE_POCKETS",
)

PRIMARY_TARGET = "ROCK2"
N_HOLDOUT = 50
RANDOM_STATE = 42

def build_feature(
    smiles: str,
    target: str,
) -> np.ndarray:
    """
    Build Morgan + KLIFS pocket feature vector.
    """

    ligand_fp = smiles_to_array(
        smiles
    )

    if ligand_fp is None:
        raise ValueError(
            f"Could not fingerprint SMILES: {smiles}"
        )

    if target not in KINASE_POCKETS:
        raise ValueError(
            f"No KLIFS pocket found for {target}"
        )

    pocket_fp = pocket_one_hot(
        KINASE_POCKETS[target][
            "pocket_sequence"
        ]
    )

    return np.concatenate(
        [
            ligand_fp,
            pocket_fp,
        ]
    ).astype(np.float32)


def choose_holdout_compounds(
    df: pd.DataFrame,
    n_holdout: int,
    random_state: int,
) -> list[str]:
    """
    Choose compounds with a ROCK2 measurement.

    Preference is given to compounds that also have
    measurements against multiple off-targets, so that
    prospective selectivity profiles can be evaluated.
    """

    coverage = (
        df.groupby("compound_id")
        ["target_name"]
        .nunique()
        .rename("n_targets")
    )

    primary_compounds = (
        df.loc[
            df["target_name"]
            == PRIMARY_TARGET,
            ["compound_id"]
        ]
        .drop_duplicates()
        .merge(
            coverage,
            on="compound_id",
            how="left",
        )
    )

    # Prefer compounds with broad target coverage.
    #
    # 4-target profiles first,
    # then 3-target profiles,
    # then 2-target profiles if necessary.
    candidates = primary_compounds[
        primary_compounds["n_targets"] >= 3
    ].copy()

    if len(candidates) < n_holdout:
        candidates = primary_compounds[
            primary_compounds["n_targets"] >= 2
        ].copy()

    if len(candidates) < n_holdout:
        raise ValueError(
            f"Only {len(candidates)} suitable "
            f"{PRIMARY_TARGET} compounds available."
        )

    sampled = candidates.sample(
        n=n_holdout,
        random_state=random_state,
    )

    return (
        sampled["compound_id"]
        .tolist()
    )


def prepare_training_data(
    df: pd.DataFrame,
):
    """
    Convert all compound-target rows into PCM features.
    """

    X = []
    y = []

    for row in df.itertuples(
        index=False
    ):
        feature = build_feature(
            row.canonical_smiles,
            row.target_name,
        )

        X.append(
            feature
        )

        y.append(
            float(row.pIC50)
        )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            y,
            dtype=float,
        ),
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    calibration_df = pd.read_csv(
        CALIBRATION_FILE
    )

    print(
        f"Calibration observations: "
        f"{len(calibration_df)}"
    )

    print(
        "Calibration observations by target:"
    )

    print(
        calibration_df[
            "target_name"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )
    required = {
        "compound_id",
        "target_name",
        "pIC50",
        "canonical_smiles",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    holdout_ids = (
        choose_holdout_compounds(
            df,
            n_holdout=N_HOLDOUT,
            random_state=RANDOM_STATE,
        )
    )

    print(
        f"Selected {len(holdout_ids)} "
        f"prospective compounds"
    )

    train_df = df[
        ~df["compound_id"].isin(
            holdout_ids
        )
    ].copy()

    test_df = df[
        df["compound_id"].isin(
            holdout_ids
        )
    ].copy()

    print(
        f"Training observations: "
        f"{len(train_df)}"
    )

    print(
        f"Held-out observations: "
        f"{len(test_df)}"
    )

    print(
        f"Training compounds: "
        f"{train_df['compound_id'].nunique()}"
    )

    print(
        f"Held-out compounds: "
        f"{test_df['compound_id'].nunique()}"
    )

    # Important leakage check:
    # no held-out compound may occur in training.
    overlap = (
        set(
            train_df["compound_id"]
        )
        &
        set(
            test_df["compound_id"]
        )
    )

    print(
        f"Compound overlap: "
        f"{len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Holdout compounds leaked "
            "into training data."
        )

    X_train, y_train = (
        prepare_training_data(
            train_df
        )
    )

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print(
        "\nTraining prospective model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    prediction_rows = []

    # One canonical SMILES per held-out compound.
    compound_smiles = (
        test_df[
            [
                "compound_id",
                "canonical_smiles",
            ]
        ]
        .drop_duplicates(
            "compound_id"
        )
    )

    targets = sorted(
        KINASE_POCKETS.keys()
    )

    print(
        "\nPredicting held-out compounds..."
    )

    for row in compound_smiles.itertuples(
        index=False
    ):
        for target in targets:
            feature = build_feature(
                row.canonical_smiles,
                target,
            )

            prediction = float(
                model.predict(
                    feature.reshape(
                        1,
                        -1,
                    )
                )[0]
            )

            observed_rows = test_df[
                (
                    test_df[
                        "compound_id"
                    ]
                    == row.compound_id
                )
                &
                (
                    test_df[
                        "target_name"
                    ]
                    == target
                )
            ]

            if observed_rows.empty:
                observed = np.nan
            else:
                observed = float(
                    observed_rows[
                        "pIC50"
                    ].median()
                )

            prediction_rows.append(
                {
                    "compound_id":
                        row.compound_id,
                    "canonical_smiles":
                        row.canonical_smiles,
                    "target_name":
                        target,
                    "observed_pIC50":
                        observed,
                    "predicted_pIC50":
                        prediction,
                }
            )

    predictions = pd.DataFrame(
        prediction_rows
    )

    # ------------------------------------------------------------
    # Calibrated threshold probabilities
    # ------------------------------------------------------------

    probability_rows = []

    for row in predictions.itertuples(index=False):
        probabilities = empirical_threshold_probabilities(
            calibration_df=calibration_df,
            predicted_pIC50=row.predicted_pIC50,
            target_name=row.target_name,
            thresholds=(6.0, 7.0, 8.0),
            n_neighbors=100,
        )

        probability_rows.append(
            {
                "compound_id": row.compound_id,
                "target_name": row.target_name,
                "P_pIC50_ge_6": probabilities["P_pIC50_ge_6"],
                "P_pIC50_ge_7": probabilities["P_pIC50_ge_7"],
                "P_pIC50_ge_8": probabilities["P_pIC50_ge_8"],
                "n_calibration": probabilities["n_calibration"],
                "calibration_mean_distance":probabilities["calibration_mean_distance"],
            }
        )

    probability_df = pd.DataFrame(
        probability_rows
    )

    predictions = predictions.merge(
        probability_df,
        on=[
            "compound_id",
            "target_name",
        ],
        how="left",
    )

    evaluable = predictions.dropna(
        subset=[
            "observed_pIC50"
        ]
    ).copy()

    evaluable["error"] = (
        evaluable[
            "predicted_pIC50"
        ]
        -
        evaluable[
            "observed_pIC50"
        ]
    )

    evaluable[
        "absolute_error"
    ] = (
        evaluable["error"]
        .abs()
    )

    print(
        "\nOverall prospective performance:"
    )

    rmse = (
        mean_squared_error(
            evaluable[
                "observed_pIC50"
            ],
            evaluable[
                "predicted_pIC50"
            ],
        )
        ** 0.5
    )

    mae = mean_absolute_error(
        evaluable[
            "observed_pIC50"
        ],
        evaluable[
            "predicted_pIC50"
        ],
    )

    r2 = r2_score(
        evaluable[
            "observed_pIC50"
        ],
        evaluable[
            "predicted_pIC50"
        ],
    )

    print(
        f"RMSE: {rmse:.3f}"
    )
    print(
        f"MAE:  {mae:.3f}"
    )
    print(
        f"R2:   {r2:.3f}"
    )

    print(
        "\nPerformance by target:"
    )

    target_summary = []

    for target, group in (
        evaluable.groupby(
            "target_name"
        )
    ):
        if len(group) < 2:
            continue

        target_rmse = (
            mean_squared_error(
                group[
                    "observed_pIC50"
                ],
                group[
                    "predicted_pIC50"
                ],
            )
            ** 0.5
        )

        target_mae = (
            mean_absolute_error(
                group[
                    "observed_pIC50"
                ],
                group[
                    "predicted_pIC50"
                ],
            )
        )

        target_r2 = (
            r2_score(
                group[
                    "observed_pIC50"
                ],
                group[
                    "predicted_pIC50"
                ],
            )
        )

        target_summary.append(
            {
                "target":
                    target,
                "n":
                    len(group),
                "RMSE":
                    target_rmse,
                "MAE":
                    target_mae,
                "R2":
                    target_r2,
            }
        )

    target_summary = pd.DataFrame(
        target_summary
    )

    print(
        target_summary
        .round(3)
        .to_string(
            index=False
        )
    )

    observed_matrix = (
        predictions.pivot(
            index="compound_id",
            columns="target_name",
            values="observed_pIC50",
        )
    )

    predicted_matrix = (
        predictions.pivot(
            index="compound_id",
            columns="target_name",
            values="predicted_pIC50",
        )
    )

    observed_matrix.columns = [
        f"observed_{column}"
        for column
        in observed_matrix.columns
    ]

    predicted_matrix.columns = [
        f"predicted_{column}"
        for column
        in predicted_matrix.columns
    ]

    comparison = (
        observed_matrix.join(
            predicted_matrix
        )
    )


    # ------------------------------------------------------------
    # Selectivity-risk calibration
    # ------------------------------------------------------------

    risk_rows = []

    for compound_id in comparison.index:
        primary_pred = comparison.loc[
            compound_id,
            f"predicted_{PRIMARY_TARGET}",
        ]

        for target in targets:
            if target == PRIMARY_TARGET:
                continue

            off_pred = comparison.loc[
                compound_id,
                f"predicted_{target}",
            ]

            try:
                risk = empirical_selectivity_risk(
                    calibration_df=calibration_df,
                    primary_target=PRIMARY_TARGET,
                    n_neighbors=25,
                    off_target=target,
                    predicted_primary_pIC50=primary_pred,
                    predicted_offtarget_pIC50=off_pred,
                    margin_threshold=1.0,
                )
            except ValueError:
                continue

            risk_rows.append(
                {
                    "compound_id": compound_id,
                    "off_target": target,
                    "predicted_margin":
                        risk["predicted_margin"],
                    "P_offtarget_within_1_log":
                        risk["P_offtarget_within_margin"],
                    "n_selectivity_calibration":
                        risk["n_calibration"],
                        "calibration_mean_distance":
                        risk["calibration_mean_distance"],
                        "calibration_status":
                        risk["calibration_status"],
                }
            )

    risk_df = pd.DataFrame(
        risk_rows
    )

    # Add predicted selectivity margins
    # relative to the primary target.
    for target in targets:
        if target == PRIMARY_TARGET:
            continue

        comparison[
            f"predicted_"
            f"{PRIMARY_TARGET}_vs_"
            f"{target}"
        ] = (
            comparison[
                f"predicted_"
                f"{PRIMARY_TARGET}"
            ]
            -
            comparison[
                f"predicted_{target}"
            ]
        )

        observed_primary = (
            f"observed_"
            f"{PRIMARY_TARGET}"
        )

        observed_target = (
            f"observed_{target}"
        )

        if (
            observed_primary
            in comparison.columns
            and
            observed_target
            in comparison.columns
        ):
            comparison[
                f"observed_"
                f"{PRIMARY_TARGET}_vs_"
                f"{target}"
            ] = (
                comparison[
                    observed_primary
                ]
                -
                comparison[
                    observed_target
                ]
            )

    predictions.to_csv(
        OUTPUT_DIR
        / "prospective_predictions_long.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "prospective_predictions_wide.csv"
    )

    target_summary.to_csv(
        OUTPUT_DIR
        / "prospective_metrics_by_target.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "compound_id":
                holdout_ids
        }
    ).to_csv(
        OUTPUT_DIR
        / "holdout_compounds.csv",
        index=False,
    )

    train_df.to_csv(
        OUTPUT_DIR
        / "training_data.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR
        / "hidden_test_data.csv",
        index=False,
    )

    risk_df.to_csv(
    OUTPUT_DIR
    / "prospective_selectivity_risk.csv",
    index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "holdout_compounds.csv"
    )

    print(
        OUTPUT_DIR
        / "prospective_predictions_long.csv"
    )

    print(
        OUTPUT_DIR
        / "prospective_predictions_wide.csv"
    )

    print(
        OUTPUT_DIR
        / "prospective_metrics_by_target.csv"
    )
    print(
    OUTPUT_DIR
    / "prospective_selectivity_risk.csv"
    )


if __name__ == "__main__":
    main()
