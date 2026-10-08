import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from selectivity_pcm.analysis.calibration import (
    empirical_selectivity_risk,
    empirical_threshold_probabilities,
)
from selectivity_pcm.config import load_python_config
from selectivity_pcm.features.pocket_features import pocket_one_hot
from selectivity_pcm.models.qsar import smiles_to_array
from selectivity_pcm.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

TRAINING_FILE = (
    DATA_DIR
    / "rock2_panel_analysis_structures.csv"
)

EXTERNAL_FILE = (
    DATA_DIR
    / "WO2025137483_ROCK2_unknowns_15.csv"
)

CALIBRATION_FILE = (
    RESULTS_DIR
    / "rock2_panel"
    / "pocket_pcm_scaffold_cv_predictions.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "rock2_panel"
    / "external_predictions"
)

KINASE_POCKETS = load_python_config(
    PROJECT_ROOT / "examples" / "rock2" / "kinase_pockets.py",
    "KINASE_POCKETS",
)

PRIMARY_TARGET = "ROCK2"
RANDOM_STATE = 42


def build_feature(
    smiles: str,
    target: str,
) -> np.ndarray:

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


def prepare_training_data(
    df: pd.DataFrame,
):
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

    train_df = pd.read_csv(
        TRAINING_FILE
    )

    external_df = pd.read_csv(
        EXTERNAL_FILE
    )

    calibration_df = pd.read_csv(
        CALIBRATION_FILE
    )

    print(
        f"Training observations: {len(train_df)}"
    )

    print(
        f"Training compounds: "
        f"{train_df['compound_id'].nunique()}"
    )

    print(
        f"External compounds: "
        f"{len(external_df)}"
    )

    print(
        f"Calibration observations: "
        f"{len(calibration_df)}"
    )

    # ------------------------------------------------------------
    # Validate external input
    # ------------------------------------------------------------

    required_external = {
        "compound_id",
        "smiles",
    }

    missing = (
        required_external
        - set(external_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing external columns: "
            f"{sorted(missing)}"
        )

    external_df = (
        external_df
        .rename(
            columns={
                "smiles":
                    "canonical_smiles"
            }
        )
        .copy()
    )

    # ------------------------------------------------------------
    # Train final model on ALL available ChEMBL data
    # ------------------------------------------------------------

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
        "\nTraining final external-prediction model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    # ------------------------------------------------------------
    # Predict all external compounds against all pocket targets
    # ------------------------------------------------------------

    targets = sorted(
        KINASE_POCKETS.keys()
    )

    print(
        "\nTargets:",
        ", ".join(targets)
    )

    prediction_rows = []

    for row in external_df.itertuples(
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

            prediction_rows.append(
                {
                    "compound_id":
                        row.compound_id,

                    "canonical_smiles":
                        row.canonical_smiles,

                    "target_name":
                        target,

                    "predicted_pIC50":
                        prediction,
                }
            )

    predictions = pd.DataFrame(
        prediction_rows
    )

    # ------------------------------------------------------------
    # Calibrated potency threshold probabilities
    # ------------------------------------------------------------

    probability_rows = []

    for row in predictions.itertuples(
        index=False
    ):

        probabilities = (
            empirical_threshold_probabilities(
                calibration_df=calibration_df,
                predicted_pIC50=
                    row.predicted_pIC50,
                target_name=
                    row.target_name,
                thresholds=(
                    6.0,
                    7.0,
                    8.0,
                ),
                n_neighbors=100,
            )
        )

        probability_rows.append(
            {
                "compound_id":
                    row.compound_id,

                "target_name":
                    row.target_name,

                "P_pIC50_ge_6":
                    probabilities[
                        "P_pIC50_ge_6"
                    ],

                "P_pIC50_ge_7":
                    probabilities[
                        "P_pIC50_ge_7"
                    ],

                "P_pIC50_ge_8":
                    probabilities[
                        "P_pIC50_ge_8"
                    ],

                "n_calibration":
                    probabilities[
                        "n_calibration"
                    ],

                "calibration_mean_distance":
                    probabilities[
                        "calibration_mean_distance"
                    ],
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

    # ------------------------------------------------------------
    # Wide prediction matrix
    # ------------------------------------------------------------

    predicted_matrix = (
        predictions.pivot(
            index="compound_id",
            columns="target_name",
            values="predicted_pIC50",
        )
    )

    predicted_matrix.columns = [
        f"predicted_{column}"
        for column
        in predicted_matrix.columns
    ]

    comparison = (
        predicted_matrix.copy()
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
                risk = (
                    empirical_selectivity_risk(
                        calibration_df=
                            calibration_df,

                        primary_target=
                            PRIMARY_TARGET,

                        n_neighbors=25,

                        off_target=
                            target,

                        predicted_primary_pIC50=
                            primary_pred,

                        predicted_offtarget_pIC50=
                            off_pred,

                        margin_threshold=1.0,
                    )
                )

            except ValueError:
                continue

            risk_rows.append(
                {
                    "compound_id":
                        compound_id,

                    "off_target":
                        target,

                    "predicted_margin":
                        risk[
                            "predicted_margin"
                        ],

                    "P_offtarget_within_1_log":
                        risk[
                            "P_offtarget_within_margin"
                        ],

                    "n_selectivity_calibration":
                        risk[
                            "n_calibration"
                        ],

                    "calibration_mean_distance":
                        risk[
                            "calibration_mean_distance"
                        ],

                    "calibration_status":
                        risk[
                            "calibration_status"
                        ],
                }
            )

    risk_df = pd.DataFrame(
        risk_rows
    )

    # ------------------------------------------------------------
    # Add explicit predicted margins to wide table
    # ------------------------------------------------------------

    for target in targets:

        if target == PRIMARY_TARGET:
            continue

        comparison[
            f"predicted_"
            f"{PRIMARY_TARGET}_vs_"
            f"{target}"
        ] = (
            comparison[
                f"predicted_{PRIMARY_TARGET}"
            ]
            -
            comparison[
                f"predicted_{target}"
            ]
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    predictions.to_csv(
        OUTPUT_DIR
        / "external_predictions_long.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "external_predictions_wide.csv"
    )

    risk_df.to_csv(
        OUTPUT_DIR
        / "external_selectivity_risk.csv",
        index=False,
    )

    print(
        "\nExternal predictions:"
    )

    display_columns = [
        f"predicted_{PRIMARY_TARGET}",
        "predicted_ROCK1",
    ]

    print(
        comparison[
            display_columns
        ]
        .round(3)
        .to_string()
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "external_predictions_long.csv"
    )

    print(
        OUTPUT_DIR
        / "external_predictions_wide.csv"
    )

    print(
        OUTPUT_DIR
        / "external_selectivity_risk.csv"
    )


if __name__ == "__main__":
    main()