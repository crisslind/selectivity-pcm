import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from selectivity_pcm.paths import RESULTS_DIR

INPUT_FILE = (
    RESULTS_DIR
    / "prospective_holdout"
    / "prospective_predictions_long.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "calibration"
)

THRESHOLDS = [
    6.0,
    7.0,
    8.0,
]

N_BINS = 5


def calibration_table(
    df: pd.DataFrame,
    threshold: float,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Build a calibration table for one pIC50 threshold.

    Example for threshold = 7:

        predicted probability bin
        mean predicted probability
        observed fraction with pIC50 >= 7
        number of compounds
    """

    probability_column = (
        f"P_pIC50_ge_{threshold:g}"
    )

    data = df.dropna(
        subset=[
            "observed_pIC50",
            probability_column,
        ]
    ).copy()

    data["outcome"] = (
        data["observed_pIC50"]
        >= threshold
    ).astype(int)

    # Fixed probability bins:
    #
    # 0.0-0.2
    # 0.2-0.4
    # ...
    # 0.8-1.0
    bins = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    data["probability_bin"] = pd.cut(
        data[probability_column],
        bins=bins,
        include_lowest=True,
        duplicates="drop",
    )

    table = (
        data.groupby(
            "probability_bin",
            observed=True,
        )
        .agg(
            n=("outcome", "size"),
            mean_predicted=(
                probability_column,
                "mean",
            ),
            observed_fraction=(
                "outcome",
                "mean",
            ),
        )
        .reset_index()
    )

    table["threshold"] = threshold

    return table


def evaluate_threshold(
    df: pd.DataFrame,
    threshold: float,
):
    """
    Calculate Brier score for one threshold.
    """

    probability_column = (
        f"P_pIC50_ge_{threshold:g}"
    )

    data = df.dropna(
        subset=[
            "observed_pIC50",
            probability_column,
        ]
    ).copy()

    observed = (
        data["observed_pIC50"]
        >= threshold
    ).astype(int)

    predicted = data[
        probability_column
    ]

    brier = brier_score_loss(
        observed,
        predicted,
    )

    return {
        "threshold": threshold,
        "n": len(data),
        "positive_rate": float(
            observed.mean()
        ),
        "mean_predicted_probability":
            float(
                predicted.mean()
            ),
        "brier_score": float(
            brier
        ),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded {len(df)} "
        f"compound-target predictions"
    )

    overall_metrics = []
    calibration_tables = []

    for threshold in THRESHOLDS:
        metrics = evaluate_threshold(
            df,
            threshold,
        )

        overall_metrics.append(
            metrics
        )

        table = calibration_table(
            df,
            threshold,
            n_bins=N_BINS,
        )

        calibration_tables.append(
            table
        )

    overall_metrics = pd.DataFrame(
        overall_metrics
    )

    calibration_tables = pd.concat(
        calibration_tables,
        ignore_index=True,
    )

    print(
        "\nOverall calibration:"
    )

    print(
        overall_metrics
        .round(3)
        .to_string(
            index=False
        )
    )

    print(
        "\nCalibration bins:"
    )

    print(
        calibration_tables
        .round(3)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Per-target Brier scores
    # --------------------------------------------------------

    target_rows = []

    for target, target_df in df.groupby(
        "target_name"
    ):
        for threshold in THRESHOLDS:
            metrics = evaluate_threshold(
                target_df,
                threshold,
            )

            metrics[
                "target_name"
            ] = target

            target_rows.append(
                metrics
            )

    target_metrics = pd.DataFrame(
        target_rows
    )

    print(
        "\nCalibration by target:"
    )

    print(
        target_metrics[
            [
                "target_name",
                "threshold",
                "n",
                "positive_rate",
                "mean_predicted_probability",
                "brier_score",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    overall_metrics.to_csv(
        OUTPUT_DIR
        / "threshold_calibration_overall.csv",
        index=False,
    )

    target_metrics.to_csv(
        OUTPUT_DIR
        / "threshold_calibration_by_target.csv",
        index=False,
    )

    calibration_tables.to_csv(
        OUTPUT_DIR
        / "threshold_calibration_bins.csv",
        index=False,
    )

    print(
        "\nSaved calibration results to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()