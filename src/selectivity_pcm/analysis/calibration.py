import numpy as np
import pandas as pd


def empirical_threshold_probabilities(
    calibration_df: pd.DataFrame,
    predicted_pIC50: float,
    target_name: str,
    thresholds=(6.0, 7.0, 8.0),
    n_neighbors: int = 25,
):
    """
    Estimate empirical probabilities that observed pIC50 exceeds
    specified thresholds.

    Uses out-of-fold historical predictions from the same target.

    The calibration observations whose predicted pIC50 values are
    closest to the new prediction are used.

    Example:
        new prediction = 7.2

    Find the 100 historical OOF predictions for that target closest
    to 7.2, then calculate what fraction actually had:

        observed pIC50 >= 6
        observed pIC50 >= 7
        observed pIC50 >= 8

    This produces a local, empirical probability estimate.
    """

    required = {
        "target_name",
        "observed_pIC50",
        "predicted_pIC50",
    }

    missing = required - set(
        calibration_df.columns
    )

    if missing:
        raise ValueError(
            "Missing required calibration columns: "
            f"{sorted(missing)}"
        )

    target_df = (
        calibration_df[
            calibration_df["target_name"]
            == target_name
        ]
        .dropna(
            subset=[
                "observed_pIC50",
                "predicted_pIC50",
            ]
        )
        .copy()
    )

    if target_df.empty:
        raise ValueError(
            f"No calibration observations "
            f"available for {target_name}"
        )

    target_df[
        "prediction_distance"
    ] = np.abs(
        target_df["predicted_pIC50"]
        - predicted_pIC50
    )

    n_use = min(
        n_neighbors,
        len(target_df),
    )

    local = (
        target_df
        .nsmallest(
            n_use,
            "prediction_distance",
        )
        .copy()
    )

    probabilities = {}

    for threshold in thresholds:
        probability = (
            local["observed_pIC50"]
            >= threshold
        ).mean()

        probabilities[
            f"P_pIC50_ge_{threshold:g}"
        ] = float(
            probability
        )

    probabilities[
        "n_calibration"
    ] = len(local)

    probabilities[
        "calibration_mean_distance"
    ] = float(
        local[
            "prediction_distance"
        ].mean()
    )

    return probabilities


def empirical_selectivity_risk(
    calibration_df: pd.DataFrame,
    primary_target: str,
    off_target: str,
    predicted_primary_pIC50: float,
    predicted_offtarget_pIC50: float,
    margin_threshold: float = 1.0,
    n_neighbors: int = 25,# 100,
    min_calibration: int = 25,
):
    """
    Estimate probability that an off-target is within a specified
    potency margin of the primary target.

    With:

        margin_threshold = 1.0

    we estimate:

        P(
            primary pIC50 - off-target pIC50 <= 1.0
        )

    In other words:

        probability that the off-target is within 10-fold potency
        of the primary target.

    Calibration is based on paired out-of-fold predictions.
    """

    required = {
        "compound_id",
        "target_name",
        "observed_pIC50",
        "predicted_pIC50",
    }

    missing = required - set(
        calibration_df.columns
    )

    if missing:
        raise ValueError(
            "Missing required calibration columns: "
            f"{sorted(missing)}"
        )

    primary = (
        calibration_df[
            calibration_df["target_name"]
            == primary_target
        ][
            [
                "compound_id",
                "observed_pIC50",
                "predicted_pIC50",
            ]
        ]
        .rename(
            columns={
                "observed_pIC50":
                    "observed_primary",
                "predicted_pIC50":
                    "predicted_primary",
            }
        )
    )

    off = (
        calibration_df[
            calibration_df["target_name"]
            == off_target
        ][
            [
                "compound_id",
                "observed_pIC50",
                "predicted_pIC50",
            ]
        ]
        .rename(
            columns={
                "observed_pIC50":
                    "observed_offtarget",
                "predicted_pIC50":
                    "predicted_offtarget",
            }
        )
    )

    paired = (
        primary.merge(
            off,
            on="compound_id",
            how="inner",
        )
        .dropna()
        .copy()
    )

    if paired.empty:
        raise ValueError(
            "No paired calibration observations for "
            f"{primary_target} vs {off_target}"
        )

    paired[
        "predicted_margin"
    ] = (
        paired["predicted_primary"]
        -
        paired["predicted_offtarget"]
    )

    paired[
        "observed_margin"
    ] = (
        paired["observed_primary"]
        -
        paired["observed_offtarget"]
    )

    predicted_margin = (
        predicted_primary_pIC50
        -
        predicted_offtarget_pIC50
    )

    paired[
        "margin_distance"
    ] = np.abs(
        paired["predicted_margin"]
        -
        predicted_margin
    )


    n_available = len(paired)

    if n_available < min_calibration:
        return {
            "predicted_margin":
                float(
                    predicted_margin
                ),

            "P_offtarget_within_margin":
                np.nan,

            "margin_threshold":
                float(
                    margin_threshold
                ),

            "n_calibration":
                int(
                    n_available
                ),

            "calibration_mean_distance":
                np.nan,

            "calibration_status":
                "INSUFFICIENT_DATA",
        }

    n_use = min(
        n_neighbors,
        n_available,
    )

    local = (
        paired
        .nsmallest(
            n_use,
            "margin_distance",
        )
        .copy()
    )

    probability = (
        local["observed_margin"]
        <= margin_threshold
    ).mean()

    return {
        "predicted_margin":
            float(
                predicted_margin
            ),

        "P_offtarget_within_margin":
            float(
                probability
            ),

        "margin_threshold":
            float(
                margin_threshold
            ),

        "n_calibration":
            len(local),

        "calibration_mean_distance":
            float(
                local[
                    "margin_distance"
                ].mean()
            ),

        "calibration_status":
            "CALIBRATED",
    }