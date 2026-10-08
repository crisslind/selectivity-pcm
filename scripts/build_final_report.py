import numpy as np
import pandas as pd

from selectivity_pcm.paths import RESULTS_DIR

# ============================================================
# Configuration
# ============================================================
PREDICTIONS_FILE = (
    RESULTS_DIR
    / "rock2_panel"
    / "prospective_holdout"
    / "prospective_predictions_long.csv"
)

RISK_FILE = (
    RESULTS_DIR
    / "rock2_panel"
    / "prospective_holdout"
    / "prospective_selectivity_risk.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "rock2_panel"
    / "prospective_holdout"
    / "final_prospective_report.csv"
)

PRIMARY_TARGET = "ROCK2"
CRITICAL_OFFTARGETS = [
    "ROCK1",
]

# ------------------------------------------------------------
# MPO weights
#
# Total = 100%
# ------------------------------------------------------------

WEIGHTS = {
    "primary_potency": 0.30,
    "primary_probability": 0.20,
    "worst_margin": 0.25,
    "offtarget_risk": 0.20,
    "confidence": 0.05,
}


# ============================================================
# Scoring functions
# ============================================================

def clipped_linear(
    value: float,
    low: float,
    high: float,
) -> float:
    """
    Convert a numeric value to a 0-100 score.

    low  -> 0
    high -> 100

    Values outside the range are clipped.
    """

    if pd.isna(value):
        return np.nan

    score = (
        (value - low)
        /
        (high - low)
        * 100.0
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


def confidence_score(
    distance: float,
) -> float:
    """
    Convert calibration distance to a 0-100 support score.

    Smaller distance = calibration examples were closer
    to the current prediction.

    This is a v1 heuristic and should remain visible
    rather than being interpreted as a learned probability.
    """

    if pd.isna(distance):
        return 0.0

    if distance <= 0.15:
        return 100.0

    if distance <= 0.40:
        # 100 -> 70
        return float(
            np.interp(
                distance,
                [0.15, 0.40],
                [100.0, 70.0],
            )
        )

    if distance <= 0.80:
        # 70 -> 40
        return float(
            np.interp(
                distance,
                [0.40, 0.80],
                [70.0, 40.0],
            )
        )

    if distance <= 1.50:
        # 40 -> 0
        return float(
            np.interp(
                distance,
                [0.80, 1.50],
                [40.0, 0.0],
            )
        )

    return 0.0


def confidence_label(
    distance: float,
) -> str:
    """
    Human-readable calibration support label.
    """

    if pd.isna(distance):
        return "UNKNOWN"

    if distance <= 0.15:
        return "HIGH"

    if distance <= 0.40:
        return "MEDIUM"

    return "LOW"


# ============================================================
# Recommendation logic
# ============================================================

def make_recommendation(
    mpo_score: float,
    primary_pred: float,
    worst_margin: float,
    worst_margin_target: str,
    max_risk: float,
    max_risk_target: str,
    confidence: str,
    n_critical_insufficient: int,
    critical_insufficient_targets: list[str],
):
    """
    Assign recommendation using both:

    1. continuous MPO score
    2. hard selectivity/potency rules

    Returns:
        recommendation
        recommendation_type
        recommendation_reason
    """

    reasons = []

    potency_failure = (
        primary_pred < 6.5
    )

    selectivity_failure = (
        worst_margin < 0
        or max_risk >= 0.85
    )

    # --------------------------------------------------------
    # Hard-stop rules
    # --------------------------------------------------------

    if potency_failure:
        reasons.append(
            f"{PRIMARY_TARGET} predicted pIC50 "
            f"is only {primary_pred:.2f}"
        )

    if worst_margin < 0:
        reasons.append(
            f"{worst_margin_target} is predicted "
            f"more potent than {PRIMARY_TARGET} "
            f"(margin {worst_margin:.2f})"
        )

    if max_risk >= 0.85:
        reasons.append(
            f"high selectivity risk for "
            f"{max_risk_target} "
            f"({max_risk:.0%} probability of being "
            f"within 1 log unit)"
        )

    if potency_failure or selectivity_failure:

        recommendation = (
            "DEPRIORITIZE"
        )

        if (
            potency_failure
            and selectivity_failure
        ):
            recommendation_type = (
                "DEPRIORITIZE_BOTH"
            )

        elif potency_failure:
            recommendation_type = (
                "DEPRIORITIZE_POTENCY"
            )

        else:
            recommendation_type = (
                "DEPRIORITIZE_SELECTIVITY"
            )

        return (
            recommendation,
            recommendation_type,
            "; ".join(reasons),
        )
    # --------------------------------------------------------
    # Insufficient selectivity calibration
    # --------------------------------------------------------

    if n_critical_insufficient > 0:
        reasons.append(
            "insufficient calibration for critical off-targets: "
            + ", ".join(
                critical_insufficient_targets
            )
        )

        return (
            "WATCH",
            "WATCH_INSUFFICIENT_DATA",
            "; ".join(reasons),
        )
    
    # --------------------------------------------------------
    # ADVANCE
    # --------------------------------------------------------

    advance = (
        mpo_score >= 70
        and primary_pred >= 7.0
        and worst_margin >= 1.0
        and max_risk <= 0.50
        and confidence != "LOW"
    )

    if advance:

        recommendation = "ADVANCE"
        recommendation_type = "ADVANCE"

        reason = (
            f"{PRIMARY_TARGET} predicted pIC50 "
            f"{primary_pred:.2f}; "
            f"worst off-target margin "
            f"{worst_margin:.2f} vs "
            f"{worst_margin_target}; "
            f"maximum off-target risk "
            f"{max_risk:.0%}; "
            f"{confidence.lower()} calibration support"
        )

        return (
            recommendation,
            recommendation_type,
            reason,
        )

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    recommendation = "WATCH"

    if confidence == "LOW":
        recommendation_type = (
            "WATCH_UNCERTAINTY"
        )

    else:
        recommendation_type = (
            "WATCH_SELECTIVITY"
        )

    if primary_pred < 7.0:
        reasons.append(
            f"{PRIMARY_TARGET} potency is moderate "
            f"({primary_pred:.2f})"
        )

    if worst_margin < 1.0:
        reasons.append(
            f"closest off-target is "
            f"{worst_margin_target} "
            f"with margin {worst_margin:.2f}"
        )

    if max_risk > 0.50:
        reasons.append(
            f"{max_risk_target} selectivity risk "
            f"is {max_risk:.0%}"
        )

    if confidence == "LOW":
        reasons.append(
            "selectivity calibration support is low"
        )

    if not reasons:
        reasons.append(
            f"MPO score {mpo_score:.1f}"
        )

    return (
        recommendation,
        recommendation_type,
        "; ".join(reasons),
    )


# ============================================================
# Main
# ============================================================

def main():
    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    risk = pd.read_csv(
        RISK_FILE
    )

    targets = sorted(
        predictions[
            "target_name"
        ].unique()
    )

    if PRIMARY_TARGET not in targets:
        raise ValueError(
            f"Primary target {PRIMARY_TARGET} "
            f"not found in prediction file."
        )

    off_targets = [
        target
        for target in targets
        if target != PRIMARY_TARGET
    ]

    critical_offtargets = [
        target
        for target in off_targets
        if target in CRITICAL_OFFTARGETS
    ]

    profile_targets = [
        target
        for target in off_targets
        if target not in CRITICAL_OFFTARGETS
    ]

    print(
        f"Primary target: {PRIMARY_TARGET}"
    )

    print(
        "Off-targets:",
        ", ".join(off_targets),
    )

    unknown_critical = (
    set(CRITICAL_OFFTARGETS)
    - set(off_targets)
    )

    if unknown_critical:        
        raise ValueError(
            "Critical off-targets not found in panel: "
            f"{sorted(unknown_critical)}"
        )

    print(
        "Critical off-targets:",
        ", ".join(critical_offtargets),
    )

    print(
        "Profile targets:",
        ", ".join(profile_targets),
    )

    print(
        f"Compounds: "
        f"{predictions['compound_id'].nunique()}"
    )

    final_rows = []

    for compound_id, compound_df in (
        predictions.groupby(
            "compound_id"
        )
    ):
        compound_df = (
            compound_df
            .set_index(
                "target_name"
            )
        )

        if PRIMARY_TARGET not in compound_df.index:
            continue

        primary_row = compound_df.loc[
            PRIMARY_TARGET
        ]

        primary_pred = float(
            primary_row[
                "predicted_pIC50"
            ]
        )

        primary_p7 = float(
            primary_row[
                "P_pIC50_ge_7"
            ]
        )

        primary_p8 = float(
            primary_row[
                "P_pIC50_ge_8"
            ]
        )

        # ----------------------------------------------------
        # Base output
        # ----------------------------------------------------

        row = {
            "compound_id":
                compound_id,

            "canonical_smiles":
                primary_row[
                    "canonical_smiles"
                ],
        }

        # ----------------------------------------------------
        # Add target-specific predictions
        # ----------------------------------------------------

        target_distances = []

        for target in targets:
            if target not in compound_df.index:
                continue

            target_row = (
                compound_df.loc[
                    target
                ]
            )

            row[
                f"{target}_predicted_pIC50"
            ] = target_row[
                "predicted_pIC50"
            ]

            row[
                f"{target}_P_ge_6"
            ] = target_row[
                "P_pIC50_ge_6"
            ]

            row[
                f"{target}_P_ge_7"
            ] = target_row[
                "P_pIC50_ge_7"
            ]

            row[
                f"{target}_P_ge_8"
            ] = target_row[
                "P_pIC50_ge_8"
            ]

            row[
                f"{target}_calibration_distance"
            ] = target_row[
                "calibration_mean_distance"
            ]

            target_distances.append(
                float(
                    target_row[
                        "calibration_mean_distance"
                    ]
                )
            )

            # Retrospective validation only.
            # For truly prospective compounds this can simply be absent.
            if (
                "observed_pIC50"
                in target_row.index
                and
                not pd.isna(
                    target_row[
                        "observed_pIC50"
                    ]
                )
            ):
                row[
                    f"{target}_observed_pIC50"
                ] = target_row[
                    "observed_pIC50"
                ]

        # ----------------------------------------------------
        # Selectivity risk information
        # ----------------------------------------------------

        compound_risk = risk[
            risk["compound_id"]
            == compound_id
        ].copy()

        margins = {}
        risks = {}
        risk_distances = {}
        calibration_statuses = {}
        insufficient_targets = []

        for off_target in off_targets:

            risk_row = compound_risk[
                compound_risk[
                    "off_target"
                ]
                == off_target
            ]

            if risk_row.empty:
                continue

            risk_row = (
                risk_row.iloc[0]
            )

            margin = float(
                risk_row[
                    "predicted_margin"
                ]
            )

            calibration_status = (
                risk_row[
                    "calibration_status"
                ]
            )

            calibration_statuses[
                off_target
            ] = calibration_status

            margins[
                off_target
            ] = margin

            row[
                f"{PRIMARY_TARGET}_vs_{off_target}_margin"
            ] = margin

            row[
                f"{off_target}_selectivity_calibration_status"
            ] = calibration_status

            row[
                f"{off_target}_n_selectivity_calibration"
            ] = risk_row[
                "n_selectivity_calibration"
            ]

            if calibration_status == "CALIBRATED":

                off_risk = float(
                    risk_row[
                        "P_offtarget_within_1_log"
                    ]
                )

                distance = float(
                    risk_row[
                        "calibration_mean_distance"
                    ]
                )

                risks[
                    off_target
                ] = off_risk

                risk_distances[
                    off_target
                ] = distance

                row[
                    f"{off_target}_within_1log_risk"
                ] = off_risk

                row[
                    f"{off_target}_selectivity_calibration_distance"
                ] = distance

            else:

                insufficient_targets.append(
                    off_target
                )

                row[
                    f"{off_target}_within_1log_risk"
                ] = np.nan

                row[
                    f"{off_target}_selectivity_calibration_distance"
                ] = np.nan

        if not margins:
            continue

        # ----------------------------------------------------
        # Overall off-target summary
        # ----------------------------------------------------

        row[
            "n_offtargets_calibrated"
        ] = len(risks)

        row[
            "n_offtargets_insufficient"
        ] = len(insufficient_targets)

        row[
            "insufficient_calibration_targets"
        ] = ",".join(
            insufficient_targets
        )

        # ----------------------------------------------------
        # Critical vs profile target split
        # ----------------------------------------------------

        critical_margins = {
            target: margin
            for target, margin
            in margins.items()
            if target
            in critical_offtargets
        }

        profile_margins = {
            target: margin
            for target, margin
            in margins.items()
            if target
            in profile_targets
        }

        critical_risks = {
            target: value
            for target, value
            in risks.items()
            if target
            in critical_offtargets
        }

        profile_risks = {
            target: value
            for target, value
            in risks.items()
            if target
            in profile_targets
        }

        critical_insufficient = [
            target
            for target
            in insufficient_targets
            if target
            in critical_offtargets
        ]

        profile_insufficient = [
            target
            for target
            in insufficient_targets
            if target
            in profile_targets
        ]

        row[
            "critical_offtargets"
        ] = ",".join(
            critical_offtargets
        )

        row[
            "profile_targets"
        ] = ",".join(
            profile_targets
        )

        row[
            "n_critical_offtargets"
        ] = len(
            critical_offtargets
        )

        row[
            "n_profile_targets"
        ] = len(
            profile_targets
        )

        row[
            "n_critical_calibrated"
        ] = len(
            critical_risks
        )

        row[
            "n_profile_calibrated"
        ] = len(
            profile_risks
        )

        row[
            "critical_insufficient_targets"
        ] = ",".join(
            critical_insufficient
        )

        row[
            "profile_insufficient_targets"
        ] = ",".join(
            profile_insufficient
        )

        # ----------------------------------------------------
        # Overall worst/selectivity-limiting off-target
        # ----------------------------------------------------

        worst_margin_target = min(
            margins,
            key=margins.get,
        )

        worst_margin = margins[
            worst_margin_target
        ]

        if risks:

            max_risk_target = max(
                risks,
                key=risks.get,
            )

            max_risk = risks[
                max_risk_target
            ]

        else:

            max_risk_target = None
            max_risk = np.nan

        row[
            "worst_offtarget"
        ] = worst_margin_target

        row[
            "worst_selectivity_margin"
        ] = worst_margin

        row[
            "highest_risk_offtarget"
        ] = max_risk_target

        row[
            "max_offtarget_within_1log_risk"
        ] = max_risk

        # ----------------------------------------------------
        # Critical off-target summary
        # ----------------------------------------------------

        if critical_margins:

            worst_critical_target = min(
                critical_margins,
                key=critical_margins.get,
            )

            worst_critical_margin = (
                critical_margins[
                    worst_critical_target
                ]
            )

        else:

            worst_critical_target = None
            worst_critical_margin = np.nan

        if critical_risks:

            max_critical_risk_target = max(
                critical_risks,
                key=critical_risks.get,
            )

            max_critical_risk = (
                critical_risks[
                    max_critical_risk_target
                ]
            )

        else:

            max_critical_risk_target = None
            max_critical_risk = np.nan

        row[
            "worst_critical_offtarget"
        ] = worst_critical_target

        row[
            "worst_critical_margin"
        ] = worst_critical_margin

        row[
            "highest_risk_critical_offtarget"
        ] = max_critical_risk_target

        row[
            "max_critical_offtarget_risk"
        ] = max_critical_risk

        # ----------------------------------------------------
        # Profile-target summary
        # ----------------------------------------------------

        if profile_margins:

            worst_profile_target = min(
                profile_margins,
                key=profile_margins.get,
            )

            worst_profile_margin = (
                profile_margins[
                    worst_profile_target
                ]
            )

        else:

            worst_profile_target = None
            worst_profile_margin = np.nan

        if profile_risks:

            max_profile_risk_target = max(
                profile_risks,
                key=profile_risks.get,
            )

            max_profile_risk = (
                profile_risks[
                    max_profile_risk_target
                ]
            )

        else:

            max_profile_risk_target = None
            max_profile_risk = np.nan

        row[
            "worst_profile_target"
        ] = worst_profile_target

        row[
            "worst_profile_margin"
        ] = worst_profile_margin

        row[
            "highest_risk_profile_target"
        ] = max_profile_risk_target

        row[
            "max_profile_target_risk"
        ] = max_profile_risk

        # ----------------------------------------------------
        # Dynamic profile status / comment
        # ----------------------------------------------------

        profile_comments = []

        profile_liability_targets = [
            target
            for target, margin
            in profile_margins.items()
            if margin < 0
        ]

        profile_close_targets = [
            target
            for target, margin
            in profile_margins.items()
            if 0 <= margin < 1.0
        ]

        profile_high_risk_targets = [
            target
            for target, risk_value
            in profile_risks.items()
            if risk_value >= 0.85
        ]

        if profile_liability_targets:
            profile_comments.append(
                "predicted more potent than "
                f"{PRIMARY_TARGET}: "
                + ", ".join(
                    profile_liability_targets
                )
            )

        if profile_close_targets:
            profile_comments.append(
                "predicted within 1 log of "
                f"{PRIMARY_TARGET}: "
                + ", ".join(
                    profile_close_targets
                )
            )

        if profile_high_risk_targets:
            profile_comments.append(
                "high calibrated selectivity risk: "
                + ", ".join(
                    profile_high_risk_targets
                )
            )

        if profile_insufficient:
            profile_comments.append(
                "insufficient calibration: "
                + ", ".join(
                    profile_insufficient
                )
            )

        has_profile_liability = (
            bool(profile_liability_targets)
            or bool(profile_high_risk_targets)
        )

        has_profile_insufficient = bool(
            profile_insufficient
        )

        if (
            has_profile_liability
            and has_profile_insufficient
        ):
            profile_status = (
                "PROFILE_LIABILITY_AND_INSUFFICIENT_DATA"
            )

        elif has_profile_liability:
            profile_status = (
                "PROFILE_LIABILITY"
            )

        elif has_profile_insufficient:
            profile_status = (
                "PROFILE_INSUFFICIENT_DATA"
            )

        elif profile_close_targets:
            profile_status = (
                "PROFILE_CLOSE"
            )

        else:
            profile_status = (
                "PROFILE_OK"
            )

        row[
            "profile_status"
        ] = profile_status

        row[
            "profile_comment"
        ] = (
            "; ".join(
                profile_comments
            )
            if profile_comments
            else "no major profile liabilities identified"
        )


        # ----------------------------------------------------
        # Selectivity liability classification
        # ----------------------------------------------------

        row[
            "selectivity_liability"
        ] = worst_margin_target

        if worst_margin < 0:
            liability_type = (
                "OFFTARGET_MORE_POTENT"
            )

        elif worst_margin < 0.5:
            liability_type = (
                "SEVERE_SELECTIVITY_RISK"
            )

        elif worst_margin < 1.0:
            liability_type = (
                "MODERATE_SELECTIVITY_RISK"
            )

        else:
            liability_type = (
                "LOW_SELECTIVITY_RISK"
            )

        row[
            "selectivity_liability_type"
        ] = liability_type

        # ----------------------------------------------------
        # Overall calibration support
        #
        # We deliberately take the worst distance because
        # selectivity decisions are limited by the weakest
        # supported target comparison.
        # ----------------------------------------------------

        all_distances = (
            target_distances
            +
            list(
                risk_distances.values()
            )
        )

        median_distance = float(
            np.median(
                all_distances
            )
        )

        primary_distance = float(
            primary_row[
                "calibration_mean_distance"
            ]
        )

        worst_offtarget_distance = max(
            risk_distances.values()
        )

        primary_confidence = (
            confidence_label(
                primary_distance
            )
        )

        offtarget_confidence = (
            confidence_label(
                worst_offtarget_distance
            )
        )

        overall_confidence = (
            confidence_label(
                median_distance
            )
        )

        support_score = (
            confidence_score(
                median_distance
            )
        )

        row[
            "primary_calibration_distance"
        ] = primary_distance

        row[
            "worst_offtarget_calibration_distance"
        ] = worst_offtarget_distance

        row[
            "median_calibration_distance"
        ] = median_distance

        row[
            "primary_confidence"
        ] = primary_confidence

        row[
            "offtarget_confidence"
        ] = offtarget_confidence

        row[
            "confidence"
        ] = overall_confidence

        # ----------------------------------------------------
        # MPO components
        # ----------------------------------------------------

        # Primary potency:
        #
        # pIC50 6 -> score 0
        # pIC50 8 -> score 100
        #
        potency_component = (
            clipped_linear(
                primary_pred,
                low=6.0,
                high=8.0,
            )
        )

        # Primary probability:
        #
        # Mostly driven by probability of >=7,
        # with a bonus for probability of >=8.
        #
        probability_component = (
            (
                0.75
                * primary_p7
            )
            +
            (
                0.25
                * primary_p8
            )
        ) * 100.0

        # Worst selectivity margin:
        #
        # 0 log selectivity -> 0
        # 1 log selectivity -> 50
        # 2 log selectivity -> 100
        #
        margin_component = (
            clipped_linear(
                worst_critical_margin,
                low=0.0,
                high=2.0,
            )
        )

        # Off-target risk:
        #
        # risk 0 -> score 100
        # risk 1 -> score 0
        #
        risk_component = (
            1.0
            - max_critical_risk
        ) * 100.0

        # Calibration support
        confidence_component = (
            support_score
        )

        # ----------------------------------------------------
        # Final weighted MPO
        # ----------------------------------------------------

        mpo_score = (
            potency_component
            * WEIGHTS[
                "primary_potency"
            ]
            +
            probability_component
            * WEIGHTS[
                "primary_probability"
            ]
            +
            margin_component
            * WEIGHTS[
                "worst_margin"
            ]
            +
            risk_component
            * WEIGHTS[
                "offtarget_risk"
            ]
            +
            confidence_component
            * WEIGHTS[
                "confidence"
            ]
        )

        mpo_score = float(
            np.clip(
                mpo_score,
                0.0,
                100.0,
            )
        )

        row[
            "MPO_primary_potency"
        ] = potency_component

        row[
            "MPO_primary_probability"
        ] = probability_component

        row[
            "MPO_worst_margin"
        ] = margin_component

        row[
            "MPO_offtarget_risk"
        ] = risk_component

        row[
            "MPO_confidence"
        ] = confidence_component

        row[
            "selectivity_MPO"
        ] = mpo_score

        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        (
            recommendation,
            recommendation_type,
            reason,
        ) = make_recommendation(
            mpo_score=mpo_score,
            primary_pred=primary_pred,
            worst_margin=worst_critical_margin,
            worst_margin_target=worst_critical_target,
            max_risk=max_critical_risk,
            max_risk_target=max_critical_risk_target,
            confidence=overall_confidence,
            n_critical_insufficient=len(critical_insufficient),
            critical_insufficient_targets=(critical_insufficient),
        )

        row[
            "recommendation"
        ] = recommendation
        
        row[
            "recommendation_type"
        ] = recommendation_type

        row[
            "recommendation_reason"
        ] = reason

        final_rows.append(
            row
        )

    final = pd.DataFrame(
        final_rows
    )

    # --------------------------------------------------------
    # MPO ranking
    #
    # Rank is relative to the compounds in the current batch.
    #
    # Ranking priority:
    #   1. higher Selectivity MPO
    #   2. higher worst selectivity margin
    #   3. lower maximum off-target risk
    #   4. higher primary-target predicted potency
    #
    # Recommendation does NOT affect MPO rank.
    # --------------------------------------------------------

    ranked = (
        final.sort_values(
            [
                "selectivity_MPO",
                "worst_selectivity_margin",
                "max_offtarget_within_1log_risk",
                f"{PRIMARY_TARGET}_predicted_pIC50",
            ],
            ascending=[
                False,
                False,
                True,
                False,
            ],
        )
        .reset_index()
    )

    ranked[
        "MPO_rank"
    ] = np.arange(
        1,
        len(ranked) + 1,
    )

    rank_map = dict(
        zip(
            ranked["index"],
            ranked["MPO_rank"],
        )
    )

    final[
        "MPO_rank"
    ] = final.index.map(
        rank_map
    )


    # --------------------------------------------------------
    # Final report ordering
    #
    # Show ADVANCE first, then WATCH, then DEPRIORITIZE.
    # Within each recommendation group, show best MPO rank first.
    # --------------------------------------------------------

    recommendation_order = {
        "ADVANCE": 0,
        "WATCH": 1,
        "DEPRIORITIZE": 2,
    }

    final[
        "_recommendation_order"
    ] = final[
        "recommendation"
    ].map(
        recommendation_order
    )

    final = (
        final.sort_values(
            [
                "_recommendation_order",
                "MPO_rank",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .drop(
            columns=[
                "_recommendation_order"
            ]
        )
        .reset_index(
            drop=True
        )
    ).reset_index(
                drop=True
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("Recommendation summary:")

    print(
        final[
            "recommendation"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Top compounds:")

    summary_columns = [
        "compound_id",
        f"{PRIMARY_TARGET}_predicted_pIC50",

        *[
            f"{target}_predicted_pIC50"
            for target in CRITICAL_OFFTARGETS
        ],

        "worst_critical_offtarget",
        "worst_critical_margin",
        "max_critical_offtarget_risk",

        "worst_profile_target",
        "worst_profile_margin",
        "max_profile_target_risk",
        "profile_insufficient_targets",
        "profile_status",
        "profile_comment",
        "primary_confidence",
        "offtarget_confidence",
        "confidence",
        "MPO_rank",
        "selectivity_MPO",
        "recommendation",
        "recommendation_type",
    ]

    print(
        final[
            summary_columns
        ]
        .head(20)
        .round(3)
        .to_string(
            index=False
        )
    )

    print()
    print(f"\nSaved to {OUTPUT_FILE}")



if __name__ == "__main__":
    main()