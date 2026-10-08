from pathlib import Path

import pandas as pd

from selectivity_pcm.paths import (
    DATA_DIR,
    RESULTS_DIR,
)

INPUT_FILE = (
    DATA_DIR
    / "cdk_chembl_analysis_structures.csv"
)

TARGETS = [
    "CDK1",
    "CDK2",
    "CDK4",
    "CDK6",
]

PRIMARY_TARGET = "CDK1"


def classify_observed_profile(
    primary_pIC50: float,
    worst_margin: float,
) -> str:
    """
    Ground-truth-like classification based only on
    experimentally observed potency/selectivity.

    This deliberately does NOT use model probabilities
    or calibration confidence.
    """

    potency_failure = (
        primary_pIC50 < 6.5
    )

    selectivity_failure = (
        worst_margin < 0.0
    )

    if (
        potency_failure
        and selectivity_failure
    ):
        return "DEPRIORITIZE_BOTH"

    if potency_failure:
        return "DEPRIORITIZE_POTENCY"

    if selectivity_failure:
        return "DEPRIORITIZE_SELECTIVITY"

    if (
        primary_pIC50 >= 7.0
        and worst_margin >= 1.0
    ):
        return "ADVANCE_LIKE"

    return "WATCH_LIKE"


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # Pivot experimental measurements:
    #
    # one row per compound
    # one column per target
    # --------------------------------------------------------

    matrix = (
        df.pivot_table(
            index=[
                "compound_id",
                "canonical_smiles",
            ],
            columns="target_name",
            values="pIC50",
            aggfunc="median",
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Only use compounds experimentally measured on
    # ALL four targets.
    #
    # This gives us an actual observed selectivity profile.
    # --------------------------------------------------------

    complete = matrix.dropna(
        subset=TARGETS
    ).copy()

    print(
        f"Fully profiled compounds: "
        f"{len(complete)}"
    )

    # --------------------------------------------------------
    # Calculate experimental selectivity
    # --------------------------------------------------------

    off_targets = [
        target
        for target in TARGETS
        if target != PRIMARY_TARGET
    ]

    complete[
        "strongest_observed_offtarget_pIC50"
    ] = complete[
        off_targets
    ].max(
        axis=1
    )

    complete[
        "observed_worst_offtarget"
    ] = complete[
        off_targets
    ].idxmax(
        axis=1
    )

    complete[
        "observed_worst_margin"
    ] = (
        complete[
            PRIMARY_TARGET
        ]
        -
        complete[
            "strongest_observed_offtarget_pIC50"
        ]
    )

    # --------------------------------------------------------
    # Assign validation category
    # --------------------------------------------------------

    complete[
        "observed_profile_class"
    ] = complete.apply(
        lambda row:
        classify_observed_profile(
            primary_pIC50=row[
                PRIMARY_TARGET
            ],
            worst_margin=row[
                "observed_worst_margin"
            ],
        ),
        axis=1,
    )

    print()
    print(
        "Observed profile classes:"
    )

    print(
        complete[
            "observed_profile_class"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Summary per class
    # --------------------------------------------------------

    summary = (
        complete.groupby(
            "observed_profile_class"
        )
        .agg(
            n=(
                "compound_id",
                "size",
            ),
            mean_CDK1=(
                PRIMARY_TARGET,
                "mean",
            ),
            median_CDK1=(
                PRIMARY_TARGET,
                "median",
            ),
            mean_worst_margin=(
                "observed_worst_margin",
                "mean",
            ),
            median_worst_margin=(
                "observed_worst_margin",
                "median",
            ),
        )
        .reset_index()
    )

    print()
    print(
        "Profile summary:"
    )

    print(
        summary
        .round(2)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Show strongest examples from each category
    # --------------------------------------------------------

    for category, category_df in (
        complete.groupby(
            "observed_profile_class"
        )
    ):

        print()
        print(
            f"=== {category} ==="
        )

        show = (
            category_df.sort_values(
                [
                    PRIMARY_TARGET,
                    "observed_worst_margin",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .head(10)
        )

        print(
            show[
                [
                    "compound_id",
                    "CDK1",
                    "CDK2",
                    "CDK4",
                    "CDK6",
                    "observed_worst_offtarget",
                    "observed_worst_margin",
                ]
            ]
            .round(2)
            .to_string(
                index=False
            )
        )

    output_file = Path(
        RESULTS_DIR
        / "mpo_validation_pool.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "Saved validation pool to:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()