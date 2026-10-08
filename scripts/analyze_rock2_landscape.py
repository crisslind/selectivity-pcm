import numpy as np
import pandas as pd

from selectivity_pcm.paths import (
    DATA_DIR,
    RESULTS_DIR,
)

INPUT_FILE = (
    DATA_DIR
    / "rock2_panel_analysis_structures.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "rock2_panel"
    / "rock2_selectivity_landscape.csv"
)

PRIMARY_TARGET = "ROCK2"

OFF_TARGETS = [
    "ROCK1",
    "CSNK2A2",
    "AURKA",
    "STK3",
]


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # Compound × target activity matrix
    # --------------------------------------------------------

    matrix = df.pivot_table(
        index="compound_id",
        columns="target_name",
        values="pIC50",
        aggfunc="median",
    )

    print(
        f"Activity matrix: "
        f"{matrix.shape[0]} compounds × "
        f"{matrix.shape[1]} targets"
    )

    # --------------------------------------------------------
    # Require primary-target activity
    # --------------------------------------------------------

    landscape = matrix[
        matrix[PRIMARY_TARGET].notna()
    ].copy()

    # Count observed off-targets
    landscape[
        "n_offtargets_measured"
    ] = landscape[
        OFF_TARGETS
    ].notna().sum(
        axis=1
    )

    # For selectivity, require at least one measured off-target
    landscape = landscape[
        landscape[
            "n_offtargets_measured"
        ] >= 1
    ].copy()

    # --------------------------------------------------------
    # Strongest observed off-target
    #
    # Highest pIC50 = most potent off-target interaction
    # --------------------------------------------------------

    landscape[
        "strongest_offtarget_pIC50"
    ] = landscape[
        OFF_TARGETS
    ].max(
        axis=1,
        skipna=True,
    )

    landscape[
        "strongest_offtarget"
    ] = landscape[
        OFF_TARGETS
    ].idxmax(
        axis=1,
        skipna=True,
    )

    # --------------------------------------------------------
    # Selectivity margin
    #
    # positive = ROCK2 more potent
    # negative = off-target more potent
    # --------------------------------------------------------

    landscape[
        "primary_pIC50"
    ] = landscape[
        PRIMARY_TARGET
    ]

    landscape[
        "observed_selectivity_margin"
    ] = (
        landscape[
            "primary_pIC50"
        ]
        -
        landscape[
            "strongest_offtarget_pIC50"
        ]
    )

    # 1 log unit = 10-fold selectivity
    landscape[
        "observed_fold_selectivity"
    ] = np.power(
        10.0,
        landscape[
            "observed_selectivity_margin"
        ],
    )

    # --------------------------------------------------------
    # Return to ordinary dataframe
    # --------------------------------------------------------

    landscape = (
        landscape
        .reset_index()
    )

    # --------------------------------------------------------
    # Coverage summary
    # --------------------------------------------------------

    print()
    print(
        f"ROCK2 compounds with >=1 off-target: "
        f"{len(landscape)}"
    )

    print()
    print(
        "Off-target coverage:"
    )

    print(
        landscape[
            "n_offtargets_measured"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Useful selectivity categories
    # --------------------------------------------------------

    print()
    print(
        "Observed selectivity margins:"
    )

    categories = pd.Series(
        {
            "margin < 0":
                (
                    landscape[
                        "observed_selectivity_margin"
                    ] < 0
                ).sum(),

            "margin >= 0":
                (
                    landscape[
                        "observed_selectivity_margin"
                    ] >= 0
                ).sum(),

            "margin >= 0.5":
                (
                    landscape[
                        "observed_selectivity_margin"
                    ] >= 0.5
                ).sum(),

            "margin >= 1.0":
                (
                    landscape[
                        "observed_selectivity_margin"
                    ] >= 1.0
                ).sum(),

            "margin >= 2.0":
                (
                    landscape[
                        "observed_selectivity_margin"
                    ] >= 2.0
                ).sum(),
        }
    )

    print(
        categories.to_string()
    )

    # --------------------------------------------------------
    # Fully profiled subset
    # --------------------------------------------------------

    fully_profiled = landscape[
        landscape[
            "n_offtargets_measured"
        ] == len(
            OFF_TARGETS
        )
    ].copy()

    print()
    print(
        f"Fully profiled compounds: "
        f"{len(fully_profiled)}"
    )

    # --------------------------------------------------------
    # Top observed ROCK2-selective compounds
    # --------------------------------------------------------

    print()
    print(
        "Top observed ROCK2 selectivity margins:"
    )

    display_columns = [
        "compound_id",
        "primary_pIC50",
        "strongest_offtarget",
        "strongest_offtarget_pIC50",
        "observed_selectivity_margin",
        "observed_fold_selectivity",
        "n_offtargets_measured",
    ]

    print(
        landscape[
            display_columns
        ]
        .sort_values(
            [
                "observed_selectivity_margin",
                "primary_pIC50",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(20)
        .round(3)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Top compounds with >=3 measured off-targets
    #
    # More meaningful than a huge margin against only
    # one measured off-target.
    # --------------------------------------------------------

    well_profiled = landscape[
        landscape[
            "n_offtargets_measured"
        ] >= 3
    ]

    print()
    print(
        "Top ROCK2-selective compounds "
        "with >=3 measured off-targets:"
    )

    print(
        well_profiled[
            display_columns
        ]
        .sort_values(
            [
                "observed_selectivity_margin",
                "primary_pIC50",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(20)
        .round(3)
        .to_string(
            index=False
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    landscape.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()