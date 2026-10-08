import pandas as pd

from selectivity_pcm.paths import (
    RESULTS_DIR,
)

INPUT_FILE = (
    RESULTS_DIR
    / "rock2_panel"
    / "strict_ic50_raw.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "rock2_panel"
    / "high_coverage_compounds.csv"
)

TARGETS = [
    "ROCK2",
    "ROCK1",
    "CSNK2A1",
    "CSNK2A2",
    "CDC42BPB",
    "BMP2K",
    "STK3",
    "AURKA",
]


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    # One compound-target pair is enough for this coverage check.
    pairs = (
        df[
            [
                "molecule_chembl_id",
                "target_name",
            ]
        ]
        .drop_duplicates()
    )

    presence = (
        pairs.assign(
            measured=1
        )
        .pivot_table(
            index="molecule_chembl_id",
            columns="target_name",
            values="measured",
            fill_value=0,
        )
    )

    presence = presence.reindex(
        columns=TARGETS,
        fill_value=0,
    )

    presence[
        "n_targets_measured"
    ] = presence.sum(
        axis=1
    )

    high = (
        presence[
            presence[
                "n_targets_measured"
            ] >= 4
        ]
        .reset_index()
        .rename(
            columns={
                "molecule_chembl_id":
                    "compound_id"
            }
        )
        .sort_values(
            "n_targets_measured",
            ascending=False,
        )
    )

    print(
        f"Compounds measured on >=4 targets: "
        f"{len(high)}"
    )

    print()
    print(
        high.to_string(
            index=False
        )
    )

    high.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()