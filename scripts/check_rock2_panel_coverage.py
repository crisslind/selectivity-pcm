import pandas as pd

from selectivity_pcm.paths import (
    RESULTS_DIR,
)

# ============================================================
# Panel definition
# ============================================================

# UNIPROT_TARGETS = {
#     "ROCK2": "O75116",   # primary
#     "ROCK1": "Q13464",   # close isoform
#     "PRKCA": "P17252",   # AGC-family off-target
#     "RSK1": "Q15418",    # known ROCK-inhibitor liability class
#     "AURKA": "O14965",   # more distant kinase / broader selectivity test
# }

TARGETS = {
    "ROCK2": {
        "uniprot": "O75116",
        "chembl": "CHEMBL2973",
    },
    "ROCK1": {
        "uniprot": "Q13464",
        "chembl": "CHEMBL3231",
    },
    "CSNK2A1": {
        "uniprot": "P68400",
        "chembl": "CHEMBL3629",
    },
    "STK3": {
        "uniprot": "Q13188",
        "chembl": "CHEMBL5408",
    },
    "AURKA": {
        "uniprot": "O14965",
        "chembl": "CHEMBL4722",
    },
}


OUTPUT_DIR = (
    RESULTS_DIR
    / "rock2_panel"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR = (
    OUTPUT_DIR
    / "cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# Strict biochemical IC50 data
# ============================================================

def fetch_strict_ic50(
    target_name: str,
    target_chembl_id: str,
):
    """
    Fetch exact biochemical single-protein IC50 measurements
    from the ChEMBL REST API.

    Uses smaller pages plus retry/backoff because the EBI
    endpoint can occasionally time out on larger requests.
    """

    import time

    import requests

    base_url = (
        "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    )

    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": "IC50",
        "limit": 200,
    }

    records = []

    url = base_url
    page = 1

    session = requests.Session()

    while url is not None:

        response = None

        for attempt in range(1, 6):

            try:

                response = session.get(
                    url,
                    params=params,
                    timeout=(10, 120),
                )

                response.raise_for_status()

                break

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
            ) as exc:

                if attempt == 5:
                    raise

                wait_seconds = attempt * 5

                print(
                    f"  {target_name}: page {page} failed "
                    f"(attempt {attempt}/5): {exc}"
                )

                print(
                    f"  Retrying in {wait_seconds} s..."
                )

                time.sleep(
                    wait_seconds
                )

        payload = response.json()

        page_records = payload.get(
            "activities",
            []
        )

        records.extend(
            page_records
        )

        print(
            f"  {target_name}: page {page}, "
            f"{len(page_records)} records "
            f"({len(records)} total)"
        )

        next_url = (
            payload
            .get(
                "page_meta",
                {}
            )
            .get(
                "next"
            )
        )

        if next_url:

            if next_url.startswith(
                "http"
            ):
                url = next_url

            else:
                url = (
                    "https://www.ebi.ac.uk"
                    + next_url
                )

            # The next URL already contains its query parameters.
            params = None

        else:
            url = None

        page += 1

    df = pd.DataFrame(
        records
    )

    if df.empty:
        return df

    # --------------------------------------------------------
    # Strict biochemical single-protein assay
    # --------------------------------------------------------

    df = df[
        (
            df["assay_type"] == "B"
        )
        &
        (
            df["bao_label"]
            == "single protein format"
        )
    ].copy()

    # --------------------------------------------------------
    # Exact measurements only
    # --------------------------------------------------------

    df = df[
        df["standard_relation"] == "="
    ].copy()

    df[
        "standard_value"
    ] = pd.to_numeric(
        df[
            "standard_value"
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "molecule_chembl_id",
            "standard_value",
        ]
    )

    df[
        "target_name"
    ] = target_name

    return df


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Resolve targets
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Target definitions
    #
    # ChEMBL IDs are fixed here because they have already
    # been resolved/verified. This avoids depending on the
    # ChEMBL Python client's /spore schema endpoint.
    # --------------------------------------------------------

    resolved_targets = []

    print(
        "Using fixed ChEMBL target mappings..."
    )

    for target_name, info in TARGETS.items():

        target = {
            "target_name":
                target_name,

            "uniprot_accession":
                info["uniprot"],

            "target_chembl_id":
                info["chembl"],
        }

        resolved_targets.append(
            target
        )

        print(
            f"{target_name:7s} "
            f"{info['uniprot']:6s}  "
            f"{info['chembl']}"
        )

    target_table = pd.DataFrame(
        resolved_targets
    )

    target_table.to_csv(
        OUTPUT_DIR
        / "resolved_targets.csv",
        index=False,
    )

     # --------------------------------------------------------
    # Fetch/load exact IC50 data
    # --------------------------------------------------------

    print()
    print(
        "Fetching strict biochemical IC50 data..."
    )

    all_activity = []
    coverage_rows = []

    for target in resolved_targets:

        name = target[
            "target_name"
        ]

        chembl_id = target[
            "target_chembl_id"
        ]

        cache_file = (
            CACHE_DIR
            / f"{name}_exact_ic50.csv"
        )

        if cache_file.exists():

            print(
                f"{name}: loading cached data "
                f"from {cache_file}"
            )

            df = pd.read_csv(
                cache_file
            )

        else:

            print(
                f"{name}: fetching from ChEMBL..."
            )

            df = fetch_strict_ic50(
                name,
                chembl_id,
            )

            df.to_csv(
                cache_file,
                index=False,
            )

            print(
                f"{name}: cached "
                f"{len(df)} rows"
            )

        all_activity.append(
            df
        )

        n_measurements = len(
            df
        )

        n_compounds = (
            df[
                "molecule_chembl_id"
            ]
            .nunique()
            if not df.empty
            else 0
        )

        coverage_rows.append(
            {
                "target_name":
                    name,

                "uniprot_accession":
                    target[
                        "uniprot_accession"
                    ],

                "target_chembl_id":
                    chembl_id,

                "n_measurements":
                    n_measurements,

                "n_unique_compounds":
                    n_compounds,
            }
        )

    coverage = pd.DataFrame(
        coverage_rows
    )

    print()
    print(
        "Per-target coverage:"
    )

    print(
        coverage.to_string(
            index=False
        )
    )

    coverage.to_csv(
        OUTPUT_DIR
        / "target_coverage.csv",
        index=False,
    )
    # --------------------------------------------------------
    # Combine all activities
    # --------------------------------------------------------

    activity = pd.concat(
        all_activity,
        ignore_index=True,
    )

    activity.to_csv(
        OUTPUT_DIR
        / "strict_ic50_raw.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Compound × target coverage matrix
    # --------------------------------------------------------

    compound_target = (
        activity[
            [
                "molecule_chembl_id",
                "target_name",
            ]
        ]
        .drop_duplicates()
    )

    presence = (
        compound_target
        .assign(
            measured=1
        )
        .pivot_table(
            index="molecule_chembl_id",
            columns="target_name",
            values="measured",
            fill_value=0,
        )
    )

    # Ensure all five columns are present.
    presence = presence.reindex(
        columns=list(
            TARGETS.keys()
        ),
        fill_value=0,
    )

    presence[
        "n_targets_measured"
    ] = presence.sum(
        axis=1
    )

    # --------------------------------------------------------
    # Coverage distribution
    # --------------------------------------------------------

    coverage_distribution = (
        presence[
            "n_targets_measured"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "n_targets_measured"
        )
        .reset_index(
            name="n_compounds"
        )
    )

    print()
    print(
        "Compound target coverage:"
    )

    print(
        coverage_distribution
        .to_string(
            index=False
        )
    )

    coverage_distribution.to_csv(
        OUTPUT_DIR
        / "compound_target_coverage.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Pairwise overlap
    # --------------------------------------------------------

    targets = list(
        TARGETS.keys()
    )

    overlap = pd.DataFrame(
        index=targets,
        columns=targets,
        dtype=int,
    )

    for target_a in targets:
        for target_b in targets:

            overlap.loc[
                target_a,
                target_b,
            ] = int(
                (
                    (
                        presence[
                            target_a
                        ]
                        == 1
                    )
                    &
                    (
                        presence[
                            target_b
                        ]
                        == 1
                    )
                ).sum()
            )

    overlap = overlap.astype(
        int
    )

    print()
    print(
        "Pairwise compound overlap:"
    )

    print(
        overlap.to_string()
    )

    overlap.to_csv(
        OUTPUT_DIR
        / "pairwise_overlap.csv"
    )

    # --------------------------------------------------------
    # ROCK2-centric overlap
    # --------------------------------------------------------

    rock2_compounds = (
        presence[
            presence[
                "ROCK2"
            ]
            == 1
        ]
    )

    print()
    print(
        "ROCK2 compounds with additional "
        "panel measurements:"
    )

    for n_targets in range(
        1,
        len(targets) + 1,
    ):

        n = int(
            (
                rock2_compounds[
                    "n_targets_measured"
                ]
                >= n_targets
            ).sum()
        )

        print(
            f">= {n_targets} target(s): {n}"
        )

    # --------------------------------------------------------
    # Save compound matrix
    # --------------------------------------------------------

    presence.reset_index().to_csv(
        OUTPUT_DIR
        / "compound_target_presence.csv",
        index=False,
    )

    print()
    print(
        "Saved coverage results to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()