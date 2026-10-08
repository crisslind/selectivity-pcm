from pathlib import Path

import numpy as np
import pandas as pd
import requests

TARGETS = {
    "ROCK2": "CHEMBL2973",
    "ROCK1": "CHEMBL3231",
    "CSNK2A2": "CHEMBL2148",
    "AURKA": "CHEMBL4722",
    "STK3": "CHEMBL5408",
}

OUTPUT_DIR = Path("data")

RAW_FILE = OUTPUT_DIR / "rock2_panel_raw.csv"
CURATED_FILE = OUTPUT_DIR / "rock2_panel_curated.csv"
ANALYSIS_FILE = OUTPUT_DIR / "rock2_panel_analysis.csv"

VARIABILITY_THRESHOLD = 1.0


def fetch_ic50_records(
    target_name: str,
    target_chembl_id: str,
) -> pd.DataFrame:

    base_url = (
        "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    )

    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": "IC50",
        "limit": 1000,
    }

    records = []
    url = base_url

    while url is not None:

        response = requests.get(
            url,
            params=params,
            timeout=60,
        )

        response.raise_for_status()

        payload = response.json()

        records.extend(
            payload.get(
                "activities",
                []
            )
        )

        next_url = (
            payload
            .get("page_meta", {})
            .get("next")
        )

        if next_url:

            if next_url.startswith("http"):
                url = next_url

            else:
                url = (
                    "https://www.ebi.ac.uk"
                    + next_url
                )

            params = None

        else:
            url = None

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df["target_name"] = target_name

    return df


def curate_measurements(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    df = raw.copy()

    # --------------------------------------------------------
    # Strict assay filtering
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
    # Require exact / usable IC50 values
    # --------------------------------------------------------

# --------------------------------------------------------
# Require exact IC50 measurements
#
# Censored values such as:
#   > 10000 nM
#   < 10 nM
#
# are useful bounds, but should not be treated as exact
# regression targets or exact selectivity margins.
# --------------------------------------------------------

    df = df[
        df["standard_relation"] == "="
    ].copy()

    df["standard_value"] = pd.to_numeric(
        df["standard_value"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "molecule_chembl_id",
            "standard_value",
            "standard_units",
        ]
    )

    # --------------------------------------------------------
    # Normalize IC50 to nM
    # --------------------------------------------------------

    unit_factors = {
        "nM": 1.0,
        "uM": 1000.0,
        "µM": 1000.0,
        "mM": 1_000_000.0,
        "pM": 0.001,
    }

    df = df[
        df["standard_units"].isin(
            unit_factors
        )
    ].copy()

    df["IC50_nM"] = (
        df["standard_value"]
        *
        df["standard_units"].map(
            unit_factors
        )
    )

    # --------------------------------------------------------
    # Keep positive finite values only
    # --------------------------------------------------------

    df = df[
        (
            df["IC50_nM"] > 0
        )
        &
        np.isfinite(
            df["IC50_nM"]
        )
    ].copy()

    # --------------------------------------------------------
    # pIC50
    #
    # pIC50 = 9 - log10(IC50_nM)
    # --------------------------------------------------------

    df["pIC50"] = (
        9.0
        -
        np.log10(
            df["IC50_nM"]
        )
    )

    df = df.rename(
        columns={
            "molecule_chembl_id":
                "compound_id",
        }
    )

    return df


def aggregate_compound_target(
    curated: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        curated.groupby(
            [
                "compound_id",
                "target_name",
            ]
        )
        .agg(
            pIC50=(
                "pIC50",
                "median",
            ),
            n_measurements=(
                "pIC50",
                "size",
            ),
            min_pIC50=(
                "pIC50",
                "min",
            ),
            max_pIC50=(
                "pIC50",
                "max",
            ),
        )
        .reset_index()
    )

    grouped[
        "pIC50_range"
    ] = (
        grouped[
            "max_pIC50"
        ]
        -
        grouped[
            "min_pIC50"
        ]
    )

    grouped[
        "measurement_flag"
    ] = np.where(
        grouped[
            "pIC50_range"
        ]
        > VARIABILITY_THRESHOLD,
        "HIGH_VARIABILITY",
        "OK",
    )

    return grouped


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Fetch
    # --------------------------------------------------------

    frames = []

    for (
        target_name,
        chembl_id,
    ) in TARGETS.items():

        print(
            f"Fetching {target_name} "
            f"({chembl_id})..."
        )

        df = fetch_ic50_records(
            target_name,
            chembl_id,
        )

        print(
            f"  raw records: {len(df)}"
        )

        frames.append(df)

    raw = pd.concat(
        frames,
        ignore_index=True,
    )

    raw.to_csv(
        RAW_FILE,
        index=False,
    )

    print()
    print(
        f"Raw measurements: {len(raw)}"
    )

    # --------------------------------------------------------
    # Curate
    # --------------------------------------------------------

    curated = curate_measurements(
        raw
    )

    curated.to_csv(
        CURATED_FILE,
        index=False,
    )

    print(
        f"Strict curated measurements: "
        f"{len(curated)}"
    )

    # --------------------------------------------------------
    # Aggregate replicates
    # --------------------------------------------------------

    aggregated = (
        aggregate_compound_target(
            curated
        )
    )

    print(
        f"Compound-target pairs: "
        f"{len(aggregated)}"
    )

    print()
    print(
        "Measurement flags:"
    )

    print(
        aggregated[
            "measurement_flag"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Primary analysis set
    # --------------------------------------------------------

    analysis = aggregated[
        aggregated[
            "measurement_flag"
        ]
        == "OK"
    ].copy()

    analysis.to_csv(
        ANALYSIS_FILE,
        index=False,
    )

    print()
    print(
        "Usable compound-target pairs:"
    )

    print(
        analysis[
            "target_name"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        f"Total usable pairs: "
        f"{len(analysis)}"
    )

    print(
        f"Unique compounds: "
        f"{analysis['compound_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Target coverage
    # --------------------------------------------------------

    matrix = (
        analysis.pivot_table(
            index="compound_id",
            columns="target_name",
            values="pIC50",
        )
    )

    coverage = (
        matrix.notna()
        .sum(axis=1)
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Compounds by number of "
        "targets measured:"
    )

    print(
        coverage.to_string()
    )

    print()
    print(
        "Saved:"
    )

    print(RAW_FILE)
    print(CURATED_FILE)
    print(ANALYSIS_FILE)


if __name__ == "__main__":
    main()