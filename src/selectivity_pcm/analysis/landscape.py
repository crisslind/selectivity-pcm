import pandas as pd


def build_activity_matrix(
    df: pd.DataFrame,
    activity_column: str,
) -> pd.DataFrame:
    """
    Build a compound x target activity matrix.

    The activity column is assumed to contain pIC50 values.
    Higher values therefore indicate stronger activity.
    """

    required_columns = {
        "compound_id",
        "target_name",
        activity_column,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    matrix = df.pivot_table(
        index="compound_id",
        columns="target_name",
        values=activity_column,
        aggfunc="median",
    )

    return matrix


def calculate_selectivity(
    matrix: pd.DataFrame,
    primary_target: str,
) -> pd.DataFrame:
    """
    Calculate selectivity relative to the strongest measured off-target.

    Assumes activity values are pIC50:
        higher pIC50 = stronger activity

    Selectivity margin:
        primary target pIC50 - strongest measured off-target pIC50

    A margin of:
        1.0 = 10-fold selectivity
        2.0 = 100-fold selectivity
    """

    if primary_target not in matrix.columns:
        available_targets = ", ".join(map(str, matrix.columns))

        raise ValueError(
            f"Primary target '{primary_target}' not found. "
            f"Available targets: {available_targets}"
        )

    off_targets = [
        column
        for column in matrix.columns
        if column != primary_target
    ]

    if not off_targets:
        raise ValueError(
            "At least one off-target is required to calculate selectivity."
        )

    # Keep only compounds with:
    # 1. a measured primary-target activity
    # 2. at least one measured off-target activity
    valid_mask = (
        matrix[primary_target].notna()
        & matrix[off_targets].notna().any(axis=1)
    )

    valid_matrix = matrix.loc[valid_mask].copy()

    results = pd.DataFrame(index=valid_matrix.index)

    results["primary_pIC50"] = valid_matrix[primary_target]

    results["n_offtargets_measured"] = (
        valid_matrix[off_targets]
        .notna()
        .sum(axis=1)
    )

    results["strongest_offtarget_pIC50"] = (
        valid_matrix[off_targets]
        .max(axis=1)
    )

    results["strongest_offtarget"] = (
        valid_matrix[off_targets]
        .idxmax(axis=1)
    )

    # results["selectivity_margin"] = (
    #     results["primary_pIC50"]
    #     - results["strongest_offtarget_pIC50"]
    # )

    # results["fold_selectivity"] = (
    #     10 ** results["selectivity_margin"]
    # )

    results["observed_selectivity_margin"] = (
        results["primary_pIC50"]
        - results["strongest_offtarget_pIC50"]
    )

    results["observed_fold_selectivity"] = (
        10 ** results["observed_selectivity_margin"]
    )

    results = results.sort_values(
        "observed_selectivity_margin",
        ascending=False,
    )

    results = results.sort_values(
        "observed_selectivity_margin",
        ascending=False,
    )

    return results