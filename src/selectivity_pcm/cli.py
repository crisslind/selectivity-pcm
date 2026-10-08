from pathlib import Path

import pandas as pd
import typer

from selectivity_pcm.analysis.landscape import (
    build_activity_matrix,
    calculate_selectivity,
)
from selectivity_pcm.analysis.plotting import (
    plot_selectivity_heatmap,
)

app = typer.Typer(
    help="Explore experimental and predicted target selectivity landscapes."
)


@app.callback()
def main():
    """
    Selectivity PCM command-line interface.
    """


@app.command()
def landscape(
    input_file: Path = typer.Option(
        ...,
        "--input-file",
        help="CSV file containing compound-target activity data.",
    ),
    primary: str = typer.Option(
        ...,
        "--primary",
        help="Primary target used for selectivity calculations.",
    ),
    activity_column: str = typer.Option(
        ...,
        "--activity-column",
        help=(
            "Column containing activity values. "
            "Values must already be converted to pIC50."
        ),
    ),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output-dir",
        help="Directory where output files will be written.",
    ),
):
    """
    Build an experimental selectivity landscape from pIC50 data.
    """

    if not input_file.exists():
        raise typer.BadParameter(
            f"Input file does not exist: {input_file}"
        )

    try:
        df = pd.read_csv(input_file)
    except Exception as exc:
        raise typer.BadParameter(
            f"Could not read input CSV: {exc}"
        ) from exc

    required_columns = {
        "compound_id",
        "target_name",
        activity_column,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        available = ", ".join(df.columns)

        raise typer.BadParameter(
            f"Missing required column(s): {missing}. "
            f"Available columns: {available}"
        )

    if not pd.api.types.is_numeric_dtype(df[activity_column]):
        raise typer.BadParameter(
            f"Activity column '{activity_column}' must contain numeric pIC50 values."
        )

    try:
        matrix = build_activity_matrix(
            df,
            activity_column=activity_column,
        )

        selectivity = calculate_selectivity(
            matrix,
            primary_target=primary,
        )

    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_file = output_dir / "activity_matrix.csv"
    selectivity_file = output_dir / "selectivity_summary.csv"

    matrix.to_csv(matrix_file)
    selectivity.to_csv(selectivity_file)

    heatmap_file = output_dir / "selectivity_heatmap.png"

    plot_selectivity_heatmap(
    matrix=matrix,
    primary_target=primary,
    output_file=heatmap_file,
    )
    print(f"  {heatmap_file}")
    
    print("\nDataset summary:")
    print(f"  Compounds in activity matrix: {len(matrix)}")
    print(f"  Targets: {len(matrix.columns)}")
    print(f"  Compounds with usable selectivity data: {len(selectivity)}")

    print("\nOff-target coverage:")
    coverage_counts = (
        selectivity["n_offtargets_measured"]
        .value_counts()
        .sort_index()
    )

    for n_targets, count in coverage_counts.items():
        print(f"  {n_targets} off-target(s): {count}")

    print("\nTop 20 by observed selectivity margin:")
    print(
        selectivity
        .head(20)
        .round(2)
        .to_string()
    )

    n_offtargets = len(matrix.columns) - 1

    complete = selectivity[
        selectivity["n_offtargets_measured"] == n_offtargets
    ]

    print(
        f"\nTop 20 with complete off-target coverage "
        f"({n_offtargets}/{n_offtargets} off-targets measured):"
    )

    print(
        complete
        .head(20)
        .round(2)
        .to_string()
    )

    print("\nOutput:")
    print(f"  {matrix_file}")
    print(f"  {selectivity_file}")


if __name__ == "__main__":
    app()