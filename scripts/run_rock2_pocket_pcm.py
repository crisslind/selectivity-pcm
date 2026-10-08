import pandas as pd

from selectivity_pcm.config import load_python_config
from selectivity_pcm.models.pocket_pcm import (
    cross_validate_pocket_pcm_scaffold,
)
from selectivity_pcm.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

INPUT_FILE = (
    DATA_DIR
    / "rock2_panel_analysis_structures.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "rock2_panel"
)

KINASE_POCKETS = load_python_config(
    PROJECT_ROOT / "examples" / "rock2" / "kinase_pockets.py",
    "KINASE_POCKETS",
)

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded observations: {len(df)}"
    )
    print(
        f"Unique compounds: "
        f"{df['compound_id'].nunique()}"
    )

    print()
    print(
        "Observations by target:"
    )
    print(
        df["target_name"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Running pocket PCM "
        "5-fold scaffold CV..."
    )

    (
        overall_metrics,
        target_metrics,
        predictions,
    ) = cross_validate_pocket_pcm_scaffold(
        df,
        pockets=KINASE_POCKETS,
        n_splits=5,
        random_state=42,
    )

    overall_file = (
        OUTPUT_DIR
        / "pocket_pcm_scaffold_cv_metrics.csv"
    )

    target_file = (
        OUTPUT_DIR
        / "pocket_pcm_scaffold_cv_target_metrics.csv"
    )

    predictions_file = (
        OUTPUT_DIR
        / "pocket_pcm_scaffold_cv_predictions.csv"
    )

    overall_metrics.to_csv(
        overall_file,
        index=False,
    )

    target_metrics.to_csv(
        target_file,
        index=False,
    )

    predictions.to_csv(
        predictions_file,
        index=False,
    )

    print()
    print("Overall CV metrics:")
    print(
        overall_metrics.to_string(
            index=False
        )
    )

    print()
    print(
        "Mean overall metrics:"
    )
    print(
        overall_metrics[
            [
                "RMSE",
                "MAE",
                "R2",
                "Spearman",
            ]
        ]
        .mean()
        .to_string()
    )

    print()
    print(
        "Mean metrics by target:"
    )
    print(
        target_metrics
        .groupby("target")[
            [
                "RMSE",
                "MAE",
                "R2",
                "Spearman",
            ]
        ]
        .mean()
        .round(3)
        .to_string()
    )

    print()
    print("Saved:")
    print(overall_file)
    print(target_file)
    print(predictions_file)


if __name__ == "__main__":
    main()
    