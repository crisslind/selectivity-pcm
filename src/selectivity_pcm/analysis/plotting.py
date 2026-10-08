from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_selectivity_heatmap(
    matrix: pd.DataFrame,
    primary_target: str,
    output_file: Path,
):
    """
    Plot compounds with complete target coverage,
    ordered by selectivity toward the primary target.
    """

    complete = matrix.dropna().copy()

    if complete.empty:
        raise ValueError("No compounds have complete target coverage.")

    off_targets = [
        column
        for column in complete.columns
        if column != primary_target
    ]

    complete["selectivity_margin"] = (
        complete[primary_target]
        - complete[off_targets].max(axis=1)
    )

    complete = complete.sort_values(
        "selectivity_margin",
        ascending=False,
    )

    plot_data = complete.drop(
        columns="selectivity_margin"
    )

    fig, ax = plt.subplots(
        figsize=(8, max(6, len(plot_data) * 0.18))
    )

    image = ax.imshow(
        plot_data.values,
        aspect="auto",
    )

    ax.set_xticks(range(len(plot_data.columns)))
    ax.set_xticklabels(plot_data.columns)

    ax.set_yticks(range(len(plot_data.index)))
    ax.set_yticklabels(plot_data.index, fontsize=6)

    ax.set_xlabel("Target")
    ax.set_ylabel("Compound")
    ax.set_title(
        f"Experimental selectivity landscape: {primary_target}"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="pIC50",
    )

    fig.tight_layout()
    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)