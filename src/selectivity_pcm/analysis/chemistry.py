from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True
)


def make_morgan_fingerprint(
    smiles: str,
):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return morgan_generator.GetFingerprint(mol)


def find_selectivity_cliffs(
    df: pd.DataFrame,
    primary_target: str,
    comparator_target: str,
    similarity_threshold: float = 0.8,
    margin_difference_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Find chemically similar compound pairs with strongly different
    selectivity between two targets.
    """

    pivot = df.pivot_table(
        index=["compound_id", "canonical_smiles"],
        columns="target_name",
        values="pIC50",
        aggfunc="median",
    )

    required = {primary_target, comparator_target}

    if not required.issubset(pivot.columns):
        raise ValueError(
            f"Required targets not present: {required}"
        )

    pivot = pivot.dropna(
        subset=[primary_target, comparator_target]
    ).copy()

    pivot["selectivity_margin"] = (
        pivot[primary_target]
        - pivot[comparator_target]
    )

    pivot = pivot.reset_index()

    fingerprints = [
        make_morgan_fingerprint(smiles)
        for smiles in pivot["canonical_smiles"]
    ]

    rows = []

    for i in range(len(pivot)):
        fp_i = fingerprints[i]

        if fp_i is None:
            continue

        for j in range(i + 1, len(pivot)):
            fp_j = fingerprints[j]

            if fp_j is None:
                continue

            similarity = DataStructs.TanimotoSimilarity(
                fp_i,
                fp_j,
            )

            if similarity < similarity_threshold:
                continue

            margin_difference = abs(
                pivot.loc[i, "selectivity_margin"]
                - pivot.loc[j, "selectivity_margin"]
            )

            if margin_difference < margin_difference_threshold:
                continue

            rows.append(
                {
                    "compound_a": pivot.loc[i, "compound_id"],
                    "compound_b": pivot.loc[j, "compound_id"],
                    "similarity": similarity,
                    "margin_a": pivot.loc[i, "selectivity_margin"],
                    "margin_b": pivot.loc[j, "selectivity_margin"],
                    "margin_difference": margin_difference,
                }
            )

    return pd.DataFrame(rows).sort_values(
        "margin_difference",
        ascending=False,
    )


def plot_selectivity_cliff(
    df: pd.DataFrame,
    compound_a: str,
    compound_b: str,
    primary_target: str,
    comparator_target: str,
    similarity: float,
    output_file: Path,
):
    subset = df[
        df["compound_id"].isin([compound_a, compound_b])
        & df["target_name"].isin(
            [primary_target, comparator_target]
        )
    ].copy()

    pivot = subset.pivot_table(
        index="compound_id",
        columns="target_name",
        values="pIC50",
        aggfunc="median",
    )

    smiles_lookup = (
        subset[
            ["compound_id", "canonical_smiles"]
        ]
        .drop_duplicates()
        .set_index("compound_id")["canonical_smiles"]
        .to_dict()
    )

    mol_a = Chem.MolFromSmiles(
        smiles_lookup[compound_a]
    )
    mol_b = Chem.MolFromSmiles(
        smiles_lookup[compound_b]
    )

    margin_a = (
        pivot.loc[compound_a, primary_target]
        - pivot.loc[compound_a, comparator_target]
    )

    margin_b = (
        pivot.loc[compound_b, primary_target]
        - pivot.loc[compound_b, comparator_target]
    )

    image = Draw.MolsToGridImage(
    [mol_a, mol_b],
    molsPerRow=2,
    subImgSize=(500, 400),
    legends=["", ""],
    useSVG=False,
)

    fig, ax = plt.subplots(
        figsize=(11, 6),
    )

    ax.imshow(image)
    ax.axis("off")

    ax.set_title(
        f"{primary_target} vs {comparator_target} selectivity cliff\n"
        f"Morgan similarity = {similarity:.2f} | "
        f"Δmargin = {abs(margin_a - margin_b):.2f}",
        fontsize=14,
        pad=20,
    )

    text_a = (
        f"{compound_a}\n"
        f"{primary_target} pIC50: "
        f"{pivot.loc[compound_a, primary_target]:.2f}\n"
        f"{comparator_target} pIC50: "
        f"{pivot.loc[compound_a, comparator_target]:.2f}\n"
        f"Selectivity margin: {margin_a:+.2f}"
    )

    text_b = (
        f"{compound_b}\n"
        f"{primary_target} pIC50: "
        f"{pivot.loc[compound_b, primary_target]:.2f}\n"
        f"{comparator_target} pIC50: "
        f"{pivot.loc[compound_b, comparator_target]:.2f}\n"
        f"Selectivity margin: {margin_b:+.2f}"
    )

    fig.text(
        0.28,
        0.04,
        text_a,
        ha="center",
        va="bottom",
        fontsize=11,
    )

    fig.text(
        0.72,
        0.04,
        text_b,
        ha="center",
        va="bottom",
        fontsize=11,
    )

    fig.subplots_adjust(
        top=0.84,
        bottom=0.22,
    )

    fig.savefig(
        output_file,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)