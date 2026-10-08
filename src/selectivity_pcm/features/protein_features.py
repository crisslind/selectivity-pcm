import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def protein_sequence_features(
    sequence: str,
) -> np.ndarray:
    """
    Generate simple physicochemical descriptors from a protein sequence.
    """

    sequence = sequence.upper()

    analysis = ProteinAnalysis(sequence)

    composition = analysis.amino_acids_percent

    features = [
        len(sequence),
    ]

    features.extend(
        composition.get(aa, 0.0)
        for aa in AMINO_ACIDS
    )

    features.extend(
        [
            analysis.molecular_weight(),
            analysis.aromaticity(),
            analysis.instability_index(),
            analysis.isoelectric_point(),
            analysis.gravy(),
        ]
    )

    return np.asarray(
        features,
        dtype=float,
    )