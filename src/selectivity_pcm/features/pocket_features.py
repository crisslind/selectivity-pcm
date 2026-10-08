import numpy as np

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
POCKET_STATES = AMINO_ACIDS + ["-"]


def pocket_one_hot(sequence: str) -> np.ndarray:
    """
    One-hot encode an aligned KLIFS pocket sequence.

    KLIFS pockets contain 85 aligned positions.
    Each position is represented by 21 states:

    - 20 standard amino acids
    - '-' for gap / unknown

    Output size:
        85 * 21 = 1785 features
    """

    sequence = sequence.upper().replace("_", "-")

    if len(sequence) != 85:
        raise ValueError(
            f"Expected an 85-residue KLIFS pocket, "
            f"got {len(sequence)} residues."
        )

    state_to_index = {
        state: index
        for index, state in enumerate(POCKET_STATES)
    }

    features = np.zeros(
        len(sequence) * len(POCKET_STATES),
        dtype=np.uint8,
    )

    for position, residue in enumerate(sequence):
        if residue not in state_to_index:
            residue = "-"

        state_index = state_to_index[residue]

        feature_index = (
            position * len(POCKET_STATES)
            + state_index
        )

        features[feature_index] = 1

    return features