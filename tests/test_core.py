import numpy as np
import pandas as pd

from selectivity_pcm.analysis.calibration import (
    empirical_selectivity_risk,
)
from selectivity_pcm.features.pocket_features import (
    pocket_one_hot,
)


def test_pocket_one_hot_shape():
    sequence = "A" * 85

    features = pocket_one_hot(sequence)

    assert features.shape == (1785,)
    assert np.isfinite(features).all()


def test_empirical_selectivity_risk_insufficient_data():
    calibration_df = pd.DataFrame(
        {
            "compound_id": ["c1", "c1"],
            "target_name": ["ROCK2", "ROCK1"],
            "observed_pIC50": [7.0, 6.0],
            "predicted_pIC50": [7.1, 6.1],
        }
    )

    result = empirical_selectivity_risk(
        calibration_df=calibration_df,
        primary_target="ROCK2",
        off_target="ROCK1",
        predicted_primary_pIC50=7.0,
        predicted_offtarget_pIC50=6.0,
        margin_threshold=1.0,
        n_neighbors=25,
        min_calibration=25,
    )

    assert result["calibration_status"] == "INSUFFICIENT_DATA"
    assert np.isnan(
        result["P_offtarget_within_margin"]
    )


def test_predicted_margin_definition():
    primary = 7.5
    off_target = 6.2

    margin = primary - off_target

    assert np.isclose(
    margin,
    1.3,
)