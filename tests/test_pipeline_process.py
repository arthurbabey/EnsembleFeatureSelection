import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.feature_selection_methods import (
    feature_selection_f_statistic,
    feature_selection_mutual_info,
    feature_selection_random_forest,
)
from src.feature_selection_pipeline import FeatureSelectionPipeline
from src.merging_strategy_methods import (
    merging_strategy_union_of_pairwise_intersections,
)


@pytest.fixture
def pipeline_instance():
    # Define the data frame
    num_samples = 100
    num_features = 5000
    feature_names = [f"Feature_{i + 1}" for i in range(num_features)]
    target_values = np.random.randint(0, 4, size=num_samples)

    data = pd.DataFrame(
        np.random.randn(num_samples, num_features), columns=feature_names
    )
    data["target"] = target_values

    # Other parameters for the pipeline
    classifier = "RF"
    fs_methods = [
        feature_selection_mutual_info,
        feature_selection_random_forest,
        feature_selection_f_statistic,
    ]
    merging_strategy = merging_strategy_union_of_pairwise_intersections
    num_repeats = 3
    task = "classification"
    threshold = None

    pipeline = FeatureSelectionPipeline(
        data=data,
        fs_methods=fs_methods,
        merging_strategy=merging_strategy,
        classifier=classifier,
        num_repeats=num_repeats,
        threshold=threshold,
        task=task,
    )

    return pipeline


def test_feature_selection_pipeline(pipeline_instance):
    (
        best_features,
        best_repeat,
        best_group_name,
    ) = pipeline_instance.iterate_pipeline()
    assert best_features is not None
    assert best_repeat is not None
    assert best_group_name is not None

    # test best_features are in the data columns
    assert all([feature in pipeline_instance.data.columns for feature in best_features])
    assert 0 <= int(best_repeat) <= pipeline_instance.num_repeats
    assert best_group_name in pipeline_instance.subgroup_names
