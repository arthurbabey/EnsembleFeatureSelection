import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.feature_selection_methods import *
from src.feature_selection_pipeline import FeatureSelectionPipeline


@pytest.fixture
def pipeline_instance():
    # Define the data frame
    num_samples = 100
    num_features = 5000
    feature_names = [f"Feature_{i+1}" for i in range(num_features)]
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
    merging_strategy = "merging_strategy_union_of_pairwise_intersections"
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


@pytest.fixture
def train_data():
    num_samples = 80
    num_features = 5000
    feature_names = [f"Feature_{i+1}" for i in range(num_features)]
    target_values = np.random.randint(0, 4, size=num_samples)

    data = pd.DataFrame(
        np.random.randn(num_samples, num_features), columns=feature_names
    )
    data["target"] = target_values

    return data


def test_compute_pareto_analysis(pipeline_instance):
    groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    names = ["Group 1", "Group 2", "Group 3"]
    best_group_name = pipeline_instance._compute_pareto_analysis(groups, names)
    assert best_group_name == "Group 3"


def test_extract_features(pipeline_instance):
    # random data frame with 10 features and 5 samples and generic names defined in headers
    headers = [
        "hello",
        "world",
        "this",
        "is",
        "a",
        "test",
        "of",
        "the",
        "emergency",
        "broadcast",
    ]
    data = pd.DataFrame(np.random.randn(5, 10), columns=headers)
    features_list = pipeline_instance._extract_features(data)

    assert len(features_list) == 10
    for i in range(10):
        assert features_list[i].name == headers[i]

    # test if data is not a dataframe should return ValueError
    with pytest.raises(ValueError):
        pipeline_instance._extract_features(np.random.randn(5, 10))


def test_generate_subgroups_names(pipeline_instance):
    subgroup_names = pipeline_instance._generate_subgroup_names()
    assert len(subgroup_names) == 4
    for group in subgroup_names:
        assert len(group) >= 2
        for name in group:
            assert isinstance(name, str)
            assert name in ["mutual_info", "random_forest", "f_statistic"]


def test_get_X_Y(pipeline_instance):
    # random data frame with 10 features and 5 samples and generic names defined in headers
    headers = [
        "hello",
        "world",
        "this",
        "is",
        "a",
        "test",
        "of",
        "the",
        "emergency",
        "broadcast",
    ]
    data = pd.DataFrame(np.random.randn(5, 10), columns=headers)
    # should raise errovalue because no target column
    with pytest.raises(ValueError):
        X, y = pipeline_instance._get_X_y(data)

    # random data frame with 10 features and 5 samples and generic names defined in headers
    headers = [
        "hello",
        "world",
        "this",
        "is",
        "a",
        "test",
        "of",
        "the",
        "emergency",
        "target",
    ]
    data = pd.DataFrame(np.random.randn(5, 10), columns=headers)
    X, y = pipeline_instance._get_X_y(data)
    assert X.shape == (5, 9)
    assert y.shape == (5,)

    # test if data is not a dataframe should return ValueError
    with pytest.raises(ValueError):
        FeatureSelectionPipeline._get_X_y(np.random.randn(5, 10))


def test_computes_feature(pipeline_instance):
    # feature scores are random positive numbers
    nbr_features = pipeline_instance.data.shape[1] - 1
    feature_scores = list(np.random.rand(nbr_features))
    selected_features_indices = list(np.argsort(feature_scores)[-500:])
    all_features = pipeline_instance._compute_features(
        selected_features_indices, feature_scores
    )

    assert len(all_features) == nbr_features
    for i in range(nbr_features):
        assert all_features[i].score == feature_scores[i]
        if i in selected_features_indices:
            assert all_features[i].selected is True
        else:
            assert all_features[i].selected is False
    # test that number of selected features is equal to the number of selected features
    assert len([feature for feature in all_features if feature.selected]) == len(
        selected_features_indices
    )

    # test if selected_features_indices is empty
    all_features = pipeline_instance._compute_features([], feature_scores)
    assert len(all_features) == nbr_features
    assert len([feature for feature in all_features if feature.selected]) == 0

    # test with feature scores being None
    all_features = pipeline_instance._compute_features(selected_features_indices, None)
    assert len(all_features) == nbr_features
    assert len([feature for feature in all_features if feature.selected]) == len(
        selected_features_indices
    )
