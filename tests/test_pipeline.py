import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_selection_pipeline import FeatureSelectionPipeline
import pytest


import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def pipeline_instance():
    # Define the data frame
    num_samples = 100
    num_features = 5000
    feature_names = [f"Feature_{i+1}" for i in range(num_features)]
    target_values = np.random.randint(0, 4, size=num_samples)

    data = pd.DataFrame(np.random.randn(num_samples, num_features), columns=feature_names)
    data['target'] = target_values

    # Other parameters for the pipeline
    classifier = "RF"
    fs_methods = ["feature_selection_mutual_info", "feature_selection_random_forest", "feature_selection_f_statistic"]
    merging_strategy = "merging_strategy_union_of_pairwise_intersections"
    num_repeats = 3
    task = "classification"
    threshold = None

    pipeline = FeatureSelectionPipeline(data=data, fs_methods=fs_methods, merging_strategy=merging_strategy, classifier=classifier, num_repeats=num_repeats, threshold=threshold, task=task)    
    
    return pipeline

@pytest.fixture
def train_data():
    num_samples = 80
    num_features = 5000
    feature_names = [f"Feature_{i+1}" for i in range(num_features)]
    target_values = np.random.randint(0, 4, size=num_samples)

    data = pd.DataFrame(np.random.randn(num_samples, num_features), columns=feature_names)
    data['target'] = target_values

    return data

def test_compute_sbst_and_scores_per_method(pipeline_instance, train_data):
    pipeline_instance._compute_sbst_and_scores_per_method(train_data, 0)
    for method in pipeline_instance.fs_methods:
        assert isinstance(self.FS_subsets[(0, method)], dict)
    

