import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_selection_methods import *
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import pytest
import numpy as np

@pytest.fixture
def fake_data():
    # Generate random 2D vector for features (100 samples, 50 features)
    X_random = np.random.randint(0, 101, size=(100, 50))
    y = np.random.randint(0, 3, size=100)  # Ensure y has the same length as X_random samples

    # Generate 10 correlated new features based on y
    new_features = np.zeros((100, 10))
    for i, yi in enumerate(y):
        if yi == 0:
            new_features[i] = np.random.randint(0, 15, size=10)  # Set values between 0 and 4
        elif yi == 1:
            new_features[i] = np.random.randint(42, 57, size=10)  # Set values between 47 and 52
        elif yi == 2:
            new_features[i] = np.random.randint(85, 100, size=10)  # Set values between 96 and 99

    # X has now 50 noisy feature and then 10 highly correlated features
    X_combined = np.concatenate((X_random, new_features), axis=1)
    # shuffle to avoid having them ordered
    np.random.shuffle(X_combined.T)

    # retrieve the columns that met the condition above i.e columns that are correlated
    ranges = [(0, 15), (42, 57), (85, 100)]    
    expected_features = []    
    for i in range(X_combined.shape[1]):
        if all(any(low <= int(val) <= high for low, high in ranges) for val in X_combined[:, i]):
            expected_features.append(i)

    return X_combined, y, expected_features


def test_feature_selection_f_statistic(fake_data):
    X, y, expected_features = fake_data
    scores, selected_features = feature_selection_f_statistic(X, y, task='classification', num_features_to_select=10)
    assert len(scores) == 60
    assert len(selected_features) == 10
    assert set(selected_features) == set(expected_features)

def test_feature_selection_mutual_info(fake_data):
    X, y, expected_features = fake_data
    scores, selected_features = feature_selection_mutual_info(X, y, task='classification', num_features_to_select=10)
    assert len(scores) == 60
    assert len(selected_features) == 10
    assert set(selected_features) == set(expected_features)

def test_feature_selection_random_forest(fake_data):
    X, y, expected_features = fake_data
    scores, selected_features = feature_selection_random_forest(X, y, task='classification', num_features_to_select=10)
    assert len(scores) == 60
    assert len(selected_features) == 10
    assert set(selected_features) == set(expected_features)

def test_feature_selection_svm(fake_data):
    X, y, expected_features = fake_data
    scores, selected_features = feature_selection_svm(X, y, task='classification', num_features_to_select=10)
    assert len(scores) == 60
    assert len(selected_features) == 10
    assert set(selected_features) == set(expected_features)

def test_feature_selection_rfe_rf(fake_data):
    X, y, expected_features = fake_data
    scores, selected_features = feature_selection_rfe_rf(X, y, task='classification', num_features_to_select=10)
    assert len(scores) == 60
    assert len(selected_features) == 10
    assert set(selected_features) == set(expected_features)

def test_feature_selection_xgboost(fake_data):
    X, y, expected_features = fake_data
    scores, selected_features = feature_selection_xgboost(X, y, task='classification', num_features_to_select=10)
    assert len(scores) == 60
    assert len(selected_features) == 10
    assert set(selected_features) == set(expected_features)

def test_feature_selection_rfe_rf(fake_data):
    X, y, expected_features = fake_data
    scores, selected_features = feature_selection_rfe_rf(X, y, task='classification', num_features_to_select=10)
    assert scores == None
    assert len(selected_features) == 10
    assert set(selected_features) == set(expected_features)
    













