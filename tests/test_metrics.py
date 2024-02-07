import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.metrics import compute_performance_metrics, compute_stability_metrics


@pytest.fixture
def int_list_similar():
    return [[1, 2, 3], [3, 2, 1], [2, 3, 1], [1, 3, 2]]


@pytest.fixture
def int_list_different():
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


@pytest.fixture
def string_list_similar():
    return [
        ["apple", "banana", "orange"],
        ["banana", "apple", "orange"],
        ["apple", "orange", "banana"],
    ]


@pytest.fixture
def string_list_different():
    return [
        ["apple", "banana", "orange1"],
        ["apple2", "banana2", "orange2"],
        ["apple3", "banana3", "orange3"],
    ]


@pytest.fixture
def data():
    # Generate synthetic data
    num_samples = 1000
    num_features = 10
    X_train = np.random.randn(
        num_samples, num_features
    )  # Generate random 2D array (1000x10) for features
    y_train = np.random.randint(
        0, 2, num_samples
    )  # Generate random 1D array (1000,) for labels (binary classification)
    X_test = np.random.randn(
        num_samples, num_features
    )  # Generate random 2D array (1000x10) for test features
    y_test = np.random.randint(
        0, 2, num_samples
    )  # Generate random 1D array (1000,) for test labels (binary classification)

    # Return the generated data
    return X_train, y_train, X_test, y_test


def test_compute_stability_metrics_with_int_list_similar(int_list_similar):
    assert compute_stability_metrics(int_list_similar)[0] == 1


def test_compute_stability_metrics_with_int_list_different(int_list_different):
    assert compute_stability_metrics(int_list_different)[0] == 0


def test_compute_stability_metrics_with_string_list_similar(
    string_list_similar,
):
    assert compute_stability_metrics(string_list_similar)[0] == 1


def test_compute_stability_metrics_with_string_list_different(
    string_list_different,
):
    assert compute_stability_metrics(string_list_different)[0] == 0


def test_compute_performance_metrics_classification(data):
    X_train, y_train, X_test, y_test = data
    for classifier in ["RF", "naiveBayes", "bagging"]:
        results = compute_performance_metrics(
            classifier, "classification", X_train, y_train, X_test, y_test
        )
        assert "accuracy" in results
        assert "AUROC" in results
        assert "MAE" in results
        assert isinstance(results["accuracy"], float)
        assert isinstance(results["AUROC"], float)
        assert isinstance(results["MAE"], float)


def test_compute_performance_metrics_regression(data):
    X_train, y_train, X_test, y_test = data
    # error raise with naivebayes is tested after
    for classifier in ["RF", "bagging"]:
        results = compute_performance_metrics(
            classifier, "regression", X_train, y_train, X_test, y_test
        )
        assert "MAE" in results
        assert "R2" in results
        assert "RMSE" in results
        assert isinstance(results["MAE"], float)
        assert isinstance(results["R2"], float)
        assert isinstance(results["RMSE"], float)


def test_compute_performance_metrics_invalid_classifier(data):
    X_train, y_train, X_test, y_test = data

    # Check if compute_performance_metrics() raises ValueError with the expected message
    with pytest.raises(
        ValueError,
        match=r"Invalid classifier name '.*'. Please choose among 'RF', 'naiveBayes', or 'bagging'.",
    ):
        compute_performance_metrics(
            "invalid_classifier",
            "classification",
            X_train,
            y_train,
            X_test,
            y_test,
        )


def test_compute_performance_metrics_invalid_task(data):
    X_train, y_train, X_test, y_test = data
    for classifier in ["RF", "naiveBayes", "bagging"]:
        with pytest.raises(ValueError, match=r"Invalid task type"):
            compute_performance_metrics(
                classifier, "invalid task", X_train, y_train, X_test, y_test
            )


def test_compute_performance_metrics_regressionnaives(data):
    X_train, y_train, X_test, y_test = data
    with pytest.raises(
        ValueError, match=r"Naive Bayes is not suitable for regression task"
    ):
        compute_performance_metrics(
            "naiveBayes", "regression", X_train, y_train, X_test, y_test
        )
