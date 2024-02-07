import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.svm import SVC

from .feature_selection_methods import *
from .feature_selection_pipeline import FeatureSelectionPipeline
from .merging_strategy_methods import merging_strategy_union_of_pairwise_intersections


def compare_performance(selected_features_baseline, selected_features_pipeline, data):
    y = data["target"]
    classifier = SVC(kernel="rbf", C=1.0, gamma="scale")

    # Initialize lists to store performance metrics across runs
    (
        accuracy_pipeline,
        precision_pipeline,
        recall_pipeline,
        f1_pipeline,
        roc_auc_pipeline,
    ) = ([], [], [], [], [])
    (
        accuracy_baseline,
        precision_baseline,
        recall_baseline,
        f1_baseline,
        roc_auc_baseline,
    ) = ([], [], [], [], [])

    for features_baseline, features_pipeline in zip(
        selected_features_baseline, selected_features_pipeline
    ):
        # pipeline
        X_pipeline = data[features_pipeline]
        X_train, X_test, y_train, y_test = train_test_split(
            X_pipeline, y, test_size=0.2
        )
        classifier.fit(X_train, y_train)
        y_pred_pipeline = classifier.predict(X_test)

        # baseline
        X_baseline = data[features_baseline]
        X_train, X_test, y_train, y_test = train_test_split(
            X_baseline, y, test_size=0.2
        )
        classifier.fit(X_train, y_train)
        y_pred_baseline = classifier.predict(X_test)

        # Calculate performance metrics for pipeline
        accuracy_pipeline.append(accuracy_score(y_test, y_pred_pipeline))
        precision_pipeline.append(precision_score(y_test, y_pred_pipeline))
        recall_pipeline.append(recall_score(y_test, y_pred_pipeline))
        f1_pipeline.append(f1_score(y_test, y_pred_pipeline))
        roc_auc_pipeline.append(
            roc_auc_score(
                label_binarize(y_test, classes=[0, 1]),
                label_binarize(y_pred_pipeline, classes=[0, 1]),
            )
        )

        # Calculate performance metrics for baseline
        accuracy_baseline.append(accuracy_score(y_test, y_pred_baseline))
        precision_baseline.append(precision_score(y_test, y_pred_baseline))
        recall_baseline.append(recall_score(y_test, y_pred_baseline))
        f1_baseline.append(f1_score(y_test, y_pred_baseline))
        roc_auc_baseline.append(
            roc_auc_score(
                label_binarize(y_test, classes=[0, 1]),
                label_binarize(y_pred_baseline, classes=[0, 1]),
            )
        )

    # Compute average metrics
    avg_accuracy_pipeline = np.mean(accuracy_pipeline)
    avg_precision_pipeline = np.mean(precision_pipeline)
    avg_recall_pipeline = np.mean(recall_pipeline)
    avg_f1_pipeline = np.mean(f1_pipeline)
    avg_roc_auc_pipeline = np.mean(roc_auc_pipeline)

    avg_accuracy_baseline = np.mean(accuracy_baseline)
    avg_precision_baseline = np.mean(precision_baseline)
    avg_recall_baseline = np.mean(recall_baseline)
    avg_f1_baseline = np.mean(f1_baseline)
    avg_roc_auc_baseline = np.mean(roc_auc_baseline)

    print(
        f"""Average Performance Metrics using features selected by pipeline:
      Accuracy: {avg_accuracy_pipeline}, Precision: {avg_precision_pipeline}, Recall: {avg_recall_pipeline},
      F1 Score: {avg_f1_pipeline}, AUC-ROC: {avg_roc_auc_pipeline}
      Number of features (average across runs): {np.mean([len(features) for features in selected_features_pipeline])}

Average Performance Metrics using features selected by baseline:
      Accuracy: {avg_accuracy_baseline}, Precision: {avg_precision_baseline}, Recall: {avg_recall_baseline},
      F1 Score: {avg_f1_baseline}, AUC-ROC: {avg_roc_auc_baseline}
      Number of features (average across runs): {np.mean([len(features) for features in selected_features_baseline])}"""
    )


def run_feature_selection(
    data, method, num_runs=5, train_size=0.8, num_features_to_select=10
):
    selected_features_list = []

    for _ in range(num_runs):
        # Randomly split the data
        _, data_subset, _, _ = train_test_split(
            data, train_size=train_size, stratify=data["target"], random_state=None
        )

        # Extract features and labels
        X_subset = data_subset.drop("target", axis=1)
        y_subset = data_subset["target"]

        # Perform feature selection
        if method == "pipeline":
            classifier = "RF"
            fs_methods = [
                feature_selection_infogain,
                feature_selection_xgboost,
                feature_selection_random_forest,
                feature_selection_chi2,
            ]
            merging_strategy = merging_strategy_union_of_pairwise_intersections
            num_repeats = 3
            pipeline = FeatureSelectionPipeline(
                data_subset, fs_methods, merging_strategy, classifier, num_repeats
            )
            selected_features, _, _ = pipeline.iterate_pipeline()

        elif method == "baseline":
            baseline_feature_selection = feature_selection_chi2
            _, baseline_idx = baseline_feature_selection(
                X=X_subset, y=y_subset, num_features_to_select=num_features_to_select
            )
            selected_features = data.columns[baseline_idx].to_list()
        else:
            raise ValueError("Invalid feature selection method")

        selected_features_list.append(set(selected_features))

    return selected_features_list


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def compare_robustness(baseline_selected_features, pipeline_selected_features):
    jaccard_similarities = []

    for baseline_features, pipeline_features in zip(
        baseline_selected_features, pipeline_selected_features
    ):
        similarity = jaccard_similarity(baseline_features, pipeline_features)
        jaccard_similarities.append(similarity)

    mean_similarity = np.mean(jaccard_similarities)
    print(f"Mean Jaccard Similarity across runs: {mean_similarity}")


if __name__ == "main":
    cores_to_use = 64  # Change this number to the desired core count
    os.sched_setaffinity(0, range(cores_to_use))

    # read and process dataset

    data = (
        pd.read_csv("breast_cancer_data/Breast_GSE45827.csv")
        .rename(columns={"type": "target"})
        .drop(columns="samples")
    )
    label_encoder = LabelEncoder()
    data["target"] = label_encoder.fit_transform(data["target"])
    target_column = data["target"]
    columns_to_normalize = data.drop("target", axis=1)
    scaler = MinMaxScaler()
    data[columns_to_normalize.columns] = scaler.fit_transform(columns_to_normalize)
    data = pd.concat([target_column, data], axis=1)

    pipeline_selected_features = run_feature_selection(
        data, method="pipeline", num_runs=5, train_size=0.8, num_features_to_select=10
    )
    baseline_selected_features = run_feature_selection(
        data,
        method="baseline",
        num_runs=5,
        train_size=0.8,
        num_features_to_select=len(pipeline_selected_features),
    )

    compare_robustness(baseline_selected_features, pipeline_selected_features)
    compare_performance(baseline_selected_features, pipeline_selected_features, data)
