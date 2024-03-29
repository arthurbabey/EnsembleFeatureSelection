import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import *
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    chi2,
    mutual_info_classif,
)
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor


def feature_selection_f_statistic(
    X, y, task="classification", num_features_to_select=None
):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Choose the F-statistic function based on the task type
    if task == "classification":
        f_stat_func = f_classif
    elif task == "regression":
        f_stat_func = f_regression
    selector = SelectKBest(score_func=f_stat_func, k=num_features_to_select)
    selector.fit(X, y)
    feature_scores = selector.scores_
    selected_features_indices = feature_scores.argsort()[::-1][:num_features_to_select]

    return feature_scores, selected_features_indices


def feature_selection_mutual_info(
    X, y, task="classification", num_features_to_select=None
):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    if task == "classification":
        mutual_info_func = mutual_info_classif
    elif task == "regression":
        mutual_info_func = mutual_info_regression
    else:
        raise ValueError(
            "Invalid task type. Please specify either 'classification' or 'regression'."
        )

    selector = SelectKBest(score_func=mutual_info_func, k=num_features_to_select)
    selector.fit(X, y)
    feature_scores = selector.scores_
    selected_features_indices = feature_scores.argsort()[::-1][:num_features_to_select]

    return feature_scores, selected_features_indices


# might not be useful in this context
def feature_selection_chi2(X, y, task=None, num_features_to_select=None):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    chi2_selector = SelectKBest(score_func=chi2, k=num_features_to_select)
    chi2_selector.fit(X, y)

    feature_scores = chi2_selector.scores_  # Get scores for each feature
    selected_features_indices = np.argsort(feature_scores)[::-1][
        :num_features_to_select
    ]

    return feature_scores, selected_features_indices


# Error with this function, selected_features_indices are not the good feature list, need to understand
# how to FS with SVM.
# now this version works but probably only for linear kernel


def feature_selection_svm(
    X, y, task="classification", num_features_to_select=None, **kwargs
):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Choose the SVM model based on the task type
    if task == "classification":
        model = SVC(kernel="linear", **kwargs)
    elif task == "regression":
        model = SVR(kernel="linear", **kwargs)
    else:
        raise ValueError(
            "Invalid task type. Please specify either 'classification' or 'regression'."
        )

    # Initialize SelectFromModel with SVM
    feature_selector = SelectFromModel(model)
    feature_selector.fit(X, y)

    feature_scores = np.abs(feature_selector.estimator_.coef_).mean(axis=0)
    selected_features_indices = feature_scores.argsort()[::-1][:num_features_to_select]

    return feature_scores, selected_features_indices


def feature_selection_random_forest(
    X, y, task="classification", num_features_to_select=None, **kwargs
):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Initialize the Random Forest model based on task type
    if task == "classification":
        model = RandomForestClassifier(**kwargs)
    elif task == "regression":
        model = RandomForestRegressor(**kwargs)
    else:
        raise ValueError(
            "Invalid task type. Please specify either 'classification' or 'regression'."
        )

    # Use Random Forest-based feature selection
    feature_selector = SelectFromModel(model)
    feature_selector.fit(X, y)

    feature_scores = feature_selector.estimator_.feature_importances_
    selected_features_indices = feature_scores.argsort()[::-1][:num_features_to_select]

    return feature_scores, selected_features_indices


def feature_selection_xgboost(
    X, y, task="classification", num_features_to_select=None, **kwargs
):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Initialize the XGBoost model based on task type
    if task == "classification":
        model = XGBClassifier(**kwargs)
    elif task == "regression":
        model = XGBRegressor(**kwargs)
    else:
        raise ValueError(
            "Invalid task type. Please specify either 'classification' or 'regression'."
        )

    # Use XGBoost-based feature selection
    feature_selector = SelectFromModel(model)
    feature_selector.fit(X, y)

    feature_scores = feature_selector.estimator_.feature_importances_
    selected_features_indices = feature_scores.argsort()[::-1][:num_features_to_select]

    return feature_scores, selected_features_indices


def feature_selection_rfe_rf(
    X, y, task="classification", num_features_to_select=None, **kwargs
):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Initialize the Random Forest model based on task type
    if task == "classification":
        model = RandomForestClassifier(**kwargs)
    elif task == "regression":
        model = RandomForestRegressor(**kwargs)
    else:
        raise ValueError(
            "Invalid task type. Please specify either 'classification' or 'regression'."
        )

    rfe = RFE(model, n_features_to_select=num_features_to_select)
    rfe.fit(X, y)

    selected_features_indices = np.where(rfe.support_)[0]

    return None, selected_features_indices


# WHAT TO DO WITH FUNCTION THAT DOES NOT DIRECTLY PROVIDE FEATURE IMPORTANCE
# WHY DID I RETURN IMPORTANCES ??
# MAYBE ONLY BECAUSE OF RANKING METHODS THAT NEED ALL ?
# IF YES SOLUTION COULD BE TO ONLY IMPLEMENT PAIRWISEOFUNION
