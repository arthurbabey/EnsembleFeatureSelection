from sklearn.feature_selection import chi2, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.base import clone

from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
import numpy as np

def feature_selection_infogain(X, y, num_features_to_select=None):

    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    kbest_selector = SelectKBest(score_func=mutual_info_classif, k=num_features_to_select)
    kbest_selector.fit(X, y)

    feature_scores = kbest_selector.scores_  # Get scores for each feature
    selected_features_indices = np.argsort(feature_scores)[::-1][:num_features_to_select]
    return feature_scores, selected_features_indices


def feature_selection_chi2(X, y, num_features_to_select=None):

    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    chi2_selector = SelectKBest(score_func=chi2, k=num_features_to_select)
    chi2_selector.fit(X, y)

    feature_scores = chi2_selector.scores_  # Get scores for each feature
    selected_features_indices = np.argsort(feature_scores)[::-1][:num_features_to_select]
    return feature_scores, selected_features_indices


def feature_selection_svm(X, y, num_features_to_select=None, **kwargs):

    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Initialize the SVM classifier
    svm = SVC(**kwargs)
    feature_selector = SelectFromModel(svm)
    feature_selector.fit(X, y)
    importances = feature_selector.estimator_.coef_.reshape(-1)
    selected_features = importances.argsort()[::-1][:num_features_to_select]

    return selected_features, importances

def feature_selection_random_forest(X, y, num_features_to_select=None, **kwargs):

    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Initialize the Random Forest classifier
    rf = RandomForestClassifier(**kwargs)

    # Use Random Forest-based feature selection
    feature_selector = SelectFromModel(rf)
    feature_selector.fit(X, y)

    feature_scores = feature_selector.estimator_.feature_importances_
    selected_features_indices = feature_scores.argsort()[::-1][:num_features_to_select]
    return feature_scores, selected_features_indices


def feature_selection_xgboost(X, y, num_features_to_select=None, **kwargs):

    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Initialize the XGBoost classifier
    xgb = XGBClassifier(**kwargs)

    # Use XGBoost-based feature selection
    feature_selector = SelectFromModel(xgb)
    feature_selector.fit(X, y)

    feature_scores = feature_selector.estimator_.feature_importances_
    selected_features_indices = feature_scores.argsort()[::-1][:num_features_to_select]
    return feature_scores, selected_features_indices

def feature_selection_rfe_rf(X, y, num_features_to_select=None, **kwargs):

    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    rf = RandomForestClassifier(**kwargs)
    rfe = RFE(rf, n_features_to_select=num_features_to_select)
    rfe.fit(X, y)

    selected_features_indices = rfe.support_

    return None, selected_features_indices

# WHAT TO DO WITH FUNCTION THAT DOES NOT DIRECTLY PROVIDE FEATURE IMPORTANCE
# WHY DID I RETURN IMPORTANCES ??
# MAYBE ONLY BECAUSE OF RANKING METHODS THAT NEED ALL ?
# IF YES SOLUTION COULD BE TO ONLY IMPLEMENT PAIRWISEOFUNION

