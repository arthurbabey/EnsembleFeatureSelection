from sklearn.feature_selection import chi2, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
import numpy as np

def feature_selection_infogain(X, y, threshold):
    kbest_selector = SelectKBest(score_func=mutual_info_classif, k=threshold)
    kbest_selector.fit(X, y)

    feature_scores = kbest_selector.scores_  # Get scores for each feature
    selected_features_indices = np.argsort(feature_scores)[::-1][:threshold]
    return feature_scores, selected_features_indices

def feature_selection_chi2(X, y, threshold):
    chi2_selector = SelectKBest(score_func=chi2, k=threshold)
    chi2_selector.fit(X, y)

    feature_scores = chi2_selector.scores_  # Get scores for each feature
    selected_features_indices = np.argsort(feature_scores)[::-1][:threshold]
    return feature_scores, selected_features_indices


"""
def feature_selection_svm(X, y, k_features, **kwargs):
    # Initialize the SVM classifier
    svm = SVC(**kwargs)
    feature_selector = SelectFromModel(svm)
    feature_selector.fit(X, y)
    importances = feature_selector.estimator_.coef_.reshape(-1)
    selected_features = importances.argsort()[::-1][:num_features]

    return selected_features
"""

def feature_selection_random_forest(X, y, threshold, **kwargs):
    # Initialize the Random Forest classifier
    rf = RandomForestClassifier(**kwargs)

    # Use Random Forest-based feature selection
    feature_selector = SelectFromModel(rf)
    feature_selector.fit(X, y)

    feature_scores = feature_selector.estimator_.feature_importances_
    selected_features_indices = feature_scores.argsort()[::-1][:threshold]
    return feature_scores, selected_features_indices


def feature_selection_xgboost(X, y, threshold, **kwargs):
    # Initialize the XGBoost classifier
    xgb = XGBClassifier(**kwargs)

    # Use XGBoost-based feature selection
    feature_selector = SelectFromModel(xgb)
    feature_selector.fit(X, y)

    feature_scores = feature_selector.estimator_.feature_importances_
    selected_features_indices = feature_scores.argsort()[::-1][:threshold]
    return feature_scores, selected_features_indices


