from sklearn.feature_selection import chi2, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import RFE
from sklearn.base import clone

from sklearn.feature_selection import *
import numpy as np


def feature_selection_f_statistic(X, y, task='classification', num_features_to_select=None):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Choose the F-statistic function based on the task type
    if task == 'classification':
        f_stat_func = f_classif
    elif task == 'regression':
        f_stat_func = f_regression

    selector = SelectKBest(score_func=f_stat_func, k=num_features_to_select)
    selector.fit(X, y)
    feature_scores = selector.scores_
    selected_features_indices = feature_scores.argsort()[::-1][:num_features_to_select]

    return feature_scores, selected_features_indices



def feature_selection_mutual_info(X, y, task='classification', num_features_to_select=None):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    if task == 'classification':
        mutual_info_func = mutual_info_classif
    elif task == 'regression':
        mutual_info_func = mutual_info_regression
    else:
        raise ValueError("Invalid task type. Please specify either 'classification' or 'regression'.")

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
    selected_features_indices = np.argsort(feature_scores)[::-1][:num_features_to_select]

    return feature_scores, selected_features_indices



#Error with this function, selected_features_indices are not the good feature list, need to understand
#how to FS with SVM.


def feature_selection_svm(X, y, num_features_to_select=None, **kwargs):

    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Initialize the SVM classifier
    svm = SVC(kernel='linear', **kwargs)
    feature_selector = SelectFromModel(svm)
    feature_selector.fit(X, y)
    importances = feature_selector.estimator_.coef_.reshape(-1)
    selected_features_indices = importances.argsort()[::-1][:num_features_to_select]
    print(selected_features_indices)

    return None, selected_features_indices



def feature_selection_random_forest(X, y, task='classification', num_features_to_select=None, **kwargs):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Initialize the Random Forest model based on task type
    if task == 'classification':
        model = RandomForestClassifier(**kwargs)
    elif task == 'regression':
        model = RandomForestRegressor(**kwargs)
    else:
        raise ValueError("Invalid task type. Please specify either 'classification' or 'regression'.")

    # Use Random Forest-based feature selection
    feature_selector = SelectFromModel(model)
    feature_selector.fit(X, y)

    feature_scores = feature_selector.estimator_.feature_importances_
    selected_features_indices = feature_scores.argsort()[::-1][:num_features_to_select]

    return feature_scores, selected_features_indices



def feature_selection_xgboost(X, y, task='classification', num_features_to_select=None, **kwargs):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Initialize the XGBoost model based on task type
    if task == 'classification':
        model = XGBClassifier(**kwargs)
    elif task == 'regression':
        model = XGBRegressor(**kwargs)
    else:
        raise ValueError("Invalid task type. Please specify either 'classification' or 'regression'.")

    # Use XGBoost-based feature selection
    feature_selector = SelectFromModel(model)
    feature_selector.fit(X, y)

    feature_scores = feature_selector.estimator_.feature_importances_
    selected_features_indices = feature_scores.argsort()[::-1][:num_features_to_select]

    return feature_scores, selected_features_indices


def feature_selection_rfe_rf(X, y, task='classification', num_features_to_select=None, **kwargs):
    if num_features_to_select is None:
        num_features_to_select = int(0.1 * X.shape[1])

    # Initialize the Random Forest model based on task type
    if task == 'classification':
        model = RandomForestClassifier(**kwargs)
    elif task == 'regression':
        model = RandomForestRegressor(**kwargs)
    else:
        raise ValueError("Invalid task type. Please specify either 'classification' or 'regression'.")

    rfe = RFE(model, n_features_to_select=num_features_to_select)
    rfe.fit(X, y)

    selected_features_indices = rfe.support_

    return None, selected_features_indices

# WHAT TO DO WITH FUNCTION THAT DOES NOT DIRECTLY PROVIDE FEATURE IMPORTANCE
# WHY DID I RETURN IMPORTANCES ??
# MAYBE ONLY BECAUSE OF RANKING METHODS THAT NEED ALL ?
# IF YES SOLUTION COULD BE TO ONLY IMPLEMENT PAIRWISEOFUNION

