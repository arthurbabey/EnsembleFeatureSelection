from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB

import rpy2
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector, StrVector, ListVector
stabm = importr('stabm')

def compute_performance_metrics(classifier, X_train, y_train, X_test, y_test):
    results = {}

    # Initialize classifier based on user input
    if classifier == 'RF':
        clf = RandomForestClassifier()
    elif classifier == 'naiveBayes':
        clf = GaussianNB()
    elif classifier == 'bagging':
        clf = BaggingClassifier()
    else:
        raise ValueError(
            f"Invalid classifier name '{classifier}'. Please choose among 'RF', 'naiveBayes', or 'bagging'.")

    # Fit the classifier and make predictions
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Compute AUROC (if classifier supports predict_proba)
    if hasattr(clf, "predict_proba"):
        num_classes = len(np.unique(y_test))
        if num_classes == 2:
            # ASSUME POSITIVE CLASS ARE STORED IN THE SECOND COLUMN
            y_proba = clf.predict_proba(X_test)[:, 1]  # Assuming positive class probabilities
            auroc = roc_auc_score(y_test, y_proba)
        else:  # Multi-class classification
            y_proba = clf.predict_proba(X_test)
            auroc = roc_auc_score(y_test, y_proba, multi_class='ovo')
    else:
        auroc = None  # Not applicable for classifiers that don't have predict_proba

    # Compute MAE
    mae = mean_absolute_error(y_test, y_pred)

    # Store results for current classifier
    results = {'accuracy': accuracy, 'AUROC': auroc, 'MAE': mae}

    return results


def compute_stability_metrics(features_list):
    # Determine the type of the first element in the list
    first_element = features_list[0][0]
    if isinstance(first_element, int):
        r_list_of_lists = ListVector({
            f"dataset_{i+1}": IntVector(inner_list) for i, inner_list in enumerate(features_list)
        })
    elif isinstance(first_element, str):
        r_list_of_lists = ListVector({
            f"dataset_{i+1}": StrVector(inner_list) for i, inner_list in enumerate(features_list)
        })
    else:
        raise ValueError("Unsupported type found in the given list")

    result = stabm.stabilityNovovicova(features=r_list_of_lists)
    return result


