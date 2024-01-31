from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB

import rpy2
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector, StrVector, ListVector
stabm = importr('stabm')

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

from sklearn.metrics import mean_squared_error
import numpy as np

def compute_performance_metrics(classifier, task, X_train, y_train, X_test, y_test):
    results = {}

    # Initialize classifier based on user input
    if classifier == 'RF':
        if task == 'classification':
            clf = RandomForestClassifier()
        elif task == 'regression':
            clf = RandomForestRegressor()
        else:
            raise ValueError("Invalid task type")
    elif classifier == 'naiveBayes':
        if task == 'classification':
            clf = GaussianNB()
        else:
            raise ValueError("Naive Bayes is not suitable for regression task")
    elif classifier == 'bagging':
        if task == 'classification':
            clf = BaggingClassifier()
        elif task == 'regression':
            clf = BaggingRegressor()
        else:
            raise ValueError("Invalid task type")
    else:
        raise ValueError(
            f"Invalid classifier name '{classifier}'. Please choose among 'RF', 'naiveBayes', or 'bagging'.")

    # Fit the classifier and make predictions
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute appropriate performance metrics based on task type
    if task == 'classification':
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

        # Store results for classification
        results = {'accuracy': accuracy, 'AUROC': auroc, 'MAE': mae}
    elif task == 'regression':
        # Compute MAE, R-squared, and RMSE
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Store results for regression
        results = {'MAE': mae, 'R-squared': r2, 'RMSE': rmse}
    else:
        raise ValueError("Invalid task type")

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


