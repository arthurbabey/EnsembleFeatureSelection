from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

def compare_performance(feature_pipeline, feature_baseline, data):
    y = data['target']
    classifier = SVC(kernel='rbf', C=1.0, gamma='scale')

    # pipeline
    X = data[feature_pipeline]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    classifier.fit(X_train, y_train)
    y_pred_1 = classifier.predict(X_test)

    # baseline
    X = data[feature_baseline]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    classifier.fit(X_train, y_train)
    y_pred_2 = classifier.predict(X_test)

    accuracy_1 = accuracy_score(y_test, y_pred_1)
    precision_1 = precision_score(y_test, y_pred_1)
    recall_1 = recall_score(y_test, y_pred_1)
    f1_1 = f1_score(y_test, y_pred_1)
    roc_auc_1 = roc_auc_score(label_binarize(y_test, classes=[0, 1]), label_binarize(y_pred_1, classes=[0, 1]))

    accuracy_2 = accuracy_score(y_test, y_pred_2)
    precision_2 = precision_score(y_test, y_pred_2)
    recall_2 = recall_score(y_test, y_pred_2)
    f1_2 = f1_score(y_test, y_pred_2)
    roc_auc_2 = roc_auc_score(label_binarize(y_test, classes=[0, 1]), label_binarize(y_pred_2, classes=[0, 1]))

    print(f"""Performance Metrics using features selected by pipeline:
      Accuracy: {accuracy_1}, Precision: {precision_1}, Recall: {recall_1}, F1 Score: {f1_1}, AUC-ROC: {roc_auc_1}
      Number of features: {len(feature_pipeline)}

Performance Metrics using features selected by baseline:
      Accuracy: {accuracy_2}, Precision: {precision_2}, Recall: {recall_2}, F1 Score: {f1_2}, AUC-ROC: {roc_auc_2}
      Number of features: {len(feature_baseline)}""")

    return accuracy_1, accuracy_2, precision_1, precision_2, recall_1, recall_2, f1_1, f1_2, roc_auc_1, roc_auc_2


