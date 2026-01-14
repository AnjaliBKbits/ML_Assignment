import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, matthews_corrcoef,
                                 precision_score, recall_score, f1_score,
                                 confusion_matrix, classification_report)

def run_decision_tree_classifier(data):
    fault_columns = [
        "Pastry",
        "Z_Scratch",
        "K_Scatch",
        "Stains",
        "Dirtiness",
        "Bumps",
        "Other_Faults"
    ]

    #Creating copy of dataset
    data = data.copy()

    #Finding the column with value 1 and using it as class label
    data["Target"] = data[fault_columns].idxmax(axis=1)

    #Seperate features and target
    X = data.drop(columns = fault_columns + ["Target"])

    y = data["Target"]

    #Encode target variable
    y = LabelEncoder().fit_transform(y)

    #Train-test data split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  #For class balance
    )

    #Decision tree model
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None, #tree grows fully"
        random_state=42
    )

    #Train the model
    model.fit(X_train,y_train)

    #Prediction for test data
    y_pred = model.predict(X_test)

    #Compute class probability for AUC score
    y_prob = model.predict_proba(X_test)

    #metrics stored in dictionary
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_score": roc_auc_score(y_test, y_prob, multi_class="ovr"),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    #Show correct vs incorrect predictions per class
    cm = confusion_matrix(y_test, y_pred)

    #Generates precision, recall , F1_score for each class
    report = classification_report(y_test, y_pred)

    return metrics, cm, report
