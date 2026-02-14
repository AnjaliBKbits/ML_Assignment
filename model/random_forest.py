import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def run_random_forest_classifier(data, fault_columns):

    # Copy dataset
    data = data.copy()

    # Create single target from fault columns
    data["Target"] = data[fault_columns].idxmax(axis=1)

    # Separate features and target
    X = data.drop(columns=fault_columns + ["Target"])
    y = data["Target"]

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train–test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Random Forest model with imbalance handling
    model = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Safe ROC-AUC calculation for multiclass
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    except ValueError:
        auc = 0.0

    # Metrics dictionary
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_score": auc,
        "mcc": matthews_corrcoef(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    # Confusion matrix with stable label order
    cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))

    # Classification report with original class names
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_
    )

    return metrics, cm, report
