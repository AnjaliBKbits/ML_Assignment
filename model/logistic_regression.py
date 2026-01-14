from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, matthews_corrcoef,
                                 precision_score, recall_score, f1_score,
                                 confusion_matrix, classification_report)

def run_logistic_regression(data,fault_columns):

    # Create SINGLE multiclass target
    data = data.copy()
    data["Target"] = data[fault_columns].idxmax(axis=1)

    # Drop original fault columns
    X = data.drop(columns=fault_columns + ["Target"])
    y = data["Target"]

    #Check if target is categorical
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Adding logistic regression model and setting max iterations to ensure convergence
    model = LogisticRegression(max_iter=1000)

    #Training the model
    model.fit(X_train, y_train)

    #Making predictions
    y_pred = model.predict(X_test)

    #Predict probability score
    y_probab = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_score": roc_auc_score(y_test, y_probab, multi_class="ovr"),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }
    
    #Show correct vs incorrect predictions per class
    cfm = confusion_matrix(y_test, y_pred)

    #Generates precision, recall , F1_score for each class
    creport =  classification_report(y_test, y_pred)
    return metrics, cfm, creport


