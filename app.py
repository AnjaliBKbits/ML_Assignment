import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model.logistic_regression import run_logistic_regression
from model.decision_tree_classifier import run_decision_tree_classifier
from model.knn_classifier import run_knn_classifier

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2",
    layout="wide"
)

# -------------------------------------------------
# Header Section
# -------------------------------------------------
st.title("Machine Learning Classification Dashboard")
st.markdown(""" 
**Interactive web application for evaluating classification models.**
""")

st.divider()

# -------------------------------------------------
# Sidebar – User Controls
# -------------------------------------------------
st.sidebar.header("Controls")

# Dataset upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# Run button
run_button = st.sidebar.button("Run Model")

# -------------------------------------------------
# Main Layout
# -------------------------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    st.markdown(f"**Selected Model:** `{model_choice}`")

    fault_columns = [
        "Pastry",
        "Z_Scratch",
        "K_Scatch",
        "Stains",
        "Dirtiness",
        "Bumps",
        "Other_Faults"
    ]

    if run_button:

        if model_choice == "Logistic Regression":
            metrics, cfm, creport = run_logistic_regression(data,fault_columns)
        elif model_choice == "Decision Tree":
            metrics, cfm, creport = run_decision_tree_classifier(data,fault_columns)
        elif model_choice == "K-Nearest Neighbors":
            metrics, cfm, creport = run_knn_classifier(data,fault_columns)
        else:
            st.warning("Selected model is not yet implemented.")
            st.stop()
        
        st.success(f"{model_choice} model executed successfully!")

        st.divider()

        # -------------------------------------------------
        # Metrics Section (Placeholder)
        # -------------------------------------------------
        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("AUC Score", f"{metrics['auc_score']:.4f}")
        col3.metric("MCC", f"{metrics['mcc']:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", f"{metrics['precision']:.4f}")
        col5.metric("Recall", f"{metrics['recall']:.4f}")
        col6.metric("F1 Score", f"{metrics['f1_score']:.4f}")

        st.divider()

        # -------------------------------------------------
        # Confusion Matrix Placeholder
        # -------------------------------------------------
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')    
        ax.set_ylabel('True Label')
        st.pyplot(fig)

        st.divider()

        # -------------------------------------------------
        # Classification Report Placeholder
        # -------------------------------------------------
        st.subheader("Classification Report")
        st.code(creport)

else:
    st.warning("Please upload a CSV file to begin.")

