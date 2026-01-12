import streamlit as st
import pandas as pd

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

    if run_button:
        st.success("Model executed successfully!")

        st.divider()

        # -------------------------------------------------
        # Metrics Section (Placeholder)
        # -------------------------------------------------
        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", "—")
        col2.metric("AUC Score", "—")
        col3.metric("MCC", "—")

        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", "—")
        col5.metric("Recall", "—")
        col6.metric("F1 Score", "—")

        st.divider()

        # -------------------------------------------------
        # Confusion Matrix Placeholder
        # -------------------------------------------------
        st.subheader("Confusion Matrix")
        st.info("Confusion matrix will be displayed here.")

        st.divider()

        # -------------------------------------------------
        # Classification Report Placeholder
        # -------------------------------------------------
        st.subheader("Classification Report")
        st.code("""
Class  Precision  Recall  F1-score
----------------------------------
Class 0     —        —        —
Class 1     —        —        —
""")

else:
    st.warning("Please upload a CSV file to begin.")

