"""
Customer Churn Prediction — Deployment Demo (Streamlit)
--------------------------------------------------------
Loads the trained pipeline (preprocessing + SMOTE + tuned XGBoost) saved by
the notebook as `churn_model_pipeline.pkl` and serves an interactive form
that returns a live churn prediction + probability for any customer profile.

Run locally with:
    pip install streamlit pandas scikit-learn xgboost imbalanced-learn joblib
    streamlit run app.py

This is a lightweight demo of how the model would sit behind a UI or,
equivalently, behind a FastAPI/Flask REST endpoint in production.
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "churn_model_pipeline.pkl"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_churn(customer: dict, model) -> dict:
    """Score a single customer record and return prediction + probability."""
    input_df = pd.DataFrame([customer])
    input_df["AvgChargePerMonth"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)
    proba = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]
    return {
        "churn_prediction": "Yes" if pred == 1 else "No",
        "churn_probability": round(float(proba), 4),
    }


st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's profile to get a live churn prediction from the "
    "trained XGBoost model (see the accompanying notebook for how it was built)."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_PATH}`. Run the notebook "
        "(`Customer_Churn_Prediction.ipynb`) first — the final model-saving "
        "cell writes this file, and this app must be run from the same folder."
    )
    st.stop()

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    result = predict_churn(customer, model)

    st.subheader("Result")
    if result["churn_prediction"] == "Yes":
        st.error(f"⚠️ High churn risk — probability: {result['churn_probability']:.1%}")
    else:
        st.success(f"✅ Low churn risk — probability: {result['churn_probability']:.1%}")

    st.progress(result["churn_probability"])
