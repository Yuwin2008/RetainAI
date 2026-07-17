import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load("model/gb_model.pkl")

st.set_page_config(
    page_title="RetainAI",
    layout="wide"
)

st.title("RetainAI")

st.markdown(
"""
### Predict customer churn before it happens.

Use machine learning to identify customers at risk of leaving and take proactive retention actions.
"""
)

st.sidebar.header(
    "Customer Information"
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    72,
    12
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges",
    0.0,
    150.0,
    70.0
)

contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    ["No","Yes"]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["No","Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependent",
    ["No","Yes"]
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["No","Yes"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No","Yes","No Phone Service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL","Fiber Optic","No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes","No","No Internet Service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes","No","No Internet Service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes","No","No Internet Service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes","No","No Internet Service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes","No","No Internet Service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes","No","No Internet Service"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes","No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

if st.button("Predict Churn"):

    total_charges = tenure * monthly_charges

    if monthly_charges < 35:
        monthly_category = "Low"
    elif monthly_charges < 70:
        monthly_category = "Medium"
    else:
        monthly_category = "High"

    if tenure < 12:
        tenure_group = "New"
    elif tenure < 48:
        tenure_group = "Regular"
    else:
        tenure_group = "Loyal"

    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "MonthlyCategory": [monthly_category],
        "TenureGroup": [tenure_group]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    churn_probability = probability * 100

    if churn_probability < 30:
        risk_level = " Low Risk"
    elif churn_probability < 70:
        risk_level = " Medium Risk"
    else:
        risk_level = " High Risk"

    st.metric(
        "Churn Probability",
        f"{churn_probability:.2f}%"
    )

    st.info(
        f"Risk Level: {risk_level}"
    )

    st.subheader("Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Gender", gender)
        st.metric("Tenure", f"{tenure} Months")

    with col2:
        st.metric("Contract", contract)
        st.metric("Internet", internet_service)

    with col3:
        st.metric(
            "Monthly Charges",
            f"${monthly_charges:.2f}"
        )
        st.metric(
            "Payment Method",
            payment_method
        )

    if prediction == 1:
        st.error(
            f" Customer likely to churn ({churn_probability:.2f}%)"
        )
    else:
        st.success(
            f" Customer likely to stay ({100 - churn_probability:.2f}%)"
        )

    X_processed = model.named_steps[
        "preprocessor"
    ].transform(input_df)

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    explainer = shap.Explainer(
        model.named_steps["model"]
    )

    shap_values = explainer(X_processed)

    shap_values.feature_names = feature_names

    st.subheader(
        "Why did the model make this prediction?"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    shap.plots.waterfall(
        shap_values[0],
        show=False
    )

    st.pyplot(fig)

    recommendations = []

    if contract == "Month-to-month":
        recommendations.append(
            "Offer a discounted yearly contract."
        )

    if tech_support == "No":
        recommendations.append(
            "Offer premium technical support."
        )

    if payment_method == "Electronic check":
        recommendations.append(
            "Encourage switching to automatic payments."
        )

    if tenure < 12:
        recommendations.append(
            "Provide loyalty incentives for new customers."
        )

    st.subheader(
        "Retention Strategy"
    )

    if recommendations:
        for rec in recommendations:
            st.write(f"• {rec}")
    else:
        st.success(
            "No immediate retention action required."
        )