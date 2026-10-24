
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_files():
    model = joblib.load("churn_model.pkl")
    preprocessor = joblib.load("churn_preprocessor.pkl")
    threshold = joblib.load("churn_threshold.pkl")
    return model, preprocessor, float(threshold)

model, preprocessor, threshold = load_files()

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to estimate whether the customer "
    "is likely to leave the service."
)

st.info(f"Model prediction threshold: {threshold:.2f}")

with st.form("customer_form"):
    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=100, value=12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges", min_value=0.0, value=70.0, step=1.0
        )
        total = st.number_input(
            "Total Charges", min_value=0.0, value=840.0, step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    try:
        transformed = preprocessor.transform(customer)
        probability = float(model.predict_proba(transformed)[0][1])

        st.subheader("Prediction Result")
        st.metric("Estimated Churn Probability", f"{probability:.1%}")

        if probability >= threshold:
            st.error("⚠️ This customer is predicted to churn.")
            st.write(
                "Consider reviewing the customer's contract, service "
                "experience, and support needs."
            )
        else:
            st.success("✅ This customer is predicted not to churn.")

        st.caption(
            "This is a model estimate, not a guarantee of customer behavior."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
