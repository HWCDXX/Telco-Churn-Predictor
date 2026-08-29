import os
import plotly.graph_objects as go
import requests
import streamlit as st

# Streamlit Page Configuration
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configurable FastAPI Endpoint (Sanitizes trailing slashes for Cloud / Docker envs)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def check_backend_health():
    """Verify connectivity to FastAPI inference service."""
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return res.status_code == 200, res.json()
    except Exception:
        return False, {}


# App Header
st.title("🔮 Telco Customer Churn Predictor")
st.markdown(
    "Evaluate customer churn probability in real time using your trained XGBoost model."
)

# Backend Health Status Indicator
is_healthy, health_data = check_backend_health()
if is_healthy:
    st.sidebar.success(f"🟢 Connected to FastAPI (`{BACKEND_URL}`)")
else:
    st.sidebar.error(
        f"🔴 Backend Disconnected (`{BACKEND_URL}`). Ensure FastAPI container is running."
    )

st.markdown("---")

# Feature Input Form UI
st.subheader("📋 Customer Profile & Service Selection")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox(
        "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
    )
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)

with col2:
    st.markdown("#### 🌐 Subscribed Services")
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security = st.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )

with col3:
    st.markdown("#### 💳 Account & Financials")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=18.0,
        max_value=150.0,
        value=70.0,
        step=1.0,
    )
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=18.0,
        max_value=9000.0,
        value=float(tenure * monthly_charges),
        step=10.0,
    )

# Format JSON Payload matching CustomerPayload Schema
payload = {
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

st.markdown("---")

# Submit & Predict
predict_btn = st.button("🚀 Calculate Churn Probability", use_container_width=True)

if predict_btn:
    if not is_healthy:
        st.error("Cannot complete prediction: Backend server is currently offline.")
    else:
        with st.spinner("Analyzing risk profile..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict", json=payload, timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    # Fallback key resolution
                    raw_prob = result.get("raw_output")
                    if raw_prob is None:
                        raw_prob = result.get("churn_probability", 0.0)
                    probability = float(raw_prob)

                    raw_class = result.get("churn_class")
                    if raw_class is None:
                        raw_class = result.get("prediction", 0)
                    prediction = int(raw_class)

                    st.markdown("## 📊 Model Inference Results")

                    res_col1, res_col2 = st.columns([1, 1])

                    with res_col1:
                        # Gauge Chart Visualization
                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=probability * 100,
                                number={"suffix": "%", "font": {"size": 36}},
                                title={"text": "Churn Risk Probability"},
                                gauge={
                                    "axis": {"range": [0, 100]},
                                    "bar": {"color": "#333333"},
                                    "steps": [
                                        {"range": [0, 30], "color": "#2ecc71"},
                                        {"range": [30, 60], "color": "#f1c40f"},
                                        {"range": [60, 100], "color": "#e74c3c"},
                                    ],
                                    "threshold": {
                                        "line": {"color": "black", "width": 4},
                                        "thickness": 0.75,
                                        "value": probability * 100,
                                    },
                                },
                            )
                        )
                        fig.update_layout(
                            height=280, margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with res_col2:
                        st.markdown("### Risk Status & Strategy")
                        if prediction == 1 or probability >= 0.5:
                            st.error("⚠️ **HIGH RISK CUSTOMER**")
                            st.write(
                                "This customer profile shows strong indicators of potential cancellation."
                            )
                            st.markdown("""
                                **Recommended Actions:**
                                * Offer annual contract conversion discounts.
                                * Upgrade internet speed or add complimentary tech support.
                                """)
                        else:
                            st.success("✅ **LOW RISK CUSTOMER**")
                            st.write(
                                "Customer displays stable usage patterns and low risk profile."
                            )
                            st.markdown("""
                                **Recommended Actions:**
                                * Cross-sell higher tier value bundles.
                                * Invite to loyalty feedback program.
                                """)
                else:
                    st.error(
                        f"Inference Error ({response.status_code}): {response.text}"
                    )

            except Exception as err:
                st.error(f"Failed to communicate with prediction service: {err}")
