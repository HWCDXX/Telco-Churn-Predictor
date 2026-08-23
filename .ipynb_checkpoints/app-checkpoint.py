import os
import requests
import streamlit as st

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.set_page_config(page_title="Telco Churn Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Telco Customer Churn Predictor")

# Backend Connection Check
try:
    health_res = requests.get(f"{FASTAPI_URL}/health", timeout=3)
    if health_res.status_code == 200:
        st.success(f"Connected to FastAPI Backend ({FASTAPI_URL})")
    else:
        st.warning("Backend service degraded.")
except Exception as e:
    st.error(f"Cannot connect to FastAPI at {FASTAPI_URL}: {e}")

st.info("Interactive input forms and probability gauge charts will be wired here next.")
