import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Medical Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load model & scaler
# -----------------------------
model = load_model("best_nn_model.h5")
scaler = joblib.load("scaler.pkl")

numeric_features = [
    "age",
    "total_claims",
    "num_conditions",
    "num_meds",
    "num_encounters"
]

categorical_features = ["gender", "race"]

# -----------------------------
# Helper
# -----------------------------
def smooth_prob(p, factor=0.9):
    return factor * p + (1 - factor) * 0.5

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div style="background:linear-gradient(to right,#4B6EFF,#61FFDA);
                padding:15px;border-radius:8px;
                text-align:center;color:white;
                font-size:22px;font-weight:bold;">
        🩺 Medical Risk Prediction Dashboard
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "Predict **multi-morbidity risk** (≥2 conditions) using clinical and demographic features."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")
smooth_factor = st.sidebar.slider("Probability Smoothing", 0.5, 1.0, 0.9)

# -----------------------------
# Single patient prediction
# -----------------------------
st.subheader("📍 Single Patient Prediction")

with st.form("patient_form"):
    age = st.number_input("Age", 0, 120, 45)
    total_claims = st.number_input("Total Claims", 0, 100000, 1000)
    num_conditions = st.number_input("Number of Conditions", 0, 20, 2)
    num_meds = st.number_input("Number of Medications", 0, 20, 2)
    num_encounters = st.number_input("Number of Encounters", 0, 50, 5)

    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    df = pd.DataFrame(
        [[age, total_claims, num_conditions, num_meds, num_encounters, gender, race]],
        columns=numeric_features + categorical_features
    )

    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    df["race"] = df["race"].map({"White": 0, "Black": 1, "Asian": 2, "Other": 3})

    df[numeric_features] = scaler.transform(df[numeric_features])

    raw_prob = model.predict(df)[0][0]
    prob = smooth_prob(raw_prob, smooth_factor)
    pred = int(prob > 0.5)

    if pred == 1:
        st.error(f"⚠️ High Risk of Multimorbidity\n\nProbability: {prob*100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Multimorbidity\n\nProbability: {prob*100:.2f}%")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if pred else "green"}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Batch prediction
# -----------------------------
st.subheader("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    required_cols = numeric_features + categorical_features

    if not all(col in data.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        data["gender"] = data["gender"].map({"Male": 0, "Female": 1})
        data["race"] = data["race"].map({"White": 0, "Black": 1, "Asian": 2, "Other": 3})

        data[numeric_features] = scaler.transform(data[numeric_features])

        probs = model.predict(data).flatten()
        data["Risk_Probability"] = [smooth_prob(p, smooth_factor) for p in probs]
        data["Risk"] = np.where(data["Risk_Probability"] > 0.5, "High", "Low")

        st.success("Predictions completed")
        st.dataframe(data)

        st.download_button(
            "📥 Download Results",
            data.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )