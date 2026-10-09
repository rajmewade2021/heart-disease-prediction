"""
Heart Disease Prediction — Streamlit App
Loads a pre-trained SVC model (model.pkl + scaler.pkl + columns.pkl)
and lets users enter patient data to get a prediction.
"""
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
)

# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, scaler, columns

model, scaler, feature_columns = load_artifacts()

# ── Helper: build encoded feature vector ─────────────────────────────────────
def build_input(age, trestbps, chol, thalach, oldpeak,
                sex, cp, fbs, restecg, exang, slope, ca, thal):
    """Return a 1-row DataFrame aligned to the training columns."""
    row = {col: 0 for col in feature_columns}
    # Continuous
    row["age"]      = age
    row["trestbps"] = trestbps
    row["chol"]     = chol
    row["thalach"]  = thalach
    row["oldpeak"]  = oldpeak
    # One-hot categoricals
    for col_name, value in [
        ("sex",     sex),
        ("cp",      cp),
        ("fbs",     fbs),
        ("restecg", restecg),
        ("exang",   exang),
        ("slope",   slope),
        ("ca",      ca),
        ("thal",    thal),
    ]:
        key = f"{col_name}_{value}"
        if key in row:
            row[key] = 1
    return pd.DataFrame([row])[feature_columns]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Prediction")
st.markdown(
    "Enter the patient's clinical measurements below and click **Predict** "
    "to assess heart disease risk using an SVM classifier trained on the "
    "[UCI Heart Disease dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)."
)

st.divider()

# ── Section 1 — Personal Info ─────────────────────────────────────────────────
st.subheader("👤 Personal Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=50,
        help="Patient age in years."
    )

with col2:
    sex_label = st.radio(
        "Sex", options=["Male", "Female"], horizontal=True,
        help="Biological sex of the patient."
    )
    sex = 1 if sex_label == "Male" else 0

st.divider()

# ── Section 2 — Vital Signs ───────────────────────────────────────────────────
st.subheader("🩺 Vital Signs")
col3, col4, col5 = st.columns(3)

with col3:
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80, max_value=220, value=120,
        help="Resting blood pressure on admission to hospital."
    )

with col4:
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=100, max_value=600, value=200,
        help="Serum cholesterol in mg/dl."
    )

with col5:
    thalach = st.number_input(
        "Max Heart Rate Achieved (bpm)",
        min_value=60, max_value=220, value=150,
        help="Maximum heart rate achieved during exercise test."
    )

st.divider()

# ── Section 3 — ECG & Stress Test ─────────────────────────────────────────────
st.subheader("📈 ECG & Stress Test Results")
col6, col7 = st.columns(2)

with col6:
    cp_options = {
        "Typical Angina (0)": 0,
        "Atypical Angina (1)": 1,
        "Non-anginal Pain (2)": 2,
        "Asymptomatic (3)": 3,
    }
    cp_label = st.selectbox(
        "Chest Pain Type",
        options=list(cp_options.keys()),
        help="Type of chest pain experienced by the patient."
    )
    cp = cp_options[cp_label]

    restecg_options = {
        "Normal (0)": 0,
        "ST-T Wave Abnormality (1)": 1,
        "Left Ventricular Hypertrophy (2)": 2,
    }
    restecg_label = st.selectbox(
        "Resting ECG Result",
        options=list(restecg_options.keys()),
        help="Resting electrocardiographic results."
    )
    restecg = restecg_options[restecg_label]

    slope_options = {
        "Downsloping (0)": 0,
        "Flat (1)": 1,
        "Upsloping (2)": 2,
    }
    slope_label = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=list(slope_options.keys()),
        help="Slope of the peak exercise ST segment."
    )
    slope = slope_options[slope_label]

with col7:
    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest."
    )

    exang_label = st.radio(
        "Exercise Induced Angina",
        options=["No", "Yes"], horizontal=True,
        help="Whether exercise induced angina."
    )
    exang = 1 if exang_label == "Yes" else 0

    fbs_label = st.radio(
        "Fasting Blood Sugar > 120 mg/dl",
        options=["No", "Yes"], horizontal=True,
        help="Whether fasting blood sugar is greater than 120 mg/dl."
    )
    fbs = 1 if fbs_label == "Yes" else 0

st.divider()

# ── Section 4 — Cardiac Details ───────────────────────────────────────────────
st.subheader("🔬 Cardiac Details")
col8, col9 = st.columns(2)

with col8:
    ca = st.selectbox(
        "Number of Major Vessels (0–4)",
        options=[0, 1, 2, 3, 4],
        help="Number of major vessels coloured by fluoroscopy."
    )

with col9:
    thal_options = {
        "Normal (1)": 1,
        "Fixed Defect (2)": 2,
        "Reversible Defect (3)": 3,
    }
    thal_label = st.selectbox(
        "Thalassemia",
        options=list(thal_options.keys()),
        help="Thalassemia type from nuclear stress test."
    )
    thal = thal_options[thal_label]

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

if predict_btn:
    input_df = build_input(
        age, trestbps, chol, thalach, oldpeak,
        sex, cp, fbs, restecg, exang, slope, ca, thal
    )
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader("🩻 Prediction Result")

    risk_prob = probability[1] * 100  # probability of heart disease

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction == 1:
            st.error("### ❤️‍🩹 Heart Disease Detected", icon="⚠️")
        else:
            st.success("### ✅ No Heart Disease Detected", icon="✅")

    with res_col2:
        st.metric(
            label="Risk Probability",
            value=f"{risk_prob:.1f}%",
            delta=f"{'High' if risk_prob >= 50 else 'Low'} Risk",
            delta_color="inverse",
        )
        st.progress(int(risk_prob))

    st.caption(
        "⚠️ This prediction is generated by a machine learning model and is "
        "**not a substitute for professional medical advice**. "
        "Please consult a qualified healthcare provider for diagnosis."
    )

    # ── Expandable input summary ───────────────────────────────────────────────
    with st.expander("📋 View submitted input values"):
        summary = {
            "Age": age, "Sex": sex_label,
            "Chest Pain Type": cp_label, "Resting BP (mm Hg)": trestbps,
            "Cholesterol (mg/dl)": chol, "Fasting Blood Sugar >120": fbs_label,
            "Resting ECG": restecg_label, "Max Heart Rate": thalach,
            "Exercise Angina": exang_label, "ST Depression": oldpeak,
            "ST Slope": slope_label, "Major Vessels": ca,
            "Thalassemia": thal_label,
        }
        st.table(pd.DataFrame(list(summary.items()), columns=["Feature", "Value"]))
