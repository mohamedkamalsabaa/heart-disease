import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Models" / "final_heart_disease_model.pkl"
DATA_PATH = BASE_DIR / "Data" / "heart_disease_dataset.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("Heart Disease Prediction App")
st.markdown("Enter patient data for prediction and explore heart disease trends.")

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    df = None

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model = None

# Sidebar inputs
st.sidebar.header("Patient Inputs")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=55)
sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0,1,2,3], index=0)
trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=130)
chol = st.sidebar.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=246)

fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0,1])
restecg = st.sidebar.selectbox("Resting ECG Results (restecg)", [0,1,2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", [0,1])
oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (slope)", [0,1,2])
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", [0,1,2,3,4])
thal = st.sidebar.selectbox("Thalassemia (thal)", [1,2,3])

sex_v = 1 if sex == "Male" else 0

input_df = pd.DataFrame([{
    'age': age,
    'sex': sex_v,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}])

st.subheader("Current Inputs")
st.table(input_df.T)

if st.button("Predict Now"):
    if model is None:
        st.error("Model not available.")
    else:
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0,1]
            else:
                proba = float(model.predict(input_df))
            st.metric("Heart Disease Probability (%)", f"{proba*100:.2f}%")

            threshold = 0.5
            if proba >= threshold:
                st.error("High risk — Consult a doctor.")
            else:
                st.success("Low risk of heart disease.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.header("Explore Heart Disease Data")
if df is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        st.bar_chart(df['age'].value_counts().sort_index())
    with col2:
        st.subheader("Cases by Sex")
        sex_counts = df['sex'].map({1:'Male',0:'Female'}).value_counts()
        st.bar_chart(sex_counts)
    st.subheader("Average Target by Age")
    if 'target' in df.columns:
        pivot = df.groupby('age')['target'].mean()
        st.line_chart(pivot)
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
else:
    st.info("No data available — Place 'heart_disease_dataset.csv' in the Data/ folder.")