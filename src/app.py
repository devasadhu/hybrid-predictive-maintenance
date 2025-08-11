import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# -----------------------
# BASE PATHS
# -----------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data")
MODELS_PATH = os.path.join(BASE_DIR, "models")

# -----------------------
# LOAD MODELS
# -----------------------
@st.cache_resource
def load_models():
    try:
        rul_model_path = os.path.join(MODELS_PATH, "rul_model.pkl")
        fm_model_path = os.path.join(MODELS_PATH, "fm_model.pkl")

        if not os.path.exists(rul_model_path) or os.path.getsize(rul_model_path) == 0:
            raise FileNotFoundError(f"Missing or empty file: {rul_model_path}")
        if not os.path.exists(fm_model_path) or os.path.getsize(fm_model_path) == 0:
            raise FileNotFoundError(f"Missing or empty file: {fm_model_path}")

        rul_model = joblib.load(rul_model_path)
        failure_mode_model = joblib.load(fm_model_path)

        # Optional — only if implemented
        hse_model = None
        return rul_model, failure_mode_model, hse_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

# -----------------------
# FEATURE ENGINEERING
# -----------------------
def calculate_features(df):
    feature_df = df.copy()
    for col in df.columns:
        if col not in ["unit_nr", "time_in_cycles", "failure_mode"]:
            feature_df[f"{col}_rolling_mean"] = df[col].rolling(window=5, min_periods=1).mean()
            feature_df[f"{col}_rolling_std"] = df[col].rolling(window=5, min_periods=1).std()
    feature_df.fillna(0, inplace=True)
    return feature_df

# -----------------------
# PREDICTION FUNCTION
# -----------------------

def predict(df, rul_model, failure_mode_model, hse_model):
    # Calculate features (use the same logic as training)
    features = calculate_features(df)

    # Load feature columns used for training
    import os
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODELS_PATH = os.path.join(BASE_DIR, "models")
    feature_cols = joblib.load(os.path.join(MODELS_PATH, "feature_cols.pkl"))

    # Select and reorder features exactly as during training
    features = features.reindex(columns=feature_cols)

    # Fill any missing values with 0
    features = features.fillna(0)

    rul_pred = rul_model.predict(features)
    fm_pred = failure_mode_model.predict_proba(features)

    latest_rul = rul_pred[-1]
    latest_fm_probs = fm_pred[-1]

    if hse_model:
        hse_pred = hse_model.predict_proba(features)
        latest_hse_probs = hse_pred[-1]
    else:
        latest_hse_probs = [0, 0]  # placeholder

    return latest_rul, latest_fm_probs, latest_hse_probs


# -----------------------
# MAIN STREAMLIT APP
# -----------------------
def main():
    st.set_page_config(page_title="Baker Hughes - Predictive Maintenance Dashboard", layout="wide")
    st.title("🔧 Predictive Maintenance Dashboard")
    st.markdown("Real-time RUL, failure mode prediction, and HS&E risk assessment.")

    rul_model, failure_mode_model, hse_model = load_models()
    if not rul_model:
        st.stop()

    st.sidebar.header("📂 Data Source")
    option = st.sidebar.radio("Choose input method:", ["Use Sample Run", "Upload CSV"])

    df = None
    if option == "Use Sample Run":
        sample_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
        if sample_files:
            selected_file = st.sidebar.selectbox("Select a run:", sample_files)
            df = pd.read_csv(os.path.join(DATA_PATH, selected_file))
        else:
            st.warning(f"No sample files found in {DATA_PATH}")
            st.stop()
    else:
        uploaded_file = st.sidebar.file_uploader("Upload sensor data CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Upload a file to proceed.")
            st.stop()

    st.subheader("📊 Latest Data Snapshot")
    st.dataframe(df.tail())

    latest_rul, latest_fm_probs, latest_hse_probs = predict(df, rul_model, failure_mode_model, hse_model)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted RUL (cycles)", f"{latest_rul:.1f}")
    col2.metric("HS&E Risk - Low", f"{latest_hse_probs[0]*100:.1f}%")
    col3.metric("HS&E Risk - High", f"{latest_hse_probs[1]*100:.1f}%")

    st.subheader("⚠ Failure Mode Probabilities")
    fm_labels = failure_mode_model.classes_
    fm_df = pd.DataFrame({
        "Failure Mode": fm_labels,
        "Probability": latest_fm_probs
    })
    st.bar_chart(fm_df.set_index("Failure Mode"))

    st.subheader("📈 Sensor Trends")
    sensor_cols = [col for col in df.columns if col not in ["unit_nr", "time_in_cycles", "failure_mode"]]
    if sensor_cols:
        selected_sensor = st.selectbox("Select sensor to view trend:", sensor_cols)
        fig = px.line(df, x="time_in_cycles", y=selected_sensor, title=f"{selected_sensor} Trend")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🛠 Maintenance Recommendation")
    if latest_rul < 20:
        st.error("Immediate inspection and maintenance required!")
    elif latest_rul < 50:
        st.warning("Schedule maintenance soon.")
    else:
        st.success("No immediate maintenance required.")

if __name__ == "__main__":
    main()
