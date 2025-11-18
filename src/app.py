# src/app.py - Gas Turbine Predictive Maintenance Dashboard (REFACTORED)
"""
Standalone turbine monitoring dashboard.
Can be run independently OR imported into master_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from utils import calculate_features

# -----------------------
# CONFIGURATION
# -----------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data")
MODELS_PATH = os.path.join(BASE_DIR, "models")

COST_PER_MW_HOUR = 50
DOWNTIME_COST_PER_HOUR = 75000
MAINTENANCE_COST_PLANNED = 25000
MAINTENANCE_COST_UNPLANNED = 150000

# -----------------------
# CUSTOM CSS
# -----------------------
DASHBOARD_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00843D;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #ffbb33;
        color: #333;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .alert-normal {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .cost-impact {
        font-size: 2rem;
        font-weight: bold;
        color: #00843D;
    }
</style>
"""

# -----------------------
# CACHED LOADERS
# -----------------------
@st.cache_resource
def load_models():
    """Load trained ML models."""
    try:
        rul_model = joblib.load(os.path.join(MODELS_PATH, "rul_model.pkl"))
        failure_mode_model = joblib.load(os.path.join(MODELS_PATH, "fm_model.pkl"))
        feature_cols = joblib.load(os.path.join(MODELS_PATH, "feature_cols.pkl"))
        metadata = joblib.load(os.path.join(MODELS_PATH, "model_metadata.pkl"))
        return rul_model, failure_mode_model, feature_cols, metadata
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None

# -----------------------
# UTILITY FUNCTIONS
# -----------------------
def calculate_cost_impact(rul_hours, power_output_mw):
    """Calculate financial impact of predicted failure."""
    if rul_hours < 100:
        est_downtime_hours = 72
        repair_cost = MAINTENANCE_COST_UNPLANNED
    elif rul_hours < 500:
        est_downtime_hours = 24
        repair_cost = MAINTENANCE_COST_PLANNED * 1.5
    else:
        est_downtime_hours = 8
        repair_cost = MAINTENANCE_COST_PLANNED
    
    lost_revenue = est_downtime_hours * power_output_mw * COST_PER_MW_HOUR
    total_cost = lost_revenue + repair_cost + (est_downtime_hours * DOWNTIME_COST_PER_HOUR / 24)
    reactive_cost = MAINTENANCE_COST_UNPLANNED + (72 * DOWNTIME_COST_PER_HOUR)
    savings = reactive_cost - total_cost if total_cost < reactive_cost else 0
    
    return {
        "estimated_downtime_hours": est_downtime_hours,
        "repair_cost": repair_cost,
        "lost_revenue": lost_revenue,
        "total_cost": total_cost,
        "potential_savings": savings
    }

def detect_anomalies(df, sensor_col, window=20, threshold=3):
    """Simple statistical anomaly detection using rolling z-score."""
    if len(df) < window:
        return [False] * len(df)
    
    rolling_mean = df[sensor_col].rolling(window=window, center=True).mean()
    rolling_std = df[sensor_col].rolling(window=window, center=True).std()
    z_scores = np.abs((df[sensor_col] - rolling_mean) / (rolling_std + 1e-6))
    anomalies = z_scores > threshold
    return anomalies.fillna(False).tolist()

def predict_rul_and_failure(df, rul_model, failure_mode_model, feature_cols):
    """Make RUL and failure mode predictions."""
    features = calculate_features(df)
    features = features.reindex(columns=feature_cols, fill_value=0)
    
    rul_pred = rul_model.predict(features)[0]
    fm_probs = failure_mode_model.predict_proba(features)[0]
    
    return rul_pred, fm_probs

# -----------------------
# MAIN DASHBOARD FUNCTION
# -----------------------
def show_turbine_dashboard(df=None, external_call=False):
    """
    Main turbine dashboard function.
    
    Args:
        df: Optional pre-loaded dataframe (for integration with master dashboard)
        external_call: True if called from master_dashboard.py
    """
    
    if not external_call:
        # Standalone mode: apply page config and CSS
        st.set_page_config(
            page_title="Gas Turbine Fleet Health",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">⚡ Gas Turbine Fleet Health Monitor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Predictive Maintenance & Asset Performance Management</p>', unsafe_allow_html=True)
    
    # Load models
    rul_model, fm_model, feature_cols, metadata = load_models()
    if not rul_model:
        st.stop()
    
    # Load data (either provided or select from files)
    if df is None:
        if not external_call:
            st.sidebar.image("https://via.placeholder.com/200x80/00843D/FFFFFF?text=Turbine+AI", width='stretch')
            st.sidebar.markdown("---")
            st.sidebar.header("🎛️ Dashboard Controls")
        
        # Data source selection
        option = st.sidebar.radio("Select Data Source:", ["📂 Historical Run Data", "📤 Upload Custom Data"])
        
        if option == "📂 Historical Run Data":
            sample_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv") and "sample_run" in f]
            if sample_files:
                selected_file = st.sidebar.selectbox("Select Turbine Run:", sorted(sample_files))
                df = pd.read_csv(os.path.join(DATA_PATH, selected_file))
            else:
                st.warning(f"No sample files found in {DATA_PATH}")
                st.stop()
        else:
            uploaded_file = st.sidebar.file_uploader("Upload CSV (turbine sensor data)", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                st.info("📁 Upload a CSV file to begin analysis")
                st.stop()
    
    # Ensure column compatibility
    if "t" in df.columns:
        df = df.rename(columns={"t": "time_in_cycles", "run_id": "unit_nr"})
    
    # Extract turbine info
    turbine_id = df["unit_nr"].iloc[0] if "unit_nr" in df.columns else "Unknown"
    current_hours = df["time_in_cycles"].iloc[-1] if "time_in_cycles" in df.columns else 0
    
    # Make predictions
    latest_rul, latest_fm_probs = predict_rul_and_failure(df, rul_model, fm_model, feature_cols)
    
    # Determine risk level
    if latest_rul < 20:
        risk_level, risk_color = "CRITICAL", "alert-critical"
    elif latest_rul < 50:
        risk_level, risk_color = "WARNING", "alert-warning"
    else:
        risk_level, risk_color = "NORMAL", "alert-normal"
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔢 Turbine ID", f"GT-{turbine_id}")
        st.metric("⏱️ Operating Hours", f"{current_hours:,.0f}")
    
    with col2:
        st.metric("🔮 Predicted RUL", f"{latest_rul:.0f} hours")
        st.metric("📊 Model Confidence", f"{metadata.get('rul_mae', 0):.1f}h MAE")
    
    with col3:
        fm_labels = fm_model.classes_
        top_failure_idx = np.argmax(latest_fm_probs)
        top_failure = fm_labels[top_failure_idx]
        top_failure_prob = latest_fm_probs[top_failure_idx]
        
        st.metric("⚠️ Primary Risk", top_failure.replace("_", " ").title())
        st.metric("📈 Probability", f"{top_failure_prob*100:.1f}%")
    
    with col4:
        st.markdown(f'<div class="{risk_color}">Health Status: {risk_level}</div>', unsafe_allow_html=True)
        
        avg_power = df["power"].iloc[-10:].mean() if "power" in df.columns else 300
        cost_info = calculate_cost_impact(latest_rul, avg_power)
        st.metric("💰 Potential Savings", f"${cost_info['potential_savings']:,.0f}")
    
    st.markdown("---")
    
    # Detailed cost breakdown
    with st.expander("💵 Cost Impact Analysis", expanded=(risk_level in ["CRITICAL", "WARNING"])):
        cost_col1, cost_col2, cost_col3 = st.columns(3)
        
        with cost_col1:
            st.markdown("### Estimated Costs")
            st.write(f"**Downtime:** {cost_info['estimated_downtime_hours']} hours")
            st.write(f"**Repair Cost:** ${cost_info['repair_cost']:,.0f}")
            st.write(f"**Lost Revenue:** ${cost_info['lost_revenue']:,.0f}")
        
        with cost_col2:
            st.markdown("### Total Impact")
            st.markdown(f'<p class="cost-impact">${cost_info["total_cost"]:,.0f}</p>', unsafe_allow_html=True)
        
        with cost_col3:
            st.markdown("### Predictive Maintenance Benefit")
            if cost_info['potential_savings'] > 0:
                st.success(f"✅ Save ${cost_info['potential_savings']:,.0f} vs reactive maintenance")
            else:
                st.info("Continue monitoring for optimal maintenance timing")
    
    # Failure mode probabilities
    st.subheader("🎯 Failure Mode Risk Assessment")
    fm_df = pd.DataFrame({
        "Failure Mode": [fm.replace("_", " ").title() for fm in fm_labels],
        "Probability (%)": latest_fm_probs * 100
    }).sort_values("Probability (%)", ascending=False)
    
    fig_fm = px.bar(
        fm_df, x="Probability (%)", y="Failure Mode",
        orientation='h', color="Probability (%)",
        color_continuous_scale="Reds",
        title="Predicted Failure Modes"
    )
    fig_fm.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_fm, width='stretch')
    
    # Sensor trends with anomaly detection
    st.subheader("📈 Real-Time Sensor Monitoring")
    
    sensor_cols = [col for col in df.columns if col not in ["unit_nr", "time_in_cycles", "failure_mode"]]
    if sensor_cols:
        tab_sensors = st.tabs(["Temperature", "Vibration", "Pressure & Flow", "Power"])
        
        with tab_sensors[0]:
            temp_cols = [c for c in sensor_cols if "temp" in c.lower()]
            if temp_cols:
                for temp_col in temp_cols:
                    anomalies = detect_anomalies(df, temp_col)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["time_in_cycles"], y=df[temp_col],
                        mode='lines', name=temp_col,
                        line=dict(color='#FF6B6B', width=2)
                    ))
                    
                    anomaly_df = df[anomalies]
                    if not anomaly_df.empty:
                        fig.add_trace(go.Scatter(
                            x=anomaly_df["time_in_cycles"], y=anomaly_df[temp_col],
                            mode='markers', name='Anomalies',
                            marker=dict(color='red', size=10, symbol='x')
                        ))
                    
                    fig.update_layout(title=f"{temp_col} Trend", height=300)
                    st.plotly_chart(fig, width='stretch')
        
        with tab_sensors[1]:
            vib_cols = [c for c in sensor_cols if "vib" in c.lower()]
            if vib_cols:
                for vib_col in vib_cols:
                    anomalies = detect_anomalies(df, vib_col)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["time_in_cycles"], y=df[vib_col],
                        mode='lines', name=vib_col,
                        line=dict(color='#4ECDC4', width=2)
                    ))
                    
                    anomaly_df = df[anomalies]
                    if not anomaly_df.empty:
                        fig.add_trace(go.Scatter(
                            x=anomaly_df["time_in_cycles"], y=anomaly_df[vib_col],
                            mode='markers', name='Anomalies',
                            marker=dict(color='red', size=10, symbol='x')
                        ))
                    
                    fig.update_layout(title=f"{vib_col} Trend", height=300)
                    st.plotly_chart(fig, width='stretch')
        
        with tab_sensors[2]:
            pressure_cols = [c for c in sensor_cols if "pressure" in c.lower() or "flow" in c.lower()]
            if pressure_cols:
                fig = make_subplots(rows=len(pressure_cols), cols=1, subplot_titles=pressure_cols)
                for idx, col in enumerate(pressure_cols, 1):
                    fig.add_trace(go.Scatter(x=df["time_in_cycles"], y=df[col], name=col), row=idx, col=1)
                fig.update_layout(height=300*len(pressure_cols), showlegend=False)
                st.plotly_chart(fig, width='stretch')
        
        with tab_sensors[3]:
            power_cols = [c for c in sensor_cols if "power" in c.lower()]
            if power_cols:
                fig = px.line(df, x="time_in_cycles", y=power_cols[0], title="Power Output Trend")
                fig.update_traces(line=dict(color='#95E1D3', width=3))
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')
    
    # Maintenance recommendations
    st.subheader("🔧 Actionable Maintenance Recommendations")
    
    recommendations = []
    
    if latest_rul < 50:
        recommendations.append({
            "Priority": "🔴 URGENT" if latest_rul < 20 else "🟡 HIGH",
            "Action": "Schedule immediate inspection" if latest_rul < 20 else "Plan maintenance within 2 weeks",
            "Estimated Cost": f"${cost_info['repair_cost']:,.0f}",
            "Timeline": "24-48 hours" if latest_rul < 20 else "1-2 weeks"
        })
    
    if top_failure_prob > 0.5:
        action_map = {
            "bearing_wear": "Inspect bearings, check lubrication system",
            "seal_leak": "Inspect seals and gaskets, check for pressure drops",
            "misalignment": "Verify rotor alignment, check foundation",
            "overheat": "Inspect cooling system, check combustor condition",
            "hot_gas_path_degradation": "Schedule hot gas path inspection"
        }
        recommendations.append({
            "Priority": "🟡 MEDIUM",
            "Action": action_map.get(top_failure, "Perform detailed diagnostic"),
            "Estimated Cost": "$15,000 - $40,000",
            "Timeline": "Next maintenance window"
        })
    
    if not recommendations:
        st.success("✅ No immediate actions required. Continue normal monitoring schedule.")
    else:
        rec_df = pd.DataFrame(recommendations)
        st.table(rec_df)
    
    # Footer
    st.markdown("---")
    st.caption(f"🤖 Model: {metadata.get('rul_model_type', 'Unknown')} | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# -----------------------
# STANDALONE EXECUTION
# -----------------------
if __name__ == "__main__":
    show_turbine_dashboard(external_call=False)