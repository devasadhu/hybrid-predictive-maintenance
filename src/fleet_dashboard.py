# src/fleet_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

def show_fleet_dashboard():
    st.title("🏭 Fleet Health Overview")
    st.markdown("Multi-Turbine Monitoring Dashboard")

    # --- Load dataset ---
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_PATH = os.path.join(BASE_DIR, "data", "simulated_dataset_enhanced.csv")

    if os.path.exists(DATA_PATH):
        fleet_df = pd.read_csv(DATA_PATH)
    else:
        st.warning("⚠️ No dataset found. Please run `python src/train_model.py` first.")
        return

    # --- Fleet health distribution ---
    col1, col2 = st.columns(2)
    with col1:
        status_counts = fleet_df["Status"].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Fleet Health Distribution",
            color=status_counts.index,
            color_discrete_map={"CRITICAL": "#ff4444", "WARNING": "#ffbb33", "NORMAL": "#00C851"}
        )
        st.plotly_chart(fig_status, width='stretch')
    with col2:
        fig_rul = px.histogram(
            fleet_df, x="RUL (hours)", nbins=20,
            title="RUL Distribution Across Fleet",
            color_discrete_sequence=["#00843D"]
        )
        fig_rul.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        fig_rul.add_vline(x=20, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        st.plotly_chart(fig_rul, width='stretch')

    # --- Failure mode analysis ---
    st.subheader("🎯 Fleet-Wide Risk Assessment")
    risk_counts = fleet_df["Top Risk"].value_counts()
    fig_risks = px.bar(
        x=risk_counts.index, y=risk_counts.values,
        title="Common Failure Modes Across Fleet",
        labels={"x": "Failure Mode", "y": "Number of Turbines"},
        color=risk_counts.values, color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_risks, width='stretch')

    # --- Priority actions ---
    st.subheader("🚨 Priority Actions Required")
    critical_turbines = fleet_df[fleet_df["Status"] == "CRITICAL"].sort_values("RUL (hours)")
    warning_turbines = fleet_df[fleet_df["Status"] == "WARNING"].sort_values("RUL (hours)")

    if not critical_turbines.empty:
        st.error(f"**{len(critical_turbines)} turbine(s) require immediate attention:**")
        for _, row in critical_turbines.iterrows():
            st.markdown(
                f"<div class='turbine-card critical'><strong>{row['Turbine ID']}</strong> - "
                f"RUL: {row['RUL (hours)']} hours<br>Risk: {row['Top Risk']} "
                f"({row['Risk Probability (%)']:.1f}%)<br>Action: Schedule emergency inspection within 24-48 hours</div>",
                unsafe_allow_html=True
            )

    if not warning_turbines.empty:
        st.warning(f"**{len(warning_turbines)} turbine(s) need planned maintenance:**")
        for _, row in warning_turbines.iterrows():
            st.markdown(
                f"<div class='turbine-card warning'><strong>{row['Turbine ID']}</strong> - "
                f"RUL: {row['RUL (hours)']} hours<br>Risk: {row['Top Risk']} "
                f"({row['Risk Probability (%)']:.1f}%)<br>Action: Plan maintenance within 1-2 weeks</div>",
                unsafe_allow_html=True
            )

    normal_count = len(fleet_df[fleet_df["Status"] == "NORMAL"])
    if normal_count > 0:
        st.success(f"✅ **{normal_count} turbine(s) operating normally**")

    # --- Detailed fleet table ---
    st.subheader("📋 Detailed Fleet Status")
    def color_status(val):
        if val == "CRITICAL":
            return 'background-color: #ffcccc'
        elif val == "WARNING":
            return 'background-color: #fff4cc'
        else:
            return 'background-color: #ccffcc'
    
    # Fixed: Use .map() instead of .map()
    styled_df = fleet_df.style.map(color_status, subset=['Status'])
    st.dataframe(styled_df, width='stretch', height=400)

    # --- Export ---
    st.subheader("📥 Export Fleet Data")
    csv = fleet_df.to_csv(index=False)
    st.download_button(
        label="Download Fleet Report (CSV)",
        data=csv,
        file_name=f"fleet_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # --- Sensor correlations ---
    st.subheader("📊 Fleet-Wide Sensor Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig_temp = px.scatter(
            fleet_df, x="Operating Hours", y="Temperature",
            color="Status", size="RUL (hours)",
            title="Temperature vs Operating Hours",
            color_discrete_map={"CRITICAL": "#ff4444", "WARNING": "#ffbb33", "NORMAL": "#00C851"}
        )
        st.plotly_chart(fig_temp, width='stretch')
    with col2:
        fig_vib = px.scatter(
            fleet_df, x="Operating Hours", y="Vibration",
            color="Status", size="RUL (hours)",
            title="Vibration vs Operating Hours",
            color_discrete_map={"CRITICAL": "#ff4444", "WARNING": "#ffbb33", "NORMAL": "#00C851"}
        )
        st.plotly_chart(fig_vib, width='stretch')

# Allow standalone run
if __name__ == "__main__":
    show_fleet_dashboard()