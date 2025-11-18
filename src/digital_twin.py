# src/digital_twin.py - Interactive Digital Twin Simulator (REFACTORED)
"""
Digital Twin Simulator for Gas Turbines - Clean version

Can be run standalone OR imported into master_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_PATH = os.path.join(BASE_DIR, "models")

COST_PER_MW_HOUR = 50
FUEL_COST_PER_UNIT = 3.5
MAINTENANCE_COST_PER_CYCLE = 1000

# Custom CSS
TWIN_CSS = """
<style>
    .twin-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00843D;
        margin-bottom: 1rem;
    }
    .optimal-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
"""

def simulate_turbine_operation(temp, pressure, vibration, power, cycles=100):
    """Simulate turbine operation over time with given parameters."""
    
    base_degradation = 0.5
    temp_factor = 1.0 + np.exp((temp - 550) / 50) * 0.1
    pressure_factor = 1.0 + abs(pressure - 280) / 280 * 0.2
    vibration_factor = 1.0 + (vibration / 2.5) ** 2 * 0.15
    
    degradation_rate = base_degradation * temp_factor * pressure_factor * vibration_factor
    
    rul_trajectory = []
    initial_rul = 150
    current_rul = initial_rul
    
    for cycle in range(cycles):
        current_rul -= degradation_rate
        rul_trajectory.append(max(0, current_rul))
        if current_rul <= 0:
            break
    
    fuel_cost = cycles * power * FUEL_COST_PER_UNIT
    revenue = cycles * power * COST_PER_MW_HOUR
    maintenance_cost = (cycles / max(current_rul, 1)) * MAINTENANCE_COST_PER_CYCLE * 10
    total_profit = revenue - fuel_cost - maintenance_cost
    
    return {
        'rul_trajectory': rul_trajectory,
        'final_rul': rul_trajectory[-1] if rul_trajectory else 0,
        'cycles_until_failure': len(rul_trajectory),
        'fuel_cost': fuel_cost,
        'revenue': revenue,
        'maintenance_cost': maintenance_cost,
        'total_profit': total_profit,
        'degradation_rate': degradation_rate
    }

def optimize_operating_point(target='max_profit'):
    """Use grid search to find optimal operating parameters."""
    
    temp_range = np.linspace(520, 620, 15)
    pressure_range = np.linspace(260, 300, 10)
    power_range = np.linspace(280, 340, 10)
    
    best_score = -np.inf
    best_params = None
    best_result = None
    
    for temp in temp_range:
        for pressure in pressure_range:
            for power in power_range:
                result = simulate_turbine_operation(temp, pressure, 2.0, power, cycles=100)
                
                if target == 'max_profit':
                    score = result['total_profit']
                elif target == 'max_lifetime':
                    score = result['final_rul']
                else:
                    score = result['total_profit'] * 0.6 + result['final_rul'] * 100
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'temperature': temp,
                        'pressure': pressure,
                        'power': power,
                        'vibration': 2.0
                    }
                    best_result = result
    
    return best_params, best_result, best_score

def plot_rul_comparison(baseline_traj, scenario_traj):
    """Plot RUL trajectory comparison."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(baseline_traj))),
        y=baseline_traj,
        mode='lines',
        name='Current Settings',
        line=dict(color='#FF6B6B', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(scenario_traj))),
        y=scenario_traj,
        mode='lines',
        name='Adjusted Settings',
        line=dict(color='#00C851', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title='RUL Degradation Over Time',
        xaxis_title='Operating Cycles',
        yaxis_title='Remaining Useful Life (hours)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_cost_breakdown(baseline_result, scenario_result):
    """Plot cost comparison between baseline and scenario."""
    categories = ['Revenue', 'Fuel Cost', 'Maintenance Cost', 'Net Profit']
    
    baseline_values = [
        baseline_result['revenue'],
        -baseline_result['fuel_cost'],
        -baseline_result['maintenance_cost'],
        baseline_result['total_profit']
    ]
    
    scenario_values = [
        scenario_result['revenue'],
        -scenario_result['fuel_cost'],
        -scenario_result['maintenance_cost'],
        scenario_result['total_profit']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current Settings',
        x=categories,
        y=baseline_values,
        marker_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Bar(
        name='Adjusted Settings',
        x=categories,
        y=scenario_values,
        marker_color='#00C851'
    ))
    
    fig.update_layout(
        title='Financial Impact Comparison (100 cycles)',
        yaxis_title='Amount ($)',
        barmode='group',
        height=400
    )
    
    return fig

def show_digital_twin(external_call=False):
    """
    Main Digital Twin function.
    
    Args:
        external_call: True if called from master_dashboard.py
    """
    
    if not external_call:
        st.set_page_config(
            page_title="Digital Twin Simulator",
            page_icon="🎮",
            layout="wide"
        )
        st.markdown(TWIN_CSS, unsafe_allow_html=True)
    
    st.markdown('<p class="twin-header">🎮 Digital Twin Simulator</p>', unsafe_allow_html=True)
    st.markdown("**Interactive What-If Analysis & Operating Point Optimization**")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs([
        "🎛️ Interactive Parameter Testing",
        "🤖 AI-Powered Optimization",
        "📊 Scenario Comparison"
    ])
    
    # TAB 1: Interactive Testing
    with tab1:
        st.subheader("🎛️ Adjust Operating Parameters")
        st.markdown("Use the sliders to see real-time impact on turbine health and economics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Baseline (Current Settings)**")
            baseline_temp = 560
            baseline_pressure = 280
            baseline_vibration = 2.2
            baseline_power = 310
            
            st.info(f"""
            **Current Operating Point:**
            - Temperature: {baseline_temp}°C
            - Fuel Pressure: {baseline_pressure} psi
            - Vibration: {baseline_vibration} mm/s
            - Power Output: {baseline_power} MW
            """)
            
            baseline_result = simulate_turbine_operation(
                baseline_temp, baseline_pressure, baseline_vibration, baseline_power
            )
            
            st.metric("Current RUL Prediction", f"{baseline_result['final_rul']:.1f} hours")
            st.metric("Degradation Rate", f"{baseline_result['degradation_rate']:.3f} hours/cycle")
            st.metric("Expected Profit (100 cycles)", f"${baseline_result['total_profit']:,.0f}")
        
        with col2:
            st.markdown("**Adjust Parameters (What-If Scenario)**")
            
            temp_adjust = st.slider(
                "🌡️ Turbine Exhaust Temperature (°C)",
                min_value=520, max_value=620, value=560, step=5,
                help="Higher temp = more power but faster degradation"
            )
            
            pressure_adjust = st.slider(
                "💨 Fuel Pressure (psi)",
                min_value=260, max_value=300, value=280, step=2,
                help="Optimal pressure balances efficiency and wear"
            )
            
            vibration_adjust = st.slider(
                "📳 Vibration Level (mm/s)",
                min_value=1.5, max_value=4.0, value=2.2, step=0.1,
                help="Lower vibration = longer life"
            )
            
            power_adjust = st.slider(
                "⚡ Power Output (MW)",
                min_value=280, max_value=340, value=310, step=5,
                help="Higher power = more revenue but increased stress"
            )
            
            scenario_result = simulate_turbine_operation(
                temp_adjust, pressure_adjust, vibration_adjust, power_adjust
            )
            
            rul_change = scenario_result['final_rul'] - baseline_result['final_rul']
            profit_change = scenario_result['total_profit'] - baseline_result['total_profit']
            
            st.metric(
                "Adjusted RUL Prediction",
                f"{scenario_result['final_rul']:.1f} hours",
                delta=f"{rul_change:+.1f} hours"
            )
            st.metric(
                "Degradation Rate",
                f"{scenario_result['degradation_rate']:.3f} hours/cycle",
                delta=f"{scenario_result['degradation_rate'] - baseline_result['degradation_rate']:+.3f}"
            )
            st.metric(
                "Expected Profit (100 cycles)",
                f"${scenario_result['total_profit']:,.0f}",
                delta=f"${profit_change:+,.0f}"
            )
        
        # Visualizations
        st.markdown("---")
        st.subheader("📈 Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rul = plot_rul_comparison(
                baseline_result['rul_trajectory'],
                scenario_result['rul_trajectory']
            )
            st.plotly_chart(fig_rul, width='stretch')
        
        with col2:
            fig_cost = plot_cost_breakdown(baseline_result, scenario_result)
            st.plotly_chart(fig_cost, width='stretch')
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 AI Recommendations")
        
        if rul_change > 0 and profit_change > 0:
            st.success(f"✅ **Excellent!** Improves both RUL (+{rul_change:.1f}h) and profit (+${profit_change:,.0f})")
        elif rul_change > 0:
            st.info(f"⚖️ **Trade-off:** Increased lifetime (+{rul_change:.1f}h) but reduced profit (${profit_change:,.0f})")
        elif profit_change > 0:
            st.warning(f"⚠️ **Short-term gain:** Increased profit (+${profit_change:,.0f}) but reduced lifetime ({rul_change:.1f}h)")
        else:
            st.error(f"❌ **Not recommended:** Both RUL ({rul_change:.1f}h) and profit (${profit_change:,.0f}) decreased")
    
    # TAB 2: AI Optimization
    with tab2:
        st.subheader("🤖 AI-Powered Operating Point Optimization")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Select Optimization Goal:**")
            
            opt_target = st.radio(
                "What should the AI optimize for?",
                options=['max_profit', 'max_lifetime', 'balanced'],
                format_func=lambda x: {
                    'max_profit': '💰 Maximum Profit',
                    'max_lifetime': '⏳ Maximum Lifetime',
                    'balanced': '⚖️ Balanced'
                }[x]
            )
            
            if st.button("🚀 Run Optimization", type="primary"):
                with st.spinner("🤖 AI searching 1,500+ parameter combinations..."):
                    optimal_params, optimal_result, best_score = optimize_operating_point(opt_target)
                    st.session_state['optimal_params'] = optimal_params
                    st.session_state['optimal_result'] = optimal_result
                    st.session_state['best_score'] = best_score
        
        with col2:
            if 'optimal_params' in st.session_state:
                params = st.session_state['optimal_params']
                result = st.session_state['optimal_result']
                
                st.markdown('<div class="optimal-banner">🎯 Optimal Operating Point Found!</div>', unsafe_allow_html=True)
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.markdown("**Optimal Settings:**")
                    st.write(f"🌡️ Temp: **{params['temperature']:.1f}°C**")
                    st.write(f"💨 Pressure: **{params['pressure']:.1f} psi**")
                    st.write(f"⚡ Power: **{params['power']:.1f} MW**")
                
                with col_b:
                    st.markdown("**Predicted Outcomes:**")
                    st.write(f"⏱️ RUL: **{result['final_rul']:.1f}h**")
                    st.write(f"💰 Profit: **${result['total_profit']:,.0f}**")
                    st.write(f"🔄 Cycles: **{result['cycles_until_failure']}**")
                
                with col_c:
                    baseline_result = simulate_turbine_operation(560, 280, 2.2, 310)
                    improvement = ((result['total_profit'] - baseline_result['total_profit']) / baseline_result['total_profit']) * 100
                    
                    st.markdown("**vs. Current:**")
                    st.metric("Profit Improvement", f"{improvement:+.1f}%")
                    st.metric("RUL Change", f"{result['final_rul'] - baseline_result['final_rul']:+.1f}h")
                
                st.markdown("---")
                fig_opt = plot_rul_comparison(baseline_result['rul_trajectory'], result['rul_trajectory'])
                st.plotly_chart(fig_opt, width='stretch')
                
                st.markdown("---")
                st.subheader("📋 Implementation Guide")
                st.markdown(f"""
                **Recommended Actions:**
                1. Gradual transition over 2-3 days
                2. Monitor exhaust temp, vibration, NOx emissions
                3. Validation period of 48 hours
                4. Expected annual ROI: **${improvement/100 * baseline_result['total_profit'] * 36:,.0f}**
                """)
            else:
                st.info("👈 Click 'Run Optimization' to find optimal parameters")
    
    # TAB 3: Scenario Comparison
    with tab3:
        st.subheader("📊 Multi-Scenario Comparison")
        
        scenarios = {
            'Current': {'temp': 560, 'pressure': 280, 'power': 310, 'vib': 2.2},
            'Conservative': {'temp': 540, 'pressure': 275, 'power': 295, 'vib': 2.0},
            'Aggressive': {'temp': 600, 'pressure': 290, 'power': 330, 'vib': 2.5},
            'Eco-Mode': {'temp': 530, 'pressure': 270, 'power': 285, 'vib': 1.8},
        }
        
        results = {}
        for name, params in scenarios.items():
            results[name] = simulate_turbine_operation(
                params['temp'], params['pressure'], params['vib'], params['power']
            )
        
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Strategy': name,
                'Final RUL (h)': f"{result['final_rul']:.1f}",
                'Cycles Until Failure': result['cycles_until_failure'],
                'Total Profit ($)': f"${result['total_profit']:,.0f}",
                'Revenue ($)': f"${result['revenue']:,.0f}",
                'Fuel Cost ($)': f"${result['fuel_cost']:,.0f}",
                'Maintenance ($)': f"${result['maintenance_cost']:,.0f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, width='stretch', hide_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rul_multi = go.Figure()
            for name, result in results.items():
                fig_rul_multi.add_trace(go.Scatter(
                    x=list(range(len(result['rul_trajectory']))),
                    y=result['rul_trajectory'],
                    mode='lines',
                    name=name,
                    line=dict(width=2)
                ))
            fig_rul_multi.update_layout(
                title='RUL Trajectories Comparison',
                xaxis_title='Cycles',
                yaxis_title='RUL (hours)',
                height=400
            )
            st.plotly_chart(fig_rul_multi, width='stretch')
        
        with col2:
            profit_values = [results[name]['total_profit'] for name in scenarios.keys()]
            fig_profit = go.Figure(data=[
                go.Bar(x=list(scenarios.keys()), y=profit_values,
                       marker_color=['#FF6B6B', '#4ECDC4', '#FF6B88', '#95E1D3'])
            ])
            fig_profit.update_layout(
                title='Profit Comparison (100 cycles)',
                yaxis_title='Total Profit ($)',
                height=400
            )
            st.plotly_chart(fig_profit, width='stretch')

if __name__ == "__main__":
    show_digital_twin(external_call=False)