# src/data_sim.py - Gas Turbine Edition
import numpy as np
import pandas as pd
import os
from tqdm import trange

OUT = "../data/simulated_dataset.csv"
os.makedirs("../data", exist_ok=True)

# Gas Turbine Parameters
TURBINE_MODELS = ["9HA.02", "7HA.03", "6B.03"]
PLANT_LOCATIONS = ["Texas_Plant_A", "Ohio_Plant_B", "Arizona_Plant_C"]

def generate_turbine_run(turbine_id, length=200, seed=None, turbine_model=None):
    """
    Simulate gas turbine sensors with realistic parameters:
    - Compressor discharge temperature (CDT)
    - Turbine exhaust temperature (TET) 
    - Bearing vibration (multiple zones)
    - Fuel pressure
    - Generator power output
    - NOx emissions
    """
    if seed is not None:
        np.random.seed(seed + int(turbine_id))
    
    t = np.arange(length)
    turbine_model = turbine_model or np.random.choice(TURBINE_MODELS)
    plant = np.random.choice(PLANT_LOCATIONS)
    
    # Base operating conditions (realistic for gas turbines)
    cdt = 400 + 0.1 * (t / length) * 50 + np.random.normal(0, 3, length)  # °C
    tet = 550 + 0.15 * (t / length) * 80 + np.random.normal(0, 5, length)  # °C
    
    # Multi-zone vibration (bearings at different locations)
    vib_bearing_1 = 2.5 + 0.02*np.sin(0.1*t) + np.random.normal(0, 0.15, length)  # mm/s
    vib_bearing_2 = 2.3 + 0.03*np.cos(0.08*t) + np.random.normal(0, 0.12, length)  # mm/s
    vib_turbine = 1.8 + 0.01*np.sin(0.15*t) + np.random.normal(0, 0.10, length)    # mm/s
    
    fuel_pressure = 280 + np.random.normal(0, 2, length) - 0.08 * t  # psi
    power_output = 300 + 0.2*t + np.random.normal(0, 5, length)  # MW
    nox_emissions = 15 + np.random.normal(0, 0.5, length)  # ppm
    
    # Cooling air flow and lube oil temperature
    cooling_flow = 850 + np.random.normal(0, 10, length) - 0.1 * t  # cfm
    lube_oil_temp = 60 + 0.05 * (t / length) * 20 + np.random.normal(0, 1, length)  # °C
    
    # Introduce degradation patterns (starts at 60% of lifecycle)
    degrade_start = int(length * 0.6)
    for i in range(degrade_start, length):
        degradation_factor = (i - degrade_start) / (length - degrade_start)
        cdt[i] += 30 * degradation_factor
        tet[i] += 40 * degradation_factor
        vib_bearing_1[i] += 1.5 * degradation_factor
        lube_oil_temp[i] += 15 * degradation_factor
    
    # Failure modes specific to gas turbines
    failure_modes = [
        "hot_gas_path_degradation",
        "bearing_degradation", 
        "combustor_damage",
        "compressor_fouling",
        "seal_wear"
    ]
    fm = np.random.choice(failure_modes)
    
    # Inject failure-specific signatures
    degradation_pattern = np.linspace(0, 1, length)
    
    if fm == "hot_gas_path_degradation":
        tet += 60 * degradation_pattern
        cdt += 35 * degradation_pattern
        power_output -= 20 * degradation_pattern
        nox_emissions += 5 * degradation_pattern
        
    elif fm == "bearing_degradation":
        vib_bearing_1 += 3.5 * degradation_pattern
        vib_bearing_2 += 2.8 * degradation_pattern
        lube_oil_temp += 20 * degradation_pattern
        
    elif fm == "combustor_damage":
        tet += 70 * degradation_pattern
        nox_emissions += 8 * degradation_pattern
        vib_turbine += 1.2 * degradation_pattern
        
    elif fm == "compressor_fouling":
        cdt += 45 * degradation_pattern
        power_output -= 25 * degradation_pattern
        fuel_pressure += 15 * degradation_pattern
        
    elif fm == "seal_wear":
        fuel_pressure -= 20 * degradation_pattern
        cooling_flow -= 100 * degradation_pattern
        power_output -= 15 * degradation_pattern
    
    # Calculate RUL (Remaining Useful Life in operating hours)
    RUL = np.arange(length - 1, -1, -1) * 24  # Convert cycles to hours (approx)
    
    # Calculate efficiency metric
    efficiency = 100 * (1 - 0.3 * degradation_pattern)
    
    df = pd.DataFrame({
        "turbine_id": turbine_id,
        "turbine_model": turbine_model,
        "plant_location": plant,
        "operating_hours": t * 24,  # Convert to hours
        "compressor_discharge_temp": cdt,
        "turbine_exhaust_temp": tet,
        "vibration_bearing_1": vib_bearing_1,
        "vibration_bearing_2": vib_bearing_2,
        "vibration_turbine": vib_turbine,
        "fuel_pressure": fuel_pressure,
        "power_output_mw": power_output,
        "nox_emissions": nox_emissions,
        "cooling_air_flow": cooling_flow,
        "lube_oil_temp": lube_oil_temp,
        "efficiency_percent": efficiency,
        "RUL_hours": RUL,
        "failure_mode": fm
    })
    
    return df

def make_fleet_dataset(n_turbines=120, run_length=150):
    """Generate complete fleet dataset with multiple turbines."""
    frames = []
    print(f"🔧 Generating Gas Turbine Fleet Data ({n_turbines} units)...")
    
    for i in trange(n_turbines, desc="Simulating turbines"):
        frames.append(generate_turbine_run(i, length=run_length, seed=42))
    
    full_df = pd.concat(frames, ignore_index=True)
    
    # Add anomaly events (5% of data points)
    n_anomalies = int(len(full_df) * 0.05)
    anomaly_indices = np.random.choice(full_df.index, n_anomalies, replace=False)
    full_df['is_anomaly'] = False
    full_df.loc[anomaly_indices, 'is_anomaly'] = True
    
    return full_df

def create_legacy_compatible_dataset(df):
    """Create backward-compatible version for existing training pipeline."""
    legacy_df = df.copy()
    
    # Map to old column names
    column_mapping = {
        "turbine_id": "run_id",
        "operating_hours": "t",
        "vibration_bearing_1": "vibration",
        "turbine_exhaust_temp": "temperature",
        "fuel_pressure": "pressure",
        "cooling_air_flow": "flow",
        "power_output_mw": "power",
        "RUL_hours": "RUL"
    }
    
    legacy_cols = []
    for old_col, new_col in column_mapping.items():
        if old_col in legacy_df.columns:
            legacy_df[new_col] = legacy_df[old_col]
            legacy_cols.append(new_col)
    
    legacy_cols.append("failure_mode")
    return legacy_df[legacy_cols]

if __name__ == "__main__":
    print("Gas Turbine Fleet Simulation")
    print("=" * 60)
    
    # Generate enhanced dataset
    df_full = make_fleet_dataset(n_turbines=120, run_length=150)
    
    # Save full enhanced dataset
    enhanced_path = OUT.replace(".csv", "_enhanced.csv")
    df_full.to_csv(enhanced_path, index=False)
    print(f"\n✅ Enhanced dataset saved to {enhanced_path}")
    print(f"   Shape: {df_full.shape}")
    print(f"   Turbine models: {df_full['turbine_model'].value_counts().to_dict()}")
    print(f"   Plant locations: {df_full['plant_location'].value_counts().to_dict()}")
    print(f"   Failure modes: {df_full['failure_mode'].value_counts().to_dict()}")
    print(f"   Anomalies detected: {df_full['is_anomaly'].sum()}")
    
    # Create legacy-compatible version for existing training pipeline
    df_legacy = create_legacy_compatible_dataset(df_full)
    df_legacy.to_csv(OUT, index=False)
    print(f"\n✅ Legacy-compatible dataset saved to {OUT}")
    print(f"   Shape: {df_legacy.shape}")
    
    print("\n🎯 Ready for training! Run: python src/train_model.py")