# src/utils.py
import numpy as np
import pandas as pd
import json
import os

def rolling_features(df, sensor_cols, window=10):
    """
    Given a dataframe with time-ordered rows for one unit run,
    compute rolling window statistical features for each sensor.
    Returns flattened features for the last timestamp of that run.
    """
    feats = {}
    for c in sensor_cols:
        if c in df.columns:
            series = df[c].rolling(window, min_periods=1)
            feats[f"{c}_mean"] = series.mean().iloc[-1]
            feats[f"{c}_std"] = series.std().iloc[-1] if window > 1 else 0.0
            feats[f"{c}_min"] = series.min().iloc[-1]
            feats[f"{c}_max"] = series.max().iloc[-1]
            
            # slope (trend) - improved calculation
            y = df[c].values
            if len(y) >= 2:
                x = np.arange(len(y))
                # Use simple linear regression for slope
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                else:
                    slope = 0.0
                feats[f"{c}_slope"] = slope
            else:
                feats[f"{c}_slope"] = 0.0
    return pd.Series(feats)

def calculate_features(df, window=12):
    """
    Calculate features for prediction - matches training format
    """
    sensor_cols = ["vibration", "temperature", "pressure", "flow", "power"]
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Sort by time
    if "time_in_cycles" in df.columns:
        df = df.sort_values("time_in_cycles").reset_index(drop=True)
    elif "t" in df.columns:
        df = df.sort_values("t").reset_index(drop=True)
    
    # Use last window rows
    window_data = df.tail(window) if len(df) >= window else df
    
    # Calculate rolling features
    rolling_feats = rolling_features(window_data, sensor_cols, window)
    
    # Add current values
    current_features = {}
    for col in sensor_cols:
        if col in df.columns:
            current_features[f"{col}_latest"] = df[col].iloc[-1]
    
    # Combine features
    all_features = {**rolling_feats.to_dict(), **current_features}
    
    return pd.DataFrame([all_features])

def make_hse_risk_score(pred_rul, fm_probs, sensor_latest, thresholds=None):
    """
    Heuristic HS&E risk:
     - high if predicted RUL is small
     - high if failure-mode probability is high
     - high if sensor thresholds are exceeded
    Returns: (risk_score_0_to_100, category_str)
    """
    if thresholds is None:
        thresholds = {
            'temperature': (35, 50),
            'vibration': (0.3, 0.8),
            'pressure': (4.5, 6.0),
            'flow': (90, 110),
            'power': (180, 220)
        }
    
    score = 0.0
    
    # RUL contribution: smaller RUL -> higher risk (max +50)
    if pred_rul < 10:
        score += 50
    elif pred_rul < 30:
        score += 30
    elif pred_rul < 50:
        score += 15
    
    # Failure mode contribution: highest predicted prob (max +30)
    fm_max = float(max(fm_probs)) if len(fm_probs) > 0 else 0.0
    score += fm_max * 30
    
    # Sensor threshold breaches (+5 each, max +20)
    sensor_violations = 0
    for s, val in sensor_latest.items():
        if s in thresholds:
            lo, hi = thresholds[s]
            if val < lo or val > hi:
                sensor_violations += 1
    
    score += min(sensor_violations * 5, 20)
    
    # Cap at 100
    score = min(100, score)
    
    # Risk category
    if score >= 70:
        cat = "HIGH"
    elif score >= 40:
        cat = "MEDIUM"
    else:
        cat = "LOW"
    
    return round(score, 1), cat

def calculate_hse_risk_simple(temp, vibration, pressure):
    """
    Simpler threshold-based HS&E risk function.
    Useful for quick testing or non-ML baseline.
    """
    risk = 0
    if temp > 50: risk += 2
    elif temp > 45: risk += 1

    if vibration > 0.8: risk += 2
    elif vibration > 0.6: risk += 1

    if pressure < 4.5: risk += 1

    if risk >= 4:
        return "HIGH"
    elif risk >= 2:
        return "MEDIUM"
    else:
        return "LOW"

def save_stream_json(out_path, record):
    """Save latest record as JSON for real-time streaming"""
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

def read_stream_json(path):
    """Read streaming JSON data"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None

def format_sensor_data(df):
    """Format dataframe for consistent column naming"""
    # Map common column variations
    column_mapping = {
        't': 'time_in_cycles',
        'time': 'time_in_cycles', 
        'run_id': 'unit_nr',
        'unit_id': 'unit_nr'
    }
    
    df_formatted = df.copy()
    for old_col, new_col in column_mapping.items():
        if old_col in df_formatted.columns and new_col not in df_formatted.columns:
            df_formatted = df_formatted.rename(columns={old_col: new_col})
    
    return df_formatted

def get_maintenance_recommendations(rul_pred, failure_mode, fm_prob, sensor_latest):
    """Generate maintenance recommendations based on predictions"""
    recommendations = []
    
    # RUL-based recommendations
    if rul_pred < 10:
        recommendations.append("🚨 **CRITICAL**: Stop equipment immediately for inspection")
    elif rul_pred < 30:
        recommendations.append("⚠️ **HIGH PRIORITY**: Schedule maintenance within 5-10 cycles")
    elif rul_pred < 50:
        recommendations.append("⏰ **MEDIUM PRIORITY**: Plan maintenance within 20-30 cycles")
    else:
        recommendations.append("✅ **LOW PRIORITY**: Continue monitoring")
    
    # Failure mode specific recommendations
    if fm_prob > 0.5:
        if failure_mode == "bearing_wear":
            recommendations.append("🔩 Inspect bearings and lubrication system")
        elif failure_mode == "seal_leak":
            recommendations.append("🔧 Check seals and gaskets for leakage")
        elif failure_mode == "misalignment":
            recommendations.append("⚖️ Verify equipment alignment and mounting")
        elif failure_mode == "overheat":
            recommendations.append("🌡️ Inspect cooling system and ventilation")
    
    # Sensor-based recommendations
    if 'temperature' in sensor_latest and sensor_latest['temperature'] > 50:
        recommendations.append("🌡️ High temperature - check cooling system")
    if 'vibration' in sensor_latest and sensor_latest['vibration'] > 0.8:
        recommendations.append("📳 High vibration - check for loose components")
    if 'pressure' in sensor_latest and sensor_latest['pressure'] < 4.5:
        recommendations.append("📉 Low pressure - inspect for leaks")
    
    return recommendations