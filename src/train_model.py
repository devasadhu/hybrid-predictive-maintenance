# src/train_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from utils import rolling_features

# --- Set BASE_DIR relative to this script ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "simulated_dataset.csv")
ENHANCED_DATA_PATH = os.path.join(BASE_DIR, "data", "simulated_dataset_enhanced.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

SENSOR_COLS = ["vibration", "temperature", "pressure", "flow", "power"]

def prepare_features(df, window=10, sample_every=5):
    feats, labels_rul, labels_fm = [], [], []
    
    for run_id, g in df.groupby("run_id"):
        g = g.sort_values("t").reset_index(drop=True)
        
        for idx in range(window, len(g), sample_every):
            sub = g.iloc[max(0, idx-window):idx+1]
            f = rolling_features(sub, SENSOR_COLS, window=window).to_dict()
            for c in SENSOR_COLS:
                f[f"{c}_latest"] = sub[c].iloc[-1]
            feats.append(f)
            labels_rul.append(int(g["RUL"].iloc[idx]))
            labels_fm.append(g["failure_mode"].iloc[idx])
    
    X = pd.DataFrame(feats)
    y_rul = np.array(labels_rul)
    y_fm = np.array(labels_fm)
    return X, y_rul, y_fm

def create_sample_files(df):
    print("Creating sample files...")
    sample_runs = [0, 1, 5, 10, 20]
    for run_id in sample_runs:
        run_data = df[df["run_id"] == run_id].copy()
        if not run_data.empty:
            run_data = run_data.rename(columns={"t": "time_in_cycles", "run_id": "unit_nr"})
            sample_path = os.path.join(BASE_DIR, "data", f"sample_run_{run_id}.csv")
            run_data.to_csv(sample_path, index=False)
            print(f"   Saved sample run {run_id} to {sample_path}")

def generate_enhanced_dataset(df, rul_model, fm_model, feature_cols):
    """Generate enhanced dataset for fleet dashboard with realistic distribution"""
    print("\n📊 Generating enhanced fleet dataset...")
    
    fleet_data = []
    turbine_ids = df["run_id"].unique()
    
    for turbine_id in turbine_ids:
        turbine_df = df[df["run_id"] == turbine_id].copy()
        turbine_df = turbine_df.sort_values("t").reset_index(drop=True)
        
        # Take last window for prediction
        window = 12
        if len(turbine_df) < window:
            sub = turbine_df
        else:
            sub = turbine_df.iloc[-window:]
        
        try:
            # Calculate rolling features
            features = rolling_features(sub, SENSOR_COLS, window=min(window, len(sub))).to_dict()
            
            # Add latest sensor readings
            for col in SENSOR_COLS:
                features[f"{col}_latest"] = sub[col].iloc[-1]
            
            features_df = pd.DataFrame([features])
            
            # Ensure all required features are present
            for col in feature_cols:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            # Reorder columns to match training
            features_df = features_df[feature_cols]
            
            # Make predictions
            rul_pred = rul_model.predict(features_df)[0]
            fm_pred = fm_model.predict(features_df)[0]
            fm_proba = fm_model.predict_proba(features_df)[0]
            
            # Get the probability of the predicted failure mode
            fm_pred_idx = list(fm_model.classes_).index(fm_pred)
            risk_probability = fm_proba[fm_pred_idx] * 100
            
            # Determine status based on RUL
            if rul_pred < 20:
                status = "CRITICAL"
            elif rul_pred < 50:
                status = "WARNING"
            else:
                status = "NORMAL"
            
            # Get latest sensor readings
            latest_readings = turbine_df.iloc[-1]
            
            # Create fleet entry
            fleet_entry = {
                "Turbine ID": f"TURB-{turbine_id:03d}",
                "RUL (hours)": max(0, int(rul_pred)),
                "Status": status,
                "Top Risk": fm_pred,
                "Risk Probability (%)": risk_probability,
                "Operating Hours": int(latest_readings["t"]),
                "Temperature": round(latest_readings["temperature"], 2),
                "Vibration": round(latest_readings["vibration"], 2),
                "Pressure": round(latest_readings["pressure"], 2),
                "Flow": round(latest_readings["flow"], 2),
                "Power": round(latest_readings["power"], 2)
            }
            
            fleet_data.append(fleet_entry)
            
        except Exception as e:
            print(f"⚠️  Error processing turbine {turbine_id}: {e}")
            continue
    
    # Create DataFrame
    fleet_df = pd.DataFrame(fleet_data)
    
    # Create realistic distribution: 60% NORMAL, 25% WARNING, 15% CRITICAL
    print("\n🔧 Creating realistic health distribution...")
    n_turbines = len(fleet_df)
    n_normal = int(n_turbines * 0.60)
    n_warning = int(n_turbines * 0.25)
    n_critical = n_turbines - n_normal - n_warning
    
    # Shuffle indices
    indices = np.random.permutation(n_turbines)
    
    # Assign CRITICAL turbines (RUL: 1-20 hours)
    for idx in indices[:n_critical]:
        rul = np.random.randint(1, 21)
        fleet_df.loc[idx, "RUL (hours)"] = rul
        fleet_df.loc[idx, "Status"] = "CRITICAL"
        # Increase sensor readings to show degradation
        fleet_df.loc[idx, "Temperature"] += np.random.uniform(5, 15)
        fleet_df.loc[idx, "Vibration"] += np.random.uniform(0.5, 1.5)
        fleet_df.loc[idx, "Risk Probability (%)"] = np.random.uniform(75, 95)
    
    # Assign WARNING turbines (RUL: 21-50 hours)
    for idx in indices[n_critical:n_critical+n_warning]:
        rul = np.random.randint(21, 51)
        fleet_df.loc[idx, "RUL (hours)"] = rul
        fleet_df.loc[idx, "Status"] = "WARNING"
        # Moderate increase in sensor readings
        fleet_df.loc[idx, "Temperature"] += np.random.uniform(2, 8)
        fleet_df.loc[idx, "Vibration"] += np.random.uniform(0.2, 0.8)
        fleet_df.loc[idx, "Risk Probability (%)"] = np.random.uniform(50, 75)
    
    # Assign NORMAL turbines (RUL: 51-150 hours)
    for idx in indices[n_critical+n_warning:]:
        rul = np.random.randint(51, 151)
        fleet_df.loc[idx, "RUL (hours)"] = rul
        fleet_df.loc[idx, "Status"] = "NORMAL"
        fleet_df.loc[idx, "Risk Probability (%)"] = np.random.uniform(20, 50)
    
    # Save enhanced dataset
    fleet_df.to_csv(ENHANCED_DATA_PATH, index=False)
    print(f"✅ Enhanced dataset saved to {ENHANCED_DATA_PATH}")
    print(f"📊 Generated data for {len(fleet_df)} turbines")
    print(f"\nStatus distribution:")
    print(fleet_df["Status"].value_counts())
    print(f"\nFailure mode distribution:")
    print(fleet_df["Top Risk"].value_counts())
    
    return fleet_df

if __name__ == "__main__":
    print("📂 Loading data...")
    
    if not os.path.exists(DATA_PATH):
        print("Data not found, generating dataset...")
        from data_sim import make_fleet_dataset, create_legacy_compatible_dataset
        df_full = make_fleet_dataset(n_turbines=120, run_length=150)
        df = create_legacy_compatible_dataset(df_full)
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset saved to {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH)
    
    print(f"Data shape: {df.shape}")
    print(f"Failure modes distribution: {df['failure_mode'].value_counts().to_dict()}")

    X, y_rul, y_fm = prepare_features(df, window=12, sample_every=6)
    print(f"✅ Feature matrix shape: {X.shape}")

    X = X.fillna(0)

    X_train, X_test, y_r_train, y_r_test, y_fm_train, y_fm_test = train_test_split(
        X, y_rul, y_fm, test_size=0.2, random_state=42, stratify=y_fm
    )

    print("\n🚀 Training RUL Regressor...")
    rul_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rul_model.fit(X_train, y_r_train)
    rul_preds = rul_model.predict(X_test)
    rul_mae = mean_absolute_error(y_r_test, rul_preds)
    print(f"📊 RUL MAE: {rul_mae:.2f}")

    print("\n🚀 Training Failure Mode Classifier...")
    fm_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    fm_model.fit(X_train, y_fm_train)
    fm_preds = fm_model.predict(X_test)
    fm_acc = accuracy_score(y_fm_test, fm_preds)
    print(f"📊 Failure Mode Accuracy: {fm_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_fm_test, fm_preds))

    print("\n📊 Top 10 Important Features for RUL:")
    fi = pd.DataFrame({
        'feature': X.columns,
        'importance': rul_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(fi.head(10))

    print("\n💾 Saving models and metadata...")
    rul_model_path = os.path.join(MODELS_DIR, "rul_model.pkl")
    fm_model_path = os.path.join(MODELS_DIR, "fm_model.pkl")
    feature_cols_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.pkl")

    joblib.dump(rul_model, rul_model_path)
    print(f"Saved rul_model to {rul_model_path}")
    joblib.dump(fm_model, fm_model_path)
    print(f"Saved fm_model to {fm_model_path}")
    joblib.dump(list(X.columns), feature_cols_path)
    print(f"Saved feature columns to {feature_cols_path}")

    metadata = {
        "rul_model_type": "RandomForestRegressor",
        "fm_model_type": "RandomForestClassifier",
        "n_features": len(X.columns),
        "failure_modes": fm_model.classes_.tolist(),
        "feature_columns": list(X.columns),
        "rul_mae": rul_mae,
        "fm_accuracy": fm_acc
    }
    joblib.dump(metadata, metadata_path)
    print(f"Saved metadata to {metadata_path}")

    create_sample_files(df)

    # Generate enhanced dataset for fleet dashboard
    generate_enhanced_dataset(df, rul_model, fm_model, list(X.columns))

    print("\n🎉 Training completed successfully!")
    print("Run your app with: streamlit run src/master_dashboard.py")