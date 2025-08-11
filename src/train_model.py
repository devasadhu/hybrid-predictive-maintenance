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

if __name__ == "__main__":
    print("📂 Loading data...")
    
    if not os.path.exists(DATA_PATH):
        print("Data not found, generating dataset...")
        from data_sim import make_dataset
        df = make_dataset(n_runs=120, run_length=150)
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

    print("\n🎉 Training completed successfully!")
    print("Run your app with: streamlit run src/app.py")
