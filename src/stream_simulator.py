# stream_simulator.py
import pandas as pd
import time
import json
import os

# Paths
DATA_CSV = "../data/simulated_dataset.csv"
STREAM_JSON = "../data/stream_latest.json"

def save_stream_json(out_path, record):
    """Save latest record as JSON for Streamlit app to read."""
    with open(out_path, "w") as f:
        json.dump(record, f)

def stream_data(file_path=DATA_CSV, delay=0.5):
    """Simple generator to stream all rows (for quick testing)."""
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        yield row.to_dict()
        time.sleep(delay)

def stream_run(run_id=0, speed=0.6):
    """Stream one run_id to JSON file in real time."""
    df = pd.read_csv(DATA_CSV)
    run_df = df[df["run_id"] == run_id].sort_values("t").reset_index(drop=True)
    for _, row in run_df.iterrows():
        rec = {
            "run_id": int(row["run_id"]),
            "t": int(row["t"]),
            "vibration": float(row["vibration"]),
            "temperature": float(row["temperature"]),
            "pressure": float(row["pressure"]),
            "flow": float(row["flow"]),
            "power": float(row["power"]),
            "RUL_true": int(row["RUL"]),
            "failure_mode": row["failure_mode"]
        }
        save_stream_json(STREAM_JSON, rec)
        time.sleep(speed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=0, help="Run ID to stream")
    parser.add_argument("--speed", type=float, default=0.8, help="Seconds between updates")
    parser.add_argument("--print_only", action="store_true", help="Print stream instead of saving JSON")
    args = parser.parse_args()

    if args.print_only:
        for row in stream_data(delay=args.speed):
            print(row)
    else:
        print(f"Streaming run {args.run} to {STREAM_JSON}...")
        stream_run(run_id=args.run, speed=args.speed)
