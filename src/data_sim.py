# src/data_sim.py
import numpy as np
import pandas as pd
import os
from tqdm import trange

OUT = "../data/simulated_dataset.csv"
os.makedirs("../data", exist_ok=True)

def generate_run(run_id, length=200, seed=None):
    """
    Simulate sensors: vibration, temperature, pressure, flow, power.
    Inject gradual degradation + occasional step anomalies representing failure mode triggers.
    Returns DataFrame with columns: run_id, t, vibration, temperature, pressure, flow, power, RUL, failure_mode
    """
    if seed is not None:
        np.random.seed(seed + int(run_id))
    t = np.arange(length)

    # base normal oscillations
    vibration = 0.5 + 0.02*np.sin(0.1*t) + np.random.normal(0, 0.02, length)
    temperature = 40 + 0.05 * (t / length) * 40 + np.random.normal(0, 0.3, length)
    pressure = 5 + 0.01*np.cos(0.05*t) + np.random.normal(0, 0.05, length)
    flow = 100 + np.random.normal(0, 1, length) - 0.05 * t
    power = 200 + 0.1*t + np.random.normal(0, 2, length)

    # introduce degradation: vibration increases, temperature spikes near end
    degrade_start = int(length * 0.6)
    for i in range(degrade_start, length):
        degradation_factor = (i - degrade_start) / (length - degrade_start)
        vibration[i] += 0.3 * degradation_factor
        temperature[i] += 10 * degradation_factor

    # failure mode: randomly pick one mode per run
    failure_modes = ["bearing_wear", "seal_leak", "misalignment", "overheat"]
    fm = np.random.choice(failure_modes)

    # inject distinctive signature depending on fm
    degradation_pattern = np.linspace(0, 1, length)
    if fm == "bearing_wear":
        vibration += 0.5 * degradation_pattern
        power += 15 * degradation_pattern
    elif fm == "seal_leak":
        pressure -= 0.8 * degradation_pattern
        flow -= 8 * degradation_pattern
    elif fm == "misalignment":
        vibration += 0.3 * np.sin(0.2*t) * degradation_pattern
        power += 20 * degradation_pattern
    elif fm == "overheat":
        temperature += 15 * degradation_pattern
        power += 25 * degradation_pattern

    # RUL: decreasing integers to 0 at end
    RUL = np.arange(length - 1, -1, -1)

    df = pd.DataFrame({
        "run_id": run_id,
        "t": t,
        "vibration": vibration,
        "temperature": temperature,
        "pressure": pressure,
        "flow": flow,
        "power": power,
        "RUL": RUL,
        "failure_mode": fm
    })
    return df

def make_dataset(n_runs=150, run_length=160):
    """Generate multiple runs and merge into one dataset."""
    frames = []
    for i in trange(n_runs, desc="Generating runs"):
        frames.append(generate_run(i, length=run_length, seed=42))
    return pd.concat(frames, ignore_index=True)

def generate_simple(n_samples=1000):
    """Generate simpler random dataset (quick version)."""
    np.random.seed(42)
    time = np.arange(n_samples)
    temp = np.random.normal(70, 5, n_samples) + np.linspace(0, 20, n_samples)
    vibration = np.random.normal(0.5, 0.05, n_samples) + np.linspace(0, 0.5, n_samples) / 100
    pressure = np.random.normal(100, 10, n_samples) - np.linear(0, 5, n_samples)
    power = np.random.normal(200, 15, n_samples) + temp * 0.5
    rul = np.maximum(0, 50 - (time / 20) + np.random.normal(0, 2, n_samples))

    failure_mode = np.where(temp > 90, "overheat",
                     np.where(vibration > 0.7, "bearing_wear", "normal"))

    df = pd.DataFrame({
        "time": time,
        "temperature": temp,
        "vibration": vibration,
        "pressure": pressure,
        "power": power,
        "RUL": rul,
        "failure_mode": failure_mode
    })
    return df

if __name__ == "__main__":
    print("Generating dataset...")
    df = make_dataset(n_runs=120, run_length=150)
    df.to_csv(OUT, index=False)
    print(f"✅ Dataset saved to {OUT} with shape {df.shape}")
    print(f"Failure modes: {df['failure_mode'].value_counts().to_dict()}")