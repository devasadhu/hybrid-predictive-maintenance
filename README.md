# Hybrid AI Predictive Maintenance Dashboard

**Hybrid AI prototype for predictive maintenance + HS&E risk + energy recommendation**

---

## Features

- Live simulated sensor data trends  
- Predicted Remaining Useful Life (RUL) using a trained Random Forest regressor  
- Failure mode classification probabilities  
- Health, Safety & Environment (HS&E) risk scoring  
- Basic maintenance and energy efficiency recommendations  

---

## Setup and Run

1. Clone the repo and create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows

2. Install dependencies:
pip install -r requirements.txt

3. Generate the dataset:
python src/data_sim.py

4. Train models:
python src/train_model.py

5. Run data stream simulator for a specific unit (e.g., run 0):
python src/stream_simulator.py --run 0

6. Launch the dashboard app:
streamlit run src/app.py


Notes
Easily swap in real datasets by modifying src/data_sim.py or directly feeding real CSV files.

Extend functionality by adding reinforcement learning for parameter tuning, richer explainability (e.g., SHAP), or persistent storage for events.

Project Structure
bash
Copy
Edit
├── data/                  # Sample CSV files and simulated dataset (~2.7 MB)
├── models/                # Trained ML models and metadata (~3.3 MB)
├── src/                   # Source code: app, data sim, model training, utils
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
└── venv/                  # Virtual environment (ignored in repo)