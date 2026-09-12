import os
import joblib
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from database import fetch_training_data

def update_model():
    print("Initiating LIF Training Pipeline...")
    df = fetch_training_data()
    
    if df is None or df.empty:
        print("No data found. Aborting training.")
        return

    model = DiscreteBayesianNetwork([
        ('food_trigger_category', 'seizure_experienced'),
        ('stress_level', 'seizure_experienced'),
        ('sleep_hours', 'travel_fatigue_score'),
        ('travel_fatigue_score', 'seizure_experienced'),
        ('missed_med_dose', 'seizure_experienced'),
        ('prodromal_symptom', 'seizure_experienced'),
        ('lunar_phase', 'seizure_experienced')
    ])

    print("Calculating Probabilities from Historical Logs...")
    
    model.fit(df)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "lif_bayesian_model_latest.pkl")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f"Pipeline Complete: Model saved to {model_path}")

if __name__ == "__main__":
    update_model()
