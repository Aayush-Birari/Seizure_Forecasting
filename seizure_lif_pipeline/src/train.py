import os
from xml.parsers.expat import model
import joblib
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from database import fetch_training_data

def update_model():
    print("Initiating LIF Training Pipeline...")
    df = fetch_training_data()

# 2. NEW: Dummy State Initialization
# This forces the AI to learn the existence of all categories to prevent KeyErrors
    dummy_data = pd.DataFrame({
        'food_trigger_category': ['Neutral'] * 6,
        'sleep_hours': ['Good'] * 6,
        'stress_level': [1] * 6,
        'travel_fatigue_score': [1] * 6,
        'missed_med_dose': [False] * 6,
        'seizure_experienced': [False] * 6,
        'prodromal_symptom': ["None", "Visual Aura", "Dizziness", "Mood Shift", "Unusual Smell", "Auditory Changes"],
        'lunar_phase': ["Unknown", "New Moon", "First Quarter", "Full Moon", "Last Quarter", "Unknown"]
    })
    df = pd.concat([df, dummy_data], ignore_index=True)

    # 3. Define the DAG
    model = DiscreteBayesianNetwork([
        ('food_trigger_category', 'seizure_experienced'),
        ('stress_level', 'seizure_experienced'),
        ('sleep_hours', 'travel_fatigue_score'),
        ('travel_fatigue_score', 'seizure_experienced'),
        ('missed_med_dose', 'seizure_experienced'),
        ('prodromal_symptom', 'seizure_experienced'),
        ('lunar_phase', 'seizure_experienced')
    ])

    # 4. Train and Save
    model.fit(df)
    # ... code that saves the .pkl file ...

    # Dynamically resolve absolute path for MLOps stability
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "lif_bayesian_model_latest.pkl")
    
    # Ensure the models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f"Pipeline Complete: Model saved to {model_path}")

if __name__ == "__main__":
    update_model()