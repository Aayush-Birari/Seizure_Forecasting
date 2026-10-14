import os
import joblib
from pgmpy.inference import VariableElimination

def forecast_seizure_risk(daily_log_evidence):
    """
    Predicts seizure risk based on today's lifestyle inputs.
    """
    # Dynamically resolve absolute path to load the artifact
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "lif_bayesian_model_latest.pkl")
    
    model = joblib.load(model_path)
    inference_engine = VariableElimination(model)
    
    print(f"Analyzing daily routine logs: {daily_log_evidence}")
    result = inference_engine.query(
        variables=['seizure_experienced'], 
        evidence=daily_log_evidence
    )
    
    # Extract and return the probability of 'True'
    risk_percentage = result.values[1] * 100 
    
    if risk_percentage > 70:
        alert = "HIGH RISK: Immediate threshold vulnerability detected."
    elif risk_percentage > 35:
        alert = "MODERATE RISK: Caution advised. Reduce physical exertion."
    else:
        alert = "LOW RISK: Baseline is stable."
        
    return f"{alert} (Probability: {risk_percentage:.2f}%)"

if __name__ == "__main__":
    # Updated to match the exact categorical values from the PostgreSQL database
    todays_inputs = {
        'food_trigger_category': 'High-Spice',  # Changed from 'High_Acidity'
        'stress_level': 4, 
        'missed_med_dose': True
    }
    
    risk_alert = forecast_seizure_risk(todays_inputs)
    print(f"\nFORECAST RESULT: {risk_alert}")