import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse
import hashlib
import uuid
from datetime import datetime

# ==========================================
# 1. AUTHENTICATION & REGISTRATION
# ==========================================
def fetch_training_data():
    """Extracts historical logs to retrain the Bayesian Network."""
    try:
        safe_password = urllib.parse.quote_plus("safe_password")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
      
        query = """
            SELECT 
                food_trigger_category, 
                sleep_hours, 
                stress_level, 
                travel_fatigue_score, 
                missed_med_dose,
                prodromal_symptom,
                lunar_phase,
                seizure_experienced 
            FROM daily_routine_logs;
        """
        
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            df['sleep_hours'] = pd.cut(df['sleep_hours'], bins=[0, 5, 7, 24], labels=['Poor', 'Average', 'Good'])
            
        return df

    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def hash_password(password):
    """Encrypts passwords using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_patient(login_id, password, birth_year, gender, diagnosis, diagnosis_year, medication):
    """Creates a new patient profile and calculates automatic metrics."""
    try:
        safe_password = urllib.parse.quote_plus("safe_password")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        current_year = datetime.now().year
        age = current_year - birth_year
        disease_duration = str(current_year - diagnosis_year)
        reg_date = datetime.now().strftime("%Y-%m-%d")
        patient_id = f"P-{uuid.uuid4().hex[:6].upper()}"
        hashed_pwd = hash_password(password)

        with engine.begin() as conn:
            master_query = text("""
                INSERT INTO master_patients 
                (patient_id, birth_year, age, gender, baseline_diagnosis, year_of_diagnosis, disease_duration, prescribed_medication, date) 
                VALUES (:pid, :by, :age, :g, :bd, :yd, :dd, :pm, :dt)
            """)
            conn.execute(master_query, {
                "pid": patient_id, "by": birth_year, "age": age, "g": gender,
                "bd": diagnosis, "yd": diagnosis_year, "dd": disease_duration,
                "pm": medication, "dt": reg_date
            })
            
            auth_query = text("INSERT INTO patient_auth (login_id, password_hash, patient_id) VALUES (:l, :p, :pid)")
            conn.execute(auth_query, {"l": login_id, "p": hashed_pwd, "pid": patient_id})
            
        return True
    except Exception as e:
        print(f"Registration failed: {e}")
        return False

def verify_login(username, password):
    """Verifies hashed credentials against the patient_auth table."""
    try:
        safe_password = urllib.parse.quote_plus("safe_password")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        hashed_pwd = hash_password(password)
        query = text("SELECT patient_id FROM patient_auth WHERE login_id = :user AND password_hash = :pwd")
        
        with engine.connect() as conn:
            result = pd.read_sql_query(query, conn, params={"user": username, "pwd": hashed_pwd})
            
        if not result.empty:
            return result['patient_id'].iloc[0]
        return None
    except Exception as e:
        print(f"Login failed: {e}")
        return None

# ==========================================
# 2. FOOD DICTIONARY MLOPS
# ==========================================
def get_all_categories():
    """Fetches all unique food categories currently in the database to populate the dropdown."""
    try:
        safe_password = urllib.parse.quote_plus("safe_password")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        query = text("SELECT DISTINCT food_trigger_category FROM food_dictionary")
        with engine.connect() as conn:
            result = pd.read_sql_query(query, conn)

        categories = result['food_trigger_category'].str.strip().str.title().unique().tolist()
        return sorted(categories)
    except Exception as e:
        print(f"Failed to fetch categories: {e}")
        return ['Fermented', 'Heavy/Sweet', 'High-Spice', 'High-Stimulant', 'Neutral']

def get_food_category(food_item):
    """Returns TWO values: The raw category for the UI, and the safe mapped state for the ML model."""
    try:
        safe_password = urllib.parse.quote_plus("safe_password")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        query = text("SELECT food_trigger_category FROM food_dictionary WHERE LOWER(food_item_raw) = LOWER(:food)")
        
        with engine.connect() as conn:
            result = pd.read_sql_query(query, conn, params={"food": food_item})
            
        if not result.empty:
            db_category = result['food_trigger_category'].iloc[0].strip().title()

            model_state_mapping = {
                'Fermented': 'Fermented', 'High-Spice': 'High-Spice', 'Heavy/Sweet': 'Heavy/Sweet', 
                'High-Stimulant': 'High-Stimulant', 'Neutral': 'Neutral', # Fixed self-mapping bug
                'Spicy': 'High-Spice', 'High Acidity / Sour': 'High-Spice', 'Street Food / Chaat': 'High-Spice', 
                'Heavy / Rich': 'Heavy/Sweet', 'Non-Vegetarian': 'Heavy/Sweet', 'Sweet / Desserts': 'Heavy/Sweet', 
                'Processed / Imflammatory': 'High-Stimulant', 'Processed / Inflammatory': 'High-Stimulant'
            }
            ml_state = model_state_mapping.get(db_category, 'Neutral')
            
            return db_category, ml_state
        
        return "Unknown", "Unknown"
    except Exception as e:
        print(f"Dictionary lookup failed: {e}")
        return "Unknown", "Unknown"

def add_new_food(food_item, category):
    """Expands the database when a user categorizes a new food."""
    try:
        safe_password = urllib.parse.quote_plus("safe_password")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        with engine.begin() as conn:
            query = text("INSERT INTO food_dictionary (food_item_raw, food_trigger_category) VALUES (:f, :c)")
            conn.execute(query, {"f": food_item.lower(), "c": category})
        return True
    except Exception as e:
        print(f"Adding food failed: {e}")
        return False
