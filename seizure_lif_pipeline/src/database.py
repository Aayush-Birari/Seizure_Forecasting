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
        # UPDATE THIS WITH YOUR ACTUAL PASSWORD
        safe_password = urllib.parse.quote_plus("Epilepsy2014")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        # body_state is intentionally excluded here to prevent data leakage
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

def register_patient(login_id, password, birth_year, gender, diagnosis, diagnosis_year, medication, historical_logs=None):
    """Creates a new patient profile and calculates automatic metrics."""
    try:
        # UPDATE THIS WITH YOUR ACTUAL PASSWORD
        safe_password = urllib.parse.quote_plus("Epilepsy2014")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        current_year = datetime.now().year
        age = current_year - birth_year
        disease_duration = str(current_year - diagnosis_year) # Adjusted to match varchar/integer compatibility
        reg_date = datetime.now().strftime("%Y-%m-%d")
        
        # Auto-generate a secure Patient ID
        patient_id = f"P-{uuid.uuid4().hex[:6].upper()}"
        hashed_pwd = hash_password(password)

        with engine.begin() as conn:
            # 1. INSERT INTO PARENT TABLE FIRST (master_patients)
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
            
            # 2. INSERT INTO CHILD TABLE SECOND (patient_auth)
            auth_query = text("INSERT INTO patient_auth (login_id, password_hash, patient_id) VALUES (:l, :p, :pid)")
            conn.execute(auth_query, {"l": login_id, "p": hashed_pwd, "pid": patient_id})

            # 3. Retroactive Historical Logging
            if historical_logs:
                for log in historical_logs:
                    log_query = text("""
                        INSERT INTO daily_routine_logs 
                        (patient_id, log_date, time_of_day_quarter, food_trigger_category, travel_fatigue_score, 
                        prior_health_issue, seizure_experienced, seizure_duration_seconds, involuntary_actions, 
                        sleep_hours, stress_level, missed_med_dose, prodromal_symptom, lunar_phase)
                        VALUES (:pid, :d, :tod, :ftc, :tfs, :phi, :se, :sds, :iact, :sh, :sl, :mmd, :ps, :lp)
                    """)
                    conn.execute(log_query, {
                        "pid": patient_id, "d": log['Date'], "tod": log['Time of Day'],
                        "ftc": log['Diet (ML Category)'], "tfs": log['Fatigue'], "phi": log['Prior Health'],
                        "se": True, "sds": log['Duration (Sec)'], "iact": log['Symptoms'],
                        "sh": 7.0, "sl": 1, "mmd": False, "ps": 'None', "lp": 'Unknown'
                    })                    
        return True
    except Exception as e:
        print(f"Registration failed: {e}")
        return False

def verify_login(username, password):
    """Verifies hashed credentials against the patient_auth table."""
    try:
        safe_password = urllib.parse.quote_plus("Epilepsy2014")
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
        safe_password = urllib.parse.quote_plus("Epilepsy2014")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        query = text("SELECT DISTINCT food_trigger_category FROM food_dictionary")
        with engine.connect() as conn:
            result = pd.read_sql_query(query, conn)
            
        # Clean, format, and return the unique list of categories
        categories = result['food_trigger_category'].str.strip().str.title().unique().tolist()
        return sorted(categories)
    except Exception as e:
        print(f"Failed to fetch categories: {e}")
        # Fallback list if database is unreachable
        return ['Fermented', 'Heavy/Sweet', 'High-Spice', 'High-Stimulant', 'Neutral']

def get_food_category(food_item):
    """Returns TWO values: The raw category for the UI, and the safe mapped state for the ML model."""
    try:
        safe_password = urllib.parse.quote_plus("Epilepsy2014")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        query = text("SELECT food_trigger_category FROM food_dictionary WHERE LOWER(food_item_raw) = LOWER(:food)")
        
        with engine.connect() as conn:
            result = pd.read_sql_query(query, conn, params={"food": food_item})
            
        if not result.empty:
            db_category = result['food_trigger_category'].iloc[0].strip().title()
            
            # This mapping ensures custom categories safely default to Neutral for the AI, 
            # while established triggers map to their mathematical states.
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
        safe_password = urllib.parse.quote_plus("Epilepsy2014")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        with engine.begin() as conn:
            query = text("INSERT INTO food_dictionary (food_item_raw, food_trigger_category) VALUES (:f, :c)")
            conn.execute(query, {"f": food_item.lower(), "c": category})
        return True
    except Exception as e:
        print(f"Adding food failed: {e}")
        return False

def get_symptom_category(symptom_input):
    """Checks if the synonym exists. Returns the ML-safe category."""
    try:
        safe_password = urllib.parse.quote_plus("Epilepsy2014")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        query = text("SELECT standardized_category FROM symptom_dictionary WHERE LOWER(raw_symptom) = LOWER(:s)")
        with engine.connect() as conn:
            result = pd.read_sql_query(query, conn, params={"s": symptom_input.strip()})
            
        if not result.empty:
            return result['standardized_category'].iloc[0]
        return "Unknown"
    except Exception as e:
        print(f"Symptom lookup failed: {e}")
        return "Unknown"

def add_new_symptom(raw_symptom, category):
    """Saves the user's mapped synonym permanently."""
    try:
        safe_password = urllib.parse.quote_plus("Epilepsy2014")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        with engine.begin() as conn:
            query = text("INSERT INTO symptom_dictionary (raw_symptom, standardized_category) VALUES (:r, :c)")
            conn.execute(query, {"r": raw_symptom.lower().strip(), "c": category})
        return True
    except Exception as e:
        print(f"Adding symptom failed: {e}")

def get_all_symptom_categories():
    """Fetches all unique standardized symptom categories from the database."""
    try:
        safe_password = urllib.parse.quote_plus("Epilepsy2014")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        query = text("SELECT DISTINCT standardized_category FROM symptom_dictionary")
        with engine.connect() as conn:
            result = pd.read_sql_query(query, conn)
            
        categories = result['standardized_category'].str.strip().str.title().unique().tolist()
        return sorted(categories)
    except Exception as e:
        print(f"Failed to fetch symptom categories: {e}")
        return ["Visual Aura", "Dizziness", "Mood Shift", "Unusual Smell", "Auditory Changes", "Speech Impairment", "Motor Dysfunction"]

def save_daily_routine(patient_id, log_date, tod, food, fatigue, health, seizure, duration, inv, sleep, stress, meds, prodromal, lunar):
    """Saves or updates the patient's daily log in the database."""
    try:
        # REPLACE WITH YOUR ACTUAL DATABASE PASSWORD
        safe_password = urllib.parse.quote_plus("Epilepsy2014")
        db_url = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/epilepsy_mlops_db"
        engine = create_engine(db_url)
        
        query = text("""
            INSERT INTO daily_routine_logs 
            (patient_id, log_date, time_of_day_quarter, food_trigger_category, travel_fatigue_score, 
            prior_health_issue, seizure_experienced, seizure_duration_seconds, involuntary_actions, 
            sleep_hours, stress_level, missed_med_dose, prodromal_symptom, lunar_phase)
            VALUES (:pid, :d, :tod, :ftc, :tfs, :phi, :se, :sds, :iact, :sh, :sl, :mmd, :ps, :lp)
            ON CONFLICT (patient_id, log_date, time_of_day_quarter) 
            DO UPDATE SET 
                food_trigger_category = EXCLUDED.food_trigger_category,
                travel_fatigue_score = EXCLUDED.travel_fatigue_score,
                prior_health_issue = EXCLUDED.prior_health_issue,
                seizure_experienced = EXCLUDED.seizure_experienced,
                seizure_duration_seconds = EXCLUDED.seizure_duration_seconds,
                involuntary_actions = EXCLUDED.involuntary_actions,
                sleep_hours = EXCLUDED.sleep_hours,
                stress_level = EXCLUDED.stress_level,
                missed_med_dose = EXCLUDED.missed_med_dose,
                prodromal_symptom = EXCLUDED.prodromal_symptom,
                lunar_phase = EXCLUDED.lunar_phase;
        """)
        
        with engine.connect() as conn:
            conn.execute(query, {
                "pid": patient_id, "d": log_date, "tod": tod, "ftc": food, "tfs": fatigue,
                "phi": health, "se": seizure, "sds": duration, "iact": inv, "sh": sleep,
                "sl": stress, "mmd": meds, "ps": prodromal, "lp": lunar
            })
            conn.commit()
        return True
    except Exception as e:
        print(f"Failed to save daily log: {e}")
        return False