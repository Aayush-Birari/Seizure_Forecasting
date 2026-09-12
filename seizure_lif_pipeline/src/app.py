import streamlit as st
import re
from inference import forecast_seizure_risk
from database import get_all_categories, get_food_category, verify_login, register_patient, add_new_food

st.set_page_config(page_title="LIF Algorithm Engine", layout="wide")

# --- SESSION & NAVIGATION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['patient_id'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'Login'

# --- HELPER: PASSWORD VALIDATION ---
def is_valid_password(pwd):
    if len(pwd) < 8: return False
    if not re.search(r"[A-Z]", pwd): return False # Capital letter
    if not re.search(r"\d", pwd): return False    # Number
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", pwd): return False # Symbol
    return True

# ==========================================
# GUEST ROUTING (Login vs Registration)
# ==========================================
if not st.session_state['logged_in']:
    
    if st.session_state['page'] == 'Login':
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.title("🔒 Patient Login")
            with st.form("login_form"):
                username = st.text_input("Login ID")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login"):
                    pid = verify_login(username, password)
                    if pid:
                        st.session_state['logged_in'] = True
                        st.session_state['patient_id'] = pid
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
            
            st.divider()
            if st.button("New Patient? Create an Account"):
                st.session_state['page'] = 'Register'
                st.rerun()

    elif st.session_state['page'] == 'Register':
        st.title("📝 Patient Registration")
        st.info("Your data is secured using SHA-256 military-grade encryption.")
        
        with st.form("reg_form"):
            st.subheader("Account Details")
            reg_user = st.text_input("Choose a Login ID")
            reg_pass = st.text_input("Choose a Password", type="password", help="Must contain: 8+ chars, 1 uppercase, 1 number, 1 symbol.")
            
            st.subheader("Medical Baseline")
            col1, col2 = st.columns(2)
            with col1:
                b_year = st.number_input("Birth Year", min_value=1900, max_value=2026, value=2000)
                gen = st.selectbox("Gender", ["Male", "Female", "Other"])
                
                # Updated Ayurvedic Diagnosis Dropdown
                b_diag = st.selectbox("Baseline Diagnosis", [
                    "Pittaja Apasmara", 
                    "Kaphaja Apasmara", 
                    "Vataja Apasmara", 
                    "Sannipataja Apasmara"
                ])
                
            with col2:
                y_diag = st.number_input("Year of Diagnosis", min_value=1900, max_value=2026, value=2020)
                meds = st.text_input("Prescribed Medication(s)")
            
            if st.form_submit_button("Complete Registration"):
                if not is_valid_password(reg_pass):
                    st.error("Password does not meet security constraints.")
                elif not reg_user or not b_diag:
                    st.error("Please fill out all required fields.")
                else:
                    if register_patient(reg_user, reg_pass, b_year, gen, b_diag, y_diag, meds):
                        st.success("Account created successfully! Please log in.")
                        st.session_state['page'] = 'Login'
                    else:
                        st.error("Registration failed. Login ID might already exist.")
                        
        if st.button("← Back to Login"):
            st.session_state['page'] = 'Login'
            st.rerun()

# ==========================================
# MAIN APPLICATION (Logged In)
# ==========================================
else:
    st.title(f"🧠 LIF Engine | Patient: {st.session_state['patient_id']}")
    if st.button("Log Out"):
        st.session_state['logged_in'] = False
        st.rerun()
        
    st.divider()
    st.subheader("📝 Daily Routine Log")

    # --- DYNAMIC FOOD DICTIONARY EXPANSION ---
    food_input = st.text_input("Dietary Log (e.g., dosa, chilli chicken, apple):").strip()
    display_category = "Neutral"
    ml_model_category = "Neutral"
    
    if food_input:
        # Unpack the two streams: UI Display vs. ML Engine constraints
        display_category, ml_model_category = get_food_category(food_input)
        
        if display_category == "Unknown":
            st.warning(f"⚠️ '{food_input}' is not in our clinical database.")
            
            # Fetch dynamic list and append the "Other" option
            dynamic_categories = get_all_categories()
            options = dynamic_categories + ["Other (Add New Category)"]
            
            selected_cat = st.selectbox("Help us learn! What category best describes this food?", options)
            
            # If the user selects "Other", trigger the custom text box
            final_category = selected_cat
            if selected_cat == "Other (Add New Category)":
                final_category = st.text_input("Enter the new category name (e.g., Fruits, Dairy):").strip().title()
                
            if st.button("Add to Database & Save"):
                if final_category and final_category != "Other (Add New Category)":
                    add_new_food(food_input, final_category)
                    st.success(f"'{food_input}' permanently mapped to {final_category}!")
                    st.rerun() 
                else:
                    st.error("Please enter a valid category name.")
        else:
            # Displays the exact custom category the user entered (e.g., "Fruits")
            st.info(f"AI Detected Trigger Category: **{display_category}**")

   # --- FULL LOGGING METRICS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        sleep = st.number_input("Sleep Hours (Last Night)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        stress = st.slider("Psychological Stress Level (1-5)", 1, 5, 1)
        health_issue = st.text_input("Prior Health Issue (e.g., Fever, Cold)")
    with col2:
        fatigue = st.slider("Travel Fatigue Score (1-10)", 1, 10, 1)
        tod_quarter = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
        
        # NEW: Lunar Phase Dropdown
        lunar = st.selectbox("Current Lunar Phase", ["Unknown", "New Moon", "First Quarter", "Full Moon", "Last Quarter"])
        
    with col3:
        missed_meds = st.radio("Missed Medication Dose?", [False, True])
        seizure_event = st.radio("Seizure Experienced Today?", [False, True])
        
        # NEW: Prodromal Symptom Dropdown
        prodromal = st.selectbox("Prodromal Symptoms (Early Warnings)", ["None", "Visual Aura", "Dizziness", "Mood Shift", "Unusual Smell", "Auditory Changes"])

    # CONDITIONAL EVENT FIELDS (Remains the same)
    if seizure_event:
        with st.expander("⚠️ Seizure Event Details", expanded=True):
            duration = st.number_input("Estimated Duration (Seconds)", min_value=0, value=30)
            involuntary_actions = st.multiselect("Involuntary Symptoms Observed", 
                ["Muscle Spasms", "Loss of Consciousness", "Staring Spell", "Aura/Sensory Changes", "Vocalization"])

    st.divider()

    # --- INFERENCE EXECUTION ---
    if st.button("Calculate Threshold Risk", type="primary"):
        # Pass ALL the exact variables your DAG model is trained on
        todays_inputs = {
            'food_trigger_category': ml_model_category, 
            'stress_level': stress,
            'missed_med_dose': missed_meds,
            'prodromal_symptom': prodromal,
            'lunar_phase': lunar
        }
        
        with st.spinner('Running Bayesian Inference...'):
            result = forecast_seizure_risk(todays_inputs)
            
        st.subheader("Forecasting Result:")
        if "HIGH RISK" in result:
            st.error(result)
        elif "MODERATE" in result:
            st.warning(result)
        else:
            st.success(result)