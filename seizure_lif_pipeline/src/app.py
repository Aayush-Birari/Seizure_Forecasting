import streamlit as st
import re
import datetime
from inference import forecast_seizure_risk
from google_fit import fetch_todays_activity
import pandas as pd
from database import (
    get_all_categories, get_food_category, verify_login, 
    register_patient, add_new_food, get_symptom_category, 
    add_new_symptom, get_all_symptom_categories, save_daily_routine
)

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
    if not re.search(r"[A-Z]", pwd): return False
    if not re.search(r"\d", pwd): return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", pwd): return False
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
        
        # Initialize the session state array to hold historical logs
        if 'historical_logs' not in st.session_state:
            st.session_state['historical_logs'] = []

        st.subheader("Account Details")
        reg_user = st.text_input("Choose a Login ID")
        reg_pass = st.text_input("Choose a Password", type="password", help="Must contain: 8+ chars, 1 uppercase, 1 number, 1 symbol.")

        st.subheader("Medical Baseline")
        col1, col2 = st.columns(2)
        with col1:
            b_year = st.number_input("Birth Year", min_value=1900, max_value=2026, value=2000)
            gen = st.selectbox("Gender", ["Male", "Female", "Other"])
            b_diag = st.selectbox("Baseline Diagnosis", ["Pittaja Apasmara", "Kaphaja Apasmara", "Vataja Apasmara", "Sannipataja Apasmara"])
        with col2:
            y_diag = st.number_input("Year of Diagnosis", min_value=1900, max_value=2026, value=2020)
            meds = st.text_input("Prescribed Medication(s)")

        st.divider()
        
        # --- OPTIONAL HISTORICAL UPLOADER ---
        st.subheader("⏳ Optional: Add Historical Seizure Logs")
        st.caption("Adding past events helps the AI model understand your unique baseline triggers immediately.")
        
        # --- INSTANT CSV UPLOADER ---
        uploaded_file = st.file_uploader("📂 Upload Backup (CSV)", type=['csv'], help="Upload a formatted CSV backup to skip manual entry.")
        if uploaded_file is not None:
            if st.button("Load 20 Entries from Backup"):
                try:
                    df = pd.read_csv(uploaded_file)
                    for index, row in df.iterrows():
                        diet_raw = str(row['Diet (Raw)']) if pd.notna(row['Diet (Raw)']) else ""
                        prior_health = str(row['Prior Health']) if pd.notna(row['Prior Health']) else ""
                        symptoms = str(row['Symptoms']) if pd.notna(row['Symptoms']) else "None"
                        
                        st.session_state['historical_logs'].append({
                            "Date": pd.to_datetime(row['Date'], format='mixed', dayfirst=True).date(),
                            "Time of Day": str(row['Time of Day']),
                            "Diet (Raw)": diet_raw,
                            "Diet (ML Category)": str(row['Diet (ML Category)']),
                            "Fatigue": int(row['Fatigue']),
                            "Prior Health": prior_health,
                            "Duration (Sec)": int(row['Duration (Sec)']),
                            "Symptoms": symptoms
                        })
                    st.rerun() 
                except Exception as e:
                    st.error(f"Failed to read CSV. Please check the format. Error: {e}")

        # --- MANUAL PAST EVENT ENTRY ---
        with st.expander("➕ Add a Past Event Manually"):
            today = datetime.datetime.today().date()
            three_years_ago = today - datetime.timedelta(days=3*365)

            h_col1, h_col2 = st.columns(2)
            with h_col1:
                h_date = st.date_input("Date of Seizure", value=today, min_value=three_years_ago, max_value=today)
                h_time = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"], key="h_time")
                h_diet = st.text_input("Diet before event (e.g., upma, coffee)", key="h_diet").strip()
                h_fatigue = st.slider("Physical/Travel Fatigue (1-10)", 1, 10, 1, key="h_fat")
            with h_col2:
                h_health = st.text_input("Prior Health Issue (e.g., Fever, Cold)", key="h_health")
                h_duration = st.number_input("Estimated Duration (Seconds)", min_value=0, value=30, key="h_dur")
                
                h_standard_symptoms = st.multiselect("Standard Involuntary Symptoms", ["Muscle Spasms", "Loss of Consciousness", "Staring Spell", "Aura/Sensory Changes", "Vocalization"], key="h_symp")
                h_custom_symptom = st.text_input("Other Custom Symptoms:", key="h_symp_other").strip()
                h_final_symptoms = list(h_standard_symptoms)

                if h_custom_symptom:
                    h_ml_cat = get_symptom_category(h_custom_symptom)
                    if h_ml_cat == "Unknown":
                        st.warning(f"⚠️ '{h_custom_symptom}' is unknown.")
                        h_dynamic_cats = get_all_symptom_categories()
                        h_options = h_dynamic_cats + ["Other (Create New Category)"]
                        h_selected_cat = st.selectbox("Map symptom to category:", h_options, key="h_dyn_symp")
                        
                        h_final_cat = h_selected_cat
                        if h_selected_cat == "Other (Create New Category)":
                            h_final_cat = st.text_input("Enter new category name:", key="h_new_symp").strip().title()
                            
                        if st.button("Save Historical Symptom Mapping", key="h_save_symp"):
                            if h_final_cat and h_final_cat != "Other (Create New Category)":
                                add_new_symptom(h_custom_symptom, h_final_cat)
                                st.success("Symptom mapped successfully!")
                                st.rerun()
                    else:
                        st.info(f"AI Categorized As: **{h_ml_cat}**")
                        h_final_symptoms.append(h_ml_cat)
            
            if st.button("Add Log to Table"):
                _, ml_cat = get_food_category(h_diet) if h_diet else ("Neutral", "Neutral")
                st.session_state['historical_logs'].append({
                    "Date": h_date,
                    "Time of Day": h_time,
                    "Diet (Raw)": h_diet,
                    "Diet (ML Category)": ml_cat,
                    "Fatigue": h_fatigue,
                    "Prior Health": h_health,
                    "Duration (Sec)": h_duration,
                    "Symptoms": ", ".join(h_final_symptoms) if h_final_symptoms else "None"
                })
                st.rerun()

        # Display the tabular data so the patient can verify it
        if len(st.session_state['historical_logs']) > 0:
            st.write("**Pending Historical Logs (Please Verify):**")
            st.dataframe(pd.DataFrame(st.session_state['historical_logs']), width="stretch")
            if st.button("Clear Table"):
                st.session_state['historical_logs'] = []
                st.rerun()
                
        st.divider()

        # --- FINAL SUBMISSION ---
        if st.button("Complete Registration", type="primary"):
            if not is_valid_password(reg_pass):
                st.error("Password does not meet security constraints.")
            elif not reg_user or not b_diag:
                st.error("Please fill out all required fields.")
            else:
                if register_patient(reg_user, reg_pass, b_year, gen, b_diag, y_diag, meds, historical_logs=st.session_state['historical_logs']):
                    st.success("Account created successfully! Please log in.")
                    st.session_state['historical_logs'] = [] 
                    st.session_state['page'] = 'Login'
                    st.rerun()
                else:
                    st.error("Registration failed. Login ID might already exist.")

        if st.button("← Back to Login"):
            st.session_state['historical_logs'] = []
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
    display_category, ml_model_category = "Neutral", "Neutral"

    if food_input:
        display_category, ml_model_category = get_food_category(food_input)
        if display_category == "Unknown":
            st.warning(f"⚠️ '{food_input}' is not in our clinical database.")
            dynamic_categories = get_all_categories()
            options = dynamic_categories + ["Other (Add New Category)"]
            selected_cat = st.selectbox("Help us learn! What category best describes this food?", options)
            final_category = selected_cat
            
            if selected_cat == "Other (Add New Category)":
                final_category = st.text_input("Enter the new category name:").strip().title()
                
            if st.button("Add to Database & Save"):
                if final_category and final_category != "Other (Add New Category)":
                    add_new_food(food_input, final_category)
                    st.success(f"'{food_input}' permanently mapped to {final_category}!")
                    st.rerun() 
                else:
                    st.error("Please enter a valid category name.")
        else:
            st.info(f"AI Detected Trigger Category: **{display_category}**")
            
    st.divider()

    # --- WEARABLE & SENSOR INTEGRATION (GOOGLE FIT) ---
    st.subheader("⌚ Physical Activity (Google Fit / Manual)")
    
    if 'google_steps' not in st.session_state:
        st.session_state['google_steps'] = 0
        st.session_state['google_distance'] = 0.0

    w_col1, w_col2, w_col3 = st.columns([1.5, 1, 1])
    with w_col1:
        st.write("Sync your phone or smartwatch to automatically calculate fatigue limits.")
        if st.button("🔗 Connect Google Fit"):
            with st.spinner("Connecting to Google Fit..."):
                steps, distance = fetch_todays_activity()
                st.session_state['google_steps'] = steps
                st.session_state['google_distance'] = distance
                st.rerun()
            
    with w_col2:
        daily_steps = st.number_input(
            "Total Daily Steps", 
            min_value=0, max_value=100000, 
            value=st.session_state['google_steps'], 
            step=500
        )
    with w_col3:
        distance_km = st.number_input(
            "Distance Traveled (km)", 
            min_value=0.0, max_value=100.0, 
            value=float(st.session_state['google_distance']), 
            step=0.5
        )

    # Translate objective steps into our subjective fatigue score (1-10)
    auto_fatigue = 1
    if daily_steps > 15000: auto_fatigue = 8
    elif daily_steps > 10000: auto_fatigue = 5
    elif daily_steps > 5000: auto_fatigue = 3
    
    if st.session_state['google_steps'] > 0:
        st.success(f"✅ Data synced. AI has adjusted your baseline fatigue score to {auto_fatigue}/10.")

    st.divider()

    # --- FULL LOGGING METRICS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        sleep = st.number_input("Sleep Hours (Last Night)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        stress = st.slider("Psychological Stress Level (1-5)", 1, 5, 1, help="1: Relaxed, 3: Overwhelmed, 5: Severe Panic/Anxiety")
        health_issue = st.text_input("Prior Health Issue (e.g., Fever, Cold)")
        
    with col2:
        # Pre-fills the fatigue slider with the Google Fit data calculation
        fatigue = st.slider("Physical/Travel Fatigue (1-10)", 1, 10, auto_fatigue, help="1: Rested/Home, 10: Exhausting Travel")
        tod_quarter = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
        lunar = st.selectbox("Current Lunar Phase", ["Unknown", "New Moon", "First Quarter", "Full Moon", "Last Quarter"])

    with col3:
        missed_meds = st.radio("Missed Medication Dose?", [False, True])
        seizure_event = st.radio("Seizure Experienced Today?", [False, True])

        raw_symptom = st.text_input("Prodromal Symptoms (e.g., babble, visual aura):", value="None").strip()
        ml_symptom = "None"
        
        if raw_symptom and raw_symptom.lower() != "none":
            ml_symptom = get_symptom_category(raw_symptom)
            if ml_symptom == "Unknown":
                st.warning(f"⚠️ '{raw_symptom}' is an unknown symptom term.")
                
                dyn_prod_cats = get_all_symptom_categories()
                prod_options = dyn_prod_cats + ["Other (Create New Category)"]
                
                selected_symp_cat = st.selectbox("Map this to a standard medical category:", prod_options)
                
                final_prod_cat = selected_symp_cat
                if selected_symp_cat == "Other (Create New Category)":
                    final_prod_cat = st.text_input("Enter new category name:", key="new_prod_symp").strip().title()
                    
                if st.button("Save Prodromal Mapping"):
                    if final_prod_cat and final_prod_cat != "Other (Create New Category)":
                        add_new_symptom(raw_symptom, final_prod_cat)
                        st.success(f"Mapped '{raw_symptom}' to {final_prod_cat}!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid category name.")
            else:
                st.info(f"AI Categorized Symptom As: **{ml_symptom}**")

    # CONDITIONAL EVENT FIELDS (Retroactive Logging)
    if seizure_event:
        with st.expander("⚠️ Seizure Event Details", expanded=True):
            seizure_date = st.date_input("Date of Seizure", value=datetime.datetime.today())
            duration = st.number_input("Estimated Duration (Seconds)", min_value=0, value=30)
            
            standard_involuntary = st.multiselect("Standard Involuntary Symptoms", 
                ["Muscle Spasms", "Loss of Consciousness", "Staring Spell", "Aura/Sensory Changes", "Vocalization"])
            
            custom_involuntary = st.text_input("Other Custom Symptoms (e.g., lip smacking, heavy breathing):").strip()
            final_involuntary_actions = list(standard_involuntary)

            if custom_involuntary:
                ml_inv_cat = get_symptom_category(custom_involuntary)
                
                if ml_inv_cat == "Unknown":
                    st.warning(f"⚠️ '{custom_involuntary}' is not in our symptom dictionary.")
                    dynamic_symp_cats = get_all_symptom_categories()
                    symp_options = dynamic_symp_cats + ["Other (Create New Category)"]
                    
                    selected_inv_cat = st.selectbox("Map this to an existing category, or create a new one:", symp_options, key="dyn_inv_symp")
                    
                    final_symp_cat = selected_inv_cat
                    if selected_inv_cat == "Other (Create New Category)":
                        final_symp_cat = st.text_input("Enter the new clinical category name:").strip().title()
                        
                    if st.button("Save Involuntary Symptom Mapping"):
                        if final_symp_cat and final_symp_cat != "Other (Create New Category)":
                            add_new_symptom(custom_involuntary, final_symp_cat)
                            st.success(f"'{custom_involuntary}' permanently mapped to {final_symp_cat}!")
                            st.rerun()
                        else:
                            st.error("Please enter a valid category name.")
                else:
                    st.info(f"AI Categorized Custom Symptom As: **{ml_inv_cat}**")
                    final_involuntary_actions.append(ml_inv_cat)

   # --- INFERENCE EXECUTION ---
   # --- ACTION BUTTONS (SAVE & INFERENCE) ---
    st.divider()
    action_col1, action_col2 = st.columns(2)
    
    todays_inputs = {
        'food_trigger_category': ml_model_category, 
        'stress_level': stress,
        'missed_med_dose': missed_meds,
        'prodromal_symptom': ml_symptom, 
        'lunar_phase': lunar
    }

    with action_col1:
        if st.button("💾 Save Daily Log to Database", use_container_width=True):
            # Format the involuntary symptoms for the database
            inv_str = ", ".join(final_involuntary_actions) if seizure_event and final_involuntary_actions else "None"
            dur_val = duration if seizure_event else 0
            
            success = save_daily_routine(
                st.session_state['patient_id'], datetime.datetime.today().date(), tod_quarter, 
                ml_model_category, fatigue, health_issue, seizure_event, dur_val, 
                inv_str, sleep, stress, missed_meds, ml_symptom, lunar
            )
            if success:
                st.success("Log permanently saved to your medical history!")
            else:
                st.error("Failed to save log. Check terminal for details.")

    with action_col2:
        if st.button("🧠 Calculate Threshold Risk", type="primary", use_container_width=True):
            with st.spinner('Running Bayesian Inference...'):
                try:
                    result = forecast_seizure_risk(todays_inputs)
                    
                    st.subheader("Forecasting Result:")
                    if "HIGH RISK" in result:
                        st.error(result)
                    elif "MODERATE" in result:
                        st.warning(result)
                    else:
                        st.success(result)
                        
                except KeyError:
                    st.warning(f"⚠️ **Model Retraining Required:** The AI model needs to learn the newly mapped category '{ml_symptom}' from your saved logs. Please click 'Save Daily Log', then run your model training script.")

    # --- MEDICAL AI DISCLAIMER ---
    st.caption("""
    **Medical AI Disclaimer:** 
    The probability calculated by the LIF Engine is powered by a Machine Learning (ML) algorithm based on historical data patterns. 
    It is **not** a substitute for professional medical advice, diagnosis, or treatment. AI systems can make incorrect predictions. 
    Always consult with your neurologist or physician regarding your condition. Ensure your daily logs are accurate, as false data will negatively impact your risk alerts.
    """)