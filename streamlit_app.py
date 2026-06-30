# streamlit_app.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime

# These imports are required so joblib can unpickle your custom classes
from custom_transformers import WaferAggregator, FeatureEngineer, SoilSensorPipeline
from all_in_one_router import AllInOneRouter
from auth import authenticate, register_user, get_all_users
from activity_logger import get_latest_logs, get_dataset_path, log_user_activity
import evaluation_utils as ev

import joblib

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'name' not in st.session_state:
    st.session_state.name = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

# Page config with white background
st.set_page_config(page_title="Sensor Fault Detection", layout="wide")

# Custom CSS for white background and black text
st.markdown("""
    <style>
    .stApp {
        background-color: #F0F2F6;
    }
    .main .block-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        max-width: 900px;
        margin: auto;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #1F2937 !important;
    }
    .stSelectbox label, .stFileUploader label {
        color: #1F2937 !important;
        font-weight: 500;
    }
    .stButton button {
        background-color: #93C5FD;
        color: white;
        width: 100%;
        padding: 0.75rem;
        font-size: 16px;
        border: none;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #60A5FA;
    }
    .stDownloadButton button {
        background-color: #93C5FD;
        color: white;
        width: 100%;
        padding: 0.75rem;
        font-size: 16px;
        border: none;
        border-radius: 8px;
    }
    .stDownloadButton button:hover {
        background-color: #60A5FA;
    }
    /* File uploader dark background styling */
    [data-testid="stFileUploader"] {
        background-color: #2D3748 !important;
        padding: 25px !important;
        border-radius: 10px !important;
        border: 2px solid #2D3748 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #2D3748 !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] label, 
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small {
        color: white !important;
    }
    /* Browse Files button styling */
    [data-testid="stFileUploader"] button {
        background-color: black !important;
        color: white !important;
        border: 2px solid white !important;
        padding: 12px 18px !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        font-size: 14px !important;
    }
    /* Reset button styling */
    button[kind="secondary"] {
        background-color: #1e6ae6 !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 12px 18px !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        font-size: 14px !important;
    }
    button[kind="secondary"]:hover {
        background-color: #1e6ae6 !important;
        color: #FFFFFF !important;
    }
    button[kind="secondary"] p {
        color: #FFFFFF !important;
        font-weight: bold !important;
    }
    button[kind="secondary"] span {
        color: #FFFFFF !important;
        font-weight: bold !important;
    }
    /* Login/Signup form styling */
    .login-container {
        max-width: 400px;
        margin: 20px auto;
        padding: 1.5rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stTextInput input {
        border-radius: 6px;
    }
    /* Center align login/signup content */
    [data-testid="stAppViewContainer"] {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
        padding: 1rem;
    }
    /* Login and Signup page buttons */
    .login-container .stButton button {
        background-color: #011f4f !important;
    }
    .login-container .stButton button:hover {
        background-color: #023a7a !important;
    }
    /* Admin Panel Styling */
    .admin-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .activity-row {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #93C5FD;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# AUTHENTICATION PAGES

def login_page():
    """Display login page"""
  # st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("## Login")
    st.markdown("---")
    
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Login", use_container_width=True):
            if username and password:
                success, role, name = authenticate(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.session_state.name = name
                    st.session_state.page = 'main'
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.warning("Please enter both username and password")
    
    with col2:
        if st.button("Sign Up", use_container_width=True):
            st.session_state.page = 'signup'
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

def signup_page():
    """Display signup page"""
    # st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("## Sign Up")
    st.markdown("---")
    
    name = st.text_input("Full Name", key="signup_name")
    username = st.text_input("Username", key="signup_username")
    password = st.text_input("Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Register", use_container_width=True):
            if name and username and password and confirm_password:
                if password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = register_user(username, password, name)
                    if success:
                        st.success(message + " - Logging you in...")
                        # Automatically log the user in
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.role = 'user'
                        st.session_state.name = name
                        st.session_state.page = 'main'
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.warning("Please fill in all fields")
    
    with col2:
        if st.button("Back to Login", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ROUTE TO APPROPRIATE PAGE

if not st.session_state.logged_in:
    if st.session_state.page == 'login':
        login_page()
    elif st.session_state.page == 'signup':
        signup_page()
    st.stop()

# ADMIN PANEL

if st.session_state.role == 'admin':
# Admin Panel Header
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.markdown("## Admin Dashboard")
    with col2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.name = None
            st.session_state.page = 'login'
            st.rerun()
    
    st.markdown("---")
    
# Get latest 5 activity logs
    logs = get_latest_logs(limit=5)
    
# Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='stat-box'>
            <h2 style='margin:0; color:white;'>{}</h2>
            <p style='margin:0; color:white;'>Total Activities</p>
        </div>
        """.format(len(logs)), unsafe_allow_html=True)
    
    with col2:
        unique_users = len(set([log['username'] for log in logs])) if logs else 0
        st.markdown("""
        <div class='stat-box'>
            <h2 style='margin:0; color:white;'>{}</h2>
            <p style='margin:0; color:white;'>Active Users</p>
        </div>
        """.format(unique_users), unsafe_allow_html=True)
    
    with col3:
        today_count = len([log for log in logs if log.get('date', '').startswith(datetime.now().strftime("%d-%m"))]) if logs else 0
        st.markdown("""
        <div class='stat-box'>
            <h2 style='margin:0; color:white;'>{}</h2>
            <p style='margin:0; color:white;'>Today's Activities</p>
        </div>
        """.format(today_count), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not logs:
        st.info("No user activity recorded yet.")
    else:
        st.markdown("### Recent User Activities (Latest 5)")
        st.markdown("<br>", unsafe_allow_html=True)
        
# Display each activity
        for idx, log in enumerate(logs):
            st.markdown(f"""
            <div class='activity-row'>
                <strong>User:</strong> {log['username']} &nbsp;|&nbsp; 
                <strong>Date:</strong> {log['date']} &nbsp;|&nbsp; 
                <strong>Sensor:</strong> {log['sensor_type']}
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 4, 4])
            
            with col1:
                st.write(f"**Record #{idx + 1}**")
            
            with col2:
                st.write(f"**Input:** `{log['input_dataset']}`")
# Download input dataset
                input_path = log.get('input_path', '')
                if input_path and os.path.exists(input_path):
                    with open(input_path, 'rb') as f:
                        st.download_button(
                            label="Download Input",
                            data=f.read(),
                            file_name=log['input_dataset'],
                            mime='text/csv',
                            key=f"input_{idx}",
                            use_container_width=True
                        )
                else:
                    st.caption("File not available")
            
            with col3:
                st.write(f"**Output:** `{log['output_dataset']}`")
# Download output dataset
                output_path = log.get('output_path', '')
                if output_path and os.path.exists(output_path):
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="Download Output",
                            data=f.read(),
                            file_name=log['output_dataset'],
                            mime='text/csv',
                            key=f"output_{idx}",
                            use_container_width=True
                        )
                else:
                    st.caption("File not available")
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Export all logs to Excel
        st.markdown("---")
        st.markdown("### Export Activity Report")
        
        if st.button("Download Activity Report (Excel)", use_container_width=True):
            try:
                # Create DataFrame from logs
                export_data = []
                for log in logs:
                    export_data.append({
                        'Username': log['username'],
                        'Timestamp': log['timestamp'],
                        'Input Dataset': log['input_dataset'],
                        'Output Dataset': log['output_dataset']
                    })
                
                df_export = pd.DataFrame(export_data)
                
                # Save to Excel
                excel_filename = f"activity_report_{datetime.now().strftime('%d-%m-%Y')}.xlsx"
                df_export.to_excel(excel_filename, index=False, engine='openpyxl')
                
                with open(excel_filename, 'rb') as f:
                    st.download_button(
                        label="Click to Download Excel File",
                        data=f.read(),
                        file_name=excel_filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key="excel_download"
                    )
                
                # Clean up temporary file
                if os.path.exists(excel_filename):
                    os.remove(excel_filename)
            except Exception as e:
                st.error(f"Error creating Excel file: {str(e)}")
                st.info("Please ensure openpyxl is installed: pip install openpyxl")
    
    st.stop()

# MAIN APPLICATION (After Login - For Users Only)

# Welcome header and Logout button on same line
col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    st.markdown(f"### Welcome, {st.session_state.name}")
with col2:
    if st.button("Reset", use_container_width=True):
        st.session_state.file_uploader_key += 1
        for k in ("pred_df", "pred_mode", "pred_input_filename"):
            st.session_state.pop(k, None)
        st.rerun()
with col3:
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.name = None
        st.session_state.page = 'login'
        st.rerun()

st.markdown("---")

# Main heading
st.markdown("## Upload Sensor Dataset")

# DROPDOWN: Select mode
mode = st.selectbox(
    "Select Sensor Type",
    [
        "-- Select a sensor --",
        "Wafer Sensor",
        "Gas Sensor",
        "Temperature Sensor",
        "Soil-Moisture Sensor",
        "Light Sensor",
        "All-in-One Sensor"
    ]
)

# File uploader
st.markdown("#### Upload CSV File")

uploaded_file = st.file_uploader(
    "Upload CSV", 
    type=["csv"], 
    label_visibility="collapsed",
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)

if mode == "-- Select a sensor --":
    st.warning("Please select a sensor type.")
    st.stop()

if not uploaded_file:
    st.info("Please upload a CSV file to start.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Helper to validate dataset matches selected sensor type
def validate_dataset(df_cols, sensor_mode):
    """Check if uploaded dataset columns match the selected sensor type"""
    from alias_utils import normalize_col
    
    raw_cols = [str(c).strip().lower() for c in df_cols]
    norm_cols = [normalize_col(c) for c in df_cols]
    
    has_sensor_value = ("sensorvalue" in norm_cols)
    
    # Wafer: Must have wafer_id OR multiple sensor_N columns 
    if sensor_mode == "Wafer Sensor":
        has_wafer_id = any("wafer" in col for col in norm_cols)
        sensor_cols = [col for col in norm_cols if col.startswith("sensor") and col != "sensorvalue"]
        has_multiple_sensors = len(sensor_cols) >= 2
        return has_wafer_id or has_multiple_sensors
    
    # Soil Moisture: Must have EXACT timestamp_ms (underscore version) + sensor_value
    if sensor_mode == "Soil-Moisture Sensor":
        has_timestamp_ms = "timestamp_ms" in raw_cols
        return has_timestamp_ms and has_sensor_value
    
    # Temperature: Must have EXACT timestamp(ms) (parentheses version) + sensor_value
    if sensor_mode == "Temperature Sensor":
        has_timestamp_paren = "timestamp(ms)" in df_cols
        return has_timestamp_paren and has_sensor_value
    
    # Gas: Must have mq2_value or (temperature AND humidity)
    if sensor_mode == "Gas Sensor":
        has_mq2 = "mq2value" in norm_cols
        has_temp_hum = ("temperature" in raw_cols and "humidity" in raw_cols)
        return has_mq2 or has_temp_hum
    
    # Light: Must have ldr_value OR ambient_light
    if sensor_mode == "Light Sensor":
        has_ldr = "ldrvalue" in norm_cols
        has_ambient = "ambientlight" in norm_cols
        return has_ldr or has_ambient
    
    return True  # Unknown mode, allow

# Helper to load specific sensor model exactly like router
def load_single_model(sensor_name):
    file_map = {
        "Wafer Sensor": "wafer_pipeline.joblib",
        "Soil-Moisture Sensor": "soil_moisture_pipeline.joblib",
        "Gas Sensor": "gas_pipeline.joblib",
        "Temperature Sensor": "temperature_pipeline.joblib",
        "Light Sensor": "ldr_pipeline.joblib",
    }
    fname = file_map.get(sensor_name)
    if not fname:
        return None

    model_path = os.path.join("models", fname)
    if not os.path.exists(model_path):
        return None

    return joblib.load(model_path)


# Prediction button
if st.button("Run Fault Detection"):
    with st.spinner("Predicting..."):

        # ALL-IN-ONE MODE - use working router
        if mode == "All-in-One Sensor":
            router = AllInOneRouter(model_dir="models")
            pred_df = router.route_and_predict(df)

        # SINGLE SENSOR MODE - load specific model
        else:
            # Validate dataset matches selected sensor type
            if not validate_dataset(df.columns, mode):
                st.error(f"❌ The dataset is not proper for {mode} sensor!")
                st.warning(f"Please upload a valid {mode} sensor dataset or use 'All-in-One (Auto Detect)' mode.")
                st.stop()
            
            model = load_single_model(mode)

            if model is None:
                st.error(f"Model not found for sensor: {mode}")
                st.stop()

            # Use the existing router logic for alias mapping
            router = AllInOneRouter(model_dir="models")

            # But override auto-detection:
            results = []
            expected_map = {
                "Wafer Sensor": "wafer",
                "Soil-Moisture Sensor": "soil",
                "Gas Sensor": "gas",
                "Temperature Sensor": "temperature",
                "Light Sensor": "light",
            }
            sensor_key = expected_map[mode]

            from alias_utils import EXPECTED_FEATURES, map_columns_with_aliases
            expected_cols = EXPECTED_FEATURES[sensor_key]

            for _, row in df.iterrows():
                row_df = row.to_frame().T

                # Alias mapping 
                rename_map, notes = map_columns_with_aliases(
                    row_df.columns, expected_cols
                )
                prepared = row_df.rename(columns=rename_map)

                # Fill missing expected columns
                for col in expected_cols:
                    if col not in prepared.columns:
                        prepared[col] = 0

                prepared = prepared[expected_cols].copy()

                # Convert to numeric
                for c in prepared.columns:
                    prepared[c] = pd.to_numeric(prepared[c], errors="coerce").fillna(0)

                # Predict
                pred = model.predict(prepared)[0]

                # Best-effort fault probability (used for confidence display
                # and for ROC/PR curves in the evaluation section below).
                fault_prob = None
                try:
                    prob_arr = ev.fault_probability(model, prepared, sensor_key)
                    if prob_arr is not None:
                        fault_prob = float(prob_arr[0])
                except Exception:
                    fault_prob = None

                out = row.to_dict()
                out["sensor_type"] = sensor_key
                out["prediction"] = pred
                out["fault_probability"] = fault_prob
                out["note"] = f"single_model:{mode}"
                results.append(out)

            pred_df = pd.DataFrame(results)

    # Reorder columns to show sensor_type, prediction and fault_probability first
    front_cols = [c for c in ['sensor_type', 'prediction', 'fault_probability'] if c in pred_df.columns]
    pred_df = pred_df[front_cols + [col for col in pred_df.columns if col not in front_cols]]

    # Log user activity immediately after prediction
    try:
        log_user_activity(
            username=st.session_state.username,
            sensor_type=mode,
            input_filename=uploaded_file.name,
            output_filename="predictions.csv",
            input_data=df,
            output_data=pred_df
        )
    except Exception as e:
        st.warning(f"Activity logging failed: {str(e)}")

    # Persist results so they survive reruns triggered by the evaluation
    # widgets below (Streamlit reruns the whole script on every interaction,
    # and st.button only returns True on the exact run it was clicked).
    st.session_state['pred_df'] = pred_df
    st.session_state['pred_mode'] = mode
    st.session_state['pred_input_filename'] = uploaded_file.name
    st.session_state['pred_input_df'] = df

# Show results + evaluation only if they belong to the file/mode currently selected
results_are_current = (
    'pred_df' in st.session_state
    and uploaded_file is not None
    and st.session_state.get('pred_input_filename') == uploaded_file.name
    and st.session_state.get('pred_mode') == mode
)

if results_are_current:
    pred_df = st.session_state['pred_df']
    input_df = st.session_state['pred_input_df']

    # Show results
    st.subheader("Predictions Preview")
    st.dataframe(pred_df.head())

    # Download button
    st.download_button(
        "Download Full Prediction CSV",
        pred_df.to_csv(index=False).encode("utf-8"),
        "predictions.csv",
        "text/csv"
    )

    # ------------------------------------------------------------------
    # MODEL EVALUATION METRICS
    # Only relevant if the uploaded file also contains the ACTUAL / true
    # fault status, so predictions can be checked against ground truth.
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Model Evaluation Metrics")
    st.caption(
        "If your uploaded CSV also contains the actual/ground-truth fault status "
        "(not just sensor readings), select that column below to evaluate this "
        "model's performance: accuracy, precision, recall, F1-score, confusion "
        "matrix, ROC-AUC and more."
    )

    label_candidates = ["-- None (skip evaluation) --"] + list(input_df.columns)
    auto_guess = ev.find_label_column(input_df.columns)
    default_idx = label_candidates.index(auto_guess) if auto_guess in label_candidates else 0

    label_col = st.selectbox(
        "Ground-truth / actual fault-status column",
        label_candidates,
        index=default_idx,
        key="eval_label_col",
    )

    if label_col != "-- None (skip evaluation) --":
        y_true_raw = input_df[label_col].reset_index(drop=True)
        y_pred_raw = pred_df["prediction"].reset_index(drop=True)

        n = min(len(y_true_raw), len(y_pred_raw))
        if len(y_true_raw) != len(y_pred_raw):
            st.warning(
                f"The label column has {len(y_true_raw)} rows but there are "
                f"{len(y_pred_raw)} predictions - comparing the first {n} rows of each."
            )
        y_true_raw = y_true_raw.iloc[:n]
        y_pred_raw = y_pred_raw.iloc[:n]

        true_unique_vals = sorted(y_true_raw.dropna().unique().tolist(), key=str)
        pred_unique_vals = sorted(y_pred_raw.dropna().unique().tolist(), key=str)

        if not true_unique_vals or not pred_unique_vals:
            st.warning("Could not find any non-empty values to evaluate in the selected column.")
        else:
            # Best-effort default guess for which value means "Faulty", based on
            # the sensor type that was actually detected/used for these rows
            # (works the same way whether the file went through a single-sensor
            # model or the All-in-One router) - the user can always override it.
            detected_sensor_types = pred_df["sensor_type"].dropna().unique().tolist()
            sensor_key_for_mode = detected_sensor_types[0] if len(detected_sensor_types) == 1 else None

            info = ev.SENSOR_LABEL_INFO.get(sensor_key_for_mode) if sensor_key_for_mode else None

            def _default_index(values, preferred):
                if preferred is not None and preferred in values:
                    return values.index(preferred)
                return 0

            col_a, col_b = st.columns(2)
            with col_a:
                true_fault_value = st.selectbox(
                    "Which value in the ground-truth column means FAULTY?",
                    true_unique_vals,
                    index=_default_index(true_unique_vals, info["fault_value"] if info else None),
                    key="eval_true_fault_value",
                )
            with col_b:
                pred_fault_value = st.selectbox(
                    "Which predicted value means FAULTY?",
                    pred_unique_vals,
                    index=_default_index(pred_unique_vals, info["fault_value"] if info else None),
                    key="eval_pred_fault_value",
                )

            if mode == "All-in-One Sensor" and pred_df["sensor_type"].nunique() > 1:
                st.info(
                    "This file was routed across more than one sensor type, so a single "
                    "'faulty' value can't perfectly describe every row - metrics below "
                    "treat the value you picked uniformly across all rows."
                )

            y_true_bin = (y_true_raw == true_fault_value).astype(int).reset_index(drop=True)
            y_pred_bin = (y_pred_raw == pred_fault_value).astype(int).reset_index(drop=True)

            # Only usable as a ROC/PR score when the whole batch came from one
            # sensor/model AND the chosen "faulty" value matches that sensor's
            # canonical fault_value (since fault_probability is P(canonical fault)).
            y_score = None
            if (
                "fault_probability" in pred_df.columns
                and pred_df["sensor_type"].nunique() == 1
            ):
                only_sensor = pred_df["sensor_type"].iloc[0]
                only_info = ev.SENSOR_LABEL_INFO.get(only_sensor)
                if only_info and only_info["fault_value"] == pred_fault_value:
                    fp = pred_df["fault_probability"].iloc[:n].reset_index(drop=True)
                    if fp.notna().all():
                        y_score = fp.values

            metrics = ev.compute_metrics(y_true_bin, y_pred_bin, pos_label=1, y_score=y_score)

            st.markdown("#### Key Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            m2.metric("Precision", f"{metrics['precision']:.3f}" if metrics['precision'] is not None else "N/A")
            m3.metric("Recall", f"{metrics['recall']:.3f}" if metrics['recall'] is not None else "N/A")
            m4.metric("F1-Score", f"{metrics['f1']:.3f}" if metrics['f1'] is not None else "N/A")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Specificity", f"{metrics['specificity']:.3f}" if metrics.get('specificity') is not None else "N/A")
            m6.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.3f}")
            m7.metric("Matthews CorrCoef", f"{metrics['mcc']:.3f}")
            m8.metric("Cohen's Kappa", f"{metrics['cohen_kappa']:.3f}")

            if metrics.get("roc_auc") is not None:
                m9, m10 = st.columns(2)
                m9.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
                m10.metric("PR-AUC (Avg. Precision)", f"{metrics['pr_auc']:.3f}")

            st.markdown("#### Confusion Matrix & Curves")
            plot_cols = st.columns(3 if metrics.get("roc_curve") else 1)

            with plot_cols[0]:
                fig_cm = ev.confusion_matrix_figure(
                    metrics["confusion_matrix"],
                    ["Normal", "Faulty"],
                    title="Confusion Matrix",
                )
                st.pyplot(fig_cm, use_container_width=True)

            if metrics.get("roc_curve"):
                fpr, tpr = metrics["roc_curve"]
                rec, prec = metrics["pr_curve"]
                with plot_cols[1]:
                    fig_roc = ev.roc_curve_figure(fpr, tpr, metrics["roc_auc"])
                    st.pyplot(fig_roc, use_container_width=True)
                with plot_cols[2]:
                    fig_pr = ev.pr_curve_figure(rec, prec, metrics["pr_auc"])
                    st.pyplot(fig_pr, use_container_width=True)
            else:
                st.caption(
                    "ROC / Precision-Recall curves need a per-row fault probability "
                    "for a single, consistently-encoded sensor model - not available "
                    "for this selection, but accuracy/precision/recall/F1 above are unaffected."
                )

            st.markdown("#### Per-Class Report")
            st.dataframe(ev.classification_report_dataframe(metrics), use_container_width=True)

            report_text = ev.metrics_to_text_report(
                metrics, title=f"Model Evaluation Report - {mode}"
            )
            st.download_button(
                "📄 Download Evaluation Report (.txt)",
                report_text.encode("utf-8"),
                "evaluation_report.txt",
                "text/plain",
            )

