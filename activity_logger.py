# activity_logger.py
import json
import os
from datetime import datetime

ACTIVITY_LOG_FILE = "activity_logs.json"
DATASETS_DIR = "user_datasets"

def ensure_directories():
    """Create necessary directories if they don't exist"""
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

def load_activity_logs():
    """Load activity logs from JSON file"""
    if os.path.exists(ACTIVITY_LOG_FILE):
        try:
            with open(ACTIVITY_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If file is corrupted, return empty list and recreate
            return []
    return []

def save_activity_logs(logs):
    """Save activity logs to JSON file"""
    try:
        with open(ACTIVITY_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving activity logs: {e}")

def log_user_activity(username, sensor_type, input_filename, output_filename, input_data, output_data):
    """
    Log user activity when they download predictions
    
    Args:
        username: The logged-in username
        sensor_type: The sensor type used
        input_filename: Original uploaded file name
        output_filename: Predicted CSV filename
        input_data: DataFrame of input data
        output_data: DataFrame of output predictions
    """
    ensure_directories()
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    date_only = datetime.now().strftime("%d-%m-%Y")
    
    # Create filenames without timestamp
    saved_input_filename = f"{username}_{sensor_type.replace(' ', '')}_input.csv"
    saved_output_filename = f"{username}_{sensor_type.replace(' ', '')}_output.csv"
    
    # Save datasets to user_datasets folder
    input_path = os.path.join(DATASETS_DIR, saved_input_filename)
    output_path = os.path.join(DATASETS_DIR, saved_output_filename)
    
    input_data.to_csv(input_path, index=False)
    output_data.to_csv(output_path, index=False)
    
    # Load existing logs
    logs = load_activity_logs()
    
    # Create new log entry
    log_entry = {
        "username": username,
        "timestamp": timestamp,
        "date": date_only,
        "sensor_type": sensor_type,
        "input_dataset": saved_input_filename,
        "output_dataset": saved_output_filename,
        "input_path": input_path,
        "output_path": output_path
    }
    
    # Add to beginning of list (most recent first)
    logs.insert(0, log_entry)
    
    # Save updated logs
    save_activity_logs(logs)
    
    return True

def get_latest_logs(limit=5):
    """Get the latest N activity logs"""
    logs = load_activity_logs()
    return logs[:limit]

def get_dataset_path(filename):
    """Get full path to a dataset file"""
    return os.path.join(DATASETS_DIR, filename)
