import json
import os
import hashlib

USERS_FILE = "users.json"

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    else:
        # Create default admin user
        default_users = {
            "admin": {
                "password": hash_password("admin123"),
                "role": "admin",
                "name": "Administrator"
            }
        }
        save_users(default_users)
        return default_users

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def authenticate(username, password):
    """Authenticate user credentials"""
    users = load_users()
    if username in users:
        if users[username]["password"] == hash_password(password):
            role = users[username]["role"]
            name = users[username].get("name", username.capitalize())
            return True, role, name
    return False, None, None

def register_user(username, password, name):
    """Register a new user"""
    users = load_users()
    
    if username in users:
        return False, "Username already exists"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    users[username] = {
        "password": hash_password(password),
        "role": "user",
        "name": name
    }
    
    save_users(users)
    return True, "Registration successful"

def get_all_users():
    """Get all users (for admin)"""
    return load_users()
