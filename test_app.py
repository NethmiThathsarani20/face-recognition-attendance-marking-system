"""Simplified test app without face recognition dependencies"""
import json
import os
from datetime import datetime
from flask import Flask, render_template, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance_records")
DATABASE_DIR = os.path.join(BASE_DIR, "database")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.route("/")
def index():
    return render_template("index.html", users=[], attendance=[])

@app.route("/add_user")
def add_user():
    """Add user page - simplified"""
    return "Add User Page (Not implemented in test)"

@app.route("/get_attendance")
def get_attendance():
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    if start_date and end_date:
        # Return filtered attendance
        from datetime import datetime, timedelta
        records = []
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            current_date = start
            while current_date <= end:
                date_str = current_date.strftime('%Y-%m-%d')
                attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.json")
                
                if os.path.exists(attendance_file):
                    with open(attendance_file) as f:
                        daily_records = json.load(f)
                        records.extend(daily_records)
                
                current_date += timedelta(days=1)
        except Exception as e:
            print(f"Error: {e}")
        
        return jsonify(records)
    else:
        # Get today's attendance
        today = datetime.now().strftime('%Y-%m-%d')
        attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.json")
        
        if os.path.exists(attendance_file):
            with open(attendance_file) as f:
                return jsonify(json.load(f))
        return jsonify([])

@app.route("/get_users")
def get_users():
    """Get list of registered users from database directory"""
    users = []
    if os.path.exists(DATABASE_DIR):
        users = [d for d in os.listdir(DATABASE_DIR) 
                if os.path.isdir(os.path.join(DATABASE_DIR, d)) and not d.startswith('.')]
    users.sort()
    return jsonify(users)

@app.route("/delete_user/<user_name>", methods=["DELETE"])
def delete_user(user_name):
    """Delete a registered user"""
    import shutil
    user_dir = os.path.join(DATABASE_DIR, user_name)
    
    if os.path.exists(user_dir):
        try:
            shutil.rmtree(user_dir)
            return jsonify({"success": True, "message": f"User '{user_name}' deleted successfully"})
        except Exception as e:
            return jsonify({"success": False, "message": f"Error deleting user: {str(e)}"})
    
    return jsonify({"success": False, "message": f"User '{user_name}' not found"})

@app.route("/initialize_system", methods=["POST"])
def initialize_system():
    """Initialize system - simplified version"""
    users = []
    if os.path.exists(DATABASE_DIR):
        users = [d for d in os.listdir(DATABASE_DIR) 
                if os.path.isdir(os.path.join(DATABASE_DIR, d)) and not d.startswith('.')]
    
    today = datetime.now().strftime('%Y-%m-%d')
    attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.json")
    
    attendance = []
    if os.path.exists(attendance_file):
        with open(attendance_file) as f:
            attendance = json.load(f)
    
    return jsonify({
        "success": True,
        "users_count": len(users),
        "attendance_count": len(attendance),
        "users": users,
        "attendance": attendance
    })

if __name__ == "__main__":
    print("Starting test web application...")
    print("Open browser at: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
