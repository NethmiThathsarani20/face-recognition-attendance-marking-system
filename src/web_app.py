"""
Simple Flask web application for attendance system.
Minimal code approach with basic functionality.
"""

import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import json
from datetime import datetime

# Handle both relative and absolute imports
try:
    from .attendance_system import AttendanceSystem
    from .config import ALLOWED_EXTENSIONS, WEB_HOST, WEB_PORT, WEB_DEBUG
except ImportError:
    from attendance_system import AttendanceSystem
    from config import ALLOWED_EXTENSIONS, WEB_HOST, WEB_PORT, WEB_DEBUG


app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize attendance system
attendance_system = AttendanceSystem()


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page with attendance marking options."""
    users = attendance_system.get_user_list()
    today_attendance = attendance_system.get_today_attendance()
    return render_template('index.html', users=users, attendance=today_attendance)


@app.route('/add_user')
def add_user():
    """Add new user page."""
    return render_template('add_user.html')


@app.route('/add_user', methods=['POST'])
def add_user_post():
    """Handle new user addition."""
    user_name = request.form.get('user_name')
    uploaded_files = request.files.getlist('user_images')
    
    if not user_name:
        return jsonify({'success': False, 'message': 'User name is required'})
    
    # Save uploaded files temporarily
    temp_files = []
    for file in uploaded_files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_path = os.path.join('temp', filename)
            os.makedirs('temp', exist_ok=True)
            file.save(temp_path)
            temp_files.append(temp_path)
    
    if not temp_files:
        return jsonify({'success': False, 'message': 'No valid image files provided'})
    
    # Add user to system
    result = attendance_system.add_new_user(user_name, temp_files)
    
    # Clean up temporary files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except:
            pass
    
    if result['success']:
        return redirect(url_for('index'))
    else:
        return jsonify(result)


@app.route('/mark_attendance_camera', methods=['POST'])
def mark_attendance_camera():
    """Mark attendance using camera capture."""
    if request.json is None:
        return jsonify({'success': False, 'message': 'No JSON data provided'})
    
    camera_source = request.json.get('camera_source', 0)
    
    # Convert camera_source to appropriate type
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)
    
    # Capture image from camera
    image = attendance_system.capture_from_camera(camera_source)
    if image is None:
        return jsonify({'success': False, 'message': 'Failed to capture from camera'})
    
    # Mark attendance
    result = attendance_system.mark_attendance(image)
    
    # Convert image to base64 for display
    if result['success']:
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        result['captured_image'] = img_base64
    
    return jsonify(result)


@app.route('/mark_attendance_upload', methods=['POST'])
def mark_attendance_upload():
    """Mark attendance using uploaded image."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided'})
    
    file = request.files['image']
    if file.filename == '' or file.filename is None:
        return jsonify({'success': False, 'message': 'No image file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file type'})
    
    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join('temp', filename)
    os.makedirs('temp', exist_ok=True)
    file.save(temp_path)
    
    # Mark attendance
    result = attendance_system.mark_attendance(temp_path)
    
    # Clean up temporary file
    try:
        os.remove(temp_path)
    except:
        pass
    
    return jsonify(result)


@app.route('/get_attendance')
def get_attendance():
    """Get today's attendance records."""
    attendance = attendance_system.get_today_attendance()
    return jsonify(attendance)


@app.route('/get_users')
def get_users():
    """Get list of registered users."""
    users = attendance_system.get_user_list()
    return jsonify(users)


@app.route('/camera_test/<path:camera_source>')
def camera_test(camera_source):
    """Test camera functionality."""
    # Convert camera_source to appropriate type
    if camera_source.isdigit():
        camera_source = int(camera_source)
    
    print(f"🧪 Testing camera: {camera_source}")
    image = attendance_system.capture_from_camera(camera_source)
    if image is not None:
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'success': True, 'image': img_base64, 'message': 'Camera working successfully!'})
    else:
        error_message = 'Camera not available'
        if isinstance(camera_source, str):
            error_message += f' - Check IP camera URL: {camera_source}'
        return jsonify({'success': False, 'message': error_message})


def run_app():
    """Run the Flask application."""
    app.run(host=WEB_HOST, port=WEB_PORT, debug=WEB_DEBUG)


if __name__ == '__main__':
    run_app()
