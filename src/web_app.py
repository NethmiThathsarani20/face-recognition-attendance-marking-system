"""Simple Flask web application for attendance system.
Minimal code approach with basic functionality.
"""

import base64
import os
from datetime import datetime

import cv2
from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

# Handle both relative and absolute imports
try:
    from .attendance_system import AttendanceSystem
    from .config import ALLOWED_EXTENSIONS, WEB_DEBUG, WEB_HOST, WEB_PORT
except ImportError:
    from attendance_system import AttendanceSystem
    from config import ALLOWED_EXTENSIONS, WEB_DEBUG, WEB_HOST, WEB_PORT


app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Initialize attendance system
attendance_system = AttendanceSystem()


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Main page with attendance marking options."""
    users = attendance_system.get_user_list()
    today_attendance = attendance_system.get_today_attendance()
    return render_template("index.html", users=users, attendance=today_attendance)


@app.route("/add_user")
def add_user():
    """Add new user page."""
    return render_template("add_user.html")


@app.route("/add_user", methods=["POST"])
def add_user_post():
    """Handle new user addition."""
    user_name = request.form.get("user_name")
    uploaded_files = request.files.getlist("user_images")

    if not user_name:
        return jsonify({"success": False, "message": "User name is required"})

    # Save uploaded files temporarily
    temp_files = []
    for file in uploaded_files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_path = os.path.join("temp", filename)
            os.makedirs("temp", exist_ok=True)
            file.save(temp_path)
            temp_files.append(temp_path)

    if not temp_files:
        return jsonify({"success": False, "message": "No valid image files provided"})

    # Add user to system
    result = attendance_system.add_new_user(user_name, temp_files)

    # Clean up temporary files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except OSError:
            pass

    if result["success"]:
        return redirect(url_for("index"))
    return jsonify(result)


@app.route("/mark_attendance_camera", methods=["POST"])
def mark_attendance_camera():
    """Mark attendance using camera capture."""
    if request.json is None:
        return jsonify({"success": False, "message": "No JSON data provided"})

    camera_source = request.json.get("camera_source", 0)
    auto_mode = request.json.get("auto_mode", False)
    save_captured = not auto_mode  # Only save in manual mode

    # Convert camera_source to appropriate type
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)

    # Capture image from camera
    image = attendance_system.capture_from_camera(camera_source)
    if image is None:
        return jsonify({"success": False, "message": "Failed to capture from camera"})

    # Mark attendance
    result = attendance_system.mark_attendance(image, save_captured=save_captured)

    # Convert image to base64 for display
    if result["success"]:
        _, buffer = cv2.imencode(".jpg", image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        result["captured_image"] = img_base64

    return jsonify(result)


@app.route("/mark_attendance_upload", methods=["POST"])
def mark_attendance_upload():
    """Mark attendance using uploaded image."""
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image file provided"})

    file = request.files["image"]
    if file.filename == "" or file.filename is None:
        return jsonify({"success": False, "message": "No image file selected"})

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Invalid file type"})

    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    file.save(temp_path)

    # Mark attendance
    result = attendance_system.mark_attendance(temp_path)

    # Clean up temporary file
    try:
        os.remove(temp_path)
    except OSError:
        pass

    return jsonify(result)


@app.route("/get_attendance")
def get_attendance():
    """Get today's attendance records."""
    attendance = attendance_system.get_today_attendance()
    return jsonify(attendance)


@app.route("/get_users")
def get_users():
    """Get list of registered users."""
    users = attendance_system.get_user_list()
    return jsonify(users)


@app.route("/camera_test/<path:camera_source>")
def camera_test(camera_source):
    """Test camera functionality."""
    # Convert camera_source to appropriate type
    if camera_source.isdigit():
        camera_source = int(camera_source)

    print(f"🧪 Testing camera: {camera_source}")
    image = attendance_system.capture_from_camera(camera_source)
    if image is not None:
        _, buffer = cv2.imencode(".jpg", image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return jsonify(
            {
                "success": True,
                "image": img_base64,
                "message": "Camera working successfully!",
            },
        )
    error_message = "Camera not available"
    if isinstance(camera_source, str):
        error_message += f" - Check IP camera URL: {camera_source}"
    return jsonify({"success": False, "message": error_message})


@app.route("/cnn_training")
def cnn_training():
    """CNN training page."""
    return render_template("cnn_training.html")


@app.route("/model_status")
def model_status():
    """Get current model status (active model and availability)."""
    try:
        status = attendance_system.get_current_model_info()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)})




@app.route("/cnn_switch_model", methods=["POST"])
def cnn_switch_model():
    """Switch between CNN, Embedding, and InsightFace models."""
    try:
        data = request.get_json()
        model_type = data.get("model_type", "insightface")

        if model_type == "cnn":
            attendance_system.switch_to_cnn_model()
        elif model_type == "embedding":
            attendance_system.switch_to_embedding_model()
        elif model_type in ("custom_embedding", "custom-embedding"):
            attendance_system.switch_to_custom_embedding_model()
        else:
            attendance_system.switch_to_insightface_model()

        return jsonify({"success": True, "message": f"Switched to {model_type} model"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# Separate switch endpoints for each model (simple to call from UI)
@app.route("/switch/insightface", methods=["POST"])
def switch_insightface():
    try:
        attendance_system.switch_to_insightface_model()
        return jsonify({"success": True, "message": "Switched to insightface model"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/switch/cnn", methods=["POST"])
def switch_cnn():
    try:
        attendance_system.switch_to_cnn_model()
        return jsonify({"success": True, "message": "Switched to cnn model"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/switch/embedding", methods=["POST"])
def switch_embedding():
    try:
        attendance_system.switch_to_embedding_model()
        return jsonify({"success": True, "message": "Switched to embedding model"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/switch/custom_embedding", methods=["POST"])
def switch_custom_embedding():
    try:
        attendance_system.switch_to_custom_embedding_model()
        return jsonify({"success": True, "message": "Switched to custom embedding model"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})




@app.route("/cnn_prepare_data", methods=["POST"])
def cnn_prepare_data():
    """Prepare training data."""
    try:
        cnn_trainer = attendance_system.get_cnn_trainer()
        success = cnn_trainer.prepare_training_data()

        if success:
            users_count = len(set(cnn_trainer.training_labels))
            samples_count = len(cnn_trainer.training_data)
            return jsonify(
                {
                    "success": True,
                    "message": "Training data prepared successfully",
                    "users_count": users_count,
                    "samples_prepared": samples_count,
                },
            )
        return jsonify({"success": False, "message": "No training data found"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/cnn_train", methods=["POST"])
def cnn_train():
    """Train the CNN model."""
    try:
        data = request.get_json()
        epochs = data.get("epochs", 50)

        cnn_trainer = attendance_system.get_cnn_trainer()

        # Prepare data first
        if not cnn_trainer.prepare_training_data():
            return jsonify({"success": False, "message": "No training data available"})

        # Train model
        result = cnn_trainer.train_model(epochs=epochs)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/cnn_add_training_images", methods=["POST"])
def cnn_add_training_images():
    """Add training images for a user."""
    try:
        user_name = request.form.get("user_name")
        uploaded_files = request.files.getlist("images")

        if not user_name:
            return jsonify({"success": False, "message": "User name is required"})

        if not uploaded_files:
            return jsonify({"success": False, "message": "No images provided"})

        # Create user directory
        from config import DATABASE_DIR

        user_dir = os.path.join(DATABASE_DIR, user_name)
        os.makedirs(user_dir, exist_ok=True)

        processed_count = 0
        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Add timestamp to avoid conflicts
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{filename}"
                file_path = os.path.join(user_dir, filename)
                file.save(file_path)
                processed_count += 1

        if processed_count > 0:
            # Update face manager database
            attendance_system.face_manager.add_user_from_database_folder(user_name)

            return jsonify(
                {
                    "success": True,
                    "message": f"Added {processed_count} training images",
                    "images_processed": processed_count,
                },
            )
        return jsonify({"success": False, "message": "No valid images processed"})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/cnn_add_training_video", methods=["POST"])
def cnn_add_training_video():
    """Extract training data from video."""
    try:
        user_name = request.form.get("user_name")
        video_file = request.files.get("video")
        frame_interval = int(request.form.get("frame_interval", 30))

        if not user_name:
            return jsonify({"success": False, "message": "User name is required"})

        if not video_file or not video_file.filename:
            return jsonify({"success": False, "message": "No video file provided"})

        # Save video temporarily
        filename = secure_filename(video_file.filename)
        temp_path = os.path.join("temp", filename)
        os.makedirs("temp", exist_ok=True)
        video_file.save(temp_path)

        try:
            # Extract training data
            cnn_trainer = attendance_system.get_cnn_trainer()
            result = cnn_trainer.add_training_data_from_video(
                temp_path, user_name, frame_interval,
            )

            # Update face manager database
            if result["success"]:
                attendance_system.face_manager.add_user_from_database_folder(user_name)

            return jsonify(result)

        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except OSError:
                pass

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


def run_app():
    """Run the Flask application."""
    app.run(host=WEB_HOST, port=WEB_PORT, debug=WEB_DEBUG)


if __name__ == "__main__":
    run_app()
