from deepface import DeepFace
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import os
import sqlite3
import tempfile

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "backend/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

def create_connection():
    conn = sqlite3.connect("database/attendance.db")
    return conn

@app.route("/")
def home():
    return "Smart Attendance Backend Running!"

@app.route("/detect", methods=["POST"])
def detect_face():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        return jsonify({"face_detected": True})
    else:
        return jsonify({"face_detected": False})

@app.route("/register", methods=["POST"])
def register_student():
    student_name = request.form.get("student_name")
    student_id = request.form.get("student_id")
    image = request.files.get("image")

    if not student_name or not student_id or not image:
        return jsonify({"success": False, "error": "Missing required fields"}), 400

    filename = f"{student_id}_{image.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    try:
        conn = create_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO students (student_name, student_id, image_path)
            VALUES (?, ?, ?)
        """, (student_name, student_id, image_path))

        conn.commit()
        conn.close()

        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "error": "Student ID already exists"}), 400

@app.route("/recognize", methods=["POST"])
def recognize_student():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    image = request.files["image"]

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    test_image_path = temp_file.name
    temp_file.close()

    image.save(test_image_path)

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT student_name, student_id, image_path FROM students")
    students = cursor.fetchall()
    conn.close()

    if not students:
        os.remove(test_image_path)
        return jsonify({"success": False, "error": "No students registered"}), 400

    for student in students:
        student_name, student_id, image_path = student

        if not os.path.exists(image_path):
            print("Missing registered image:", image_path)
            continue

        try:
            result = DeepFace.verify(
                img1_path=test_image_path,
                img2_path=image_path,
                enforce_detection=False
            )

            if result["verified"]:
                os.remove(test_image_path)
                return jsonify({
                    "success": True,
                    "student_name": student_name,
                    "student_id": student_id
                })

        except Exception as e:
            print("Error comparing face:", e)

    os.remove(test_image_path)
    return jsonify({
        "success": False,
        "error": "No matching student found"
    })

if __name__ == "__main__":
    app.run(debug=True)