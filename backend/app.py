from deepface import DeepFace
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
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
    return sqlite3.connect("database/attendance.db")

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

    return jsonify({"face_detected": bool(results.detections)})

@app.route("/register", methods=["POST"])
def register_student():
    student_name = request.form.get("student_name")
    student_id = request.form.get("student_id")
    image = request.files.get("image")

    if not student_name or not student_id or not image:
        return jsonify({"success": False, "error": "Missing required fields"}), 400

    filename = f"{student_id}_Default_{image.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    try:
        conn = create_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO students (student_name, student_id, image_path)
            VALUES (?, ?, ?)
        """, (student_name, student_id, image_path))

        cursor.execute("""
            INSERT INTO student_images (student_id, image_path, appearance_label)
            VALUES (?, ?, ?)
        """, (student_id, image_path, "Default"))

        conn.commit()
        conn.close()

        return jsonify({"success": True})

    except sqlite3.IntegrityError:
        return jsonify({"success": False, "error": "Student ID already exists"}), 400

@app.route("/add-appearance", methods=["POST"])
def add_appearance():
    student_id = request.form.get("student_id")
    appearance_label = request.form.get("appearance_label") or "Alternate"
    image = request.files.get("image")

    if not student_id or not image:
        return jsonify({"success": False, "error": "Missing student ID or image"}), 400

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    student = cursor.fetchone()

    if not student:
        conn.close()
        return jsonify({"success": False, "error": "Student not found"}), 404

    safe_label = appearance_label.replace(" ", "_")
    filename = f"{student_id}_{safe_label}_{image.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    cursor.execute("""
        INSERT INTO student_images (student_id, image_path, appearance_label)
        VALUES (?, ?, ?)
    """, (student_id, image_path, appearance_label))

    conn.commit()
    conn.close()

    return jsonify({"success": True})

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

    cursor.execute("""
        SELECT students.student_name, students.student_id, student_images.image_path
        FROM students
        JOIN student_images
        ON students.student_id = student_images.student_id
    """)

    students = cursor.fetchall()
    conn.close()

    if not students:
        os.remove(test_image_path)
        return jsonify({"success": False, "error": "No student appearance profiles found"}), 400

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
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                attendance_conn = create_connection()
                attendance_cursor = attendance_conn.cursor()

                attendance_cursor.execute("""
                    INSERT INTO attendance (student_id, student_name, timestamp)
                    VALUES (?, ?, ?)
                """, (student_id, student_name, current_time))

                attendance_conn.commit()
                attendance_conn.close()

                os.remove(test_image_path)

                return jsonify({
                    "success": True,
                    "student_name": student_name,
                    "student_id": student_id,
                    "attendance_marked": True,
                    "timestamp": current_time
                })

        except Exception as e:
            print("Error comparing face:", e)

    os.remove(test_image_path)

    return jsonify({
        "success": False,
        "error": "No matching student found"
    })

@app.route("/attendance", methods=["GET"])
def get_attendance():
    selected_date = request.args.get("date")

    conn = create_connection()
    cursor = conn.cursor()

    if selected_date:
        cursor.execute("""
            SELECT id, student_id, student_name, timestamp
            FROM attendance
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp DESC
        """, (selected_date,))
    else:
        cursor.execute("""
            SELECT id, student_id, student_name, timestamp
            FROM attendance
            ORDER BY timestamp DESC
        """)

    rows = cursor.fetchall()
    conn.close()

    attendance_records = []

    for row in rows:
        attendance_records.append({
            "id": row[0],
            "student_id": row[1],
            "student_name": row[2],
            "timestamp": row[3]
        })

    return jsonify({
        "success": True,
        "attendance": attendance_records
    })

@app.route("/dashboard-stats", methods=["GET"])
def dashboard_stats():
    selected_date = request.args.get("date")

    if not selected_date:
        selected_date = datetime.now().strftime("%Y-%m-%d")

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM attendance")
    total_attendance_records = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM attendance
        WHERE DATE(timestamp) = ?
    """, (selected_date,))
    selected_date_attendance = cursor.fetchone()[0]

    cursor.execute("""
        SELECT student_name, student_id, timestamp
        FROM attendance
        WHERE DATE(timestamp) = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (selected_date,))
    latest = cursor.fetchone()

    conn.close()

    latest_attendance = None

    if latest:
        latest_attendance = {
            "student_name": latest[0],
            "student_id": latest[1],
            "timestamp": latest[2]
        }

    return jsonify({
        "success": True,
        "selected_date": selected_date,
        "total_students": total_students,
        "total_attendance_records": total_attendance_records,
        "selected_date_attendance": selected_date_attendance,
        "latest_attendance": latest_attendance
    })

if __name__ == "__main__":
    app.run(debug=True)