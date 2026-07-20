from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import csv
import io
import os
import sqlite3
import tempfile
import threading

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


# ---------------------------------------------------------
# Azure-safe absolute paths
# ---------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

UPLOAD_FOLDER = os.path.join(BACKEND_DIR, "uploads")
DATABASE_PATH = os.path.join(PROJECT_ROOT, "database", "attendance.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

# ---------------------------------------------------------
# Lazy-loaded AI dependencies
# ---------------------------------------------------------
_deepface = None
_face_detector = None
_ai_lock = threading.Lock()


def get_deepface():
    global _deepface

    if _deepface is None:
        with _ai_lock:
            if _deepface is None:
                print("Loading DeepFace...", flush=True)
                from deepface import DeepFace

                _deepface = DeepFace
                print("DeepFace loaded.", flush=True)

    return _deepface


def get_face_detector():
    global _face_detector

    if _face_detector is None:
        with _ai_lock:
            if _face_detector is None:
                print("Loading MediaPipe face detector...", flush=True)
                import mediapipe as mp

                _face_detector = (
                    mp.solutions.face_detection.FaceDetection(
                        model_selection=0,
                        min_detection_confidence=0.5,
                    )
                )

                print("MediaPipe face detector loaded.", flush=True)

    return _face_detector


def create_connection():
    conn = sqlite3.connect(DATABASE_PATH, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as error:
        print(
            f"Could not remove temporary file {path}: {error}",
            flush=True,
        )


# ---------------------------------------------------------
# Health routes
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "running",
            "service": "Smart Attendance Backend",
        }
    ), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "success": True,
            "status": "healthy",
        }
    ), 200


# ---------------------------------------------------------
# Face detection
# ---------------------------------------------------------
@app.route("/detect", methods=["POST"])
def detect_face():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        import cv2
        import numpy as np

        file = request.files["image"]

        image_bytes = file.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detector = get_face_detector()
        results = detector.process(rgb)

        return jsonify(
            {
                "success": True,
                "face_detected": bool(results.detections),
            }
        ), 200

    except Exception as error:
        print(f"Face detection error: {error}", flush=True)

        return jsonify(
            {
                "success": False,
                "error": "Face detection failed",
            }
        ), 500


# ---------------------------------------------------------
# Student registration
# ---------------------------------------------------------
@app.route("/register", methods=["POST"])
def register_student():
    student_name = request.form.get("student_name")
    student_id = request.form.get("student_id")
    image = request.files.get("image")

    if not student_name or not student_id or not image:
        return jsonify(
            {
                "success": False,
                "error": "Missing required fields",
            }
        ), 400

    original_filename = os.path.basename(
        image.filename or "image.jpg"
    )

    filename = (
        f"{student_id}_Default_{original_filename}"
    )

    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename,
    )

    image.save(image_path)

    conn = None

    try:
        conn = create_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO students
                (student_name, student_id, image_path)
            VALUES (?, ?, ?)
            """,
            (
                student_name,
                student_id,
                image_path,
            ),
        )

        cursor.execute(
            """
            INSERT INTO student_images
                (
                    student_id,
                    image_path,
                    appearance_label
                )
            VALUES (?, ?, ?)
            """,
            (
                student_id,
                image_path,
                "Default",
            ),
        )

        conn.commit()

        return jsonify(
            {
                "success": True,
            }
        ), 201

    except sqlite3.IntegrityError:
        safe_remove(image_path)

        return jsonify(
            {
                "success": False,
                "error": "Student ID already exists",
            }
        ), 400

    except Exception as error:
        safe_remove(image_path)

        print(
            f"Student registration error: {error}",
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": "Could not register student",
            }
        ), 500

    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------
# Alternate appearances
# ---------------------------------------------------------
@app.route("/add-appearance", methods=["POST"])
def add_appearance():
    student_id = request.form.get("student_id")
    appearance_label = (
        request.form.get("appearance_label")
        or "Alternate"
    )

    image = request.files.get("image")

    if not student_id or not image:
        return jsonify(
            {
                "success": False,
                "error": "Missing student ID or image",
            }
        ), 400

    conn = None
    image_path = None

    try:
        conn = create_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT student_id
            FROM students
            WHERE student_id = ?
            """,
            (student_id,),
        )

        student = cursor.fetchone()

        if not student:
            return jsonify(
                {
                    "success": False,
                    "error": "Student not found",
                }
            ), 404

        safe_label = "".join(
            character
            if character.isalnum()
            or character in "-_"
            else "_"
            for character in appearance_label
        )

        original_filename = os.path.basename(
            image.filename or "image.jpg"
        )

        filename = (
            f"{student_id}_"
            f"{safe_label}_"
            f"{original_filename}"
        )

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename,
        )

        image.save(image_path)

        cursor.execute(
            """
            INSERT INTO student_images
                (
                    student_id,
                    image_path,
                    appearance_label
                )
            VALUES (?, ?, ?)
            """,
            (
                student_id,
                image_path,
                appearance_label,
            ),
        )

        conn.commit()

        return jsonify(
            {
                "success": True,
            }
        ), 201

    except Exception as error:
        if image_path:
            safe_remove(image_path)

        print(
            f"Add appearance error: {error}",
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": "Could not add appearance",
            }
        ), 500

    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------
# Students
# ---------------------------------------------------------
@app.route("/students", methods=["GET"])
def get_students():
    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT student_id, student_name
            FROM students
            ORDER BY student_name ASC
            """
        )

        rows = cursor.fetchall()

        students = []

        for row in rows:
            students.append(
                {
                    "student_id": row[0],
                    "student_name": row[1],
                }
            )

        return jsonify(
            {
                "success": True,
                "students": students,
            }
        ), 200

    finally:
        conn.close()


# ---------------------------------------------------------
# Classes
# ---------------------------------------------------------
@app.route("/classes", methods=["POST"])
def create_class():
    course_name = request.form.get("course_name")
    course_code = request.form.get("course_code")
    professor_name = request.form.get(
        "professor_name"
    )

    if (
        not course_name
        or not course_code
        or not professor_name
    ):
        return jsonify(
            {
                "success": False,
                "error": "Missing required fields",
            }
        ), 400

    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO classes
                (
                    course_name,
                    course_code,
                    professor_name
                )
            VALUES (?, ?, ?)
            """,
            (
                course_name,
                course_code,
                professor_name,
            ),
        )

        conn.commit()

        return jsonify(
            {
                "success": True,
                "class_id": cursor.lastrowid,
            }
        ), 201

    except sqlite3.IntegrityError as error:
        return jsonify(
            {
                "success": False,
                "error": str(error),
            }
        ), 400

    finally:
        conn.close()


@app.route("/classes", methods=["GET"])
def get_classes():
    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                course_name,
                course_code,
                professor_name
            FROM classes
            ORDER BY id DESC
            """
        )

        rows = cursor.fetchall()

        classes = []

        for row in rows:
            classes.append(
                {
                    "id": row[0],
                    "course_name": row[1],
                    "course_code": row[2],
                    "professor_name": row[3],
                }
            )

        return jsonify(
            {
                "success": True,
                "classes": classes,
            }
        ), 200

    finally:
        conn.close()


# ---------------------------------------------------------
# Class enrollment
# ---------------------------------------------------------
@app.route("/class-students", methods=["POST"])
def add_student_to_class():
    class_id = request.form.get("class_id")
    student_id = request.form.get("student_id")

    if not class_id or not student_id:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Missing class ID or student ID"
                ),
            }
        ), 400

    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id
            FROM classes
            WHERE id = ?
            """,
            (class_id,),
        )

        if not cursor.fetchone():
            return jsonify(
                {
                    "success": False,
                    "error": "Class not found",
                }
            ), 404

        cursor.execute(
            """
            SELECT student_id
            FROM students
            WHERE student_id = ?
            """,
            (student_id,),
        )

        if not cursor.fetchone():
            return jsonify(
                {
                    "success": False,
                    "error": "Student not found",
                }
            ), 404

        cursor.execute(
            """
            INSERT INTO class_students
                (class_id, student_id)
            VALUES (?, ?)
            """,
            (
                class_id,
                student_id,
            ),
        )

        conn.commit()

        return jsonify(
            {
                "success": True,
            }
        ), 201

    except sqlite3.IntegrityError:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Student is already enrolled "
                    "in this class"
                ),
            }
        ), 400

    finally:
        conn.close()


@app.route(
    "/class-students/<int:class_id>",
    methods=["GET"],
)
def get_class_students(class_id):
    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                students.student_id,
                students.student_name
            FROM class_students
            JOIN students
                ON class_students.student_id =
                   students.student_id
            WHERE class_students.class_id = ?
            ORDER BY students.student_name ASC
            """,
            (class_id,),
        )

        rows = cursor.fetchall()

        students = []

        for row in rows:
            students.append(
                {
                    "student_id": row[0],
                    "student_name": row[1],
                }
            )

        return jsonify(
            {
                "success": True,
                "students": students,
            }
        ), 200

    finally:
        conn.close()


@app.route(
    "/class-students/<int:class_id>/<student_id>",
    methods=["DELETE"],
)
def remove_student_from_class(
    class_id,
    student_id,
):
    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM class_students
            WHERE class_id = ?
              AND student_id = ?
            """,
            (
                class_id,
                student_id,
            ),
        )

        conn.commit()

        return jsonify(
            {
                "success": True,
                "removed": cursor.rowcount > 0,
            }
        ), 200

    finally:
        conn.close()


# ---------------------------------------------------------
# Face recognition
# ---------------------------------------------------------
@app.route("/recognize", methods=["POST"])
def recognize_student():
    if "image" not in request.files:
        return jsonify(
            {
                "success": False,
                "error": "No image uploaded",
            }
        ), 400

    class_id = request.form.get("class_id")

    if not class_id:
        return jsonify(
            {
                "success": False,
                "error": "Please select a class",
            }
        ), 400

    image = request.files["image"]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg",
    )

    test_image_path = temp_file.name
    temp_file.close()

    image.save(test_image_path)

    try:
        conn = create_connection()

        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    students.student_name,
                    students.student_id,
                    student_images.image_path
                FROM students
                JOIN student_images
                    ON students.student_id =
                       student_images.student_id
                JOIN class_students
                    ON students.student_id =
                       class_students.student_id
                WHERE class_students.class_id = ?
                """,
                (class_id,),
            )

            students = cursor.fetchall()

        finally:
            conn.close()

        if not students:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "No student appearance profiles "
                        "found for this class"
                    ),
                }
            ), 400

        deepface = get_deepface()

        for student in students:
            student_name = student[0]
            student_id = student[1]
            image_path = student[2]

            if not os.path.exists(image_path):
                print(
                    "Missing registered image:",
                    image_path,
                    flush=True,
                )
                continue

            try:
                result = deepface.verify(
                    img1_path=test_image_path,
                    img2_path=image_path,
                    enforce_detection=False,
                )

                if result.get("verified"):
                    current_time = (
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )

                    attendance_conn = (
                        create_connection()
                    )

                    try:
                        attendance_cursor = (
                            attendance_conn.cursor()
                        )

                        attendance_cursor.execute(
                            """
                            SELECT id
                            FROM attendance
                            WHERE class_id = ?
                              AND student_id = ?
                              AND DATE(timestamp) =
                                  DATE(?)
                            LIMIT 1
                            """,
                            (
                                class_id,
                                student_id,
                                current_time,
                            ),
                        )

                        existing_record = (
                            attendance_cursor.fetchone()
                        )

                        if existing_record:
                            attendance_cursor.execute(
                                """
                                UPDATE attendance
                                SET
                                    student_name = ?,
                                    timestamp = ?,
                                    status = 'Present'
                                WHERE id = ?
                                """,
                                (
                                    student_name,
                                    current_time,
                                    existing_record[0],
                                ),
                            )

                        else:
                            attendance_cursor.execute(
                                """
                                INSERT INTO attendance
                                    (
                                        class_id,
                                        student_id,
                                        student_name,
                                        timestamp,
                                        status
                                    )
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (
                                    class_id,
                                    student_id,
                                    student_name,
                                    current_time,
                                    "Present",
                                ),
                            )

                        attendance_conn.commit()

                    finally:
                        attendance_conn.close()

                    return jsonify(
                        {
                            "success": True,
                            "student_name": student_name,
                            "student_id": student_id,
                            "class_id": class_id,
                            "attendance_marked": True,
                            "status": "Present",
                            "timestamp": current_time,
                        }
                    ), 200

            except Exception as error:
                print(
                    (
                        "Error comparing face for "
                        f"{student_id}: {error}"
                    ),
                    flush=True,
                )

        return jsonify(
            {
                "success": False,
                "error": (
                    "No matching student found "
                    "in selected class"
                ),
            }
        ), 404

    except Exception as error:
        print(
            f"Recognition error: {error}",
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": "Face recognition failed",
            }
        ), 500

    finally:
        safe_remove(test_image_path)


# ---------------------------------------------------------
# Attendance
# ---------------------------------------------------------
@app.route("/attendance", methods=["GET"])
def get_attendance():
    selected_date = request.args.get("date")
    class_id = request.args.get("class_id")

    if not selected_date or not class_id:
        return jsonify(
            {
                "success": True,
                "attendance": [],
                "message": (
                    "Please select a class and date"
                ),
            }
        ), 200

    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                attendance.id,
                attendance.student_id,
                attendance.student_name,
                attendance.timestamp,
                attendance.status,
                classes.course_code,
                classes.course_name
            FROM attendance
            LEFT JOIN classes
                ON attendance.class_id =
                   classes.id
            WHERE DATE(attendance.timestamp) = ?
              AND attendance.class_id = ?
            ORDER BY attendance.timestamp DESC
            """,
            (
                selected_date,
                class_id,
            ),
        )

        rows = cursor.fetchall()

        attendance_records = []

        for row in rows:
            attendance_records.append(
                {
                    "id": row[0],
                    "student_id": row[1],
                    "student_name": row[2],
                    "timestamp": row[3],
                    "status": row[4],
                    "course_code": row[5],
                    "course_name": row[6],
                }
            )

        return jsonify(
            {
                "success": True,
                "attendance": attendance_records,
            }
        ), 200

    finally:
        conn.close()


# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------
@app.route("/dashboard-stats", methods=["GET"])
def dashboard_stats():
    selected_date = request.args.get("date")

    if not selected_date:
        selected_date = datetime.now().strftime(
            "%Y-%m-%d"
        )

    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM students"
        )

        total_students = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM attendance"
        )

        total_attendance_records = (
            cursor.fetchone()[0]
        )

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM attendance
            WHERE DATE(timestamp) = ?
            """,
            (selected_date,),
        )

        selected_date_attendance = (
            cursor.fetchone()[0]
        )

        cursor.execute(
            """
            SELECT
                student_name,
                student_id,
                timestamp
            FROM attendance
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (selected_date,),
        )

        latest = cursor.fetchone()

        latest_attendance = None

        if latest:
            latest_attendance = {
                "student_name": latest[0],
                "student_id": latest[1],
                "timestamp": latest[2],
            }

        return jsonify(
            {
                "success": True,
                "selected_date": selected_date,
                "total_students": total_students,
                "total_attendance_records": (
                    total_attendance_records
                ),
                "selected_date_attendance": (
                    selected_date_attendance
                ),
                "latest_attendance": (
                    latest_attendance
                ),
            }
        ), 200

    finally:
        conn.close()


# ---------------------------------------------------------
# Attendance policies
# ---------------------------------------------------------
@app.route(
    "/attendance-policies",
    methods=["POST"],
)
def create_attendance_policy():
    class_id = request.form.get("class_id")
    policy_name = request.form.get(
        "policy_name"
    )

    absence_limit = request.form.get(
        "absence_limit"
    )

    late_limit = request.form.get(
        "late_limit"
    )

    late_minutes = request.form.get(
        "late_minutes"
    )

    attendance_weight = request.form.get(
        "attendance_weight"
    )

    consequence = request.form.get(
        "consequence"
    )

    excuse_counts = (
        request.form.get("excuse_counts")
        or "No"
    )

    if not class_id or not policy_name:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Class and policy name "
                    "are required"
                ),
            }
        ), 400

    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO attendance_policies
                (
                    class_id,
                    policy_name,
                    absence_limit,
                    late_limit,
                    late_minutes,
                    attendance_weight,
                    consequence,
                    excuse_counts
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                class_id,
                policy_name,
                absence_limit,
                late_limit,
                late_minutes,
                attendance_weight,
                consequence,
                excuse_counts,
            ),
        )

        conn.commit()

        return jsonify(
            {
                "success": True,
            }
        ), 201

    finally:
        conn.close()


@app.route(
    "/attendance-policies/<int:class_id>",
    methods=["GET"],
)
def get_attendance_policies(class_id):
    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                class_id,
                policy_name,
                absence_limit,
                late_limit,
                late_minutes,
                attendance_weight,
                consequence,
                excuse_counts
            FROM attendance_policies
            WHERE class_id = ?
            ORDER BY id DESC
            """,
            (class_id,),
        )

        rows = cursor.fetchall()

        policies = []

        for row in rows:
            policies.append(
                {
                    "id": row[0],
                    "class_id": row[1],
                    "policy_name": row[2],
                    "absence_limit": row[3],
                    "late_limit": row[4],
                    "late_minutes": row[5],
                    "attendance_weight": row[6],
                    "consequence": row[7],
                    "excuse_counts": row[8],
                }
            )

        return jsonify(
            {
                "success": True,
                "policies": policies,
            }
        ), 200

    finally:
        conn.close()


@app.route(
    "/attendance-policies/<int:policy_id>",
    methods=["PUT"],
)
def update_attendance_policy(policy_id):
    data = request.get_json(silent=True) or {}

    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE attendance_policies
            SET
                policy_name = ?,
                absence_limit = ?,
                late_limit = ?,
                late_minutes = ?,
                attendance_weight = ?,
                consequence = ?,
                excuse_counts = ?
            WHERE id = ?
            """,
            (
                data.get("policy_name"),
                data.get("absence_limit"),
                data.get("late_limit"),
                data.get("late_minutes"),
                data.get("attendance_weight"),
                data.get("consequence"),
                data.get("excuse_counts"),
                policy_id,
            ),
        )

        conn.commit()

        return jsonify(
            {
                "success": True,
                "updated": cursor.rowcount > 0,
            }
        ), 200

    finally:
        conn.close()


@app.route(
    "/attendance-policies/<int:policy_id>",
    methods=["DELETE"],
)
def delete_attendance_policy(policy_id):
    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM attendance_policies
            WHERE id = ?
            """,
            (policy_id,),
        )

        conn.commit()

        return jsonify(
            {
                "success": True,
                "deleted": cursor.rowcount > 0,
            }
        ), 200

    finally:
        conn.close()


# ---------------------------------------------------------
# Policy risk
# ---------------------------------------------------------
@app.route(
    "/policy-risk/<int:class_id>",
    methods=["GET"],
)
def policy_risk(class_id):
    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                policy_name,
                absence_limit,
                consequence
            FROM attendance_policies
            WHERE class_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (class_id,),
        )

        policy = cursor.fetchone()

        if not policy:
            return jsonify(
                {
                    "success": True,
                    "has_policy": False,
                    "message": (
                        "No attendance policy set "
                        "for this class"
                    ),
                    "at_risk": [],
                }
            ), 200

        policy_id = policy[0]
        policy_name = policy[1]
        absence_limit = policy[2]
        consequence = policy[3]

        cursor.execute(
            """
            SELECT
                students.student_id,
                students.student_name
            FROM class_students
            JOIN students
                ON class_students.student_id =
                   students.student_id
            WHERE class_students.class_id = ?
            """,
            (class_id,),
        )

        enrolled_students = cursor.fetchall()

        cursor.execute(
            """
            SELECT COUNT(
                DISTINCT DATE(timestamp)
            )
            FROM attendance
            WHERE class_id = ?
            """,
            (class_id,),
        )

        total_class_days = cursor.fetchone()[0]

        at_risk = []

        for student in enrolled_students:
            student_id = student[0]
            student_name = student[1]

            cursor.execute(
                """
                SELECT COUNT(
                    DISTINCT DATE(timestamp)
                )
                FROM attendance
                WHERE class_id = ?
                  AND student_id = ?
                  AND status = 'Present'
                """,
                (
                    class_id,
                    student_id,
                ),
            )

            present_days = cursor.fetchone()[0]

            absences = max(
                total_class_days - present_days,
                0,
            )

            if (
                absence_limit is not None
                and absences >= absence_limit
            ):
                at_risk.append(
                    {
                        "student_id": student_id,
                        "student_name": (
                            student_name
                        ),
                        "absences": absences,
                        "absence_limit": (
                            absence_limit
                        ),
                    }
                )

        return jsonify(
            {
                "success": True,
                "has_policy": True,
                "policy_id": policy_id,
                "policy_name": policy_name,
                "absence_limit": absence_limit,
                "consequence": consequence,
                "total_class_days": (
                    total_class_days
                ),
                "at_risk": at_risk,
            }
        ), 200

    finally:
        conn.close()


# ---------------------------------------------------------
# Attendance trend
# ---------------------------------------------------------
@app.route(
    "/attendance-trend/<int:class_id>",
    methods=["GET"],
)
def attendance_trend(class_id):
    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM class_students
            WHERE class_id = ?
            """,
            (class_id,),
        )

        total_students = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT
                DATE(timestamp)
                    AS attendance_date,
                COUNT(DISTINCT student_id)
                    AS present_count
            FROM attendance
            WHERE class_id = ?
              AND status = 'Present'
            GROUP BY DATE(timestamp)
            ORDER BY DATE(timestamp) DESC
            LIMIT 5
            """,
            (class_id,),
        )

        rows = cursor.fetchall()

        trend = []

        for row in reversed(rows):
            attendance_date = row[0]
            present_count = row[1]

            rate = 0

            if total_students > 0:
                rate = round(
                    (
                        present_count
                        / total_students
                    )
                    * 100
                )

            trend.append(
                {
                    "date": attendance_date,
                    "present_count": (
                        present_count
                    ),
                    "total_students": (
                        total_students
                    ),
                    "attendance_rate": rate,
                }
            )

        return jsonify(
            {
                "success": True,
                "trend": trend,
            }
        ), 200

    finally:
        conn.close()


# ---------------------------------------------------------
# CSV export
# ---------------------------------------------------------
@app.route(
    "/export-attendance",
    methods=["GET"],
)
def export_attendance():
    class_id = request.args.get("class_id")
    selected_date = request.args.get("date")

    if not class_id or not selected_date:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Class and date are required"
                ),
            }
        ), 400

    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                classes.course_code,
                classes.course_name,
                attendance.student_id,
                attendance.student_name,
                attendance.timestamp,
                attendance.status
            FROM attendance
            JOIN classes
                ON attendance.class_id =
                   classes.id
            WHERE attendance.class_id = ?
              AND DATE(attendance.timestamp) = ?
            ORDER BY attendance.timestamp ASC
            """,
            (
                class_id,
                selected_date,
            ),
        )

        rows = cursor.fetchall()

    finally:
        conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "Course Code",
            "Course Name",
            "Student ID",
            "Student Name",
            "Timestamp",
            "Status",
        ]
    )

    writer.writerows(rows)

    response = app.response_class(
        response=output.getvalue(),
        status=200,
        mimetype="text/csv",
    )

    response.headers["Content-Disposition"] = (
        "attachment; "
        f"filename=attendance_"
        f"{class_id}_"
        f"{selected_date}.csv"
    )

    return response


# ---------------------------------------------------------
# Excuses
# ---------------------------------------------------------
@app.route("/excuses", methods=["POST"])
def submit_excuse():
    student_id = request.form.get("student_id")
    class_id = request.form.get("class_id")
    excuse_date = request.form.get(
        "excuse_date"
    )
    reason = request.form.get("reason")

    if (
        not student_id
        or not class_id
        or not excuse_date
        or not reason
    ):
        return jsonify(
            {
                "success": False,
                "error": "Missing required fields",
            }
        ), 400

    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT student_id
            FROM students
            WHERE student_id = ?
            """,
            (student_id,),
        )

        if not cursor.fetchone():
            return jsonify(
                {
                    "success": False,
                    "error": "Student not found",
                }
            ), 404

        cursor.execute(
            """
            SELECT id
            FROM classes
            WHERE id = ?
            """,
            (class_id,),
        )

        if not cursor.fetchone():
            return jsonify(
                {
                    "success": False,
                    "error": "Class not found",
                }
            ), 404

        cursor.execute(
            """
            INSERT INTO excuses
                (
                    student_id,
                    class_id,
                    excuse_date,
                    reason,
                    status
                )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                student_id,
                class_id,
                excuse_date,
                reason,
                "Pending",
            ),
        )

        conn.commit()

        return jsonify(
            {
                "success": True,
            }
        ), 201

    finally:
        conn.close()


@app.route("/excuses", methods=["GET"])
def get_excuses():
    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                excuses.id,
                excuses.student_id,
                students.student_name,
                excuses.class_id,
                classes.course_code,
                classes.course_name,
                excuses.excuse_date,
                excuses.reason,
                excuses.status,
                excuses.submitted_at
            FROM excuses
            JOIN students
                ON excuses.student_id =
                   students.student_id
            JOIN classes
                ON excuses.class_id =
                   classes.id
            ORDER BY excuses.submitted_at DESC
            """
        )

        rows = cursor.fetchall()

        excuses = []

        for row in rows:
            excuses.append(
                {
                    "id": row[0],
                    "student_id": row[1],
                    "student_name": row[2],
                    "class_id": row[3],
                    "course_code": row[4],
                    "course_name": row[5],
                    "excuse_date": row[6],
                    "reason": row[7],
                    "status": row[8],
                    "submitted_at": row[9],
                }
            )

        return jsonify(
            {
                "success": True,
                "excuses": excuses,
            }
        ), 200

    finally:
        conn.close()


@app.route(
    "/excuses/<int:excuse_id>",
    methods=["PUT"],
)
def update_excuse_status(excuse_id):
    data = request.get_json(silent=True) or {}
    status = data.get("status")

    if status not in [
        "Approved",
        "Rejected",
        "Pending",
    ]:
        return jsonify(
            {
                "success": False,
                "error": "Invalid status",
            }
        ), 400

    conn = create_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE excuses
            SET status = ?
            WHERE id = ?
            """,
            (
                status,
                excuse_id,
            ),
        )

        if cursor.rowcount == 0:
            return jsonify(
                {
                    "success": False,
                    "error": "Excuse not found",
                }
            ), 404

        if status == "Approved":
            cursor.execute(
                """
                SELECT
                    student_id,
                    class_id,
                    excuse_date
                FROM excuses
                WHERE id = ?
                """,
                (excuse_id,),
            )

            excuse = cursor.fetchone()

            if excuse:
                student_id = excuse[0]
                class_id = excuse[1]
                excuse_date = excuse[2]

                cursor.execute(
                    """
                    SELECT student_name
                    FROM students
                    WHERE student_id = ?
                    """,
                    (student_id,),
                )

                student = cursor.fetchone()

                student_name = (
                    student[0]
                    if student
                    else "Unknown"
                )

                cursor.execute(
                    """
                    SELECT id
                    FROM attendance
                    WHERE student_id = ?
                      AND class_id = ?
                      AND DATE(timestamp) = ?
                    LIMIT 1
                    """,
                    (
                        student_id,
                        class_id,
                        excuse_date,
                    ),
                )

                attendance_record = (
                    cursor.fetchone()
                )

                if attendance_record:
                    cursor.execute(
                        """
                        UPDATE attendance
                        SET status = 'Excused'
                        WHERE id = ?
                        """,
                        (
                            attendance_record[0],
                        ),
                    )

                else:
                    excuse_timestamp = (
                        f"{excuse_date} "
                        "00:00:00"
                    )

                    cursor.execute(
                        """
                        INSERT INTO attendance
                            (
                                class_id,
                                student_id,
                                student_name,
                                timestamp,
                                status
                            )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            class_id,
                            student_id,
                            student_name,
                            excuse_timestamp,
                            "Excused",
                        ),
                    )

        conn.commit()

        return jsonify(
            {
                "success": True,
            }
        ), 200

    finally:
        conn.close()
        
@app.route("/")
def serve_home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_frontend(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# ---------------------------------------------------------
# Local development only
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get("PORT", 5000)
        ),
        debug=True,
    )