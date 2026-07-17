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


@app.route("/students", methods=["GET"])
def get_students():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT student_id, student_name
        FROM students
        ORDER BY student_name ASC
    """)

    rows = cursor.fetchall()
    conn.close()

    students = []

    for row in rows:
        students.append({
            "student_id": row[0],
            "student_name": row[1]
        })

    return jsonify({"success": True, "students": students})


@app.route("/classes", methods=["POST"])
def create_class():
    course_name = request.form.get("course_name")
    course_code = request.form.get("course_code")
    professor_name = request.form.get("professor_name")

    if not course_name or not course_code or not professor_name:
        return jsonify({"success": False, "error": "Missing required fields"}), 400

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO classes (course_name, course_code, professor_name)
        VALUES (?, ?, ?)
    """, (course_name, course_code, professor_name))

    conn.commit()
    conn.close()

    return jsonify({"success": True})


@app.route("/classes", methods=["GET"])
def get_classes():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, course_name, course_code, professor_name
        FROM classes
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    classes = []

    for row in rows:
        classes.append({
            "id": row[0],
            "course_name": row[1],
            "course_code": row[2],
            "professor_name": row[3]
        })

    return jsonify({"success": True, "classes": classes})


@app.route("/class-students", methods=["POST"])
def add_student_to_class():
    class_id = request.form.get("class_id")
    student_id = request.form.get("student_id")

    if not class_id or not student_id:
        return jsonify({"success": False, "error": "Missing class ID or student ID"}), 400

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM classes WHERE id = ?", (class_id,))
    class_record = cursor.fetchone()

    if not class_record:
        conn.close()
        return jsonify({"success": False, "error": "Class not found"}), 404

    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    student_record = cursor.fetchone()

    if not student_record:
        conn.close()
        return jsonify({"success": False, "error": "Student not found"}), 404

    cursor.execute("""
        INSERT INTO class_students (class_id, student_id)
        VALUES (?, ?)
    """, (class_id, student_id))

    conn.commit()
    conn.close()

    return jsonify({"success": True})


@app.route("/class-students/<int:class_id>", methods=["GET"])
def get_class_students(class_id):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT students.student_id, students.student_name
        FROM class_students
        JOIN students
        ON class_students.student_id = students.student_id
        WHERE class_students.class_id = ?
    """, (class_id,))

    rows = cursor.fetchall()
    conn.close()

    students = []

    for row in rows:
        students.append({
            "student_id": row[0],
            "student_name": row[1]
        })

    return jsonify({"success": True, "students": students})


@app.route("/class-students/<int:class_id>/<student_id>", methods=["DELETE"])
def remove_student_from_class(class_id, student_id):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM class_students
        WHERE class_id = ? AND student_id = ?
    """, (class_id, student_id))

    conn.commit()
    conn.close()

    return jsonify({"success": True})


@app.route("/recognize", methods=["POST"])
def recognize_student():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    class_id = request.form.get("class_id")

    if not class_id:
        return jsonify({"success": False, "error": "Please select a class"}), 400

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
        JOIN class_students
        ON students.student_id = class_students.student_id
        WHERE class_students.class_id = ?
    """, (class_id,))

    students = cursor.fetchall()
    conn.close()

    if not students:
        os.remove(test_image_path)
        return jsonify({
            "success": False,
            "error": "No student appearance profiles found for this class"
        }), 400

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
                    INSERT INTO attendance (class_id, student_id, student_name, timestamp, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (class_id, student_id, student_name, current_time, "Present"))

                attendance_conn.commit()
                attendance_conn.close()

                os.remove(test_image_path)

                return jsonify({
                    "success": True,
                    "student_name": student_name,
                    "student_id": student_id,
                    "class_id": class_id,
                    "attendance_marked": True,
                    "status": "Present",
                    "timestamp": current_time
                })

        except Exception as e:
            print("Error comparing face:", e)

    os.remove(test_image_path)

    return jsonify({
        "success": False,
        "error": "No matching student found in selected class"
    })


@app.route("/attendance", methods=["GET"])
def get_attendance():
    selected_date = request.args.get("date")
    class_id = request.args.get("class_id")

    if not selected_date or not class_id:
        return jsonify({
            "success": True,
            "attendance": [],
            "message": "Please select a class and date"
        })

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT attendance.id,
               attendance.student_id,
               attendance.student_name,
               attendance.timestamp,
               attendance.status,
               classes.course_code,
               classes.course_name
        FROM attendance
        LEFT JOIN classes
        ON attendance.class_id = classes.id
        WHERE DATE(attendance.timestamp) = ?
        AND attendance.class_id = ?
        ORDER BY attendance.timestamp DESC
    """, (selected_date, class_id))

    rows = cursor.fetchall()
    conn.close()

    attendance_records = []

    for row in rows:
        attendance_records.append({
            "id": row[0],
            "student_id": row[1],
            "student_name": row[2],
            "timestamp": row[3],
            "status": row[4],
            "course_code": row[5],
            "course_name": row[6]
        })

    return jsonify({"success": True, "attendance": attendance_records})


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


@app.route("/attendance-policies", methods=["POST"])
def create_attendance_policy():
    class_id = request.form.get("class_id")
    policy_name = request.form.get("policy_name")
    absence_limit = request.form.get("absence_limit")
    late_limit = request.form.get("late_limit")
    late_minutes = request.form.get("late_minutes")
    attendance_weight = request.form.get("attendance_weight")
    consequence = request.form.get("consequence")
    excuse_counts = request.form.get("excuse_counts") or "No"

    if not class_id or not policy_name:
        return jsonify({"success": False, "error": "Class and policy name are required"}), 400

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO attendance_policies
        (class_id, policy_name, absence_limit, late_limit, late_minutes,
         attendance_weight, consequence, excuse_counts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        class_id,
        policy_name,
        absence_limit,
        late_limit,
        late_minutes,
        attendance_weight,
        consequence,
        excuse_counts
    ))

    conn.commit()
    conn.close()

    return jsonify({"success": True})


@app.route("/attendance-policies/<int:class_id>", methods=["GET"])
def get_attendance_policies(class_id):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, class_id, policy_name, absence_limit, late_limit,
               late_minutes, attendance_weight, consequence, excuse_counts
        FROM attendance_policies
        WHERE class_id = ?
        ORDER BY id DESC
    """, (class_id,))

    rows = cursor.fetchall()
    conn.close()

    policies = []

    for row in rows:
        policies.append({
            "id": row[0],
            "class_id": row[1],
            "policy_name": row[2],
            "absence_limit": row[3],
            "late_limit": row[4],
            "late_minutes": row[5],
            "attendance_weight": row[6],
            "consequence": row[7],
            "excuse_counts": row[8]
        })

    return jsonify({"success": True, "policies": policies})


@app.route("/attendance-policies/<int:policy_id>", methods=["PUT"])
def update_attendance_policy(policy_id):
    data = request.get_json()

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE attendance_policies
        SET policy_name = ?,
            absence_limit = ?,
            late_limit = ?,
            late_minutes = ?,
            attendance_weight = ?,
            consequence = ?,
            excuse_counts = ?
        WHERE id = ?
    """, (
        data.get("policy_name"),
        data.get("absence_limit"),
        data.get("late_limit"),
        data.get("late_minutes"),
        data.get("attendance_weight"),
        data.get("consequence"),
        data.get("excuse_counts"),
        policy_id
    ))

    conn.commit()
    conn.close()

    return jsonify({"success": True})


@app.route("/attendance-policies/<int:policy_id>", methods=["DELETE"])
def delete_attendance_policy(policy_id):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM attendance_policies WHERE id = ?", (policy_id,))

    conn.commit()
    conn.close()

    return jsonify({"success": True})


@app.route("/policy-risk/<int:class_id>", methods=["GET"])
def policy_risk(class_id):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, policy_name, absence_limit, consequence
        FROM attendance_policies
        WHERE class_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (class_id,))

    policy = cursor.fetchone()

    if not policy:
        conn.close()
        return jsonify({
            "success": True,
            "has_policy": False,
            "message": "No attendance policy set for this class",
            "at_risk": []
        })

    policy_id, policy_name, absence_limit, consequence = policy

    cursor.execute("""
        SELECT students.student_id, students.student_name
        FROM class_students
        JOIN students
        ON class_students.student_id = students.student_id
        WHERE class_students.class_id = ?
    """, (class_id,))

    enrolled_students = cursor.fetchall()

    cursor.execute("""
        SELECT COUNT(DISTINCT DATE(timestamp))
        FROM attendance
        WHERE class_id = ?
    """, (class_id,))

    total_class_days = cursor.fetchone()[0]

    at_risk = []

    for student in enrolled_students:
        student_id = student[0]
        student_name = student[1]

        cursor.execute("""
            SELECT COUNT(DISTINCT DATE(timestamp))
            FROM attendance
            WHERE class_id = ?
            AND student_id = ?
            AND status = 'Present'
        """, (class_id, student_id))

        present_days = cursor.fetchone()[0]
        absences = max(total_class_days - present_days, 0)

        if absence_limit is not None and absences >= absence_limit:
            at_risk.append({
                "student_id": student_id,
                "student_name": student_name,
                "absences": absences,
                "absence_limit": absence_limit
            })

    conn.close()

    return jsonify({
        "success": True,
        "has_policy": True,
        "policy_name": policy_name,
        "absence_limit": absence_limit,
        "consequence": consequence,
        "total_class_days": total_class_days,
        "at_risk": at_risk
    })


@app.route("/attendance-trend/<int:class_id>", methods=["GET"])
def attendance_trend(class_id):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM class_students
        WHERE class_id = ?
    """, (class_id,))
    total_students = cursor.fetchone()[0]

    cursor.execute("""
        SELECT DATE(timestamp) as attendance_date,
               COUNT(DISTINCT student_id) as present_count
        FROM attendance
        WHERE class_id = ?
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp) DESC
        LIMIT 5
    """, (class_id,))

    rows = cursor.fetchall()
    conn.close()

    trend = []

    for row in reversed(rows):
        attendance_date = row[0]
        present_count = row[1]

        rate = 0
        if total_students > 0:
            rate = round((present_count / total_students) * 100)

        trend.append({
            "date": attendance_date,
            "present_count": present_count,
            "total_students": total_students,
            "attendance_rate": rate
        })

    return jsonify({"success": True, "trend": trend})


@app.route("/export-attendance", methods=["GET"])
def export_attendance():
    class_id = request.args.get("class_id")
    selected_date = request.args.get("date")

    if not class_id or not selected_date:
        return jsonify({"success": False, "error": "Class and date are required"}), 400

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT classes.course_code,
               classes.course_name,
               attendance.student_id,
               attendance.student_name,
               attendance.timestamp,
               attendance.status
        FROM attendance
        JOIN classes
        ON attendance.class_id = classes.id
        WHERE attendance.class_id = ?
        AND DATE(attendance.timestamp) = ?
        ORDER BY attendance.timestamp ASC
    """, (class_id, selected_date))

    rows = cursor.fetchall()
    conn.close()

    csv_data = "Course Code,Course Name,Student ID,Student Name,Timestamp,Status\n"

    for row in rows:
        csv_data += f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n"

    response = app.response_class(
        response=csv_data,
        status=200,
        mimetype="text/csv"
    )

    response.headers["Content-Disposition"] = (
        f"attachment; filename=attendance_{class_id}_{selected_date}.csv"
    )

    return response


@app.route("/excuses", methods=["POST"])
def submit_excuse():
    student_id = request.form.get("student_id")
    class_id = request.form.get("class_id")
    excuse_date = request.form.get("excuse_date")
    reason = request.form.get("reason")

    if not student_id or not class_id or not excuse_date or not reason:
        return jsonify({"success": False, "error": "Missing required fields"}), 400

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    student = cursor.fetchone()

    if not student:
        conn.close()
        return jsonify({"success": False, "error": "Student not found"}), 404

    cursor.execute("SELECT * FROM classes WHERE id = ?", (class_id,))
    class_record = cursor.fetchone()

    if not class_record:
        conn.close()
        return jsonify({"success": False, "error": "Class not found"}), 404

    cursor.execute("""
        INSERT INTO excuses (student_id, class_id, excuse_date, reason, status)
        VALUES (?, ?, ?, ?, ?)
    """, (student_id, class_id, excuse_date, reason, "Pending"))

    conn.commit()
    conn.close()

    return jsonify({"success": True})


@app.route("/excuses", methods=["GET"])
def get_excuses():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT excuses.id,
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
        ON excuses.student_id = students.student_id
        JOIN classes
        ON excuses.class_id = classes.id
        ORDER BY excuses.submitted_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    excuses = []

    for row in rows:
        excuses.append({
            "id": row[0],
            "student_id": row[1],
            "student_name": row[2],
            "class_id": row[3],
            "course_code": row[4],
            "course_name": row[5],
            "excuse_date": row[6],
            "reason": row[7],
            "status": row[8],
            "submitted_at": row[9]
        })

    return jsonify({"success": True, "excuses": excuses})


@app.route("/excuses/<int:excuse_id>", methods=["PUT"])
def update_excuse_status(excuse_id):
    data = request.get_json()
    status = data.get("status")

    if status not in ["Approved", "Rejected", "Pending"]:
        return jsonify({"success": False, "error": "Invalid status"}), 400

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE excuses
        SET status = ?
        WHERE id = ?
    """, (status, excuse_id))

    if status == "Approved":
        cursor.execute("""
            SELECT student_id, class_id, excuse_date
            FROM excuses
            WHERE id = ?
        """, (excuse_id,))

        excuse = cursor.fetchone()

        if excuse:
            student_id, class_id, excuse_date = excuse

            cursor.execute("""
                SELECT student_name
                FROM students
                WHERE student_id = ?
            """, (student_id,))

            student = cursor.fetchone()
            student_name = student[0] if student else "Unknown"

            cursor.execute("""
                SELECT id
                FROM attendance
                WHERE student_id = ?
                AND class_id = ?
                AND DATE(timestamp) = ?
            """, (student_id, class_id, excuse_date))

            attendance_record = cursor.fetchone()

            if attendance_record:
                cursor.execute("""
                    UPDATE attendance
                    SET status = 'Excused'
                    WHERE id = ?
                """, (attendance_record[0],))
            else:
                excuse_timestamp = excuse_date + " 00:00:00"

                cursor.execute("""
                    INSERT INTO attendance
                    (class_id, student_id, student_name, timestamp, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (class_id, student_id, student_name, excuse_timestamp, "Excused"))

    conn.commit()
    conn.close()

    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)