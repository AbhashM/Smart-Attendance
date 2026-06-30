import sqlite3

conn = sqlite3.connect("database/attendance.db")
cursor = conn.cursor()

# Students table
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name TEXT NOT NULL,
    student_id TEXT UNIQUE NOT NULL,
    image_path TEXT NOT NULL
)
""")

# Multiple student images / alternate appearances
cursor.execute("""
CREATE TABLE IF NOT EXISTS student_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    appearance_label TEXT,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
)
""")

# Classes table
cursor.execute("""
CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_name TEXT NOT NULL,
    course_code TEXT NOT NULL,
    professor_name TEXT NOT NULL
)
""")

# Students assigned to classes
cursor.execute("""
CREATE TABLE IF NOT EXISTS class_students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL,
    student_id TEXT NOT NULL,
    FOREIGN KEY (class_id) REFERENCES classes(id),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
)
""")

# Attendance table
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    student_name TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database initialized successfully.")