# database/view_attendance.py

import sqlite3

conn = sqlite3.connect("database/attendance.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM attendance")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()