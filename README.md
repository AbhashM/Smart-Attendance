# Smart-Attendance

Smart Attendance is a full-stack AI-based attendance system that uses face detection and face recognition concepts to automate student attendance.

## Current Status
- Sprint 1: Project setup, camera test, and face detection completed
- Sprint 2: Frontend connected to backend for face detection
- Sprint 3: Student registration system completed with image upload and database storage

## Tech Stack
- Python
- OpenCV
- MediaPipe
- Flask
- Flask-CORS
- SQLite
- HTML, CSS, JavaScript

## Project Structure
- backend/   - backend API and AI logic
- frontend/  - frontend pages
- database/  - database scripts and SQLite database setup
- models/    - face recognition models / embeddings
- docs/      - sprint plans, diagrams, reports
- tests/     - testing scripts

## Features Completed
- Webcam test with OpenCV
- Face detection using MediaPipe
- Flask backend API
- Frontend image upload for face detection
- Student registration form
- Upload and save student images
- Store student records in SQLite database

## Setup Instructions

1. Install Python 3.10

2. Create a virtual environment:
   py -3.10 -m venv venv

3. Activate the virtual environment:
   venv\Scripts\activate

4. Install dependencies:
   pip install -r requirements.txt

5. Run the database setup:
   python database/db.py

6. Start the backend server:
   python backend/app.py

7. Open the frontend pages with Live Server:
   - frontend/index.html
   - frontend/register.html

## Notes
- Uploaded images are stored locally in `backend/uploads/`
- The SQLite database file is ignored in GitHub
- The uploads folder is tracked using `.gitkeep`