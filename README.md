# Smart-Attendance

Smart Attendance is a full-stack AI-based attendance system that uses face recognition to mark attendance automatically.

## Tech Stack
-opencv-python
-mediapipe==0.10.14
-flask

## Project Structure
backend/   - backend API + AI logic  
frontend/  - UI pages  
database/  - database scripts  
models/    - face recognition models/embeddings  
docs/      - sprint plans, diagrams, reports  
tests/     - testing scripts  

## Setup Instructions

1. Install Python 3.10
2. Create virtual environment:
   py -3.10 -m venv venv
3. Activate:
   venv\Scripts\activate
4. Install dependencies:
   pip install -r requirements.txt
5. Run face detection:
   python backend/face_detection.py

