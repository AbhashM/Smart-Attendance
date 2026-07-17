#!/bin/bash
set -e

echo "Installing OpenCV Linux libraries..."

apt-get update
apt-get install -y libxcb1 libgl1 libglib2.0-0

echo "Starting Smart Attendance..."

exec gunicorn \
  --bind=0.0.0.0:8000 \
  --timeout 1200 \
  --workers 1 \
  --chdir backend \
  app:app