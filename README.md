cp <filepath/filename.pdf> ./schedule.pdf

python3 -m venv venv

source venv/bin/activate

python3 -m pip install flask opencv-python numpy pdf2image

python3 ./activities.py

on other terminal
chromium-browser --kiosk http://localhost:5000