cp <filepath/filename.pdf> ./schedule.pdf

python3 -m venv venv

source venv/bin/activate

python3 -m pip install opencv-python numpy pdf2image

python3 ./activities.py