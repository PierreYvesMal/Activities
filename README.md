cp <filepath/filename.pdf> ./schedule.pdf

python3 -m venv venv

source venv/bin/activate

pip install opencv-python numpy pdf2image
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

python3 ./activities.py