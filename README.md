# Requirements
## Python virtual environment
```
python3 -m venv venv
```
```
source venv/bin/activate
```
## Python requirements
```
python3 -m pip install flask opencv-python numpy pdf2image
```

# Usage
## Prepare
```
cp <filepath/filename.pdf> ./schedule.pdf
```
## Run
```
python3 ./activities.py
```
on other terminal
```
chromium-browser --kiosk http://localhost:5000
```
# On tablet (options to explore)
## Open near full screen
```
termux-open-url http://127.0.0.1:5000
```
or

Install Kiwi Browser or use Chrome:
Then manually once:
open page
tap “Add to Home screen”
open from home screen → standalone mode

## autostart on boot 
pkg install termux-services