import io

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import os
import pickle

SCOPES = ["https://www.googleapis.com/auth/drive"]

# Authenticate
creds = None

if os.path.exists("token.pickle"):
    with open("token.pickle", "rb") as f:
        creds = pickle.load(f)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json",
            SCOPES
        )
        creds = flow.run_local_server(port=0)

    with open("token.pickle", "wb") as f:
        pickle.dump(creds, f)

service = build("drive", "v3", credentials=creds)

folder_id = "1rYLC7Dqrb5CpzmLpmdvTkjCiiHAk2Uff"

results = service.files().list(
    q=f"name='schedule.pdf' and '{folder_id}' in parents and trashed=false",
    fields="files(id, name)"
).execute()

files = results.get("files", [])

if not files:
    print("File not found")
else:
    file = files[0]
    print(file["name"], file["id"])
    fileId = file["id"]
request = service.files().get_media(fileId=fileId)

with io.FileIO("schedule.pdf", "wb") as fh:
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"{status.progress() * 100:.0f}%")

print("Downloaded schedule.pdf")

# file = service.files().get(
#     fileId=results["files"][0]["id"],
#     fields="parents"
# ).execute()

processed_folder_id = "1CDgW498BhuhpJ1uX1F2an7sGsRtxllB1"

previous_parents = ",".join(file.get("parents", []))

service.files().update(
    fileId=fileId,
    addParents=processed_folder_id,
    removeParents=previous_parents,
    fields="id, parents"
).execute()

print("File moved.")