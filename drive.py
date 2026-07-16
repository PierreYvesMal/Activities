import io
import os
import pickle
import fnmatch

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload

class GoogleDriveClient:
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    def __init__(
        self,
        credentials_file="credentials.json",
        token_file="token.pickle",
    ):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = self._authenticate()

    def _authenticate(self):
        creds = None

        if os.path.exists(self.token_file):
            with open(self.token_file, "rb") as f:
                creds = pickle.load(f)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file,
                    self.SCOPES,
                )
                creds = flow.run_local_server(port=0)

            with open(self.token_file, "wb") as f:
                pickle.dump(creds, f)

        return build("drive", "v3", credentials=creds)

    def find_file(self, folder_id, filename):
        results = self.service.files().list(
            q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
            fields="files(id,name,parents)",
        ).execute()

        files = results.get("files", [])

        if not files:
            return None

        return files[0]

    def find_file_pattern(self, folder_id, pattern):
        results = self.service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id,name,parents)",
        ).execute()

        for file in results.get("files", []):
            if fnmatch.fnmatch(file["name"], pattern):
                return file

        return None

    def download_file(self, file_id, destination):
        request = self.service.files().get_media(fileId=file_id)

        with io.FileIO(destination, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"{status.progress() * 100:.0f}%")

    def move_file(self, file_id, destination_folder_id):
        file = self.service.files().get(
            fileId=file_id,
            fields="parents",
        ).execute()

        previous_parents = ",".join(file.get("parents", []))

        self.service.files().update(
            fileId=file_id,
            addParents=destination_folder_id,
            removeParents=previous_parents,
            fields="id,parents",
        ).execute()

    def download_file_by_name(self, folder_id, filename, destination=None):
        file = self.find_file(folder_id, filename)

        if file is None:
            raise FileNotFoundError(filename)

        if destination is None:
            destination = filename

        self.download_file(file["id"], destination)

        return file

    def move_file_by_name(self, source_folder_id, filename, destination_folder_id):
        file = self.find_file(source_folder_id, filename)

        if file is None:
            raise FileNotFoundError(filename)

        self.move_file(file["id"], destination_folder_id)

        return file
    
    def exists(self, folder_id, filename):
        return self.find_file(folder_id, filename) is not None
    
    def push_file(self, folder_id, path):
        filename = os.path.basename(path)

        metadata = {
            "name": filename,
            "parents": [folder_id],
        }

        media = MediaFileUpload(
            path,
            resumable=True,
        )

        file = self.service.files().create(
            body=metadata,
            media_body=media,
            fields="id,name",
        ).execute()

        return file