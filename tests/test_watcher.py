from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import watcher
from playwright.sync_api import sync_playwright

class RecordingDriveClient:
    def __init__(self):
        self.uploads = []

    def push_file(self, folder_id, path):
        self.uploads.append((folder_id, path))
        return {"name": Path(path).name}


def test_activity_pdf_uploads_to_activities_folder(tmp_path, monkeypatch):
    pdf_path = tmp_path / "Weekly-ACTIVITE.pdf"
    pdf_path.touch()
    drive_client = RecordingDriveClient()

    monkeypatch.setattr(watcher, "DRIVE_CLIENT", None)
    monkeypatch.setattr(watcher, "GoogleDriveClient", lambda: drive_client)

    watcher.process_pdf(pdf_path)

    assert drive_client.uploads == [
        (watcher.ACTIVITIES_INPUT_FOLDER, str(pdf_path))
    ]
    assert watcher.DRIVE_CLIENT is drive_client


def test_non_activity_pdf_is_skipped_without_creating_drive_client(
    tmp_path, monkeypatch
):
    pdf_path = tmp_path / "newsletter.pdf"

    def unexpected_drive_client():
        raise AssertionError("Drive client should not be created")

    monkeypatch.setattr(watcher, "DRIVE_CLIENT", None)
    monkeypatch.setattr(watcher, "GoogleDriveClient", unexpected_drive_client)

    watcher.process_pdf(pdf_path)

    assert watcher.DRIVE_CLIENT is None


def test_poll_downloads_new_pdf_from_local_newsfeed(tmp_path, monkeypatch):
    existing_pdf = b"%PDF-1.4\nexisting schedule\n%%EOF\n"
    new_pdf = b"%PDF-1.4\nnew activity schedule\n%%EOF\n"
    feed_state = {"paths": ["/pdf/existing.pdf"]}
    pdf_files = {
        "/pdf/existing.pdf": existing_pdf,
        "/pdf/Activiteiten-week-44.pdf": new_pdf,
    }

    class NewsfeedHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                anchors = "".join(
                    f'<a class="post-file" href="{item}">PDF</a>'
                    for item in feed_state["paths"]
                )
                content = f"<html><body>{anchors}</body></html>".encode()
                content_type = "text/html"
                status = 200
            elif path in pdf_files:
                content = pdf_files[path]
                content_type = "application/pdf"
                status = 200
            else:
                content = b"Not found"
                content_type = "text/plain"
                status = 404

            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), NewsfeedHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    downloads_dir = tmp_path / "downloads"
    processed = []
    seen_updates = []
    monkeypatch.setattr(watcher, "DOWNLOAD_DIR", downloads_dir)
    monkeypatch.setattr(watcher, "process_pdf", processed.append)
    monkeypatch.setattr(watcher, "notify", lambda path: None)
    monkeypatch.setattr(watcher, "save_seen", lambda seen: seen_updates.append(set(seen)))

    try:
        feed_url = f"http://127.0.0.1:{server.server_port}/"
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context()
                page = context.new_page()
                page.goto(feed_url, wait_until="domcontentloaded")
                seen = {item["key"] for item in watcher.collect_pdf_links(page)}

                feed_state["paths"] = ["/pdf/Activiteiten-week-44.pdf"]
                watcher.poll_feed_once(page, context, seen, feed_url)

                downloaded = downloads_dir / "Activiteiten-week-44.pdf"
                assert downloaded.read_bytes() == new_pdf
                assert processed == [downloaded]
                assert watcher.attachment_key(
                    f"{feed_url}pdf/Activiteiten-week-44.pdf"
                ) in seen
                assert seen_updates == [seen]
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()