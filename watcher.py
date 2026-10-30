#!/usr/bin/env python3
"""
My Armonea newsfeed PDF watcher.

First run:
  - Opens a Chromium window.
  - Log in to My Armonea normally (including any 2FA).
  - The browser profile is retained under ~/.local/share/armonea-watcher.

Later runs can be headless:
  ARMONEA_HEADLESS=1 python3 armonea_watcher.py

The first run records the PDFs already present as a baseline; it does not
notify about old posts. New PDF attachments are downloaded once and passed
to process_pdf().
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from drive import GoogleDriveClient
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

NEWSFEED_URL = "https://myarmonea.com/"
POLL_SECONDS = int(os.getenv("ARMONEA_POLL_SECONDS", "120"))
HEADLESS = os.getenv("ARMONEA_HEADLESS", "0").lower() in {"1", "true", "yes"}

DATA_DIR = Path.home() / ".local" / "share" / "armonea-watcher"
PROFILE_DIR = DATA_DIR / "browser-profile"
DOWNLOAD_DIR = Path(os.getenv("ARMONEA_DOWNLOAD_DIR", str(DATA_DIR / "pdf")))
STATE_FILE = DATA_DIR / "seen_attachments.json"
ACTIVITIES_INPUT_FOLDER = "1rYLC7Dqrb5CpzmLpmdvTkjCiiHAk2Uff"

LOG = logging.getLogger("armonea-watcher")
DRIVE_CLIENT = None


def attachment_key(url: str) -> str:
    """Stable identity: the media-attachment ID, not the expiring S3 URL."""
    match = re.search(r"/media-attachment/(\d+)/", url)
    if match:
        return match.group(1)
    return urlparse(url).path


def filename_from_url(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    return name if name.lower().endswith(".pdf") else f"{name or 'attachment'}.pdf"


def load_seen() -> set[str]:
    try:
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return set()


def save_seen(seen: set[str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(sorted(seen), indent=2), encoding="utf-8")
    temporary.replace(STATE_FILE)


def process_pdf(pdf_path: Path) -> None:
    """Upload activity PDFs to the Drive folder polled by activities.py."""
    if "activi" not in pdf_path.name.casefold():
        LOG.info("Skipping Drive upload for non-activity PDF: %s", pdf_path.name)
        return

    global DRIVE_CLIENT
    if DRIVE_CLIENT is None:
        DRIVE_CLIENT = GoogleDriveClient()

    uploaded = DRIVE_CLIENT.push_file(
        ACTIVITIES_INPUT_FOLDER,
        str(pdf_path),
    )
    LOG.info("Uploaded activity PDF to Drive: %s", uploaded["name"])


def notify(pdf_path: Path) -> None:
    """Local Linux desktop notification, when notify-send is available."""
    message = f"New My Armonea PDF: {pdf_path.name}"
    if shutil.which("notify-send"):
        subprocess.run(
            ["notify-send", "--app-name=Armonea watcher",
             "New activity/document posted", message],
            check=False,
        )
    LOG.info("%s", message)


def collect_pdf_links(page) -> list[dict[str, str]]:
    # The saved page shows attachments as <a class="post-file" href="...pdf">.
    # Use the DOM after Vue/Nuxt has rendered it, rather than parsing raw JS.
    links = page.locator("a.post-file").evaluate_all(
        """els => els.map(a => ({
            href: a.href,
            label: (a.innerText || a.textContent || '').trim()
        }))"""
    )

    result = []
    for item in links:
        url = item["href"]
        if urlparse(url).path.lower().endswith(".pdf"):
            result.append({
                "key": attachment_key(url),
                "url": url,
                "filename": filename_from_url(url),
            })
    return result


def wait_for_authenticated_feed(page) -> None:
    page.goto(NEWSFEED_URL, wait_until="domcontentloaded", timeout=60000)
    LOG.info("Waiting for the authenticated newsfeed.")
    LOG.info("If a login page appears, log in manually in the Chromium window.")

    # At least one PDF currently appears in the supplied newsfeed snapshot.
    # Waiting for the attachment also avoids treating a login page as an empty feed.
    try:
        page.locator("a.post-file").first.wait_for(state="attached", timeout=600000)
    except PlaywrightTimeoutError as exc:
        raise RuntimeError(
            "No PDF attachment appeared. Check that login completed and that "
            "the selected resident's newsfeed contains PDF posts."
        ) from exc


def poll_feed_once(page, context, seen, feed_url=NEWSFEED_URL) -> None:
    page.goto(feed_url, wait_until="domcontentloaded", timeout=60000)
    page.locator("a.post-file").first.wait_for(
        state="attached", timeout=60000
    )
    links = collect_pdf_links(page)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    new_links = [item for item in links if item["key"] not in seen]
    for item in reversed(new_links):
        destination = DOWNLOAD_DIR / item["filename"]
        temporary = destination.with_suffix(destination.suffix + ".part")

        response = context.request.get(item["url"], timeout=60000)
        if not response.ok:
            raise RuntimeError(f"PDF download failed: HTTP {response.status}")

        content = response.body()
        if not content.startswith(b"%PDF-"):
            raise RuntimeError(
                f"Downloaded file is not a PDF: {item['filename']}"
            )

        temporary.write_bytes(content)
        temporary.replace(destination)

        process_pdf(destination)
        notify(destination)
        seen.add(item["key"])
        save_seen(seen)

        LOG.info("Downloaded attachment %s (%d bytes)", destination, len(content))


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=HEADLESS,
            accept_downloads=False,
            viewport={"width": 1440, "height": 1000},
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            wait_for_authenticated_feed(page)

            seen = load_seen()
            initial_links = collect_pdf_links(page)

            if not seen:
                seen = {item["key"] for item in initial_links}
                save_seen(seen)
                LOG.info(
                    "Initial baseline stored (%d existing PDFs); "
                    "only PDFs posted after this point will trigger processing.",
                    len(seen),
                )

            while True:
                try:
                    poll_feed_once(page, context, seen)
                except Exception:
                    LOG.exception("Newsfeed check failed; will retry.")

                time.sleep(POLL_SECONDS)

        finally:
            context.close()


if __name__ == "__main__":
    run()
