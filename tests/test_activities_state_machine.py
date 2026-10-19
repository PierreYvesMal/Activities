from datetime import datetime

import numpy as np

import activities


def test_classify_time_uses_day_evening_and_night_windows():
    assert activities.classify_time(datetime(2026, 10, 26, 7)) is activities.State.ACTIVITIES
    assert activities.classify_time(datetime(2026, 10, 26, 19, 59)) is activities.State.ACTIVITIES
    assert activities.classify_time(datetime(2026, 10, 26, 20)) is activities.State.DISPLAY_ERROR
    assert activities.classify_time(datetime(2026, 10, 26, 21, 59)) is activities.State.DISPLAY_ERROR
    assert activities.classify_time(datetime(2026, 10, 26, 22)) is activities.State.DISPLAY_BLACK
    assert activities.classify_time(datetime(2026, 10, 26, 6, 59)) is activities.State.DISPLAY_BLACK
    assert activities.classify_time(datetime(2026, 10, 31, 12)) is activities.State.DISPLAY_BLACK


def test_no_new_file_keeps_rejected_schedule_on_error():
    assert activities.state_without_new_file(
        False, has_processed_image=True, has_local_pdf=False
    ) is activities.State.DISPLAY_ERROR


def test_no_new_file_displays_valid_or_unchecked_cache():
    assert activities.state_without_new_file(
        True, has_processed_image=True, has_local_pdf=False
    ) is activities.State.DISPLAY_PROCESSED
    assert activities.state_without_new_file(
        None, has_processed_image=True, has_local_pdf=False
    ) is activities.State.DISPLAY_PROCESSED


def test_no_new_file_processes_local_pdf_or_errors_without_input():
    assert activities.state_without_new_file(
        None, has_processed_image=False, has_local_pdf=True
    ) is activities.State.PROCESS_IMAGE
    assert activities.state_without_new_file(
        None, has_processed_image=False, has_local_pdf=False
    ) is activities.State.DISPLAY_ERROR


def test_load_drive_photo_downloads_and_decodes_newest_photo(monkeypatch):
    class FakeDriveClient:
        def find_image_file(self, folder_id):
            assert folder_id == "photos-folder-id"
            return {"id": "photo-id", "name": "photo.png"}

        def download_file(self, file_id, destination):
            assert file_id == "photo-id"
            assert destination == activities.PHOTO_IMAGE_PATH

    expected_image = np.zeros((20, 30, 3), dtype=np.uint8)
    monkeypatch.setattr(activities, "PHOTO_FOLDER_ID", "photos-folder-id")
    monkeypatch.setattr(
        activities.cv2,
        "imread",
        lambda path, mode: expected_image
        if path == activities.PHOTO_IMAGE_PATH and mode == activities.cv2.IMREAD_COLOR
        else None,
    )

    assert activities.load_drive_photo(FakeDriveClient()) is expected_image


def test_load_drive_photo_is_disabled_without_folder_id(monkeypatch):
    monkeypatch.setattr(activities, "PHOTO_FOLDER_ID", "")

    assert activities.load_drive_photo(object()) is None


def test_schedule_validation_requires_matching_day_number(monkeypatch):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    vertical_peaks = [0, 50]
    horizontal_peaks = [0, 10, 20, 30, 40, 50, 60]
    today = datetime(2026, 10, 30)

    monkeypatch.setattr(activities, "ocr", lambda crop: "Vendredi 30")
    assert activities.schedule_is_up_to_date(
        image, vertical_peaks, horizontal_peaks, today
    )

    monkeypatch.setattr(activities, "ocr", lambda crop: "Vendredi 23")
    assert not activities.schedule_is_up_to_date(
        image, vertical_peaks, horizontal_peaks, today
    )