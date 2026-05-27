from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Callable

import cv2
import numpy as np

from .detector import DetectionResult, PoseFallDetector


@dataclass(frozen=True)
class ProcessedMedia:
    output_path: Path
    detection_count: int
    fall_found: bool


def create_output_path(directory: Path, source: Path, extension: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = extension or source.suffix.lower()
    return directory / f"{source.stem}_annotated_{stamp}{suffix}"


def read_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def write_image(path: Path, frame: np.ndarray) -> bool:
    extension = path.suffix if path.suffix else ".jpg"
    success, buffer = cv2.imencode(extension, frame)
    if not success:
        return False
    try:
        buffer.tofile(path)
        return True
    except OSError:
        return False


def process_image(
    detector: PoseFallDetector,
    source: Path,
    output_directory: Path,
    imgsz: int,
    conf: float,
    device: str,
) -> tuple[ProcessedMedia, DetectionResult]:
    frame = read_image(source)
    if frame is None:
        raise ValueError("Không thể đọc file ảnh đã chọn.")
    result = detector.predict(frame, imgsz=imgsz, conf=conf, device=device)
    output_path = create_output_path(output_directory, source)
    if not write_image(output_path, result.annotated_frame):
        raise OSError("Không thể lưu ảnh kết quả.")
    return (
        ProcessedMedia(output_path, result.detection_count, result.has_fall),
        result,
    )


def process_video(
    detector: PoseFallDetector,
    source: Path,
    output_directory: Path,
    imgsz: int,
    conf: float,
    device: str,
    cancelled: Event,
    on_preview: Callable[[DetectionResult, int, int], None] | None = None,
) -> ProcessedMedia:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Không thể mở file video đã chọn.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_path = create_output_path(output_directory, source, ".mp4")
    writer = None
    detection_count = 0
    fall_found = False
    frame_index = 0

    try:
        while not cancelled.is_set():
            ok, frame = capture.read()
            if not ok:
                break
            result = detector.predict(frame, imgsz=imgsz, conf=conf, device=device)
            if writer is None:
                height, width = result.annotated_frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise OSError("Không thể tạo file video kết quả.")
            writer.write(result.annotated_frame)
            detection_count += result.detection_count
            fall_found = fall_found or result.has_fall
            frame_index += 1
            if on_preview and (frame_index == 1 or frame_index % 5 == 0):
                on_preview(result, frame_index, total_frames)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if cancelled.is_set():
        if output_path.exists():
            output_path.unlink()
        raise InterruptedError("Đã hủy xử lý video.")
    if writer is None:
        raise ValueError("Video không có frame đọc được.")
    return ProcessedMedia(output_path, detection_count, fall_found)


def find_camera_indices(limit: int = 10) -> list[int]:
    available: list[int] = []
    for index in range(limit):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                available.append(index)
        capture.release()
    return available
