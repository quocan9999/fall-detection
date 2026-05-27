from __future__ import annotations

from queue import Queue
from threading import Event, Thread
from typing import Any

import cv2

from .detector import PoseFallDetector


class CameraWorker:
    def __init__(self, detector: PoseFallDetector, events: Queue[tuple[str, Any]]) -> None:
        self.detector = detector
        self.events = events
        self._stop_event = Event()
        self._thread: Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, index: int, imgsz: int, conf: float, device: str) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = Thread(
            target=self._run,
            args=(index, imgsz, conf, device),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self, index: int, imgsz: int, conf: float, device: str) -> None:
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            self.events.put(("error", "Không thể mở camera đã chọn."))
            self.events.put(("camera_stopped", None))
            return

        self.events.put(("status", "Camera đang chạy."))
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    self.events.put(("error", "Mất kết nối hoặc không đọc được camera."))
                    break
                result = self.detector.predict(
                    frame, imgsz=imgsz, conf=conf, device=device
                )
                self.events.put(("camera_frame", result))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            capture.release()
            self.events.put(("camera_stopped", None))
