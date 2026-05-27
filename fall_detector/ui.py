from __future__ import annotations

import os
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .alarm import FallAlarm
from .camera import CameraWorker
from .detector import DetectionResult, ModelInfo, PoseFallDetector
from .media import find_camera_indices, process_image, process_video


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_MODEL = PROJECT_ROOT / "weights" / "best.pt"
IMAGE_OUTPUTS = PROJECT_ROOT / "outputs" / "images"
VIDEO_OUTPUTS = PROJECT_ROOT / "outputs" / "videos"


class MainWindow(QMainWindow):
    def __init__(self, on_close: Any) -> None:
        super().__init__()
        self._on_close = on_close

    def closeEvent(self, event: QCloseEvent) -> None:
        self._on_close()
        super().closeEvent(event)


class FallDetectionApp:
    def __init__(self) -> None:
        self.qt_app = QApplication.instance() or QApplication([])
        self.window = MainWindow(self._shutdown)
        self.window.setWindowTitle("Fall Detection - YOLO Pose")
        self.window.resize(1280, 820)
        self.window.setMinimumSize(1040, 700)

        self.detector = PoseFallDetector()
        self.alarm = FallAlarm(cooldown_seconds=2.0)
        self.events: Queue[tuple[str, Any]] = Queue()
        self.camera_worker = CameraWorker(self.detector, self.events)
        self.cancel_video = Event()
        self.active_task: str | None = None
        self.model_loading = False
        self.last_camera_detection: str | None = None

        self._build_ui()
        self._set_controls()
        self.timer = QTimer()
        self.timer.timeout.connect(self._poll_events)
        self.timer.start(50)
        self._load_model_async(DEFAULT_MODEL, is_default=True)

    def run(self) -> None:
        self.window.show()
        self.qt_app.exec()

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        self.window.setCentralWidget(central)

        settings = QGroupBox("Cấu hình")
        settings_layout = QGridLayout(settings)
        self.model_label = QLabel("Model: chưa nạp")
        settings_layout.addWidget(self.model_label, 0, 0, 1, 7)
        self.model_button = QPushButton("Nạp model .pt")
        self.model_button.clicked.connect(self._pick_model)
        settings_layout.addWidget(self.model_button, 0, 7)

        settings_layout.addWidget(QLabel("Confidence"), 1, 0)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(25)
        self.conf_slider.setFixedWidth(180)
        self.conf_slider.valueChanged.connect(self._update_conf_label)
        settings_layout.addWidget(self.conf_slider, 1, 1)
        self.conf_label = QLabel("0.25")
        settings_layout.addWidget(self.conf_label, 1, 2)

        settings_layout.addWidget(QLabel("Kích thước inference"), 1, 3)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "480", "640", "960"])
        self.imgsz_combo.setCurrentText("960")
        settings_layout.addWidget(self.imgsz_combo, 1, 4)

        settings_layout.addWidget(QLabel("Thiết bị"), 1, 5)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "auto"])
        settings_layout.addWidget(self.device_combo, 1, 6)
        outer.addWidget(settings)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)
        actions = QWidget()
        actions.setMaximumWidth(340)
        actions_layout = QVBoxLayout(actions)
        tabs = QTabWidget()
        actions_layout.addWidget(tabs)
        splitter.addWidget(actions)

        camera_tab = QWidget()
        camera_layout = QVBoxLayout(camera_tab)
        camera_hint = QLabel("Webcam PC/laptop hoặc camera ảo Iriun:")
        camera_hint.setWordWrap(True)
        camera_layout.addWidget(camera_hint)
        self.camera_combo = QComboBox()
        camera_layout.addWidget(self.camera_combo)
        self.scan_button = QPushButton("Quét camera")
        self.scan_button.clicked.connect(self._scan_cameras)
        camera_layout.addWidget(self.scan_button)
        self.start_camera_button = QPushButton("Bắt đầu realtime")
        self.start_camera_button.clicked.connect(self._start_camera)
        camera_layout.addWidget(self.start_camera_button)
        self.stop_camera_button = QPushButton("Dừng camera")
        self.stop_camera_button.clicked.connect(self._stop_camera)
        camera_layout.addWidget(self.stop_camera_button)
        camera_layout.addStretch()
        tabs.addTab(camera_tab, "Camera")

        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        image_hint = QLabel("Kết quả được lưu vào outputs/images.")
        image_hint.setWordWrap(True)
        image_layout.addWidget(image_hint)
        self.image_button = QPushButton("Chọn và xử lý ảnh")
        self.image_button.clicked.connect(self._pick_image)
        image_layout.addWidget(self.image_button)
        image_layout.addStretch()
        tabs.addTab(image_tab, "Ảnh")

        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)
        video_hint = QLabel("Kết quả MP4 annotate không giữ audio gốc.")
        video_hint.setWordWrap(True)
        video_layout.addWidget(video_hint)
        self.video_button = QPushButton("Chọn và xử lý video")
        self.video_button.clicked.connect(self._pick_video)
        video_layout.addWidget(self.video_button)
        self.cancel_video_button = QPushButton("Hủy xử lý")
        self.cancel_video_button.clicked.connect(self._cancel_video)
        video_layout.addWidget(self.cancel_video_button)
        video_layout.addStretch()
        tabs.addTab(video_tab, "Video")

        log_group = QGroupBox("Thông báo")
        log_layout = QVBoxLayout(log_group)
        log_actions = QHBoxLayout()
        self.open_outputs_button = QPushButton("Mở folder outputs")
        self.open_outputs_button.clicked.connect(self._open_outputs_folder)
        log_actions.addWidget(self.open_outputs_button)
        self.clear_log_button = QPushButton("Clear thông báo")
        self.clear_log_button.clicked.connect(self._clear_log)
        log_actions.addWidget(self.clear_log_button)
        log_layout.addLayout(log_actions)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        log_layout.addWidget(self.log)
        actions_layout.addWidget(log_group, stretch=1)

        preview_area = QWidget()
        preview_layout = QVBoxLayout(preview_area)
        self.preview = QLabel("Kết quả nhận diện sẽ hiển thị tại đây")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 480)
        self.preview.setStyleSheet("QLabel { background-color: #202124; color: #dddddd; }")
        preview_layout.addWidget(self.preview, stretch=1)
        self.status_label = QLabel("Đang khởi tạo ứng dụng...")
        self.status_label.setWordWrap(True)
        preview_layout.addWidget(self.status_label)
        splitter.addWidget(preview_area)
        splitter.setStretchFactor(1, 1)

    def _settings(self) -> tuple[int, float, str]:
        return (
            int(self.imgsz_combo.currentText()),
            self.conf_slider.value() / 100.0,
            self.device_combo.currentText(),
        )

    def _update_conf_label(self, value: int) -> None:
        self.conf_label.setText(f"{value / 100.0:.2f}")

    def _set_controls(self) -> None:
        ready = self.detector.ready and not self.model_loading
        idle = self.active_task is None
        enabled = ready and idle
        self.model_button.setEnabled(idle and not self.model_loading)
        self.scan_button.setEnabled(enabled)
        self.start_camera_button.setEnabled(enabled and self.camera_combo.count() > 0)
        self.stop_camera_button.setEnabled(self.active_task == "camera")
        self.image_button.setEnabled(enabled)
        self.video_button.setEnabled(enabled)
        self.cancel_video_button.setEnabled(self.active_task == "video")
        self.imgsz_combo.setEnabled(enabled)
        self.device_combo.setEnabled(enabled)
        self.conf_slider.setEnabled(enabled)
        self.open_outputs_button.setEnabled(True)
        self.clear_log_button.setEnabled(True)

    def _append_log(self, message: str) -> None:
        print(message)
        self.log.append(message)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _clear_log(self) -> None:
        self.log.clear()
        self._append_log("Đã clear thông báo.")

    def _open_outputs_folder(self) -> None:
        OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(OUTPUTS_ROOT))
        except OSError as exc:
            self._show_error(f"Không thể mở folder outputs.\n{exc}")

    def _load_model_async(self, path: Path, is_default: bool = False) -> None:
        self.model_loading = True
        self._set_status(f"Đang nạp model: {path.name}")
        self._set_controls()

        def load() -> None:
            try:
                info = self.detector.load_model(path)
                self.events.put(("model_loaded", (info, is_default)))
            except Exception as exc:
                self.events.put(("model_error", (str(exc), is_default)))

        Thread(target=load, daemon=True).start()

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Chọn checkpoint YOLO Pose", "", "PyTorch checkpoint (*.pt);;Tất cả file (*)"
        )
        if path:
            self._load_model_async(Path(path))

    def _scan_cameras(self) -> None:
        self.active_task = "scan"
        self._set_status("Đang quét camera Windows, bao gồm Iriun nếu đã kết nối...")
        self._set_controls()

        def scan() -> None:
            self.events.put(("cameras", find_camera_indices()))

        Thread(target=scan, daemon=True).start()

    def _start_camera(self) -> None:
        if not self.camera_combo.currentText():
            self._show_error("Chưa chọn camera.")
            return
        index = int(self.camera_combo.currentText().split(" ")[1])
        imgsz, conf, device = self._settings()
        self.active_task = "camera"
        self.last_camera_detection = None
        self.alarm.reset()
        self._set_controls()
        self.camera_worker.start(index, imgsz, conf, device)

    def _stop_camera(self) -> None:
        self.camera_worker.stop()
        self._set_status("Đang dừng camera...")

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Chọn ảnh", "", "Ảnh (*.jpg *.jpeg *.png *.bmp);;Tất cả file (*)"
        )
        if not path:
            return
        imgsz, conf, device = self._settings()
        self.active_task = "image"
        self._set_status("Đang xử lý ảnh...")
        self._set_controls()

        def run_image() -> None:
            try:
                media, result = process_image(
                    self.detector, Path(path), IMAGE_OUTPUTS, imgsz, conf, device
                )
                self.events.put(("image_done", (media, result)))
            except Exception as exc:
                self.events.put(("task_error", str(exc)))

        Thread(target=run_image, daemon=True).start()

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Chọn video", "", "Video (*.mp4 *.avi *.mov *.mkv);;Tất cả file (*)"
        )
        if not path:
            return
        imgsz, conf, device = self._settings()
        self.active_task = "video"
        self.cancel_video.clear()
        self._set_status("Đang xử lý video...")
        self._set_controls()

        def preview(result: DetectionResult, current: int, total: int) -> None:
            self.events.put(("video_preview", (result, current, total)))

        def run_video() -> None:
            try:
                media = process_video(
                    self.detector,
                    Path(path),
                    VIDEO_OUTPUTS,
                    imgsz,
                    conf,
                    device,
                    self.cancel_video,
                    preview,
                )
                self.events.put(("video_done", media))
            except InterruptedError as exc:
                self.events.put(("task_cancelled", str(exc)))
            except Exception as exc:
                self.events.put(("task_error", str(exc)))

        Thread(target=run_video, daemon=True).start()

    def _cancel_video(self) -> None:
        self.cancel_video.set()
        self._set_status("Đang hủy xử lý video...")

    def _show_frame(self, frame: Any) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)

    def _handle_detection_message(self, result: DetectionResult, realtime: bool) -> None:
        if realtime:
            state = "fall" if result.has_fall else ("detected" if result.has_detection else "none")
            if state != self.last_camera_detection:
                if state == "fall":
                    self._append_log("Realtime: phát hiện FALL.")
                elif state == "detected":
                    self._append_log("Realtime: phát hiện no_fall.")
                else:
                    self._append_log("Realtime: không nhận diện được đối tượng.")
                self.last_camera_detection = state
            self.alarm.notify_if_needed(result.has_fall)
        if result.warning:
            self._append_log(result.warning)

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                self._handle_event(event, payload)
        except Empty:
            return

    def _handle_event(self, event: str, payload: Any) -> None:
        if event == "model_loaded":
            info, is_default = payload
            self.model_loading = False
            self._describe_model(info, is_default)
        elif event == "model_error":
            error, is_default = payload
            self.model_loading = False
            self._show_error(f"Không thể nạp model.\n{error}")
            self._set_status(
                "Model mặc định không sẵn sàng." if is_default else "Giữ nguyên model đang hoạt động."
            )
        elif event == "cameras":
            self.active_task = None
            self.camera_combo.clear()
            self.camera_combo.addItems([f"Camera {index}" for index in payload])
            if payload:
                self._set_status(f"Tìm thấy {len(payload)} camera.")
                self._append_log(
                    "Camera khả dụng: " + ", ".join(f"Camera {index}" for index in payload)
                )
            else:
                self._set_status("Không tìm thấy camera. Kiểm tra webcam hoặc Iriun.")
                self._append_log("Không tìm thấy camera Windows khả dụng.")
        elif event == "status":
            self._set_status(payload)
        elif event == "error":
            self._show_error(payload)
        elif event == "camera_frame":
            self._show_frame(payload.annotated_frame)
            self._handle_detection_message(payload, realtime=True)
            if payload.has_fall:
                self._set_status("Cảnh báo: phát hiện FALL.")
            elif payload.has_detection:
                self._set_status("Realtime: phát hiện no_fall.")
            else:
                self._set_status("Realtime: không nhận diện được đối tượng.")
        elif event == "camera_stopped":
            self.active_task = None
            self._set_status("Camera đã dừng.")
        elif event == "image_done":
            media, result = payload
            self.active_task = None
            self._show_frame(result.annotated_frame)
            label = (
                "fall"
                if result.has_fall
                else ("no_fall" if result.has_detection else "không nhận diện được")
            )
            message = f"Ảnh: {label}. Đã lưu: {media.output_path}"
            self._set_status(message)
            self._append_log(message)
            if not result.has_detection:
                QMessageBox.information(self.window, "Kết quả", "Không nhận diện được đối tượng trong ảnh.")
        elif event == "video_preview":
            result, current, total = payload
            self._show_frame(result.annotated_frame)
            progress = f"{current}/{total}" if total else str(current)
            self._set_status(f"Đang xử lý video: frame {progress}")
        elif event == "video_done":
            self.active_task = None
            if payload.detection_count:
                outcome = "có fall" if payload.fall_found else "chỉ có no_fall"
            else:
                outcome = "không nhận diện được đối tượng"
            message = f"Video: {outcome}. Đã lưu: {payload.output_path}"
            self._set_status(message)
            self._append_log(message)
            if not payload.detection_count:
                QMessageBox.information(
                    self.window, "Kết quả", "Không nhận diện được đối tượng trong video."
                )
        elif event == "task_cancelled":
            self.active_task = None
            self._set_status(payload)
            self._append_log(payload)
        elif event == "task_error":
            self.active_task = None
            self._show_error(payload)
        self._set_controls()

    def _describe_model(self, info: ModelInfo, is_default: bool) -> None:
        origin = "mặc định" if is_default else "được chọn"
        self.model_label.setText(
            f"Model {origin}: {info.path.name} | labels: no_fall/fall | keypoints: {info.kpt_shape[0]}"
        )
        message = f"Đã nạp model {origin}: {info.path}"
        self._set_status(message)
        self._append_log(message)

    def _show_error(self, message: str) -> None:
        self._set_status("Có lỗi. Xem thông báo để xử lý.")
        self._append_log("LỖI: " + message)
        QMessageBox.critical(self.window, "Lỗi", message)

    def _shutdown(self) -> None:
        self.camera_worker.stop()
        self.cancel_video.set()
