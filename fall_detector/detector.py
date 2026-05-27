from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any


class ModelValidationError(ValueError):
    """Raised when a checkpoint does not satisfy the fall-pose contract."""


class InferenceError(RuntimeError):
    """Raised when inference cannot complete."""


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    names: dict[int, str]
    kpt_shape: tuple[int, int]


@dataclass
class DetectionResult:
    annotated_frame: Any
    has_detection: bool
    has_fall: bool
    detection_count: int
    warning: str | None = None


def _normalized_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        try:
            return {int(key): str(value) for key, value in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("Danh sách label của model không hợp lệ.") from exc
    if isinstance(raw_names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise ModelValidationError("Không đọc được label của model.")


def _read_kpt_shape(model: Any) -> tuple[int, int] | None:
    inner_model = getattr(model, "model", None)
    shape = getattr(inner_model, "kpt_shape", None)
    if shape is None:
        layers = getattr(inner_model, "model", None)
        if layers:
            shape = getattr(layers[-1], "kpt_shape", None)
    if shape is None:
        return None
    try:
        return int(shape[0]), int(shape[1])
    except (IndexError, TypeError, ValueError):
        return None


def validate_pose_model(model: Any, path: Path) -> ModelInfo:
    if getattr(model, "task", None) != "pose":
        raise ModelValidationError(
            "Model không phải YOLO Pose. Hãy chọn best.pt được train cho fall pose."
        )

    names = _normalized_names(getattr(model, "names", None))
    if names != {0: "no_fall", 1: "fall"}:
        raise ModelValidationError(
            "Model phải có đúng hai label: 0=no_fall và 1=fall."
        )

    kpt_shape = _read_kpt_shape(model)
    if kpt_shape is None or kpt_shape[0] < 1 or kpt_shape[1] < 2:
        raise ModelValidationError("Model pose không chứa cấu hình keypoint hợp lệ.")

    return ModelInfo(path=path, names=names, kpt_shape=kpt_shape)


class PoseFallDetector:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._info: ModelInfo | None = None
        self._lock = Lock()

    @property
    def info(self) -> ModelInfo | None:
        return self._info

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load_model(self, path: str | Path) -> ModelInfo:
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise ModelValidationError(f"Không tìm thấy model: {checkpoint}")
        if checkpoint.suffix.lower() != ".pt":
            raise ModelValidationError("Chỉ hỗ trợ checkpoint Ultralytics định dạng .pt.")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelValidationError(
                "Chưa cài ultralytics. Hãy cài dependencies từ requirements.txt."
            ) from exc

        try:
            candidate = YOLO(str(checkpoint))
            info = validate_pose_model(candidate, checkpoint)
        except ModelValidationError:
            raise
        except Exception as exc:
            raise ModelValidationError(f"Không thể nạp model: {exc}") from exc

        with self._lock:
            self._model = candidate
            self._info = info
        return info

    def predict(
        self, frame: Any, imgsz: int = 960, conf: float = 0.25, device: str = "cpu"
    ) -> DetectionResult:
        with self._lock:
            if self._model is None or self._info is None:
                raise InferenceError("Chưa nạp model để nhận diện.")

            warning = None
            predict_device = None if device == "auto" else device
            try:
                result = self._predict_frame(frame, imgsz, conf, predict_device)
            except Exception as exc:
                if device != "auto":
                    raise InferenceError(f"Lỗi nhận diện: {exc}") from exc
                try:
                    result = self._predict_frame(frame, imgsz, conf, "cpu")
                    warning = "Không dùng được thiết bị Auto; đã chuyển sang CPU."
                except Exception as retry_exc:
                    raise InferenceError(f"Lỗi nhận diện: {retry_exc}") from retry_exc

        boxes = getattr(result, "boxes", None)
        classes = []
        if boxes is not None and getattr(boxes, "cls", None) is not None:
            classes = [int(value) for value in boxes.cls.cpu().tolist()]
        return DetectionResult(
            annotated_frame=result.plot(boxes=True, labels=True, kpt_line=True),
            has_detection=bool(classes),
            has_fall=1 in classes,
            detection_count=len(classes),
            warning=warning,
        )

    def _predict_frame(
        self, frame: Any, imgsz: int, conf: float, device: str | None
    ) -> Any:
        results = self._model.predict(
            source=frame, imgsz=imgsz, conf=conf, device=device, verbose=False
        )
        if not results:
            raise InferenceError("Model không trả về kết quả.")
        return results[0]
