from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .detector import DetectionResult, PoseDetection, PostprocessStatus


@dataclass(frozen=True)
class BedROI:
    enabled: bool
    points: tuple[tuple[float, float], ...]

    @classmethod
    def load(cls, path: Path) -> BedROI:
        if not path.is_file():
            return cls(enabled=False, points=())
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        points = tuple(
            (float(point[0]), float(point[1]))
            for point in data.get("points", [])
        )
        return cls(enabled=bool(data.get("enabled", False)) and len(points) >= 3, points=points)

    def pixel_points(self, width: int, height: int) -> np.ndarray | None:
        if not self.enabled:
            return None
        return np.array(
            [[int(x * width), int(y * height)] for x, y in self.points],
            dtype=np.int32,
        )

    def contains(self, x: float, y: float, width: int, height: int) -> bool:
        polygon = self.pixel_points(width, height)
        if polygon is None:
            return False
        return _point_inside_polygon(float(x), float(y), polygon)


@dataclass
class _FrameState:
    raw_has_fall: bool
    center: tuple[float, float] | None
    size: tuple[float, float] | None
    aspect: float | None
    inside_bed_roi: bool


class FallPostProcessor:
    def __init__(self, fps: float, bed_roi: BedROI | None = None) -> None:
        self.fps = max(float(fps), 1.0)
        self.bed_roi = bed_roi or BedROI(enabled=False, points=())
        self.smoothing_frames = max(5, int(round(self.fps * 1.0)))
        self.smoothing_required = max(3, int(round(self.smoothing_frames * 0.65)))
        self.drop_frames = max(3, int(round(self.fps * 0.8)))
        self.still_frames = max(5, int(round(self.fps * 1.0)))
        self.possible_fall_frames = max(6, int(round(self.fps * 1.0)))
        self.confirm_grace_frames = max(3, int(round(self.fps * 0.3)))
        self.recent_bed_frames = max(5, int(round(self.fps * 2.0)))
        self._drop_latch = 0
        self._possible_fall_latch = 0
        self._possible_fall_age = 0
        self._recent_bed_latch = 0
        self._last_floor_like_off_bed = False
        self.history: deque[_FrameState] = deque(
            maxlen=max(
                self.smoothing_frames,
                self.drop_frames,
                self.still_frames,
                self.possible_fall_frames,
            )
            + 2
        )
        self.state = "NORMAL"

    def apply(self, result: DetectionResult) -> DetectionResult:
        frame = result.annotated_frame
        height, width = frame.shape[:2]
        target = self._select_target(result.detections or [])
        raw_has_fall = any(detection.class_id == 1 for detection in result.detections or [])

        center = self._center(target) if target else None
        size = self._size(target) if target else None
        aspect = (size[0] / size[1]) if size and size[1] > 0 else None
        inside_bed_roi = self._inside_bed_roi(target, width, height) if target else False

        self.history.append(
            _FrameState(
                raw_has_fall=raw_has_fall,
                center=center,
                size=size,
                aspect=aspect,
                inside_bed_roi=inside_bed_roi,
            )
        )

        smoothed_fall = self._smoothed_fall()
        raw_sudden_drop = self._sudden_drop(height)
        if raw_sudden_drop:
            self._drop_latch = self.smoothing_frames + self.drop_frames
        sudden_drop = raw_sudden_drop or self._drop_latch > 0
        still = self._still(width, height)
        self._update_state(
            raw_has_fall=raw_has_fall,
            smoothed_fall=smoothed_fall,
            sudden_drop=sudden_drop,
            inside_bed_roi=inside_bed_roi,
            still=still,
            has_detection=result.has_detection,
            aspect=aspect,
            center=center,
            size=size,
            frame_height=height,
        )
        if self._drop_latch > 0:
            self._drop_latch -= 1
        if self._recent_bed_latch > 0:
            self._recent_bed_latch -= 1

        result.has_fall = self.state == "FALL"
        result.postprocess = PostprocessStatus(
            state=self.state,
            raw_has_fall=raw_has_fall,
            smoothed_fall=smoothed_fall,
            sudden_drop=sudden_drop,
            inside_bed_roi=inside_bed_roi,
            still=still,
        )
        self._draw_overlay(result, width, height)
        return result

    def _select_target(self, detections: Iterable[PoseDetection]) -> PoseDetection | None:
        detections = list(detections)
        if not detections:
            return None
        fall_detections = [detection for detection in detections if detection.class_id == 1]
        candidates = fall_detections or detections
        return max(candidates, key=lambda detection: self._area(detection))

    def _update_state(
        self,
        raw_has_fall: bool,
        smoothed_fall: bool,
        sudden_drop: bool,
        inside_bed_roi: bool,
        still: bool,
        has_detection: bool,
        aspect: float | None,
        center: tuple[float, float] | None,
        size: tuple[float, float] | None,
        frame_height: int,
    ) -> None:
        if inside_bed_roi:
            self._recent_bed_latch = self.recent_bed_frames
            self._clear_possible_fall()
            self.state = "ON_BED"
            return

        if not has_detection and self._recent_bed_latch > 0:
            self._clear_possible_fall()
            self.state = "ON_BED"
            return

        current_floor_like = (
            has_detection
            and self._recent_bed_latch == 0
            and self._lying_or_low(aspect, center, size, frame_height)
        )

        fall_motion = (
            self.state != "FALL"
            and sudden_drop
            and (raw_has_fall or smoothed_fall)
            and self._recent_bed_latch == 0
        )
        if fall_motion:
            if self.state != "POSSIBLE_FALL":
                self._possible_fall_age = 0
            self._possible_fall_latch = self.possible_fall_frames
            self.state = "POSSIBLE_FALL"
            self._last_floor_like_off_bed = current_floor_like

        if self.state == "FALL":
            return

        if self.state != "POSSIBLE_FALL":
            self.state = "NORMAL"
            return

        self._possible_fall_age += 1
        if current_floor_like:
            self._last_floor_like_off_bed = True

        if self._recovered_upright(raw_has_fall, has_detection, aspect, center, frame_height):
            self._clear_possible_fall()
            self.state = "NORMAL"
            return

        confirmation_ready = self._possible_fall_age >= self.confirm_grace_frames
        loss_confirms_fall = not has_detection and self._last_floor_like_off_bed
        pose_confirms_fall = has_detection and current_floor_like and (
            still or raw_has_fall or smoothed_fall
        )
        if confirmation_ready and (loss_confirms_fall or pose_confirms_fall):
            self.state = "FALL"
            return

        self._possible_fall_latch -= 1
        if self._possible_fall_latch <= 0:
            self._clear_possible_fall()
            self.state = "NORMAL"

    def _clear_possible_fall(self) -> None:
        self._possible_fall_latch = 0
        self._possible_fall_age = 0
        self._last_floor_like_off_bed = False

    def _smoothed_fall(self) -> bool:
        recent = list(self.history)[-self.smoothing_frames :]
        return sum(1 for item in recent if item.raw_has_fall) >= self.smoothing_required

    def _sudden_drop(self, frame_height: int) -> bool:
        current = self.history[-1]
        if current.center is None or current.aspect is None:
            return False
        previous_items = [
            item
            for item in list(self.history)[-self.drop_frames - 1 : -1]
            if item.center is not None and item.aspect is not None
        ]
        if not previous_items:
            return False
        previous = previous_items[0]
        dy = current.center[1] - previous.center[1]
        vertical_drop = dy > frame_height * 0.10
        became_horizontal = previous.aspect < 0.9 and current.aspect > 1.15
        return vertical_drop or became_horizontal

    def _still(self, width: int, height: int) -> bool:
        recent = [
            item
            for item in list(self.history)[-self.still_frames :]
            if item.center is not None and item.size is not None
        ]
        if len(recent) < self.still_frames:
            return False
        xs = [item.center[0] for item in recent if item.center is not None]
        ys = [item.center[1] for item in recent if item.center is not None]
        areas = [
            item.size[0] * item.size[1]
            for item in recent
            if item.size is not None
        ]
        max_center_shift = max(max(xs) - min(xs), max(ys) - min(ys))
        area_span = max(areas) - min(areas)
        mean_area = sum(areas) / len(areas)
        return max_center_shift < max(width, height) * 0.03 and area_span < mean_area * 0.20

    def _lying_or_low(
        self,
        aspect: float | None,
        center: tuple[float, float] | None,
        size: tuple[float, float] | None,
        frame_height: int,
    ) -> bool:
        if aspect is not None and aspect > 1.15:
            return True
        if center is None or size is None:
            return False
        bottom = center[1] + size[1] / 2.0
        return center[1] > frame_height * 0.68 or bottom > frame_height * 0.86

    def _recovered_upright(
        self,
        raw_has_fall: bool,
        has_detection: bool,
        aspect: float | None,
        center: tuple[float, float] | None,
        frame_height: int,
    ) -> bool:
        if raw_has_fall or not has_detection or aspect is None or center is None:
            return False
        return aspect < 0.85 and center[1] < frame_height * 0.72

    def _inside_bed_roi(
        self, detection: PoseDetection, width: int, height: int
    ) -> bool:
        if not self.bed_roi.enabled:
            return False
        x1, y1, x2, y2 = detection.box_xyxy
        center_x, center_y = self._center(detection)
        sample_points = [
            (center_x, center_y),
            ((x1 + x2) / 2.0, y2),
            (x1 + (x2 - x1) * 0.25, y1 + (y2 - y1) * 0.75),
            (x1 + (x2 - x1) * 0.50, y1 + (y2 - y1) * 0.75),
            (x1 + (x2 - x1) * 0.75, y1 + (y2 - y1) * 0.75),
        ]
        hits = sum(
            1
            for point_x, point_y in sample_points
            if self.bed_roi.contains(point_x, point_y, width, height)
        )
        return hits >= 2

    def _draw_overlay(self, result: DetectionResult, width: int, height: int) -> None:
        frame = result.annotated_frame
        try:
            import cv2
        except ImportError:
            return

        polygon = self.bed_roi.pixel_points(width, height)
        if polygon is not None:
            cv2.polylines(frame, [polygon], isClosed=True, color=(255, 180, 0), thickness=2)

        status = result.postprocess
        if status is None:
            return
        final_label = "FALL" if result.has_fall else "NO_FALL"
        color = (0, 0, 255) if result.has_fall else (0, 180, 0)
        lines = [
            f"FINAL: {final_label} | state={status.state}",
            (
                "raw_fall={raw} smooth={smooth} drop={drop} bed={bed} still={still}"
            ).format(
                raw=int(status.raw_has_fall),
                smooth=int(status.smoothed_fall),
                drop=int(status.sudden_drop),
                bed=int(status.inside_bed_roi),
                still=int(status.still),
            ),
        ]
        x, y = 12, 28
        for index, line in enumerate(lines):
            baseline = y + index * 28
            cv2.putText(
                frame,
                line,
                (x, baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (x, baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    def _center(self, detection: PoseDetection) -> tuple[float, float]:
        x1, y1, x2, y2 = detection.box_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _size(self, detection: PoseDetection) -> tuple[float, float]:
        x1, y1, x2, y2 = detection.box_xyxy
        return max(x2 - x1, 0.0), max(y2 - y1, 0.0)

    def _area(self, detection: PoseDetection) -> float:
        width, height = self._size(detection)
        return width * height


def _point_inside_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    inside = False
    points = polygon.reshape(-1, 2)
    j = len(points) - 1
    for i, point in enumerate(points):
        xi, yi = float(point[0]), float(point[1])
        xj, yj = float(points[j][0]), float(points[j][1])
        if (yi > y) != (yj > y):
            intersection_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x <= intersection_x:
                inside = not inside
        j = i
    return inside
