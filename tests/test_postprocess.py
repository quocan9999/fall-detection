import unittest

import numpy as np

from fall_detector.detector import DetectionResult, PoseDetection
from fall_detector.postprocess import BedROI, FallPostProcessor


def result_for(class_id, box):
    return DetectionResult(
        annotated_frame=np.zeros((200, 200, 3), dtype=np.uint8),
        has_detection=True,
        has_fall=class_id == 1,
        detection_count=1,
        detections=[PoseDetection(class_id=class_id, confidence=0.9, box_xyxy=box)],
    )


def no_detection_result():
    return DetectionResult(
        annotated_frame=np.zeros((200, 200, 3), dtype=np.uint8),
        has_detection=False,
        has_fall=False,
        detection_count=0,
        detections=[],
    )


class PostProcessTests(unittest.TestCase):
    def test_bed_roi_suppresses_lying_without_drop(self):
        processor = FallPostProcessor(
            fps=5,
            bed_roi=BedROI(
                enabled=True,
                points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            ),
        )

        final = None
        for _ in range(12):
            final = processor.apply(result_for(1, (40, 100, 170, 150)))

        self.assertIsNotNone(final)
        self.assertFalse(final.has_fall)
        self.assertEqual(final.postprocess.state, "ON_BED")

    def test_drop_then_stillness_confirms_fall(self):
        processor = FallPostProcessor(fps=5)

        for _ in range(4):
            processor.apply(result_for(0, (80, 20, 120, 170)))

        final = None
        for _ in range(14):
            final = processor.apply(result_for(1, (35, 130, 170, 175)))

        self.assertIsNotNone(final)
        self.assertTrue(final.has_fall)
        self.assertEqual(final.postprocess.state, "FALL")

    def test_short_raw_fall_then_detection_loss_confirms_fall(self):
        processor = FallPostProcessor(fps=5)

        for _ in range(4):
            processor.apply(result_for(0, (80, 20, 120, 170)))

        processor.apply(result_for(1, (35, 110, 170, 175)))

        final = None
        for _ in range(4):
            final = processor.apply(no_detection_result())

        self.assertIsNotNone(final)
        self.assertTrue(final.has_fall)
        self.assertEqual(final.postprocess.state, "FALL")

    def test_low_raw_fall_pose_confirms_without_stillness(self):
        processor = FallPostProcessor(fps=5)

        for _ in range(4):
            processor.apply(result_for(0, (80, 20, 120, 170)))

        processor.apply(result_for(1, (35, 95, 170, 175)))

        final = None
        for _ in range(3):
            final = processor.apply(result_for(1, (35, 120, 190, 175)))

        self.assertIsNotNone(final)
        self.assertTrue(final.has_fall)
        self.assertEqual(final.postprocess.state, "FALL")

    def test_detection_loss_after_bed_state_does_not_confirm_fall(self):
        processor = FallPostProcessor(
            fps=5,
            bed_roi=BedROI(
                enabled=True,
                points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            ),
        )

        for _ in range(5):
            processor.apply(result_for(1, (40, 100, 170, 150)))

        final = None
        for _ in range(4):
            final = processor.apply(no_detection_result())

        self.assertIsNotNone(final)
        self.assertFalse(final.has_fall)
        self.assertEqual(final.postprocess.state, "ON_BED")

    def test_detection_loss_without_floor_like_pose_does_not_confirm_fall(self):
        processor = FallPostProcessor(fps=5)

        for _ in range(4):
            processor.apply(result_for(0, (80, 20, 120, 170)))

        processor.apply(result_for(1, (80, 61, 120, 171)))

        final = None
        for _ in range(4):
            final = processor.apply(no_detection_result())

        self.assertIsNotNone(final)
        self.assertFalse(final.has_fall)
        self.assertEqual(final.postprocess.state, "POSSIBLE_FALL")


if __name__ == "__main__":
    unittest.main()
