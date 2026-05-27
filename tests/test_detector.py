import unittest
from pathlib import Path

from fall_detector.detector import ModelValidationError, PoseFallDetector, validate_pose_model


class FakeInnerModel:
    kpt_shape = [18, 3]


class FakeModel:
    task = "pose"
    names = {0: "no_fall", 1: "fall"}
    model = FakeInnerModel()


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, values):
        self.cls = FakeTensor(values)


class FakeResult:
    def __init__(self, values):
        self.boxes = FakeBoxes(values)

    def plot(self, **_kwargs):
        return "annotated"


class DetectorContractTests(unittest.TestCase):
    def test_valid_pose_model_contract_is_accepted(self):
        info = validate_pose_model(FakeModel(), Path("best.pt"))
        self.assertEqual(info.names, {0: "no_fall", 1: "fall"})
        self.assertEqual(info.kpt_shape, (18, 3))

    def test_wrong_task_is_rejected(self):
        model = FakeModel()
        model.task = "detect"
        with self.assertRaisesRegex(ModelValidationError, "YOLO Pose"):
            validate_pose_model(model, Path("detect.pt"))

    def test_wrong_labels_are_rejected(self):
        model = FakeModel()
        model.names = {0: "person", 1: "fall"}
        with self.assertRaisesRegex(ModelValidationError, "no_fall"):
            validate_pose_model(model, Path("wrong.pt"))

    def test_predict_flags_fall_and_no_detection(self):
        detector = PoseFallDetector()
        detector._info = validate_pose_model(FakeModel(), Path("best.pt"))
        detector._model = FakeModel()
        detector._model.predict = lambda **_kwargs: [FakeResult([0.0, 1.0])]

        result = detector.predict("frame")
        self.assertTrue(result.has_detection)
        self.assertTrue(result.has_fall)
        self.assertEqual(result.detection_count, 2)

        detector._model.predict = lambda **_kwargs: [FakeResult([])]
        result = detector.predict("frame")
        self.assertFalse(result.has_detection)
        self.assertFalse(result.has_fall)


if __name__ == "__main__":
    unittest.main()

