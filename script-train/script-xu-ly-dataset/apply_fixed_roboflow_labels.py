from __future__ import annotations

import csv
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
ORIGIN_DATASET_DIR = ROOT / "origin-dataset"
OUTPUTS_DIR = ORIGIN_DATASET_DIR / "outputs"
SCRIPT_OUTPUT_DIR = OUTPUTS_DIR / Path(__file__).name
PATCH_ZIP = ORIGIN_DATASET_DIR / "Fall detection.v7-chinh-sua-9-keypoint.yolov8.zip"
OUTPUT_BASE = OUTPUTS_DIR
DATASET_5CLASS = OUTPUT_BASE / "fall-detection-clean-fastdup"
DATASET_2CLASS = OUTPUT_BASE / "fall-detection-clean-fastdup-2class"
ZIP_5CLASS = OUTPUT_BASE / "fall-detection-clean-fastdup.zip"
ZIP_2CLASS = OUTPUT_BASE / "fall-detection-clean-fastdup-2class.zip"
REPORT_DIR = SCRIPT_OUTPUT_DIR / "reports"

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}
EXPECTED_VALUES = 59
EXPECTED_KEYPOINTS = 18
CLASS_5_TO_2 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}
CLASS_5_NAMES = ["Sitting", "Sleeping", "Standing", "Walking", "falling"]
CLASS_2_NAMES = ["no_fall", "fall"]

TARGETS = {
    "img_3586": "train",
    "img_5794": "train",
    "img_5799": "train",
    "img_5836": "train",
    "img_5936": "train",
    "img_6088": "train",
    "img_6091": "train",
    "img_5939": "valid",
    "img_6083": "test",
}


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def fail(message: str) -> None:
    print(f"[LOI] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def short_id(path_or_name: str) -> str:
    name = Path(path_or_name).name
    if "_jpg.rf." in name:
        return name.split("_jpg.rf.", 1)[0]
    if "_jpeg.rf." in name:
        return name.split("_jpeg.rf.", 1)[0]
    if "_png.rf." in name:
        return name.split("_png.rf.", 1)[0]
    return Path(name).stem


def list_split_files(dataset_root: Path, split: str, kind: str, suffixes: set[str]) -> dict[str, Path]:
    directory = dataset_root / split / kind
    if not directory.is_dir():
        fail(f"Thieu thu muc: {directory}")
    result: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            sid = short_id(path.name)
            if sid in result:
                fail(f"Trung short id {sid} trong {directory}")
            result[sid] = path
    return result


def read_zip_index(zip_path: Path) -> dict[str, dict[str, str]]:
    if not zip_path.exists():
        fail(f"Khong tim thay ZIP patch: {zip_path}")
    index: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            parts = Path(name).parts
            if len(parts) < 3 or parts[0] not in SPLITS:
                continue
            split, kind = parts[0], parts[1]
            suffix = Path(name).suffix.lower()
            if kind == "labels" and suffix == ".txt":
                index.setdefault(short_id(name), {})["label"] = name
                index[short_id(name)]["split"] = split
            elif kind == "images" and suffix in IMAGE_EXTS:
                index.setdefault(short_id(name), {})["image"] = name
                index[short_id(name)]["split"] = split
    return index


def parse_label_text(text: str, expected_nc: int, allow_empty: bool = False) -> list[list[str]]:
    rows: list[list[str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != EXPECTED_VALUES:
            fail(f"Label patch sai so cot o dong {line_number}: {len(parts)} != {EXPECTED_VALUES}")
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts]
        except ValueError:
            fail(f"Label patch co gia tri khong phai so o dong {line_number}")
        if not 0 <= class_id < expected_nc:
            fail(f"Label patch co class ngoai range o dong {line_number}: {class_id}")
        if float(parts[0]) != class_id:
            fail(f"Class id patch khong phai so nguyen o dong {line_number}: {parts[0]}")
        rows.append(parts)
    if not rows and not allow_empty:
        fail("Label patch rong.")
    return rows


def geometry_errors(parts: list[str]) -> list[str]:
    nums = [float(value) for value in parts]
    errors: list[str] = []
    x, y, w, h = nums[1:5]
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        errors.append("bbox_normalized_invalid")
    if x - w / 2 < -0.02 or y - h / 2 < -0.02 or x + w / 2 > 1.02 or y + h / 2 > 1.02:
        errors.append("bbox_outside_image")
    bad_keypoints = 0
    for offset in range(5, EXPECTED_VALUES, 3):
        kx, ky, visibility = nums[offset], nums[offset + 1], nums[offset + 2]
        if visibility > 0 and not (0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0):
            bad_keypoints += 1
    if bad_keypoints:
        errors.append(f"visible_keypoints_outside_image={bad_keypoints}")
    return errors


def map_label_rows_to_2class(rows: list[list[str]]) -> list[list[str]]:
    mapped: list[list[str]] = []
    for parts in rows:
        class_5 = int(float(parts[0]))
        if class_5 not in CLASS_5_TO_2:
            fail(f"Khong map duoc class 5-class: {class_5}")
        new_parts = parts.copy()
        new_parts[0] = str(CLASS_5_TO_2[class_5])
        mapped.append(new_parts)
    return mapped


def write_label(path: Path, rows: list[list[str]]) -> None:
    path.write_text("\n".join(" ".join(row) for row in rows) + "\n", encoding="utf-8")


def remove_sample(dataset_root: Path, split: str, sid: str) -> tuple[list[str], list[str]]:
    images = list_split_files(dataset_root, split, "images", IMAGE_EXTS)
    labels = list_split_files(dataset_root, split, "labels", {".txt"})
    removed_images: list[str] = []
    removed_labels: list[str] = []
    image = images.get(sid)
    label = labels.get(sid)
    if image is not None:
        os.chmod(image, 0o666)
        image.unlink()
        removed_images.append(str(image.relative_to(dataset_root)))
    if label is not None:
        os.chmod(label, 0o666)
        label.unlink()
        removed_labels.append(str(label.relative_to(dataset_root)))
    return removed_images, removed_labels


def validate_dataset(dataset_root: Path, expected_nc: int) -> dict:
    summary = {
        "image_counts": Counter(),
        "label_counts": Counter(),
        "class_counts": {split: Counter() for split in SPLITS},
        "missing_labels": [],
        "orphan_labels": [],
        "bad_lines": [],
        "geometry_warnings": [],
        "corrupt_images": [],
    }
    for split in SPLITS:
        images = list_split_files(dataset_root, split, "images", IMAGE_EXTS)
        labels = list_split_files(dataset_root, split, "labels", {".txt"})
        summary["image_counts"][split] = len(images)
        summary["label_counts"][split] = len(labels)
        for sid in sorted(set(images) - set(labels)):
            summary["missing_labels"].append({"split": split, "image": images[sid].name})
        for sid in sorted(set(labels) - set(images)):
            summary["orphan_labels"].append({"split": split, "label": labels[sid].name})
        for sid, image_path in images.items():
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                summary["corrupt_images"].append({"split": split, "image": image_path.name, "error": str(exc)})
        for label_path in labels.values():
            rows = parse_label_text(label_path.read_text(encoding="utf-8"), expected_nc, allow_empty=True)
            for line_number, parts in enumerate(rows, start=1):
                class_id = int(float(parts[0]))
                summary["class_counts"][split][class_id] += 1
                errors = geometry_errors(parts)
                if errors:
                    summary["geometry_warnings"].append(
                        {
                            "split": split,
                            "label": label_path.name,
                            "line": line_number,
                            "class_id": class_id,
                            "errors": ";".join(errors),
                        }
                    )
    if summary["missing_labels"] or summary["orphan_labels"] or summary["bad_lines"] or summary["corrupt_images"]:
        fail(f"Dataset khong dat dieu kien train: {dataset_root}")
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def zip_dataset(dataset_root: Path, output_zip: Path) -> None:
    tmp_zip = output_zip.with_suffix(output_zip.suffix + ".tmp")
    if tmp_zip.exists():
        tmp_zip.unlink()
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in dataset_root.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(dataset_root).as_posix())
    tmp_zip.replace(output_zip)


def main() -> None:
    log("Tiến hành chuẩn bị patch label từ ZIP Roboflow cho các dataset resize hiện có.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_root in (DATASET_5CLASS, DATASET_2CLASS):
        if not dataset_root.is_dir():
            fail(f"Khong tim thay dataset: {dataset_root}")

    log("Tiến hành đọc ZIP patch và thay/xoá các label mục tiêu.")
    patch_index = read_zip_index(PATCH_ZIP)
    patch_rows: list[dict] = []
    deleted_rows: list[dict] = []

    with zipfile.ZipFile(PATCH_ZIP) as archive:
        for sid, expected_split in TARGETS.items():
            item = patch_index.get(sid)
            if item is None or "label" not in item:
                removed_5_images, removed_5_labels = remove_sample(DATASET_5CLASS, expected_split, sid)
                removed_2_images, removed_2_labels = remove_sample(DATASET_2CLASS, expected_split, sid)
                deleted_rows.append(
                    {
                        "image_id": sid,
                        "expected_split": expected_split,
                        "reason": "missing_in_patch_zip",
                        "removed_5class_images": ";".join(removed_5_images),
                        "removed_5class_labels": ";".join(removed_5_labels),
                        "removed_2class_images": ";".join(removed_2_images),
                        "removed_2class_labels": ";".join(removed_2_labels),
                    }
                )
                continue
            patch_split = item["split"]
            if patch_split != expected_split:
                fail(f"{sid} nam o split {patch_split} trong ZIP, mong doi {expected_split}")
            label_text = archive.read(item["label"]).decode("utf-8")
            rows_5 = parse_label_text(label_text, expected_nc=5)
            rows_2 = map_label_rows_to_2class(rows_5)

            labels_5 = list_split_files(DATASET_5CLASS, expected_split, "labels", {".txt"})
            labels_2 = list_split_files(DATASET_2CLASS, expected_split, "labels", {".txt"})
            if sid not in labels_5 or sid not in labels_2:
                fail(f"Khong tim thay label hien tai cho {sid} trong dataset clean.")

            write_label(labels_5[sid], rows_5)
            write_label(labels_2[sid], rows_2)
            patch_rows.append(
                {
                    "image_id": sid,
                    "split": expected_split,
                    "patch_label": item["label"],
                    "clean_5class_label": str(labels_5[sid].relative_to(DATASET_5CLASS)),
                    "clean_2class_label": str(labels_2[sid].relative_to(DATASET_2CLASS)),
                    "old_source_class": "Sleeping",
                    "new_5class_ids": ";".join(str(int(float(row[0]))) for row in rows_5),
                    "new_5class_names": ";".join(CLASS_5_NAMES[int(float(row[0]))] for row in rows_5),
                    "new_2class_ids": ";".join(str(int(float(row[0]))) for row in rows_2),
                    "new_2class_names": ";".join(CLASS_2_NAMES[int(float(row[0]))] for row in rows_2),
                    "geometry_errors_after_patch": ";".join(error for row in rows_5 for error in geometry_errors(row)),
                }
            )

    log("Tiến hành xuất báo cáo các label đã patch và ảnh đã xoá.")
    write_csv(
        REPORT_DIR / "01_applied_label_patches.csv",
        patch_rows,
        [
            "image_id",
            "split",
            "patch_label",
            "clean_5class_label",
            "clean_2class_label",
            "old_source_class",
            "new_5class_ids",
            "new_5class_names",
            "new_2class_ids",
            "new_2class_names",
            "geometry_errors_after_patch",
        ],
    )
    write_csv(
        REPORT_DIR / "02_deleted_missing_patch_images.csv",
        deleted_rows,
        [
            "image_id",
            "expected_split",
            "reason",
            "removed_5class_images",
            "removed_5class_labels",
            "removed_2class_images",
            "removed_2class_labels",
        ],
    )

    log("Tiến hành kiểm tra dataset 5-class sau khi áp patch.")
    summary_5 = validate_dataset(DATASET_5CLASS, expected_nc=5)
    log("Tiến hành kiểm tra dataset 2-class sau khi áp patch.")
    summary_2 = validate_dataset(DATASET_2CLASS, expected_nc=2)

    write_csv(
        REPORT_DIR / "03_geometry_warnings_5class_after_patch.csv",
        summary_5["geometry_warnings"],
        ["split", "label", "line", "class_id", "errors"],
    )
    write_csv(
        REPORT_DIR / "04_geometry_warnings_2class_after_patch.csv",
        summary_2["geometry_warnings"],
        ["split", "label", "line", "class_id", "errors"],
    )

    summary_lines = []
    for title, summary, names in (
        ("5-class", summary_5, CLASS_5_NAMES),
        ("2-class", summary_2, CLASS_2_NAMES),
    ):
        summary_lines.append(f"[{title}]")
        for split in SPLITS:
            summary_lines.append(
                f"{split}: images={summary['image_counts'][split]}, labels={summary['label_counts'][split]}"
            )
            for class_id in sorted(summary["class_counts"][split]):
                summary_lines.append(f"  class {class_id} {names[class_id]}: {summary['class_counts'][split][class_id]}")
        total_images = sum(summary["image_counts"].values())
        total_labels = sum(summary["label_counts"].values())
        summary_lines.append(f"total: images={total_images}, labels={total_labels}")
        summary_lines.append(f"geometry_warnings={len(summary['geometry_warnings'])}")
        summary_lines.append("")
    (REPORT_DIR / "05_dataset_summary_after_patch.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    log("Tiến hành tạo lại ZIP 5-class.")
    zip_dataset(DATASET_5CLASS, ZIP_5CLASS)
    log("Tiến hành tạo lại ZIP 2-class.")
    zip_dataset(DATASET_2CLASS, ZIP_2CLASS)

    log(f"Da ap {len(patch_rows)} label patch, xoa {len(deleted_rows)} anh khong con trong ZIP patch.")
    log(f"Report: {REPORT_DIR}")
    log(f"ZIP 5-class: {ZIP_5CLASS}")
    log(f"ZIP 2-class: {ZIP_2CLASS}")


if __name__ == "__main__":
    main()
