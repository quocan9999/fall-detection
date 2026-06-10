"""Export image-level and object-level statistics for one YOLO pose class."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
ORIGIN_DATASET_DIR = ROOT / "origin-dataset"
OUTPUTS_DIR = ORIGIN_DATASET_DIR / "outputs"
DEFAULT_DATASET = OUTPUTS_DIR / "fall-detection-clean-fastdup"
DEFAULT_OUTPUT = OUTPUTS_DIR / Path(__file__).name
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thong ke anh co chua mot class trong YOLO pose dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Dataset nguon 5 class.")
    parser.add_argument("--class-name", default="Sleeping", help="Ten class can thong ke.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Thu muc report dau ra.")
    return parser.parse_args()


def load_class_id(dataset: Path, requested_name: str) -> tuple[int, str]:
    config_path = dataset / "data.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    names_data = data.get("names", [])
    if isinstance(names_data, dict):
        names = {int(key): str(value) for key, value in names_data.items()}
    else:
        names = {index: str(value) for index, value in enumerate(names_data)}
    for class_id, class_name in names.items():
        if class_name.casefold() == requested_name.casefold():
            return class_id, class_name
    raise ValueError(f"Khong tim thay class {requested_name!r} trong {config_path}. Co cac class: {list(names.values())}")


def find_image(images_dir: Path, relative_label: Path) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / relative_label.parent / f"{relative_label.stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Khong tim thay anh tuong ung cho label: {relative_label}")


def short_image_name(filename: str) -> str:
    match = re.search(r"((?:img|image)_\d+)", filename, flags=re.IGNORECASE)
    return match.group(1) if match else Path(filename).stem


def image_sort_key(row: dict[str, str | int]) -> tuple[int, int, str]:
    split_rank = SPLITS.index(str(row["split"]))
    match = re.search(r"(\d+)$", str(row["image_name"]))
    number = int(match.group(1)) if match else 10**12
    return split_rank, number, str(row["image_name"])


def collect_rows(dataset: Path, class_id: int) -> list[dict[str, str | int]]:
    rows = []
    for split in SPLITS:
        labels_dir = dataset / split / "labels"
        images_dir = dataset / split / "images"
        for label_path in labels_dir.rglob("*.txt"):
            objects = 0
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    line_class_id = int(float(parts[0]))
                except ValueError:
                    continue
                if line_class_id == class_id:
                    objects += 1
            if objects == 0:
                continue
            relative_label = label_path.relative_to(labels_dir)
            image_path = find_image(images_dir, relative_label)
            rows.append(
                {
                    "split": split,
                    "image_name": short_image_name(image_path.name),
                    "full_filename": image_path.name,
                    "objects_of_class": objects,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                }
            )
    return sorted(rows, key=image_sort_key)


def write_reports(output: Path, class_id: int, class_name: str, rows: list[dict[str, str | int]]) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    csv_path = output / "sleeping_images_by_split.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "split", "image_name", "full_filename", "objects_of_class", "image_path", "label_path"
        ])
        writer.writeheader()
        writer.writerows(rows)

    image_counts = Counter(str(row["split"]) for row in rows)
    object_counts = Counter()
    for row in rows:
        object_counts[str(row["split"])] += int(row["objects_of_class"])
    summary_lines = [
        f"Class: {class_name} (id={class_id})",
        "",
        "split,images,objects",
    ]
    for split in SPLITS:
        summary_lines.append(f"{split},{image_counts[split]},{object_counts[split]}")
    summary_lines.append(f"total,{len(rows)},{sum(object_counts.values())}")
    (output / "sleeping_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    for split in SPLITS:
        names_path = output / f"{split}_image_names.txt"
        names = [str(row["image_name"]) for row in rows if row["split"] == split]
        names_path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser()
    output_path = args.output.expanduser()
    if not dataset_path.is_absolute():
        dataset_path = ORIGIN_DATASET_DIR / dataset_path
    if not output_path.is_absolute():
        output_path = OUTPUTS_DIR / output_path
    dataset = dataset_path.resolve()
    output = output_path.resolve()
    print(f"[INFO] Tiến hành xác định class cần thống kê trong dataset: {dataset}", flush=True)
    class_id, class_name = load_class_id(dataset, args.class_name)
    print(f"[INFO] Tiến hành thống kê ảnh và object của class {class_name} theo train/valid/test.", flush=True)
    rows = collect_rows(dataset, class_id)
    print(f"[INFO] Tiến hành xuất report thống kê class vào: {output}", flush=True)
    write_reports(output, class_id, class_name, rows)

    split_images = Counter(str(row["split"]) for row in rows)
    split_objects = Counter()
    for row in rows:
        split_objects[str(row["split"])] += int(row["objects_of_class"])
    print(f"Class: {class_name} (id={class_id})")
    for split in SPLITS:
        print(f"{split}: {split_images[split]} images, {split_objects[split]} objects")
    print(f"total: {len(rows)} images, {sum(split_objects.values())} objects")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
