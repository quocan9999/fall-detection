"""Randomly visualize pose keypoint indices from the clean two-class dataset."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from visualize_keypoint_indices import IMAGE_EXTENSIONS, KEYPOINT_COLORS, load_font


ROOT_DIR = Path(__file__).resolve().parent
ORIGIN_DATASET_DIR = ROOT_DIR / "origin-dataset"
OUTPUTS_DIR = ORIGIN_DATASET_DIR / "outputs"
DEFAULT_DATASET_DIR = OUTPUTS_DIR / "fall-detection-clean-fastdup-2class"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / Path(__file__).name
SPLITS = ("train", "valid", "test")
CLASS_BOX_COLORS = {
    "no_fall": (0, 205, 255),
    "fall": (255, 45, 45),
}


@dataclass(frozen=True)
class PoseObject:
    class_id: int
    class_name: str
    values: list[float]


@dataclass(frozen=True)
class Sample:
    split: str
    image_path: Path
    label_path: Path
    objects: list[PoseObject]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ve keypoint index tren 20 anh ngau nhien moi split cua dataset pose 2 class."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset pose 2 class chua data.yaml va train/valid/test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Thu muc xuat 60 overlay va manifest CSV.",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=20,
        help="So anh ngau nhien cho moi split; mac dinh 20.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed de lap lai dung bo mau; mac dinh 42.",
    )
    parser.add_argument(
        "--keep-old",
        action="store_true",
        help="Khong xoa thu muc overlay cu truoc khi xuat.",
    )
    return parser.parse_args()


def load_dataset_config(dataset_dir: Path) -> tuple[int, list[str]]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Khong tim thay data.yaml: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    kpt_shape = data.get("kpt_shape")
    if kpt_shape != [18, 3]:
        raise ValueError(f"Dataset khong co kpt_shape [18, 3]: {kpt_shape}")
    names_data = data.get("names", [])
    if isinstance(names_data, dict):
        names = [str(names_data[key]) for key in sorted(names_data, key=lambda key: int(key))]
    else:
        names = [str(name) for name in names_data]
    if names != ["no_fall", "fall"]:
        raise ValueError(f"Dataset khong dung schema 2 class no_fall/fall: {names}")
    return int(kpt_shape[0]), names


def find_image(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def parse_label(label_path: Path, kpt_count: int, names: list[str]) -> list[PoseObject] | None:
    expected_columns = 5 + kpt_count * 3
    objects = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != expected_columns:
            return None
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            return None
        if class_id < 0 or class_id >= len(names):
            return None
        objects.append(PoseObject(class_id=class_id, class_name=names[class_id], values=values))
    return objects or None


def collect_samples(dataset_dir: Path, split: str, kpt_count: int, names: list[str]) -> list[Sample]:
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Khong tim thay split {split} trong {dataset_dir}")
    samples = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        objects = parse_label(label_path, kpt_count, names)
        image_path = find_image(images_dir, label_path.stem)
        if objects is None or image_path is None:
            continue
        samples.append(
            Sample(
                split=split,
                image_path=image_path,
                label_path=label_path,
                objects=objects,
            )
        )
    return samples


def visible_keypoints(pose_object: PoseObject) -> int:
    keypoints = pose_object.values[4:]
    return sum(1 for index in range(0, len(keypoints), 3) if keypoints[index + 2] > 0)


def draw_overlay(sample: Sample, output_path: Path, kpt_count: int) -> None:
    image = Image.open(sample.image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    point_font = load_font(max(9, min(12, min(width, height) // 48)))
    tag_font = load_font(max(10, min(14, min(width, height) // 42)))
    header_font = load_font(max(13, min(18, min(width, height) // 35)))
    radius = max(3, min(5, min(width, height) // 145))

    for object_index, pose_object in enumerate(sample.objects, start=1):
        box_color = CLASS_BOX_COLORS.get(pose_object.class_name, (0, 220, 255))
        x_center, y_center, bbox_width, bbox_height = pose_object.values[:4]
        x0 = (x_center - bbox_width / 2) * width
        y0 = (y_center - bbox_height / 2) * height
        x1 = (x_center + bbox_width / 2) * width
        y1 = (y_center + bbox_height / 2) * height
        draw.rectangle((x0, y0, x1, y1), outline=box_color, width=3)
        tag = f"#{object_index} {pose_object.class_name}"
        tag_box = draw.textbbox((0, 0), tag, font=tag_font)
        tag_y = max(0, y0 - tag_box[3] - 5)
        draw.rectangle((max(0, x0), tag_y, max(0, x0) + tag_box[2] + 8, tag_y + tag_box[3] + 4), fill=(0, 0, 0))
        draw.text((max(0, x0) + 4, tag_y + 2), tag, font=tag_font, fill=box_color)

        keypoints = pose_object.values[4:]
        for index in range(kpt_count):
            x, y, visibility = keypoints[index * 3 : index * 3 + 3]
            if visibility <= 0:
                continue
            px = x * width
            py = y * height
            color = KEYPOINT_COLORS[index % len(KEYPOINT_COLORS)]
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=color,
                outline=(255, 255, 255),
                width=1,
            )
            draw.text(
                (px + radius + 4, py - radius - 2),
                str(index),
                font=point_font,
                fill=color,
                stroke_width=1,
                stroke_fill=(255, 255, 255),
            )

    classes = ", ".join(pose_object.class_name for pose_object in sample.objects)
    visible_total = sum(visible_keypoints(pose_object) for pose_object in sample.objects)
    header = f"{sample.split} | {classes} | objects: {len(sample.objects)} | visible: {visible_total}"
    header_box = draw.textbbox((0, 0), header, font=header_font)
    draw.rectangle((0, 0, min(width, header_box[2] + 18), header_box[3] + 14), fill=(0, 0, 0))
    draw.text((8, 6), header, font=header_font, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser()
    output_path = args.output.expanduser()
    if not dataset_path.is_absolute():
        dataset_path = ORIGIN_DATASET_DIR / dataset_path
    if not output_path.is_absolute():
        output_path = OUTPUTS_DIR / output_path
    dataset_dir = dataset_path.resolve()
    output_dir = output_path.resolve()
    if args.samples_per_split <= 0:
        raise ValueError("--samples-per-split phai lon hon 0.")

    print(f"[INFO] Tiến hành đọc cấu hình pose từ dataset 2 class: {dataset_dir}", flush=True)
    kpt_count, names = load_dataset_config(dataset_dir)
    rng = random.Random(args.seed)
    selected: dict[str, list[Sample]] = {}
    print("[INFO] Tiến hành chọn ngẫu nhiên ảnh pose hợp lệ cho từng split.", flush=True)
    for split in SPLITS:
        available = collect_samples(dataset_dir, split, kpt_count, names)
        if len(available) < args.samples_per_split:
            raise RuntimeError(
                f"Split {split} chi co {len(available)} anh pose hop le, "
                f"khong du {args.samples_per_split} mau."
            )
        selected[split] = rng.sample(available, args.samples_per_split)

    if output_dir.exists() and not args.keep_old:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_requested = args.samples_per_split * len(SPLITS)
    print(f"[INFO] Tiến hành vẽ keypoint trên {total_requested} ảnh mẫu của dataset 2 class.", flush=True)
    manifest_path = output_dir / "random_keypoint_samples_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "sample_in_split",
                "object_count",
                "classes",
                "visible_keypoints_total",
                "source_image",
                "source_label",
                "overlay_image",
                "seed",
            ],
        )
        writer.writeheader()
        total = 0
        for split in SPLITS:
            split_dir = output_dir / split
            for sample_index, sample in enumerate(selected[split], start=1):
                output_path = split_dir / f"{sample_index:02d}_{split}_{sample.image_path.stem}_indexed.jpg"
                draw_overlay(sample, output_path, kpt_count)
                writer.writerow(
                    {
                        "split": split,
                        "sample_in_split": sample_index,
                        "object_count": len(sample.objects),
                        "classes": ";".join(pose_object.class_name for pose_object in sample.objects),
                        "visible_keypoints_total": sum(visible_keypoints(obj) for obj in sample.objects),
                        "source_image": sample.image_path,
                        "source_label": sample.label_path,
                        "overlay_image": output_path,
                        "seed": args.seed,
                    }
                )
                total += 1
                print(f"[OK] {output_path}")

    print(f"\nDa xuat {total} anh ({args.samples_per_split} anh/split) vao: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
