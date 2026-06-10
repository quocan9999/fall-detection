"""Draw YOLO pose keypoint indices on clear samples for flip_idx verification."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import yaml


ROOT_DIR = Path(__file__).resolve().parent
ORIGIN_DATASET_DIR = ROOT_DIR / "origin-dataset"
OUTPUTS_DIR = ORIGIN_DATASET_DIR / "outputs"
DEFAULT_DATASET_DIR = OUTPUTS_DIR / "fall-detection-clean-fastdup"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / Path(__file__).name
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
KEYPOINT_COLORS = (
    (255, 32, 32),
    (255, 128, 0),
    (255, 214, 10),
    (120, 200, 0),
    (0, 190, 70),
    (0, 196, 196),
    (0, 130, 255),
    (74, 70, 255),
    (152, 62, 255),
    (232, 45, 206),
    (255, 70, 132),
    (160, 72, 20),
    (240, 155, 80),
    (105, 180, 155),
    (50, 110, 170),
    (115, 90, 170),
    (220, 100, 145),
    (110, 110, 110),
)


@dataclass(frozen=True)
class Candidate:
    split: str
    image_path: Path
    label_path: Path
    class_id: int
    class_name: str
    visible_keypoints: int
    bbox_area: float
    values: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ve so index 0..N-1 len cac keypoint cua anh YOLO pose."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Thu muc dataset clean chua data.yaml va cac split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Thu muc xuat anh da ve index va manifest CSV.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test", "all"),
        default="train",
        help="Split de chon anh xem; mac dinh train.",
    )
    parser.add_argument(
        "--class-name",
        default="Standing",
        help="Uu tien class de anh de nhin bo phan trai/phai; mac dinh Standing.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="So anh can xuat; mac dinh 12.",
    )
    parser.add_argument(
        "--keep-old",
        action="store_true",
        help="Khong xoa cac overlay cu trong thu muc output truoc khi xuat.",
    )
    return parser.parse_args()


def load_dataset_config(dataset_dir: Path) -> tuple[int, list[str]]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Khong tim thay data.yaml: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    kpt_shape = data.get("kpt_shape")
    if not isinstance(kpt_shape, list) or not kpt_shape:
        raise ValueError("data.yaml khong co kpt_shape hop le.")

    names_data = data.get("names", [])
    if isinstance(names_data, dict):
        names = [str(names_data[key]) for key in sorted(names_data, key=lambda x: int(x))]
    else:
        names = [str(name) for name in names_data]
    return int(kpt_shape[0]), names


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        image_path = images_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path
    return None


def parse_single_object_label(
    label_path: Path, kpt_count: int
) -> tuple[int, list[float]] | None:
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        return None

    parts = lines[0].split()
    expected_columns = 5 + kpt_count * 3
    if len(parts) != expected_columns:
        return None

    try:
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    return class_id, values


def collect_candidates(
    dataset_dir: Path, splits: list[str], kpt_count: int, names: list[str]
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for split in splits:
        labels_dir = dataset_dir / split / "labels"
        images_dir = dataset_dir / split / "images"
        if not labels_dir.exists() or not images_dir.exists():
            continue
        for label_path in labels_dir.glob("*.txt"):
            parsed = parse_single_object_label(label_path, kpt_count)
            if parsed is None:
                continue
            class_id, values = parsed
            image_path = find_image(images_dir, label_path.stem)
            if image_path is None:
                continue
            keypoint_values = values[4:]
            visible = sum(
                1
                for i in range(0, len(keypoint_values), 3)
                if keypoint_values[i + 2] > 0
            )
            bbox_area = values[2] * values[3]
            class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            candidates.append(
                Candidate(
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    class_id=class_id,
                    class_name=class_name,
                    visible_keypoints=visible,
                    bbox_area=bbox_area,
                    values=values,
                )
            )
    return candidates


def choose_candidates(
    candidates: list[Candidate], preferred_class_name: str, samples: int
) -> list[Candidate]:
    preferred = [
        candidate
        for candidate in candidates
        if candidate.class_name.casefold() == preferred_class_name.casefold()
    ]
    pool = preferred if preferred else candidates
    return sorted(
        pool,
        key=lambda item: (item.visible_keypoints, item.bbox_area),
        reverse=True,
    )[:samples]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_paths = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for font_path in font_paths:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def draw_overlay(candidate: Candidate, output_path: Path, kpt_count: int) -> None:
    image = Image.open(candidate.image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    point_font = load_font(max(9, min(12, min(width, height) // 48)))
    header_font = load_font(max(16, min(width, height) // 35))

    x_center, y_center, bbox_width, bbox_height = candidate.values[:4]
    x0 = (x_center - bbox_width / 2) * width
    y0 = (y_center - bbox_height / 2) * height
    x1 = (x_center + bbox_width / 2) * width
    y1 = (y_center + bbox_height / 2) * height
    draw.rectangle((x0, y0, x1, y1), outline=(0, 220, 255), width=3)

    keypoints = candidate.values[4:]
    radius = max(3, min(5, min(width, height) // 145))
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

    header = (
        f"{candidate.split} | {candidate.class_name} | "
        f"visible keypoints: {candidate.visible_keypoints}/{kpt_count}"
    )
    header_box = draw.textbbox((0, 0), header, font=header_font, stroke_width=1)
    draw.rectangle(
        (0, 0, min(width, header_box[2] + 18), header_box[3] + 14),
        fill=(0, 0, 0),
    )
    draw.text(
        (8, 6),
        header,
        font=header_font,
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
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
    if args.samples <= 0:
        raise ValueError("--samples phai lon hon 0.")

    print(f"[INFO] Tiến hành đọc cấu hình pose từ dataset 5 class: {dataset_dir}", flush=True)
    kpt_count, names = load_dataset_config(dataset_dir)
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    print("[INFO] Tiến hành thu thập ảnh có keypoint hợp lệ để kiểm tra bằng mắt.", flush=True)
    candidates = collect_candidates(dataset_dir, splits, kpt_count, names)
    selected = choose_candidates(candidates, args.class_name, args.samples)
    if not selected:
        raise RuntimeError("Khong tim thay anh mot nguoi co label pose hop le de ve index.")

    if output_dir.exists() and not args.keep_old:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Tiến hành vẽ index keypoint trên {len(selected)} ảnh mẫu.", flush=True)
    manifest_path = output_dir / "kpt_index_check_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "sample",
                "split",
                "class_id",
                "class_name",
                "visible_keypoints",
                "source_image",
                "source_label",
                "overlay_image",
            ],
        )
        writer.writeheader()
        for sample_index, candidate in enumerate(selected, start=1):
            output_path = output_dir / (
                f"{sample_index:02d}_{candidate.split}_{candidate.class_name}_"
                f"{candidate.image_path.stem}_indexed.jpg"
            )
            draw_overlay(candidate, output_path, kpt_count)
            writer.writerow(
                {
                    "sample": sample_index,
                    "split": candidate.split,
                    "class_id": candidate.class_id,
                    "class_name": candidate.class_name,
                    "visible_keypoints": candidate.visible_keypoints,
                    "source_image": candidate.image_path,
                    "source_label": candidate.label_path,
                    "overlay_image": output_path,
                }
            )
            print(f"[OK] {output_path}")

    print(f"\nDa xuat {len(selected)} anh vao: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("Mo cac anh *_indexed.jpg va ghi lai vi tri index cua tung bo phan.")


if __name__ == "__main__":
    main()
