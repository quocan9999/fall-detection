from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
ORIGIN_DATASET_DIR = ROOT / "origin-dataset"
OUTPUT_DIR = ORIGIN_DATASET_DIR / "outputs"
SCRIPT_OUTPUT_DIR = OUTPUT_DIR / Path(__file__).name
DEFAULT_OUTPUT_BASE = OUTPUT_DIR
REPORTS_DIR = SCRIPT_OUTPUT_DIR / "reports"
WORK_DIR = SCRIPT_OUTPUT_DIR / "work"
DEFAULT_SOURCE = DEFAULT_OUTPUT_BASE / "fall-detection-clean-fastdup"
DEFAULT_OUTPUT_NAME = "fall-detection-clean-fastdup-2class"
PROCESS_REPORTS_DIR = OUTPUT_DIR / "process_dataset_fastdup.py" / "reports"
FINAL_MANIFEST = PROCESS_REPORTS_DIR / "11_final_manifest_after_rebalance.csv"
FINAL_LEAK_REPORT = PROCESS_REPORTS_DIR / "12_final_split_duplicate_leak_check.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}
SPLITS = ("train", "valid", "test")
SPLIT_RATIOS = {"train": 0.70, "valid": 0.20, "test": 0.10}
EXPECTED_KPT_SHAPE = [18, 3]
EXPECTED_FLIP_IDX = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
SOURCE_CLASS_NAMES = ["Sitting", "Sleeping", "Standing", "Walking", "falling"]
OUTPUT_CLASS_NAMES = ["no_fall", "fall"]
CLASS_REMAP = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def fail(message: str) -> None:
    print(f"[LOI] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict:
    if not path.exists():
        fail(f"Khong tim thay data.yaml: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_integer_class(token: str) -> int | None:
    try:
        value = float(token)
    except ValueError:
        return None
    if not value.is_integer():
        return None
    return int(value)


def list_files(directory: Path, suffixes: set[str] | None = None) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    result = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        result[path.relative_to(directory).as_posix()] = path
    return result


def collect_inventory(dataset_root: Path) -> dict[str, dict[str, dict[str, Path]]]:
    inventory = {}
    for split in SPLITS:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            fail(f"Dataset thieu thu muc {split}/images hoac {split}/labels: {dataset_root}")
        inventory[split] = {
            "images": list_files(images_dir, IMAGE_EXTS),
            "labels": list_files(labels_dir, {".txt"}),
        }
    return inventory


def validate_source_yaml(source_root: Path) -> dict:
    data = read_yaml(source_root / "data.yaml")
    if int(data.get("nc", -1)) != 5 or list(data.get("names", [])) != SOURCE_CLASS_NAMES:
        fail(
            "Dataset nguon khong dung schema 5 class mong doi: "
            f"nc={data.get('nc')}, names={data.get('names')}"
        )
    if list(data.get("kpt_shape", [])) != EXPECTED_KPT_SHAPE:
        fail(f"kpt_shape nguon khong dung {EXPECTED_KPT_SHAPE}: {data.get('kpt_shape')}")
    if list(data.get("flip_idx", [])) != EXPECTED_FLIP_IDX:
        fail("flip_idx nguon chua dung mapping da xac minh cho fliplr.")
    return data


def snapshot_source(source_root: Path, inventory: dict) -> dict[str, dict[str, str]]:
    paths = {"data.yaml": ("yaml", source_root / "data.yaml")}
    for split in SPLITS:
        for rel, path in inventory[split]["images"].items():
            paths[f"{split}/images/{rel}"] = ("image", path)
        for rel, path in inventory[split]["labels"].items():
            paths[f"{split}/labels/{rel}"] = ("label", path)
    snapshot = {}
    for rel, (kind, path) in tqdm(paths.items(), desc="Hash source", unit="file"):
        snapshot[rel] = {"kind": kind, "sha256": sha256_file(path)}
    return snapshot


def geometry_errors(nums: list[float], kpt_dims: int) -> list[str]:
    errors = []
    x, y, w, h = nums[1:5]
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        errors.append("bbox_normalized_invalid")
    if x - w / 2 < -0.02 or y - h / 2 < -0.02 or x + w / 2 > 1.02 or y + h / 2 > 1.02:
        errors.append("bbox_outside_image")
    bad_keypoints = 0
    keypoints = nums[5:]
    for start in range(0, len(keypoints), kpt_dims):
        kx, ky = keypoints[start], keypoints[start + 1]
        visibility = keypoints[start + 2] if kpt_dims >= 3 else 1.0
        if visibility > 0 and not (0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0):
            bad_keypoints += 1
    if bad_keypoints:
        errors.append(f"visible_keypoints_outside_image={bad_keypoints}")
    return errors


def inspect_dataset(dataset_root: Path, expected_nc: int) -> dict:
    data = read_yaml(dataset_root / "data.yaml")
    kpt_shape = list(data.get("kpt_shape", []))
    if kpt_shape != EXPECTED_KPT_SHAPE:
        fail(f"Dataset co kpt_shape khong dung {EXPECTED_KPT_SHAPE}: {dataset_root}")
    expected_values = 5 + kpt_shape[0] * kpt_shape[1]
    inventory = collect_inventory(dataset_root)
    result = {
        "inventory": inventory,
        "split_images": {},
        "split_labels": {},
        "class_counts": {split: Counter() for split in SPLITS},
        "label_value_lengths": Counter(),
        "missing_labels": [],
        "orphan_labels": [],
        "corrupt_images": [],
        "bad_label_lines": [],
        "geometry_warnings": [],
        "expected_values": expected_values,
    }

    for split in SPLITS:
        images = inventory[split]["images"]
        labels = inventory[split]["labels"]
        result["split_images"][split] = len(images)
        result["split_labels"][split] = len(labels)
        expected_labels = {str(Path(rel).with_suffix(".txt")).replace("\\", "/") for rel in images}
        label_paths = set(labels)
        result["missing_labels"].extend(f"{split}/{rel}" for rel in sorted(expected_labels - label_paths))
        result["orphan_labels"].extend(f"{split}/{rel}" for rel in sorted(label_paths - expected_labels))

        for rel, image_path in tqdm(images.items(), desc=f"Verify {split} images", unit="image"):
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                result["corrupt_images"].append({"image": f"{split}/{rel}", "error": str(exc)})

        for rel, label_path in labels.items():
            with label_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    result["label_value_lengths"][len(parts)] += 1
                    class_id = parse_integer_class(parts[0])
                    if len(parts) != expected_values or class_id is None or not 0 <= class_id < expected_nc:
                        result["bad_label_lines"].append(
                            {"label": f"{split}/{rel}", "line": line_number, "num_values": len(parts), "class": class_id}
                        )
                        continue
                    try:
                        nums = [float(value) for value in parts]
                    except ValueError:
                        result["bad_label_lines"].append(
                            {"label": f"{split}/{rel}", "line": line_number, "num_values": len(parts), "class": class_id}
                        )
                        continue
                    result["class_counts"][split][class_id] += 1
                    errors = geometry_errors(nums, kpt_shape[1])
                    if errors:
                        result["geometry_warnings"].append(
                            {"split": split, "label": rel, "line": line_number, "class_id": class_id, "errors": errors}
                        )
    return result


def require_trainable_structure(result: dict, name: str) -> None:
    if result["missing_labels"]:
        fail(f"{name} thieu label: {len(result['missing_labels'])} file.")
    if result["orphan_labels"]:
        fail(f"{name} co label khong co anh: {len(result['orphan_labels'])} file.")
    if result["corrupt_images"]:
        fail(f"{name} co anh hong: {len(result['corrupt_images'])} file.")
    if result["bad_label_lines"]:
        fail(f"{name} co label sai format/class: {len(result['bad_label_lines'])} dong.")
    if set(result["label_value_lengths"]) != {59}:
        fail(f"{name} khong duy tri dung 59 gia tri moi dong pose: {dict(result['label_value_lengths'])}")


def expected_split_counts(total_images: int) -> dict[str, int]:
    valid = round(total_images * SPLIT_RATIOS["valid"])
    test = round(total_images * SPLIT_RATIOS["test"])
    return {"train": total_images - valid - test, "valid": valid, "test": test}


def require_split_ratio(result: dict, name: str) -> dict[str, int]:
    total = sum(result["split_images"].values())
    targets = expected_split_counts(total)
    if result["split_images"] != targets:
        fail(f"{name} khong dung split 70/20/10: actual={result['split_images']}, expected={targets}")
    return targets


def external_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw_path)
    if os.name == "nt" and match:
        return Path(f"{match.group(1).upper()}:/{match.group(2)}")
    return path


def verify_clean_lineage(source_result: dict, source_snapshot: dict) -> tuple[Counter, dict]:
    if not FINAL_MANIFEST.exists() or not FINAL_LEAK_REPORT.exists():
        fail("Thieu report 11 hoac 12 tu pipeline clean; khong the chung minh trang thai dataleak.")
    with FINAL_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    source_keys = {
        (split, rel): path
        for split in SPLITS
        for rel, path in source_result["inventory"][split]["images"].items()
    }
    manifest_by_key = {(row["final_split"], row["rel_image"]): row for row in manifest_rows}
    if len(manifest_rows) != len(manifest_by_key) or set(manifest_by_key) != set(source_keys):
        fail("Inventory cua dataset clean khong con khop reports/11_final_manifest_after_rebalance.csv.")

    compared_to_analyzed_images = 0
    unavailable_analyzed_images = 0
    for key, source_path in tqdm(source_keys.items(), desc="Verify clean lineage", unit="image"):
        row = manifest_by_key[key]
        analyzed_path = external_path(row["image_path"])
        if not analyzed_path.exists():
            unavailable_analyzed_images += 1
            continue
        source_rel = f"{key[0]}/images/{key[1]}"
        if sha256_file(analyzed_path) != source_snapshot[source_rel]["sha256"]:
            fail(f"Anh clean khong trung voi anh da duoc scan leak: {source_path}")
        compared_to_analyzed_images += 1

    with FINAL_LEAK_REPORT.open("r", encoding="utf-8-sig", newline="") as handle:
        leak_counts = Counter(row.get("leak_type", "") for row in csv.DictReader(handle))
    critical = leak_counts.get("CRITICAL_train_valid", 0) + leak_counts.get("CRITICAL_train_test", 0)
    if critical:
        fail(f"Report clean con critical leak train-valid/train-test: {critical}")
    lineage_stats = {
        "manifest_images": len(manifest_rows),
        "images_compared_to_analyzed_copy": compared_to_analyzed_images,
        "analyzed_copy_unavailable": unavailable_analyzed_images,
    }
    return leak_counts, lineage_stats


def write_binary_yaml(source_yaml: dict, output_root: Path) -> None:
    output_yaml = dict(source_yaml)
    output_yaml["train"] = "train/images"
    output_yaml["val"] = "valid/images"
    output_yaml["test"] = "test/images"
    output_yaml["kpt_shape"] = EXPECTED_KPT_SHAPE
    output_yaml["flip_idx"] = EXPECTED_FLIP_IDX
    output_yaml["nc"] = 2
    output_yaml["names"] = OUTPUT_CLASS_NAMES
    with (output_root / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output_yaml, handle, sort_keys=False, allow_unicode=True)


def remap_label_file(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                converted.append(line)
                continue
            match = re.match(r"^(\s*)(\S+)(.*)$", line, flags=re.DOTALL)
            if not match:
                fail(f"Khong doc duoc label {source_path}, dong {line_number}.")
            source_id = parse_integer_class(match.group(2))
            if source_id not in CLASS_REMAP:
                fail(f"Class ID ngoai mapping trong {source_path}, dong {line_number}: {match.group(2)}")
            converted.append(f"{match.group(1)}{CLASS_REMAP[source_id]}{match.group(3)}")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.writelines(converted)


def convert_dataset(source_root: Path, output_root: Path, source_yaml: dict, source_result: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)
        for rel, image_path in tqdm(
            source_result["inventory"][split]["images"].items(), desc=f"Copy {split} images", unit="image"
        ):
            target = output_root / split / "images" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target)
        for rel, label_path in tqdm(
            source_result["inventory"][split]["labels"].items(), desc=f"Remap {split} labels", unit="label"
        ):
            remap_label_file(label_path, output_root / split / "labels" / rel)
    write_binary_yaml(source_yaml, output_root)


def verify_conversion(source_result: dict, output_result: dict, source_snapshot: dict, output_root: Path) -> None:
    if source_result["split_images"] != output_result["split_images"]:
        fail("So luong anh output khong giong source.")
    if source_result["split_labels"] != output_result["split_labels"]:
        fail("So luong label output khong giong source.")

    for split in SPLITS:
        source_images = source_result["inventory"][split]["images"]
        output_images = output_result["inventory"][split]["images"]
        if set(source_images) != set(output_images):
            fail(f"Danh sach anh split {split} bi thay doi.")
        for rel, output_path in tqdm(output_images.items(), desc=f"Compare {split} images", unit="image"):
            source_rel = f"{split}/images/{rel}"
            if sha256_file(output_path) != source_snapshot[source_rel]["sha256"]:
                fail(f"Anh output khong giong byte nguon: {split}/images/{rel}")

        for rel, source_label in source_result["inventory"][split]["labels"].items():
            output_label = output_result["inventory"][split]["labels"].get(rel)
            if output_label is None:
                fail(f"Output thieu label: {split}/labels/{rel}")
            source_lines = [line.split() for line in source_label.read_text(encoding="utf-8").splitlines() if line.strip()]
            output_lines = [line.split() for line in output_label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(source_lines) != len(output_lines):
                fail(f"So object thay doi sau remap: {split}/labels/{rel}")
            for line_number, (source_parts, output_parts) in enumerate(zip(source_lines, output_lines), start=1):
                source_id = parse_integer_class(source_parts[0])
                output_id = parse_integer_class(output_parts[0])
                if output_id != CLASS_REMAP.get(source_id):
                    fail(f"Class remap sai: {split}/labels/{rel}, dong {line_number}")
                if source_parts[1:] != output_parts[1:]:
                    fail(f"BBox/keypoint bi thay doi: {split}/labels/{rel}, dong {line_number}")

    source_geometry = {
        (row["split"], row["label"], row["line"], tuple(row["errors"])) for row in source_result["geometry_warnings"]
    }
    output_geometry = {
        (row["split"], row["label"], row["line"], tuple(row["errors"])) for row in output_result["geometry_warnings"]
    }
    if source_geometry != output_geometry:
        fail("Output phat sinh hoac lam mat geometry warning so voi source.")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    report_root: Path,
    source_root: Path,
    output_root: Path,
    source_result: dict,
    output_result: dict,
    snapshot_before: dict,
    snapshot_after: dict,
    leak_counts: Counter,
    lineage_stats: dict,
    targets: dict[str, int],
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    integrity_rows = []
    for rel, before in snapshot_before.items():
        after = snapshot_after.get(rel, {})
        integrity_rows.append(
            {
                "relative_path": rel,
                "kind": before["kind"],
                "sha256_before": before["sha256"],
                "sha256_after": after.get("sha256", ""),
                "unchanged": before["sha256"] == after.get("sha256"),
            }
        )
    write_csv(
        report_root / "source_integrity.csv",
        ["relative_path", "kind", "sha256_before", "sha256_after", "unchanged"],
        integrity_rows,
    )

    remap_rows = []
    for split in SPLITS:
        output_totals = output_result["class_counts"][split]
        for source_id, source_name in enumerate(SOURCE_CLASS_NAMES):
            output_id = CLASS_REMAP[source_id]
            remap_rows.append(
                {
                    "split": split,
                    "source_class_id": source_id,
                    "source_class_name": source_name,
                    "source_objects": source_result["class_counts"][split].get(source_id, 0),
                    "output_class_id": output_id,
                    "output_class_name": OUTPUT_CLASS_NAMES[output_id],
                    "output_total_objects": output_totals.get(output_id, 0),
                }
            )
    write_csv(
        report_root / "class_remap_summary.csv",
        [
            "split",
            "source_class_id",
            "source_class_name",
            "source_objects",
            "output_class_id",
            "output_class_name",
            "output_total_objects",
        ],
        remap_rows,
    )

    output_warning_by_key = {
        (row["split"], row["label"], row["line"], tuple(row["errors"])): row
        for row in output_result["geometry_warnings"]
    }
    inherited_rows = []
    for source_warning in source_result["geometry_warnings"]:
        key = (
            source_warning["split"],
            source_warning["label"],
            source_warning["line"],
            tuple(source_warning["errors"]),
        )
        output_warning = output_warning_by_key[key]
        inherited_rows.append(
            {
                "split": source_warning["split"],
                "label": source_warning["label"],
                "line": source_warning["line"],
                "source_class_id": source_warning["class_id"],
                "output_class_id": output_warning["class_id"],
                "errors": ";".join(source_warning["errors"]),
            }
        )
    write_csv(
        report_root / "inherited_geometry_warnings.csv",
        ["split", "label", "line", "source_class_id", "output_class_id", "errors"],
        inherited_rows,
    )

    pose_lines = [
        f"Source dataset: {source_root}",
        f"Output dataset: {output_root}",
        "Format: YOLO pose/keypoint",
        f"kpt_shape: {EXPECTED_KPT_SHAPE}",
        "Expected values per object line: 59",
        f"flip_idx: {EXPECTED_FLIP_IDX}",
        "nc: 2",
        f"names: {OUTPUT_CLASS_NAMES}",
    ]
    for split in SPLITS:
        counts = output_result["class_counts"][split]
        pose_lines.append(
            f"[{split}] images={output_result['split_images'][split]}, labels={output_result['split_labels'][split]}, "
            f"no_fall={counts.get(0, 0)}, fall={counts.get(1, 0)}"
        )
    pose_lines += [
        f"Missing labels: {len(output_result['missing_labels'])}",
        f"Orphan labels: {len(output_result['orphan_labels'])}",
        f"Corrupt images: {len(output_result['corrupt_images'])}",
        f"Bad label format/class lines: {len(output_result['bad_label_lines'])}",
        f"Inherited geometry warnings: {len(output_result['geometry_warnings'])}",
        "Result: PASS - output remains a trainable YOLO pose dataset; geometry warnings are inherited unchanged.",
    ]
    (report_root / "pose_validation.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")

    ratio_lines = ["Split ratio validation (image counts):"]
    total = sum(output_result["split_images"].values())
    for split in SPLITS:
        count = output_result["split_images"][split]
        ratio_lines.append(
            f"{split}: actual={count}, target={targets[split]}, percent={count / total * 100:.3f}%"
        )
    ratio_lines.append("Result: PASS - split membership is unchanged and satisfies the 70/20/10 target counts.")
    (report_root / "split_ratio_check.txt").write_text("\n".join(ratio_lines) + "\n", encoding="utf-8")

    leak_lines = [
        "Dataleak verification for binary dataset:",
        "Method: binary images and split membership are byte-identical to the clean source; source inventory matches its final manifest.",
        f"Manifest images matched: {lineage_stats['manifest_images']}",
        f"Images compared to retained analyzed copy: {lineage_stats['images_compared_to_analyzed_copy']}",
        f"Analyzed copy unavailable for byte comparison: {lineage_stats['analyzed_copy_unavailable']}",
        f"CRITICAL train-valid: {leak_counts.get('CRITICAL_train_valid', 0)}",
        f"CRITICAL train-test: {leak_counts.get('CRITICAL_train_test', 0)}",
        f"Duplicate valid-test only: {leak_counts.get('valid_test_only', 0)}",
        "Result: PASS - class remapping introduces no train-valid or train-test leak; leak status is inherited from the clean-source report.",
    ]
    (report_root / "leak_verification.txt").write_text("\n".join(leak_lines) + "\n", encoding="utf-8")


def replace_directory(source: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.rename(backup)
    try:
        source.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert clean YOLO pose dataset from 5 classes to 2 classes.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Path to the existing clean 5-class dataset.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="New folder name created next to the source dataset.")
    args = parser.parse_args()

    source_path = Path(args.source).expanduser()
    if not source_path.is_absolute():
        source_path = ORIGIN_DATASET_DIR / source_path
    source_root = source_path.resolve()
    if not source_root.is_dir():
        fail(f"Dataset source khong ton tai: {source_root}")
    output_name = args.output_name.strip()
    if not output_name or Path(output_name).name != output_name:
        fail("--output-name chi duoc la ten folder, khong duoc chua path.")
    output_root = (source_root.parent / output_name).resolve()
    if source_root == output_root:
        fail("Output dataset phai khac source dataset 5 class.")

    log(f"Source 5 class: {source_root}")
    log(f"Output 2 class: {output_root}")
    log("Tiến hành kiểm tra data.yaml và cấu trúc pose của dataset clean 5 class.")
    source_yaml = validate_source_yaml(source_root)
    source_result = inspect_dataset(source_root, expected_nc=5)
    require_trainable_structure(source_result, "Dataset source")
    targets = require_split_ratio(source_result, "Dataset source")
    log("Tiến hành chụp kiểm kê dataset nguồn và xác minh report dataleak từ pipeline clean.")
    source_snapshot_before = snapshot_source(source_root, source_result["inventory"])
    leak_counts, lineage_stats = verify_clean_lineage(source_result, source_snapshot_before)

    stage_root = WORK_DIR / f".convert_2class_stage_{os.getpid()}"
    stage_dataset = stage_root / output_name
    stage_reports = stage_root / "reports"
    stage_zip = stage_root / f"{output_name}.zip"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    try:
        log("Tiến hành chuyển label từ 5 class sang 2 class, giữ nguyên ảnh và keypoint.")
        convert_dataset(source_root, stage_dataset, source_yaml, source_result)
        output_yaml = read_yaml(stage_dataset / "data.yaml")
        if (
            int(output_yaml.get("nc", -1)) != 2
            or list(output_yaml.get("names", [])) != OUTPUT_CLASS_NAMES
            or list(output_yaml.get("flip_idx", [])) != EXPECTED_FLIP_IDX
        ):
            fail("data.yaml output khong dung schema 2 class/flip_idx.")

        log("Tiến hành kiểm tra dataset 2 class còn đúng định dạng YOLO pose và tỷ lệ split.")
        output_result = inspect_dataset(stage_dataset, expected_nc=2)
        require_trainable_structure(output_result, "Dataset output")
        output_targets = require_split_ratio(output_result, "Dataset output")
        if targets != output_targets:
            fail("Split target output khong trung voi source.")
        log("Tiến hành đối chiếu ảnh, bbox và keypoint giữa dataset 5 class và 2 class.")
        verify_conversion(source_result, output_result, source_snapshot_before, stage_dataset)

        log("Tiến hành kiểm tra dataset 5 class không bị thay đổi trong quá trình conversion.")
        source_snapshot_after = snapshot_source(source_root, source_result["inventory"])
        if source_snapshot_before != source_snapshot_after:
            fail("Dataset source 5 class bi thay doi trong luc conversion; huy publish output.")

        log("Tiến hành xuất báo cáo kiểm tra dataset 2 class.")
        write_reports(
            stage_reports,
            source_root,
            output_root,
            source_result,
            output_result,
            source_snapshot_before,
            source_snapshot_after,
            leak_counts,
            lineage_stats,
            targets,
        )
        log("Tiến hành nén dataset clean 2 class thành file ZIP để train.")
        made_zip = Path(shutil.make_archive(str(stage_root / output_name), "zip", root_dir=stage_dataset))
        if made_zip != stage_zip:
            fail(f"Khong tao duoc ZIP dung vi tri staging: {made_zip}")

        log("Tiến hành publish dataset, report và ZIP 2 class vào origin-dataset/outputs.")
        source_root.parent.mkdir(parents=True, exist_ok=True)
        replace_directory(stage_dataset, output_root)
        replace_directory(stage_reports, REPORTS_DIR)
        final_zip = source_root.parent / f"{output_name}.zip"
        os.replace(stage_zip, final_zip)
        log(f"Da tao dataset 2 class: {output_root}")
        log(f"Da tao ZIP train: {final_zip}")
        log(f"Bao cao: {REPORTS_DIR}")
        log("PASS: pose format, flip_idx, split 70/20/10 va critical dataleak da duoc kiem tra.")
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)


if __name__ == "__main__":
    main()
