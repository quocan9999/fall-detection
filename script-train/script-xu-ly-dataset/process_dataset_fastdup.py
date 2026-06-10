from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
ORIGIN_DATASET_DIR = ROOT / "origin-dataset"
OUTPUT_DIR = ORIGIN_DATASET_DIR / "outputs"
SCRIPT_OUTPUT_DIR = OUTPUT_DIR / Path(__file__).name
WORK_DIR = SCRIPT_OUTPUT_DIR / "work"
REPORT_DIR = SCRIPT_OUTPUT_DIR / "reports"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}
SPLIT_ALIASES = {
    "train": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "test": "test",
}


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[LỖI] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def load_config(path: Path) -> dict:
    if not path.exists():
        die(f"Không tìm thấy file cấu hình: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def output_base(cfg: dict) -> Path:
    group = str(cfg.get("OUTPUT_GROUP", "")).strip()
    return OUTPUT_DIR / group if group else OUTPUT_DIR


def clean_previous_run(enabled: bool, cfg: dict) -> None:
    if not enabled:
        return
    log("Tiến hành dọn dữ liệu trung gian, report và output no-resize của lần chạy trước.")
    for path in [WORK_DIR, REPORT_DIR, output_base(cfg)]:
        if path.exists():
            shutil.rmtree(path)


def prepare_dirs(cfg: dict | None = None) -> None:
    paths = [WORK_DIR, REPORT_DIR, OUTPUT_DIR]
    if cfg is not None:
        paths.append(output_base(cfg))
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def has_split_layout(path: Path) -> bool:
    if not path.is_dir():
        return False
    found = 0
    for name in SPLIT_ALIASES:
        split_dir = path / name
        if (split_dir / "images").is_dir() and (split_dir / "labels").is_dir():
            found += 1
    return found >= 2


def find_dataset_root(path: Path) -> Path:
    if has_split_layout(path):
        return path
    if (path / "data.yaml").exists() and has_split_layout(path):
        return path
    candidates = []
    for child in path.rglob("data.yaml"):
        parent = child.parent
        if has_split_layout(parent):
            candidates.append(parent)
    if candidates:
        return sorted(candidates, key=lambda p: len(str(p)))[0]
    for child in path.rglob("*"):
        if child.is_dir() and has_split_layout(child):
            candidates.append(child)
    if candidates:
        return sorted(candidates, key=lambda p: len(str(p)))[0]
    die(f"Không tìm thấy layout YOLO train/valid/test trong: {path}")


def resolve_dataset_source(cfg: dict) -> Path:
    source = str(cfg.get("DATASET_SOURCE", "auto")).strip()
    hint = str(cfg.get("DATASET_NAME_HINT", "fall-detection-no-augment")).strip()

    if source.lower() != "auto":
        src = Path(source).expanduser()
        if not src.is_absolute():
            src = ORIGIN_DATASET_DIR / src
        if not src.exists():
            die(f"DATASET_SOURCE không tồn tại: {src}")
        return materialize_source(src)

    candidates = [
        ORIGIN_DATASET_DIR / hint,
        ORIGIN_DATASET_DIR / f"{hint}.zip",
    ]
    if ORIGIN_DATASET_DIR.exists():
        candidates += sorted(ORIGIN_DATASET_DIR.glob(f"*{hint}*.zip"))
        candidates += [p for p in sorted(ORIGIN_DATASET_DIR.iterdir()) if p.is_dir() and hint.lower() in p.name.lower()]
        candidates += [p for p in sorted(ORIGIN_DATASET_DIR.iterdir()) if p.is_dir() and has_split_layout(p)]

    for candidate in candidates:
        if candidate.exists():
            return materialize_source(candidate)

    die(
        "Không tự tìm thấy dataset. Hãy đặt folder/zip dataset vào origin-dataset "
        "hoặc sửa DATASET_SOURCE trong config.yaml."
    )


def materialize_source(src: Path) -> Path:
    if src.is_file() and src.suffix.lower() == ".zip":
        extract_dir = WORK_DIR / "extracted_original"
        extract_dir.mkdir(parents=True, exist_ok=True)
        log(f"Tiến hành giải nén dataset: {src}")
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(extract_dir)
        return find_dataset_root(extract_dir)
    if src.is_dir():
        return find_dataset_root(src)
    die(f"DATASET_SOURCE phải là folder hoặc file .zip: {src}")


def split_dirs(dataset_root: Path) -> dict[str, Path]:
    result = {}
    for child in dataset_root.iterdir():
        if child.is_dir():
            canonical = SPLIT_ALIASES.get(child.name.lower())
            if canonical and (child / "images").is_dir() and (child / "labels").is_dir():
                result[canonical] = child
    missing = [s for s in ["train", "valid", "test"] if s not in result]
    if missing:
        die(f"Dataset thiếu split: {missing}. Root đang dùng: {dataset_root}")
    return result


def image_display_name(path: Path) -> str:
    stem = path.stem
    match = re.search(r"(?:image|img)[_\-]?(\d+)", stem, flags=re.IGNORECASE)
    if match:
        return f"image_{match.group(1)}"
    return stem


def collect_manifest(dataset_root: Path) -> pd.DataFrame:
    log("Tiến hành lập danh sách ảnh và label của dataset nguồn.")
    splits = split_dirs(dataset_root)
    rows = []
    for split, split_dir in splits.items():
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        images = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        for img in sorted(images):
            rel_img = img.relative_to(images_dir)
            label = labels_dir / rel_img.with_suffix(".txt")
            rows.append(
                {
                    "split": split,
                    "image_path": str(img.resolve()),
                    "label_path": str(label.resolve()),
                    "rel_image": rel_img.as_posix(),
                    "rel_label": rel_img.with_suffix(".txt").as_posix(),
                    "file_name": img.name,
                    "stem": img.stem,
                    "display_name": image_display_name(img),
                    "has_label": label.exists(),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        die(f"Không tìm thấy ảnh trong dataset: {dataset_root}")
    df.to_csv(REPORT_DIR / "00_manifest.csv", index=False, encoding="utf-8-sig")
    log(f"Tìm thấy {len(df)} ảnh. Manifest: reports/00_manifest.csv")
    return df


def run_fastdup(manifest: pd.DataFrame) -> pd.DataFrame:
    try:
        import fastdup
    except Exception as exc:
        warning = (
            "Không import được fastdup, sẽ fallback sang pHash local bằng Python. "
            f"Chi tiết lỗi: {exc}"
        )
        log(warning)
        (REPORT_DIR / "fastdup_warnings.txt").write_text(warning, encoding="utf-8")
        return run_phash_fallback(manifest, float(load_config(ROOT / "config.yaml").get("FASTDUP_MAX_DISTANCE", 0.05)))

    fd_work = WORK_DIR / "fastdup"
    fd_work.mkdir(parents=True, exist_ok=True)
    file_list = WORK_DIR / "fastdup_input_files.txt"
    file_list.write_text("\n".join(manifest["image_path"].tolist()) + "\n", encoding="utf-8")

    log("Tiến hành chạy fastdup để tìm ảnh trùng/gần trùng. Bước này có thể mất vài phút.")
    sim_df = None
    errors = []

    try:
        fd = fastdup.create(work_dir=str(fd_work), input_dir=str(file_list))
        fd.run()
        sim_df = fd.similarity()
        try:
            invalid = fd.invalid_instances()
            invalid.to_csv(REPORT_DIR / "01_fastdup_invalid_images.csv", index=False, encoding="utf-8-sig")
        except Exception as exc:
            errors.append(f"Không lấy được invalid_instances(): {exc}")
    except Exception as exc:
        errors.append(f"API fastdup.create thất bại: {exc}")

    if sim_df is None:
        try:
            ret = fastdup.run(input_dir=str(file_list), work_dir=str(fd_work))
            if ret not in [0, None]:
                errors.append(f"fastdup.run trả về mã: {ret}")
            sim_path = fd_work / "similarity.csv"
            if sim_path.exists():
                sim_df = pd.read_csv(sim_path)
        except Exception as exc:
            errors.append(f"API fastdup.run thất bại: {exc}")

    if sim_df is None:
        warning = (
            "fastdup không sinh được similarity dataframe nên fallback sang pHash local bằng Python.\n"
            + "\n".join(errors)
        )
        log(warning)
        (REPORT_DIR / "fastdup_warnings.txt").write_text(warning, encoding="utf-8")
        return run_phash_fallback(manifest, float(load_config(ROOT / "config.yaml").get("FASTDUP_MAX_DISTANCE", 0.05)))

    if errors:
        (REPORT_DIR / "fastdup_warnings.txt").write_text("\n".join(errors), encoding="utf-8")

    sim_df.to_csv(REPORT_DIR / "01_fastdup_similarity_all_raw.csv", index=False, encoding="utf-8-sig")
    log(f"fastdup trả về {len(sim_df)} dòng similarity.")
    return sim_df


def run_phash_fallback(manifest: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    try:
        import imagehash
    except Exception as exc:
        die(
            "Không import được ImageHash để chạy fallback. "
            "Chạy: pip install -r requirements.txt. "
            f"Chi tiết lỗi: {exc}"
        )

    max_bits = max(0, int(max_distance * 64))
    log(
        "Tiến hành dùng fallback pHash local vì fastdup lỗi. "
        f"Threshold normalized={max_distance}, tương đương <= {max_bits} bit khác nhau."
    )

    hashes = []
    bad_rows = []
    log("Tiến hành tính pHash cho toàn bộ ảnh.")
    for row in tqdm(manifest.to_dict("records"), desc="Tính pHash"):
        try:
            with Image.open(row["image_path"]) as img:
                h = imagehash.phash(img.convert("RGB"))
            hashes.append((row["image_path"], int(str(h), 16)))
        except Exception as exc:
            bad_rows.append({"image_path": row["image_path"], "error": str(exc)})

    if bad_rows:
        pd.DataFrame(bad_rows).to_csv(REPORT_DIR / "01_phash_fallback_bad_images.csv", index=False, encoding="utf-8-sig")

    rows = []
    n = len(hashes)
    log("Tiến hành so sánh pHash giữa các ảnh để tạo danh sách cặp gần trùng.")
    for i in tqdm(range(n), desc="So sánh pHash"):
        path_i, hash_i = hashes[i]
        for j in range(i + 1, n):
            path_j, hash_j = hashes[j]
            bits = (hash_i ^ hash_j).bit_count()
            if bits <= max_bits:
                rows.append(
                    {
                        "filename_from": path_i,
                        "filename_to": path_j,
                        "distance": bits / 64.0,
                        "phash_hamming_bits": bits,
                        "backend": "phash_fallback",
                    }
                )

    sim_df = pd.DataFrame(rows)
    sim_df.to_csv(REPORT_DIR / "01_phash_fallback_similarity.csv", index=False, encoding="utf-8-sig")
    log(f"Fallback pHash tìm thấy {len(sim_df)} cặp dưới threshold.")
    return sim_df


def normalize_fastdup_pairs(sim_df: pd.DataFrame, manifest: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    log("Tiến hành chuẩn hoá kết quả duplicate và lọc theo threshold.")
    path_lookup = {}
    for row in manifest.to_dict("records"):
        path = str(Path(row["image_path"]).resolve())
        path_lookup[path] = row
        path_lookup[Path(path).as_posix()] = row
        path_lookup[Path(path).name] = row

    def first_existing(row: pd.Series, columns: list[str]):
        for col in columns:
            if col in row and pd.notna(row[col]):
                return row[col]
        return None

    rows = []
    for _, r in sim_df.iterrows():
        a = first_existing(r, ["filename_from", "from", "from_filename"])
        b = first_existing(r, ["filename_to", "to", "to_filename"])
        dist = first_existing(r, ["distance", "score", "similarity"])
        if a is None or b is None or dist is None:
            continue

        try:
            dist_float = float(dist)
        except Exception:
            continue
        if dist_float > max_distance:
            continue

        a_key = str(a)
        b_key = str(b)
        a_abs = str(Path(a_key).resolve()) if not a_key.isdigit() else a_key
        b_abs = str(Path(b_key).resolve()) if not b_key.isdigit() else b_key
        left = path_lookup.get(a_abs) or path_lookup.get(Path(a_key).as_posix()) or path_lookup.get(Path(a_key).name)
        right = path_lookup.get(b_abs) or path_lookup.get(Path(b_key).as_posix()) or path_lookup.get(Path(b_key).name)
        if not left or not right:
            continue
        if left["image_path"] == right["image_path"]:
            continue

        rows.append(
            {
                "from_split": left["split"],
                "to_split": right["split"],
                "from_display_name": left["display_name"],
                "to_display_name": right["display_name"],
                "from_image": left["image_path"],
                "to_image": right["image_path"],
                "from_label": left["label_path"],
                "to_label": right["label_path"],
                "distance": dist_float,
            }
        )

    pairs = pd.DataFrame(rows).drop_duplicates()
    pairs = pairs.sort_values("distance", kind="stable") if not pairs.empty else pairs
    pairs.to_csv(REPORT_DIR / "02_fastdup_pairs_under_threshold.csv", index=False, encoding="utf-8-sig")
    log(f"Số cặp trùng/gần trùng dưới threshold {max_distance}: {len(pairs)}")
    return pairs


class DSU:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_duplicate_components(pairs: pd.DataFrame, image_paths: list[str]) -> dict[str, str]:
    dsu = DSU()
    for p in image_paths:
        dsu.find(p)
    for _, r in pairs.iterrows():
        dsu.union(r["from_image"], r["to_image"])
    return {p: dsu.find(p) for p in image_paths}


def leak_reports_and_removal(pairs: pd.DataFrame, manifest: pd.DataFrame, cfg: dict) -> tuple[set[str], dict[str, str]]:
    log("Tiến hành phân tích dataleak train-valid/train-test và xuất report duplicate.")
    def is_pair(a: str, b: str, x: str, y: str) -> bool:
        return {a, b} == {x, y}

    train_valid = pairs[pairs.apply(lambda r: is_pair(r["from_split"], r["to_split"], "train", "valid"), axis=1)]
    train_test = pairs[pairs.apply(lambda r: is_pair(r["from_split"], r["to_split"], "train", "test"), axis=1)]
    valid_test = pairs[pairs.apply(lambda r: is_pair(r["from_split"], r["to_split"], "valid", "test"), axis=1)]

    train_valid.to_csv(REPORT_DIR / "03_CRITICAL_leak_train_valid_pairs.csv", index=False, encoding="utf-8-sig")
    train_test.to_csv(REPORT_DIR / "04_CRITICAL_leak_train_test_pairs.csv", index=False, encoding="utf-8-sig")
    valid_test.to_csv(REPORT_DIR / "05_duplicate_valid_test_only_pairs.csv", index=False, encoding="utf-8-sig")

    path_to_split = dict(zip(manifest["image_path"], manifest["split"]))
    component_by_image = build_duplicate_components(pairs, manifest["image_path"].tolist())

    groups = defaultdict(list)
    for path in path_to_split:
        groups[component_by_image[path]].append(path)

    remove = set()
    component_rows = []
    for group_id, paths in groups.items():
        splits = {path_to_split[p] for p in paths}
        has_critical = "train" in splits and ("valid" in splits or "test" in splits)
        for p in paths:
            if has_critical and path_to_split[p] in {"valid", "test"}:
                remove.add(p)
        if len(paths) > 1:
            component_rows.append(
                {
                    "component_id": group_id,
                    "num_images": len(paths),
                    "splits": ",".join(sorted(splits)),
                    "is_critical_train_leak": has_critical,
                    "num_removed": sum(1 for p in paths if p in remove),
                    "images": " | ".join(paths),
                }
            )

    components = pd.DataFrame(component_rows)
    components.to_csv(REPORT_DIR / "06_duplicate_components.csv", index=False, encoding="utf-8-sig")

    removed_rows = manifest[manifest["image_path"].isin(remove)].copy()
    removed_rows.to_csv(REPORT_DIR / "07_images_removed_from_valid_test.csv", index=False, encoding="utf-8-sig")

    log(f"CRITICAL leak train-valid pairs: {len(train_valid)}")
    log(f"CRITICAL leak train-test pairs: {len(train_test)}")
    log(f"Duplicate valid-test only pairs: {len(valid_test)}")
    log(f"Số ảnh valid/test sẽ bị loại vì dính train: {len(remove)}")
    make_review_samples(pd.concat([train_valid, train_test], ignore_index=True), cfg)
    return remove, component_by_image


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:120]


def make_review_samples(critical_pairs: pd.DataFrame, cfg: dict) -> None:
    max_samples = int(cfg.get("MAX_REVIEW_SAMPLES", 120))
    review_dir = REPORT_DIR / "leak_review_samples"
    review_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, r in critical_pairs.head(max_samples).reset_index(drop=True).iterrows():
        folder = review_dir / f"{idx + 1:04d}_{safe_name(r['from_display_name'])}__{safe_name(r['to_display_name'])}"
        folder.mkdir(parents=True, exist_ok=True)
        left = Path(r["from_image"])
        right = Path(r["to_image"])
        shutil.copy2(left, folder / f"{r['from_split']}__{left.name}")
        shutil.copy2(right, folder / f"{r['to_split']}__{right.name}")
        rows.append({"sample_folder": str(folder), **r.to_dict()})
    pd.DataFrame(rows).to_csv(review_dir / "index.csv", index=False, encoding="utf-8-sig")


def label_class_counts(label_path: str) -> Counter:
    counts = Counter()
    path = Path(label_path)
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                counts[int(float(parts[0]))] += 1
            except Exception:
                continue
    return counts


def add_counter(a: Counter, b: Counter) -> Counter:
    out = Counter(a)
    out.update(b)
    return out


def counter_abs_distance(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys))


def rebalance_after_cleaning(
    manifest: pd.DataFrame,
    remove_images: set[str],
    component_by_image: dict[str, str],
    cfg: dict,
) -> pd.DataFrame:
    log("Tiến hành loại ảnh leak và cân bằng lại tỷ lệ train/valid/test theo class.")
    keep = manifest[~manifest["image_path"].isin(remove_images)].copy()
    keep["final_split"] = keep["split"]
    keep["component_id"] = keep["image_path"].map(component_by_image).fillna(keep["image_path"])
    keep["class_counts_obj"] = keep["label_path"].map(label_class_counts)

    if not bool(cfg.get("REBALANCE_AFTER_CLEANING", True)):
        keep.to_csv(REPORT_DIR / "09_final_manifest_no_rebalance.csv", index=False, encoding="utf-8-sig")
        return keep

    ratios = cfg.get("SPLIT_RATIOS", {"train": 0.70, "valid": 0.20, "test": 0.10})
    ratios = {k: float(v) for k, v in ratios.items()}
    total_ratio = sum(ratios.values())
    ratios = {k: v / total_ratio for k, v in ratios.items()}

    total_images = len(keep)
    target_images = {
        "train": total_images - round(total_images * ratios.get("valid", 0.20)) - round(total_images * ratios.get("test", 0.10)),
        "valid": round(total_images * ratios.get("valid", 0.20)),
        "test": round(total_images * ratios.get("test", 0.10)),
    }

    total_classes = Counter()
    for c in keep["class_counts_obj"]:
        total_classes.update(c)
    target_classes = {
        split: Counter({cls: round(count * ratios[split]) for cls, count in total_classes.items()})
        for split in ["train", "valid", "test"]
    }

    def current_counts(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, Counter]]:
        image_counts = {split: int((df["final_split"] == split).sum()) for split in ["train", "valid", "test"]}
        class_counts = {split: Counter() for split in ["train", "valid", "test"]}
        for _, row in df.iterrows():
            class_counts[row["final_split"]].update(row["class_counts_obj"])
        return image_counts, class_counts

    groups = []
    for component_id, g in keep.groupby("component_id", sort=False):
        splits = set(g["final_split"].tolist())
        group_classes = Counter()
        for c in g["class_counts_obj"]:
            group_classes.update(c)
        groups.append(
            {
                "component_id": component_id,
                "image_paths": g["image_path"].tolist(),
                "original_splits": splits,
                "num_images": len(g),
                "class_counts": group_classes,
            }
        )

    movable = [
        g for g in groups
        if g["original_splits"] == {"train"} and g["num_images"] < total_images
    ]

    moved_rows = []
    image_counts, class_counts = current_counts(keep)

    for target_split in ["test", "valid"]:
        while image_counts[target_split] < target_images[target_split]:
            need_images = target_images[target_split] - image_counts[target_split]
            candidates = [g for g in movable if g["num_images"] <= max(need_images, 1)]
            if not candidates:
                candidates = movable[:]
            if not candidates:
                log(f"Không còn train-only component để bù cho {target_split}.")
                break

            before_distance = counter_abs_distance(class_counts[target_split], target_classes[target_split])

            def score(group: dict) -> tuple[float, int, int]:
                new_counts = add_counter(class_counts[target_split], group["class_counts"])
                after_distance = counter_abs_distance(new_counts, target_classes[target_split])
                image_over = max(0, image_counts[target_split] + group["num_images"] - target_images[target_split])
                return (after_distance - before_distance + image_over * 1000, image_over, group["num_images"])

            chosen = min(candidates, key=score)
            movable.remove(chosen)
            keep.loc[keep["image_path"].isin(chosen["image_paths"]), "final_split"] = target_split
            image_counts["train"] -= chosen["num_images"]
            image_counts[target_split] += chosen["num_images"]
            class_counts["train"].subtract(chosen["class_counts"])
            class_counts[target_split].update(chosen["class_counts"])
            moved_rows.append(
                {
                    "component_id": chosen["component_id"],
                    "to_split": target_split,
                    "num_images": chosen["num_images"],
                    "class_counts": dict(chosen["class_counts"]),
                    "images": " | ".join(chosen["image_paths"]),
                }
            )

    report_rows = []
    final_image_counts, final_class_counts = current_counts(keep)
    for split in ["train", "valid", "test"]:
        report_rows.append(
            {
                "split": split,
                "target_images": target_images[split],
                "final_images": final_image_counts[split],
                "target_class_counts": dict(target_classes[split]),
                "final_class_counts": dict(final_class_counts[split]),
            }
        )

    pd.DataFrame(moved_rows).to_csv(REPORT_DIR / "09_rebalance_moved_train_to_valid_test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report_rows).to_csv(REPORT_DIR / "10_rebalance_summary.csv", index=False, encoding="utf-8-sig")

    export_manifest = keep.drop(columns=["class_counts_obj"]).copy()
    export_manifest.to_csv(REPORT_DIR / "11_final_manifest_after_rebalance.csv", index=False, encoding="utf-8-sig")

    log(f"Rebalance xong. Ảnh final: train={final_image_counts['train']}, valid={final_image_counts['valid']}, test={final_image_counts['test']}")
    return keep


def check_final_leaks(pairs: pd.DataFrame, final_manifest: pd.DataFrame) -> pd.DataFrame:
    log("Tiến hành kiểm tra lại dataleak sau khi cân bằng split.")
    final_split_by_image = dict(zip(final_manifest["image_path"], final_manifest["final_split"]))
    rows = []
    for _, r in pairs.iterrows():
        a_split = final_split_by_image.get(r["from_image"])
        b_split = final_split_by_image.get(r["to_image"])
        if not a_split or not b_split or a_split == b_split:
            continue
        leak_type = "other"
        if {a_split, b_split} == {"train", "valid"}:
            leak_type = "CRITICAL_train_valid"
        elif {a_split, b_split} == {"train", "test"}:
            leak_type = "CRITICAL_train_test"
        elif {a_split, b_split} == {"valid", "test"}:
            leak_type = "valid_test_only"
        rows.append(
            {
                "leak_type": leak_type,
                "from_final_split": a_split,
                "to_final_split": b_split,
                **r.to_dict(),
            }
        )
    final_leaks = pd.DataFrame(rows)
    final_leaks.to_csv(REPORT_DIR / "12_final_split_duplicate_leak_check.csv", index=False, encoding="utf-8-sig")
    counts = Counter(final_leaks["leak_type"].tolist()) if not final_leaks.empty else Counter()
    summary = [
        "Kiểm tra duplicate/dataleak sau khi rebalance:",
        f"CRITICAL train-valid: {counts.get('CRITICAL_train_valid', 0)}",
        f"CRITICAL train-test: {counts.get('CRITICAL_train_test', 0)}",
        f"Duplicate valid-test only: {counts.get('valid_test_only', 0)}",
    ]
    (REPORT_DIR / "12_final_split_duplicate_leak_check.txt").write_text("\n".join(summary), encoding="utf-8")
    log(summary[0])
    log(summary[1])
    log(summary[2])
    log(summary[3])
    return final_leaks


def copy_clean_dataset(dataset_root: Path, final_manifest: pd.DataFrame, cfg: dict) -> Path:
    log("Tiến hành sao chép ảnh/label để tạo dataset clean 5 class.")
    out_name = str(cfg.get("OUTPUT_DATASET_NAME", "fall-detection-clean-fastdup")).strip()
    out_root = output_base(cfg) / out_name
    if out_root.exists():
        shutil.rmtree(out_root)

    for _, row in tqdm(final_manifest.iterrows(), total=len(final_manifest), desc="Copy dataset clean"):
        split = row["final_split"]
        dst_img = out_root / split / "images" / row["rel_image"]
        dst_lbl = out_root / split / "labels" / row["rel_label"]
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["image_path"], dst_img)
        if Path(row["label_path"]).exists():
            shutil.copy2(row["label_path"], dst_lbl)

    write_data_yaml(dataset_root, out_root, cfg)
    log(f"Dataset clean đã tạo tại: {out_root}")
    return out_root


def write_data_yaml(source_root: Path, out_root: Path, cfg: dict) -> None:
    old_yaml = source_root / "data.yaml"
    data = {}
    if old_yaml.exists():
        with old_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"
    data.setdefault("kpt_shape", cfg.get("EXPECTED_KPT_SHAPE", [18, 3]))
    kpt_count = int(data["kpt_shape"][0])
    verified_flip_idx = cfg.get("VERIFIED_FLIP_IDX")
    if verified_flip_idx is not None:
        try:
            verified_flip_idx = [int(index) for index in verified_flip_idx]
        except (TypeError, ValueError):
            die("VERIFIED_FLIP_IDX phải là danh sách số nguyên trong config.yaml.")
        if len(verified_flip_idx) != kpt_count or sorted(verified_flip_idx) != list(range(kpt_count)):
            die(
                "VERIFIED_FLIP_IDX không hợp lệ: phải chứa đúng một lần mỗi index "
                f"từ 0 đến {kpt_count - 1}."
            )
        data["flip_idx"] = verified_flip_idx
        log(f"Đã áp dụng VERIFIED_FLIP_IDX vào data.yaml output: {verified_flip_idx}")
    if "names" not in data:
        data["names"] = ["Sitting", "Sleeping", "Standing", "Walking", "falling"]
    if "nc" not in data:
        data["nc"] = len(data["names"]) if isinstance(data["names"], list) else len(data["names"].keys())

    with (out_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def validate_dataset(out_root: Path, cfg: dict) -> dict:
    log(f"Tiến hành kiểm tra cấu trúc YOLO pose, label và geometry: {out_root}")
    data_yaml = out_root / "data.yaml"
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    kpt_shape = data.get("kpt_shape") or cfg.get("EXPECTED_KPT_SHAPE", [18, 3])
    kpt_count = int(kpt_shape[0])
    kpt_dims = int(kpt_shape[1])
    expected_values = 5 + kpt_count * kpt_dims
    names = data.get("names", [])
    nc = int(data.get("nc", len(names)))

    summary = {
        "dataset": str(out_root),
        "kpt_shape": kpt_shape,
        "expected_values_per_label_line": expected_values,
        "splits": {},
        "label_value_lengths": Counter(),
        "class_counts": Counter(),
        "missing_labels": [],
        "orphan_labels": [],
        "corrupt_images": [],
        "bad_label_lines": [],
        "bad_label_geometry": [],
    }

    visual_dir = REPORT_DIR / "label_visual_check_samples"
    visual_dir.mkdir(parents=True, exist_ok=True)
    max_visual_samples = int(cfg.get("MAX_LABEL_VISUAL_SAMPLES", 80))
    visual_count = 0
    bad_geometry_visual_dir = REPORT_DIR / "bad_label_geometry_visuals"
    if bad_geometry_visual_dir.exists():
        shutil.rmtree(bad_geometry_visual_dir)
    bad_geometry_visual_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        img_dir = out_root / split / "images"
        lbl_dir = out_root / split / "labels"
        images = sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        labels = sorted([p for p in lbl_dir.rglob("*.txt") if p.is_file()])
        summary["splits"][split] = {"images": len(images), "labels": len(labels)}

        image_rels = {p.relative_to(img_dir).with_suffix(".txt").as_posix() for p in images}
        label_rels = {p.relative_to(lbl_dir).as_posix() for p in labels}
        for rel in sorted(image_rels - label_rels):
            summary["missing_labels"].append(f"{split}/{rel}")
        for rel in sorted(label_rels - image_rels):
            summary["orphan_labels"].append(f"{split}/{rel}")

        for img in images:
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception as exc:
                summary["corrupt_images"].append({"image": str(img), "error": str(exc)})

        image_by_rel_label = {p.relative_to(img_dir).with_suffix(".txt").as_posix(): p for p in images}
        for label in labels:
            rel_label = label.relative_to(lbl_dir).as_posix()
            image_path = image_by_rel_label.get(rel_label)
            image_size = None
            if image_path and image_path.exists():
                try:
                    with Image.open(image_path) as im:
                        image_size = im.size
                except Exception:
                    image_size = None

            parsed_objects = []
            geometry_row_indexes = []
            with label.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    summary["label_value_lengths"][len(parts)] += 1
                    try:
                        cls = int(float(parts[0]))
                        summary["class_counts"][cls] += 1
                    except Exception:
                        cls = None
                    if len(parts) != expected_values or cls is None or cls < 0 or cls >= nc:
                        summary["bad_label_lines"].append(
                            {
                                "label": str(label),
                                "line": line_no,
                                "num_values": len(parts),
                                "class": cls,
                            }
                        )
                        continue

                    nums = [float(x) for x in parts]
                    bbox = nums[1:5]
                    keypoints = nums[5:]
                    geom_errors = []
                    x, y, w, h = bbox
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        geom_errors.append("bbox_normalized_invalid")
                    if x - w / 2 < -0.02 or y - h / 2 < -0.02 or x + w / 2 > 1.02 or y + h / 2 > 1.02:
                        geom_errors.append("bbox_outside_image")

                    kpt_bad = 0
                    for idx in range(0, len(keypoints), kpt_dims):
                        kx, ky = keypoints[idx], keypoints[idx + 1]
                        kv = keypoints[idx + 2] if kpt_dims >= 3 else 1
                        if kv > 0 and not (0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0):
                            kpt_bad += 1
                    if kpt_bad:
                        geom_errors.append(f"visible_keypoints_outside_image={kpt_bad}")

                    if geom_errors:
                        summary["bad_label_geometry"].append(
                            {
                                "label": str(label),
                                "image": str(image_path) if image_path else None,
                                "line": line_no,
                                "class": cls,
                                "errors": geom_errors,
                                "visual_overlay": None,
                            }
                        )
                        geometry_row_indexes.append(len(summary["bad_label_geometry"]) - 1)
                    parsed_objects.append((cls, bbox, keypoints))

            if image_path and image_size and parsed_objects and visual_count < max_visual_samples:
                out_name = f"{visual_count + 1:04d}_{split}_{safe_name(image_path.stem)}.jpg"
                draw_label_overlay(image_path, parsed_objects, visual_dir / out_name)
                visual_count += 1
            if image_path and image_size and parsed_objects and geometry_row_indexes:
                out_name = f"{split}_{safe_name(Path(rel_label).with_suffix('').as_posix())}.jpg"
                overlay_path = bad_geometry_visual_dir / out_name
                draw_label_overlay(image_path, parsed_objects, overlay_path)
                for row_idx in geometry_row_indexes:
                    summary["bad_label_geometry"][row_idx]["visual_overlay"] = str(overlay_path)

    serializable = json.loads(json.dumps(summary, default=lambda x: dict(x), ensure_ascii=False))
    with (REPORT_DIR / "08_dataset_check_summary.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    pd.DataFrame(serializable["bad_label_geometry"]).to_csv(
        REPORT_DIR / "08_bad_label_geometry.csv",
        index=False,
        encoding="utf-8-sig",
    )

    text_lines = []
    text_lines.append(f"Dataset: {out_root}")
    text_lines.append(f"kpt_shape: {kpt_shape}")
    text_lines.append(f"Số giá trị label mong đợi mỗi dòng: {expected_values}")
    for split, counts in serializable["splits"].items():
        text_lines.append(f"[{split}] images={counts['images']}, labels={counts['labels']}")
    text_lines.append(f"Thiếu label: {len(serializable['missing_labels'])}")
    text_lines.append(f"Label không có ảnh: {len(serializable['orphan_labels'])}")
    text_lines.append(f"Ảnh lỗi/corrupt: {len(serializable['corrupt_images'])}")
    text_lines.append(f"Dòng label sai format/class: {len(serializable['bad_label_lines'])}")
    text_lines.append(f"Dòng label có bbox/keypoint bất thường: {len(serializable['bad_label_geometry'])}")
    text_lines.append(f"Ảnh overlay kiểm tra label: {visual_dir}")
    text_lines.append(f"Ảnh overlay riêng cho label hình học bất thường: {bad_geometry_visual_dir}")
    text_lines.append(f"Phân bố độ dài label: {serializable['label_value_lengths']}")
    text_lines.append(f"Phân bố class: {serializable['class_counts']}")
    text_lines.append(
        "KẾT LUẬN: Dataset có vẻ là YOLO pose và label không có lỗi hình học rõ ràng."
        if not serializable["bad_label_lines"] and not serializable["bad_label_geometry"] and expected_values in map(int, serializable["label_value_lengths"].keys())
        else "KẾT LUẬN: Cần kiểm tra lại format label."
    )
    (REPORT_DIR / "08_dataset_check_summary.txt").write_text("\n".join(text_lines), encoding="utf-8")
    log("Đã kiểm tra dataset clean. Xem reports/08_dataset_check_summary.txt")
    return serializable


def draw_label_overlay(image_path: Path, objects: list[tuple[int, list[float], list[float]]], out_path: Path) -> None:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        width, height = im.size
        colors = ["red", "lime", "cyan", "yellow", "magenta", "orange"]
        for obj_idx, (cls, bbox, keypoints) in enumerate(objects):
            color = colors[obj_idx % len(colors)]
            x, y, w, h = bbox
            x1 = (x - w / 2) * width
            y1 = (y - h / 2) * height
            x2 = (x + w / 2) * width
            y2 = (y + h / 2) * height
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((max(0, x1), max(0, y1 - 12)), f"class {cls}", fill=color)
            for idx in range(0, len(keypoints), 3):
                kx, ky, kv = keypoints[idx], keypoints[idx + 1], keypoints[idx + 2]
                if kv <= 0:
                    continue
                px = kx * width
                py = ky * height
                r = max(2, int(min(width, height) * 0.006))
                draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="black")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, quality=92)


def zip_dataset(out_root: Path) -> Path:
    log("Tiến hành nén dataset clean 5 class thành file ZIP để train.")
    zip_base = out_root.parent / out_root.name
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=out_root))
    log(f"Đã nén dataset clean: {zip_path}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Xử lý dataleak YOLO pose dataset bằng fastdup.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Đường dẫn config.yaml")
    parser.add_argument(
        "--check-output-only",
        action="store_true",
        help="Chỉ kiểm tra/vẽ overlay trên dataset clean hiện có, không quét duplicate lại.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.check_output_only:
        prepare_dirs(cfg)
        out_name = str(cfg.get("OUTPUT_DATASET_NAME", "fall-detection-clean-fastdup")).strip()
        out_root = output_base(cfg) / out_name
        if not out_root.exists():
            die(f"Không tìm thấy dataset clean hiện có: {out_root}")
        validate_dataset(out_root, cfg)
        log("Đã tạo lại report/overlay từ dataset clean hiện có; không chạy lại duplicate scan.")
        return

    clean_previous_run(bool(cfg.get("CLEAN_PREVIOUS_RUN", True)), cfg)
    prepare_dirs(cfg)

    log("Tiến hành xác định và chuẩn bị dataset nguồn.")
    dataset_root = resolve_dataset_source(cfg)
    log(f"Dataset root đang dùng: {dataset_root}")

    manifest = collect_manifest(dataset_root)
    sim_df = run_fastdup(manifest)
    max_distance = float(cfg.get("FASTDUP_MAX_DISTANCE", 0.05))
    pairs = normalize_fastdup_pairs(sim_df, manifest, max_distance)
    remove_images, component_by_image = leak_reports_and_removal(pairs, manifest, cfg)

    if not bool(cfg.get("APPLY_CLEANING", True)):
        log("APPLY_CLEANING=false nên chỉ xuất report, chưa tạo dataset clean.")
        return

    final_manifest = rebalance_after_cleaning(manifest, remove_images, component_by_image, cfg)
    final_leaks = check_final_leaks(pairs, final_manifest)
    critical_after = 0
    if not final_leaks.empty:
        critical_after = int(final_leaks["leak_type"].isin(["CRITICAL_train_valid", "CRITICAL_train_test"]).sum())
    if critical_after > 0:
        die(
            "Sau rebalance vẫn còn critical leak train-valid/train-test. "
            "Xem reports/12_final_split_duplicate_leak_check.csv"
        )

    out_root = copy_clean_dataset(dataset_root, final_manifest, cfg)
    validate_dataset(out_root, cfg)
    zip_dataset(out_root)
    log(f"Hoàn tất. Report nằm trong {REPORT_DIR}, dataset và file zip nằm trong {output_base(cfg)}.")


if __name__ == "__main__":
    main()
