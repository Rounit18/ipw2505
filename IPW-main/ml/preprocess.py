import csv
import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from PIL import UnidentifiedImageError

from class_config import (
    CLASS_TO_ID,
    EXPECTED_CLASS_NAMES,
    class_category,
    normalize_class_name,
)
from create_mappings import build_mappings, write_json, write_label_map
from image_preprocessing import (
    PREPROCESSING_CONFIG,
    iter_image_paths,
    preprocess_image_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CONFIG = ROOT / "ml" / "configs" / "dataset_config.json"
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "indian-sign-language-isl" / "Indian"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "isl_processed"
DEFAULT_SPLITS_DIR = ROOT / "data" / "splits"
DEFAULT_REPORTS_DIR = ROOT / "ml" / "reports"
DEFAULT_BACKEND_MAPPINGS = ROOT / "backend" / "data" / "mappings.json"
DEFAULT_FRONTEND_MAPPINGS = ROOT / "frontend" / "src" / "data" / "mappings.json"
DEFAULT_LABEL_MAP = ROOT / "ml" / "configs" / "label_map.json"


def load_dataset_config(path=DEFAULT_DATASET_CONFIG):
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_path(config, key, fallback):
    value = config.get(key)
    if not value:
        return fallback
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def project_path(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def discover_class_folders(raw_dir, allow_extra_folders=False):
    folders = {}
    extras = []

    for path in sorted(raw_dir.iterdir()):
        if not path.is_dir():
            continue
        class_name = normalize_class_name(path.name)
        if class_name in CLASS_TO_ID:
            folders[class_name] = path
        else:
            extras.append(path.name)

    missing = [class_name for class_name in EXPECTED_CLASS_NAMES if class_name not in folders]

    if missing:
        raise ValueError(f"Missing class folders: {', '.join(missing)}")
    if extras and not allow_extra_folders:
        raise ValueError(
            "Unexpected class folders: "
            + ", ".join(extras)
            + ". Pass --allow-extra-folders to ignore them."
        )

    return folders, extras


def split_class_records(records, train_ratio, val_ratio, test_ratio, seed):
    ratio_sum = round(train_ratio + val_ratio + test_ratio, 6)
    if ratio_sum != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    rng = random.Random(seed)
    split_records = []

    for class_name in EXPECTED_CLASS_NAMES:
        class_records = list(records[class_name])
        rng.shuffle(class_records)
        total = len(class_records)

        if total < 3:
            raise ValueError(f"Class {class_name} needs at least 3 images for stratified splits.")

        test_count = max(1, round(total * test_ratio))
        val_count = max(1, round(total * val_ratio))
        train_count = total - val_count - test_count

        if train_count < 1:
            train_count = 1
            if val_count > test_count:
                val_count -= 1
            else:
                test_count -= 1

        boundaries = {
            "train": train_count,
            "val": train_count + val_count,
        }

        for index, record in enumerate(class_records):
            if index < boundaries["train"]:
                split = "train"
            elif index < boundaries["val"]:
                split = "val"
            else:
                split = "test"
            split_records.append({**record, "split": split})

    return split_records


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "split",
        "gesture_id",
        "class_name",
        "category",
        "path",
        "source_path",
        "is_augmented",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def class_split_summary(records):
    summary = {
        class_name: {
            "total": 0,
            "train": 0,
            "val": 0,
            "test": 0,
        }
        for class_name in EXPECTED_CLASS_NAMES
    }

    for record in records:
        item = summary[record["class_name"]]
        item["total"] += 1
        item[record["split"]] += 1

    return summary


def process_dataset(
    raw_dir,
    processed_dir,
    splits_dir,
    reports_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=258,
    allow_extra_folders=False,
    fail_on_invalid=False,
):
    raw_dir = raw_dir.resolve()
    processed_dir = processed_dir.resolve()
    splits_dir = splits_dir.resolve()
    reports_dir = reports_dir.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir}")

    class_folders, extra_folders = discover_class_folders(
        raw_dir,
        allow_extra_folders=allow_extra_folders,
    )
    records_by_class = defaultdict(list)
    invalid_images = []

    for class_name in EXPECTED_CLASS_NAMES:
        class_dir = class_folders[class_name]
        image_paths = iter_image_paths(class_dir)
        if not image_paths:
            raise ValueError(f"Class folder has no supported images: {class_dir}")

        for source_path in image_paths:
            relative = source_path.relative_to(class_dir).with_suffix(".png")
            target_path = processed_dir / class_name / relative

            try:
                preprocess_image_file(source_path, target_path)
            except (OSError, UnidentifiedImageError) as exc:
                invalid_images.append(
                    {
                        "class_name": class_name,
                        "path": str(source_path),
                        "error": str(exc),
                    }
                )
                if fail_on_invalid:
                    raise
                continue

            record = {
                "record_id": f"{class_name}-{len(records_by_class[class_name]):06d}",
                "gesture_id": CLASS_TO_ID[class_name],
                "class_name": class_name,
                "category": class_category(class_name),
                "path": project_path(target_path),
                "source_path": str(source_path),
                "is_augmented": "0",
            }
            records_by_class[class_name].append(record)

    split_records = split_class_records(
        records_by_class,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    split_records = sorted(
        split_records,
        key=lambda record: (record["split"], record["gesture_id"], record["record_id"]),
    )

    all_csv = splits_dir / "all.csv"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    write_csv(all_csv, split_records)
    write_csv(train_csv, [record for record in split_records if record["split"] == "train"])
    write_csv(val_csv, [record for record in split_records if record["split"] == "val"])
    write_csv(test_csv, [record for record in split_records if record["split"] == "test"])

    mappings = build_mappings(existing_path=DEFAULT_BACKEND_MAPPINGS)
    write_json(DEFAULT_BACKEND_MAPPINGS, mappings)
    write_json(DEFAULT_FRONTEND_MAPPINGS, mappings)
    label_map = write_label_map(DEFAULT_LABEL_MAP)

    summary = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "splits_dir": str(splits_dir),
        "reports_dir": str(reports_dir),
        "preprocessing": PREPROCESSING_CONFIG,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "class_count": len(EXPECTED_CLASS_NAMES),
        "classes": EXPECTED_CLASS_NAMES,
        "label_map": label_map,
        "extra_folders_ignored": extra_folders,
        "invalid_images": invalid_images,
        "total_processed_images": len(split_records),
        "split_counts": {
            "train": sum(1 for record in split_records if record["split"] == "train"),
            "val": sum(1 for record in split_records if record["split"] == "val"),
            "test": sum(1 for record in split_records if record["split"] == "test"),
        },
        "class_split_summary": class_split_summary(split_records),
        "generated_files": {
            "all_csv": project_path(all_csv),
            "train_csv": project_path(train_csv),
            "val_csv": project_path(val_csv),
            "test_csv": project_path(test_csv),
            "label_map": project_path(DEFAULT_LABEL_MAP),
            "backend_mappings": project_path(DEFAULT_BACKEND_MAPPINGS),
            "frontend_mappings": project_path(DEFAULT_FRONTEND_MAPPINGS),
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "dataset_summary.json", summary)
    return summary


def main():
    dataset_config = load_dataset_config()
    configured_raw_dir = config_path(dataset_config, "raw_class_dir", DEFAULT_RAW_DIR)
    configured_processed_dir = config_path(
        dataset_config,
        "processed_dir",
        DEFAULT_PROCESSED_DIR,
    )
    configured_splits_dir = config_path(dataset_config, "splits_dir", DEFAULT_SPLITS_DIR)
    configured_reports_dir = config_path(dataset_config, "reports_dir", DEFAULT_REPORTS_DIR)
    split_ratios = dataset_config.get("split_ratios", {})

    parser = ArgumentParser(description="Prepare the SignBridge ISL dataset.")
    parser.add_argument(
        "--raw-dir",
        default=str(configured_raw_dir),
        help="Raw Kaggle ISL class directory. Defaults to the downloaded dataset path.",
    )
    parser.add_argument("--processed-dir", default=str(configured_processed_dir))
    parser.add_argument("--splits-dir", default=str(configured_splits_dir))
    parser.add_argument("--reports-dir", default=str(configured_reports_dir))
    parser.add_argument("--train-ratio", type=float, default=split_ratios.get("train", 0.7))
    parser.add_argument("--val-ratio", type=float, default=split_ratios.get("val", 0.15))
    parser.add_argument("--test-ratio", type=float, default=split_ratios.get("test", 0.15))
    parser.add_argument("--seed", type=int, default=dataset_config.get("seed", 258))
    parser.add_argument("--allow-extra-folders", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=0,
        help="If greater than 0, create training-only augmented images after preprocessing.",
    )
    args = parser.parse_args()

    summary = process_dataset(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        splits_dir=Path(args.splits_dir),
        reports_dir=Path(args.reports_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        allow_extra_folders=args.allow_extra_folders,
        fail_on_invalid=args.fail_on_invalid,
    )

    if args.augment_copies > 0:
        from augment import augment_training_set

        augmented_summary = augment_training_set(
            train_csv=Path(args.splits_dir) / "train.csv",
            output_dir=Path(args.processed_dir).parent / "isl_augmented",
            copies=args.augment_copies,
            seed=args.seed,
            reports_dir=Path(args.reports_dir),
        )
        summary["augmentation"] = augmented_summary
        write_json(Path(args.reports_dir) / "dataset_summary.json", summary)

    print(json.dumps(summary["split_counts"], indent=2))
    print(f"Wrote report: {Path(args.reports_dir) / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
