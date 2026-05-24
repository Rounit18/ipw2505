import csv
import json
import random
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

from image_preprocessing import IMAGE_SIZE


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = ROOT / "ml" / "reports"


def project_path(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_record_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def read_records(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_records(path, records):
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


def corner_fill(image):
    return image.getpixel((0, 0))


def random_zoom(image, rng, limit=0.1):
    width, height = image.size
    factor = rng.uniform(1 - limit, 1 + limit)

    if factor >= 1.0:
        crop_width = max(1, int(width / factor))
        crop_height = max(1, int(height / factor))
        left = rng.randint(0, width - crop_width)
        top = rng.randint(0, height - crop_height)
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.Resampling.LANCZOS)

    resized = image.resize(
        (max(1, int(width * factor)), max(1, int(height * factor))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), corner_fill(image))
    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def augment_image(image, rng):
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    image = image.rotate(
        rng.uniform(-15, 15),
        resample=Image.Resampling.BICUBIC,
        fillcolor=corner_fill(image),
    )
    image = random_zoom(image, rng, limit=0.1)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    return image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def augment_training_set(train_csv, output_dir, copies=3, seed=258, reports_dir=DEFAULT_REPORTS_DIR):
    train_csv = train_csv.resolve()
    output_dir = output_dir.resolve()
    reports_dir = reports_dir.resolve()

    records = [
        record
        for record in read_records(train_csv)
        if record.get("split", "train") == "train" and record.get("is_augmented", "0") == "0"
    ]

    rng = random.Random(seed)
    augmented_records = []

    for record in records:
        source_path = resolve_record_path(record["path"])
        class_name = record["class_name"]

        with Image.open(source_path) as image:
            for copy_index in range(copies):
                augmented = augment_image(image, rng)
                target = (
                    output_dir
                    / class_name
                    / f"{source_path.stem}__aug_{copy_index + 1:02d}.png"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                augmented.save(target, format="PNG")

                augmented_records.append(
                    {
                        **record,
                        "record_id": f"{record['record_id']}-aug-{copy_index + 1:02d}",
                        "path": project_path(target),
                        "source_path": record["path"],
                        "is_augmented": "1",
                    }
                )

    augmented_csv = train_csv.parent / "train_augmented_only.csv"
    combined_csv = train_csv.parent / "train_with_augmented.csv"
    write_records(augmented_csv, augmented_records)
    write_records(combined_csv, records + augmented_records)

    summary = {
        "train_csv": project_path(train_csv),
        "output_dir": project_path(output_dir),
        "copies_per_image": copies,
        "seed": seed,
        "original_train_images": len(records),
        "augmented_images": len(augmented_records),
        "combined_train_images": len(records) + len(augmented_records),
        "generated_files": {
            "augmented_only_csv": project_path(augmented_csv),
            "combined_train_csv": project_path(combined_csv),
        },
    }
    write_json(reports_dir / "augmentation_summary.json", summary)
    return summary


def main():
    parser = ArgumentParser(description="Training-only augmentation entry point.")
    parser.add_argument("--train-csv", required=True, help="Path to train.csv from preprocess.py.")
    parser.add_argument("--output-dir", required=True, help="Augmented image output directory.")
    parser.add_argument("--copies", type=int, default=3, help="Augmented copies per source image.")
    parser.add_argument("--seed", type=int, default=258)
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    args = parser.parse_args()

    summary = augment_training_set(
        train_csv=Path(args.train_csv),
        output_dir=Path(args.output_dir),
        copies=args.copies,
        seed=args.seed,
        reports_dir=Path(args.reports_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
