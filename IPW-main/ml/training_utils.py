import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for training. Install backend requirements first."
        ) from exc

    return torch, Dataset, DataLoader


def resolve_device(requested_device="auto"):
    torch, _, _ = import_torch()
    requested = str(requested_device or "auto").lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    return torch.device(requested)


def load_json(path):
    path = resolve_project_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def project_path(path):
    path = resolve_project_path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path):
    path = resolve_project_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_records(path):
    path = resolve_project_path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_label_map(path):
    data = load_json(path)
    return {class_name: int(index) for class_name, index in data.items()}


def invert_label_map(label_map):
    return {index: class_name for class_name, index in label_map.items()}


def load_training_config(path):
    config = load_json(path)
    config["_config_path"] = project_path(path)
    return config


def model_input_size(config):
    size = config.get("input_size", [64, 64])
    return int(size[0]), int(size[1])


def build_transform(config):
    torch, _, _ = import_torch()
    import numpy as np

    width, height = model_input_size(config)

    def transform(path):
        image = Image.open(resolve_project_path(path)).convert("RGB")
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    return transform


def make_dataset_class():
    torch, Dataset, _ = import_torch()

    class SignBridgeCsvDataset(Dataset):
        def __init__(self, csv_path, label_map, transform):
            self.csv_path = resolve_project_path(csv_path)
            self.records = read_csv_records(self.csv_path)
            self.label_map = label_map
            self.transform = transform

            if not self.records:
                raise ValueError(f"No records found in {self.csv_path}")

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            record = self.records[index]
            image = self.transform(record["path"])
            label = int(record.get("gesture_id") or self.label_map[record["class_name"]])
            return image, torch.tensor(label, dtype=torch.long)

    return SignBridgeCsvDataset


def make_dataloader(csv_path, label_map, config, shuffle):
    _, _, DataLoader = import_torch()
    DatasetClass = make_dataset_class()
    dataset = DatasetClass(csv_path, label_map, build_transform(config))
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
    )


def empty_confusion_matrix(num_classes):
    return [[0 for _ in range(num_classes)] for _ in range(num_classes)]


def update_confusion_matrix(matrix, labels, predictions):
    for actual, predicted in zip(labels, predictions):
        matrix[int(actual)][int(predicted)] += 1


def add_confusion_matrices(left, right):
    for row_index, row in enumerate(right):
        for col_index, value in enumerate(row):
            left[row_index][col_index] += value
    return left


def metrics_from_confusion(matrix):
    num_classes = len(matrix)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(num_classes))
    per_class = {}

    for index in range(num_classes):
        actual_total = sum(matrix[index])
        predicted_total = sum(row[index] for row in matrix)
        true_positive = matrix[index][index]
        recall = true_positive / actual_total if actual_total else 0.0
        precision = true_positive / predicted_total if predicted_total else 0.0
        per_class[index] = {
            "support": actual_total,
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
        }

    return {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "per_class": per_class,
    }


def format_percent(value):
    return f"{value * 100:.2f}%"


def write_history_csv(path, history):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_top3_accuracy",
        "learning_rate",
        "epoch_seconds",
        "is_best",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_confusion_csv(path, matrix, index_to_class):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [index_to_class[index] for index in range(len(matrix))]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *labels])
        for index, row in enumerate(matrix):
            writer.writerow([labels[index], *row])


def save_confusion_image(path, matrix, index_to_class):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = len(matrix)
    cell = 22
    margin_left = 82
    margin_top = 82
    width = margin_left + num_classes * cell + 20
    height = margin_top + num_classes * cell + 30
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    max_value = max((max(row) for row in matrix), default=1) or 1

    for index in range(num_classes):
        label = index_to_class[index]
        x = margin_left + index * cell
        y = margin_top + index * cell
        draw.text((x + 5, 10), label, fill="black", font=font)
        draw.text((10, y + 5), label, fill="black", font=font)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = int(255 - (value / max_value) * 210)
            color = (intensity, intensity, 255)
            left = margin_left + col_index * cell
            top = margin_top + row_index * cell
            draw.rectangle(
                [left, top, left + cell - 1, top + cell - 1],
                fill=color,
                outline=(220, 220, 220),
            )
            if row_index == col_index and value:
                draw.text((left + 4, top + 5), str(value), fill="black", font=font)

    draw.text((margin_left, height - 20), "Rows = actual, columns = predicted", fill="black")
    image.save(path)


def summarize_weak_classes(matrix, index_to_class, recall_target=0.85):
    metrics = metrics_from_confusion(matrix)
    weak = []

    for index, item in metrics["per_class"].items():
        recall = item["recall"]
        if recall >= recall_target:
            continue

        row = matrix[index]
        confusions = []
        for predicted_index, count in sorted(
            enumerate(row),
            key=lambda entry: entry[1],
            reverse=True,
        ):
            if predicted_index == index or count == 0:
                continue
            confusions.append(
                {
                    "predicted_class": index_to_class[predicted_index],
                    "count": count,
                }
            )

        weak.append(
            {
                "class_name": index_to_class[index],
                "recall": recall,
                "support": item["support"],
                "top_confusions": confusions[:3],
            }
        )

    return sorted(weak, key=lambda item: item["recall"])


def recommended_augmentation_for_class(item):
    recs = [
        "collect or inspect real webcam examples for this class",
        "add targeted brightness and contrast variation",
        "review hand centering and crop consistency",
    ]
    if item["top_confusions"]:
        confusing = ", ".join(confusion["predicted_class"] for confusion in item["top_confusions"])
        recs.append(f"add pairwise augmentation against confused classes: {confusing}")
    return recs


def write_weak_class_report(path, weak_classes, recall_target):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Weak Class Report",
        "",
        f"Recall target: {format_percent(recall_target)}",
        "",
    ]

    if not weak_classes:
        lines.append("All classes meet the recall target.")
    else:
        for item in weak_classes:
            lines.extend(
                [
                    f"## {item['class_name']}",
                    "",
                    f"- Recall: {format_percent(item['recall'])}",
                    f"- Support: {item['support']}",
                ]
            )
            if item["top_confusions"]:
                confusions = ", ".join(
                    f"{entry['predicted_class']} ({entry['count']})"
                    for entry in item["top_confusions"]
                )
                lines.append(f"- Top confusions: {confusions}")
            lines.append("- Recommended changes:")
            for recommendation in recommended_augmentation_for_class(item):
                lines.append(f"  - {recommendation}")
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_counts_by_class(csv_path):
    counts = defaultdict(int)
    for record in read_csv_records(csv_path):
        counts[record["class_name"]] += 1
    return dict(sorted(counts.items()))
