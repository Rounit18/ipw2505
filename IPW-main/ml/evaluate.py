import json
from argparse import ArgumentParser

from training_utils import (
    empty_confusion_matrix,
    format_percent,
    import_torch,
    invert_label_map,
    load_label_map,
    load_training_config,
    make_dataloader,
    metrics_from_confusion,
    project_path,
    resolve_project_path,
    resolve_device,
    save_confusion_image,
    summarize_weak_classes,
    update_confusion_matrix,
    write_confusion_csv,
    write_json,
    write_weak_class_report,
)


def load_model_class():
    from src.model import SignBridgeCNN

    return SignBridgeCNN


def evaluate(config_path, model_path=None, split_csv=None, output_prefix="test"):
    torch, _, _ = import_torch()
    SignBridgeCNN = load_model_class()
    config = load_training_config(config_path)
    label_map_path = resolve_project_path(config["label_map"])
    label_map = load_label_map(label_map_path)
    index_to_class = invert_label_map(label_map)
    num_classes = int(config.get("num_classes", len(label_map)))
    top_k = int(config.get("top_k", 3))
    recall_target = float(config.get("min_recall_target", 0.85))
    device = resolve_device(config.get("device", "auto"))
    model_path = resolve_project_path(model_path or config["model_output"])
    split_csv = resolve_project_path(split_csv or config["test_csv"])
    reports_dir = resolve_project_path(config.get("reports_dir", "ml/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    model = SignBridgeCNN(num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dataloader = make_dataloader(split_csv, label_map, config, shuffle=False)
    matrix = empty_confusion_matrix(num_classes)
    total = 0
    correct = 0
    topk_correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            _, top_indices = torch.topk(logits, k=min(top_k, num_classes), dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            topk_correct += top_indices.eq(labels.view(-1, 1)).any(dim=1).sum().item()
            update_confusion_matrix(
                matrix,
                labels.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
            )

    metrics = metrics_from_confusion(matrix)
    per_class = {}
    min_recall = 1.0
    for index, item in metrics["per_class"].items():
        class_name = index_to_class[index]
        min_recall = min(min_recall, item["recall"])
        per_class[class_name] = {
            "support": item["support"],
            "true_positive": item["true_positive"],
            "precision": round(item["precision"], 6),
            "recall": round(item["recall"], 6),
        }

    weak_classes = summarize_weak_classes(
        matrix,
        index_to_class,
        recall_target=recall_target,
    )

    confusion_csv = reports_dir / f"{output_prefix}_confusion_matrix.csv"
    confusion_png = reports_dir / f"{output_prefix}_confusion_matrix.png"
    weak_report = reports_dir / f"{output_prefix}_weak_classes.md"
    write_confusion_csv(confusion_csv, matrix, index_to_class)
    save_confusion_image(confusion_png, matrix, index_to_class)
    write_weak_class_report(weak_report, weak_classes, recall_target)

    summary = {
        "split_csv": project_path(split_csv),
        "model_path": project_path(model_path),
        "records": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        f"top{top_k}_accuracy": topk_correct / total if total else 0.0,
        "min_recall": min_recall if total else 0.0,
        "recall_target": recall_target,
        "weak_class_count": len(weak_classes),
        "weak_classes": weak_classes,
        "per_class": per_class,
        "generated_files": {
            "confusion_csv": project_path(confusion_csv),
            "confusion_png": project_path(confusion_png),
            "weak_report": project_path(weak_report),
        },
    }
    write_json(reports_dir / f"{output_prefix}_evaluation.json", summary)
    return summary


def main():
    parser = ArgumentParser(description="Evaluate SignBridgeCNN.")
    parser.add_argument("--config", default="ml/configs/model_config.json")
    parser.add_argument("--model", help="Path to isl_cnn.pth.")
    parser.add_argument("--split-csv", help="CSV split to evaluate. Defaults to configured test_csv.")
    parser.add_argument("--output-prefix", default="test")
    args = parser.parse_args()
    summary = evaluate(
        config_path=args.config,
        model_path=args.model,
        split_csv=args.split_csv,
        output_prefix=args.output_prefix,
    )
    print(
        json.dumps(
            {
                "records": summary["records"],
                "accuracy": summary["accuracy"],
                "top3_accuracy": summary.get("top3_accuracy"),
                "min_recall": summary["min_recall"],
                "weak_class_count": summary["weak_class_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
