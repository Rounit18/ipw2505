import json
import platform
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

from training_utils import (
    ROOT,
    add_confusion_matrices,
    empty_confusion_matrix,
    format_percent,
    group_counts_by_class,
    import_torch,
    invert_label_map,
    load_label_map,
    load_training_config,
    make_dataloader,
    metrics_from_confusion,
    project_path,
    resolve_project_path,
    resolve_device,
    sha256_file,
    summarize_weak_classes,
    update_confusion_matrix,
    write_confusion_csv,
    write_history_csv,
    write_json,
    write_weak_class_report,
)


def current_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    torch, _, _ = import_torch()
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    matrix = empty_confusion_matrix(num_classes)

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = torch.argmax(logits, dim=1)
        total += batch_size
        correct += (predictions == labels).sum().item()
        update_confusion_matrix(
            matrix,
            labels.detach().cpu().tolist(),
            predictions.detach().cpu().tolist(),
        )

    return {
        "loss": running_loss / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "confusion_matrix": matrix,
    }


def evaluate_split(model, dataloader, criterion, device, num_classes, top_k=3):
    torch, _, _ = import_torch()
    model.eval()
    running_loss = 0.0
    total = 0
    top1_correct = 0
    topk_correct = 0
    matrix = empty_confusion_matrix(num_classes)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            predictions = torch.argmax(logits, dim=1)
            _, top_indices = torch.topk(logits, k=min(top_k, num_classes), dim=1)

            total += batch_size
            top1_correct += (predictions == labels).sum().item()
            topk_correct += top_indices.eq(labels.view(-1, 1)).any(dim=1).sum().item()
            update_confusion_matrix(
                matrix,
                labels.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
            )

    return {
        "loss": running_loss / total if total else 0.0,
        "accuracy": top1_correct / total if total else 0.0,
        f"top{top_k}_accuracy": topk_correct / total if total else 0.0,
        "confusion_matrix": matrix,
    }


def save_checkpoint(path, model, optimizer, epoch, best_val_accuracy, config):
    torch, _, _ = import_torch()
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_accuracy": best_val_accuracy,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def load_model_class():
    from src.model import SignBridgeCNN

    return SignBridgeCNN


def train(config_path, epochs_override=None, batch_size_override=None, evaluate_test=True):
    torch, _, _ = import_torch()
    SignBridgeCNN = load_model_class()
    config = load_training_config(config_path)

    if epochs_override is not None:
        config["epochs"] = epochs_override
    if batch_size_override is not None:
        config["batch_size"] = batch_size_override

    reports_dir = resolve_project_path(config.get("reports_dir", "ml/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_output = resolve_project_path(config["model_output"])
    metadata_output = model_output.with_name("model_metadata.json")
    checkpoint_output = model_output.with_name("isl_cnn_checkpoint.pth")
    label_map_path = resolve_project_path(config["label_map"])
    train_csv = resolve_project_path(config["train_csv"])
    val_csv = resolve_project_path(config["val_csv"])
    test_csv = resolve_project_path(config["test_csv"])

    label_map = load_label_map(label_map_path)
    index_to_class = invert_label_map(label_map)
    num_classes = int(config.get("num_classes", len(label_map)))
    device = resolve_device(config.get("device", "auto"))
    recall_target = float(config.get("min_recall_target", 0.85))
    top_k = int(config.get("top_k", 3))

    train_loader = make_dataloader(train_csv, label_map, config, shuffle=True)
    val_loader = make_dataloader(val_csv, label_map, config, shuffle=False)

    model = SignBridgeCNN(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.001)))

    best_val_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    best_val_matrix = empty_confusion_matrix(num_classes)
    patience = int(config.get("early_stopping_patience", 5))
    started = time.time()

    print(f"Training on {device} with {len(train_loader.dataset)} train records.")
    print(f"Validation records: {len(val_loader.dataset)}")

    for epoch in tqdm(range(1, int(config.get("epochs", 30)) + 1), desc="Epochs", unit="epoch"):
        epoch_started = time.time()
        train_result = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes,
        )
        val_result = evaluate_split(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            top_k=top_k,
        )

        is_best = val_result["accuracy"] > best_val_accuracy
        if is_best:
            best_val_accuracy = val_result["accuracy"]
            best_epoch = epoch
            stale_epochs = 0
            best_val_matrix = val_result["confusion_matrix"]
            model_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_output)
            save_checkpoint(
                checkpoint_output,
                model,
                optimizer,
                epoch,
                best_val_accuracy,
                config,
            )
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch,
            "train_loss": round(train_result["loss"], 6),
            "train_accuracy": round(train_result["accuracy"], 6),
            "val_loss": round(val_result["loss"], 6),
            "val_accuracy": round(val_result["accuracy"], 6),
            "val_top3_accuracy": round(val_result[f"top{top_k}_accuracy"], 6),
            "learning_rate": current_learning_rate(optimizer),
            "epoch_seconds": round(time.time() - epoch_started, 2),
            "is_best": is_best,
        }
        history.append(row)

        print(
            f"Epoch {epoch}: "
            f"train_acc={format_percent(train_result['accuracy'])} "
            f"val_acc={format_percent(val_result['accuracy'])} "
            f"val_top{top_k}={format_percent(val_result[f'top{top_k}_accuracy'])} "
            f"best={format_percent(best_val_accuracy)}"
        )

        if stale_epochs >= patience:
            print(f"Early stopping after {stale_epochs} stale epochs.")
            break

    write_history_csv(reports_dir / "training_history.csv", history)
    write_json(reports_dir / "training_history.json", history)

    val_metrics = metrics_from_confusion(best_val_matrix)
    weak_val_classes = summarize_weak_classes(
        best_val_matrix,
        index_to_class,
        recall_target=recall_target,
    )
    write_confusion_csv(reports_dir / "validation_confusion_matrix.csv", best_val_matrix, index_to_class)
    write_weak_class_report(reports_dir / "validation_weak_classes.md", weak_val_classes, recall_target)

    metadata = {
        "model_name": "SignBridgeCNN",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "num_classes": num_classes,
        "input_size": config.get("input_size", [64, 64]),
        "preprocessing_config": project_path("preprocessing_config.json"),
        "preprocessing_hash": sha256_file("preprocessing_config.json"),
        "label_map": project_path(label_map_path),
        "label_map_hash": sha256_file(label_map_path),
        "train_csv": project_path(train_csv),
        "val_csv": project_path(val_csv),
        "test_csv": project_path(test_csv),
        "train_records": len(train_loader.dataset),
        "val_records": len(val_loader.dataset),
        "test_records": len(group_counts_by_class(test_csv)) and sum(group_counts_by_class(test_csv).values()),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_recall_min": min(
            (item["recall"] for item in val_metrics["per_class"].values()),
            default=0.0,
        ),
        "epochs_completed": len(history),
        "training_seconds": round(time.time() - started, 2),
        "state_dict_path": project_path(model_output),
        "checkpoint_path": project_path(checkpoint_output),
        "reports_dir": project_path(reports_dir),
        "class_counts": {
            "train": group_counts_by_class(train_csv),
            "val": group_counts_by_class(val_csv),
            "test": group_counts_by_class(test_csv),
        },
    }

    test_summary = None
    if evaluate_test:
        from evaluate import evaluate

        test_summary = evaluate(
            config_path=config_path,
            model_path=model_output,
            split_csv=test_csv,
            output_prefix="test",
        )
        metadata["test_accuracy"] = test_summary["accuracy"]
        metadata["test_top3_accuracy"] = test_summary[f"top{top_k}_accuracy"]
        metadata["test_recall_min"] = test_summary["min_recall"]

    write_json(metadata_output, metadata)
    write_json(reports_dir / "training_summary.json", metadata)

    print(f"Saved best state_dict: {model_output}")
    print(f"Saved metadata: {metadata_output}")
    return metadata


def main():
    parser = ArgumentParser(description="Train SignBridgeCNN.")
    parser.add_argument("--config", default="ml/configs/model_config.json")
    parser.add_argument("--epochs", type=int, help="Override configured epochs.")
    parser.add_argument("--batch-size", type=int, help="Override configured batch size.")
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip final test evaluation after training.",
    )
    args = parser.parse_args()
    train(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        evaluate_test=not args.skip_test_eval,
    )


if __name__ == "__main__":
    main()
