import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

from class_config import EXPECTED_CLASS_NAMES, class_category, default_description


ROOT = Path(__file__).resolve().parents[1]
BACKEND_MAPPINGS = ROOT / "backend" / "data" / "mappings.json"
FRONTEND_MAPPINGS = ROOT / "frontend" / "src" / "data" / "mappings.json"
LABEL_MAP = ROOT / "ml" / "configs" / "label_map.json"


def _load_existing(path):
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        return {}

    return {entry.get("class_name"): entry for entry in data if "class_name" in entry}


def build_mappings(existing_path=BACKEND_MAPPINGS):
    existing = _load_existing(existing_path)
    mappings = []

    for gesture_id, class_name in enumerate(EXPECTED_CLASS_NAMES):
        previous = existing.get(class_name, {})
        mappings.append(
            {
                "gesture_id": gesture_id,
                "class_name": class_name,
                "category": class_category(class_name),
                "english": previous.get("english", class_name),
                "hindi": previous.get("hindi", class_name),
                "kannada": previous.get("kannada", class_name),
                "word_builder_value": previous.get("word_builder_value", class_name),
                "description": previous.get("description", default_description(class_name)),
                "notes": previous.get("notes"),
            }
        )

    return mappings


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_label_map(path):
    label_map = {class_name: index for index, class_name in enumerate(EXPECTED_CLASS_NAMES)}
    write_json(path, label_map)
    return label_map


def main():
    parser = ArgumentParser(description="Generate canonical SignBridge mappings.")
    parser.add_argument("--backend-output", default=str(BACKEND_MAPPINGS))
    parser.add_argument("--frontend-output", default=str(FRONTEND_MAPPINGS))
    parser.add_argument("--label-map-output", default=str(LABEL_MAP))
    parser.add_argument(
        "--existing",
        default=str(BACKEND_MAPPINGS),
        help="Existing mappings file to preserve validated labels/descriptions from.",
    )
    parser.add_argument(
        "--skip-frontend-copy",
        action="store_true",
        help="Generate only backend mappings and label_map.",
    )
    args = parser.parse_args()

    backend_output = Path(args.backend_output)
    frontend_output = Path(args.frontend_output)
    label_map_output = Path(args.label_map_output)
    mappings = build_mappings(existing_path=Path(args.existing))

    if len(mappings) != 35:
        raise ValueError(f"Expected 35 mappings, found {len(mappings)}.")

    write_json(backend_output, mappings)
    write_label_map(label_map_output)

    if not args.skip_frontend_copy:
        frontend_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(backend_output, frontend_output)

    print(f"Wrote mappings: {backend_output}")
    print(f"Wrote label map: {label_map_output}")
    if not args.skip_frontend_copy:
        print(f"Copied frontend fallback mappings: {frontend_output}")


if __name__ == "__main__":
    main()
