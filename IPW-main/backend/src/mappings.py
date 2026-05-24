import hashlib
import json


REQUIRED_FIELDS = {
    "gesture_id",
    "class_name",
    "category",
    "english",
    "hindi",
    "kannada",
    "word_builder_value",
    "description",
}


def load_mappings(path):
    with open(path, "r", encoding="utf-8") as handle:
        mappings = json.load(handle)

    if not isinstance(mappings, list):
        raise ValueError("mappings.json must contain a list.")

    for index, entry in enumerate(mappings):
        missing = REQUIRED_FIELDS.difference(entry.keys())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Mapping at index {index} is missing: {missing_list}")
        if entry["gesture_id"] != index:
            raise ValueError(
                f"Mapping index drift: expected gesture_id {index}, "
                f"found {entry['gesture_id']}."
            )

    return mappings


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()

