import json
from pathlib import Path

from PIL import Image

from class_config import VALID_IMAGE_SUFFIXES


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "preprocessing_config.json"


def load_preprocessing_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


PREPROCESSING_CONFIG = load_preprocessing_config()
IMAGE_SIZE = tuple(PREPROCESSING_CONFIG["image_size"])


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES


def iter_image_paths(folder):
    return sorted(path for path in folder.rglob("*") if is_image_file(path))


def load_rgb_image(path):
    return Image.open(path).convert(PREPROCESSING_CONFIG["color_mode"])


def resize_for_model(image):
    return image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def preprocess_image_file(source_path, target_path):
    image = resize_for_model(load_rgb_image(source_path))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(target_path, format="PNG")
    return target_path

