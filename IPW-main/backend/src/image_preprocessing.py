import base64
import binascii
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "preprocessing_config.json"


def load_preprocessing_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


PREPROCESSING_CONFIG = load_preprocessing_config()
IMAGE_SIZE = tuple(PREPROCESSING_CONFIG["image_size"])


def decode_base64_image(image_base64):
    if not isinstance(image_base64, str):
        raise ValueError("Image payload must be a base64 string.")

    if image_base64.startswith("data:") and "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    try:
        raw = base64.b64decode("".join(image_base64.split()), validate=True)
        image = Image.open(io.BytesIO(raw))
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("Only PNG and JPEG frames are supported.")
        return image.convert(PREPROCESSING_CONFIG["color_mode"])
    except (binascii.Error, ValueError, UnidentifiedImageError) as exc:
        raise ValueError("Invalid base64 image payload.") from exc


def preprocess_pil_to_chw_float(image):
    image = image.convert(PREPROCESSING_CONFIG["color_mode"])
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))
