from string import ascii_uppercase


EXPECTED_CLASS_NAMES = list(ascii_uppercase) + [str(value) for value in range(1, 10)]
CLASS_TO_ID = {class_name: index for index, class_name in enumerate(EXPECTED_CLASS_NAMES)}
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_class_name(value):
    normalized = value.strip().upper()
    if normalized in CLASS_TO_ID:
        return normalized
    return value.strip()


def class_category(class_name):
    return "digit" if class_name.isdigit() else "alphabet"


def default_description(class_name):
    if class_name in {"J", "Z"}:
        return (
            f"Static dataset proxy label for {class_name}. "
            f"True dynamic {class_name} motion is out of v1 scope."
        )
    if class_name.isdigit():
        return f"Static dataset gesture for digit {class_name}."
    return f"Static dataset gesture for {class_name}."

