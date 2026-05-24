import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app
from src.errors import LowConfidenceError, ModelUnavailableError


def make_mapping(index=0, class_name="A"):
    return {
        "gesture_id": index,
        "class_name": class_name,
        "category": "alphabet",
        "english": class_name,
        "hindi": class_name,
        "kannada": class_name,
        "word_builder_value": class_name,
        "description": f"Static dataset gesture for {class_name}.",
    }


def make_mappings():
    letters = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    digits = [str(value) for value in range(1, 10)]
    entries = []
    for index, class_name in enumerate(letters + digits):
        entry = make_mapping(index, class_name)
        entry["category"] = "digit" if class_name.isdigit() else "alphabet"
        entries.append(entry)
    return entries


class FakePredictor:
    model_loaded = True
    model_error = None
    device_name = "cpu"
    torch_version = "test"

    def predict_base64(self, image, threshold=0.8, top_k=3):
        return {
            "gesture_id": 0,
            "class_name": "A",
            "english": "A",
            "hindi": "A",
            "kannada": "A",
            "word_builder_value": "A",
            "description": "Static dataset gesture for A.",
            "confidence": 0.99,
            "threshold": threshold,
            "top_k": [
                {
                    "gesture_id": 0,
                    "class_name": "A",
                    "confidence": 0.99,
                }
            ][:top_k],
            "latency_ms": 1.23,
        }


class LowConfidencePredictor(FakePredictor):
    def predict_base64(self, image, threshold=0.8, top_k=3):
        raise LowConfidenceError(
            confidence=0.42,
            threshold=threshold,
            details={
                "confidence": 0.42,
                "threshold": threshold,
                "top_k": [{"class_name": "B", "confidence": 0.42}],
            },
        )


class MissingModelPredictor(FakePredictor):
    model_loaded = False
    model_error = "Model file not found."

    def predict_base64(self, image, threshold=0.8, top_k=3):
        raise ModelUnavailableError("Model is not loaded.")


def make_runtime(predictor=None, mappings=None, mappings_error=None):
    return {
        "mappings": make_mappings() if mappings is None else mappings,
        "mappings_error": mappings_error,
        "mappings_hash": "test-hash" if mappings_error is None else None,
        "model_path": "test-model-path",
        "mappings_path": "test-mappings-path",
        "predictor": predictor or FakePredictor(),
    }


def test_health_returns_status():
    client = create_app(runtime=make_runtime()).test_client()
    response = client.get("/health")
    data = response.get_json()

    assert response.status_code == 200
    assert data["success"] is True
    assert data["data"]["status"] == "ok"
    assert data["data"]["model_loaded"] is True
    assert data["data"]["mappings_loaded"] is True


def test_health_reports_degraded_when_model_missing():
    client = create_app(runtime=make_runtime(predictor=MissingModelPredictor())).test_client()
    response = client.get("/health")
    data = response.get_json()

    assert response.status_code == 200
    assert data["success"] is True
    assert data["data"]["status"] == "degraded"
    assert data["data"]["model_loaded"] is False


def test_classes_returns_35_entries():
    client = create_app(runtime=make_runtime()).test_client()
    response = client.get("/classes")
    data = response.get_json()

    assert response.status_code == 200
    assert data["success"] is True
    assert data["data"]["count"] == 35
    assert data["data"]["classes"][0]["class_name"] == "A"


def test_classes_returns_503_when_mappings_unavailable():
    client = create_app(
        runtime=make_runtime(mappings=[], mappings_error="bad mappings")
    ).test_client()
    response = client.get("/classes")
    data = response.get_json()

    assert response.status_code == 503
    assert data["success"] is False
    assert data["error"]["code"] == "mappings_unavailable"


def test_predict_requires_image_field():
    client = create_app(runtime=make_runtime()).test_client()
    response = client.post("/predict", json={})
    data = response.get_json()

    assert response.status_code == 400
    assert data["success"] is False
    assert data["error"]["code"] == "bad_request"


def test_predict_requires_json_body():
    client = create_app(runtime=make_runtime()).test_client()
    response = client.post("/predict", data="not json", content_type="text/plain")
    data = response.get_json()

    assert response.status_code == 400
    assert data["success"] is False
    assert data["error"]["message"] == "Request body must be JSON."


def test_predict_rejects_empty_image_string():
    client = create_app(runtime=make_runtime()).test_client()
    response = client.post("/predict", json={"image": "   "})
    data = response.get_json()

    assert response.status_code == 400
    assert data["success"] is False
    assert data["error"]["message"] == "'image' must be a non-empty base64 string."


def test_predict_returns_prediction_payload():
    client = create_app(runtime=make_runtime()).test_client()
    response = client.post(
        "/predict",
        json={"image": "base64-frame", "threshold": 0.75, "top_k": 1},
    )
    data = response.get_json()

    assert response.status_code == 200
    assert data["success"] is True
    assert data["data"]["class_name"] == "A"
    assert data["data"]["confidence"] == 0.99
    assert len(data["data"]["top_k"]) == 1


def test_predict_returns_422_for_low_confidence():
    client = create_app(runtime=make_runtime(predictor=LowConfidencePredictor())).test_client()
    response = client.post("/predict", json={"image": "base64-frame", "threshold": 0.8})
    data = response.get_json()

    assert response.status_code == 422
    assert data["success"] is False
    assert data["error"]["code"] == "low_confidence"
    assert data["error"]["details"]["confidence"] == 0.42


def test_predict_returns_503_when_model_missing():
    client = create_app(runtime=make_runtime(predictor=MissingModelPredictor())).test_client()
    response = client.post("/predict", json={"image": "base64-frame"})
    data = response.get_json()

    assert response.status_code == 503
    assert data["success"] is False
    assert data["error"]["code"] == "model_unavailable"


def test_unknown_route_returns_json_error():
    client = create_app(runtime=make_runtime()).test_client()
    response = client.get("/missing")
    data = response.get_json()

    assert response.status_code == 404
    assert data["success"] is False
    assert data["error"]["code"] == "http_error"


def test_cors_allows_local_vite_origin():
    client = create_app(runtime=make_runtime()).test_client()
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
