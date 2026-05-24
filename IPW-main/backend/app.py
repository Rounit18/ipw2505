import os
import time
from datetime import datetime, timezone

from werkzeug.exceptions import HTTPException
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.errors import ApiError, error_payload
from src.image_preprocessing import PREPROCESSING_CONFIG
from src.mappings import load_mappings, sha256_file
from src.predict import GesturePredictor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPINGS_PATH = os.path.join(BASE_DIR, "data", "mappings.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "isl_cnn.pth")
START_TIME = time.time()
LOCAL_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def success_response(data, status_code=200):
    return jsonify({"success": True, "data": data}), status_code


def _load_runtime(model_path=MODEL_PATH, mappings_path=MAPPINGS_PATH):
    runtime = {
        "mappings": [],
        "mappings_error": None,
        "mappings_hash": None,
        "model_path": model_path,
        "mappings_path": mappings_path,
    }

    try:
        runtime["mappings"] = load_mappings(mappings_path)
        runtime["mappings_hash"] = sha256_file(mappings_path)
    except Exception as exc:
        runtime["mappings_error"] = str(exc)

    runtime["predictor"] = GesturePredictor(
        model_path=model_path,
        mappings=runtime["mappings"],
    )
    return runtime


def create_app(runtime=None, cors_origins=None):
    app = Flask(__name__)
    runtime = runtime or _load_runtime()
    cors_origins = cors_origins or LOCAL_CORS_ORIGINS

    CORS(
        app,
        resources={
            r"/*": {
                "origins": cors_origins,
            }
        },
    )

    @app.get("/health")
    def health():
        mappings_loaded = bool(runtime["mappings"]) and runtime["mappings_error"] is None
        predictor = runtime["predictor"]
        model_loaded = predictor.model_loaded
        status = "ok" if model_loaded and mappings_loaded else "degraded"

        return success_response(
            {
                "status": status,
                "model_loaded": model_loaded,
                "model_error": predictor.model_error,
                "mappings_loaded": mappings_loaded,
                "mappings_error": runtime["mappings_error"],
                "device": predictor.device_name,
                "num_classes": len(runtime["mappings"]),
                "pytorch_version": predictor.torch_version,
                "uptime_seconds": round(time.time() - START_TIME, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "expected_input_size": PREPROCESSING_CONFIG["image_size"],
                "preprocessing_version": PREPROCESSING_CONFIG["version"],
                "mappings_hash": runtime["mappings_hash"],
            }
        )

    @app.get("/classes")
    def classes():
        if runtime["mappings_error"] is not None:
            raise ApiError(
                "Mappings are not loaded.",
                status_code=503,
                code="mappings_unavailable",
                details={"message": runtime["mappings_error"]},
            )

        return success_response(
            {
                "count": len(runtime["mappings"]),
                "classes": runtime["mappings"],
            }
        )

    @app.post("/predict")
    def predict():
        body = parse_json_body()
        image = body.get("image")
        threshold = body.get("threshold", 0.8)
        top_k = body.get("top_k", 3)

        if "image" not in body:
            raise ApiError("Missing 'image' field in request body.", status_code=400)
        if not isinstance(image, str) or not image.strip():
            raise ApiError("'image' must be a non-empty base64 string.", status_code=400)

        result = runtime["predictor"].predict_base64(
            image,
            threshold=threshold,
            top_k=top_k,
        )
        return success_response(result)

    @app.errorhandler(ApiError)
    def handle_api_error(exc):
        return jsonify(error_payload(exc.message, exc.code, exc.details)), exc.status_code

    @app.errorhandler(HTTPException)
    def handle_http_error(exc):
        return (
            jsonify(error_payload(exc.description, "http_error", {"status_code": exc.code})),
            exc.code,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc):
        return (
            jsonify(error_payload("Unexpected server error.", "internal_error")),
            500,
        )

    return app


def parse_json_body():
    if not request.is_json:
        raise ApiError("Request body must be JSON.", status_code=400)

    body = request.get_json(silent=True)
    if body is None:
        raise ApiError("Malformed JSON request body.", status_code=400)
    if not isinstance(body, dict):
        raise ApiError("JSON request body must be an object.", status_code=400)

    return body


RUNTIME = _load_runtime()
app = create_app(RUNTIME)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
