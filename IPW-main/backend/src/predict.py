import os
import time

from src.errors import ApiError, LowConfidenceError, ModelUnavailableError
from src.image_preprocessing import decode_base64_image, preprocess_pil_to_chw_float
from src.model import SignBridgeCNN, torch


DEFAULT_TOP_K = 3


class GesturePredictor:
    def __init__(self, model_path, mappings):
        self.model_path = model_path
        self.mappings = mappings
        self.model = None
        self.model_error = None
        self.device_name = "cpu"
        self.torch_version = getattr(torch, "__version__", None) if torch else None

        self._load_model()

    @property
    def model_loaded(self):
        return self.model is not None

    def _load_model(self):
        if torch is None:
            self.model_error = "PyTorch is not installed."
            return

        if not self.mappings:
            self.model_error = "Mappings are not loaded."
            return

        if not os.path.exists(self.model_path):
            self.model_error = f"Model file not found: {self.model_path}"
            return

        try:
            device = torch.device("cpu")
            self.device_name = str(device)
            model = SignBridgeCNN(num_classes=len(self.mappings))
            state = torch.load(self.model_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            self.model = model
            self.model_error = None
        except Exception as exc:
            self.model_error = str(exc)
            self.model = None

    def predict_base64(self, image_base64, threshold=0.8, top_k=DEFAULT_TOP_K):
        if not self.model_loaded:
            raise ModelUnavailableError(
                "Model is not loaded. Add backend/models/isl_cnn.pth and restart the server."
            )

        threshold = self._validate_threshold(threshold)
        top_k = self._validate_top_k(top_k)

        started = time.perf_counter()
        tensor = self._preprocess_base64(image_base64)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_values, top_indices = torch.topk(
                probabilities,
                k=min(top_k, len(self.mappings)),
            )

        top_k_predictions = []
        for confidence, index in zip(top_values.tolist(), top_indices.tolist()):
            entry = self.mappings[index]
            top_k_predictions.append(
                {
                    "gesture_id": entry["gesture_id"],
                    "class_name": entry["class_name"],
                    "english": entry["english"],
                    "hindi": entry["hindi"],
                    "kannada": entry["kannada"],
                    "word_builder_value": entry["word_builder_value"],
                    "confidence": round(float(confidence), 4),
                }
            )

        winner = self.mappings[top_indices[0].item()]
        confidence = float(top_values[0].item())
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        result = self._prediction_payload(
            winner=winner,
            confidence=confidence,
            threshold=threshold,
            top_k_predictions=top_k_predictions,
            latency_ms=latency_ms,
        )

        if confidence < threshold:
            raise LowConfidenceError(
                confidence=confidence,
                threshold=threshold,
                details=result,
            )

        return result

    def _prediction_payload(
        self,
        winner,
        confidence,
        threshold,
        top_k_predictions,
        latency_ms,
    ):
        return {
            "gesture_id": winner["gesture_id"],
            "class_name": winner["class_name"],
            "english": winner["english"],
            "hindi": winner["hindi"],
            "kannada": winner["kannada"],
            "word_builder_value": winner["word_builder_value"],
            "description": winner["description"],
            "confidence": round(confidence, 4),
            "threshold": threshold,
            "top_k": top_k_predictions,
            "latency_ms": latency_ms,
        }

    def _preprocess_base64(self, image_base64):
        try:
            image = decode_base64_image(image_base64)
            array = preprocess_pil_to_chw_float(image)
        except ValueError as exc:
            raise ApiError(str(exc), status_code=400)

        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device_name)
        return tensor

    def _validate_threshold(self, threshold):
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise ApiError("'threshold' must be a number between 0 and 1.", status_code=400)

        if threshold < 0 or threshold > 1:
            raise ApiError("'threshold' must be between 0 and 1.", status_code=400)

        return threshold

    def _validate_top_k(self, top_k):
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            raise ApiError("'top_k' must be an integer.", status_code=400)

        if top_k < 1:
            raise ApiError("'top_k' must be at least 1.", status_code=400)

        return min(top_k, len(self.mappings))
