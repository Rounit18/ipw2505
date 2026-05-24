class ApiError(Exception):
    def __init__(self, message, status_code=400, code="bad_request", details=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}


class ModelUnavailableError(ApiError):
    def __init__(self, message="Model is not loaded."):
        super().__init__(message, status_code=503, code="model_unavailable")


class LowConfidenceError(ApiError):
    def __init__(self, confidence, threshold, details=None):
        message = (
            f"Confidence {confidence:.2f} is below threshold {threshold:.2f}. "
            "Gesture unclear."
        )
        super().__init__(
            message,
            status_code=422,
            code="low_confidence",
            details=details,
        )


def error_payload(message, code, details=None):
    payload = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if details:
        payload["error"]["details"] = details
    return payload
