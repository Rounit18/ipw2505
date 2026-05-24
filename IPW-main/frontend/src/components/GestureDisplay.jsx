export default function GestureDisplay({ health, predictionState, cameraState }) {
  const prediction = predictionState.stablePrediction;
  const latest = predictionState.latest;

  let label = "Waiting";
  let detail = "Hold one hand inside the guide box.";
  let tone = "neutral";

  if (health.loading) {
    detail = "Checking backend health.";
  } else if (health.error || health.data?.status !== "ok") {
    label = "Offline";
    detail = health.data?.model_error ?? health.error ?? "Backend is not ready.";
    tone = "danger";
  } else if (cameraState === "denied") {
    label = "Camera";
    detail = "Camera permission denied.";
    tone = "danger";
  } else if (cameraState === "unavailable") {
    label = "Camera";
    detail = "No usable camera was found.";
    tone = "danger";
  } else if (predictionState.error) {
    if (predictionState.status === "low_confidence") {
      label = "Unclear";
      detail = latest?.confidence
        ? `${Math.round(latest.confidence * 100)}% confidence`
        : predictionState.error;
      tone = "warning";
    } else {
      label = "Error";
      detail = predictionState.error;
      tone = "danger";
    }
  } else if (prediction) {
    label = prediction.class_name;
    detail = `${Math.round(prediction.confidence * 100)}% confidence`;
    tone = "success";
  } else if (predictionState.status === "no_hand" || latest?.state === "no_hand") {
    label = "No hand";
    detail = latest?.message ?? "No hand detected inside the guide box.";
    tone = "warning";
  } else if (predictionState.isPolling) {
    detail = "Reading gesture.";
  }

  return (
    <section className={`info-panel gesture-display tone-${tone}`}>
      <span className="panel-label">Prediction</span>
      <div className="gesture-badge">{label}</div>
      <p>{detail}</p>
    </section>
  );
}
