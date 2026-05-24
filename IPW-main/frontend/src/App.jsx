import { useCallback, useEffect, useRef, useState } from "react";

import GestureDisplay from "./components/GestureDisplay.jsx";
import HistoryLog from "./components/HistoryLog.jsx";
import SettingsBar from "./components/SettingsBar.jsx";
import TranslationPanel from "./components/TranslationPanel.jsx";
import VideoFeed from "./components/VideoFeed.jsx";
import WordBuilder from "./components/WordBuilder.jsx";
import { useHealth } from "./hooks/useHealth.js";
import { usePrediction } from "./hooks/usePrediction.js";
import { useWordBuilder } from "./hooks/useWordBuilder.js";
import { getClasses } from "./services/api.js";

export default function App() {
  const videoRef = useRef(null);
  const [threshold, setThreshold] = useState(0.8);
  const [intervalMs, setIntervalMs] = useState(500);
  const [showHindi, setShowHindi] = useState(true);
  const [showKannada, setShowKannada] = useState(true);
  const [classes, setClasses] = useState([]);
  const [cameraState, setCameraState] = useState("disabled");

  const health = useHealth();
  const apiStatus = health.error ? "unreachable" : health.data?.status ?? "checking";
  const captureFrame = useCallback(() => videoRef.current?.captureFrame(), []);
  const prediction = usePrediction({
    enabled: health.data?.status === "ok",
    captureFrame,
    threshold,
    intervalMs,
  });
  const wordBuilder = useWordBuilder(prediction.stablePrediction, prediction.status);

  useEffect(() => {
    getClasses()
      .then((data) => setClasses(data.classes))
      .catch(() => setClasses([]));
  }, []);

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Local ISL prototype</p>
          <h1>SignBridge</h1>
        </div>
        <SettingsBar
          threshold={threshold}
          onThresholdChange={setThreshold}
          intervalMs={intervalMs}
          onIntervalChange={setIntervalMs}
          showHindi={showHindi}
          onShowHindiChange={setShowHindi}
          showKannada={showKannada}
          onShowKannadaChange={setShowKannada}
        />
      </header>

      <section className="status-strip">
        <span>API: {apiStatus}</span>
        <span>Camera: {cameraState}</span>
        <span>Prediction: {prediction.status}</span>
        <span>Builder: {wordBuilder.mode}</span>
        {health.error ? <span>{health.error}</span> : null}
      </section>

      <section className="workspace-grid">
        <VideoFeed
          ref={videoRef}
          disabled={health.data?.status !== "ok"}
          onCameraStateChange={setCameraState}
        />
        <div className="panel-stack">
          <GestureDisplay
            health={health}
            predictionState={prediction}
            cameraState={cameraState}
          />
          <TranslationPanel
            prediction={prediction.stablePrediction}
            showHindi={showHindi}
            showKannada={showKannada}
            classes={classes}
          />
          <WordBuilder
            value={wordBuilder.currentToken}
            heldGesture={wordBuilder.heldGesture}
            builderMode={wordBuilder.mode}
            onBackspace={wordBuilder.backspace}
            onConfirm={wordBuilder.confirm}
            onClear={wordBuilder.clearCurrent}
          />
          <HistoryLog
            history={wordBuilder.history}
            transcript={wordBuilder.transcript}
            onClear={wordBuilder.clearHistory}
          />
        </div>
      </section>
    </main>
  );
}
