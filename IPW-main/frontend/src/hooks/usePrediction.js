import { useEffect, useRef, useState } from "react";

import { getApiError, predictFrame } from "../services/api.js";

const VALID_ACCEPTED_STATE = "accepted";
const BUFFER_SIZE = 3;

function normalizeAcceptedPrediction(result) {
  return {
    ...result,
    state: VALID_ACCEPTED_STATE,
  };
}

function majorityVote(items) {
  const counts = new Map();
  for (const item of items) {
    if (!item || item.state !== VALID_ACCEPTED_STATE) {
      continue;
    }
    counts.set(item.class_name, (counts.get(item.class_name) ?? 0) + 1);
  }

  let winner = null;
  let winnerCount = 0;
  for (const [className, count] of counts.entries()) {
    if (count > winnerCount) {
      winner = className;
      winnerCount = count;
    }
  }

  if (!winner || winnerCount < 2) {
    return null;
  }

  return [...items].reverse().find((item) => item?.class_name === winner) ?? null;
}

export function usePrediction({ enabled, captureFrame, threshold, intervalMs }) {
  const [latest, setLatest] = useState(null);
  const [stablePrediction, setStablePrediction] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState("idle");
  const [isPolling, setIsPolling] = useState(false);
  const bufferRef = useRef([]);
  const inFlightRef = useRef(false);
  const emptyFrameCountRef = useRef(0);

  function resetPredictionState(nextStatus = "idle") {
    setLatest(null);
    setStablePrediction(null);
    setError(null);
    setStatus(nextStatus);
    bufferRef.current = [];
    emptyFrameCountRef.current = 0;
  }

  function pushAccepted(result) {
    const accepted = normalizeAcceptedPrediction(result);
    setLatest(accepted);
    bufferRef.current = [...bufferRef.current, accepted].slice(-BUFFER_SIZE);
    setStablePrediction(majorityVote(bufferRef.current));
    setStatus("predicting");
    setError(null);
  }

  function setTransientState(nextLatest, nextStatus, nextError = null) {
    setLatest(nextLatest);
    setStablePrediction(null);
    setStatus(nextStatus);
    setError(nextError);
    bufferRef.current = [];
  }

  useEffect(() => {
    if (!enabled) {
      resetPredictionState("disabled");
      return undefined;
    }

    async function tick() {
      if (inFlightRef.current || document.hidden) {
        return;
      }

      const image = captureFrame?.();
      if (!image) {
        emptyFrameCountRef.current += 1;
        if (emptyFrameCountRef.current >= 2) {
          setTransientState(
            { state: "no_hand", message: "No camera frame available." },
            "no_hand",
          );
        }
        return;
      }

      if (image.state === "no_hand") {
        emptyFrameCountRef.current += 1;
        setTransientState(image, "no_hand");
        return;
      }

      emptyFrameCountRef.current = 0;
      inFlightRef.current = true;
      setIsPolling(true);
      try {
        const result = await predictFrame(image.image, threshold, 3);
        pushAccepted(result);
      } catch (exc) {
        const apiError = getApiError(exc);
        if (apiError.code === "low_confidence") {
          setTransientState(
            {
              ...(apiError.details ?? {}),
              state: "low_confidence",
            },
            "low_confidence",
            apiError.message,
          );
        } else if (apiError.code === "model_unavailable") {
          setTransientState(
            { state: "api_down", message: apiError.message },
            "api_down",
            apiError.message,
          );
        } else {
          setTransientState(
            { state: "error", message: apiError.message },
            "error",
            apiError.message,
          );
        }
      } finally {
        inFlightRef.current = false;
        setIsPolling(false);
      }
    }

    tick();
    const id = window.setInterval(tick, intervalMs);
    return () => window.clearInterval(id);
  }, [enabled, captureFrame, threshold, intervalMs]);

  return {
    latest,
    stablePrediction,
    error,
    status,
    isPolling,
  };
}
