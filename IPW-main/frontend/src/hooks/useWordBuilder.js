import { useEffect, useMemo, useReducer } from "react";

const RESET_STATUSES = new Set([
  "no_hand",
  "low_confidence",
  "api_down",
  "disabled",
  "error",
]);

const initialState = {
  currentToken: "",
  history: [],
  mode: "ready",
  heldGesture: null,
  lastAcceptedAt: null,
};

function timestamp() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function appendGesture(state, prediction) {
  const nextGesture = prediction.class_name;
  const shouldAppend =
    state.mode !== "holding" ||
    state.heldGesture === null ||
    state.heldGesture !== nextGesture;

  if (!shouldAppend) {
    return state;
  }

  return {
    ...state,
    currentToken: state.currentToken + prediction.word_builder_value,
    mode: "holding",
    heldGesture: nextGesture,
    lastAcceptedAt: Date.now(),
  };
}

function reducer(state, action) {
  switch (action.type) {
    case "prediction":
      return appendGesture(state, action.prediction);

    case "unlock":
      if (state.mode === "ready" && state.heldGesture === null) {
        return state;
      }
      return {
        ...state,
        mode: "ready",
        heldGesture: null,
      };

    case "backspace":
      return {
        ...state,
        currentToken: state.currentToken.slice(0, -1),
      };

    case "clear_current":
      return {
        ...state,
        currentToken: "",
        mode: "ready",
        heldGesture: null,
      };

    case "confirm": {
      const value = state.currentToken.trim();
      if (!value) {
        return state;
      }

      return {
        ...state,
        currentToken: "",
        history: [
          ...state.history,
          {
            id: `${Date.now()}-${state.history.length}`,
            value,
            timestamp: timestamp(),
          },
        ],
        mode: "ready",
        heldGesture: null,
      };
    }

    case "clear_history":
      return {
        ...state,
        history: [],
      };

    default:
      return state;
  }
}

export function useWordBuilder(stablePrediction, predictionStatus) {
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    if (stablePrediction?.state === "accepted") {
      dispatch({ type: "prediction", prediction: stablePrediction });
      return;
    }

    if (RESET_STATUSES.has(predictionStatus)) {
      dispatch({ type: "unlock" });
    }
  }, [stablePrediction, predictionStatus]);

  const transcript = useMemo(
    () => state.history.map((item) => `${item.timestamp} ${item.value}`).join("\n"),
    [state.history],
  );

  return {
    currentToken: state.currentToken,
    history: state.history,
    transcript,
    mode: state.mode,
    heldGesture: state.heldGesture,
    hasCurrentToken: state.currentToken.trim().length > 0,
    backspace: () => dispatch({ type: "backspace" }),
    clearCurrent: () => dispatch({ type: "clear_current" }),
    confirm: () => dispatch({ type: "confirm" }),
    clearHistory: () => dispatch({ type: "clear_history" }),
  };
}

