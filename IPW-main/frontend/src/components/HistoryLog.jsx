import { useEffect, useRef, useState } from "react";

export default function HistoryLog({ history, transcript, onClear }) {
  const listRef = useRef(null);
  const [copyState, setCopyState] = useState("idle");

  useEffect(() => {
    const list = listRef.current;
    if (list) {
      list.scrollTop = list.scrollHeight;
    }
  }, [history]);

  async function copyTranscript() {
    try {
      await navigator.clipboard.writeText(transcript);
      setCopyState("copied");
      window.setTimeout(() => setCopyState("idle"), 1200);
    } catch (error) {
      setCopyState("failed");
      window.setTimeout(() => setCopyState("idle"), 1600);
    }
  }

  return (
    <section className="info-panel history-panel">
      <div className="panel-heading">
        <span className="panel-label">Session history</span>
        <div className="button-row compact">
          <button type="button" onClick={copyTranscript} disabled={!history.length}>
            {copyState === "copied" ? "Copied" : "Copy"}
          </button>
          <button type="button" onClick={onClear} disabled={!history.length}>Clear</button>
        </div>
      </div>
      <div className="history-list" ref={listRef}>
        {history.length ? (
          history.map((item) => (
            <div className="history-item" key={item.id}>
              <span>{item.timestamp}</span>
              <strong>{item.value}</strong>
            </div>
          ))
        ) : (
          <p>No confirmed tokens yet.</p>
        )}
      </div>
      {copyState === "failed" ? <p className="copy-error">Clipboard unavailable.</p> : null}
    </section>
  );
}
