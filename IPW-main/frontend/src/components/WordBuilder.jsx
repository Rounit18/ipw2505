export default function WordBuilder({
  value,
  heldGesture,
  builderMode,
  onBackspace,
  onConfirm,
  onClear,
}) {
  return (
    <section className="info-panel">
      <div className="panel-heading">
        <span className="panel-label">Current token</span>
        <span className="confidence-chip">{heldGesture ?? builderMode}</span>
      </div>
      <div className="current-token">{value || "-"}</div>
      <div className="button-row">
        <button type="button" onClick={onBackspace} disabled={!value}>Backspace</button>
        <button type="button" onClick={onConfirm} disabled={!value.trim()}>Confirm</button>
        <button type="button" onClick={onClear} disabled={!value}>Clear</button>
      </div>
    </section>
  );
}
