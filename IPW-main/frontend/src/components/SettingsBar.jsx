export default function SettingsBar({
  threshold,
  onThresholdChange,
  intervalMs,
  onIntervalChange,
  showHindi,
  onShowHindiChange,
  showKannada,
  onShowKannadaChange,
}) {
  return (
    <section className="settings-bar" aria-label="Settings">
      <label>
        Threshold
        <input
          type="range"
          min="0.5"
          max="0.95"
          step="0.05"
          value={threshold}
          onChange={(event) => onThresholdChange(Number(event.target.value))}
        />
        <span className="setting-value">{Math.round(threshold * 100)}%</span>
      </label>
      <label>
        Interval
        <select
          value={intervalMs}
          onChange={(event) => onIntervalChange(Number(event.target.value))}
        >
          <option value="300">300 ms</option>
          <option value="500">500 ms</option>
          <option value="1000">1000 ms</option>
        </select>
      </label>
      <label className="inline-toggle">
        <input
          type="checkbox"
          checked={showHindi}
          onChange={(event) => onShowHindiChange(event.target.checked)}
        />
        Hindi
      </label>
      <label className="inline-toggle">
        <input
          type="checkbox"
          checked={showKannada}
          onChange={(event) => onShowKannadaChange(event.target.checked)}
        />
        Kannada
      </label>
    </section>
  );
}
