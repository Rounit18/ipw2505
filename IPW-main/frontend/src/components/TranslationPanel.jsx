function findClassEntry(classes, prediction) {
  if (!prediction) {
    return null;
  }
  return classes.find((entry) => entry.class_name === prediction.class_name) ?? prediction;
}

export default function TranslationPanel({ prediction, showHindi, showKannada, classes }) {
  const entry = findClassEntry(classes, prediction);
  const confidence = prediction?.confidence ? `${Math.round(prediction.confidence * 100)}%` : "-";

  return (
    <section className="info-panel">
      <div className="panel-heading">
        <span className="panel-label">Translations</span>
        <span className="confidence-chip">{confidence}</span>
      </div>
      <div className="translation-row">
        <span>English</span>
        <strong>{entry?.english ?? "-"}</strong>
      </div>
      {showHindi ? (
        <div className="translation-row">
          <span>Hindi</span>
          <strong lang="hi">{entry?.hindi ?? "-"}</strong>
        </div>
      ) : null}
      {showKannada ? (
        <div className="translation-row">
          <span>Kannada</span>
          <strong lang="kn">{entry?.kannada ?? "-"}</strong>
        </div>
      ) : null}
      <div className="translation-row muted-row">
        <span>Append value</span>
        <strong>{entry?.word_builder_value ?? "-"}</strong>
      </div>
    </section>
  );
}
