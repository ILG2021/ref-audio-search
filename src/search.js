const DEFAULT_WEIGHTS = [1.4, 1.2, 0.8, 0.5, 0.7, 0.35, 0.15];

// Shared by the offline evaluator. Runtime search lives in gradio_app.py.
export function distance(a, b, weights = DEFAULT_WEIGHTS) {
  const scales = [0.35, 0.8, 0.8, 0.08, 0.12, 0.12, 20];
  let sum = 0;
  let weightSum = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const weight = weights[i] ?? 1;
    const delta = (a[i] - b[i]) / scales[i];
    sum += weight * delta * delta;
    weightSum += weight;
  }
  return Math.sqrt(sum / Math.max(weightSum, 1e-9));
}

// Shared by the Node indexer when it builds transcript bigrams.
export function normalizeContent(text) {
  return String(text || "").normalize("NFKC").toLowerCase().replace(/[\p{P}\p{S}\s]+/gu, "");
}
