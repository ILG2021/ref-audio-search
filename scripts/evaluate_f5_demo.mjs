import path from "node:path";
import fs from "node:fs";

const root = path.resolve("test/f5-tts-demo");
process.env.DB_PATH ||= path.resolve(".data/f5-baseline.db");

const { closeDb, listAudio } = await import("../src/db.js");
const { distance } = await import("../src/search.js");

const items = listAudio().filter(item => item.path.startsWith(root));

function category(item) {
  return path.basename(path.dirname(item.path));
}

function cosineDistance(a, b) {
  let dot = 0, aa = 0, bb = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) { dot += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return 1 - dot / Math.max(Math.sqrt(aa * bb), 1e-12);
}

function rank(query, candidates, vectorOf, metric) {
  const queryVector = vectorOf(query);
  return candidates
    .filter(item => item.id !== query.id)
    .map(item => ({ item, score: metric(queryVector, vectorOf(item)) }))
    .sort((a, b) => a.score - b.score);
}

function evaluate(name, queries, candidatesFor, labelOf, vectorOf, featureSource, metric, ks = [1, 3, 5]) {
  const hits = Object.fromEntries(ks.map(k => [k, 0]));
  const reciprocalRanks = [];
  const failures = [];
  for (const query of queries) {
    const expected = labelOf(query);
    const ranked = rank(query, candidatesFor(query), vectorOf, metric);
    const foundAt = ranked.findIndex(result => labelOf(result.item) === expected);
    for (const k of ks) if (foundAt >= 0 && foundAt < k) hits[k]++;
    reciprocalRanks.push(foundAt >= 0 ? 1 / (foundAt + 1) : 0);
    if (foundAt !== 0) failures.push({
      query: query.name,
      expected,
      top: ranked.slice(0, 3).map(result => ({ name: result.item.name, label: labelOf(result.item), distance: Number(result.score.toFixed(4)) }))
    });
  }
  return {
    name,
    featureSource,
    queries: queries.length,
    ...Object.fromEntries(ks.map(k => [`recallAt${k}`, queries.length ? hits[k] / queries.length : 0])),
    mrr: queries.length ? reciprocalRanks.reduce((a, b) => a + b, 0) / queries.length : 0,
    failures: failures.slice(0, 12)
  };
}

const prosody = item => item.features.prosody.vector;
const style = item => item.features.style?.embedding;
const emotionVector = item => item.features.emotion?.embedding;
const emotion = items.filter(item => category(item) === "emotion");
const robustness = items.filter(item => category(item) === "robustness");
const speed = items.filter(item => category(item) === "speed_control" && /_(0\.7|1\.0|1\.3)x\.wav$/i.test(item.name));
const zeroShot = items.filter(item => category(item) === "zero_shot");

const speedFamily = item => item.name.replace(/_(0\.7|1\.0|1\.3)x\.wav$/i, "");
const hasStyle = items.length > 0 && items.every(item => style(item)?.length);
const hasEmotion = items.length > 0 && items.every(item => emotionVector(item)?.length);
const styleVector = hasStyle ? style : prosody;
const selectedEmotionVector = hasEmotion ? emotionVector : prosody;
const styleMetric = hasStyle ? cosineDistance : distance;
const emotionMetric = hasEmotion ? cosineDistance : distance;
const report = {
  generatedAt: new Date().toISOString(),
  database: process.env.DB_PATH,
  featureVersion: [...new Set(items.map(item => item.feature_version))],
  audioCount: items.length,
  evaluations: [
    evaluate("emotion-label", emotion, () => emotion, item => item.name.split("_")[0], selectedEmotionVector, hasEmotion ? "indextts2-emovec" : "prosody-v1", emotionMetric),
    evaluate("robustness-ref-gen-pair", robustness, () => robustness, item => item.name.replace(/_(ref|gen)\.wav$/i, ""), styleVector, hasStyle ? "indextts2-condition" : "prosody-v1", styleMetric),
    evaluate("speed-factor-cross-content", speed, query => speed.filter(item => speedFamily(item) !== speedFamily(query)), item => item.name.match(/_(0\.7|1\.0|1\.3)x\.wav$/i)?.[1], styleVector, hasStyle ? "indextts2-condition" : "prosody-v1", styleMetric),
    evaluate("zero-shot-reference-group", zeroShot, () => zeroShot, item => item.name.replace(/_gen_(zh|en|code_switch)\.wav$/i, "").replace(/\.wav$/i, ""), styleVector, hasStyle ? "indextts2-condition" : "prosody-v1", styleMetric)
  ]
};

const output = path.resolve(process.argv[2] || ".data/f5-eval-report.json");
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
for (const result of report.evaluations) {
  console.log(`${result.name} [${result.featureSource}]: n=${result.queries} R@1=${(result.recallAt1 * 100).toFixed(1)}% R@3=${(result.recallAt3 * 100).toFixed(1)}% MRR=${result.mrr.toFixed(3)}`);
}
console.log(`Report: ${output}`);
closeDb();
