"""Evaluate an F5-TTS demo index with leave-one-out retrieval metrics."""
import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def category(item):
    return Path(item["path"]).parent.name


def cosine_distance(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return 1 - float(left @ right / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def prosody_distance(left, right):
    weights = np.asarray([1.4, 1.2, .8, .5, .7, .35, .15])
    scales = np.asarray([.35, .8, .8, .08, .12, .12, 20])
    return float(np.sqrt(np.sum(weights * ((np.asarray(left) - right) / scales) ** 2) / weights.sum()))


def evaluate(name, queries, candidates_for, label_of, vector_of, feature_source, metric, ks=(1, 3, 5)):
    hits = {k: 0 for k in ks}
    reciprocal, failures = [], []
    for query in queries:
        ranked = sorted(
            ((item, metric(vector_of(query), vector_of(item))) for item in candidates_for(query) if item["id"] != query["id"]),
            key=lambda pair: pair[1],
        )
        expected = label_of(query)
        found = next((index for index, (item, _) in enumerate(ranked) if label_of(item) == expected), -1)
        for k in ks:
            hits[k] += 0 <= found < k
        reciprocal.append(1 / (found + 1) if found >= 0 else 0)
        if found != 0:
            failures.append({"query": query["name"], "expected": expected, "top": [
                {"name": item["name"], "label": label_of(item), "distance": round(score, 4)}
                for item, score in ranked[:3]
            ]})
    count = len(queries)
    return {"name": name, "featureSource": feature_source, "queries": count,
            **{f"recallAt{k}": hits[k] / count if count else 0 for k in ks},
            "mrr": sum(reciprocal) / count if count else 0, "failures": failures[:12]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.environ.get("DB_PATH", ROOT / ".data" / "f5-baseline.db"))
    parser.add_argument("--audio-root", default=ROOT / "test" / "f5-tts-demo")
    parser.add_argument("--output", default=ROOT / ".data" / "f5-eval-report.json")
    args = parser.parse_args()
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    audio_root = str(Path(args.audio_root).resolve())
    items = [{**dict(row), "features": json.loads(row["features_json"])} for row in connection.execute("SELECT * FROM audio_items")
             if row["path"].startswith(audio_root)]
    connection.close()
    has_style = bool(items) and all(item["features"].get("style", {}).get("embedding") for item in items)
    has_emotion = bool(items) and all(item["features"].get("emotion", {}).get("embedding") for item in items)
    prosody = lambda item: item["features"]["prosody"]["vector"]
    style = (lambda item: item["features"]["style"]["embedding"]) if has_style else prosody
    emotion_vector = (lambda item: item["features"]["emotion"]["embedding"]) if has_emotion else prosody
    style_metric = cosine_distance if has_style else prosody_distance
    emotion_metric = cosine_distance if has_emotion else prosody_distance
    emotion = [item for item in items if category(item) == "emotion"]
    robustness = [item for item in items if category(item) == "robustness"]
    speed = [item for item in items if category(item) == "speed_control" and re_search_speed(item["name"])]
    zero_shot = [item for item in items if category(item) == "zero_shot"]
    speed_family = lambda item: re_strip_speed(item["name"])
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(), "database": str(args.db),
        "featureVersion": sorted({item["feature_version"] for item in items}), "audioCount": len(items),
        "evaluations": [
            evaluate("emotion-label", emotion, lambda _: emotion, lambda item: item["name"].split("_")[0],
                     emotion_vector, "indextts2-emovec" if has_emotion else "prosody-v1", emotion_metric),
            evaluate("robustness-ref-gen-pair", robustness, lambda _: robustness,
                     lambda item: item["name"].replace("_ref.wav", "").replace("_gen.wav", ""),
                     style, "indextts2-condition" if has_style else "prosody-v1", style_metric),
            evaluate("speed-factor-cross-content", speed, lambda query: [item for item in speed if speed_family(item) != speed_family(query)],
                     lambda item: re_search_speed(item["name"]), style,
                     "indextts2-condition" if has_style else "prosody-v1", style_metric),
            evaluate("zero-shot-reference-group", zero_shot, lambda _: zero_shot,
                     lambda item: item["name"].split("_gen_")[0].removesuffix(".wav"), style,
                     "indextts2-condition" if has_style else "prosody-v1", style_metric),
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in report["evaluations"]:
        print(f"{result['name']} [{result['featureSource']}]: n={result['queries']} "
              f"R@1={result['recallAt1']:.1%} R@3={result['recallAt3']:.1%} MRR={result['mrr']:.3f}")
    print(f"Report: {output}")


def re_search_speed(name):
    import re
    match = re.search(r"_(0\.7|1\.0|1\.3)x\.wav$", name, re.IGNORECASE)
    return match.group(1) if match else None


def re_strip_speed(name):
    import re
    return re.sub(r"_(0\.7|1\.0|1\.3)x\.wav$", "", name, flags=re.IGNORECASE)


if __name__ == "__main__":
    main()
