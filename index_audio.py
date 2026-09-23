"""Build and incrementally update the reference-audio SQLite index."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import numpy as np

from model_worker import Models


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("DB_PATH", ROOT / ".data" / "audio-search.db")).resolve()
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif"}
FEATURE_VERSION = "indextts2-native-v1"


def normalize_content(text):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(text or "").lower())


def connect(path=DB_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS audio_items (
          id INTEGER PRIMARY KEY,
          path TEXT NOT NULL UNIQUE,
          name TEXT NOT NULL,
          size INTEGER NOT NULL,
          mtime_ms REAL NOT NULL,
          duration REAL NOT NULL,
          sample_rate INTEGER NOT NULL,
          features_json TEXT NOT NULL,
          feature_version TEXT NOT NULL,
          transcript TEXT NOT NULL DEFAULT '',
          transcript_source TEXT NOT NULL DEFAULT '',
          indexed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_audio_name ON audio_items(name);
        CREATE TABLE IF NOT EXISTS content_grams (
          audio_id INTEGER NOT NULL,
          gram TEXT NOT NULL,
          PRIMARY KEY(audio_id, gram)
        );
        CREATE INDEX IF NOT EXISTS idx_content_gram ON content_grams(gram);
    """)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(audio_items)")}
    if "transcript" not in columns:
        connection.execute("ALTER TABLE audio_items ADD COLUMN transcript TEXT NOT NULL DEFAULT ''")
    if "transcript_source" not in columns:
        connection.execute("ALTER TABLE audio_items ADD COLUMN transcript_source TEXT NOT NULL DEFAULT ''")
    return connection


def find_audio_files(root):
    return sorted(path.resolve() for path in Path(root).rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


def load_metadata(root):
    metadata_path = Path(root) / "metadata.csv"
    if not metadata_path.is_file():
        return None
    text = metadata_path.read_text(encoding="utf-8-sig")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    delimiter = "|" if "|" in first_line else ","
    mapping = {}
    for row in csv.reader(text.splitlines(), delimiter=delimiter):
        if len(row) < 2 or not row[0].strip() or not row[1].strip():
            continue
        name = row[0].strip().replace("\\", "/")
        transcript = row[1].strip()
        path = Path(name)
        mapping[path.name.lower()] = transcript
        if not path.suffix:
            mapping[path.stem.lower()] = transcript
    return mapping


def metadata_text(metadata, audio_path):
    if metadata is None:
        return None
    path = Path(audio_path)
    return metadata.get(path.name.lower(), metadata.get(path.stem.lower()))


def extract_prosody(samples, sample_rate=16000):
    samples = np.asarray(samples, dtype=np.float32)
    frame = round(sample_rate * .025)
    hop = round(sample_rate * .01)
    rms, zcr = [], []
    for start in range(0, max(0, len(samples) - frame + 1), hop):
        values = samples[start:start + frame]
        rms.append(float(np.sqrt(np.mean(values * values))))
        zcr.append(float(np.mean((values[1:] >= 0) != (values[:-1] >= 0))))
    rms_array = np.asarray(rms, dtype=np.float32)
    zcr_array = np.asarray(zcr, dtype=np.float32)
    noise_floor = float(np.quantile(rms_array, .2)) if len(rms_array) else 0
    peak = float(rms_array.max()) if len(rms_array) else 0
    threshold = max(noise_floor * 2.5, peak * .08, 1e-4)
    voiced = rms_array >= threshold
    pauses, pause_run, pause_lengths = 0, 0, []
    for active in voiced:
        if active:
            if pause_run >= 20:
                pauses += 1
                pause_lengths.append(pause_run * .01)
            pause_run = 0
        else:
            pause_run += 1
    if pause_run >= 20:
        pauses += 1
        pause_lengths.append(pause_run * .01)
    active_rms = rms_array[voiced]
    active_zcr = zcr_array[voiced]
    duration = len(samples) / sample_rate
    vector = [
        float(voiced.mean()) if len(voiced) else 0,
        pauses / max(duration, .1),
        float(np.mean(pause_lengths)) if pause_lengths else 0,
        float(active_rms.mean()) if len(active_rms) else 0,
        float(np.quantile(active_rms, .9) - np.quantile(active_rms, .1)) if len(active_rms) else 0,
        float(active_zcr.mean()) if len(active_zcr) else 0,
        duration,
    ]
    return {
        "duration": duration, "voicedRatio": vector[0], "pausesPerSecond": vector[1],
        "meanPauseDuration": vector[2], "rmsMean": vector[3], "dynamicRange": vector[4],
        "zeroCrossingRate": vector[5], "vector": vector,
    }


def analyze_audio(path):
    process = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
        check=True, capture_output=True,
    )
    return extract_prosody(np.frombuffer(process.stdout, dtype="<f4").copy())


def upsert_audio(connection, item):
    with connection:
        connection.execute("""
            INSERT INTO audio_items(path,name,size,mtime_ms,duration,sample_rate,features_json,feature_version,
                                    transcript,transcript_source,indexed_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,datetime('now'))
            ON CONFLICT(path) DO UPDATE SET
              name=excluded.name,size=excluded.size,mtime_ms=excluded.mtime_ms,duration=excluded.duration,
              sample_rate=excluded.sample_rate,features_json=excluded.features_json,
              feature_version=excluded.feature_version,transcript=excluded.transcript,
              transcript_source=excluded.transcript_source,indexed_at=excluded.indexed_at
        """, (item["path"], item["name"], item["size"], item["mtime_ms"], item["duration"], 16000,
              json.dumps(item["features"], ensure_ascii=False), FEATURE_VERSION, item["transcript"], item["transcript_source"]))
        audio_id = connection.execute("SELECT id FROM audio_items WHERE path=?", (item["path"],)).fetchone()[0]
        connection.execute("DELETE FROM content_grams WHERE audio_id=?", (audio_id,))
        normalized = normalize_content(item["transcript"])
        grams = {normalized[index:index + 2] for index in range(max(0, len(normalized) - 1))}
        connection.executemany("INSERT INTO content_grams(audio_id,gram) VALUES(?,?)",
                               ((audio_id, gram) for gram in grams))


def build_index(root, database_path=DB_PATH):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"音频目录不存在：{root}")
    files = find_audio_files(root)
    metadata = load_metadata(root)
    models = None
    indexed = skipped = 0
    errors = []
    with closing(connect(database_path)) as connection:
        for position, path in enumerate(files, 1):
            try:
                stat = path.stat()
                existing = connection.execute("SELECT * FROM audio_items WHERE path=?", (str(path),)).fetchone()
                existing_features = json.loads(existing["features_json"]) if existing else None
                transcript = metadata_text(metadata, path)
                mtime_ms = stat.st_mtime_ns / 1_000_000
                audio_unchanged = bool(
                    existing and existing["size"] == stat.st_size and existing["mtime_ms"] == mtime_ms
                    and existing["feature_version"] == FEATURE_VERSION
                )
                transcript_unchanged = bool(
                    existing and ((transcript is not None and existing["transcript_source"] == "metadata"
                                   and existing["transcript"] == transcript)
                                  or (transcript is None and existing["transcript_source"] == "faster-whisper"))
                )
                if audio_unchanged and transcript_unchanged:
                    skipped += 1
                    print(f"\r{position}/{len(files)} 跳过 {path.name:60.60}", end="", flush=True)
                    continue
                if models is None:
                    models = Models()
                prosody = existing_features["prosody"] if audio_unchanged else analyze_audio(path)
                extracted = ({"style": existing_features["style"]["embedding"],
                              "emotion": existing_features["emotion"]["embedding"]}
                             if audio_unchanged else models.extract(str(path)))
                if transcript is None:
                    transcript = models.transcribe(str(path))
                    transcript_source = "faster-whisper"
                else:
                    transcript_source = "metadata"
                upsert_audio(connection, {
                    "path": str(path), "name": path.name, "size": stat.st_size, "mtime_ms": mtime_ms,
                    "duration": prosody["duration"], "features": {
                        "prosody": prosody, "style": {"embedding": extracted["style"]},
                        "emotion": {"embedding": extracted["emotion"]},
                    }, "transcript": transcript, "transcript_source": transcript_source,
                })
                indexed += 1
                print(f"\r{position}/{len(files)} 已索引 {path.name:58.58}", end="", flush=True)
            except Exception as error:  # keep indexing independent files
                errors.append({"file": str(path), "error": str(error)})
                print(f"\r{position}/{len(files)} 失败 {path.name:60.60}", end="", flush=True)
    print()
    return {"total": len(files), "indexed": indexed, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="建立或增量更新参考音频索引")
    parser.add_argument("audio_directory", help="要递归扫描的音频目录")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite 索引路径")
    args = parser.parse_args()
    result = build_index(args.audio_directory, args.db)
    print(f"完成：新增或更新 {result['indexed']}，跳过未变化 {result['skipped']}，总计 {result['total']}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"失败：{error['file']}：{error['error']}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
