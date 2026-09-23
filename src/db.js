import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { DB_PATH } from "./config.js";
import { normalizeContent } from "./search.js";

let database;

export function db() {
  if (database) return database;
  fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
  database = new DatabaseSync(DB_PATH);
  database.exec(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS audio_items (
      id INTEGER PRIMARY KEY,
      path TEXT NOT NULL UNIQUE,
      name TEXT NOT NULL,
      size INTEGER NOT NULL,
      mtime_ms INTEGER NOT NULL,
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
  `);
  const columns = new Set(database.prepare("PRAGMA table_info(audio_items)").all().map(column => column.name));
  if (!columns.has("transcript")) database.exec("ALTER TABLE audio_items ADD COLUMN transcript TEXT NOT NULL DEFAULT ''");
  if (!columns.has("transcript_source")) database.exec("ALTER TABLE audio_items ADD COLUMN transcript_source TEXT NOT NULL DEFAULT ''");
  return database;
}

export function closeDb() {
  if (!database) return;
  database.close();
  database = undefined;
}

export function upsertAudio(item) {
  const connection = db();
  connection.exec("BEGIN");
  try {
  connection.prepare(`
    INSERT INTO audio_items(path,name,size,mtime_ms,duration,sample_rate,features_json,feature_version,transcript,transcript_source,indexed_at)
    VALUES(?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(path) DO UPDATE SET
      name=excluded.name,size=excluded.size,mtime_ms=excluded.mtime_ms,
      duration=excluded.duration,sample_rate=excluded.sample_rate,
      features_json=excluded.features_json,feature_version=excluded.feature_version,
      transcript=excluded.transcript,transcript_source=excluded.transcript_source,indexed_at=excluded.indexed_at
  `).run(item.path, item.name, item.size, item.mtimeMs, item.duration, item.sampleRate,
    JSON.stringify(item.features), item.featureVersion, item.transcript || "", item.transcriptSource || "", new Date().toISOString());
  const id = connection.prepare("SELECT id FROM audio_items WHERE path=?").get(item.path).id;
  connection.prepare("DELETE FROM content_grams WHERE audio_id=?").run(id);
  const normalized = Array.from(normalizeContent(item.transcript));
  const insert = connection.prepare("INSERT INTO content_grams(audio_id,gram) VALUES(?,?)");
  for (const gram of new Set(normalized.slice(0, -1).map((char, index) => char + normalized[index + 1]))) insert.run(id, gram);
  connection.exec("COMMIT");
  } catch (error) { connection.exec("ROLLBACK"); throw error; }
}

export function contentCandidateIds(query) {
  const chars = Array.from(normalizeContent(query));
  if (chars.length < 2) return db().prepare("SELECT id FROM audio_items WHERE transcript != ''").all().map(row => row.id);
  const grams = [...new Set(chars.slice(0, -1).map((char, index) => char + chars[index + 1]))];
  const placeholders = grams.map(() => "?").join(",");
  return db().prepare(`SELECT audio_id AS id FROM content_grams WHERE gram IN (${placeholders}) GROUP BY audio_id HAVING COUNT(*) = ?`)
    .all(...grams, grams.length).map(row => row.id);
}

export function listAudio() {
  return db().prepare("SELECT * FROM audio_items ORDER BY id").all().map(row => ({
    ...row,
    features: JSON.parse(row.features_json)
  }));
}

export function getAudioByPath(filePath) {
  const row = db().prepare("SELECT * FROM audio_items WHERE path = ?").get(filePath);
  return row ? { ...row, features: JSON.parse(row.features_json) } : null;
}

export function removeMissing(existingPaths, rootPath) {
  const known = db().prepare("SELECT id,path FROM audio_items").all();
  const remove = db().prepare("DELETE FROM audio_items WHERE id=?");
  let count = 0;
  db().exec("BEGIN");
  try {
    for (const row of known) {
      const inScope = row.path.toLowerCase().startsWith(rootPath.toLowerCase());
      if (inScope && !existingPaths.has(row.path)) { db().prepare("DELETE FROM content_grams WHERE audio_id=?").run(row.id); remove.run(row.id); count++; }
    }
    db().exec("COMMIT");
  } catch (error) {
    db().exec("ROLLBACK");
    throw error;
  }
  return count;
}
