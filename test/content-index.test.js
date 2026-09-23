import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

test("transcripts persist and the character index narrows content candidates", async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "ref-content-index-"));
  process.env.DB_PATH = path.join(directory, "test.db");
  const { upsertAudio, contentCandidateIds, listAudio, closeDb } = await import("../src/db.js");
  t.after(() => { closeDb(); fs.rmSync(directory, { recursive: true, force: true }); });
  const item = {
    path: path.join(directory, "sample.wav"), name: "sample.wav", size: 10, mtimeMs: 1,
    duration: 1, sampleRate: 16000, features: { prosody: { vector: [] } },
    featureVersion: "test", transcript: "今天，天气真好！", transcriptSource: "metadata"
  };
  upsertAudio(item);
  assert.equal(listAudio()[0].transcript, item.transcript);
  assert.deepEqual(contentCandidateIds("天气真好"), [1]);
  assert.deepEqual(contentCandidateIds("下雨"), []);
  upsertAudio({ ...item, transcript: "明天下雨" });
  assert.deepEqual(contentCandidateIds("天气真好"), []);
  assert.deepEqual(contentCandidateIds("下雨"), [1]);
});
