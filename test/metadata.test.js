import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { loadMetadata } from "../src/metadata.js";

test("metadata.csv imports quoted text and names with or without extension", t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "ref-metadata-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  assert.equal(loadMetadata(directory), null);
  fs.writeFileSync(path.join(directory, "metadata.csv"), 'b.wav,"你好，世界"\nc,今天天气好\n');
  const metadata = loadMetadata(directory);
  assert.equal(metadata.get(path.join(directory, "wavs", "b.wav")), "你好，世界");
  assert.equal(metadata.get(path.join(directory, "wavs", "c.wav")), "今天天气好");
  fs.writeFileSync(path.join(directory, "metadata.csv"), "d|这是台词|normalized text\n");
  assert.equal(loadMetadata(directory).get(path.join(directory, "wavs", "d.wav")), "这是台词");
});
