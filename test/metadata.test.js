import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { loadMetadata } from "../src/metadata.js";
import { searchByContent, searchByStyleText, textToPrototype } from "../src/search.js";

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

test("style and content text searches stay separate", () => {
  const prototype = textToPrototype("轻声、快语速");
  assert.deepEqual(prototype.attributes, ["快速", "轻声"]);
  const items = [
    { id: 1, name: "one.wav", duration: 2, transcript: "今天，天气真好！", features: { prosody: { vector: prototype.vector } } },
    { id: 2, name: "two.wav", duration: 2, transcript: "今天下雨", features: { prosody: { vector: [0, 0, 0, 0, 0, 0, 1] } } }
  ];
  assert.equal(searchByStyleText(items, prototype)[0].id, 1);
  assert.equal(searchByContent(items, "天气 真好")[0].id, 1);
  assert.equal(searchByContent(items, "不存在").length, 0);
});
