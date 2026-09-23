import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

test("audio ranking mode is inside the audio form rather than search settings", () => {
  const html = fs.readFileSync(new URL("../public/index.html", import.meta.url), "utf8");
  const audioForm = html.match(/<form id="audio-form">([\s\S]*?)<\/form>/)?.[1] || "";
  const filters = html.match(/<details class="filters">([\s\S]*?)<\/details>/)?.[1] || "";

  assert.match(audioForm, /音频检索类型<select id="search-mode"/);
  assert.doesNotMatch(filters, /id="search-mode"/);
});

test("emotion text uses the tokenizer single-input path", () => {
  const worker = fs.readFileSync(new URL("../model_worker.py", import.meta.url), "utf8");

  assert.match(worker, /qwen\.tokenizer\(str\(rendered\), return_tensors="pt"\)/);
  assert.doesNotMatch(worker, /qwen\.tokenizer\(\[text\], return_tensors="pt"\)/);
});

test("audio upload supports click and drag-and-drop", () => {
  const html = fs.readFileSync(new URL("../public/index.html", import.meta.url), "utf8");
  const app = fs.readFileSync(new URL("../public/app.js", import.meta.url), "utf8");

  assert.match(html, /点击选择或拖入一段查询音频/);
  assert.match(app, /audioDropZone\.addEventListener\("drop"/);
  assert.match(app, /audioFileInput\.files = transfer\.files/);
});
