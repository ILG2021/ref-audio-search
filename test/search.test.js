import test from "node:test";
import assert from "node:assert/strict";
import { extractProsody } from "../src/audio.js";

test("extractProsody returns stable finite features", () => {
  const samples = new Float32Array(32000);
  for (let i = 0; i < samples.length; i++) samples[i] = Math.sin(i / 15) * (i < 16000 ? 0.2 : 0.05);
  const result = extractProsody(samples);
  assert.equal(result.vector.length, 7);
  assert.ok(result.vector.every(Number.isFinite));
  assert.equal(result.duration, 2);
});
