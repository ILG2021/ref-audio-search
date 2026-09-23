import test from "node:test";
import assert from "node:assert/strict";
import { extractProsody } from "../src/audio.js";
import { searchByModel, searchByVector, textToPrototype } from "../src/search.js";

test("extractProsody returns stable finite features", () => {
  const samples = new Float32Array(32000);
  for (let i = 0; i < samples.length; i++) samples[i] = Math.sin(i / 15) * (i < 16000 ? 0.2 : 0.05);
  const result = extractProsody(samples);
  assert.equal(result.vector.length, 7);
  assert.ok(result.vector.every(Number.isFinite));
  assert.equal(result.duration, 2);
});

test("search ranks an identical vector first", () => {
  const items = [
    { id: 1, name: "far.wav", duration: 2, features: { prosody: { vector: [0,0,0,0,0,0,2] } } },
    { id: 2, name: "near.wav", duration: 2, features: { prosody: { vector: [1,1,1,1,1,1,2] } } }
  ];
  assert.equal(searchByVector(items, [1,1,1,1,1,1,2])[0].id, 2);
});

test("emotion text is interpreted", () => {
  assert.equal(textToPrototype("悲伤但克制").emotion, "sad");
  assert.equal(textToPrototype("没有明确描述").emotion, "neutral");
  assert.ok(textToPrototype("慢速，停顿多，轻声克制").attributes.includes("慢速"));
  assert.ok(!textToPrototype("愤怒但克制，不要喊叫").attributes.includes("有力"));
});

test("model search separates style and emotion ranking", () => {
  const items = [
    { id: 1, name: "style.wav", duration: 2, features: { prosody: { vector: [0] }, style: { embedding: [1, 0] }, emotion: { embedding: [0, 1] } } },
    { id: 2, name: "emotion.wav", duration: 2, features: { prosody: { vector: [0] }, style: { embedding: [0, 1] }, emotion: { embedding: [1, 0] } } }
  ];
  const query = { style: [1, 0], emotion: [1, 0] };
  assert.equal(searchByModel(items, query, { mode: "style" })[0].id, 1);
  assert.equal(searchByModel(items, query, { mode: "emotion" })[0].id, 2);
});

test("text prosody attributes change ranking within the same emotion", () => {
  const slow = textToPrototype("悲伤，语速慢，停顿多");
  const fast = textToPrototype("悲伤，语速快，少停顿");
  const items = [
    { id: 1, name: "slow.wav", duration: 2, features: { prosody: { vector: slow.vector }, emotion: { embedding: [1, 0] } } },
    { id: 2, name: "fast.wav", duration: 2, features: { prosody: { vector: fast.vector }, emotion: { embedding: [1, 0] } } }
  ];
  const query = { emotion: [1, 0] };
  assert.equal(searchByModel(items, query, { mode: "emotion", prosody: slow })[0].id, 1);
  assert.equal(searchByModel(items, query, { mode: "emotion", prosody: fast })[0].id, 2);
});
