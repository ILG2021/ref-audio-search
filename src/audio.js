import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { windowsHide: true });
    let stderr = "";
    child.stderr.on("data", chunk => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", code => code === 0 ? resolve() : reject(new Error(stderr || `${command} exited ${code}`)));
  });
}

export async function decodeToPcm(file) {
  const temp = path.join(os.tmpdir(), `ref-audio-${randomUUID()}.f32le`);
  await run("ffmpeg", ["-v", "error", "-i", file, "-ac", "1", "-ar", "16000", "-f", "f32le", "-y", temp]);
  try {
    const raw = fs.readFileSync(temp);
    return new Float32Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 4)).slice();
  } finally {
    fs.rmSync(temp, { force: true });
  }
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

export function extractProsody(samples, sampleRate = 16000) {
  const frame = Math.round(sampleRate * 0.025);
  const hop = Math.round(sampleRate * 0.01);
  const rms = [];
  const zcr = [];
  for (let start = 0; start + frame <= samples.length; start += hop) {
    let energy = 0;
    let crossings = 0;
    for (let i = 0; i < frame; i++) {
      const value = samples[start + i];
      energy += value * value;
      if (i && (value >= 0) !== (samples[start + i - 1] >= 0)) crossings++;
    }
    rms.push(Math.sqrt(energy / frame));
    zcr.push(crossings / frame);
  }
  const noiseFloor = quantile(rms, 0.2);
  const peak = Math.max(...rms, 0);
  const threshold = Math.max(noiseFloor * 2.5, peak * 0.08, 1e-4);
  const voiced = rms.map(value => value >= threshold);
  let voicedFrames = 0;
  let pauses = 0;
  let pauseRun = 0;
  const pauseLengths = [];
  for (const active of voiced) {
    if (active) {
      voicedFrames++;
      if (pauseRun >= 20) { pauses++; pauseLengths.push(pauseRun * 0.01); }
      pauseRun = 0;
    } else pauseRun++;
  }
  if (pauseRun >= 20) { pauses++; pauseLengths.push(pauseRun * 0.01); }
  const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  const activeRms = rms.filter((_, index) => voiced[index]);
  const duration = samples.length / sampleRate;
  const vector = [
    voicedFrames / Math.max(1, voiced.length),
    pauses / Math.max(duration, 0.1),
    mean(pauseLengths),
    mean(activeRms),
    quantile(activeRms, 0.9) - quantile(activeRms, 0.1),
    mean(zcr.filter((_, index) => voiced[index])),
    duration
  ];
  return {
    duration,
    voicedRatio: vector[0],
    pausesPerSecond: vector[1],
    meanPauseDuration: vector[2],
    rmsMean: vector[3],
    dynamicRange: vector[4],
    zeroCrossingRate: vector[5],
    vector
  };
}

export async function analyzeAudio(file) {
  const samples = await decodeToPcm(file);
  return extractProsody(samples, 16000);
}
