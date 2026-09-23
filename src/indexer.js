import fs from "node:fs";
import path from "node:path";
import { AUDIO_EXTENSIONS } from "./config.js";
import { analyzeAudio } from "./audio.js";
import { getAudioByPath, removeMissing, upsertAudio } from "./db.js";
import { modelProvider } from "./model-provider.js";
import { loadMetadata } from "./metadata.js";

export function findAudioFiles(root) {
  const result = [];
  const visit = directory => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const full = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(full);
      else if (AUDIO_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) result.push(full);
    }
  };
  visit(path.resolve(root));
  return result;
}

export async function indexDirectory(root, onProgress = () => {}, options = {}) {
  if (!modelProvider.status.enabled) throw new Error("MODEL_PYTHON 未配置，IndexTTS2 不可用");
  await modelProvider.ensureReady();
  const files = findAudioFiles(root);
  const metadata = loadMetadata(root);
  const errors = [];
  let completed = 0;
  let skipped = 0;
  for (const file of files) {
    try {
      const stat = fs.statSync(file);
      const resolved = path.resolve(file);
      const existing = getAudioByPath(resolved);
      const targetVersion = "indextts2-native-v1";
      const metadataText = metadata?.get(file) ?? null;
      const audioUnchanged = existing && existing.size === stat.size && existing.mtime_ms === stat.mtimeMs && existing.feature_version === targetVersion;
      const transcriptUnchanged = metadataText != null
        ? existing?.transcript_source === "metadata" && existing.transcript === metadataText
        : existing?.transcript_source === "faster-whisper";
      if (audioUnchanged && transcriptUnchanged) {
        skipped++;
        completed++;
        onProgress({ completed, total: files.length, file, skipped: true });
        continue;
      }
      const prosody = audioUnchanged ? existing.features.prosody : await analyzeAudio(file);
      const model = audioUnchanged ? { style: existing.features.style.embedding, emotion: existing.features.emotion.embedding } : await modelProvider.extract(file);
      const transcript = metadataText ?? await modelProvider.transcribe(file);
      upsertAudio({
        path: resolved, name: path.basename(file), size: stat.size, mtimeMs: stat.mtimeMs,
        duration: prosody.duration, sampleRate: 16000,
        features: { prosody, style: { embedding: model.style }, emotion: { embedding: model.emotion } },
        featureVersion: targetVersion,
        transcript,
        transcriptSource: metadataText != null ? "metadata" : "faster-whisper"
      });
    } catch (error) {
      errors.push({ file, error: error.message });
    }
    completed++;
    onProgress({ completed, total: files.length, file });
  }
  const removed = options.removeMissing ? removeMissing(new Set(files.map(file => path.resolve(file))), path.resolve(root) + path.sep) : 0;
  return { total: files.length, indexed: files.length - errors.length - skipped, skipped, removed, errors };
}
