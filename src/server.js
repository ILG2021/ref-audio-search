import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import os from "node:os";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { analyzeAudio } from "./audio.js";
import { addEvent, closeDb, contentCandidateIds, favoriteIds, getAudio, listAudio, listEvents, listFavorites, setFavorite, stats } from "./db.js";
import { searchByContent, searchByModel, searchByStyleText, textToPrototype } from "./search.js";
import { modelProvider } from "./model-provider.js";
import { PORT } from "./config.js";

const publicDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../public");
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8" };

function json(res, status, value) {
  res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(value));
}

function streamFile(req, res, file, download = false) {
  const stat = fs.statSync(file);
  const range = req.headers.range;
  const audioTypes = { ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".opus": "audio/ogg", ".m4a": "audio/mp4", ".aac": "audio/aac" };
  const headers = { "accept-ranges": "bytes", "content-type": audioTypes[path.extname(file).toLowerCase()] || "application/octet-stream" };
  if (download) headers["content-disposition"] = `attachment; filename*=UTF-8''${encodeURIComponent(path.basename(file))}`;
  if (stat.size === 0 && !range) {
    res.writeHead(200, { ...headers, "content-length": 0 });
    return res.end();
  }
  let start = 0;
  let end = stat.size - 1;
  if (range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (!match || (!match[1] && !match[2])) {
      res.writeHead(416, { "content-range": `bytes */${stat.size}` });
      return res.end();
    }
    if (match[1]) {
      start = Number(match[1]);
      end = match[2] ? Number(match[2]) : end;
    } else {
      const suffix = Number(match[2]);
      start = Math.max(0, stat.size - suffix);
    }
    end = Math.min(end, stat.size - 1);
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || start > end || start >= stat.size) {
      res.writeHead(416, { "content-range": `bytes */${stat.size}` });
      return res.end();
    }
  }
  const stream = fs.createReadStream(file, { start, end });
  stream.on("error", error => {
    if (res.headersSent) res.destroy(error);
    else json(res, 500, { error: error.message });
  });
  if (!range) {
    res.writeHead(200, { ...headers, "content-length": stat.size });
    return stream.pipe(res);
  }
  res.writeHead(206, { ...headers, "content-length": end - start + 1, "content-range": `bytes ${start}-${end}/${stat.size}` });
  stream.pipe(res);
}

async function readBody(req, max = 100 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > max) throw new Error("Upload exceeds 100 MB");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);
    if (url.pathname === "/api/health") return json(res, 200, { ok: modelProvider.status.enabled, stats: stats(), featureBackend: "IndexTTS2 style + emotion", model: modelProvider.status });
    if (url.pathname === "/api/library") {
      const onlyFavorites = url.searchParams.get("favorites") === "1";
      const items = onlyFavorites ? listFavorites() : listAudio();
      const favorites = favoriteIds();
      return json(res, 200, { items: items.map(item => ({ id: item.id, name: item.name, duration: item.duration, transcript: item.transcript, favorite: favorites.has(item.id), features: item.features.prosody })) });
    }
    if (url.pathname === "/api/events" && req.method === "POST") {
      addEvent(JSON.parse((await readBody(req)).toString("utf8")));
      return json(res, 201, { ok: true });
    }
    if (url.pathname === "/api/events/export" && req.method === "GET") {
      const rows = listEvents().map(event => JSON.stringify(event)).join("\n");
      res.writeHead(200, {
        "content-type": "application/x-ndjson; charset=utf-8",
        "content-disposition": `attachment; filename="preference-events-${new Date().toISOString().slice(0, 10)}.jsonl"`
      });
      return res.end(rows ? `${rows}\n` : "");
    }
    const favoriteMatch = url.pathname.match(/^\/api\/favorites\/(\d+)$/);
    if (favoriteMatch && req.method === "PUT") {
      const { favorite } = JSON.parse((await readBody(req)).toString("utf8"));
      setFavorite(Number(favoriteMatch[1]), Boolean(favorite));
      return json(res, 200, { ok: true, favorite: Boolean(favorite) });
    }
    if (url.pathname === "/api/search/text" && req.method === "POST") {
      if (!modelProvider.status.enabled) return json(res, 503, { error: "MODEL_PYTHON 未配置，IndexTTS2 不可用" });
      const { text, type, limit = 20, minDuration = 0, maxDuration = 0 } = JSON.parse((await readBody(req)).toString("utf8"));
      if (!String(text || "").trim()) return json(res, 400, { error: "请输入搜索文字" });
      if (!["emotion", "style", "content"].includes(type)) return json(res, 400, { error: "请选择情绪、发音风格或内容搜索" });
      const options = { limit, minDuration, maxDuration: maxDuration || Infinity, favorites: favoriteIds() };
      if (type === "content") return json(res, 200, { queryId: randomUUID(), query: { text, type }, ranking: { backend: "transcript", mode: "content" }, results: searchByContent(contentCandidateIds(text).map(getAudio).filter(Boolean), text, options) });
      const parsed = textToPrototype(text || "");
      if (type === "style") {
        if (!parsed.attributes.length) return json(res, 400, { error: "未识别到发音风格提示词，请使用语速、停顿、音量或表现力描述" });
        return json(res, 200, { queryId: randomUUID(), query: { text, type }, interpretation: { attributes: parsed.attributes }, ranking: { backend: "prosody", mode: "style-text" }, results: searchByStyleText(listAudio(), parsed, options) });
      }
      const modelText = await modelProvider.textEmotion(text || "");
      const modelResults = searchByModel(listAudio(), { style: null, emotion: modelText.emotion }, { ...options, mode: "emotion" });
      return json(res, 200, { queryId: randomUUID(), query: { text, type }, interpretation: { distribution: modelText.distribution, backend: "QwenEmotion" }, ranking: { backend: "IndexTTS2", mode: "emotion-text" }, results: modelResults });
    }
    if (url.pathname === "/api/search/audio" && req.method === "POST") {
      if (!modelProvider.status.enabled) return json(res, 503, { error: "MODEL_PYTHON 未配置，IndexTTS2 不可用" });
      const extension = path.extname(req.headers["x-filename"] || "query.wav") || ".wav";
      const temp = path.join(os.tmpdir(), `ref-query-${randomUUID()}${extension}`);
      fs.writeFileSync(temp, await readBody(req));
      try {
        const prosody = await analyzeAudio(temp);
        const settings = {
          limit: Number(url.searchParams.get("limit") || 20),
          minDuration: Number(url.searchParams.get("minDuration") || 0),
          maxDuration: Number(url.searchParams.get("maxDuration") || Infinity),
          favorites: favoriteIds()
        };
        const model = await modelProvider.extract(temp);
        settings.mode = url.searchParams.get("mode") || "mixed";
        return json(res, 200, { queryId: randomUUID(), query: prosody, model: model.modelVersion, ranking: { backend: "IndexTTS2", mode: settings.mode }, results: searchByModel(listAudio(), model, settings) });
      } finally { fs.rmSync(temp, { force: true }); }
    }
    const audioMatch = url.pathname.match(/^\/api\/audio\/(\d+)(\/download)?$/);
    if (audioMatch) {
      const item = getAudio(Number(audioMatch[1]));
      if (!item || !fs.existsSync(item.path)) return json(res, 404, { error: "Audio not found" });
      return streamFile(req, res, item.path, Boolean(audioMatch[2]));
    }
    const relative = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
    const file = path.resolve(publicDir, relative);
    if (!file.startsWith(publicDir) || !fs.existsSync(file)) return json(res, 404, { error: "Not found" });
    res.writeHead(200, { "content-type": types[path.extname(file)] || "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  } catch (error) {
    if (res.headersSent) res.destroy(error);
    else json(res, 500, { error: error.message });
  }
});

server.listen(PORT, "127.0.0.1", () => console.log(`Ref Audio Search: http://127.0.0.1:${PORT}`));

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.close(() => {
    modelProvider.stop();
    closeDb();
    process.exit(0);
  }));
}
