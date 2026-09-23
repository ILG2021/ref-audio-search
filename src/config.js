import path from "node:path";

export const ROOT = path.resolve(process.cwd());
export const DATA_DIR = path.resolve(process.env.DATA_DIR || path.join(ROOT, ".data"));
export const DB_PATH = path.resolve(process.env.DB_PATH || path.join(DATA_DIR, "audio-search.db"));
export const CACHE_DIR = path.resolve(process.env.CACHE_DIR || path.join(DATA_DIR, "cache"));
export const PORT = Number(process.env.PORT || 4173);
export const AUDIO_EXTENSIONS = new Set([".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif"]);
