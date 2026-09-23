import fs from "node:fs";
import path from "node:path";

function parseCsv(text, delimiter) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (char === delimiter && !quoted) { row.push(field); field = ""; }
    else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[i + 1] === "\n") i++;
      row.push(field); if (row.some(value => value.trim())) rows.push(row);
      row = []; field = "";
    } else field += char;
  }
  if (quoted) throw new Error("metadata.csv 包含未闭合的引号");
  row.push(field); if (row.some(value => value.trim())) rows.push(row);
  return rows;
}

export function loadMetadata(root) {
  const file = path.join(root, "metadata.csv");
  if (!fs.existsSync(file)) return null;
  const text = fs.readFileSync(file, "utf8").replace(/^\uFEFF/, "");
  const firstLine = text.split(/\r?\n/, 1)[0];
  const rows = parseCsv(text, firstLine.includes("|") ? "|" : ",");
  const byKey = new Map();
  for (const row of rows) {
    const [rawName, transcript] = row;
    if (!rawName?.trim() || !transcript?.trim()) continue;
    const name = rawName.trim().replaceAll("\\", "/");
    const stem = path.posix.parse(name).name.toLowerCase();
    const key = path.posix.basename(name).toLowerCase();
    byKey.set(key, transcript.trim());
    if (!path.posix.extname(name)) byKey.set(stem, transcript.trim());
  }
  return {
    file,
    get(audioPath) {
      const basename = path.basename(audioPath).toLowerCase();
      return byKey.get(basename) ?? byKey.get(path.parse(basename).name) ?? null;
    }
  };
}
