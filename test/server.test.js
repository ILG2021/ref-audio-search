import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import net from "node:net";
import { spawn } from "node:child_process";
import { once } from "node:events";

test("invalid audio ranges return 416 while the server stays available", async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "ref-audio-http-test-"));
  const file = path.join(directory, "sample.wav");
  fs.writeFileSync(file, Buffer.from("0123456789"));
  process.env.DB_PATH = path.join(directory, "test.db");
  const { upsertAudio, closeDb } = await import("../src/db.js");
  upsertAudio({ path: file, name: "sample.wav", size: 10, mtimeMs: 0, duration: 1, sampleRate: 16000, features: { prosody: { vector: [0, 0, 0, 0, 0, 0, 1] } }, featureVersion: "prosody-v1" });

  const reservation = net.createServer();
  reservation.listen(0, "127.0.0.1");
  await once(reservation, "listening");
  const port = reservation.address().port;
  await new Promise(resolve => reservation.close(resolve));
  const child = spawn(process.execPath, ["--disable-warning=ExperimentalWarning", "src/server.js"], {
    cwd: path.resolve("."),
    env: { ...process.env, PORT: String(port), MODEL_PYTHON: "" },
    stdio: "ignore"
  });
  t.after(async () => {
    if (child.exitCode === null && child.signalCode === null) {
      const exited = once(child, "exit");
      child.kill();
      await exited;
    }
    closeDb();
    fs.rmSync(directory, { recursive: true, force: true });
  });

  const base = `http://127.0.0.1:${port}`;
  let ready = false;
  for (let attempt = 0; attempt < 50; attempt++) {
    try { ready = (await fetch(`${base}/api/health`)).ok; } catch { /* startup */ }
    if (ready) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.ok(ready, "server did not start");
  const health = await (await fetch(`${base}/api/health`)).json();
  assert.equal(health.ok, false);

  const invalid = await fetch(`${base}/api/audio/1`, { headers: { range: "bytes=999999999-" } });
  assert.equal(invalid.status, 416);
  assert.equal(invalid.headers.get("content-range"), "bytes */10");
  const suffix = await fetch(`${base}/api/audio/1`, { headers: { range: "bytes=-3" } });
  assert.equal(suffix.status, 206);
  assert.equal(await suffix.text(), "789");
  assert.equal((await fetch(`${base}/api/health`)).status, 200);

  assert.equal((await fetch(`${base}/api/index`)).status, 404);
  assert.equal((await fetch(`${base}/api/index/status`)).status, 404);
  const textSearch = await fetch(`${base}/api/search/text`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text: "开心" })
  });
  assert.equal(textSearch.status, 503);
});
