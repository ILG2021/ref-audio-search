import path from "node:path";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { ROOT } from "./config.js";

class ModelProvider {
  constructor() {
    this.pending = new Map();
    this.sequence = 0;
    this.ready = null;
    this.status = { enabled: Boolean(process.env.MODEL_PYTHON), ready: false, error: null };
  }

  start() {
    if (!this.status.enabled || this.child) return;
    this.child = spawn(process.env.MODEL_PYTHON, [path.join(ROOT, "model_worker.py")], {
      cwd: ROOT, windowsHide: true,
      env: process.env
    });
    this.ready = new Promise((resolve, reject) => {
      const lines = createInterface({ input: this.child.stdout });
      lines.on("line", line => {
        let message;
        try { message = JSON.parse(line); } catch { return; }
        if ("ready" in message) {
          this.status.ready = message.ready; this.status.error = message.error || null;
          return message.ready ? resolve() : reject(new Error(message.error));
        }
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        message.error ? pending.reject(new Error(message.error)) : pending.resolve(message.result);
      });
      this.child.stderr.on("data", chunk => { this.status.error = chunk.toString().trim(); });
      this.child.on("exit", code => {
        this.status.ready = false;
        const error = new Error(`Model worker exited with code ${code}`);
        for (const request of this.pending.values()) request.reject(error);
        this.pending.clear(); this.child = null;
        reject(error);
      });
    });
  }

  async ensureReady() {
    this.start();
    if (!this.ready) throw new Error("MODEL_PYTHON 未配置，IndexTTS2 不可用");
    await this.ready;
  }

  async extract(file) {
    await this.ensureReady();
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.child.stdin.write(`${JSON.stringify({ id, action: "extract", path: path.resolve(file) })}\n`);
    });
  }

  async textEmotion(text) {
    await this.ensureReady();
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.child.stdin.write(`${JSON.stringify({ id, action: "text_emotion", text })}\n`);
    });
  }

  async transcribe(file) {
    await this.ensureReady();
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.child.stdin.write(`${JSON.stringify({ id, action: "transcribe", path: path.resolve(file) })}\n`);
    });
  }

  stop() {
    if (this.child) this.child.kill();
    this.child = null;
  }
}

export const modelProvider = new ModelProvider();
