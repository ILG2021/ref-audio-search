const $ = selector => document.querySelector(selector);
const results = $("#results");
const sessionId = localStorage.refAudioSession || (localStorage.refAudioSession = crypto.randomUUID());
const favoriteStorageKey = "refAudioFavorites";
const localFavorites = new Set(readLocalFavorites());
let currentQueryId = null;
let currentMode = "audio";

function readLocalFavorites() {
  try {
    const value = JSON.parse(localStorage.getItem(favoriteStorageKey) || "[]");
    return Array.isArray(value) ? value.map(Number).filter(Number.isSafeInteger) : [];
  } catch { return []; }
}

function saveLocalFavorites() {
  localStorage.setItem(favoriteStorageKey, JSON.stringify([...localFavorites].sort((a, b) => a - b)));
}

function setLocalFavorite(id, favorite) {
  if (favorite) localFavorites.add(id); else localFavorites.delete(id);
  saveLocalFavorites();
}

async function refreshHealth() {
  const health = await fetch("/api/health").then(response => response.json()).catch(() => null);
  $("#status").textContent = health?.ok
    ? `已索引 ${health.stats.count} 条音频 · ${(health.stats.duration / 3600).toFixed(1)} 小时 · ${health.stats.favorites} 个收藏 · ${health.featureBackend}`
    : health ? "IndexTTS2 未配置：请设置 MODEL_PYTHON 后重启服务。" : "服务连接失败";
  updateModeHelp();
}
await refreshHealth();

function updateModeHelp() {
  const mode = $("#search-mode").value;
  $("#mode-help").textContent = mode === "mixed"
      ? "综合排序分数 = 0.65 × 风格相似度 + 0.35 × 情绪相似度。"
      : mode === "style" ? "按 IndexTTS2 风格相似度排序。" : "按 IndexTTS2 情绪相似度排序。";
}
$("#search-mode").addEventListener("change", updateModeHelp);

const textTypeHints = {
  emotion: { hint: "提示词：开心、悲伤、愤怒、害怕、平静、失望。只描述想要的情绪。", placeholder: "例如：悲伤、失望，但不要愤怒" },
  style: { hint: "提示词：轻声、有力、语速快、语速慢、停顿多、少停顿、克制、夸张。可组合描述。", placeholder: "例如：轻声、语速快、停顿多" },
  content: { hint: "输入台词或其中一段文字；忽略标点和空格，按转录内容匹配。", placeholder: "例如：今天的天气真好" }
};
function updateTextHints() {
  const hints = textTypeHints[$("#text-type").value];
  $("#text-hints").textContent = hints.hint;
  $("#emotion-text").placeholder = hints.placeholder;
}
$("#text-type").addEventListener("change", updateTextHints);

function formatScore(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—";
}

function rankingExplanation(payload) {
  const ranking = payload.ranking;
  if (currentMode === "text") {
    if (ranking?.mode === "content") return "按转录文本匹配；忽略标点和空格，完整匹配优先。";
    if (ranking?.mode === "style-text") return "按识别到的发音提示词比较韵律特征，不使用情绪分数。";
    if (ranking?.mode === "emotion-text") return "按 IndexTTS2 情绪特征排序，不混入发音风格提示词。";
    return "";
  }
  if (currentMode !== "audio") return "";
  if (ranking?.backend !== "IndexTTS2") return "";
  if (ranking.mode === "mixed") return "综合排序：0.65 × 风格相似度 + 0.35 × 情绪相似度。下方为各项原始余弦相似度及最终排序分数，均不是准确率。";
  return ranking.mode === "style" ? "按风格相似度排序；分数为余弦相似度，不是准确率。" : "按情绪相似度排序；分数为余弦相似度，不是准确率。";
}

document.querySelectorAll(".tabs button").forEach(button => button.addEventListener("click", () => {
  currentMode = button.dataset.mode;
  document.querySelectorAll(".tabs button").forEach(item => item.classList.toggle("active", item === button));
  $("#audio-form").hidden = currentMode !== "audio";
  $("#text-form").hidden = currentMode !== "text";
}));

$("#audio-file").addEventListener("change", event => {
  const file = event.target.files[0];
  if (file) document.querySelector(".drop span").textContent = file.name;
});

function searchSettings() {
  return {
    limit: Math.min(100, Math.max(1, Number($("#result-limit").value) || 20)),
    minDuration: Math.max(0, Number($("#min-duration").value) || 0),
    maxDuration: Math.max(0, Number($("#max-duration").value) || 0)
  };
}

function logEvent(eventType, candidateId, position, payload = {}) {
  fetch("/api/events", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({
    sessionId, queryId: currentQueryId, candidateId, eventType, position, searchMode: currentMode, payload
  }) }).catch(() => {});
}

function render(payload, { recordSearch = true } = {}) {
  currentQueryId = payload.queryId || crypto.randomUUID();
  const interpretation = payload.interpretation;
  $("#interpretation").textContent = interpretation
    ? `${interpretation.emotion ? `识别为：${interpretation.emotion}` : ""}${interpretation.attributes?.length ? `识别到：${interpretation.attributes.join("、")}` : ""}` : "";
  const explanation = rankingExplanation(payload);
  $("#ranking-explanation").textContent = explanation;
  $("#ranking-explanation").hidden = !explanation;
  if (recordSearch) logEvent("search", null, null, {
    query: payload.query || null,
    interpretation: payload.interpretation || null,
    resultIds: (payload.results || []).map(item => item.id),
    scores: (payload.results || []).map(item => ({ id: item.id, score: item.score ?? null, style: item.styleScore ?? null, emotion: item.emotionScore ?? null }))
  });
  if (!payload.results?.length) {
    results.innerHTML = '<p class="empty">没有符合当前条件的音频。</p>';
    return;
  }
  for (const item of payload.results) {
    if (item.favorite) setLocalFavorite(Number(item.id), true);
    item.favorite = item.favorite || localFavorites.has(Number(item.id));
  }
  results.innerHTML = payload.results.map((item, index) => `
    <article class="result" data-id="${item.id}" data-position="${index + 1}">
      <div class="rank">${String(index + 1).padStart(2, "0")}</div>
      <div><div class="name">${escapeHtml(item.name)}</div><div class="meta">${item.duration.toFixed(1)} 秒 · 停顿 ${item.features.pausesPerSecond.toFixed(2)}/秒</div>${item.transcript ? `<div class="transcript">${escapeHtml(item.transcript)}</div>` : ""}${payload.ranking?.backend === "IndexTTS2" && payload.ranking.mode !== "emotion-text" ? `<div class="score-breakdown">风格 ${formatScore(item.styleScore)} · 情绪 ${formatScore(item.emotionScore)} · 排序分数 ${formatScore(item.score)}</div>` : item.score == null ? "" : `<div class="score-breakdown">排序分数 ${formatScore(item.score)}</div>`}</div>
      <div class="actions"><button data-play>试听</button><button data-favorite class="${item.favorite ? "selected" : ""}">${item.favorite ? "已收藏" : "收藏"}</button><a data-download href="/api/audio/${item.id}/download">下载</a></div>
      <div class="feedback"><small>这个结果：</small><button data-feedback="similar">相似</button><button data-feedback="different">不相似</button></div>
    </article>`).join("");
  bindResultActions();
  if (recordSearch) payload.results.forEach((item, index) => logEvent("impression", item.id, index + 1, { score: item.score ?? null }));
}

function bindResultActions() {
  document.querySelectorAll(".result").forEach(article => {
    const id = Number(article.dataset.id);
    const position = Number(article.dataset.position);
    article.querySelector("[data-play]").addEventListener("click", () => {
      let player = article.querySelector("audio");
      if (!player) {
        player = document.createElement("audio"); player.controls = true; player.src = `/api/audio/${id}`; article.append(player);
        player.addEventListener("ended", () => logEvent("play_complete", id, position));
        const milestones = new Set();
        player.addEventListener("timeupdate", () => {
          if (!player.duration) return;
          const ratio = player.currentTime / player.duration;
          for (const mark of [0.25, 0.5, 0.75]) {
            if (ratio >= mark && !milestones.has(mark)) { milestones.add(mark); logEvent("play_progress", id, position, { ratio: mark, seconds: player.currentTime }); }
          }
        });
        player.addEventListener("pause", () => {
          if (!player.ended && player.currentTime > 0) logEvent("play_pause", id, position, { seconds: player.currentTime, ratio: player.duration ? player.currentTime / player.duration : null });
        });
      }
      player.play(); logEvent("play", id, position);
    });
    article.querySelector("[data-download]").addEventListener("click", () => logEvent("download", id, position));
    article.querySelector("[data-favorite]").addEventListener("click", async event => {
      const favorite = !event.currentTarget.classList.contains("selected");
      const response = await fetch(`/api/favorites/${id}`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ favorite }) });
      if (!response.ok) return;
      setLocalFavorite(id, favorite);
      event.currentTarget.classList.toggle("selected", favorite); event.currentTarget.textContent = favorite ? "已收藏" : "收藏";
      logEvent(favorite ? "favorite" : "unfavorite", id, position); refreshHealth();
    });
    article.querySelectorAll("[data-feedback]").forEach(button => button.addEventListener("click", () => {
      article.querySelectorAll("[data-feedback]").forEach(item => item.classList.remove("selected"));
      button.classList.add("selected"); logEvent(button.dataset.feedback, id, position);
    }));
  });
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}

async function submit(form, work) {
  const button = form.querySelector("button.primary");
  button.disabled = true; const original = button.textContent; button.textContent = "处理中…";
  results.innerHTML = '<p class="empty">正在分析并搜索…</p>';
  try { render(await work()); }
  catch (error) { results.innerHTML = `<p class="empty">操作失败：${escapeHtml(error.message)}</p>`; }
  finally { button.disabled = false; button.textContent = original; }
}

$("#audio-form").addEventListener("submit", event => {
  event.preventDefault(); const file = $("#audio-file").files[0]; const settings = searchSettings();
  submit(event.currentTarget, async () => {
    const params = new URLSearchParams({ limit: settings.limit, minDuration: settings.minDuration });
    params.set("mode", $("#search-mode").value);
    if (settings.maxDuration) params.set("maxDuration", settings.maxDuration);
    const response = await fetch(`/api/search/audio?${params}`, { method: "POST", headers: { "content-type": file.type || "application/octet-stream", "x-filename": file.name }, body: file });
    if (!response.ok) throw new Error((await response.json()).error); return response.json();
  });
});

$("#text-form").addEventListener("submit", event => {
  event.preventDefault(); const settings = searchSettings();
  submit(event.currentTarget, async () => {
    const response = await fetch("/api/search/text", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text: $("#emotion-text").value, type: $("#text-type").value, ...settings }) });
    if (!response.ok) throw new Error((await response.json()).error); return response.json();
  });
});

$("#favorites-button").addEventListener("click", async () => {
  currentMode = "favorites";
  const payload = await fetch("/api/library?favorites=1").then(response => response.json());
  render({ results: payload.items }, { recordSearch: false });
});
