"""Single-process Gradio UI for reference-audio search."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np

from model_worker import Models


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("DB_PATH", ROOT / ".data" / "audio-search.db")).resolve()
PORT = int(os.environ.get("PORT", "7860"))
RESULT_HEADERS = ["排名", "ID", "文件名", "时长（秒）", "排序分数", "风格", "情绪", "收藏", "转录文本"]
MODEL = None
MODEL_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def load_items(favorites_only=False):
    with connect() as connection:
        sql = "SELECT a.* FROM audio_items a"
        if favorites_only:
            sql += " JOIN favorites f ON f.audio_id=a.id ORDER BY f.created_at DESC"
        else:
            sql += " ORDER BY a.id"
        rows = connection.execute(sql).fetchall()
        favorite_ids = {row[0] for row in connection.execute("SELECT audio_id FROM favorites")}
    return [{**dict(row), "features": json.loads(row["features_json"]), "favorite": row["id"] in favorite_ids} for row in rows]


def get_model():
    global MODEL
    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                MODEL = Models()
    return MODEL


def cosine(left, right):
    if left is None or right is None or len(left) != len(right):
        return None
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def filtered(items, minimum, maximum):
    maximum = float(maximum or 0) or float("inf")
    return [item for item in items if float(minimum or 0) <= item["duration"] <= maximum]


def model_search(items, query, mode, limit, minimum, maximum):
    matches = []
    for item in filtered(items, minimum, maximum):
        style = cosine(query.get("style"), item["features"].get("style", {}).get("embedding"))
        emotion = cosine(query.get("emotion"), item["features"].get("emotion", {}).get("embedding"))
        if style is None and emotion is None:
            continue
        score = style if mode == "style" else emotion if mode == "emotion" else (style or 0) * .65 + (emotion or 0) * .35
        matches.append({**item, "score": score, "style_score": style, "emotion_score": emotion})
    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[: max(1, min(100, int(limit or 20)))]


def normalize_content(text):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(text or "").lower())


def content_search(items, text, limit, minimum, maximum):
    needle = normalize_content(text)
    matches = []
    for item in filtered(items, minimum, maximum):
        content = normalize_content(item.get("transcript"))
        if needle and needle in content:
            matches.append({**item, "score": 1.0 if content == needle else .9})
    matches.sort(key=lambda item: (-item["score"], len(normalize_content(item.get("transcript")))))
    return matches[: max(1, min(100, int(limit or 20)))]


STYLE_RULES = [
    (r"慢速|语速慢|缓慢", "慢速", [0, 1, 2, 6]), (r"快速|语速快|急促", "快速", [0, 1, 2, 6]),
    (r"停顿多|多停顿|犹豫", "停顿多", [1, 2]), (r"少停顿|连贯|流畅", "少停顿", [1, 2]),
    (r"轻声|轻柔|低声|耳语", "轻声", [3, 4]), (r"大声|有力|高能量|强烈", "有力", [3, 4]),
    (r"克制|平缓|平稳|冷静", "克制", [4, 5]), (r"夸张|戏剧化|表现力", "夸张", [4, 5]),
]


def style_search(items, text, limit, minimum, maximum):
    matched = [(name, dimensions) for pattern, name, dimensions in STYLE_RULES if re.search(pattern, text, re.I)]
    if not matched:
        raise gr.Error("未识别到发音风格提示词，请使用语速、停顿、音量或表现力描述。")
    target = np.array([.84, .22, .42, .04, .035, .05, 6.0], dtype=np.float32)
    if "慢速" in [entry[0] for entry in matched]: target[[0, 1, 2, 6]] += [-.12, .12, .18, 1.5]
    if "快速" in [entry[0] for entry in matched]: target[[0, 1, 2, 6]] += [.08, -.08, -.12, -1]
    dimensions = sorted({index for _, indices in matched for index in indices})
    scales = np.array([.35, .8, .8, .08, .12, .12, 20], dtype=np.float32)
    matches = []
    for item in filtered(items, minimum, maximum):
        vector = np.asarray(item["features"]["prosody"]["vector"], dtype=np.float32)
        score = float(np.exp(-np.sqrt(np.mean(((target[dimensions] - vector[dimensions]) / scales[dimensions]) ** 2))))
        matches.append({**item, "score": score})
    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[: max(1, min(100, int(limit or 20)))], "、".join(name for name, _ in matched)


def table_rows(results):
    def score(value): return "—" if value is None else f"{value:.3f}"
    return [[index, item["id"], item["name"], round(item["duration"], 2), score(item.get("score")),
             score(item.get("style_score")), score(item.get("emotion_score")), "★" if item.get("favorite") else "", item.get("transcript", "")]
            for index, item in enumerate(results, 1)]


def record_event(session_id, query_id, event_type, candidate_id=None, position=None, search_mode=None, payload=None):
    with connect() as connection:
        connection.execute("INSERT INTO user_events(session_id,query_id,candidate_id,event_type,position,search_mode,payload_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                           (session_id, query_id, candidate_id, event_type, position, search_mode, json.dumps(payload or {}, ensure_ascii=False), now()))


def finish_search(results, session_id, mode, query, explanation):
    query_id = str(uuid.uuid4())
    record_event(session_id, query_id, "search", search_mode=mode, payload={"query": query, "resultIds": [item["id"] for item in results]})
    for position, item in enumerate(results, 1):
        record_event(session_id, query_id, "impression", item["id"], position, mode, {"score": item.get("score")})
    return table_rows(results), results, query_id, explanation


def search_audio(file_path, mode_label, limit, minimum, maximum, request: gr.Request):
    if not file_path:
        raise gr.Error("请上传一段查询音频。")
    mode = {"综合：风格 65% + 情绪 35%": "mixed", "仅发音风格": "style", "仅情绪": "emotion"}[mode_label]
    query = get_model().extract(file_path)
    results = model_search(load_items(), query, mode, limit, minimum, maximum)
    explanation = {"mixed": "综合排序：65% 风格 + 35% 情绪", "style": "按发音风格相似度排序", "emotion": "按情绪相似度排序"}[mode]
    return finish_search(results, request.session_hash, "audio", {"file": Path(file_path).name, "mode": mode}, explanation)


def search_text(text, type_label, limit, minimum, maximum, request: gr.Request):
    text = str(text or "").strip()
    if not text:
        raise gr.Error("请输入搜索文字。")
    items = load_items()
    if type_label == "情绪":
        query = get_model().text_emotion(text)
        results = model_search(items, {"emotion": query["emotion"]}, "emotion", limit, minimum, maximum)
        explanation = "按 IndexTTS2 QwenEmotion 情绪特征排序"
    elif type_label == "发音风格":
        results, attributes = style_search(items, text, limit, minimum, maximum)
        explanation = f"识别到：{attributes}"
    else:
        results = content_search(items, text, limit, minimum, maximum)
        explanation = "按转录文本匹配，忽略标点和空格"
    return finish_search(results, request.session_hash, "text", {"text": text, "type": type_label}, explanation)


def select_result(results, event: gr.SelectData):
    if not results or not event.index:
        return None, "请选择一条结果。", None, gr.DownloadButton(visible=False)
    row = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
    item = results[int(row)]
    details = f"**{item['name']}**  ·  {item['duration']:.1f} 秒  ·  排序分数 {item.get('score', 0):.3f}"
    return item["path"], details, item["id"], gr.DownloadButton(value=item["path"], visible=True)


def toggle_favorite(audio_id, results):
    if audio_id is None:
        raise gr.Error("请先在结果表格中选择一条音频。")
    with connect() as connection:
        exists = connection.execute("SELECT 1 FROM favorites WHERE audio_id=?", (audio_id,)).fetchone()
        if exists:
            connection.execute("DELETE FROM favorites WHERE audio_id=?", (audio_id,))
        else:
            connection.execute("INSERT INTO favorites(audio_id,created_at) VALUES(?,?)", (audio_id, now()))
    for item in results or []:
        if item["id"] == audio_id:
            item["favorite"] = not bool(exists)
    return table_rows(results or []), results, "已取消收藏" if exists else "已收藏"


def feedback(kind, audio_id, query_id, results, request: gr.Request):
    if audio_id is None:
        raise gr.Error("请先在结果表格中选择一条音频。")
    position = next((index for index, item in enumerate(results or [], 1) if item["id"] == audio_id), None)
    record_event(request.session_hash, query_id, kind, audio_id, position, "gradio")
    return "反馈已记录，谢谢。"


def selected_event(kind, audio_id, query_id, results, request: gr.Request):
    if audio_id is None:
        return
    position = next((index for index, item in enumerate(results or [], 1) if item["id"] == audio_id), None)
    record_event(request.session_hash, query_id, kind, audio_id, position, "gradio")


def similar_feedback(audio_id, query_id, results, request: gr.Request):
    return feedback("similar", audio_id, query_id, results, request)


def different_feedback(audio_id, query_id, results, request: gr.Request):
    return feedback("different", audio_id, query_id, results, request)


def record_play(audio_id, query_id, results, request: gr.Request):
    return selected_event("play", audio_id, query_id, results, request)


def record_download(audio_id, query_id, results, request: gr.Request):
    return selected_event("download", audio_id, query_id, results, request)


def record_pause(audio_id, query_id, results, request: gr.Request):
    return selected_event("play_pause", audio_id, query_id, results, request)


def export_events():
    target = ROOT / ".data" / f"preference-events-{datetime.now().date().isoformat()}.jsonl"
    with connect() as connection, target.open("w", encoding="utf-8") as output:
        for row in connection.execute("SELECT * FROM user_events ORDER BY id"):
            event = dict(row)
            output.write(json.dumps(event, ensure_ascii=False) + "\n")
    return str(target)


def show_favorites():
    results = [{**item, "score": None} for item in load_items(True)]
    return table_rows(results), results, "收藏列表"


def status_text():
    with connect() as connection:
        row = connection.execute("SELECT COUNT(*) count, COALESCE(SUM(duration),0) duration FROM audio_items").fetchone()
        favorites = connection.execute("SELECT COUNT(*) FROM favorites").fetchone()[0]
    return f"已索引 {row['count']} 条音频 · {row['duration'] / 3600:.1f} 小时 · {favorites} 个收藏"


CSS = """
.gradio-container {max-width: 1080px !important; margin: auto !important;}
#title h1 {font-family: Georgia, serif; font-size: 3.2rem; font-weight: 400;}
#results table {font-size: 13px;}
"""

with gr.Blocks(title="参考音频搜索", css=CSS, fill_width=False) as app:
    result_state = gr.State([])
    query_id_state = gr.State(None)
    selected_id = gr.State(None)
    gr.Markdown("# 找到对的表达方式", elem_id="title")
    with gr.Row():
        status = gr.Markdown(status_text())
        export_button = gr.DownloadButton("导出偏好数据", value=export_events)

    with gr.Tabs():
        with gr.Tab("参考音频"):
            audio_mode = gr.Dropdown(["综合：风格 65% + 情绪 35%", "仅发音风格", "仅情绪"], value="综合：风格 65% + 情绪 35%", label="音频检索类型")
            audio_input = gr.Audio(type="filepath", sources=["upload"], label="点击或拖入查询音频")
            audio_search_button = gr.Button("搜索相似表达", variant="primary")
        with gr.Tab("文字描述"):
            text_type = gr.Dropdown(["情绪", "发音风格", "内容／转录文本"], value="情绪", label="文字搜索类型")
            text_input = gr.Textbox(lines=4, label="文字描述", placeholder="例如：悲伤、失望，但不要愤怒")
            text_search_button = gr.Button("搜索匹配表达", variant="primary")

    with gr.Accordion("搜索设置", open=False):
        with gr.Row():
            result_limit = gr.Number(value=20, minimum=1, maximum=100, precision=0, label="返回数量")
            min_duration = gr.Number(value=0, minimum=0, label="最短时长（秒）")
            max_duration = gr.Number(value=0, minimum=0, label="最长时长（秒，0 表示不限）")

    with gr.Row():
        explanation = gr.Markdown("输入查询后，结果会出现在这里。")
        favorites_button = gr.Button("查看收藏", size="sm")
    results_table = gr.Dataframe(headers=RESULT_HEADERS, datatype=["number", "number", "str", "number", "str", "str", "str", "str", "str"], interactive=False, wrap=True, elem_id="results")

    gr.Markdown("### 选中结果")
    selected_details = gr.Markdown("请点击结果表格中的任意单元格。")
    selected_audio = gr.Audio(label="试听", interactive=False)
    with gr.Row():
        favorite_button = gr.Button("收藏／取消收藏")
        similar_button = gr.Button("相似")
        different_button = gr.Button("不相似")
        download_button = gr.DownloadButton("下载", visible=False)
    action_status = gr.Markdown()

    search_outputs = [results_table, result_state, query_id_state, explanation]
    audio_search_button.click(search_audio, [audio_input, audio_mode, result_limit, min_duration, max_duration], search_outputs)
    text_search_button.click(search_text, [text_input, text_type, result_limit, min_duration, max_duration], search_outputs)
    results_table.select(select_result, [result_state], [selected_audio, selected_details, selected_id, download_button])
    favorite_button.click(toggle_favorite, [selected_id, result_state], [results_table, result_state, action_status])
    similar_button.click(similar_feedback, [selected_id, query_id_state, result_state], action_status)
    different_button.click(different_feedback, [selected_id, query_id_state, result_state], action_status)
    favorites_button.click(show_favorites, outputs=[results_table, result_state, explanation])
    selected_audio.play(record_play, [selected_id, query_id_state, result_state], None)
    selected_audio.pause(record_pause, [selected_id, query_id_state, result_state], None)
    selected_audio.stop(record_pause, [selected_id, query_id_state, result_state], None)
    download_button.click(record_download, [selected_id, query_id_state, result_state], None)


if __name__ == "__main__":
    allowed_audio = [item["path"] for item in load_items() if Path(item["path"]).is_file()]
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=PORT, show_error=True, allowed_paths=allowed_audio)
