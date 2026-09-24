"""Single-process Gradio UI for reference-audio search."""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import threading
from pathlib import Path

# Disable Gradio usage analytics before importing it. This process-level flag
# also covers telemetry initialized outside the Blocks instance.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np

from model_worker import Models


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("DB_PATH", ROOT / ".data" / "audio-search.db")).resolve()
PORT = int(os.environ.get("PORT", "7860"))
ROOT_PATH = os.environ.get("ROOT_PATH", "").strip()
MODEL = None
MODEL_LOCK = threading.Lock()
LIBRARY_LOCK = threading.Lock()
LIBRARY_SIGNATURE = None
LIBRARY_ITEMS = []
LIBRARY_BY_ID = {}
LIBRARY_STYLE = None
LIBRARY_EMOTION = None
LIBRARY_STYLE_VALID = None
LIBRARY_EMOTION_VALID = None
LIBRARY_DURATIONS = None
LIBRARY_PROSODY = None


def connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def library_signature(connection):
    row = connection.execute(
        "SELECT COUNT(*) count, COALESCE(MAX(id),0) max_id, COALESCE(MAX(indexed_at),'') newest FROM audio_items"
    ).fetchone()
    return row["count"], row["max_id"], row["newest"]


def load_library():
    global LIBRARY_SIGNATURE, LIBRARY_ITEMS, LIBRARY_BY_ID, LIBRARY_DURATIONS, LIBRARY_PROSODY
    global LIBRARY_STYLE, LIBRARY_EMOTION, LIBRARY_STYLE_VALID, LIBRARY_EMOTION_VALID
    with connect() as connection:
        signature = library_signature(connection)
    if signature == LIBRARY_SIGNATURE:
        return LIBRARY_ITEMS
    with LIBRARY_LOCK, connect() as connection:
        signature = library_signature(connection)
        if signature != LIBRARY_SIGNATURE:
            rows = connection.execute("SELECT * FROM audio_items ORDER BY id").fetchall()
            LIBRARY_ITEMS = [{**dict(row), "features": json.loads(row["features_json"])} for row in rows]
            for index, item in enumerate(LIBRARY_ITEMS):
                item["_cache_index"] = index

            def embedding_matrix(kind):
                vectors = [item["features"].get(kind, {}).get("embedding") for item in LIBRARY_ITEMS]
                dimension = next((len(vector) for vector in vectors if vector), 0)
                matrix = np.zeros((len(vectors), dimension), dtype=np.float32)
                valid = np.zeros(len(vectors), dtype=bool)
                for index, vector in enumerate(vectors):
                    if vector is None or len(vector) != dimension:
                        continue
                    array = np.asarray(vector, dtype=np.float32)
                    norm = float(np.linalg.norm(array))
                    if norm > 1e-12:
                        matrix[index] = array / norm
                        valid[index] = True
                return matrix, valid

            LIBRARY_STYLE, LIBRARY_STYLE_VALID = embedding_matrix("style")
            LIBRARY_EMOTION, LIBRARY_EMOTION_VALID = embedding_matrix("emotion")
            LIBRARY_DURATIONS = np.fromiter(
                (item["duration"] for item in LIBRARY_ITEMS), dtype=np.float32, count=len(LIBRARY_ITEMS)
            )
            LIBRARY_PROSODY = np.asarray(
                [item["features"].get("prosody", {}).get("vector", [0] * 7) for item in LIBRARY_ITEMS],
                dtype=np.float32,
            )
            # The dense matrices above are the compact runtime copy of the
            # embeddings. Keeping the original JSON lists as well would
            # multiply RAM usage for libraries with tens of thousands of files.
            for item in LIBRARY_ITEMS:
                item["features"] = {"prosody": item["features"].get("prosody", {})}
            LIBRARY_BY_ID = {item["id"]: item for item in LIBRARY_ITEMS}
            LIBRARY_SIGNATURE = signature
    return LIBRARY_ITEMS


def normalize_favorite_ids(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(int(audio_id) for audio_id in value if str(audio_id).isdigit()))


def serialize_favorite_ids(value):
    return json.dumps(normalize_favorite_ids(value), separators=(",", ":"))


def load_items(favorite_ids=None):
    # The library cache is immutable between database signature changes. Do not
    # copy tens of thousands of dictionaries merely to decorate search results.
    return load_library()


def decorate_favorites(items, favorite_ids):
    favorites = set(normalize_favorite_ids(favorite_ids))
    return [{**item, "favorite": item["id"] in favorites} for item in items]


def content_candidate_ids(text):
    normalized = normalize_content(text)
    chars = list(normalized)
    if len(chars) < 2:
        with connect() as connection:
            return {row[0] for row in connection.execute("SELECT id FROM audio_items WHERE transcript != ''")}
    grams = list(dict.fromkeys(a + b for a, b in zip(chars, chars[1:])))
    if len(grams) > 900:
        return None
    placeholders = ",".join("?" for _ in grams)
    sql = (f"SELECT audio_id FROM content_grams WHERE gram IN ({placeholders}) "
           "GROUP BY audio_id HAVING COUNT(*) = ?")
    with connect() as connection:
        return {row[0] for row in connection.execute(sql, (*grams, len(grams)))}


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
    if items and all("_cache_index" in item for item in items):
        indices = np.fromiter((item["_cache_index"] for item in items), dtype=np.int64, count=len(items))
        if items is LIBRARY_ITEMS and LIBRARY_DURATIONS is not None:
            durations = LIBRARY_DURATIONS
        else:
            durations = LIBRARY_DURATIONS[indices]
        duration_valid = durations >= float(minimum or 0)
        if float(maximum or 0) > 0:
            duration_valid &= durations <= float(maximum)

        def similarities(vector, matrix, valid):
            if vector is None or matrix is None or not matrix.shape[1]:
                return np.zeros(len(items), dtype=np.float32), np.zeros(len(items), dtype=bool)
            vector = np.asarray(vector, dtype=np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            return matrix[indices] @ vector, valid[indices]

        style, style_valid = similarities(query.get("style"), LIBRARY_STYLE, LIBRARY_STYLE_VALID)
        emotion, emotion_valid = similarities(query.get("emotion"), LIBRARY_EMOTION, LIBRARY_EMOTION_VALID)
        if mode == "style":
            scores, eligible = style, style_valid
        elif mode == "emotion":
            scores, eligible = emotion, emotion_valid
        else:
            scores, eligible = style * .65 + emotion * .35, style_valid | emotion_valid
        eligible &= duration_valid
        count = max(1, min(100, int(limit or 10)))
        eligible_indices = np.flatnonzero(eligible)
        if len(eligible_indices) > count:
            candidate_scores = scores[eligible_indices]
            top = np.argpartition(candidate_scores, -count)[-count:]
            order = eligible_indices[top[np.argsort(-candidate_scores[top], kind="stable")]]
        else:
            order = eligible_indices[np.argsort(-scores[eligible_indices], kind="stable")]
        results = []
        for position in order:
            item = items[int(position)]
            results.append({**item, "score": float(scores[position]),
                            "style_score": float(style[position]) if style_valid[position] else None,
                            "emotion_score": float(emotion[position]) if emotion_valid[position] else None})
        return results

    matches = []
    for item in filtered(items, minimum, maximum):
        style = cosine(query.get("style"), item["features"].get("style", {}).get("embedding"))
        emotion = cosine(query.get("emotion"), item["features"].get("emotion", {}).get("embedding"))
        if style is None and emotion is None:
            continue
        score = style if mode == "style" else emotion if mode == "emotion" else (style or 0) * .65 + (emotion or 0) * .35
        matches.append({**item, "score": score, "style_score": style, "emotion_score": emotion})
    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[: max(1, min(100, int(limit or 10)))]


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
    return matches[: max(1, min(100, int(limit or 10)))]


STYLE_RULES = [
    (r"慢速|语速慢|缓慢", "慢速", [0, 1, 2, 6]), (r"快速|语速快|急促", "快速", [0, 1, 2, 6]),
    (r"停顿多|多停顿|犹豫", "停顿多", [1, 2]), (r"少停顿|连贯|流畅", "少停顿", [1, 2]),
    (r"轻声|轻柔|低声|耳语", "轻声", [3, 4]), (r"大声|有力|高能量|强烈", "有力", [3, 4]),
    (r"克制|平缓|平稳|冷静", "克制", [4, 5]), (r"夸张|戏剧化|表现力", "夸张", [4, 5]),
]


def style_search(items, text, limit, minimum, maximum):
    matched = [
        (name, dimensions)
        for pattern, name, dimensions in STYLE_RULES
        if re.search(pattern, text, re.IGNORECASE)
    ]
    if not matched:
        raise gr.Error("未识别到发音风格提示词，请使用语速、停顿、音量或表现力描述。")
    target = np.array([.84, .22, .42, .04, .035, .05, 6.0], dtype=np.float32)
    names = {entry[0] for entry in matched}
    if "慢速" in names: target[[0, 1, 2, 6]] += [-.12, .12, .18, 1.5]
    if "快速" in names: target[[0, 1, 2, 6]] += [.08, -.08, -.12, -1]
    if "停顿多" in names: target[[1, 2]] += [.35, .25]
    if "少停顿" in names: target[[1, 2]] += [-.14, -.15]
    if "轻声" in names: target[[3, 4]] *= [.58, .70]
    if "有力" in names: target[[3, 4]] *= [1.55, 1.40]
    if "克制" in names: target[[4, 5]] *= [.62, .75]
    if "夸张" in names: target[[4, 5]] *= [1.55, 1.30]
    dimensions = sorted({index for _, indices in matched for index in indices})
    scales = np.array([.35, .8, .8, .08, .12, .12, 20], dtype=np.float32)
    if items and all("_cache_index" in item for item in items) and LIBRARY_PROSODY is not None:
        indices = np.fromiter((item["_cache_index"] for item in items), dtype=np.int64, count=len(items))
        vectors = LIBRARY_PROSODY if items is LIBRARY_ITEMS else LIBRARY_PROSODY[indices]
        durations = LIBRARY_DURATIONS if items is LIBRARY_ITEMS else LIBRARY_DURATIONS[indices]
        scores = np.exp(-np.sqrt(np.mean(
            ((target[dimensions] - vectors[:, dimensions]) / scales[dimensions]) ** 2, axis=1
        )))
        eligible = durations >= float(minimum or 0)
        if float(maximum or 0) > 0:
            eligible &= durations <= float(maximum)
        eligible_indices = np.flatnonzero(eligible)
        count = max(1, min(100, int(limit or 10)))
        if len(eligible_indices) > count:
            candidate_scores = scores[eligible_indices]
            top = np.argpartition(candidate_scores, -count)[-count:]
            order = eligible_indices[top[np.argsort(-candidate_scores[top], kind="stable")]]
        else:
            order = eligible_indices[np.argsort(-scores[eligible_indices], kind="stable")]
        results = [{**items[int(position)], "score": float(scores[position])} for position in order]
        return results, "、".join(name for name, _ in matched)

    matches = []
    for item in filtered(items, minimum, maximum):
        vector = np.asarray(item["features"]["prosody"]["vector"], dtype=np.float32)
        score = float(np.exp(-np.sqrt(np.mean(((target[dimensions] - vector[dimensions]) / scales[dimensions]) ** 2))))
        matches.append({**item, "score": score})
    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[: max(1, min(100, int(limit or 10)))], "、".join(name for name, _ in matched)


def result_choices(results):
    choices = []
    for index, item in enumerate(results, 1):
        score = "—" if item.get("score") is None else f"{item['score']:.3f}"
        favorite = "★ " if item.get("favorite") else ""
        transcript = re.sub(r"\s+", " ", item.get("transcript", "")).strip()
        suffix = f" ｜ {transcript[:80]}" if transcript else ""
        choices.append(f"{index:02d}　{favorite}{item['name']}　·　{item['duration']:.1f} 秒　·　分数 {score}{suffix}")
    return choices


def result_selector(results, selected_id=None):
    choices = result_choices(results)
    selected_index = next(
        (index for index, item in enumerate(results) if item["id"] == selected_id), None
    )
    selected_value = choices[selected_index] if selected_index is not None else None
    return gr.Radio(choices=choices, value=selected_value, type="index")


def finish_search(results, explanation, favorite_ids=None):
    results = decorate_favorites(results, favorite_ids)
    return (result_selector(results), results, explanation, None,
            "请选择一条结果。", None, gr.DownloadButton(visible=False))


def search_audio(file_path, mode_label, limit, minimum, maximum, favorite_ids):
    if not file_path:
        raise gr.Error("请上传一段查询音频。")
    mode = {"综合：风格 65% + 情绪 35%": "mixed", "仅发音风格": "style", "仅情绪": "emotion"}[mode_label]
    query = get_model().extract(file_path)
    results = model_search(load_library(), query, mode, limit, minimum, maximum)
    explanation = {"mixed": "综合排序：65% 风格 + 35% 情绪", "style": "按发音风格相似度排序", "emotion": "按情绪相似度排序"}[mode]
    return finish_search(results, explanation, favorite_ids)


def search_text(text, type_label, limit, minimum, maximum, favorite_ids):
    text = str(text or "").strip()
    if not text:
        raise gr.Error("请输入搜索文字。")
    items = load_library()
    if type_label == "情绪":
        query = get_model().text_emotion(text)
        results = model_search(items, {"emotion": query["emotion"]}, "emotion", limit, minimum, maximum)
        explanation = "按 IndexTTS2 QwenEmotion 情绪特征排序"
    elif type_label == "发音风格":
        results, attributes = style_search(items, text, limit, minimum, maximum)
        explanation = f"识别到：{attributes}"
    else:
        candidate_ids = content_candidate_ids(text)
        if candidate_ids is None:
            candidates = items
        else:
            candidates = [LIBRARY_BY_ID[audio_id] for audio_id in candidate_ids if audio_id in LIBRARY_BY_ID]
        results = content_search(candidates, text, limit, minimum, maximum)
        explanation = "按转录文本匹配，忽略标点和空格"
    return finish_search(results, explanation, favorite_ids)


def select_result(index, results):
    if index is None or not results:
        # A result list refresh (for example after toggling a favorite) may
        # briefly report no selection. Keep the current preview instead of
        # clearing an audio the user just selected.
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    item = results[int(index)]
    score = "—" if item.get("score") is None else f"{item['score']:.3f}"
    details = f"**{item['name']}**  ·  {item['duration']:.1f} 秒  ·  排序分数 {score}"
    return item["path"], details, item["id"], gr.DownloadButton(value=item["path"], visible=True)


def toggle_favorite(audio_id, results, favorite_ids):
    if audio_id is None:
        raise gr.Error("请先在结果表格中选择一条音频。")
    favorite_ids = normalize_favorite_ids(favorite_ids)
    exists = audio_id in favorite_ids
    favorite_ids = [item_id for item_id in favorite_ids if item_id != audio_id]
    if not exists:
        favorite_ids.insert(0, audio_id)
    for item in results or []:
        if item["id"] == audio_id:
            item["favorite"] = not exists
    return (result_selector(results or [], selected_id=audio_id), results, serialize_favorite_ids(favorite_ids),
            "已取消收藏" if exists else "已收藏")


def toggle_favorite_in_list(audio_id, results, favorite_ids):
    _, updated, favorite_ids, message = toggle_favorite(audio_id, results, favorite_ids)
    remaining = [item for item in updated if item.get("favorite")]
    return result_selector(remaining), remaining, favorite_ids, message


def show_favorites(favorite_ids):
    order = normalize_favorite_ids(favorite_ids)
    # localStorage contains only compact IDs; resolve their display metadata
    # from the already-loaded in-memory index without touching SQLite again.
    if not LIBRARY_BY_ID:
        load_library()
    results = [
        {**LIBRARY_BY_ID[audio_id], "score": None, "favorite": True}
        for audio_id in order if audio_id in LIBRARY_BY_ID
    ]
    return (result_selector(results), results, "收藏列表", None,
            "请选择一条收藏。", None, gr.DownloadButton(visible=False))


def status_text():
    with connect() as connection:
        row = connection.execute("SELECT COUNT(*) count, COALESCE(SUM(duration),0) duration FROM audio_items").fetchone()
    return f"已索引 {row['count']} 条音频 · {row['duration'] / 3600:.1f} 小时"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="参考音频搜索 Gradio 服务")
    parser.add_argument(
        "--root-path",
        default=ROOT_PATH,
        help="反向代理部署的 URL 前缀，例如 /audio-search（默认读取 ROOT_PATH）",
    )
    return parser.parse_args(argv)


CSS = """
#title h1 {font-family: Georgia, serif; font-size: 3.2rem; font-weight: 400;}
.result-list label {padding: 10px 12px !important;}
"""

def build_result_panel(empty_text):
    explanation = gr.Markdown(empty_text)
    selector = gr.Radio([], type="index", label="检索结果", elem_classes="result-list")
    gr.Markdown("### 选中结果")
    details = gr.Markdown("请选择一条结果。")
    audio = gr.Audio(label="试听", interactive=False)
    with gr.Row():
        favorite = gr.Button("收藏／取消收藏")
        download = gr.DownloadButton("下载", visible=False)
    action_status = gr.Markdown()
    return selector, explanation, details, audio, favorite, download, action_status


def wire_result_panel(selector, result_state, selected_id, favorite_ids, details, audio,
                      favorite, download, action_status, favorite_handler=toggle_favorite):
    # Only a user's selection should load the preview. Updating the result list
    # to redraw a favorite star must not reload or clear the current audio.
    selector.input(select_result, [selector, result_state], [audio, details, selected_id, download])
    favorite.click(favorite_handler, [selected_id, result_state, favorite_ids],
                   [selector, result_state, favorite_ids, action_status])


with gr.Blocks(
    title="参考音频搜索",
    css=CSS,
    fill_width=True,
    analytics_enabled=False,
    delete_cache=(300, 300),
) as app:
    favorite_ids = gr.Textbox(value="[]", visible=False)
    gr.Markdown("# 找到对的表达方式", elem_id="title")
    status = gr.Markdown(status_text())

    with gr.Accordion("搜索设置", open=False):
        with gr.Row():
            result_limit = gr.Number(value=10, minimum=1, maximum=100, precision=0, label="返回数量")
            min_duration = gr.Number(value=0, minimum=0, label="最短时长（秒）")
            max_duration = gr.Number(value=0, minimum=0, label="最长时长（秒，0 表示不限）")

    with gr.Tabs():
        with gr.Tab("音频搜索"):
            audio_results = gr.State([])
            audio_selected_id = gr.State(None)
            audio_mode = gr.Dropdown(["综合：风格 65% + 情绪 35%", "仅发音风格", "仅情绪"], value="综合：风格 65% + 情绪 35%", label="音频检索类型")
            audio_input = gr.Audio(type="filepath", sources=["upload"], label="点击或拖入查询音频")
            audio_search_button = gr.Button("搜索相似表达", variant="primary")
            (audio_selector, audio_explanation, audio_details, audio_player, audio_favorite,
             audio_download, audio_action_status) = build_result_panel("上传音频后开始搜索。")

        with gr.Tab("文字搜索"):
            text_results = gr.State([])
            text_selected_id = gr.State(None)
            text_type = gr.Dropdown(["情绪", "发音风格", "内容／转录文本"], value="情绪", label="文字搜索类型")
            text_input = gr.Textbox(lines=4, label="文字描述", placeholder="例如：悲伤、失望，但不要愤怒")
            text_search_button = gr.Button("搜索匹配表达", variant="primary")
            (text_selector, text_explanation, text_details, text_player, text_favorite,
             text_download, text_action_status) = build_result_panel("输入文字后开始搜索。")

        with gr.Tab("收藏列表") as favorites_tab:
            favorite_results = gr.State([])
            favorite_selected_id = gr.State(None)
            (favorite_selector, favorite_explanation, favorite_details, favorite_player, favorite_toggle,
             favorite_download, favorite_action_status) = build_result_panel("切换到此标签时加载收藏。")

    audio_search_outputs = [audio_selector, audio_results, audio_explanation,
                            audio_player, audio_details, audio_selected_id, audio_download]
    text_search_outputs = [text_selector, text_results, text_explanation,
                           text_player, text_details, text_selected_id, text_download]
    audio_search_button.click(search_audio, [audio_input, audio_mode, result_limit, min_duration, max_duration, favorite_ids], audio_search_outputs)
    text_search_button.click(search_text, [text_input, text_type, result_limit, min_duration, max_duration, favorite_ids], text_search_outputs)
    favorites_tab.select(show_favorites, inputs=favorite_ids, outputs=[favorite_selector, favorite_results, favorite_explanation,
                                                  favorite_player, favorite_details, favorite_selected_id, favorite_download])

    wire_result_panel(audio_selector, audio_results, audio_selected_id, favorite_ids, audio_details, audio_player,
                      audio_favorite, audio_download, audio_action_status)
    wire_result_panel(text_selector, text_results, text_selected_id, favorite_ids, text_details, text_player,
                      text_favorite, text_download, text_action_status)
    wire_result_panel(favorite_selector, favorite_results, favorite_selected_id, favorite_ids, favorite_details,
                      favorite_player, favorite_toggle, favorite_download, favorite_action_status,
                      toggle_favorite_in_list)

    app.load(
        None,
        outputs=favorite_ids,
        js='() => localStorage.getItem("ref_audio_search_favorites") || "[]"',
    )
    favorite_ids.change(
        None,
        inputs=favorite_ids,
        js="""(value) => {
            try {
                const parsed = JSON.parse(value || "[]");
                localStorage.setItem(
                    "ref_audio_search_favorites",
                    JSON.stringify(Array.isArray(parsed) ? parsed : [])
                );
            } catch (_) {
                localStorage.setItem("ref_audio_search_favorites", "[]");
            }
        }""",
    )


if __name__ == "__main__":
    args = parse_args()
    # Gradio checks allowed paths when serving a selected file. Supplying every
    # one of 50k files makes each click expensive; a deduplicated directory list
    # keeps that check small while retaining the same indexed-library boundary.
    allowed_audio = sorted({str(Path(item["path"]).resolve().parent) for item in load_library()})
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=PORT,
        root_path=args.root_path.strip(),
        show_error=True,
        allowed_paths=allowed_audio,
    )
