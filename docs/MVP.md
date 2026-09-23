# 参考音频搜索 MVP

面向单说话人录音数据集的参考音频搜索网站。管理员通过命令行扫描服务器数据集、提取 IndexTTS2 声学与情绪特征并保存转录文本；公开网站只提供音频与文字检索、筛选、试听、收藏、反馈和下载。

> IndexTTS2 是运行必需项。Gradio 搜索服务直接运行在项目虚拟环境中；Node 建库命令仍通过 `MODEL_PYTHON` 指定该环境。模型不可用时不会退回 `prosody-v1`。

## 启用真实模型

应用已经接入 IndexTTS2 `UnifiedVoice.get_conditioning()`、`get_emovec()` 和 QwenEmotion。IndexTTS2 源码位于项目内的 `index-tts/`，模型位于 `.models/IndexTTS-2`。

IndexTTS2 要求 Python 3.10 或 3.11。首次配置时，在项目根目录创建标准 Python 虚拟环境并安装统一依赖：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

安装依赖后直接启动 Gradio 单进程服务：

```powershell
npm start
```

Gradio 与 IndexTTS2 在同一个 Python 进程中运行，不再需要为网页服务配置 `MODEL_PYTHON`。程序会自动使用项目内的 `index-tts/` 源码和 `.models/IndexTTS-2/` 模型。建库命令仍由 Node CLI 执行，并通过 `MODEL_PYTHON` 调用同一个 Python 环境。

模型权重不会随源码自动复制。若 `.models/IndexTTS-2/config.yaml` 等模型文件不存在，先在项目根目录执行：

```powershell
.\.venv\Scripts\indextts2.exe download --source huggingface --model-dir .\.models\IndexTTS-2
```

也可以把已有的完整模型目录放到 `.models/IndexTTS-2`，或在运行前用 `INDEXTTS_MODEL_DIR` 指向它。建库会在扫描音频前校验模型；模型未就绪时立即退出，不会为每个音频重复报告同一错误。

当前验证环境为 Python 3.11.16、PyTorch 2.8.0+cu128 和 RTX 4060 Laptop GPU。`indextts2 check` 已确认模型文件、Python 依赖和 CUDA 均可用。

音频搜索可以选择“仅发音风格”“仅情绪”或二者混合。模型启动或提取失败会让建库任务明确失败，不会静默使用基础特征代替。

## 环境

- Python 3.10 或 3.11（Gradio Web 服务与 IndexTTS2）
- Node.js 22.5+（仅建库 CLI 与旧版回退服务）
- FFmpeg/FFprobe 已加入 PATH

不需要安装 npm 依赖。

## 建立索引

```powershell
$env:MODEL_PYTHON=".\.venv\Scripts\python.exe"
npm run index -- .\test\f5-tts-demo
```

索引保存在 `.data/audio-search.db`。源音频不会被复制、修改或删除。数据集根目录存在 `metadata.csv` 时，按两列 CSV 读取“文件名（可带后缀）,转录文本”，直接写入音频索引；文件名与音频按基本文件名匹配。没有 `metadata.csv`，或某条音频找不到对应行时，使用 faster-whisper `large-v3-turbo` 自动转录。可用 `WHISPER_DEVICE`、`WHISPER_COMPUTE_TYPE` 和 `WHISPER_MODEL` 覆盖。首次自动转录会从 Hugging Face 获取模型权重，需要网络；可将 `WHISPER_MODEL` 设为本地模型目录以离线运行。再次建库时，音频和转录均未变化的条目会跳过；只改动 metadata 文本时复用已提取的音频特征。

文字搜索必须在页面上明确选择类型：情绪使用 IndexTTS2 QwenEmotion；发音风格使用“语速、停顿、音量、表现力”等提示词对应的韵律特征，不会把文字映射到 condition embedding；内容使用转录文本的字符片段索引，忽略标点与空格，完整匹配优先。三类搜索互不混用。

## 启动

```powershell
npm start
```

浏览器打开 <http://127.0.0.1:7860>。

当前网页使用 Gradio Blocks：搜索结果显示为表格，点击任意结果行后可在下方试听、下载、收藏或提交相似性反馈。旧版 Node/HTML 服务保留为 `npm run start:legacy`，仅用于迁移期回退。

索引操作只允许管理员在服务器命令行执行，网站不提供建库接口，也不会接受服务器目录路径。

系统自动把以下本地行为写入 `user_events`：

- 每次搜索的查询、完整结果列表和各路得分；
- 每个候选的曝光位置；
- 点击播放、25%/50%/75% 播放进度、暂停位置和完整播放；
- 下载、收藏、取消收藏；
- 用户明确选择的“相似”和“不相似”。

事件包含会话、查询、候选、位置、搜索模式和时间，可通过页面右上角“导出偏好数据”下载 JSONL。空结果搜索同样会记录。收藏以本机 SQLite 为持久化主记录，同时在当前浏览器的 `localStorage.refAudioFavorites` 保存音频 ID 镜像；收藏与取消收藏成功后会同步更新两处。搜索音频及偏好事件只写入本机 SQLite，不会由本应用发送到外部服务。首次启动模型时，若 IndexTTS2 辅助权重缺失，上游下载模块会访问 Hugging Face 补齐权重；模型文件齐全时无需此下载。

音频特征搜索采用内存精确扫描，不依赖近似向量索引；内容搜索先用 SQLite 字符二元组索引缩小候选，再核对转录文本。

可使用环境变量改变端口和数据库位置：

```powershell
$env:PORT=8080
$env:DB_PATH="D:\audio-index\search.db"
npm start
```

## API

- `GET /api/health`：索引状态
- `POST /api/search/audio`：请求体为音频二进制，`X-Filename` 提供文件名
- `POST /api/search/text`：`{"type":"emotion","text":"悲伤"}`、`{"type":"style","text":"轻声、快语速"}` 或 `{"type":"content","text":"今天的天气"}`
- `GET /api/library?favorites=1`：收藏列表
- `PUT /api/favorites/:id`：更新收藏
- `POST /api/events`：记录搜索偏好事件
- `GET /api/events/export`：导出偏好事件 JSONL
- `GET /api/audio/:id`：支持 Range 的试听流
- `GET /api/audio/:id/download`：下载源音频

## 测试

```powershell
npm test
```

### F5-TTS demo 检索评测

使用独立数据库建立测试索引，不污染正式数据：

```powershell
$env:DB_PATH="$PWD\.data\f5-baseline.db"
npm run index -- test\f5-tts-demo
npm run eval:f5
```

评测采用 leave-one-out，并报告 Recall@1、Recall@3、Recall@5 和 MRR：

- `emotion-label`：同情绪标签召回；
- `robustness-ref-gen-pair`：同编号 ref/gen 成对召回；
- `speed-factor-cross-content`：排除相同文本后召回相同速度倍率；
- `zero-shot-reference-group`：召回相同参考组。

正式搜索要求索引含真实模型特征：情绪任务使用 IndexTTS2 emovec，风格任务使用 IndexTTS2 condition。文字情绪搜索若包含明确的语速、停顿、音量或表现力描述，还会将相应韵律特征与情绪得分融合。下表的 `prosody` 基线仅供离线评测对比，正式搜索不使用该回退。完整错误案例保存到 `.data/f5-eval-report.json`。

### 本机实际结果（111 条）

| 任务 | prosody 基线 R@1 / R@3 | IndexTTS2 R@1 / R@3 |
|---|---:|---:|
| emotion-label | 0.0% / 16.7% | **88.9% / 100.0%** |
| robustness-ref-gen-pair | 3.1% / 9.4% | **100.0% / 100.0%** |
| speed-factor-cross-content | 60.0% / 86.7% | 60.0% / **93.3%** |
| zero-shot-reference-group | 36.4% / 63.6% | **100.0% / 100.0%** |

真实索引保存在 `.data/f5-indextts2.db`。再次索引已验证能够跳过全部 111 个未变化文件并正常退出。中文情绪描述“非常愤怒，但压抑克制，不要喊叫”的 QwenEmotion 实测分布以 angry=0.85 为主，并成功投影到与音频 emovec 相同的检索空间。

## 下一步模型接口

索引中的 `features_json` 允许并存多个版本：

```json
{
  "prosody": { "vector": [] },
  "emotion": { "embedding": [], "distribution": {} },
  "style": { "embedding": [] }
}
```

当前实测维度为：style condition mean/std pooling 2560 维、emotion 1280 维、prosody 7 维。

模型升级时写入新的 `feature_version` 并重建索引，避免不同版本的向量混用。
