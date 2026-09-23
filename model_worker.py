"""Persistent JSONL worker using only IndexTTS2 style and emotion features."""
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


class Models:
    def __init__(self):
        project_root = Path(__file__).resolve().parent
        index_root = project_root / "index-tts"
        sys.path.insert(0, str(index_root))
        from indextts.infer_v2 import IndexTTS2
        import torch
        import torchaudio

        self.torch = torch
        self.torchaudio = torchaudio
        model_dir = Path(os.environ.get("INDEXTTS_MODEL_DIR", project_root / ".models" / "IndexTTS-2"))
        config_path = model_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"IndexTTS2 模型不完整，缺少 {config_path}。"
                "请运行 .\\.venv\\Scripts\\indextts2.exe download --source huggingface "
                "--model-dir .\\.models\\IndexTTS-2，或设置 INDEXTTS_MODEL_DIR 指向已有模型目录。"
            )
        self.tts = IndexTTS2(
            cfg_path=str(config_path),
            model_dir=str(model_dir),
            use_fp16=torch.cuda.is_available(),
            use_cuda_kernel=False,
            retrieval_only=True,
        )
        self.whisper = None

    def transcribe(self, audio_path):
        if self.whisper is None:
            from faster_whisper import WhisperModel
            model_name = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
            device = os.environ.get("WHISPER_DEVICE", "cuda")
            compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "float16")
            self.whisper = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments, _ = self.whisper.transcribe(audio_path, beam_size=5, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def semantic_features(self, audio_path):
        audio, sr = self.tts._load_and_cut_audio(audio_path, 15, False)
        audio_16k = self.torchaudio.transforms.Resample(sr, 16000)(audio)
        inputs = self.tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        features = inputs["input_features"].to(self.tts.device)
        mask = inputs["attention_mask"].to(self.tts.device)
        semantic = self.tts.get_emb(features, mask)
        # The semantic encoder stays in fp32 while IndexTTS2's GPT is loaded
        # in fp16 on CUDA. Match the conditioning encoders' parameter dtype.
        semantic = semantic.to(dtype=next(self.tts.gpt.parameters()).dtype)
        lengths = self.torch.tensor([semantic.shape[1]], device=semantic.device)
        return semantic, lengths

    def extract(self, audio_path):
        with self.torch.inference_mode():
            semantic, lengths = self.semantic_features(audio_path)

            # Speaker/style conditioning used by IndexTTS2 generation.
            # UnifiedVoice expects channel-first semantic features here; the
            # method transposes them back internally for its conformer.
            tokens = self.tts.gpt.get_conditioning(semantic.transpose(1, 2), lengths)
            tokens_np = tokens[0].float().cpu().numpy()
            style = normalized(np.concatenate([tokens_np.mean(axis=0), tokens_np.std(axis=0)]))

            # Emotion conditioning used by IndexTTS2 generation. This replaces
            # the previous external emotion2vec dependency entirely.
            emotion = self.tts.gpt.get_emovec(semantic, lengths)
            emotion = normalized(emotion[0].float().cpu().numpy())

        return {
            "style": style.tolist(),
            "emotion": emotion.tolist(),
            "modelVersion": "indextts2-condition+emovec-v1",
        }

    def text_emotion(self, text):
        # QwenEmotion 2.0.0 batches the rendered prompt as ``[text]``. With
        # tokenizers 0.21 this can reject some non-ASCII prompts, while the
        # equivalent single-input path is stable. Keep the official prompt,
        # generation and conversion semantics, but tokenize one string.
        qwen = self.tts.qwen_emo
        text = str(text)
        messages = [
            {"role": "system", "content": qwen.prompt},
            {"role": "user", "content": text},
        ]
        rendered = qwen.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = qwen.tokenizer(str(rendered), return_tensors="pt").to(qwen.model.device)
        generated_ids = qwen.model.generate(
            **model_inputs,
            max_new_tokens=512,
            pad_token_id=qwen.tokenizer.eos_token_id,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            think_end = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            think_end = 0
        content = qwen.tokenizer.decode(output_ids[think_end:], skip_special_tokens=True)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {
                match.group(1): float(match.group(2))
                for match in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }
        lowered = text.lower()
        if any(word in lowered for word in qwen.melancholic_words):
            parsed["悲伤"], parsed["低落"] = parsed.get("低落", 0.0), parsed.get("悲伤", 0.0)
        distribution = dict(qwen.convert(parsed))
        weights = np.asarray(list(distribution.values()), dtype=np.float32)
        prototypes = []
        for group in self.tts.emo_matrix:
            prototypes.append(group.float().mean(dim=0).cpu().numpy())
        prototype_matrix = np.stack(prototypes).astype(np.float32)
        if len(weights) != prototype_matrix.shape[0]:
            raise RuntimeError(
                f"QwenEmotion returned {len(weights)} dimensions, "
                f"but IndexTTS2 has {prototype_matrix.shape[0]} emotion groups"
            )
        emotion = normalized(weights @ prototype_matrix)
        return {"distribution": distribution, "emotion": emotion.tolist()}


def emit(value):
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)


def main():
    try:
        models = Models()
        emit({"ready": True})
    except Exception as exc:
        emit({"ready": False, "error": str(exc), "trace": traceback.format_exc()})
        raise SystemExit(1)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("action") == "text_emotion":
                result = models.text_emotion(request["text"])
            elif request.get("action") == "transcribe":
                result = models.transcribe(request["path"])
            else:
                result = models.extract(request["path"])
            emit({"id": request["id"], "result": result})
        except Exception as exc:
            emit({"id": request.get("id"), "error": str(exc), "trace": traceback.format_exc()})


if __name__ == "__main__":
    main()
