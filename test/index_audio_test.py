import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from index_audio import build_index, connect, extract_prosody, load_metadata, metadata_text, upsert_audio


class IndexAudioTests(unittest.TestCase):
    def test_metadata_csv_supports_quotes_pipe_and_extensionless_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.csv").write_text('b.wav,"你好，世界"\nc,今天天气好\n', encoding="utf-8")
            metadata = load_metadata(root)
            self.assertEqual(metadata_text(metadata, root / "wavs" / "b.wav"), "你好，世界")
            self.assertEqual(metadata_text(metadata, root / "wavs" / "c.wav"), "今天天气好")
            (root / "metadata.csv").write_text("d|这是台词|normalized text\n", encoding="utf-8")
            self.assertEqual(metadata_text(load_metadata(root), root / "d.wav"), "这是台词")

    def test_prosody_is_finite_and_stable(self):
        samples = np.sin(np.arange(32000) / 15) * np.where(np.arange(32000) < 16000, .2, .05)
        result = extract_prosody(samples)
        self.assertEqual(len(result["vector"]), 7)
        self.assertTrue(np.isfinite(result["vector"]).all())
        self.assertEqual(result["duration"], 2)

    def test_transcript_bigrams_are_replaced_on_update(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "test.db"
            connection = connect(database)
            item = {
                "path": str(Path(directory) / "sample.wav"), "name": "sample.wav", "size": 10,
                "mtime_ms": 1, "duration": 1, "features": {"prosody": {"vector": []}},
                "transcript": "今天，天气真好！", "transcript_source": "metadata",
            }
            upsert_audio(connection, item)
            self.assertTrue(connection.execute("SELECT 1 FROM content_grams WHERE gram='天气'").fetchone())
            item["transcript"] = "明天下雨"
            upsert_audio(connection, item)
            self.assertFalse(connection.execute("SELECT 1 FROM content_grams WHERE gram='天气'").fetchone())
            self.assertTrue(connection.execute("SELECT 1 FROM content_grams WHERE gram='下雨'").fetchone())
            connection.close()

    def test_build_index_then_skip_unchanged_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            samples = (np.sin(np.arange(16000) / 15) * 10000).astype("<i2")
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(samples.tobytes())
            (root / "metadata.csv").write_text("sample.wav,测试文本\n", encoding="utf-8")
            database = root / "index.db"

            class FakeModels:
                def extract(self, _):
                    return {"style": [1.0, 0.0], "emotion": [0.0, 1.0]}

                def transcribe(self, _):
                    raise AssertionError("metadata transcript should be used")

            with patch("index_audio.Models", FakeModels):
                first = build_index(root, database)
            self.assertEqual(first["indexed"], 1)
            with patch("index_audio.Models", side_effect=AssertionError("unchanged index must not load models")):
                second = build_index(root, database)
            self.assertEqual(second["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
