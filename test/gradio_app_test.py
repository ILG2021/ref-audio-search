import unittest

from gradio_app import content_search, model_search, style_search, table_rows


def item(item_id, transcript, vector):
    return {
        "id": item_id,
        "name": f"{item_id}.wav",
        "path": f"{item_id}.wav",
        "duration": 2.0,
        "transcript": transcript,
        "favorite": False,
        "features": {"prosody": {"vector": vector}},
    }


class GradioSearchTests(unittest.TestCase):
    def test_content_search_ignores_punctuation(self):
        items = [item(1, "今天，天气真好！", [.84, .22, .42, .04, .035, .05, 6])]
        self.assertEqual(content_search(items, "天气 真好", 20, 0, 0)[0]["id"], 1)

    def test_style_results_render_as_table(self):
        items = [item(1, "", [.72, .34, .60, .04, .035, .05, 7.5])]
        results, attributes = style_search(items, "缓慢、克制", 20, 0, 0)
        self.assertIn("慢速", attributes)
        self.assertEqual(table_rows(results)[0][1], 1)

    def test_model_search_ranks_the_entire_library_before_limiting(self):
        items = []
        for index in range(150):
            candidate = item(index, "", [.84, .22, .42, .04, .035, .05, 6])
            candidate["features"]["style"] = {"embedding": [0, 1]}
            candidate["features"]["emotion"] = {"embedding": [0, 1]}
            items.append(candidate)
        items[-1]["features"]["emotion"] = {"embedding": [1, 0]}

        results = model_search(items, {"emotion": [1, 0]}, "emotion", 1, 0, 0)
        self.assertEqual(results[0]["id"], 149)


if __name__ == "__main__":
    unittest.main()
