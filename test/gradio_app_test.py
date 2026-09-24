import unittest

from gradio_app import content_search, model_search, parse_args, result_choices, result_selector, style_search, toggle_favorite


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
    def test_root_path_command_line_argument(self):
        self.assertEqual(parse_args(["--root-path", "/audio-search"]).root_path, "/audio-search")

    def test_content_search_ignores_punctuation(self):
        items = [item(1, "今天，天气真好！", [.84, .22, .42, .04, .035, .05, 6])]
        self.assertEqual(content_search(items, "天气 真好", 20, 0, 0)[0]["id"], 1)

    def test_style_results_render_as_selectable_choices(self):
        items = [item(1, "", [.72, .34, .60, .04, .035, .05, 7.5])]
        results, attributes = style_search(items, "缓慢、克制", 20, 0, 0)
        self.assertIn("慢速", attributes)
        self.assertIn("1.wav", result_choices(results)[0])

    def test_toggling_favorite_preserves_selected_result(self):
        results = [item(1, "", [.84, .22, .42, .04, .035, .05, 6])]
        selector, updated, favorite_ids, _ = toggle_favorite(1, results, "[]")
        self.assertEqual(selector.value, result_choices(updated)[0])
        self.assertEqual(favorite_ids, "[1]")

    def test_result_selector_can_restore_selection_by_id(self):
        results = [item(1, "", [.84, .22, .42, .04, .035, .05, 6])]
        self.assertEqual(result_selector(results, selected_id=1).value, result_choices(results)[0])

    def test_opposite_style_descriptions_produce_opposite_rankings(self):
        quiet = item(1, "", [.84, .22, .42, .020, .020, .05, 6])
        forceful = item(2, "", [.84, .22, .42, .070, .060, .05, 6])
        self.assertEqual(style_search([quiet, forceful], "轻声", 10, 0, 0)[0][0]["id"], 1)
        self.assertEqual(style_search([quiet, forceful], "有力", 10, 0, 0)[0][0]["id"], 2)

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
