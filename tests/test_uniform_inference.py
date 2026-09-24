"""CPU integration tests for the uniform-frame inference control flow."""

import unittest

import numpy as np

from video_qa.hermes_vqa import HermesVQA


class FakeQAModel:
    token_trace_enabled = False

    def __init__(self):
        self.clear_calls = 0
        self.init_calls = 0
        self.encoded_chunk_sizes = []

    def clear_cache(self):
        self.clear_calls += 1

    def encode_init_prompt(self):
        self.init_calls += 1

    def encode_video_chunk(self, chunk):
        self.encoded_chunk_sizes.append(len(chunk))

    def predict_and_compress(self):
        raise AssertionError("uniform inference must not run streaming compression")

    def get_prompt(self, prompt, mc=False):
        return prompt

    def question_answering(self, input_text, **kwargs):
        return "answer"


class UniformInferenceTest(unittest.TestCase):
    def test_resets_and_encodes_fixed_context_for_each_question(self):
        model = FakeQAModel()
        analyzer = HermesVQA(
            anno=[],
            save_dir=".",
            sample_fps=0.5,
            qa_model=model,
        )
        analyzer.frame_sampling = "uniform"
        analyzer.uniform_num_frames = 5
        analyzer.encode_chunk_size = 4
        analyzer.load_uniform_video = lambda *args, **kwargs: (
            np.zeros((5, 2, 2, 3), dtype=np.uint8),
            [0, 10, 20, 30, 40],
        )
        sample = {
            "video_id": "video-1",
            "video_path": "video-1.mp4",
            "duration": 30.0,
            "conversations": [
                {"question": "first?", "answer": "one", "end_time": 10.0},
                {"question": "second?", "answer": "two", "end_time": 20.0},
            ],
        }

        analyzer.analyze_a_video(sample)

        self.assertEqual(model.clear_calls, 2)
        self.assertEqual(model.init_calls, 2)
        self.assertEqual(model.encoded_chunk_sizes, [4, 1, 4, 1])
        self.assertEqual(len(analyzer.record), 2)
        self.assertTrue(
            all(row["frame_sampling"] == "uniform" for row in analyzer.record)
        )
        self.assertTrue(all(row["num_input_frames"] == 5 for row in analyzer.record))


if __name__ == "__main__":
    unittest.main()
