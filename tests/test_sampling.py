"""Unit tests for fixed-count video frame sampling."""

import unittest

from video_qa.sampling import frame_end_exclusive, uniform_frame_indices


class UniformFrameSamplingTest(unittest.TestCase):
    def test_selects_fixed_count_including_zero_and_question_time(self):
        end = frame_end_exclusive(total_frames=301, fps=30.0, end_time=10.0)

        self.assertEqual(end, 301)
        self.assertEqual(
            uniform_frame_indices(
                total_frames=301,
                num_frames=5,
                end_frame_exclusive=end,
            ),
            [0, 75, 150, 225, 300],
        )

    def test_caps_at_available_frames_without_duplicates(self):
        self.assertEqual(
            uniform_frame_indices(
                total_frames=4,
                num_frames=8,
                end_frame_exclusive=3,
            ),
            [0, 1, 2],
        )

    def test_question_time_is_clamped_to_video_length(self):
        self.assertEqual(
            frame_end_exclusive(total_frames=90, fps=30.0, end_time=99.0),
            90,
        )

    def test_rejects_invalid_frame_count(self):
        with self.assertRaisesRegex(ValueError, "num_frames must be positive"):
            uniform_frame_indices(total_frames=10, num_frames=0)


if __name__ == "__main__":
    unittest.main()
