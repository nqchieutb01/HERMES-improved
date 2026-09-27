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


class SampleScheduleTest(unittest.TestCase):
    def test_parses_stages_with_open_last_stage(self):
        from video_qa.sampling import parse_sample_schedule

        self.assertEqual(
            parse_sample_schedule("1.0:60,0.5:160,0.2"),
            [(1.0, 60.0), (0.5, 160.0), (0.2, float("inf"))],
        )
        with self.assertRaises(ValueError):
            parse_sample_schedule("1.0,0.2")  # only the last stage may omit its end
        with self.assertRaises(ValueError):
            parse_sample_schedule("1.0:60,0.5:30,0.2")  # end times must increase

    def test_single_rate_matches_fixed_fps(self):
        from video_qa.sampling import parse_sample_schedule, schedule_frame_times

        times = schedule_frame_times(parse_sample_schedule("0.2"), 600)
        self.assertEqual(times, [5.0 * i for i in range(120)])

    def test_rate_changes_at_stage_boundaries(self):
        from video_qa.sampling import parse_sample_schedule, schedule_frame_times

        times = schedule_frame_times(parse_sample_schedule("1.0:60,0.5:160,0.2"), 300)
        self.assertEqual(times[58:63], [58.0, 59.0, 60.0, 62.0, 64.0])
        self.assertEqual(len([t for t in times if t < 300]), 60 + 50 + 28)


class GridPlanTest(unittest.TestCase):
    def test_spacing_doubles_up_to_cap(self):
        from video_qa.sampling import GridPlan

        plan = GridPlan.parse("grid:32:8")
        self.assertEqual([plan.spacing(t) for t in (10, 64, 65, 129, 1000)], [2, 2, 4, 8, 8])

    def test_grids_are_nested_so_thinning_never_needs_evicted_frames(self):
        from video_qa.sampling import GridPlan

        plan = GridPlan.parse("grid:32:16")
        times, anchors = plan.frame_times(600)
        for now in (60, 160, 320, 600):
            kept = {a for t, a in zip(times, anchors) if t < now and plan.on_grid(a, now)}
            later = {a for t, a in zip(times, anchors) if t < now and plan.on_grid(a, now + 300)}
            self.assertLessEqual(later, kept)

    def test_pairs_share_an_anchor_and_spread_pairs_split_the_cell(self):
        from video_qa.sampling import GridPlan

        times, anchors = GridPlan.parse("grid:32:8").frame_times(600)
        self.assertEqual(times[:4], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(anchors[:4], [0.0, 0.0, 2.0, 2.0])
        spread_times, _ = GridPlan.parse("grid:32:8:spread").frame_times(600)
        self.assertIn(252.0, spread_times)
        self.assertIn(256.0, spread_times)
        with self.assertRaises(ValueError):
            GridPlan.parse("grid:32:8:wide")


if __name__ == "__main__":
    unittest.main()
