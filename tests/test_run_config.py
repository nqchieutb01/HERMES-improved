"""CPU-only tests for the Hydra workflow runner."""

from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.run import (
    evaluation_commands,
    inference_command,
    merge_chunks,
    validate_config,
)
from video_qa.run_infer import translate_legacy_args


ROOT = Path(__file__).resolve().parents[1]


def compose_config(*overrides):
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"), version_base="1.3"
    ):
        cfg = compose(config_name="config", overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


class HydraConfigTest(unittest.TestCase):
    def test_default_config_is_valid(self):
        cfg = compose_config()
        validate_config(cfg)
        self.assertEqual(cfg.model.name, "llava_ov_7b")
        self.assertEqual(cfg.dataset.name, "streamingbench")

    def test_smoke_experiment_composes_model_dataset_and_parameters(self):
        cfg = compose_config("experiment=streamingbench_smoke")
        validate_config(cfg)
        self.assertEqual(cfg.model.name, "llava_ov_0.5b")
        self.assertEqual(cfg.dataset.name, "streamingbench_subset")
        self.assertEqual(cfg.run.kv_size, 1024)
        self.assertIn("smoke-fps0.5-kv1024", cfg.paths.save_dir)

    def test_worker_command_contains_composed_overrides(self):
        cfg = compose_config(
            "model=llava_ov_0.5b",
            "dataset=streamingbench_subset",
            "run.min_tokens_per_frame=4",
        )
        command = inference_command(
            cfg,
            python="python",
            annotation=ROOT / cfg.dataset.annotation,
            save_dir=ROOT / "outputs/test",
            chunk_index=0,
        )
        self.assertEqual(
            command[:4], ["python", "-u", "-m", "video_qa.hermes_vqa"]
        )
        self.assertEqual(command[command.index("--model") + 1], "llava_ov_0.5b")
        self.assertEqual(
            command[command.index("--min_tokens_per_frame") + 1], "4"
        )

    def test_all_experiment_profiles_compose(self):
        profiles = [path.stem for path in (ROOT / "configs/experiment").glob("*.yaml")]
        for profile in profiles:
            with self.subTest(profile=profile):
                validate_config(compose_config(f"experiment={profile}"))

    def test_sember_smoke_uses_direct_adapter_and_small_subset(self):
        cfg = compose_config("experiment=sember_smoke")
        validate_config(cfg)
        self.assertEqual(cfg.model.name, "llava_ov_0.5b")
        self.assertEqual(cfg.dataset.adapter, "sember_grounding")
        self.assertEqual(cfg.dataset.max_videos, 1)
        self.assertFalse(cfg.run.use_history)
        command = inference_command(
            cfg,
            python="python",
            annotation=Path(cfg.dataset.annotation),
            save_dir=ROOT / "outputs/sember-test",
            chunk_index=0,
        )
        self.assertEqual(
            command[command.index("--dataset_adapter") + 1], "sember_grounding"
        )
        self.assertEqual(command[command.index("--max_videos") + 1], "1")
        self.assertEqual(
            command[command.index("--video_root") + 1], cfg.dataset.video_root
        )

    def test_sember_dataset_selects_grounding_evaluator(self):
        cfg = compose_config("dataset=sember_grounding", "run.mode=evaluate")
        commands = evaluation_commands(
            cfg,
            python="python",
            results_path=ROOT / "results.csv",
            save_dir=ROOT / "results",
            annotation=Path(cfg.dataset.annotation),
        )
        self.assertEqual(len(commands), 1)
        self.assertIn("eval/sember/eval_grounding.py", commands[0])

    def test_sember_mcq_smoke_uses_direct_adapter_and_evaluator(self):
        cfg = compose_config("experiment=sember_mcq_smoke")
        validate_config(cfg)
        self.assertEqual(cfg.model.name, "llava_ov_0.5b")
        self.assertEqual(cfg.dataset.adapter, "sember_mcq")
        self.assertEqual(cfg.dataset.max_videos, 1)
        command = inference_command(
            cfg,
            python="python",
            annotation=Path(cfg.dataset.annotation),
            save_dir=ROOT / "outputs/sember-mcq-test",
            chunk_index=0,
        )
        self.assertEqual(command[command.index("--dataset_adapter") + 1], "sember_mcq")
        commands = evaluation_commands(
            cfg,
            python="python",
            results_path=ROOT / "results.csv",
            save_dir=ROOT / "results",
            annotation=Path(cfg.dataset.annotation),
        )
        self.assertEqual(len(commands), 1)
        self.assertIn("eval/sember/eval_mcq.py", commands[0])

    def test_sember_mcq_time_count_location_profile_filters_tasks(self):
        cfg = compose_config("experiment=sember_mcq_time_count_location_smoke")
        validate_config(cfg)
        expected_categories = [
            "time_duration",
            "counting_objects_events",
            "location_trace",
        ]
        self.assertEqual(list(cfg.dataset.question_categories), expected_categories)
        self.assertEqual(cfg.dataset.max_videos, 12)
        command = inference_command(
            cfg,
            python="python",
            annotation=Path(cfg.dataset.annotation),
            save_dir=ROOT / "outputs/sember-tcl-test",
            chunk_index=0,
        )
        start = command.index("--question_categories") + 1
        self.assertEqual(command[start:start + 3], expected_categories)

    def test_dataset_profile_selects_its_evaluator(self):
        cfg = compose_config("dataset=videomme", "run.mode=evaluate")
        commands = evaluation_commands(
            cfg,
            python="python",
            results_path=ROOT / "results.csv",
            save_dir=ROOT / "results",
            annotation=ROOT / cfg.dataset.annotation,
        )
        self.assertEqual([command[2] for command in commands], ["general", "videomme"])

    def test_legacy_cli_is_translated_to_hydra_overrides(self):
        self.assertEqual(
            translate_legacy_args(
                [
                    "--model",
                    "llava_ov_0.5b",
                    "--dataset",
                    "streamingbench",
                    "--sample_fps",
                    "0.5",
                    "--only_eval",
                ]
            ),
            [
                "model=llava_ov_0.5b",
                "dataset=streamingbench",
                "run.sample_fps=0.5",
                "run.mode=evaluate",
            ],
        )


class ChunkMergeTest(unittest.TestCase):
    def test_merge_chunks_writes_one_header_and_all_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "2_0.csv").write_text("video_id,pred_answer\na,yes\n")
            (root / "2_1.csv").write_text("video_id,pred_answer\nb,no\n")
            output = root / "results.csv"
            merge_chunks(root, output, 2, keep_chunks=False)
            self.assertEqual(
                output.read_text().splitlines(),
                ["video_id,pred_answer", "a,yes", "b,no"],
            )
            self.assertFalse((root / "2_0.csv").exists())
            self.assertFalse((root / "2_1.csv").exists())


if __name__ == "__main__":
    unittest.main()
