"""Compatibility entry point for the Hydra runner in :mod:`scripts.run`.

New commands should use ``python scripts/run.py``. The former argparse options
are translated to Hydra overrides here so existing automation keeps working.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run import main  # noqa: E402


def translate_legacy_args(arguments: list[str]) -> list[str]:
    value_options = {
        "--model": "model",
        "--dataset": "dataset",
        "--num_chunks": "run.num_chunks",
        "--sample_fps": "run.sample_fps",
        "--debug": "run.debug",
        "--kv_size": "run.kv_size",
    }
    translated = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--only_eval":
            translated.append("run.mode=evaluate")
            index += 1
        elif argument in value_options:
            if index + 1 >= len(arguments):
                raise SystemExit(f"Expected a value after {argument}")
            translated.append(f"{value_options[argument]}={arguments[index + 1]}")
            index += 2
        else:
            translated.append(argument)
            index += 1
    return translated


if __name__ == "__main__":
    sys.argv[1:] = translate_legacy_args(sys.argv[1:])
    main()
