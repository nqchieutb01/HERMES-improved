<h1 align="center">
  <img src="./asset/logo.png" width="40" alt="logo"> HERMES
</h1>
<p align="center">
  <b>[ACL 2026] KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding</b>
</p>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://hermes-streaming.github.io/)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2601.14724)
[![HF Paper](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/papers/2601.14724)

</div>

## 🔥 News
- **[2026.04.06]** HERMES is accepted to ACL 2026 Main 🎉
- **[2026.03.23]** Full code released!
- **[2025.01.23]** HERMES reached **#3 Paper of the day** on [Hugging Face Daily Papers](https://huggingface.co/papers/2601.14724)!
- **[2025.01.21]** HERMES is available on [arXiv](https://arxiv.org/abs/2601.14724).


## 🛠️ Installation

For **LLaVA** model inference:
```bash
conda create -n hermes-llava python=3.12 -y
conda activate hermes-llava
pip install -r requirements_llava.txt
pip install flash-attn --no-build-isolation
```

For **Qwen2.5-VL** model inference:
```bash
conda create -n hermes-qwen python=3.12 -y
conda activate hermes-qwen
pip install -r requirements_qwen.txt
pip install flash-attn --no-build-isolation
```


## 📦 Preparation

### Model Preparation

Create a `models` directory and download the model weights from HuggingFace:

```bash
mkdir models
```

We support the following models (choose one or more):

| Model Family | Model | HuggingFace Link |
|:---:|:---:|:---:|
| LLaVA-OneVision | llava-onevision-qwen2-0.5b-ov-hf | [llava-hf/llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) |
| LLaVA-OneVision | llava-onevision-qwen2-7b-ov-hf | [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) |
| LLaVA-OneVision | llava-onevision-qwen2-72b-ov-hf | [llava-hf/llava-onevision-qwen2-72b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf) |
| Qwen2.5-VL | Qwen2.5-VL-3B-Instruct | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| Qwen2.5-VL | Qwen2.5-VL-7B-Instruct | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Qwen2.5-VL | Qwen2.5-VL-32B-Instruct | [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) |


### Data Preparation

Download the benchmark videos from their official sources and place them according to the paths specified in the annotation files:

**Streaming Benchmarks:**

| Benchmark | Video Path | Official Source |
|:---:|:---:|:---:|
| StreamingBench | `/data/streamingbench/videos/` | 🤗 [StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench) |
| OVO-Bench | `/data/ovobench/videos/` | 🤗 [OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench) |
| RVS-Ego | `/data/rvs/ego/videos/` | 🤗 [RVS](https://huggingface.co/datasets/Becomebright/RVS) |
| RVS-Movie | `/data/rvs/movie/videos/` | 🤗 [RVS](https://huggingface.co/datasets/Becomebright/RVS) |

**Offline Benchmarks:**

| Benchmark | Video Path | Official Source |
|:---:|:---:|:---:|
| VideoMME | `/data/videomme/videos/` | 🤗 [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME) |
| MVBench | `/data/mvbench/videos/` | 🤗 [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) |
| EgoSchema | `/data/egoschema/videos/` | 🤗 [EgoSchema](https://huggingface.co/datasets/lmms-lab/egoschema) |

The annotation JSON files contain the same information as officially provided, with formatting adjustments to adapt to our codebase.

After preparation, the project structure should look like this:

```
HERMES/
├── asset/
│   └── logo.png
├── data/
│   ├── egoschema/
│   │   ├── videos/
│   │   └── egoschema.json
│   ├── mvbench/
│   │   ├── videos/
│   │   └── mvbench.json
│   ├── ovobench/
│   │   ├── videos/
│   │   └── ovobench_realtime_backeward.json
│   ├── rvs/
│   │   ├── ego/
│   │   │   ├── videos/
│   │   │   └── ego4d_oe.json
│   │   └── movie/
│   │       ├── videos/
│   │       └── movienet_oe.json
│   ├── streamingbench/
│   │   ├── videos/
│   │   └── streamingbench_realtime.json
│   └── videomme/
│       ├── videos/
│       └── videomme.json
├── configs/
│   ├── dataset/
│   ├── experiment/
│   └── model/
├── eval/
│   ├── rvs/
│   ├── eval_multiple_choice.py
│   └── eval_open_ended.py
├── inference/
│   ├── abstract_hermes.py
│   ├── llavaov_hermes.py
│   ├── qwenvl_hermes.py
│   ├── reindex_1d.py
│   └── reindex_3d.py
├── models/
│   ├── llava-onevision-qwen2-0.5b-ov-hf/
│   ├── llava-onevision-qwen2-7b-ov-hf/
│   ├── llava-onevision-qwen2-72b-ov-hf/
│   ├── Qwen2.5-VL-3B-Instruct/
│   ├── Qwen2.5-VL-7B-Instruct/
│   └── Qwen2.5-VL-32B-Instruct/
├── scripts/
│   ├── data/
│   ├── slurm/
│   ├── sweeps/
│   └── run.py
├── tests/
├── video_qa/
│   ├── base.py
│   ├── hermes_vqa.py
│   └── run_infer.py
├── LICENSE
├── README.md
├── requirements_llava.txt
└── requirements_qwen.txt
```


## 🚀 Inference and evaluation with Hydra

All public run configuration is under `configs/` and composed by
`scripts/run.py`. Model, dataset, and experiment settings are separate config
groups, so a run and its evaluation use one resolved configuration.

Run StreamingBench inference and its configured evaluation:

```bash
python scripts/run.py \
    model=llava_ov_7b dataset=streamingbench \
    run.num_chunks=8 run.sample_fps=0.5 run.kv_size=6000
```

Hydra overrides replace the old argparse flags and launcher-specific environment
variables. Useful settings include `run.mode`, `run.num_chunks`,
`run.sample_fps`, `run.kv_size`, `run.min_tokens_per_frame`,
`run.encode_chunk_size`, `run.debug`, `paths.save_dir`, and
`paths.results_path`. Run `python scripts/run.py --cfg job --resolve` to inspect
the complete configuration without loading a model.

The available modes are:

| Mode | Behavior |
|:---|:---|
| `full` | Run inference, merge chunk CSVs, and run all dataset evaluators |
| `infer` | Run inference and merge predictions only |
| `evaluate` | Evaluate an existing `results.csv` without loading a model |
| `validate` | Run the dataset's configured integrity checks only |

For example, evaluate existing StreamingBench predictions with the same Hydra
dataset profile used for inference:

```bash
python scripts/run.py dataset=streamingbench run.mode=evaluate \
    paths.results_path=results/llava_ov_7b/streamingbench/fps0.5-kv6000/results.csv
```

The three-video smoke profile is run with:

```bash
python scripts/run.py experiment=streamingbench_smoke
```

Hydra multirun provides parameter sweeps. Each value below has an isolated
output directory and token trace:

```bash
python scripts/run.py -m experiment=streamingbench_min_tokens \
    run.min_tokens_per_frame=0,1,4,8,16
```

The per-frame token floor may increase the effective KV budget. With `k=1`, a
frame with no selected patch token receives a position-aligned mean-pooled
summary token.

Cluster launchers contain only Slurm resources and environment setup; experiment
parameters remain in Hydra profiles:

```bash
sbatch scripts/slurm/streamingbench_subset.sh
sbatch scripts/slurm/streamingbench_cr_su_eu.sh
```

Prepare the focused CR/SU/EU dataset separately when needed:

```bash
python scripts/data/download_streamingbench_tasks.py --dry-run
python scripts/data/download_streamingbench_tasks.py --tasks CR SU EU
```

Python utilities are organized by responsibility:

```text
configs/             Hydra model, dataset, and experiment profiles
eval/                benchmark metrics
eval/rvs/            RVS validation and judging
scripts/data/         dataset preparation
scripts/slurm/        cluster launchers
scripts/sweeps/       sweep controllers
tests/                CPU regression tests
```

Run the configuration and CPU regression tests with:

```bash
python -m unittest discover -s tests
```

## S-EMBER grounded streaming evaluation

S-EMBER can be read directly from its official JSONL layout; no converted copy
of the 369 GB video dataset is needed. The adapter resolves videos from NFS,
groups questions by video, and sorts them by `question_time` before causal
streaming inference.

Run the one-video, three-question smoke test with the local 0.5B checkpoint:

```bash
source .venv/hermes/bin/activate
python scripts/run.py experiment=sember_smoke
```

On Slurm:

```bash
sbatch scripts/slurm/sember_smoke.sh
```

The smoke profile reads
`/nfs-stor/chieu.nguyen/s-ember/sember_grounding.jsonl`, samples at 0.2 FPS,
and writes predictions plus deterministic S-EMBER temporal grounding metrics
under `results/llava_ov_0.5b/sember_grounding/smoke-fps0.2-kv1024/`.

Inspect the resolved command without loading a model:

```bash
python scripts/run.py experiment=sember_smoke runtime.dry_run=true
```

For the full dataset, remove the smoke limit and choose the intended model and
sampling settings explicitly, for example:

```bash
python scripts/run.py dataset=sember_grounding model=llava_ov_0.5b \
    dataset.max_videos=null run.sample_fps=0.5 run.kv_size=6000 \
    run.use_history=false
```

The model is prompted to return both a short answer and
`Time: [start_seconds, end_seconds]`. Evaluation reports temporal mIoU,
R@1 at IoU >= 0.5, parse rate, and per-category results. LLM-based semantic
answer judging remains a separate optional stage.

### S-EMBER MCQ

The five-way MCQ split is also read directly from the official JSONL. Run its
one-video, two-question smoke test with:

```bash
python scripts/run.py experiment=sember_mcq_smoke
```

On Slurm:

```bash
sbatch scripts/slurm/sember_mcq_smoke.sh
```

The adapter uses the official prompt, preserves the A-E options, and requires
the model to return only one letter. Results are written under
`results/llava_ov_0.5b/sember_mcq/smoke-fps0.2-kv1024/`; evaluation reports
overall accuracy, parse rate, per-category accuracy, and predicted-letter
distribution. Run the full split by removing the video limit:

```bash
python scripts/run.py dataset=sember_mcq model=llava_ov_0.5b \
    dataset.max_videos=null run.sample_fps=0.5 run.kv_size=6000 \
    run.use_history=false
```

To evaluate only Time Duration, Counting, and Location Trace, use the focused
profiles. The smoke profile covers all three categories with 22 questions from
12 videos:

```bash
python scripts/run.py experiment=sember_mcq_time_count_location_smoke
```

The full profile evaluates 4,558 questions from 2,752 videos (1,935 Time
Duration, 1,627 Counting, and 996 Location Trace):

```bash
python scripts/run.py experiment=sember_mcq_time_count_location
```

The category filter is applied before `max_videos`, so a limited run selects
the first N videos containing one of the requested tasks. The resulting MCQ
metrics file includes the combined accuracy and a separate score for each task.


## 📧 Contact

For any questions regarding the paper or the technical implementation, please feel free to contact haowei.zhang123@gmail.com


## 🙏 Acknowledgements

Our codebase is built upon [ReKV](https://github.com/Becomebright/ReKV). We gratefully acknowledge their contributions to the community.


## 📝 Citation

If you find our work useful for research, please cite our paper and give us a precious star 😄:

```bibtex
@misc{zhang2026hermeskvcachehierarchical,
      title={HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding}, 
      author={Haowei Zhang and Shudong Yang and Jinlan Fu and See-Kiong Ng and Xipeng Qiu},
      year={2026},
      eprint={2601.14724},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.14724}, 
}
```
