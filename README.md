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
├── eval/
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
│   └── run_infer.sh
├── video_qa/
│   ├── base.py
│   ├── hermes_vqa.py
│   └── run_infer.py
├── LICENSE
├── README.md
├── requirements_llava.txt
└── requirements_qwen.txt
```


## 🚀 Inference

Simply run the inference script:

```bash
bash scripts/run_infer.sh
```

Here is the content of `scripts/run_infer.sh`:

```bash
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH

num_chunks=8
model=llava_ov_7b
dataset=streamingbench

python video_qa/run_infer.py \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 0.5 \
    --kv_size 6000
```

To run only the Real-Time Visual Understanding subtasks Causal Reasoning (CR),
Spatial Understanding (SU), and Event Understanding (EU), use the focused
launcher below. It first downloads only the referenced video members from the
official ZIP archives into shared storage, then runs inference and evaluation:

```bash
sbatch scripts/run_streamingbench_cr_su_eu.sh
```

The script requests one GPU from the `cscc-gpu-p` production partition with
the same Slurm account/QoS settings as the repository's other batch jobs. It
runs the default `k=1,0` sweep sequentially inside that one allocation and
can also be run with `bash` from an existing GPU allocation. It defaults to
`llava_ov_0.5b`, 0.5 FPS, one GPU, and a 6,000-token
KV budget. Override `STREAMINGBENCH_MODEL`, `STREAMINGBENCH_SAMPLE_FPS`,
`STREAMINGBENCH_KV_SIZE`, `STREAMINGBENCH_ROOT`,
`STREAMINGBENCH_MIN_TOKENS_SWEEP`, or `STREAMINGBENCH_SAVE_DIR` as needed. Set
`STREAMINGBENCH_MIN_TOKENS_PER_FRAME` to run only one `k` value.
Set `STREAMINGBENCH_DEBUG=true` for a one-video validation run, or
`STREAMINGBENCH_SKIP_DOWNLOAD=true` when the prepared manifest is already
complete. The downloader can also be
inspected without network access using:

```bash
python scripts/download_streamingbench_tasks.py --dry-run
```

For the prepared three-video subset under `data/streamingbench/subset/`, run:

```bash
sbatch scripts/run_streamingbench_subset.sh
```

This submits one Slurm job with one GPU and runs the default `k=1,0` sweep
sequentially inside that allocation. Each `k` writes isolated predictions and
evaluation under `results/<model>/streamingbench_subset/`; for example,
`min-k4/` contains the `k=4` result. To run one configuration directly with an
existing GPU allocation, use `bash` instead. Override
`STREAMINGBENCH_MODEL`, `STREAMINGBENCH_SAMPLE_FPS`, `STREAMINGBENCH_KV_SIZE`,
`STREAMINGBENCH_SUBSET_ROOT`, `STREAMINGBENCH_ANNO_PATH`, or
`STREAMINGBENCH_SAVE_DIR` as needed. The launcher also writes per-frame token
retention to `token_retention.csv`; override its location with
`STREAMINGBENCH_TOKEN_TRACE_PATH`. It also writes
`token_retention_summary.csv` with per-event/per-layer and overall averages.
The CSV trace remains enabled, but detailed per-event/per-layer console output
is disabled by default; pass `--verbose_token_trace true` to the inference
script when that diagnostic logging is needed.
Set `STREAMINGBENCH_MIN_TOKENS_SWEEP`, for example to `1,4,8,16`, to change the
sequential sweep. Set
`STREAMINGBENCH_MIN_TOKENS_PER_FRAME` to a positive value, such as `4`, when
running directly with `bash` to first select the normal `kv_size` tokens and
then top up frames with fewer than that many selected tokens in every layer.
For `k=1`, a frame with no selected patch token receives one synthetic summary
token: its K/V values are mean-pooled from that frame after position-aware
RoPE alignment, and its trace records the frame's mean attention score.
The effective cache can then exceed
`kv_size` by the amount needed for this guarantee. Set
`STREAMINGBENCH_DEBUG=true` to run only the first video as a quick validation.

**Arguments:**

| Argument | Description |
|:---|:---|
| `model` | Model to use. Options: `llava_ov_0.5b`, `llava_ov_7b`, `llava_ov_72b`, `qwen2.5_vl_3b`, `qwen2.5_vl_7b`, `qwen2.5_vl_32b` |
| `dataset` | Benchmark dataset. Options: `videomme`, `mvbench`, `egoschema`, `rvs_ego`, `rvs_movie`, `ovobench`, `streamingbench` |
| `num_chunks` | Number of parallel processes for evaluation, typically set to the number of GPUs |
| `sample_fps` | Frame sampling rate (frames per second) from the video |
| `kv_size` | Maximum KV cache size for HERMES hierarchical memory management |
| `min_tokens_per_frame` | Optional per-frame visual-token floor; may increase the effective KV budget |
| `only_eval` | If set, skip inference and only run evaluation on existing results |


## 📊 Evaluation

The evaluation scripts compute metrics on the inference results:

- **Multiple-choice benchmarks** (VideoMME, MVBench, EgoSchema, OVBench, StreamingBench) are evaluated by `eval/eval_multiple_choice.py`, which takes a subcommand as its first argument:

| Subcommand | Description | Used by |
|:---|:---|:---|
| `general` | Compute overall accuracy, task-specific breakdown (auto-detects OVBench / StreamingBench), and prediction error analysis | MVBench, OVBench, StreamingBench, VideoMME |
| `videomme` | Report accuracy broken down by video duration (short / medium / long) | VideoMME |
| `egoschema` | Generate EgoSchema submission CSV file | EgoSchema |

```bash
python eval/eval_multiple_choice.py general --results_path results/llava_ov_7b/streamingbench/fps0.5-kv6000/results.csv
```

- **Open-ended benchmarks** (RVS-Ego, RVS-Movie) are evaluated by `eval/eval_open_ended.py`, which uses GPT for answer scoring:

```bash
python eval/eval_open_ended.py \
    --pred_path results/llava_ov_7b/rvs_ego/fps0.5-kv6000/results.csv \
    --output_dir results/llava_ov_7b/rvs_ego/fps0.5-kv6000/tmp \
    --output_json results/llava_ov_7b/rvs_ego/fps0.5-kv6000/results.json
```


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
