# Which Knowledge Should LLMs Trust?

<p align="center">
  <b>Knowledge Conflict Steering for Large Language Models</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue">
  <img src="https://img.shields.io/badge/PyTorch-CUDA-orange">
</p>

## Overview

Large language models often face conflicts between their **internal parametric memory** and **external contextual knowledge**.

This repository provides the implementation of **knowledge conflict steering**, showing that a **single residual-stream direction** can control whether an instruction-tuned LLM follows internal memory or contextual information.

The discovered direction generalizes across:

- different conflict formats,
- factual relations,
- datasets,
- retrieval-augmented generation,
- memory tasks,
- instruction-following tasks.

<p align="center">
<img src="docs/paper_figures/png/overview_new.png" width="90%">
</p>

<table>
<tr>
<td align="center"><b>Context → Parameter</b><br><img src="docs/paper_figures/steering_direction.gif" width="96%" alt="Real Llama saved output changing from 336 to 291 Dragon Ball Z episodes"></td>
<td align="center"><b>Parameter → Context</b><br><img src="docs/paper_figures/steering_inner_to_outer.gif" width="96%" alt="Real Llama saved output changing from Cathy Dennis and Rob Davis to Jeannie Mai"></td>
</tr>
</table>

<p align="center"><sub>These animations use the exact saved first-sentence outputs from <code>docs/paper_figures/case_data_llama31.json</code>. Context → Parameter starts at α=0 on the left; Parameter → Context ends at α=0 on the right.</sub></p>

## Main Results

A single steering direction enables bidirectional control. The table below is
**Table 1 from the paper**: in-domain source-match rate (%) on NQ-Swap and
MacNoise. `Acc_par` measures parametric-source matches; `Acc_ctx` measures
contextual-source matches. Bold values are the best in each column.

| Method | NQ Llama<br>par | NQ Qwen<br>par | NQ Gemma<br>par | NQ Mistral<br>par | Mac Llama<br>par | Mac Qwen<br>par | Mac Gemma<br>par | Mac Mistral<br>par |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unsteered | 33.7 | 46.5 | 32.8 | 42.2 | 46.1 | 50.8 | 45.5 | 50.3 |
| Random split | 32.2 ± 2.6 | 45.7 ± 1.9 | 33.5 ± 2.3 | 41.6 ± 2.1 | 45.8 ± 3.1 | 50.2 ± 2.4 | 44.9 ± 2.0 | 49.7 ± 2.5 |
| ContextFocus | 44.6 | 49.1 | 45.2 | 47.7 | 50.2 | 52.5 | 51.3 | 54.2 |
| CCS | 53.5 | 50.8 | 58.6 | 52.1 | 55.2 | 53.1 | 59.4 | 56.8 |
| CSKS | 52.4 | 51.7 | 66.3 | 54.3 | 53.7 | 53.8 | 63.6 | 58.4 |
| CK-PLUG | 54.5 | 52.9 | 74.7 | 56.2 | 55.7 | 53.4 | 69.5 | 59.7 |
| PH3 | 48.3 | 47.8 | 61.5 | 49.5 | 49.5 | 51.7 | 54.7 | 52.6 |
| IRCAN | 49.5 | 51.4 | 59.3 | 50.8 | 50.0 | 52.4 | 63.2 | 53.9 |
| JUICE | 53.4 | 53.3 | 78.4 | 57.1 | 53.2 | **56.1** | 74.2 | 60.5 |
| SpARE | **58.7** | 49.2 | 77.3 | 55.4 | **59.5** | 52.3 | 72.3 | 57.6 |
| **Ours** | 56.8 | **54.6** | **80.9** | **59.0** | 57.5 | 54.5 | **76.9** | **61.6** |

| Method | NQ Llama<br>ctx | NQ Qwen<br>ctx | NQ Gemma<br>ctx | NQ Mistral<br>ctx | Mac Llama<br>ctx | Mac Qwen<br>ctx | Mac Gemma<br>ctx | Mac Mistral<br>ctx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unsteered | 49.4 | 36.7 | 52.6 | 45.8 | 27.3 | 22.2 | 25.4 | 23.8 |
| Random split | 49.2 ± 2.8 | 37.1 ± 2.2 | 51.9 ± 2.6 | 45.3 ± 1.8 | 26.1 ± 0.7 | 21.8 ± 1.4 | 24.9 ± 1.6 | 23.4 ± 1.3 |
| ContextFocus | 56.8 | 42.3 | 58.1 | 51.2 | 31.5 | 25.7 | 29.3 | 27.1 |
| CCS | 52.3 | 46.1 | 60.7 | 54.6 | 30.1 | 27.4 | 32.5 | 28.6 |
| CSKS | 57.2 | 49.7 | 62.9 | 56.8 | 32.2 | 30.1 | 35.4 | 29.7 |
| CK-PLUG | 53.4 | 55.2 | 64.3 | 57.7 | 30.9 | 29.5 | 37.7 | 30.2 |
| PH3 | 52.4 | 51.5 | 54.3 | 49.1 | 28.9 | 28.5 | 32.4 | 25.7 |
| IRCAN | 55.2 | 47.4 | 57.3 | 52.4 | 31.7 | 32.4 | 36.1 | 28.8 |
| JUICE | 54.5 | 48.4 | 66.9 | 58.3 | **34.0** | 34.2 | 38.5 | 30.4 |
| SpARE | 52.7 | 54.4 | **68.3** | 55.9 | 30.9 | 34.3 | **42.4** | 29.2 |
| **Ours** | **55.5** | **56.0** | 65.4 | **60.2** | 32.8 | **35.1** | 40.6 | **31.1** |

<p align="center">
<img src="docs/paper_figures/png/main_curves.png" width="90%">
</p>

## Quick Start

### Installation

```bash
git clone <repository-url>

cd knowledge_conflict_steering

pip install torch transformers tokenizers numpy matplotlib
```

For complex-task experiments:

```bash
pip install -r complex_tasks/requirements.txt
```

### Prepare Models

Place model checkpoints under:

```text
models/
```

Supported models:

| Model | Default steering layer |
|---|---:|
| Llama-3.1-8B-Instruct | 10 |
| Qwen3-8B | 21 |
| Gemma-2-9B-Instruct | 16 |
| Mistral-7B-Instruct-v0.3 | 16 |

### Run Steering

Dump activations:

```bash
GPU=0 MODEL=llama3.1-8b-it DATASETS=NQ-Swap \
bash run/run_dump_all_datasets.sh
```

Estimate and apply steering direction:

```bash
GPU=0 MODEL=llama3.1-8b-it DATASET=NQ-Swap CATEGORY=dev LAYERS=10 \
bash run/run_resid_projection_ablation.sh
```

## Experiments

The repository supports:

- In-domain knowledge conflict steering
- Cross-format transfer
- Cross-knowledge-type transfer
- Cross-dataset transfer
- RAG conflict steering
- Memory conflict steering
- Instruction-following control

## Command Cookbook

All commands run from the repository root. Set `GPU`, `MODEL`, and the path
variables for your machine. Results are written under `output/` by default.

### Core five-dataset pipeline

```bash
# Dump NQ-Swap, MacNoise, ParaConflict, CounterFact, and MemoTrap.
GPU=0 MODEL=llama3.1-8b-it ./run/run_dump_all_datasets.sh

# Print the planned commands without loading a model.
GPU=0 MODEL=llama3.1-8b-it DATASETS=NQ-Swap,counterfact DRY_RUN=1 \
  ./run/run_dump_all_datasets.sh

# Estimate a direction and sweep alpha for one category.
GPU=0 MODEL=llama3.1-8b-it DATASET=NQ-Swap CATEGORY=dev LAYERS=10 \
  ./run/run_resid_projection_ablation.sh

# Run all categories with train/validation/test selection.
GPU=0 MODEL=llama3.1-8b-it LAYERS=10 SKIP_EXISTING=1 \
  ./run/run_resid_projection_ablation_all_final.sh
```

### Transfer and controls

```bash
# NQ-Swap direction -> MacNoise, ParaConflict, CounterFact, MemoTrap.
GPU=0 MODEL=llama3.1-8b-it LAYERS=10 \
  ./run/run_resid_projection_nqswap_dir_cross_datasets.sh

# ParaConflict substitution -> coherent-format transfer.
GPU=0 MODEL=llama3.1-8b-it LAYERS=10 \
  ./run/run_dump_paraconflict_substitution_cross_format.sh
GPU=0 MODEL=llama3.1-8b-it LAYERS=10 \
  ./run/run_resid_projection_paraconflict_cross_format.sh

# CounterFact relation/knowledge-type transfer.
GPU=0 MODEL=llama3.1-8b-it LAYERS=10 \
  ./run/run_resid_projection_knowledge_type_cross_steer.sh

# Four-condition prompt-structure control.
GPU=0 MODEL=llama3.1-8b-it ./run/run_four_condition_control.sh
GPU=0 MODEL=llama3.1-8b-it ./run/run_four_condition_fixed_full.sh
```

### Complex-task experiments from the paper

```bash
# Build processed manifests for the paper application datasets.
COMPLEX_RAW_ROOT="$PWD/data" PROCESSED_ROOT="$PWD/data/processed" \
  python3 complex_tasks/scripts/build_processed_jsonl.py \
  --datasets RGB,CRAG,LoCoMo,LoCoMo-Plus,LongMemEval,IFEval,FollowBench

# Dump activations for RAG, memory, and instruction-following tasks.
MODEL=llama3.1-8b-it GPU=0 \
  DATASETS=RGB,CRAG,LoCoMo,LoCoMo-Plus,LongMemEval,IFEval,FollowBench \
  ./complex_tasks/run/run_dump_taxonomy.sh

# Transfer the NQ-Swap direction to those tasks.
MODEL=llama3.1-8b-it GPU=0 \
  DATASETS=RGB,CRAG,LoCoMo,LoCoMo-Plus,LongMemEval,IFEval,FollowBench \
  ./complex_tasks/run/run_nqswap_dir_cross_taxonomy.sh

# Reuse mounted model/data/dump locations.
MODEL_ROOT=/path/to/models DUMP_ROOT=/path/to/dumps \
  STEER_OUT_ROOT=/path/to/results COMPLEX_RAW_ROOT=/path/to/data \
  ./run/run_paper_llama31_existing.sh
```

### Figures and local API judge

```bash
# Generate plots from completed outputs.
./run/run_plot_angle_fig_all.sh
python3 src/plot_nqswap_alpha_curves_multi_model.py --data-root output/steer_result
python3 src/plot_cross_format_paraconflict_bars.py --data-root output/cross_format
python3 src/plot_cross_relation_alpha_curves.py --data-root output/knowledge_type

# Configure the OpenAI-compatible judge used by complex-task scoring.
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_API_KEY=local
export LLM_MODEL=local-model
```

## Supported Datasets

| Dataset | Purpose |
|---|---|
| NQ-Swap | Factual knowledge conflicts |
| MacNoise | Contextual noise conflicts |
| ParaConflict | Structured knowledge conflicts |
| MemoTrap | Memory conflicts |
| CounterFact | Knowledge editing conflicts |

<details>
<summary><b>Repository Structure</b></summary>

```text
knowledge_conflict_steering/

├── data/
├── models/
├── src/
├── run/
├── complex_tasks/
├── docs/
└── output/
```

</details>

<details>
<summary><b>Detailed Reproduction Instructions</b></summary>

### Activation Extraction

```bash
GPU=0 ./run/run_dump_all_datasets.sh
```

### In-domain Steering

```bash
GPU=0 MODEL=llama3-8b-it CATEGORY=dev LAYERS=10 \
./run/run_resid_projection_ablation.sh
```

### Cross-domain Experiments

Available experiments:

- Cross-format transfer
- Cross-relation transfer
- Cross-dataset transfer

</details>
