## Description
Repository for CMSC498K final project. Tests whether using a specialized model for each step within Chain-of-Thought improves performance compared to having a singular model generating the whole reasoning chain.

## Get Started
Package management is done via `uv`. Install it from https://docs.astral.sh/uv/getting-started/installation/.

After `uv` is installed, run `uv sync` to install necessary packages. If you have a CUDA enabled GPU, use `uv sync --extra cu128` to install torch with CUDA.

To run a standard LoRA finetuning script on the ScienceQA dataset, do

```
cd code
uv run preprocess.py
uv run finetune.py
```