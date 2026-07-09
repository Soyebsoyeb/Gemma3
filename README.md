<div align="center">

# SLM: A Small Language Model built from scratch

</div>

<p align="center">
  A minimal, hackable implementation of a small (270M parameter) language model, using a Gemma 3-style architecture with grouped query attention, RoPE, and sliding-window attention.
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#model-architecture">Architecture</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#license">License</a>
</p>

---

## About

This repository contains a from-scratch implementation of a small language model (SLM), covering the full pipeline from raw text to a trained, generating model:

- Tokenization and efficient on-disk storage of large datasets
- A Gemma 3-style transformer (RMSNorm, grouped query attention, RoPE, alternating local/global attention)
- A training loop with mixed precision, gradient accumulation, and learning rate scheduling
- Autoregressive text generation with temperature and top-k sampling

It is intended as a reference implementation for understanding and experimenting with modern small-model architectures, not as a production training framework.

## Installation

```bash
git clone https://github.com/<your-username>/slm_project.git
cd slm_project
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.x, and a CUDA-capable GPU (recommended, not required).

## Quick Start

```bash
# 1. Prepare and tokenize the dataset
python scripts/prepare_data.py --config config/data.yaml

# 2. Train the model
python scripts/train.py --config config/train.yaml

# 3. Run inference with a trained checkpoint
python scripts/inference.py --checkpoint checkpoints/best_model_params.pt --prompt "Once upon a time"
```

## Model Architecture

The model follows the Gemma 3 design:

| Component | Description |
|---|---|
| RMSNorm | Zero-centered weights; normalization computed in float32 regardless of working dtype |
| Grouped Query Attention | Fewer key/value heads than query heads, with optional QK-norm |
| RoPE | Rotary position embeddings with separate base frequencies for local and global layers |
| Attention pattern | Sliding-window attention on most layers, full attention every 6th layer |
| FeedForward | Gated MLP with GELU (tanh approximation) |

### Default configuration (270M)

| Parameter | Value |
|---|---|
| Vocabulary size | 50,257 |
| Context length | 32,768 |
| Embedding dimension | 640 |
| Attention heads | 4 |
| KV groups | 1 |
| Layers | 18 |
| FFN hidden dimension | 2,048 |
| Head dimension | 256 |
| Sliding window | 512 |
| Dtype | bfloat16 |

Model configs are defined in `config/` and loaded by `src/models/`.

## Training

Key training hyperparameters (see `config/train.yaml`):

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Max iterations | 150,000 |
| Warmup steps | 1,000 |
| Batch size | 32 |
| Block size | 128 |
| Gradient accumulation steps | 32 |
| Optimizer | AdamW (betas 0.9/0.95, weight decay 0.1) |
| LR schedule | Linear warmup + cosine decay |

Training data is read from memory-mapped `.bin` files produced by `scripts/prepare_data.py`, so datasets larger than available RAM can be used without issue. Checkpoints are saved to `checkpoints/` whenever validation loss improves.

To resume or monitor training:

```bash
python scripts/train.py --config config/train.yaml --resume checkpoints/last.pt
```

## Inference

```bash
python scripts/inference.py \
  --checkpoint checkpoints/best_model_params.pt \
  --prompt "A little girl went to the woods" \
  --max_new_tokens 200 \
  --temperature 1.0 \
  --top_k 50
```

## Project Structure

```
slm_project/
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   ├── models/             # Model architecture
│   ├── training/           # Training utilities
│   └── inference/          # Inference utilities
├── scripts/                # Executable scripts
│   ├── prepare_data.py     # Dataset preparation
│   ├── train.py            # Training script
│   └── inference.py        # Inference script
├── config/                 # Configuration files
├── data/                   # Dataset storage
├── notebooks/               # Jupyter notebooks
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
└── README.md                 # This file
```

## Notes

- Double-check `min_lr` against `learning_rate` in your config before a long run — a cosine schedule with `min_lr` set higher than the peak learning rate will not decay as expected.
- Reduce `max_iters` for a quick smoke test before committing to a full 150k-step run.

## Citation

If you use this code in your research, please cite it as:

```bibtex
@misc{slm_project,
  title  = {SLM: A Small Language Model built from scratch},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/<your-username>/slm_project}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
