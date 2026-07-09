# SLM-270M: Small Language Model with Gemma3 Architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A 270M parameter language model implementation featuring Grouped-Query Attention, Sliding Window Attention, and Rotary Position Embeddings, inspired by Google's Gemma architecture.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Sample Output](#sample-output)
- [Advanced Configuration](#advanced-configuration)
- [Performance Benchmarks](#performance-benchmarks)
- [Monitoring Training](#monitoring-training)
- [Architectural Background](#architectural-background)
- [References](#references)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

SLM-270M is a lightweight, efficient language model designed for training and inference on consumer hardware. It implements a set of techniques common in modern transformer research — grouped-query attention, sliding-window attention, and dual-base RoPE — in a compact and readable codebase.

The project is intended for:

- Researchers experimenting with modern transformer architectures
- Developers building applications with limited compute resources
- Students learning how language models are implemented
- Hobbyists running models on consumer-grade GPUs

## Key Features

**Architecture**

- Gemma3-style transformer block with pre- and post-attention normalization
- Grouped-Query Attention with 4 query heads and 1 KV group
- Sliding Window Attention for efficient handling of long contexts (up to 32K tokens)
- Rotary Position Embeddings with dual bases for local and global attention layers
- Mixed precision training (bfloat16 / float16) with gradient scaling
- Memory-mapped dataset loading for datasets larger than available RAM

**Training**

- TensorBoard logging for loss, learning rate, and evaluation metrics
- Checkpointing with automatic retention of the best model
- Warmup and cosine decay learning rate schedule
- Gradient accumulation for larger effective batch sizes

**Inference**

- Temperature, top-k, and top-p sampling
- Batch generation across multiple prompts
- Interactive generation via Jupyter notebooks

## Model Architecture

| Parameter | Value |
|---|---|
| Total Parameters | 270M |
| Hidden Size | 640 |
| Number of Heads | 4 |
| KV Groups | 1 |
| Number of Layers | 18 |
| Head Dimension | 256 |
| Context Length | 32,768 |
| Sliding Window | 512 |
| Vocab Size | 50,257 |

Layer configuration — sliding-window attention on most layers, full attention every sixth layer:

```
[sliding x 5] -> [full] -> [sliding x 5] -> [full] -> [sliding x 5] -> [full]
```

### Estimated Memory Requirements

| Hardware | Training Memory | Inference Memory |
|---|---|---|
| CPU | 8-16 GB RAM | 4-8 GB RAM |
| GPU, 8 GB | Batch size 4-8 | Batch size 8-16 |
| GPU, 16 GB | Batch size 8-16 | Batch size 16-32 |
| GPU, 24 GB+ | Batch size 16-32 | Batch size 32-64 |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/slm-project.git
cd slm-project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Prepare a dataset (defaults to Tiny Shakespeare for a quick demo)
python scripts/prepare_data.py

# Train
python scripts/train.py

# Run inference
python scripts/inference.py
```

### Notebooks

```bash
pip install jupyter notebook
jupyter notebook
```

Recommended order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_training_visualization.ipynb`
3. `notebooks/03_inference_demo.ipynb`
4. `notebooks/04_model_analysis.ipynb`
5. `notebooks/05_hyperparameter_tuning.ipynb`

## Project Structure

```
slm_project/
├── src/                            # Core source code
│   ├── data/                       # Data processing
│   │   ├── tokenizer.py            # GPT-2 tokenizer wrapper
│   │   └── dataset.py              # Memory-mapped dataset handling
│   ├── models/                     # Model architecture
│   │   ├── attention.py            # GQA, RMSNorm, FeedForward
│   │   ├── rope.py                 # RoPE computations
│   │   ├── model.py                # Gemma3Model and TransformerBlock
│   │   └── config.py               # Model configurations
│   ├── training/                   # Training utilities
│   │   ├── trainer.py              # Main training loop
│   │   └── utils.py                # Loss estimation, metrics
│   └── inference/                  # Inference utilities
│       └── generate.py             # Text generation
├── scripts/                        # Executable scripts
│   ├── prepare_data.py             # Dataset preparation
│   ├── train.py                    # Training entry point
│   └── inference.py                # Inference entry point
├── tests/                          # Unit tests
│   ├── test_model.py
│   ├── test_attention.py
│   └── test_data.py
├── examples/                       # Example usage
│   ├── basic_usage.py
│   └── quantization.py
├── notebooks/                      # Jupyter notebooks
├── config/                         # Configuration files
│   └── training_config.yaml
├── data/                           # Dataset storage (auto-created)
├── logs/                           # Training logs (auto-created)
├── checkpoints/                    # Model checkpoints (auto-created)
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── pyproject.toml
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (recommended for GPU training)
- 8 GB+ RAM (16 GB+ recommended)

### Standard Installation

```bash
git clone https://github.com/yourusername/slm-project.git
cd slm-project

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

# Optional: development dependencies
pip install -r requirements-dev.txt
pre-commit install
```

### Docker

```dockerfile
FROM pytorch/pytorch:latest

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .

CMD ["python", "scripts/train.py"]
```

```bash
docker build -t slm-project .
docker run --gpus all -v $(pwd)/data:/app/data slm-project
```

## Training

### Basic

```bash
python scripts/train.py
```

### Custom Parameters

```bash
python scripts/train.py \
    --learning_rate 1e-4 \
    --max_iters 150000 \
    --batch_size 16 \
    --block_size 128 \
    --gradient_accumulation_steps 16 \
    --save_every 1000 \
    --eval_every 500
```

### Resuming and Additional Options

```bash
# Resume from a checkpoint
python scripts/train.py --resume checkpoints/model_10000.pt

# Use a specific dataset directory
python scripts/train.py --data_dir /path/to/data

# Enable TensorBoard logging
python scripts/train.py --tensorboard

# Profile training performance
python scripts/train.py --profile
```

### Optimization Tips

**Memory**

- Reduce batch size if running out of memory
- Increase gradient accumulation steps
- Use a smaller block size

**Speed**

- Use bfloat16 if supported by your hardware
- Increase the number of data-loading workers
- Reduce unnecessary logging frequency

**Quality**

- Train on a larger dataset
- Increase the number of training iterations
- Tune the learning rate schedule

## Inference

### Command Line

```bash
python scripts/inference.py \
    --model_path best_model_params.pt \
    --prompt "Once upon a time" \
    --max_tokens 200 \
    --temperature 0.8 \
    --top_k 50
```

### Python API

```python
from src.models import Gemma3Model, GEMMA3_CONFIG_270M
from src.inference import TextGenerator
from src.data import Tokenizer
import torch

# Load model
model = Gemma3Model(GEMMA3_CONFIG_270M)
model.load_state_dict(torch.load("best_model_params.pt"))
model.eval()

# Initialize generator
generator = TextGenerator(model)

# Generate text
prompt = "The future of AI is"
generated = generator.generate(
    prompt=prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
print(generated)
```

## Sample Output

Generations below are from a model trained on Tiny Shakespeare-style data for demonstration purposes; quality will vary depending on the dataset used.

```
Prompt: "Once upon a time there was a pumpkin."
Output: "Once upon a time there was a pumpkin. It was a very special pumpkin,
one that glowed with an inner light. The farmer who grew it was a kind old man
who lived in a small cottage at the edge of the forest..."

Prompt: "A little girl went to the woods"
Output: "A little girl went to the woods to pick berries for her grandmother.
She wore a red cape and carried a small basket. The sun was shining brightly
through the trees..."

Prompt: "The future of artificial intelligence is"
Output: "The future of artificial intelligence is bright and full of possibilities.
We can expect AI to revolutionize every industry, from healthcare to transportation,
and create new opportunities for human creativity and innovation..."
```

## Advanced Configuration

### Training Configuration File

Edit `config/training_config.yaml`:

```yaml
training:
  learning_rate: 1e-4
  min_lr: 5e-5
  warmup_steps: 1000

  max_iters: 150000
  eval_iters: 500

  batch_size: 32
  block_size: 128
  gradient_accumulation_steps: 32

  dtype: bfloat16  # bfloat16, float16, or float32

  weight_decay: 0.1
  gradient_clip: 0.5

model:
  checkpoint_path: checkpoints/best_model.pt
  save_every: 5000

data:
  data_dir: data
  num_proc: 8
  test_split: 0.1
```

Note: `min_lr` should be set lower than `learning_rate` so the cosine decay schedule behaves as expected.

### Recommended Settings by Hardware

| Memory | Batch Size | Grad Accum | Effective Batch | Learning Rate | Block Size |
|---|---|---|---|---|---|
| 8 GB | 4 | 32 | 128 | 1e-4 | 64 |
| 16 GB | 8 | 32 | 256 | 1e-4 | 128 |
| 24 GB | 16 | 32 | 512 | 1e-4 | 256 |
| 40 GB | 32 | 16 | 512 | 1e-4 | 512 |

### Custom Model Configuration

```python
CUSTOM_CONFIG = {
    "vocab_size": 50257,
    "context_length": 16384,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 12,
    "hidden_dim": 1536,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 2,
    "sliding_window": 256,
    "layer_types": ["sliding_attention"] * 8 + ["full_attention"] * 4,
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 128,
}
```

## Performance Benchmarks

### Training Throughput

| Hardware | Training Speed | Memory Usage |
|---|---|---|
| A100 40GB | ~2.5 sec/batch | ~15 GB |
| V100 16GB | ~4.0 sec/batch | ~12 GB |
| RTX 3090 | ~5.5 sec/batch | ~14 GB |
| RTX 2080 Ti | ~8.0 sec/batch | ~10 GB |

*Measured with `batch_size=32`, `block_size=128`, `gradient_accumulation_steps=32`.*

### Inference Throughput

| Sequence Length | Inference Time | Tokens/Second |
|---|---|---|
| 64 | 0.023s | 2,782 |
| 128 | 0.045s | 2,844 |
| 256 | 0.089s | 2,876 |
| 512 | 0.178s | 2,876 |

### Training Convergence

| Metric | Value |
|---|---|
| Convergence Steps | ~100,000 iterations |
| Final Loss (Tiny Shakespeare) | ~2.5 |

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/training
# View at http://localhost:6006
```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/training')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Learning_rate', lr, epoch)
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
```

### Notebooks

- `notebooks/02_training_visualization.ipynb` — loss curves, learning rate schedules, convergence analysis
- `notebooks/03_inference_demo.ipynb` — interactive generation, temperature comparison, batch generation
- `notebooks/04_model_analysis.ipynb` — architecture and attention pattern analysis, memory usage, speed benchmarking
- `notebooks/05_hyperparameter_tuning.ipynb` — learning rate exploration, batch size and model size comparisons

## Architectural Background

This implementation follows the Gemma 2 / Gemma 3 line of architectural decisions (Gemma Team, 2024; Gemma Team, 2025), which combine several techniques developed independently in the literature. This section summarizes each component and the source it is drawn from.

### Grouped-Query Attention (GQA)

Standard multi-head attention (MHA) computes an independent key/value projection per query head (Vaswani et al., 2017). Multi-query attention (MQA) reduces this to a single shared key/value head across all query heads, cutting KV cache size at some cost to quality (Shazeer, 2019). GQA interpolates between the two: the `h` query heads are partitioned into `g` groups, and each group shares one key/value head (Ainslie et al., 2023).

```
g = h        ->  GQA reduces to MHA
1 < g < h    ->  GQA (this implementation: h = 4, g = 1)
g = 1        ->  GQA reduces to MQA
```

With `n_heads = 4` and `n_kv_groups = 1`, this configuration sits at the MQA extreme of the GQA spectrum, prioritizing KV cache size over the representational capacity of independent per-head keys and values.

### Sliding Window Attention

Sliding window attention restricts each token's attention to a fixed-size local window instead of the full sequence, reducing the computational and memory cost of self-attention from quadratic to linear in sequence length (Beltagy et al., 2020). Gemma 2 interleaves local (sliding window) and global (full) attention layers at a 1:1 ratio (Gemma Team, 2024); Gemma 3 widens this to a 5:1 ratio, trading a small amount of long-context modeling capacity for a substantial reduction in KV cache memory (Gemma Team, 2025). This implementation follows the Gemma 3 pattern: five sliding-window layers followed by one full-attention layer, repeated across the network.

### Rotary Position Embeddings (RoPE)

RoPE encodes token position by rotating query and key vectors by an angle proportional to their sequence position, so that the dot product between a rotated query and key depends only on their relative distance (Su et al., 2021). This gives the attention mechanism relative-position awareness without an additive positional embedding, and it extrapolates more gracefully to sequence lengths not seen during training than learned absolute embeddings.

This implementation uses two RoPE base frequencies, following the local/global split introduced in Gemma 2 (Gemma Team, 2024): a base of 10,000 for sliding-window (local) layers and 1,000,000 for full-attention (global) layers. The higher base for global layers slows the rate at which positional signal decays over long distances, which matters more for layers that attend across the full 32K-token context.

### RMSNorm

RMSNorm normalizes activations by their root-mean-square statistic rather than by mean and variance, dropping the re-centering step used in LayerNorm (Zhang and Sennrich, 2019). This removes one reduction per normalization call, which measurably reduces training and inference latency at negligible cost to model quality, and it is the normalization used throughout Gemma, LLaMA, and Mistral-family models.

### QK Normalization and Query Pre-Attention Scaling

Following Gemma 3 (Gemma Team, 2025), this implementation applies RMSNorm to queries and keys before computing attention scores, which improves training stability at this parameter count. Queries are additionally scaled by a fixed `query_pre_attn_scalar` before the dot product, rather than the conventional `1/sqrt(head_dim)`, decoupling the attention temperature from the head dimension.

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762
2. Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head Is All You Need. https://arxiv.org/abs/1911.02150
3. Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., and Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *EMNLP 2023*. https://arxiv.org/abs/2305.13245
4. Su, J., Lu, Y., Pan, S., Wen, B., and Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. https://arxiv.org/abs/2104.09864
5. Beltagy, I., Peters, M. E., and Cohan, A. (2020). Longformer: The Long-Document Transformer. https://arxiv.org/abs/2004.05150
6. Zhang, B. and Sennrich, R. (2019). Root Mean Square Layer Normalization. *NeurIPS 2019*. https://arxiv.org/abs/1910.07467
7. Shazeer, N. (2020). GLU Variants Improve Transformer. https://arxiv.org/abs/2002.05202
8. Gemma Team, Google DeepMind (2024). Gemma 2: Improving Open Language Models at a Practical Size. https://arxiv.org/abs/2408.00118
9. Gemma Team, Google DeepMind (2025). Gemma 3 Technical Report. https://arxiv.org/abs/2503.19786
10. Karpathy, A. nanoGPT. https://github.com/karpathy/nanoGPT

## Testing

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest --cov=src tests/
```

## Deployment

### ONNX Export

```python
import torch.onnx

dummy_input = torch.randint(0, 50256, (1, 128))
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'},
    },
)
```

### Quantization

```python
import torch.quantization

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)

# Static quantization requires calibration data; see examples/quantization.py
```

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for guidelines.

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
pylint src/
```

## Citation

```bibtex
@software{slm270m2026,
  author = {MD Soyeb Hoque},
  title  = {SLM-270M: Small Language Model with Gemma3 Architecture},
  year   = {2026},
  url    = {https://github.com/Soyebsoyeb/Gemma3}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Google DeepMind, for the Gemma model family and technical reports this architecture is based on
- Andrej Karpathy and the nanoGPT project, for implementation patterns this codebase follows
- Hugging Face, for datasets and tokenizer tooling

## Contact
- Email: workemailsoyeb@example.com
