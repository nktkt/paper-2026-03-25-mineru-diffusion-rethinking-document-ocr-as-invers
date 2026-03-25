# MinerU-Diffusion: Document OCR as Inverse Rendering via Diffusion Decoding

**Unofficial** PyTorch implementation of the core algorithms from [MinerU-Diffusion](https://arxiv.org/abs/2603.22458) (arXiv:2603.22458) by Dong, Niu, Wang, Zeng, Zhang & He (Shanghai AI Lab, 2026).

> **Disclaimer**: This is an independent reimplementation for educational and research purposes. It is not affiliated with the original authors. See [Limitations](#limitations) for what is and is not included.

## What This Project Implements

MinerU-Diffusion reframes document OCR as **inverse rendering via discrete diffusion**, replacing autoregressive (left-to-right) decoding with parallel diffusion denoising. This implementation provides the core algorithmic components:

| Component | Paper Section | Status |
|-----------|--------------|--------|
| Discrete diffusion forward/reverse process | Eq. 3-5 | Implemented |
| Block-attention mask (bidirectional within, causal across) | Eq. M_ij | Implemented |
| Block-attention Transformer decoder | Section 3.2 | Implemented |
| ELBO training objective | Eq. 5 | Implemented |
| Confidence-based adaptive scheduling | Section 3.3 | Implemented |
| Uncertainty-driven curriculum learning | Section 3.4 | Implemented |
| Semantic Shuffle benchmark | Section 4.3 | Implemented |
| Vision encoder (Qwen2-VL-7B) | Section 3.1 | Stub only |

## Intentionally Omitted

- Pretrained Qwen2-VL-7B vision encoder (7B parameters)
- Pretrained SDAR-1.7B decoder weights
- Full 6.9M sample OCR training dataset
- OmniDocBench / OCRBench evaluation infrastructure
- Layout detection pipeline
- OTSL table output format
- Multi-stage curriculum training on real data

## Project Structure

```
mineru_diffusion/
├── __init__.py              # Package exports
├── config.py                # Model, diffusion, scheduler, curriculum configs
├── diffusion.py             # Discrete diffusion: forward process, ELBO loss
├── block_attention.py       # Block-attention mask & multi-head attention
├── model.py                 # Full MinerU-Diffusion model
├── scheduler.py             # Confidence-based adaptive inference scheduler
├── curriculum.py            # Uncertainty-driven hard-case mining
└── benchmark.py             # Semantic Shuffle benchmark
tests/                       # 58 tests across all components
examples/
└── demo.py                  # End-to-end demonstration
docs/
├── plan.md                  # Implementation plan & architecture
├── assumptions.md           # Assumptions & known limitations
└── review.md                # Quality review results
```

## Key Concepts

### Block-Attention Mask

The central architectural innovation: within each block of tokens, full bidirectional attention allows parallel diffusion denoising. Across blocks, causal attention maintains autoregressive structure:

```
M_ij = 1  if b(i) = b(j)    (same block - bidirectional)
     = 1  if b(j) < b(i)    (preceding block - causal)
     = 0  otherwise          (future block - masked)
```

This reduces complexity from O(L^2) to O(B * L'^2) where L = B * L'.

### Discrete Diffusion

Forward process independently masks each token with probability t:

```
q(x_t | x_0) = Prod_i Cat(x_t^i; (1-t) * delta_{x_0} + t * delta_{[MASK]})
```

Training minimizes the ELBO, predicting original tokens at masked positions.

### Adaptive Scheduling

At inference, tokens are progressively confirmed when model confidence exceeds threshold tau. Higher tau = more accurate but slower; lower tau = faster but noisier.

## Requirements

- Python >= 3.9
- PyTorch >= 2.1.0

## Installation

```bash
git clone https://github.com/<your-username>/paper-2026-03-25-mineru-diffusion-rethinking-document-ocr-as-invers.git
cd paper-2026-03-25-mineru-diffusion-rethinking-document-ocr-as-invers
pip install -e ".[dev]"
```

## Usage

```python
import torch
from mineru_diffusion import ModelConfig, MinerUDiffusion, PRESET_CONFIGS

# Create model
config = PRESET_CONFIGS["small"]  # or "tiny", "medium"
model = MinerUDiffusion(config)
print(f"Parameters: {model.count_parameters():,}")

# Training step
x_0 = torch.randint(1, config.vocab_size, (4, 64))
result = model.compute_loss(x_0)
result["loss"].backward()

# Generation with adaptive scheduling
output = model.generate(seq_len=64, batch_size=1, num_steps=10, confidence_threshold=0.95)
```

### Block-Attention Mask Visualization

```python
from mineru_diffusion import create_block_attention_mask

mask = create_block_attention_mask(seq_len=12, block_size=4)
# Within blocks: full attention (bidirectional)
# Across blocks: causal (can see past, not future)
```

### Semantic Shuffle Benchmark

```python
from mineru_diffusion import shuffle_words, create_benchmark_dataset

samples = create_benchmark_dataset(
    texts=["The neural network achieved state-of-the-art results"],
    distortion_levels=[0.0, 0.25, 0.5, 0.75, 1.0],
)
```

## Example

```bash
python examples/demo.py
```

## Testing

```bash
pytest tests/ -v
# 58 tests covering: diffusion, block-attention, model, scheduler, benchmark
```

## Limitations

1. **No real OCR capability** - Without pretrained weights and vision encoder, the model cannot perform actual document OCR
2. **Accuracy numbers not reproducible** - Paper results require full training pipeline with 6.9M samples
3. **Vision encoder is a stub** - Real implementation requires Qwen2-VL-7B (7B parameters)
4. **ELBO approximation** - Uses single-sample Monte Carlo; paper may use specific quadrature
5. **Untrained model outputs are uninformative** - Expected behavior; demonstrates algorithm correctness, not OCR quality

## Future Work

- Integration with HuggingFace Transformers for real vision encoders
- Support for loading pretrained SDAR decoder weights
- Full OmniDocBench evaluation pipeline
- Layout detection integration
- Distributed training support

## Citation

```bibtex
@misc{dong2026minerudiffusion,
    title={MinerU-Diffusion: Rethinking Document OCR as Inverse Rendering via Diffusion Decoding},
    author={Hejun Dong and Junbo Niu and Bin Wang and Weijun Zeng and Wentao Zhang and Conghui He},
    year={2026},
    eprint={2603.22458},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License

MIT - This is an unofficial implementation for research purposes.
