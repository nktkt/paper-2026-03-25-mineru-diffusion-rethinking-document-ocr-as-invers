# Implementation Plan: MinerU-Diffusion

## Summary

MinerU-Diffusion reframes document OCR as **inverse rendering via discrete diffusion**,
replacing autoregressive (left-to-right) decoding with parallel diffusion denoising.
Key innovations: block-attention diffusion decoder (full attention within blocks,
causal across blocks), confidence-based adaptive scheduling, and uncertainty-driven
curriculum learning. Achieves 2-3x speedup over AR baselines with competitive accuracy.

## MVP Scope

Implement the **core algorithmic contributions** that can run without
massive compute, proprietary data, or pretrained VL models:

### Included (MVP)
1. **Discrete diffusion process** - Forward masking + reverse denoising (MDLM-style ELBO)
2. **Block-attention mechanism** - The paper's key architectural contribution
3. **Block-attention diffusion decoder** - Transformer with block-causal mask
4. **Confidence-based adaptive scheduler** - Dynamic token confirmation at inference
5. **Semantic shuffle benchmark** - Word-level shuffle for evaluating visual grounding
6. **Uncertainty-driven curriculum utilities** - Hard-case mining via inference consistency
7. **End-to-end small model** - A miniature MinerU-Diffusion for testing/demo
8. **Comprehensive tests** for all components

### Explicitly NOT included (Non-goals)
- Qwen2-VL-7B vision encoder (7B params, proprietary weights)
- SDAR-1.7B-Chat-b32 pretrained decoder
- Full 6.9M sample OCR training dataset
- OmniDocBench / OCRBench evaluation infrastructure
- Layout detection pipeline
- Multi-stage curriculum training on full data
- OTSL table output format

## Architecture

```
Input Image  -->  [Vision Encoder (stub)]  -->  Visual Features
                                                      |
                                                      v
[MASK] tokens  -->  Block-Attention Diffusion Decoder  -->  Denoised tokens
                    (causal across blocks,                   |
                     bidirectional within blocks)             v
                                                      Confidence Filter
                                                      (τ threshold)
                                                           |
                                                           v
                                                     Final Output
```

### Key Equations

**Forward process:** q(x_t | x_0) = Π Cat(x_t^i; (1-t)δ_{x_0^i} + t·δ_{[MASK]})

**ELBO objective:**
J(x_0, Q, θ) = ∫₀¹ 1/(t|x_0|) E_{q(x_t|x_0)} [Σ_{i:masked} log p_θ(x_0^i|x_t, Q)] dt

**Block-attention mask:**
M_ij = 1 if b(i)=b(j) (same block) or b(j)<b(i) (preceding block), else 0

**Confidence scheduling:** Token confirmed when confidence > τ (default 0.95)

## File Plan

| File | Purpose |
|------|---------|
| `mineru_diffusion/config.py` | Configuration dataclasses |
| `mineru_diffusion/diffusion.py` | Discrete diffusion forward/reverse process, ELBO loss |
| `mineru_diffusion/block_attention.py` | Block-attention mask generation and module |
| `mineru_diffusion/model.py` | Full MinerU-Diffusion model (encoder + decoder) |
| `mineru_diffusion/scheduler.py` | Confidence-based adaptive inference scheduler |
| `mineru_diffusion/curriculum.py` | Uncertainty-driven hard-case mining |
| `mineru_diffusion/benchmark.py` | Semantic shuffle benchmark |
| `tests/test_diffusion.py` | Diffusion process tests |
| `tests/test_block_attention.py` | Block-attention tests |
| `tests/test_model.py` | Model integration tests |
| `tests/test_scheduler.py` | Scheduler tests |
| `tests/test_benchmark.py` | Benchmark tests |
| `examples/demo.py` | End-to-end demo |

## Test Plan

1. **Diffusion process**: Forward masking rates, reverse denoising shapes, ELBO loss computation
2. **Block attention**: Mask correctness, causal property, within-block bidirectionality
3. **Model**: Forward pass shape, loss computation, gradient flow, generation
4. **Scheduler**: Confidence filtering, token confirmation, adaptive step count
5. **Benchmark**: Shuffle correctness, distortion levels, metric computation
6. **Integration**: End-to-end forward + generate on small model

## Risks

1. Discrete diffusion numerical stability (log-prob of masked tokens)
2. Block attention mask edge cases (last block smaller than block_size)
3. Confidence calibration without pretrained model may be unrealistic
4. Without real OCR data, demo is synthetic only

## Explicit Non-goals

- Reproducing the paper's exact accuracy numbers
- Training on real OCR datasets
- Integrating with actual vision encoders
- Production-ready OCR pipeline
