# Assumptions and Limitations

## Assumptions Made During Implementation

### Architecture
1. **Vision encoder is treated as a stub.** The paper uses Qwen2-VL-7B as the vision
   encoder. We implement a simple linear projection as a placeholder since the 7B model
   requires significant compute and specific weights.

2. **Block size = 32** is used as default, matching the paper's SDAR-1.7B-Chat-b32 configuration.

3. **Discrete diffusion follows MDLM formulation.** The paper cites MDLM (Masked Diffusion
   Language Models) as the basis. We implement the continuous-time ELBO with absorbing
   state ([MASK] token) as described in Eq. 3-5 of the paper.

4. **Block-attention mask** is implemented exactly as described: full bidirectional attention
   within each block, causal attention across blocks (each block can attend to all
   preceding blocks but not future blocks).

### Training
5. **Curriculum learning is provided as utilities only.** The two-stage curriculum
   (diversity-driven + uncertainty-driven) requires the full 6.9M dataset and multiple
   training stages. We implement the hard-case mining algorithm and weighted loss, but
   do not run full curriculum training.

6. **We assume the confidence metric C(x) = 1 - variance across T stochastic passes.**
   The paper describes inference consistency but does not give the exact formula for C(x).

### Inference
7. **Confidence threshold τ = 0.95 is the default** for adaptive scheduling, as this
   gives the best accuracy-speed tradeoff per the paper's ablations.

8. **Number of diffusion steps is set to 10 by default** for the reverse process.
   The paper does not specify an exact number but discusses adaptive scheduling.

### Evaluation
9. **Semantic Shuffle benchmark** is implemented as described: word-level shuffling
   within each text region while preserving visual layout. Distortion levels are
   controlled by shuffle probability.

## Known Limitations

1. **No real OCR capability.** Without a trained vision encoder and decoder, the model
   cannot perform actual document OCR. The implementation demonstrates the algorithm
   on synthetic/toy data.

2. **No pretrained weights.** The paper's models (Qwen2-VL-7B encoder, SDAR-1.7B decoder)
   are not available or require significant resources.

3. **Accuracy numbers are not reproducible** without the full training pipeline, data,
   and pretrained components.

4. **Training speed comparisons** (TPS metrics) are not meaningful on our small models.

5. **The ELBO integral** is approximated via Monte Carlo sampling over t ∈ [0, 1],
   which introduces variance. The paper may use specific quadrature schemes not detailed.

## Unverified Claims

- The exact forward/reverse process formulation matches the paper's intent
  (only equations are available, not source code)
- Block-attention memory savings match the theoretical O(BL'^2) claim
- Confidence calibration behaves similarly to the paper without proper training
