"""End-to-end demo of MinerU-Diffusion components."""

import torch
from mineru_diffusion import (
    ModelConfig, PRESET_CONFIGS, MinerUDiffusion,
    forward_process, compute_elbo_loss, sample_timesteps,
    create_block_attention_mask,
    AdaptiveScheduler, SchedulerConfig,
    shuffle_words, create_benchmark_dataset, evaluate_robustness,
)
from mineru_diffusion.curriculum import (
    compute_inference_consistency, select_hard_cases, compute_curriculum_weights,
)


def main():
    print("=" * 70)
    print("MinerU-Diffusion: Document OCR via Diffusion Decoding - Demo")
    print("=" * 70)

    # 1. Block-attention mask
    print("\n--- Block-Attention Mask (block_size=4, seq_len=12) ---")
    mask = create_block_attention_mask(12, block_size=4)
    for i in range(12):
        row = "".join(["#" if mask[i, j] == 1 else "." for j in range(12)])
        print(f"  pos {i:2d}: {row}")

    # 2. Forward diffusion
    print("\n--- Discrete Diffusion Forward Process ---")
    x_0 = torch.randint(1, 100, (1, 16))
    print(f"  x_0: {x_0[0].tolist()}")
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = torch.tensor([t_val])
        x_t = forward_process(x_0, t, mask_token_id=0)
        n_masked = (x_t == 0).sum().item()
        print(f"  t={t_val:.2f}: {x_t[0].tolist()} ({n_masked}/{16} masked)")

    # 3. Model forward + loss
    print("\n--- Model Forward Pass ---")
    config = PRESET_CONFIGS["tiny"]
    model = MinerUDiffusion(config)
    print(f"  Config: {config.num_layers}L, {config.hidden_dim}D, "
          f"block_size={config.block_size}")
    print(f"  Parameters: {model.count_parameters():,}")

    x_0 = torch.randint(1, config.vocab_size, (4, 32))
    result = model.compute_loss(x_0)
    print(f"  Loss: {result['loss']:.4f}")
    print(f"  Masked tokens: {result['num_masked']:.0f}")

    # 4. Training demo
    print("\n--- Training (20 steps) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(20):
        result = model.compute_loss(x_0)
        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()
        if step % 5 == 0:
            print(f"  Step {step}: loss={result['loss']:.4f}")

    # 5. Generation
    print("\n--- Generation (adaptive scheduler) ---")
    output = model.generate(seq_len=32, batch_size=1, num_steps=10,
                            confidence_threshold=0.95)
    print(f"  Generated: {output[0].tolist()}")
    print(f"  No [MASK] tokens: {(output != 0).all()}")

    # 6. Scheduler demo
    print("\n--- Adaptive Scheduler ---")
    for tau in [0.5, 0.8, 0.95, 0.99]:
        sched = AdaptiveScheduler(SchedulerConfig(confidence_threshold=tau, max_steps=20))
        state = sched.create_state(seq_len=32, batch_size=1)
        confirmed = torch.zeros(1, 32, dtype=torch.bool)
        steps = 0
        for _ in range(20):
            probs = torch.randn(1, 32, config.vocab_size).softmax(dim=-1)
            newly, _, state = sched.step(probs, confirmed, state)
            confirmed = confirmed | newly
            steps += 1
            if sched.should_stop(state):
                break
        print(f"  tau={tau}: {steps} steps, "
              f"{state.fraction_confirmed:.1%} confirmed")

    # 7. Curriculum learning
    print("\n--- Curriculum Learning (uncertainty mining) ---")
    consistency = compute_inference_consistency(model, x_0[:2], num_passes=3)
    print(f"  Consistency scores: {consistency.tolist()}")
    hard = select_hard_cases(consistency, threshold=0.8)
    print(f"  Hard cases: {hard.tolist()}")
    weights = compute_curriculum_weights(consistency, beta=1.0)
    print(f"  Curriculum weights: {weights.tolist()}")

    # 8. Semantic Shuffle benchmark
    print("\n--- Semantic Shuffle Benchmark ---")
    texts = [
        "The neural network achieved state-of-the-art results on multiple benchmarks",
        "Attention mechanisms have revolutionized natural language processing",
    ]
    samples = create_benchmark_dataset(texts, distortion_levels=[0.0, 0.5, 1.0])
    for s in samples[:6]:
        print(f"  d={s.distortion_level:.1f}: \"{s.shuffled_text[:60]}...\"")

    # Mock evaluation
    preds = [s.shuffled_text for s in samples[:6]]
    refs = [s.original_text for s in samples[:6]]
    metrics = evaluate_robustness(preds, refs)
    print(f"  Char accuracy: {metrics['char_accuracy']:.1%}")
    print(f"  Exact match: {metrics['exact_match']:.1%}")

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    main()
