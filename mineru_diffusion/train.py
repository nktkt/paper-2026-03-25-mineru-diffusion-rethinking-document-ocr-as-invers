"""Complete training pipeline for MinerU-Diffusion.

Implements the paper's three-stage training:
  Stage 0: Modality alignment (vision-language pretraining)
  Stage 1: OCR adaptation (diversity-driven foundational learning)
  Stage 2: Hard-case specialization (uncertainty-driven refinement)

Each stage has its own data, sequence length, batch size, and learning rate.
"""

import argparse
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ModelConfig, DiffusionConfig, CurriculumConfig, PRESET_CONFIGS
from .model import MinerUDiffusion
from .vision_encoder import CNNVisionEncoder
from .data import SyntheticOCRDataset
from .diffusion import forward_process, compute_elbo_loss, sample_timesteps
from .curriculum import compute_inference_consistency, compute_curriculum_weights


def create_model_with_vision(config: ModelConfig, image_size: int = 224) -> nn.Module:
    """Create MinerU-Diffusion with a real CNN vision encoder.

    Args:
        config: Model configuration.
        image_size: Input image size.

    Returns:
        Combined model (vision encoder + diffusion decoder).
    """
    model = MinerUDiffusion(config)
    # Replace stub encoder with CNN
    model.vision_encoder = CNNVisionEncoder(
        in_channels=3,
        hidden_dim=config.hidden_dim,
        num_visual_tokens=64,
    )
    return model


def get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_stage(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: ModelConfig,
    num_epochs: int = 1,
    max_grad_norm: float = 1.0,
    log_interval: int = 10,
    save_dir: Optional[str] = None,
    stage_name: str = "train",
    device: str = "cpu",
    curriculum_config: Optional[CurriculumConfig] = None,
) -> list:
    """Train for one stage.

    Args:
        model: MinerU-Diffusion model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        scheduler: LR scheduler.
        config: Model config.
        num_epochs: Epochs for this stage.
        max_grad_norm: Gradient clipping norm.
        log_interval: Steps between logging.
        save_dir: Checkpoint directory.
        stage_name: Name for logging.
        device: Device string.
        curriculum_config: If set, apply uncertainty-driven weighting.

    Returns:
        List of per-step metrics.
    """
    model.train()
    mask_id = config.diffusion.mask_token_id
    metrics = []
    global_step = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            images = batch["image"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            B, L = input_ids.shape

            # Sample timesteps
            t = sample_timesteps(B, device)

            # Forward diffusion
            x_t = forward_process(input_ids, t, mask_id)

            # Mask out padding from the diffusion process
            x_t = x_t * attn_mask + mask_id * (1 - attn_mask)

            # Model prediction (pass raw images; model handles encoding)
            logits = model(x_t, t, visual_features=images)

            # ELBO loss
            loss = compute_elbo_loss(logits, input_ids, x_t, t, mask_id)

            # Apply curriculum weighting if configured
            if curriculum_config is not None:
                with torch.no_grad():
                    consistency = compute_inference_consistency(
                        model, input_ids,
                        num_passes=curriculum_config.num_stochastic_passes,
                        mask_token_id=mask_id,
                    )
                    weights = compute_curriculum_weights(
                        consistency, curriculum_config.beta
                    )
                loss = loss * weights.mean()

            # Backward + update
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            step_metrics = {
                "stage": stage_name,
                "epoch": epoch,
                "step": global_step,
                "loss": loss.item(),
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "lr": scheduler.get_last_lr()[0],
            }
            metrics.append(step_metrics)

            if global_step % log_interval == 0:
                elapsed = time.time() - t0
                print(
                    f"[{stage_name}] Epoch {epoch+1} Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {step_metrics['lr']:.2e} | "
                    f"Grad: {step_metrics['grad_norm']:.2f} | "
                    f"Time: {elapsed:.1f}s"
                )
                t0 = time.time()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{stage_name}_epoch{epoch}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "stage": stage_name,
                "epoch": epoch,
                "step": global_step,
            }, path)
            print(f"Checkpoint saved: {path}")

    return metrics


def train_full_pipeline(
    config: ModelConfig,
    save_dir: str = "checkpoints",
    device: str = "auto",
    num_samples: int = 1000,
    batch_size: int = 8,
    image_size: int = 224,
):
    """Run the complete 3-stage training pipeline.

    Stage 0: Modality alignment (LR=1e-3, short seqs)
    Stage 1: OCR adaptation (LR=4e-5, medium seqs, diverse data)
    Stage 2: Hard-case specialization (LR=2e-5, long seqs, curriculum)

    Args:
        config: Model configuration.
        save_dir: Directory for checkpoints.
        device: Training device.
        num_samples: Synthetic dataset size per stage.
        batch_size: Training batch size.
        image_size: Input image resolution.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    print(f"Training on: {device}")
    model = create_model_with_vision(config, image_size).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # --- Stage 0: Modality Alignment ---
    print("\n" + "=" * 60)
    print("Stage 0: Modality Alignment")
    print("=" * 60)
    stage0_data = SyntheticOCRDataset(
        num_samples=num_samples, vocab_size=config.vocab_size,
        max_seq_len=min(config.max_seq_len, 128), image_size=image_size,
    )
    stage0_loader = DataLoader(stage0_data, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    total_steps = len(stage0_loader) * 1
    scheduler = get_cosine_schedule(optimizer, int(0.05 * total_steps), total_steps)

    train_stage(
        model, stage0_loader, optimizer, scheduler, config,
        num_epochs=1, stage_name="stage0_alignment",
        save_dir=save_dir, device=device,
    )

    # --- Stage 1: OCR Adaptation ---
    print("\n" + "=" * 60)
    print("Stage 1: OCR Adaptation (Diversity-Driven)")
    print("=" * 60)
    stage1_data = SyntheticOCRDataset(
        num_samples=num_samples, vocab_size=config.vocab_size,
        max_seq_len=min(config.max_seq_len, 256), image_size=image_size,
    )
    stage1_loader = DataLoader(stage1_data, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    total_steps = len(stage1_loader) * 1
    scheduler = get_cosine_schedule(optimizer, int(0.05 * total_steps), total_steps)

    train_stage(
        model, stage1_loader, optimizer, scheduler, config,
        num_epochs=1, stage_name="stage1_ocr_adaptation",
        save_dir=save_dir, device=device,
    )

    # --- Stage 2: Hard-Case Specialization ---
    print("\n" + "=" * 60)
    print("Stage 2: Hard-Case Specialization (Uncertainty-Driven)")
    print("=" * 60)
    curriculum_cfg = CurriculumConfig(
        num_stochastic_passes=3, confidence_threshold=0.8, beta=1.0,
    )
    stage2_data = SyntheticOCRDataset(
        num_samples=max(100, num_samples // 10), vocab_size=config.vocab_size,
        max_seq_len=min(config.max_seq_len, 256), image_size=image_size,
    )
    stage2_loader = DataLoader(stage2_data, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(stage2_loader) * 1
    scheduler = get_cosine_schedule(optimizer, int(0.05 * total_steps), total_steps)

    train_stage(
        model, stage2_loader, optimizer, scheduler, config,
        num_epochs=1, stage_name="stage2_hard_case",
        save_dir=save_dir, device=device,
        curriculum_config=curriculum_cfg,
    )

    # Save final
    final_path = os.path.join(save_dir, "final.pt")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    print("Training complete!")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train MinerU-Diffusion")
    parser.add_argument("--config", type=str, default="tiny", choices=list(PRESET_CONFIGS.keys()))
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=112)
    args = parser.parse_args()

    config = PRESET_CONFIGS[args.config]
    train_full_pipeline(
        config, args.save_dir, args.device,
        args.num_samples, args.batch_size, args.image_size,
    )


if __name__ == "__main__":
    main()
