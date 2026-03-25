"""OCR inference pipeline for MinerU-Diffusion.

Implements the full inference flow:
1. Load image -> vision encoder -> visual features
2. Initialize [MASK] sequence
3. Iterative denoising with confidence-based adaptive scheduling
4. Decode tokens to text/layout/formula/table

Supports the paper's task-specific prompt templates:
- Text Recognition
- Formula Recognition
- Table Recognition
- Layout Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .config import ModelConfig, SchedulerConfig
from .model import MinerUDiffusion
from .scheduler import AdaptiveScheduler
from .tokenizer import OCRTokenizer
from .diffusion import get_masked_positions


TASK_PROMPTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
    "layout": "Layout Detection:",
}


class OCRPipeline:
    """End-to-end OCR inference pipeline.

    Handles image encoding, diffusion generation with adaptive scheduling,
    and token decoding back to text.
    """

    def __init__(
        self,
        model: MinerUDiffusion,
        tokenizer: OCRTokenizer,
        scheduler_config: Optional[SchedulerConfig] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.scheduler = AdaptiveScheduler(scheduler_config or SchedulerConfig())
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        task: str = "text",
        max_seq_len: int = 512,
        num_steps: int = 15,
    ) -> str:
        """Run OCR on a single image.

        Args:
            image: Input image [channels, height, width] or [1, C, H, W].
            task: Task type ("text", "formula", "table", "layout").
            max_seq_len: Maximum output sequence length.
            num_steps: Maximum denoising steps.

        Returns:
            Decoded text string.
        """
        self.model.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        # Pass raw images directly; model.forward handles encoding
        output_ids = self._generate_with_scheduler(
            image, max_seq_len, num_steps,
        )

        # Decode
        text = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        return text

    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
        task: str = "text",
        max_seq_len: int = 512,
        num_steps: int = 15,
    ) -> List[str]:
        """Run OCR on a batch of images.

        Args:
            images: Input images [batch, channels, height, width].
            task: Task type.
            max_seq_len: Maximum output sequence length.
            num_steps: Maximum denoising steps.

        Returns:
            List of decoded text strings.
        """
        self.model.eval()
        images = images.to(self.device)

        output_ids = self._generate_with_scheduler(
            images, max_seq_len, num_steps,
        )

        results = []
        for i in range(output_ids.shape[0]):
            text = self.tokenizer.decode(output_ids[i].tolist(), skip_special_tokens=True)
            results.append(text)

        return results

    def _generate_with_scheduler(
        self,
        vis_features: torch.Tensor,
        seq_len: int,
        max_steps: int,
    ) -> torch.Tensor:
        """Generate tokens using adaptive confidence-based scheduling.

        Args:
            vis_features: Visual features [batch, num_vis, hidden_dim].
            seq_len: Length of sequence to generate.
            max_steps: Maximum denoising steps.

        Returns:
            Generated token IDs [batch, seq_len].
        """
        B = vis_features.shape[0]
        mask_id = self.model.config.diffusion.mask_token_id

        # Start fully masked
        x = torch.full((B, seq_len), mask_id, dtype=torch.long, device=self.device)
        confirmed = torch.zeros(B, seq_len, dtype=torch.bool, device=self.device)

        state = self.scheduler.create_state(seq_len, B)

        for step in range(max_steps):
            t_val = self.scheduler.get_timestep(state)
            t = torch.full((B,), t_val, device=self.device)
            state.step = step

            logits = self.model(x, t, visual_features=vis_features)
            probs = F.softmax(logits, dim=-1)

            newly_confirmed, predicted, state = self.scheduler.step(
                probs, confirmed, state,
            )

            # Update confirmed tokens
            confirmed = confirmed | newly_confirmed
            x[confirmed] = predicted[confirmed]

            if self.scheduler.should_stop(state):
                break

        # Fill remaining with argmax
        remaining = ~confirmed
        if remaining.any():
            t_final = torch.full((B,), 0.01, device=self.device)
            logits = self.model(x, t_final, visual_features=vis_features)
            final_pred = logits.argmax(dim=-1)
            x[remaining] = final_pred[remaining]

        return x

    @torch.no_grad()
    def get_confidence_map(
        self,
        image: torch.Tensor,
        max_seq_len: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get per-token confidence scores for an image.

        Useful for debugging and visualizing model uncertainty.

        Args:
            image: Input image [C, H, W] or [1, C, H, W].
            max_seq_len: Sequence length.

        Returns:
            Tuple of (token_ids [seq_len], confidences [seq_len]).
        """
        self.model.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        mask_id = self.model.config.diffusion.mask_token_id
        x = torch.full((1, max_seq_len), mask_id, dtype=torch.long, device=self.device)
        t = torch.tensor([0.01], device=self.device)

        logits = self.model(x, t, visual_features=image)
        probs = F.softmax(logits, dim=-1)
        confidences, tokens = probs.max(dim=-1)

        return tokens[0], confidences[0]


def load_pipeline(
    checkpoint_path: str,
    device: str = "cpu",
    confidence_threshold: float = 0.95,
) -> OCRPipeline:
    """Load a trained model and create an inference pipeline.

    Args:
        checkpoint_path: Path to checkpoint .pt file.
        device: Device for inference.
        confidence_threshold: Scheduler confidence threshold.

    Returns:
        Configured OCRPipeline.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MinerUDiffusion(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    tokenizer = OCRTokenizer(config.vocab_size)
    scheduler_config = SchedulerConfig(confidence_threshold=confidence_threshold)

    return OCRPipeline(model, tokenizer, scheduler_config, device)
