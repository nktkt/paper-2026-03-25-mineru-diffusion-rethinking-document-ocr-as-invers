"""MinerU-Diffusion: Document OCR as Inverse Rendering via Diffusion Decoding.

Unofficial implementation of "MinerU-Diffusion" (arXiv:2603.22458).
"""

from .config import ModelConfig, DiffusionConfig, SchedulerConfig, CurriculumConfig, PRESET_CONFIGS
from .diffusion import forward_process, compute_elbo_loss, sample_timesteps
from .block_attention import BlockAttention, create_block_attention_mask, create_block_attention_bias
from .model import MinerUDiffusion
from .scheduler import AdaptiveScheduler, SchedulerState
from .curriculum import compute_inference_consistency, select_hard_cases, compute_curriculum_weights
from .benchmark import shuffle_words, create_benchmark_dataset, evaluate_robustness
from .tokenizer import OCRTokenizer
from .vision_encoder import CNNVisionEncoder, ViTVisionEncoder, DynamicResolutionEncoder
from .losses import (
    edit_distance, normalized_edit_distance, character_error_rate,
    word_error_rate, page_iou, teds_similarity, compute_task_metrics,
)
from .inference import OCRPipeline, load_pipeline
from .train import train_full_pipeline, create_model_with_vision

__version__ = "0.1.0"
