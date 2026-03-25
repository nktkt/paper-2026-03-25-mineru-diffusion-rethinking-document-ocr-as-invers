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

__version__ = "0.1.0"
