"""GPU utilities for memory reporting and device detection."""

from loguru import logger

import torch


def get_device() -> torch.device:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_gpu_count() -> int:
    """Return the number of available CUDA GPUs."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage for all devices."""
    if not torch.cuda.is_available():
        return

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        label = f"{prefix} [GPU {i}]" if prefix else f"[GPU {i}]"
        logger.info(
            f"{label} allocated={allocated:.2f}GB reserved={reserved:.2f}GB "
            f"total={total:.2f}GB"
        )


def get_peak_memory() -> float:
    """Return peak GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def reset_peak_memory() -> None:
    """Reset peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
