"""Small runtime helpers."""

from __future__ import annotations

import random
import os

import numpy as np


def set_devices(visible_devices: str) -> None:
    value = str(visible_devices).strip()
    if value:
        os.environ["CUDA_VISIBLE_DEVICES"] = value


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(requested: str) -> str:
    value = str(requested).lower()
    if value != "auto":
        return value
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def parse_csv(text: str, cast=str) -> list:
    if isinstance(text, (list, tuple)):
        return [cast(item) for item in text]
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]
