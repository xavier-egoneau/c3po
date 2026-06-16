"""Primitives expérimentales image -> observation structurée."""

from .gemma4 import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    VisionObservationResult,
    observe_image,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT",
    "VisionObservationResult",
    "observe_image",
]
