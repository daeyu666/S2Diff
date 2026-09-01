"""Model package for S2Diff."""

from .predictor import CleanHSIPredictor
from .predictor_v2 import SpectralSpatialCleanHSIPredictor

__all__ = [
    "CleanHSIPredictor",
    "SpectralSpatialCleanHSIPredictor",
]
