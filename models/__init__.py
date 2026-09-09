"""Model package for S2Diff."""

from .predictor import CleanHSIPredictor
from .predictor_v2 import SpectralSpatialCleanHSIPredictor
from .predictor_v3 import MSIHighFrequencyGuidedPredictor
from .predictor_v3_ablation import (
    MSIAblationGuidedPredictor,
    VALID_MSI_ABLATIONS,
)
from .predictor_v4_alignment import (
    AlignmentDescriptor,
    CoarseAlignmentDiagnostics,
    DegradationDomainCoarseAligner,
    LearnedGlobalRigidAligner,
    SparseProgressiveLocalAligner,
    StateMatchedCoarseAlignedPredictor,
    spectral_project_hsi,
)

__all__ = [
    "CleanHSIPredictor",
    "SpectralSpatialCleanHSIPredictor",
    "MSIHighFrequencyGuidedPredictor",
    "MSIAblationGuidedPredictor",
    "VALID_MSI_ABLATIONS",
    "AlignmentDescriptor",
    "CoarseAlignmentDiagnostics",
    "DegradationDomainCoarseAligner",
    "LearnedGlobalRigidAligner",
    "SparseProgressiveLocalAligner",
    "StateMatchedCoarseAlignedPredictor",
    "spectral_project_hsi",
]
