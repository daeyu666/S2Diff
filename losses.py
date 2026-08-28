# losses.py
import torch
import torch.nn as nn


class SAMLoss(nn.Module):
    """Numerically stable Spectral Angle Mapper loss.

    Inputs are BxCxHxW hyperspectral tensors.  The previous acos(cosine)
    implementation used eps=1e-8 inside a float32 clamp.  Because
    float32(1 - 1e-8) == 1, nearly collinear spectra could reach acos(1):
    the forward value is finite but its derivative is infinite, which can
    poison the predictor weights with NaN after the first optimizer step.

    We instead compute the exact angle between normalized spectra via

        angle(u, v) = 2 * atan2(||u-v||_2, ||u+v||_2)

    for unit vectors u and v.  This form is well behaved at zero angle and
    does not require an artificial angular clamp/floor.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()

        pred_norm = torch.linalg.vector_norm(
            pred, ord=2, dim=1, keepdim=True
        ).clamp_min(self.eps)
        target_norm = torch.linalg.vector_norm(
            target, ord=2, dim=1, keepdim=True
        ).clamp_min(self.eps)

        pred_unit = pred / pred_norm
        target_unit = target / target_norm

        chord = torch.linalg.vector_norm(
            pred_unit - target_unit, ord=2, dim=1
        )
        anti_chord = torch.linalg.vector_norm(
            pred_unit + target_unit, ord=2, dim=1
        ).clamp_min(self.eps)

        angle = 2.0 * torch.atan2(chord, anti_chord)
        return angle.mean()
