"""Training and inference engine for the progressive degradation framework.

V1/V2 predictors receive only (x_t, t). Innovation 2 V3 declares
``requires_msi=True`` and receives (x_t, HR-MSI, t). The frozen degradation
trajectory and reverse update remain identical across all predictor versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from degradations import ProgressiveDegradation, build_degradation
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from utils import AverageMeter


@dataclass
class Innovation1TrainStats:
    loss: float
    l1: float
    sam: float
    deg: float


def build_progressive_process(cfg) -> ProgressiveDegradation:
    """Build the frozen D_t / U_t / D~_t trajectory from experiment config."""
    mode = getattr(cfg, "degradation_mode", "physical")
    kwargs = {}
    if mode == "physical":
        kwargs.update(
            mtf_nyquist=float(getattr(cfg, "mtf_nyquist", 0.2)),
            truncate=float(getattr(cfg, "psf_truncate", 3.0)),
        )
    elif mode == "gaussian_bicubic":
        kwargs.update(
            sigma=float(getattr(cfg, "gaussian_sigma", 2.0)),
            kernel_size=int(getattr(cfg, "gaussian_kernel_size", 5)),
        )

    operator = build_degradation(
        mode,
        scale_ratio=int(cfg.scale_ratio),
        **kwargs,
    )

    requested_lift = getattr(cfg, "lift_mode", "auto")
    default_lift_mode = None if requested_lift == "auto" else requested_lift
    return ProgressiveDegradation(
        operator=operator,
        total_steps=int(getattr(cfg, "diffusion_steps", 12)),
        default_lift_mode=default_lift_mode,
    )


def batch_state_at(
    process: ProgressiveDegradation,
    x: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Evaluate D~_t(x) for a batch whose samples may use different t."""
    if x.ndim != 4:
        raise ValueError(f"x must be BxCxHxW, got {tuple(x.shape)}")
    if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
        raise ValueError(
            f"timesteps must have shape [B={x.shape[0]}], got {tuple(timesteps.shape)}"
        )

    order = torch.argsort(timesteps)
    inverse = torch.argsort(order)
    x_sorted = x.index_select(0, order)
    t_sorted = timesteps.index_select(0, order)

    outputs = []
    for t_value in torch.unique(t_sorted, sorted=True):
        mask = t_sorted == t_value
        outputs.append(process.state_at(x_sorted[mask], int(t_value.item())))

    out_sorted = torch.cat(outputs, dim=0)
    return out_sorted.index_select(0, inverse)


def degradation_consistency_loss(
    process: ProgressiveDegradation,
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Optional native-sensor-domain consistency loss."""
    total = pred_x0.new_zeros(())
    batch_size = pred_x0.shape[0]

    for t_value in torch.unique(timesteps, sorted=True):
        mask = timesteps == t_value
        count = int(mask.sum().item())
        pred_native = process.degrade_at(pred_x0[mask], int(t_value.item()))
        with torch.no_grad():
            target_native = process.degrade_at(
                target_x0[mask], int(t_value.item())
            )
        group_loss = F.l1_loss(pred_native, target_native)
        total = total + group_loss * (float(count) / float(batch_size))

    return total


def _ensure_finite(
    name: str,
    value: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
) -> None:
    if torch.isfinite(value).all():
        return
    message = f"Non-finite value detected in {name}"
    if timesteps is not None:
        message += f"; timesteps={timesteps.detach().cpu().tolist()}"
    finite = value.detach()[torch.isfinite(value.detach())]
    if finite.numel() > 0:
        message += (
            f"; finite_min={finite.min().item():.6e}"
            f"; finite_max={finite.max().item():.6e}"
        )
    raise FloatingPointError(message)


def model_predict(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    hr_msi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch V1/V2 single-modal or V3 MSI-guided predictor calls."""
    if bool(getattr(model, "requires_msi", False)):
        if hr_msi is None:
            raise ValueError("This predictor requires HR-MSI, but hr_msi is None")
        return model(x_t, hr_msi, timesteps)
    return model(x_t, timesteps)


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    process: ProgressiveDegradation,
    device: torch.device,
    *,
    lambda_l1: float = 1.0,
    lambda_sam: float = 0.1,
    lambda_deg: float = 0.0,
    boundary_probability: float = 0.2,
    boundary_radius: int = 1,
    grad_clip: float = 1.0,
) -> Innovation1TrainStats:
    """Train the selected predictor to directly estimate clean HR-HSI X."""
    model.train()
    sam_loss_fn = SAMLoss()

    loss_meter = AverageMeter()
    l1_meter = AverageMeter()
    sam_meter = AverageMeter()
    deg_meter = AverageMeter()

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = None
        if bool(getattr(model, "requires_msi", False)):
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            _ensure_finite("hr_msi", hr_msi)

        batch_size = gt.shape[0]
        _ensure_finite("gt", gt)

        timesteps = process.sample_timesteps(
            batch_size,
            boundary_probability=boundary_probability,
            boundary_radius=boundary_radius,
            device=device,
        )

        with torch.no_grad():
            x_t = batch_state_at(process, gt, timesteps)
        _ensure_finite("x_t", x_t, timesteps)

        pred_x0 = model_predict(model, x_t, timesteps, hr_msi=hr_msi)
        _ensure_finite("pred_x0", pred_x0, timesteps)

        l1 = F.l1_loss(pred_x0, gt)
        sam = sam_loss_fn(pred_x0, gt)
        _ensure_finite("L1 loss", l1, timesteps)
        _ensure_finite("SAM loss", sam, timesteps)

        if lambda_deg > 0.0:
            deg = degradation_consistency_loss(process, pred_x0, gt, timesteps)
            _ensure_finite("degradation consistency loss", deg, timesteps)
        else:
            deg = pred_x0.new_zeros(())

        loss = lambda_l1 * l1 + lambda_sam * sam + lambda_deg * deg
        _ensure_finite("total loss", loss, timesteps)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        max_norm = (
            float(grad_clip)
            if grad_clip is not None and grad_clip > 0.0
            else float("inf")
        )
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()

        loss_meter.update(loss.item(), batch_size)
        l1_meter.update(l1.item(), batch_size)
        sam_meter.update(sam.item(), batch_size)
        deg_meter.update(deg.item(), batch_size)

    return Innovation1TrainStats(
        loss=loss_meter.avg,
        l1=l1_meter.avg,
        sam=sam_meter.avg,
        deg=deg_meter.avg,
    )


@torch.no_grad()
def reconstruct_from_terminal_lr(
    model: torch.nn.Module,
    process: ProgressiveDegradation,
    lr_hsi: torch.Tensor,
    *,
    target_size: Tuple[int, int],
    hr_msi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the fixed deterministic reverse recursion from an LR observation."""
    model.eval()
    x_t = process.terminal_state(lr_hsi, target_size=target_size)

    for t in range(process.total_steps, 0, -1):
        timestep = torch.full(
            (x_t.shape[0],), t, dtype=torch.long, device=x_t.device
        )
        pred_x0 = model_predict(model, x_t, timestep, hr_msi=hr_msi)
        x_t = process.reverse_update(x_t, pred_x0, t)

    return x_t


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    process: ProgressiveDegradation,
    device: torch.device,
    *,
    scale_ratio: int,
) -> Dict[str, float]:
    """Evaluate the full T->0 reverse process under the selected degradation."""
    model.eval()
    final_meter = MetricAverager()
    init_meter = MetricAverager()

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = None
        if bool(getattr(model, "requires_msi", False)):
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)

        terminal_lr = process.terminal_observation(gt)
        init_state = process.terminal_state(
            terminal_lr, target_size=tuple(gt.shape[-2:])
        )
        pred = reconstruct_from_terminal_lr(
            model,
            process,
            terminal_lr,
            target_size=tuple(gt.shape[-2:]),
            hr_msi=hr_msi,
        )

        final_meter.update(calc_metrics(pred, gt, scale_ratio))
        init_meter.update(calc_metrics(init_state, gt, scale_ratio))

    final_metrics = final_meter.average()
    initial_metrics = init_meter.average()
    final_metrics["INIT_PSNR"] = initial_metrics.get("PSNR", float("nan"))
    final_metrics["INIT_SAM"] = initial_metrics.get("SAM", float("nan"))
    return final_metrics
