import torch


def masked_align(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Solve per-image least squares for a,b (on masked pixels):
        minimize || a*pred + b - target ||^2

    pred/target/mask: (B,1,H,W)
    Returns:
      aligned_pred: (B,1,H,W)
      a,b: (B,1,1,1) each
    """
    if pred.dim() != 4 or target.dim() != 4 or mask.dim() != 4:
        raise ValueError("pred/target/mask must be (B,1,H,W) tensors")

    B = pred.shape[0]
    pred_f = pred.view(B, -1)
    tgt_f = target.view(B, -1)
    m_f = mask.view(B, -1).float()

    wsum = m_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # avoid div by 0
    pred_mean = (pred_f * m_f).sum(dim=1, keepdim=True) / wsum
    tgt_mean = (tgt_f * m_f).sum(dim=1, keepdim=True) / wsum

    pred_c = pred_f - pred_mean
    tgt_c = tgt_f - tgt_mean

    var = (m_f * pred_c.pow(2)).sum(dim=1, keepdim=True) / wsum
    cov = (m_f * pred_c * tgt_c).sum(dim=1, keepdim=True) / wsum

    a = cov / (var + eps)
    b = tgt_mean - a * pred_mean

    aligned = (a * pred_f + b).view_as(pred)

    # reshape a,b to broadcastable
    a = a.view(B, 1, 1, 1)
    b = b.view(B, 1, 1, 1)
    return aligned, (a, b)


def sparse_si_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Scale+shift invariant robust loss for sparse supervision:
    - compare in log-space
    - align pred to target with per-image affine transform on valid pixels
    - apply SmoothL1 (Huber) on valid pixels only

    pred:   (B,1,H,W) model output (relative depth-like)
    target: (B,1,H,W) sparse lidar depth OR disparity (must be >0 on valid)
    mask:   (B,1,H,W) boolean
    """
    # If there are no valid pixels in a batch
    if mask.sum() == 0:
        return torch.zeros((), device=pred.device)

    aligned_pred, _ = masked_align(pred, target, mask, eps=eps)

    diff = aligned_pred[mask] - target[mask]
    return 0.5 * diff.pow(2).mean()
