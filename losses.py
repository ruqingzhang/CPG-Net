"""Joint segmentation, classification, and lesion-balanced score supervision."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F


def _positive_patch_sets(mask: np.ndarray, patch_size: int, ignore_index: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Build one positive patch set per eight-connected lesion component."""
    height, width = mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Mask dimensions must be divisible by the token patch size")
    foreground = mask == 1
    valid = mask != ignore_index
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    patch_h, patch_w = height // patch_size, width // patch_size

    def patch_any(array: np.ndarray) -> np.ndarray:
        return array.reshape(patch_h, patch_size, patch_w, patch_size).any(axis=(1, 3))

    positive_sets = [np.flatnonzero(patch_any(labels == idx).reshape(-1)) for idx in range(1, count + 1)]
    fully_valid = valid.reshape(patch_h, patch_size, patch_w, patch_size).all(axis=(1, 3))
    negative_mask = fully_valid & ~patch_any(foreground)
    return positive_sets, np.flatnonzero(negative_mask.reshape(-1))


def balanced_evidence_loss(
    scores: torch.Tensor, masks: torch.Tensor, *, patch_size: int = 16, ignore_index: int = 255
) -> torch.Tensor:
    """Mean positive BCE within lesions, then equal-weight mean across lesions.

    Negative BCE is averaged across fully valid background patches. Positive
    and negative terms receive equal weight when both are available.
    """
    if scores.ndim != 2 or masks.ndim != 3 or scores.shape[0] != masks.shape[0]:
        raise ValueError("Expected scores [B,N] and binary/ignored masks [B,H,W]")
    if scores.shape[1] != (masks.shape[-1] // patch_size) * (masks.shape[-2] // patch_size):
        raise ValueError("Patch scores and target patch grid do not align")
    mask_arrays = masks.detach().cpu().numpy()
    per_sample = []
    for idx, mask in enumerate(mask_arrays):
        positive_patch_sets, negative_patch_ids = _positive_patch_sets(mask, patch_size, ignore_index)
        terms = []
        lesion_losses = []
        for positive_ids in positive_patch_sets:
            ids = torch.as_tensor(positive_ids, dtype=torch.long, device=scores.device)
            prob = scores[idx].index_select(0, ids).float()
            lesion_losses.append(F.binary_cross_entropy(prob, torch.ones_like(prob)))
        if lesion_losses:
            terms.append(torch.stack(lesion_losses).mean())
        if negative_patch_ids.size:
            ids = torch.as_tensor(negative_patch_ids, dtype=torch.long, device=scores.device)
            prob = scores[idx].index_select(0, ids).float()
            terms.append(F.binary_cross_entropy(prob, torch.zeros_like(prob)))
        per_sample.append(torch.stack(terms).mean() if terms else scores[idx].sum() * 0.0)
    return torch.stack(per_sample).mean()


def segmentation_per_image(
    seg_logits: torch.Tensor, masks: torch.Tensor, *, ignore_index: int = 255
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image masked BCE and soft Dice losses in FP32."""
    if seg_logits.ndim != 4 or seg_logits.shape[1] != 1:
        raise ValueError("Segmentation logits must have shape [B,1,H,W]")
    if seg_logits.shape[0] != masks.shape[0] or seg_logits.shape[-2:] != masks.shape[-2:]:
        raise ValueError("Segmentation predictions and masks have incompatible shapes")
    logits = seg_logits[:, 0].float()
    target = (masks == 1).float()
    valid = (masks != ignore_index).float()
    dims = (-2, -1)
    pixel_count = valid.sum(dim=dims)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum(dim=dims)
    bce = torch.where(pixel_count > 0, bce / pixel_count.clamp_min(1.0), torch.zeros_like(bce))
    probability = torch.sigmoid(logits) * valid
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice = torch.where(pixel_count > 0, dice, torch.zeros_like(dice))
    return bce, dice


class JointCriterion(nn.Module):
    """Per-image BCE + Dice + 0.25 classification CE + 0.5 evidence BCE."""

    def __init__(
        self,
        *,
        classification_weight: float = 0.25,
        evidence_weight: float = 0.5,
        patch_size: int = 16,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if classification_weight < 0 or evidence_weight < 0:
            raise ValueError("Loss weights must be nonnegative")
        self.classification_weight = classification_weight
        self.evidence_weight = evidence_weight
        self.patch_size = patch_size
        self.ignore_index = ignore_index

    def forward(
        self, outputs: dict[str, torch.Tensor], masks: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        seg_bce, seg_dice = segmentation_per_image(
            outputs["seg_logits"], masks, ignore_index=self.ignore_index
        )
        cls = F.cross_entropy(outputs["cls_logits"].float(), labels.long(), reduction="none")
        score_loss = balanced_evidence_loss(
            outputs["scores"],
            masks,
            patch_size=self.patch_size,
            ignore_index=self.ignore_index,
        )
        segmentation = (seg_bce + seg_dice).mean()
        classification = cls.mean()
        total = segmentation + self.classification_weight * classification + self.evidence_weight * score_loss
        return {
            "total": total,
            "segmentation": segmentation,
            "bce": seg_bce.mean(),
            "dice": seg_dice.mean(),
            "classification": classification,
            "evidence": score_loss,
        }
