"""Lesion-level, pixel-level, classification, and runtime evaluation."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy import ndimage
import torch
from sklearn.metrics import f1_score

from losses import segmentation_per_image
from selection import token_coordinates


def boundary_f1(prediction: np.ndarray, ground_truth: np.ndarray, tolerance: int = 2) -> float:
    """Symmetric boundary matching with a two-pixel spatial tolerance."""
    structure = np.ones((3, 3), dtype=np.uint8)
    pred_edge = prediction & ~ndimage.binary_erosion(prediction, structure=structure)
    truth_edge = ground_truth & ~ndimage.binary_erosion(ground_truth, structure=structure)
    count_pred = int(pred_edge.sum())
    count_truth = int(truth_edge.sum())
    if count_pred == count_truth == 0:
        return 1.0
    if count_pred == 0 or count_truth == 0:
        return 0.0
    truth_near = ndimage.binary_dilation(truth_edge, iterations=tolerance)
    pred_near = ndimage.binary_dilation(pred_edge, iterations=tolerance)
    precision = float((pred_edge & truth_near).sum()) / count_pred
    recall = float((truth_edge & pred_near).sum()) / count_truth
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


class MetricAccumulator:
    """Accumulate fixed-protocol metrics without averaging per-batch IoU."""

    def __init__(
        self,
        *,
        num_classes: int,
        patch_size: int = 16,
        coverage_radius: float | None = None,
        small_component_max_area: int = 64,
        component_hit_fraction: float = 0.10,
        boundary_tolerance: int = 2,
        ignore_index: int = 255,
    ) -> None:
        if not 0 < component_hit_fraction <= 1:
            raise ValueError("component_hit_fraction must lie in (0,1]")
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.coverage_radius = coverage_radius
        if coverage_radius is not None and coverage_radius <= 0:
            raise ValueError("coverage_radius must be positive when specified")
        self.small_component_max_area = small_component_max_area
        self.component_hit_fraction = component_hit_fraction
        self.boundary_tolerance = boundary_tolerance
        self.ignore_index = ignore_index
        self.intersection = 0
        self.union = 0
        self.false_positive = 0
        self.background_pixels = 0
        self.boundary_scores: list[float] = []
        self.class_truth: list[int] = []
        self.class_predictions: list[int] = []
        self.component_coverages: list[float] = []
        self.small_hits = 0
        self.small_total = 0
        self.sample_count = 0

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        predicted_masks = (torch.sigmoid(outputs["seg_logits"].detach().float()) >= 0.5)
        predicted_masks = predicted_masks[:, 0].cpu().numpy()
        target_masks = masks.detach().cpu().numpy()
        selected_indices = outputs["selected_indices"].detach().cpu().numpy()
        predicted_classes = torch.argmax(outputs["cls_logits"].detach(), dim=1).cpu().tolist()
        true_classes = labels.detach().cpu().tolist()
        side = int(math.isqrt(outputs["scores"].shape[1]))
        if side * side != outputs["scores"].shape[1]:
            raise ValueError("Selected tokens must form a square grid")
        coordinates = token_coordinates(side, device=torch.device("cpu")).numpy()

        for prediction, target, selected, truth_class, pred_class in zip(
            predicted_masks, target_masks, selected_indices, true_classes, predicted_classes
        ):
            valid = target != self.ignore_index
            truth = target == 1
            pred = prediction & valid
            self.intersection += int((pred & truth).sum())
            self.union += int((pred | truth).sum())
            background = valid & ~truth
            self.false_positive += int((pred & background).sum())
            self.background_pixels += int(background.sum())
            self.boundary_scores.append(
                boundary_f1(pred, truth, tolerance=self.boundary_tolerance)
            )
            self.class_truth.append(int(truth_class))
            self.class_predictions.append(int(pred_class))
            self.sample_count += 1
            components, count = ndimage.label(truth, structure=np.ones((3, 3), dtype=np.uint8))
            height, width = target.shape
            patch_size = self.patch_size
            if height % patch_size or width % patch_size:
                raise ValueError("Metric masks must align with the token patch grid")
            selected_coordinates = coordinates[selected]
            for component_id in range(1, count + 1):
                component = components == component_id
                area = int(component.sum())
                if area <= self.small_component_max_area:
                    overlap = int((pred & component).sum())
                    self.small_hits += int(overlap / area >= self.component_hit_fraction)
                    self.small_total += 1
                patch_mask = component.reshape(
                    height // patch_size, patch_size, width // patch_size, patch_size
                ).any(axis=(1, 3)).reshape(-1)
                occupied_coordinates = coordinates[patch_mask]
                squared_distance = ((occupied_coordinates[:, None] - selected_coordinates[None]) ** 2).sum(-1)
                evaluation_radius = self.coverage_radius if self.coverage_radius is not None else 1.0 / side
                covered = (squared_distance.min(axis=1) <= evaluation_radius**2 + 1e-10).mean()
                self.component_coverages.append(float(covered))

    def compute(self) -> dict[str, float | int]:
        iou = self.intersection / self.union if self.union else 1.0
        dice = 2.0 * iou / (1.0 + iou)
        return {
            "samples": self.sample_count,
            "iou": 100.0 * iou,
            "dice": 100.0 * dice,
            "bf1": 100.0 * (float(np.mean(self.boundary_scores)) if self.boundary_scores else 0.0),
            "macro_f1": 100.0 * float(
                f1_score(
                    self.class_truth,
                    self.class_predictions,
                    labels=list(range(self.num_classes)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "small_component_recall": 100.0 * self.small_hits / self.small_total if self.small_total else 0.0,
            "small_component_count": self.small_total,
            "component_coverage": 100.0 * float(np.mean(self.component_coverages)) if self.component_coverages else 0.0,
            "false_positive_area_ratio": 100.0 * self.false_positive / self.background_pixels if self.background_pixels else 0.0,
        }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    metric_config: dict[str, Any] | None = None,
    use_bf16: bool = False,
    selection_mode: str = "coverage",
) -> dict[str, float | int]:
    """Evaluate the full dataset under a fixed node-selection rule."""
    model.eval()
    metrics = MetricAccumulator(num_classes=model.config.num_classes, **(metric_config or {}))
    accumulated_loss = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and use_bf16)
        ):
            outputs = model(images, selection_mode=selection_mode)
        bce, dice = segmentation_per_image(outputs["seg_logits"], masks)
        accumulated_loss += float((bce + dice).sum().item())
        count += images.shape[0]
        metrics.update(outputs, masks, labels)
    results = metrics.compute()
    results["segmentation_loss"] = accumulated_loss / count if count else float("nan")
    return results


@torch.no_grad()
def benchmark_model(
    model: torch.nn.Module,
    *,
    device: torch.device,
    image_size: int = 256,
    warmup: int = 100,
    passes: int = 500,
    use_bf16: bool = True,
    selection_mode: str = "coverage",
) -> dict[str, float]:
    """Batch-one end-to-end wall-clock inference with optional GPU peak memory."""
    if warmup < 0 or passes < 1:
        raise ValueError("Benchmark requires nonnegative warmup and positive passes")
    model.eval()
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    autocast_enabled = device.type == "cuda" and use_bf16
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            model(sample, selection_mode=selection_mode)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(passes):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            model(sample, selection_mode=selection_mode)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000.0 / passes
    measurement = {
        "latency_ms": latency_ms,
        "throughput_samples_per_second": 1000.0 / latency_ms,
        "parameters_millions": sum(p.numel() for p in model.parameters()) / 1e6,
    }
    if device.type == "cuda":
        measurement["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
    return measurement
