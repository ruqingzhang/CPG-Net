"""Checkpoint evaluation, end-to-end benchmarks, and executable regression checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from dataclasses import replace
import tempfile

import numpy as np
from PIL import Image
import torch

from data import LesionManifestDataset, make_loader
from losses import JointCriterion, balanced_evidence_loss, segmentation_per_image
from metrics import MetricAccumulator, benchmark_model, boundary_f1, evaluate_model
from model import CPGConfig, CPGNet
from selection import coverage_kernel, coverage_value, select_nodes, token_coordinates


def run_self_tests() -> dict[str, str]:
    """Run CPU-compatible checks covering geometry, supervision, and training."""
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    torch.manual_seed(37)
    passed: dict[str, str] = {}

    def record(name: str, predicate: bool) -> None:
        if not bool(predicate):
            raise AssertionError(f"Self-test failed: {name}")
        passed[name] = "PASS"
        print(f"[PASS] {name}")

    positions = token_coordinates(4, device=torch.device("cpu"))
    record("row_major_normalized_coordinates", positions.shape == (16, 2) and
           torch.allclose(positions[0], torch.tensor([0.125, 0.125])) and
           torch.allclose(positions[4], torch.tensor([0.125, 0.375])))
    scores = torch.tensor([[0.90, 0.91, 0.10, 0.10, 0.50, 0.20, 0.35, 0.30,
                            0.10, 0.40, 0.70, 0.12, 0.45, 0.15, 0.05, 0.60]])
    features = torch.randn(1, 16, 12)
    for mode in ("fps", "topk", "coverage"):
        choices = select_nodes(scores, features, budget=8, mode=mode)
        record(f"fixed_budget_unique_{mode}", choices.shape == (1, 8) and
               len(torch.unique(choices)) == 8)
        record(f"deterministic_{mode}", torch.equal(
            choices, select_nodes(scores, features, budget=8, mode=mode)
        ))
    record("topk_preserves_score_order", torch.equal(
        select_nodes(scores, features, budget=2, mode="topk"), torch.tensor([[1, 0]])
    ))
    kernel = coverage_kernel(features, positions, coverage_radius=0.25)
    record("kernel_shape_diagonal_cutoff", kernel.shape == (1, 16, 16) and
           torch.allclose(kernel.diagonal(dim1=1, dim2=2), torch.ones(1, 16), atol=1e-5) and
           kernel[0, 0, 15] == 0)
    chosen = select_nodes(scores, features, budget=8, mode="coverage")
    values = [float(coverage_value(chosen[:, :k], scores, kernel)[0]) for k in range(1, 9)]
    record("monotone_coverage_objective", all(
        a <= b + 1e-6 for a, b in zip(values[:-1], values[1:])
    ))
    full_indices = select_nodes(scores, features, budget=16, mode="all")
    record("all_node_no_pruning", torch.equal(full_indices, torch.arange(16)[None]))
    zero_indices = select_nodes(torch.zeros_like(scores), features, budget=8, mode="coverage")
    record("zero_gain_fps_fallback", len(torch.unique(zero_indices)) == 8)

    # A large component covers many patches, but each positive component has equal weight.
    mask = torch.zeros(1, 64, 64, dtype=torch.long)
    mask[:, 0:30, 0:30] = 1
    mask[:, 49:55, 49:55] = 1
    predictions = torch.full((1, 16), 0.5, requires_grad=True)
    evidence = balanced_evidence_loss(predictions, mask, patch_size=16)
    record("component_balanced_loss_finite", torch.isfinite(evidence) and
           abs(float(evidence.detach()) - np.log(2.0)) < 1e-5)
    evidence.backward()
    record("evidence_loss_backpropagates", predictions.grad is not None and
           torch.isfinite(predictions.grad).all())
    ignored = torch.full((1, 64, 64), 255, dtype=torch.long)
    neutral = balanced_evidence_loss(torch.full((1, 16), 0.5, requires_grad=True),
                                     ignored, patch_size=16)
    record("ignored_mask_zero_evidence", float(neutral) == 0.0)
    mask_logits = torch.randn(1, 1, 64, 64)
    bce, dice = segmentation_per_image(mask_logits, ignored)
    record("ignored_mask_zero_segmentation", float(bce[0]) == 0.0 and float(dice[0]) == 0.0)

    config = CPGConfig(num_classes=2, image_size=64, vit_dim=48, vit_depth=1,
                       vit_heads=4, budget=8)
    model = CPGNet(config)
    inputs = torch.randn(2, 3, 64, 64)
    outputs = model(inputs, selection_mode="coverage")
    record("end_to_end_tensor_shapes", outputs["seg_logits"].shape == (2, 1, 64, 64)
           and outputs["cls_logits"].shape == (2, 2) and
           outputs["scores"].shape == (2, 16))
    record("selection_not_detached_from_values", outputs["seg_logits"].requires_grad)
    mixed_mask = torch.zeros(2, 64, 64, dtype=torch.long)
    mixed_mask[0, 16:35, 14:37] = 1
    criterion = JointCriterion(patch_size=16)
    result = criterion(outputs, mixed_mask, torch.tensor([0, 1]))
    result["total"].backward()
    record("all_objectives_finite_gradients", torch.isfinite(result["total"]) and
           model.scorer.fc2.weight.grad is not None and
           model.global_encoder.patch_embedding.weight.grad is not None and
           model.local_encoder.stage1[0].conv1.weight.grad is not None)

    ground_truth = np.zeros((64, 64), dtype=bool)
    ground_truth[10:20, 10:20] = True
    record("perfect_boundary_f1", boundary_f1(ground_truth, ground_truth) == 1.0)
    metric = MetricAccumulator(num_classes=2, patch_size=16, small_component_max_area=200)
    perfect = {
        "seg_logits": torch.tensor(np.where(ground_truth, 15.0, -15.0), dtype=torch.float32)[None, None],
        "cls_logits": torch.tensor([[10.0, -10.0]]),
        "selected_indices": torch.arange(16)[None],
        "scores": torch.full((1, 16), 0.5),
    }
    metric.update(perfect, torch.tensor(ground_truth.astype(np.int64))[None], torch.tensor([0]))
    perfect_values = metric.compute()
    record("perfect_segmentation_and_coverage", all(
        abs(float(perfect_values[key]) - 100) < 1e-6
        for key in ("iou", "dice", "bf1", "component_coverage", "small_component_recall")
    ))
    uniform_grid = MetricAccumulator(num_classes=2, patch_size=4)
    dense_truth = torch.ones(1, 64, 64, dtype=torch.long)
    regular_positions = [row * 16 + col for row in range(0, 16, 2) for col in range(0, 16, 2)]
    grid_outputs = {
        "seg_logits": torch.full((1, 1, 64, 64), 15.0),
        "cls_logits": torch.tensor([[10.0, -10.0]]),
        "selected_indices": torch.tensor([regular_positions]),
        "scores": torch.full((1, 256), 0.5),
    }
    uniform_grid.update(grid_outputs, dense_truth, torch.tensor([0]))
    record("one_step_coverage_grid_geometry", abs(
        uniform_grid.compute()["component_coverage"] - 75.0
    ) < 1e-5)

    with tempfile.TemporaryDirectory(prefix="cpgnet-check-") as folder:
        root = Path(folder)
        rows = []
        for idx, split in enumerate(("train", "train", "val", "test")):
            rgb = np.full((64, 64, 3), 80 + 20 * idx, dtype=np.uint8)
            binary = np.zeros((64, 64), dtype=np.uint8)
            if idx != 2:
                binary[12:24, 8 + idx : 20 + idx] = 255
            Image.fromarray(rgb).save(root / f"image_{idx}.png")
            Image.fromarray(binary).save(root / f"mask_{idx}.png")
            rows.append({"image": f"image_{idx}.png", "mask": f"mask_{idx}.png",
                         "label": str(idx % 2), "split": split})
        with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "mask", "label", "split"])
            writer.writeheader()
            writer.writerows(rows)
        train_dataset = LesionManifestDataset(root / "manifest.csv", split="train", num_classes=2, image_size=64)
        val_dataset = LesionManifestDataset(root / "manifest.csv", split="val", num_classes=2, image_size=64)
        record("manifest_pairing_and_binary_masks", len(train_dataset) == 2 and
               int(train_dataset[0]["mask"].sum()) == 144 and
               int(val_dataset[0]["mask"].sum()) == 0)
        train_loader = make_loader(train_dataset, batch_size=2, seed=11)
        val_loader = make_loader(val_dataset, batch_size=1, seed=11)
        from train import train_one_epoch

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = model.scorer.fc2.weight.detach().clone()
        training_result = train_one_epoch(
            model, train_loader, optimizer, criterion, device=torch.device("cpu"),
            epoch=1, warmup_epochs=5, accumulation_steps=1, use_bf16=False
        )
        record("one_step_training_and_weight_update", training_result["total"] > 0 and
               not torch.equal(before, model.scorer.fc2.weight.detach()))
        validation = evaluate_model(model, val_loader, device=torch.device("cpu"),
                                    metric_config={"small_component_max_area": 64})
        record("validation_pipeline_finite", np.isfinite(validation["segmentation_loss"]) and
               validation["samples"] == 1)
        checkpoint_path = root / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "config": model.architecture_config()}, checkpoint_path)
        restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        replica = CPGNet(CPGConfig(**restored["config"]))
        replica.load_state_dict(restored["model"])
        model.eval()
        replica.eval()
        with torch.no_grad():
            first = model(inputs)["seg_logits"]
            second = replica(inputs)["seg_logits"]
        record("checkpoint_round_trip", torch.allclose(first, second, atol=1e-6))
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CPG-Net checkpoints")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--selection-mode", choices=("coverage", "topk", "fps", "all"), default=None)
    parser.add_argument("--budget", type=int, default=None, help="Override node count without changing trained weights")
    parser.add_argument("--benchmark-warmup", type=int, default=100)
    parser.add_argument("--benchmark-passes", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        results = run_self_tests()
        print(json.dumps({"passed": len(results), "tests": results}, indent=2))
        return
    if args.checkpoint is None or args.manifest is None:
        parser.error("--checkpoint and --manifest are required unless --self-test is used")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = CPGConfig(**checkpoint["config"])
    if args.budget is not None:
        config = replace(config, budget=args.budget)
    model = CPGNet(config).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = LesionManifestDataset(
        args.manifest,
        split=args.split,
        num_classes=config.num_classes,
        image_size=config.image_size,
        data_root=args.data_root,
        mask_threshold=checkpoint.get("mask_threshold"),
        mask_ignore_value=checkpoint.get("mask_ignore_value"),
    )
    loader = make_loader(dataset, batch_size=args.batch_size, workers=args.workers, seed=11,
                         pin_memory=device.type == "cuda")
    token_count = (config.image_size // config.patch_size) ** 2
    selection_mode = args.selection_mode or (
        "all" if config.budget == token_count else checkpoint.get("selection_mode", "coverage")
    )
    results = evaluate_model(
        model, loader, device=device, use_bf16=args.bf16,
        metric_config=checkpoint.get("metric_config"), selection_mode=selection_mode,
    )
    results["selection_mode"] = selection_mode
    results["node_budget"] = config.budget
    if args.benchmark:
        results["benchmark"] = benchmark_model(
            model, device=device, image_size=config.image_size, warmup=args.benchmark_warmup,
            passes=args.benchmark_passes, use_bf16=args.bf16, selection_mode=selection_mode,
        )
    print(json.dumps(results, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
