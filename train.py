"""Five-seed training, validation selection, checkpoints, and test reporting."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch

from data import LesionManifestDataset, make_loader
from losses import JointCriterion
from metrics import evaluate_model
from model import CPGConfig, CPGNet

DEFAULT_SEEDS = (11, 23, 37, 53, 71)


def set_reproducibility(seed: int, *, strict: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_one_epoch(
    model: CPGNet,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: JointCriterion,
    *,
    device: torch.device,
    epoch: int,
    warmup_epochs: int,
    accumulation_steps: int,
    use_bf16: bool,
    selection_mode: str = "coverage",
) -> dict[str, float]:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    active_selection = "fps" if epoch <= warmup_epochs else selection_mode
    totals = {"total": 0.0, "segmentation": 0.0, "classification": 0.0, "evidence": 0.0}
    count = 0
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    pending_samples = 0
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            outputs = model(images, selection_mode=active_selection)
        with torch.autocast(device_type=device.type, enabled=False):
            losses = criterion(outputs, masks, labels)
        # Accumulate sample-weighted gradients; normalize by actual group size.
        (losses["total"] * images.shape[0]).backward()
        pending += 1
        pending_samples += images.shape[0]
        for key in totals:
            totals[key] += float(losses[key].detach().item()) * images.shape[0]
        count += images.shape[0]
        if pending == accumulation_steps:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(pending_samples)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            pending_samples = 0
    if pending:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(pending_samples)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {key: value / count for key, value in totals.items()}


def mean_std(reports: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    if not reports:
        return {}
    output = {}
    for key in reports[0]:
        if key in ("seed", "selected_epoch", "samples", "small_component_count"):
            continue
        values = [float(report[key]) for report in reports if key in report]
        if len(values) != len(reports):
            continue
        output[key] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return output


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CPG-Net lesion segmentation/classification network")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV manifest with image, mask, label, split")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/cpgnet"))
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--selection-mode", choices=("coverage", "topk", "fps", "all"), default="coverage")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--augment", action="store_true", help="Enable optional random left-right/up-down flips")
    parser.add_argument("--mask-threshold", type=int, default=None)
    parser.add_argument("--mask-ignore-value", type=int, default=None)
    parser.add_argument("--small-component-max-area", type=int, default=64)
    parser.add_argument("--component-hit-fraction", type=float, default=0.10)
    parser.add_argument("--component-coverage-radius", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vit-dim", type=int, default=384)
    parser.add_argument("--vit-depth", type=int, default=12)
    parser.add_argument("--vit-heads", type=int, default=6)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--context-fraction", type=float, default=0.125)
    parser.add_argument("--coverage-radius", type=float, default=0.25)
    parser.add_argument("--attention-radius", type=float, default=0.35)
    parser.add_argument("--attention-distance-beta", type=float, default=1.0)
    parser.add_argument("--evidence-weight", type=float, default=0.5)
    parser.add_argument("--classification-weight", type=float, default=0.25)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.weight_decay < 0 or args.warmup_epochs < 0:
        raise ValueError("Check epochs, learning rate, weight decay, and warmup epochs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no compatible GPU is available")
    args.output.mkdir(parents=True, exist_ok=True)
    architecture = CPGConfig(
        num_classes=args.num_classes,
        image_size=args.image_size,
        vit_dim=args.vit_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        budget=args.budget,
        context_fraction=args.context_fraction,
        coverage_radius=args.coverage_radius,
        attention_radius=args.attention_radius,
        attention_distance_beta=args.attention_distance_beta,
    )
    architecture.validate()
    dataset_args = {
        "manifest": args.manifest,
        "num_classes": args.num_classes,
        "image_size": args.image_size,
        "data_root": args.data_root,
        "mask_threshold": args.mask_threshold,
        "mask_ignore_value": args.mask_ignore_value,
    }
    training = LesionManifestDataset(**dataset_args, split="train", augment=args.augment)
    validation = LesionManifestDataset(**dataset_args, split="val")
    try:
        testing = LesionManifestDataset(**dataset_args, split="test")
    except ValueError as error:
        if "No records found" not in str(error):
            raise
        testing = None
    print(json.dumps({"train": len(training), "val": len(validation), "test": len(testing) if testing else 0}))
    metric_args = {
        "small_component_max_area": args.small_component_max_area,
        "component_hit_fraction": args.component_hit_fraction,
        "coverage_radius": args.component_coverage_radius,
    }
    all_tests = []
    for seed in args.seeds:
        set_reproducibility(seed, strict=args.strict_determinism)
        run_folder = args.output / f"seed_{seed}"
        run_folder.mkdir(parents=True, exist_ok=True)
        train_loader = make_loader(
            training,
            batch_size=args.batch_size,
            seed=seed,
            shuffle=True,
            workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = make_loader(
            validation,
            batch_size=args.batch_size,
            seed=seed,
            workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        test_loader = (
            make_loader(testing, batch_size=args.batch_size, seed=seed, workers=args.workers,
                        pin_memory=device.type == "cuda")
            if testing is not None else None
        )
        model = CPGNet(architecture).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = JointCriterion(
            classification_weight=args.classification_weight,
            evidence_weight=args.evidence_weight,
        )
        best_val = float("inf")
        best_epoch = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            training_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device=device, epoch=epoch,
                warmup_epochs=args.warmup_epochs, accumulation_steps=args.accumulation_steps,
                use_bf16=args.bf16, selection_mode=args.selection_mode,
            )
            val_result = evaluate_model(
                model, val_loader, device=device, metric_config=metric_args, use_bf16=args.bf16,
                selection_mode=args.selection_mode,
            )
            record = {"epoch": epoch, "train": training_loss, "validation": val_result}
            history.append(record)
            if val_result["segmentation_loss"] < best_val:
                best_val = val_result["segmentation_loss"]
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(architecture),
                        "seed": seed,
                        "epoch": epoch,
                        "best_val_seg_loss": best_val,
                        "mask_threshold": args.mask_threshold,
                        "mask_ignore_value": args.mask_ignore_value,
                        "metric_config": metric_args,
                        "selection_mode": args.selection_mode,
                    },
                    run_folder / "best.pt",
                )
            print(
                f"seed={seed} epoch={epoch:03d} train={training_loss['total']:.4f} "
                f"val_seg={val_result['segmentation_loss']:.4f} val_IoU={val_result['iou']:.2f} "
                f"best_epoch={best_epoch}"
            )
            (run_folder / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if test_loader is not None:
            checkpoint = torch.load(run_folder / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"])
            test_result = evaluate_model(
                model, test_loader, device=device, metric_config=metric_args, use_bf16=args.bf16,
                selection_mode=args.selection_mode,
            )
            test_result["seed"] = seed
            test_result["selected_epoch"] = best_epoch
            (run_folder / "test_metrics.json").write_text(
                json.dumps(test_result, indent=2), encoding="utf-8"
            )
            all_tests.append(test_result)
            print(json.dumps({"test_seed": seed, **test_result}))
    if all_tests:
        summary = mean_std(all_tests)
        (args.output / "five_seed_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
