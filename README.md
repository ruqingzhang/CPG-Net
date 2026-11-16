# 🍃 CPG-Net

**Coverage-Aware Pyramid Guidance Network for Joint Segmentation and Classification of Dispersed Plant Lesions**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
![Tasks](https://img.shields.io/badge/Tasks-Segmentation%20%7C%20Classification-405E6A)
![Validation](https://img.shields.io/badge/Software%20Tests-26%20Passed-3C7F5D)

> **Research implementation.** Lesion-balanced supervision · Spatial task-coverage selection · Four-scale local/global interaction · Joint lesion segmentation and disease classification.

**Contents:** [Description](#1-description) · [Dataset Information](#2-dataset-information) · [Code Information](#3-code-information) · [Requirements](#4-requirements) · [Methodology](#5-methodology) · [Usage Instructions](#6-usage-instructions) · [Evaluation](#7-evaluation-and-output-files) · [Reproducibility](#8-reproducibility-and-implementation-notes) · [Citations](#9-citations) · [License & Contributions](#10-license-and-contribution-guidelines)

---

## 1. Description

Dispersed plant lesions challenge fixed-budget interaction schemes because large lesion regions may dominate patch-level supervision, while score-only node selection can repeatedly sample nearby locations. **CPG-Net** combines lesion-balanced location-score learning, greedy marginal-coverage selection, and coordinate-guided feature aggregation. At the default 256 × 256 resolution, independent ResNet-18 and ViT-S/16 encoders produce four local feature grids and 256 global tokens. A deterministic set of eight context nodes is complemented by 56 coverage-driven selections; the resulting 64-node set guides four local feature scales before a feature pyramid predicts lesion masks and disease categories. The implementation includes five-seed training, evaluation, checkpointing, controlled selection alternatives, and executable regression tests.

> **Reproducibility scope.** This release implements the method specified in the CPG-Net manuscript. Choices not fully determined by the manuscript are documented in [Implementation Notes](#implementation-notes). The 26 automated checks validate software behavior, not numerical reproduction of the reported PlantSeg and RiceSeg results.

## 2. Dataset Information

The dataset loader consumes a CSV manifest. Each image and lesion mask must be **spatially aligned**. Class labels are **zero-based integers** with a consistent label mapping across training, validation, and testing. The manifest controls the split; this code never changes the declared split assignments.

```csv
image,mask,label,split
images/leaf_001.jpg,masks/leaf_001.png,0,train
images/leaf_002.jpg,masks/leaf_002.png,1,val
images/leaf_003.jpg,masks/leaf_003.png,0,test
```

Relative paths are resolved against the manifest directory, unless `--data-root` specifies an alternative base. Each mask is resized with nearest-neighbor interpolation, while RGB images use bilinear interpolation. Standard ImageNet channel normalization is applied to RGB inputs. Binary 0/1 and 0/255 mask conventions are recognized automatically; grayscale masks can use `--mask-threshold`. Use `--mask-ignore-value` only when that raw pixel value explicitly denotes ignored regions; for ordinary 0/255 lesion masks, **do not mark 255 as ignored**.

| Dataset | Images | Train | Validation | Test | Disease classes |
|:--|--:|--:|--:|--:|--:|
| PlantSeg v7 | 7,774 | 5,442 | 777 | 1,555 | 115 |
| RiceSeg-5932 | 5,932 | 4,152 | 593 | 1,187 | 4 |

Data sources identified in the manuscript:

- **PlantSeg v7:** [Zenodo dataset](https://doi.org/10.5281/zenodo.17719108)
- **RiceSeg-5932 masks (version 1):** [Mendeley Data](https://doi.org/10.17632/92jc6w6mcy.1)
- **Rice leaf RGB images (version 2):** [Mendeley Data](https://doi.org/10.17632/fwcj7stb8r.2)

For RiceSeg, pair the version-1 masks with the corresponding version-2 RGB images **before** constructing the manifest. Validate alignment and class-ID mappings independently; this implementation does not infer pairings from ambiguous filenames.

**Dataset validation.** Inspect each image/mask pair before training, ensure disease labels use the same encoding in all splits, and check that the number of CSV rows matches the intended experimental split. The program does not download either dataset or determine which RGB image corresponds to a given segmentation mask.

## 3. Code Information

Exactly **seven Python source files** implement the network and its experimental pipeline.

```text
CPG-Net/
├── model.py                 ResNet-18, ViT-S/16, patch scoring, attention, FPN
├── selection.py             Deterministic FPS and greedy marginal coverage
├── losses.py                Component-balanced evidence and joint-task losses
├── data.py                  Aligned RGB/mask datasets and deterministic loaders
├── metrics.py               Segmentation, lesion detection, class F1, benchmarking
├── train.py                 Five-seed training, validation, and checkpointing
├── evaluate.py              Independent evaluation and self-test suite
├── manifest.example.csv     Required dataset record format
├── requirements.txt         Python dependencies
└── README.md                Documentation and verification record
```

### Model inputs and outputs

| Interface | Tensor shape at 256 × 256 | Meaning |
|:--|:--|:--|
| Input `images` | `[B, 3, 256, 256]` | Normalized RGB input batch |
| Output `seg_logits` | `[B, 1, 256, 256]` | Binary lesion logits; apply `sigmoid` and threshold for masks |
| Output `cls_logits` | `[B, C]` | Disease-class logits; use `argmax` for predicted class |
| Output `scores` | `[B, 256]` | Sigmoid location scores on the 16 × 16 token grid |
| Output `selected_indices` | `[B, 64]` | Distinct selected token indices in the default configuration |

`B` is batch size and `C` is the dataset-specific disease-class count. See `CPGConfig` in `model.py` for architecture options, `train.py --help` for training options, and `evaluate.py --help` for evaluation options.

## 4. Requirements

**Recommended:** Python 3.11, a recent PyTorch/CUDA combination, and an NVIDIA GPU for full-resolution multi-seed experiments. CPU execution is supported for checks and reduced-scale development. The source manuscript reports experiments with Python 3.11.11, PyTorch 2.7.1+cu128, and an RTX 5090D; this release was functionally validated in a CPU-only PyTorch environment, not on that specific GPU.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with the local CUDA driver if GPU acceleration is required; see the [official PyTorch installer](https://pytorch.org/get-started/locally/).

**Dependencies** are declared in `requirements.txt`: `torch>=2.3`, `torchvision>=0.18`, `numpy>=1.26`, `scipy>=1.11`, `Pillow>=10.0`, and `scikit-learn>=1.4`. A compatible PyTorch/Torchvision pairing is required; on CUDA machines, select the wheels for the installed GPU driver.

Verify the environment:

```bash
python -c "import torch, torchvision; print('PyTorch:', torch.__version__, '| Torchvision:', torchvision.__version__, '| CUDA:', torch.cuda.is_available())"
```

## 5. Methodology

```text
                         Input RGB image [B, 3, 256, 256]
                               /                 \
                              /                   \
              ResNet-18 local encoder          ViT-S/16 global encoder
              64² → 32² → 16² → 8²              16 × 16 = 256 tokens
                       |                            |
                       +→ Patch mean/max → score q  |
                                       |            |
                          Balanced score supervision
                                 (training only)
                                       |            |
                             Coverage selection ←--+
                      8 context nodes + 56 greedy nodes
                                   64 nodes
                                       |
                          Four-scale local-to-node attention
                           64², 32², 16², and 8²
                                       |
                            128-channel feature pyramid
                                    /       \
                                   /         \
                        Binary lesion mask   Disease class
```

**Lesion-balanced scoring.** Let \(P_m\) denote patches overlapping connected lesion \(m\). Positive binary-cross-entropy losses are averaged inside each lesion, then equally averaged across lesions. Negative losses are averaged over fully valid background patches. When both positive and negative supervision exist, the two terms receive equal weight. This prevents a larger lesion from receiving greater *positive-supervision weight solely because it occupies more patches*.

**Task–spatial coverage.** For tokens \(z_i\) with normalized coordinates \(p_i\) and location scores \(q_i\), the truncated kernel and set utility are

\[
\kappa_{ij}=\exp\left(-\frac{1-\cos(z_i,z_j)}{\tau_f}\right)
\exp\left(-\frac{\|p_i-p_j\|_2^2}{2\tau_p^2}\right)
\mathbf{1}\left[\|p_i-p_j\|_2\le r_{\mathrm{cov}}\right],
\qquad
F(S)=\sum_i q_i\max_{j\in S}(q_j\kappa_{ij}).
\]

Starting with deterministic farthest-point-sampled context nodes, selection repeatedly adds the candidate with the greatest \(F(S\cup\{j\})-F(S)\). Gradients do not propagate through the discrete selection indices. When all remaining gains are negligible, a spatial farthest-point fallback fills the budget.

**Local interaction and joint objective.** Every local grid position remains present and attends to its nearest eight selected nodes within normalized radius \(r_{\mathrm{att}}\). Attention combines projected content similarity with a spatial-distance penalty. A residual update and a top-down 128-channel FPN support both output heads. The optimization objective is

\[
\mathcal{L}=\mathcal{L}_{\mathrm{BCE}}+\mathcal{L}_{\mathrm{Dice}}+
0.25\,\mathcal{L}_{\mathrm{cls}}+0.5\,\mathcal{L}_{\mathrm{ev}}.
\]

### Method-to-code correspondence

| Research component | Implementation | Computation |
|:--|:--|:--|
| Lesion-balanced score supervision | `losses.py`, `PatchScorer` in `model.py` | Patch-wise BCE, averaged per connected lesion and across lesions |
| Spatial task-coverage selection | `selection.py` | FPS context seeds followed by greedy marginal coverage gains |
| Four-scale interaction | `LocalNodeInteraction` in `model.py` | Nearest selected node attention with a distance penalty and residual update |
| Joint prediction and optimization | `PyramidDecoder` in `model.py`, `JointCriterion` in `losses.py` | FPN, binary segmentation, disease classification, and weighted losses |

## 6. Usage Instructions

**Implementation sequence:** (1) install dependencies, (2) acquire and align the dataset images and masks, (3) create a CSV manifest with `train`, `val`, and `test` entries, (4) train a dataset-specific model, and (5) evaluate its selected checkpoint. All commands below run from the `CPG-Net/` directory.

### 6.1 Train models

**PlantSeg (115 classes)**

```bash
python train.py \
  --manifest /path/to/plantseg_manifest.csv \
  --num-classes 115 \
  --output runs/plantseg
```

**RiceSeg-5932 (4 classes)**

```bash
python train.py \
  --manifest /path/to/riceseg_manifest.csv \
  --num-classes 4 \
  --output runs/riceseg
```

Default full-resolution settings follow the specified manuscript values:

| Setting | Value | Setting | Value |
|:--|:--|:--|:--|
| Input resolution | 256 × 256 | Global tokens | 256 |
| Local encoder | ResNet-18 | Global encoder | ViT-S/16-style, 12 blocks / 6 heads |
| Node budget \(K\) | 64 | Context fraction \(\rho\) | 0.125 (8 nodes) |
| Feature temperature \(\tau_f\) | 0.2 | Spatial kernel scale \(\tau_p\) | 0.125 |
| Coverage kernel radius | 0.25 | Attention radius | 0.35 |
| Nearest interaction nodes | 8 | Local attention | 4 heads × 32 dimensions |
| FPN width | 128 | Classification dropout | 0.1 |
| Optimizer | AdamW | Learning rate | 0.0001 |
| Weight decay | 0.01 | Effective batch size | 16 |
| Training | 150 epochs | FPS warm-up | 5 epochs |
| Seeds | 11, 23, 37, 53, 71 | Checkpoint criterion | Lowest validation segmentation loss |

All epochs use the full joint loss, including the score-supervision term. The warm-up changes **only the selection rule**: FPS is used for epochs 1–5, and task-weighted coverage selection is used thereafter. Validation always uses the configured post-warm-up selection rule. With the default command, that rule is `coverage`.

On smaller GPUs, gradient accumulation preserves the intended effective batch size:

```bash
python train.py \
  --manifest /path/to/plantseg_manifest.csv \
  --num-classes 115 \
  --batch-size 4 \
  --accumulation-steps 4 \
  --workers 4 \
  --output runs/plantseg_accumulated
```

Augmentation is **off by default**, since the manuscript does not specify a complete augmentation pipeline. Optional horizontal/vertical flips can be enabled with `--augment` and should be disclosed when reporting experimental outcomes. `--strict-determinism` requests deterministic PyTorch operations where the backend supports them.

### 6.2 Evaluate checkpoints and run comparisons

Evaluate the best checkpoint on the held-out test split:

```bash
python evaluate.py \
  --checkpoint runs/plantseg/seed_11/best.pt \
  --manifest /path/to/plantseg_manifest.csv \
  --split test \
  --output runs/plantseg/seed_11/independent_test.json
```

Measure batch-one end-to-end latency and memory using the manuscript's **100 warm-up + 500 timed passes** protocol:

```bash
python evaluate.py \
  --checkpoint runs/plantseg/seed_11/best.pt \
  --manifest /path/to/plantseg_manifest.csv \
  --benchmark \
  --benchmark-warmup 100 \
  --benchmark-passes 500
```

Change only the selection rule while retaining checkpoint weights:

```bash
python evaluate.py \
  --checkpoint runs/plantseg/seed_11/best.pt \
  --manifest /path/to/plantseg_manifest.csv \
  --selection-mode topk
```

Use all 256 global tokens for the **all-node comparison**, bypassing the greedy selection loop:

```bash
python evaluate.py \
  --checkpoint runs/plantseg/seed_11/best.pt \
  --manifest /path/to/plantseg_manifest.csv \
  --budget 256 \
  --selection-mode all \
  --benchmark
```

To retrain a Top-K baseline with the same network, pass `--selection-mode topk` to `train.py`. To test other budgets, use `--budget` during training or independent evaluation. The `all` mode requires a budget equal to the token count.

## 7. Evaluation and Output Files

### Evaluated metrics

- **Segmentation:** foreground IoU, Dice converted from aggregate IoU, BF1 with a two-pixel tolerance.
- **Classification:** Macro-F1 across the configured disease-label set.
- **Lesion-level:** small-component recall, equal-component coverage, and false-positive area ratio.
- **Runtime:** average latency, throughput, parameter count, and CUDA peak allocated memory where available.
- **Training:** per-epoch losses, validation metrics, and a checkpoint chosen by validation segmentation loss.

Each seed produces `history.json`, `best.pt`, and `test_metrics.json`; the training command also writes `five_seed_summary.json` with mean and **sample** standard deviation. The parameter/latency report is measured locally. The paper's particular main-operation GMAC accounting is **not asserted** by this implementation.

### Checkpoint and run layout

```text
runs/plantseg/
├── seed_11/
│   ├── best.pt                 Best validation segmentation-loss checkpoint
│   ├── history.json            Per-epoch training and validation records
│   └── test_metrics.json       Held-out test measurements
├── seed_23/
├── seed_37/
├── seed_53/
├── seed_71/
└── five_seed_summary.json      Cross-seed means and sample standard deviations
```

For experiments with no `test` entries in the manifest, `test_metrics.json` and `five_seed_summary.json` are not written. Metrics are percentages unless noted otherwise. The independent evaluation command prints JSON to standard output; pass `--output` to save the result to a file.

## 8. Reproducibility and Implementation Notes

### Software verification

Run the regression suite with no external image dataset:

```bash
python evaluate.py --self-test
```

Three development validation/revision rounds were recorded in the preceding implementation; the full 26-check suite was also rerun successfully when revising this README:

| Round | Checks | Outcome | Changes / additional verification |
|:--|:--:|:--|:--|
| 1 · Architecture & supervision | 24 | Passed | Verified tensor shapes, unique node indices, loss gradients, masks, perfect-prediction metrics, and checkpoint reload |
| 2 · Runtime & optimization | 24 | Passed | Vectorized coverage selection across batches, cached FPS initialization, and corrected sample-weighted gradient accumulation |
| 3 · Protocol & integration | 26 | Passed | Separated metric radius from selection radius; tested regular-grid 75% geometric coverage; added all-node mode; executed command-line train/evaluate and full 256 × 256 inference |

The end-to-end integration run used a **small artificial dataset and a reduced transformer** to check execution and checkpoint correctness. Its metric values are not experimental claims. Full-resolution inference was additionally checked with 256 tokens and 64 unique selected nodes. **Training on the actual PlantSeg and RiceSeg datasets and validation on the manuscript's GPU were not performed here.**

### Implementation notes

Several details are under-specified in the manuscript and therefore require explicit implementation choices:

1. **Initialization and ViT mechanics.** The two encoders are initialized from scratch. The transformer uses learned patch-position embeddings, 12 pre-normalized attention blocks, and 256 spatial patch tokens without a classification token. External pretrained weights are not silently assumed.
2. **FPN and classification head.** Four lateral projections and spatial refinement convolutions form a 128-channel top-down FPN. Global average pooling over the finest fused feature map supplies the classification head. These decoder choices can affect model size and accuracy.
3. **Local spatial penalty.** The manuscript defines the attention distance coefficient \(\beta\) but does not give its default numeric value; this release uses \(\beta=1\) and exposes `--attention-distance-beta`.
4. **Component evaluation.** Because the source does not specify the threshold for small components or the criterion for a component hit, defaults are **≤64 target pixels** and **≥10% component overlap**. Both are configurable. Component coverage is computed over GT-positive token patches using a radius of **one token-grid spacing** (1/16 at the default resolution), with equal weight per GT component; the metric radius is independently configurable via `--component-coverage-radius`. It is distinct from the kernel's 0.25 selection radius.
5. **Pixel preprocessing.** Resizing, RGB channel normalization, binary-mask thresholding, and optional flips are defined here so experiments can be repeated; they should not be assumed identical to any unspecified original preprocessing.
6. **Parameter accounting.** This default 115-class implementation contains approximately **34.243 million parameters**, compared with **33.908 million** reported in the manuscript. The difference is consistent with implementation choices that are not completely defined by the paper; this release does not claim bitwise or numerical identity with the original experiment.

These choices are documented to support auditability. They should be fixed across controlled comparisons and stated in any resulting publication.

### Reproducibility checklist

- [ ] Verify RGB/mask alignment and zero-based disease-label mapping.
- [ ] Preserve the declared train/validation/test assignments.
- [ ] Record the exact dependency versions, accelerator, and CUDA driver.
- [ ] Fix preprocessing, node budget, selection rule, and component definitions.
- [ ] Run the five listed seeds and retain each best checkpoint.
- [ ] Report mean ± sample standard deviation, with the same evaluation conventions across baselines.
- [ ] Distinguish measured GPU runtime from model-operation estimates.

---

<div align="center">

**CPG-Net · Reproducible implementation · Transparent experimental scope**

</div>

## 9. Citations

If the code contributes to a publication, cite the method manuscript:

> *Coverage-Aware Pyramid Guidance Network for Joint Segmentation and Classification of Dispersed Plant Lesions.* Full author, venue, year, and DOI metadata should be taken from the published paper when available; they are not assigned in this repository.

The datasets are separate scholarly resources. Cite the versions actually used and comply with the dataset providers' terms. The following BibTeX entries follow the sources listed in the manuscript:

```bibtex
@article{wei2026plantseg,
  author  = {Wei, T. and Chen, Z. and Yu, X. and Chapman, S. and Melloy, P. and Huang, Z.},
  title   = {A Large-Scale In-the-wild Dataset for Plant Disease Segmentation},
  journal = {Scientific Data},
  volume  = {13},
  pages   = {205},
  year    = {2026},
  doi     = {10.1038/s41597-025-06513-4}
}

@dataset{wei2025plantsegv7,
  author    = {Wei, T. and Chen, Z. and Yu, X. and Chapman, S. and Melloy, P. and Huang, Z.},
  title     = {A Large-Scale In-the-wild Dataset for Plant Disease Segmentation},
  publisher = {Zenodo},
  version   = {7},
  year      = {2025},
  doi       = {10.5281/zenodo.17719108}
}

@dataset{zinsou2025riceseg5932,
  author    = {Zinsou, K. M. S. and Houndji, V. R.},
  title     = {RiceSeg-5932: Complete Pixel-Level Segmentation Masks for Rice Leaf Disease Images Samples},
  publisher = {Mendeley Data},
  version   = {1},
  year      = {2025},
  doi       = {10.17632/92jc6w6mcy.1}
}

@dataset{sethy2024riceleaf,
  author    = {Sethy, P. K.},
  title     = {Rice Leaf Disease Image Samples},
  publisher = {Mendeley Data},
  version   = {2},
  year      = {2024},
  doi       = {10.17632/fwcj7stb8r.2}
}
```

## 10. License and Contribution Guidelines

### License

**No explicit software license is included in this distribution.** The project owner should choose a license and add a `LICENSE` file before public redistribution. This README does not grant additional software-use or redistribution permissions. PlantSeg and RiceSeg data are supplied by their original providers, and the terms attached to those datasets must be respected separately.

### Contribution guidelines

Contributions are welcome if a public development repository and contribution channel are established. Proposed changes should:

1. Identify the component being changed and explain the methodological or engineering reason.
2. Preserve data-split integrity, input/output contracts, and documented default parameters unless a change is explicitly justified.
3. Include focused regression tests and run `python evaluate.py --self-test` before submission.
4. Document any new dependencies, hyperparameters, metrics, or deviations from the manuscript.
5. Keep source-code comments, docstrings, commit descriptions, and public technical documentation in English.

Changes to dataset handling should be tested with aligned sample pairs and do not include redistribution of dataset files without the rights holder's permission. Please report reproducibility discrepancies with the manifest schema, package versions, device information, command line, random seed, and observed error or metric.

---

**CPG-Net · Joint lesion learning · Reproducible engineering documentation**
