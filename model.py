"""Coverage-Aware Pyramid Guidance Network for joint lesion tasks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

from selection import select_nodes, token_coordinates


@dataclass(frozen=True)
class CPGConfig:
    """Architecture and fixed-budget selection settings."""

    num_classes: int = 115
    image_size: int = 256
    patch_size: int = 16
    vit_dim: int = 384
    vit_depth: int = 12
    vit_heads: int = 6
    fpn_dim: int = 128
    score_hidden: int = 64
    attention_heads: int = 4
    attention_head_dim: int = 32
    neighbors: int = 8
    attention_radius: float = 0.35
    attention_distance_beta: float = 1.0
    budget: int = 64
    context_fraction: float = 0.125
    coverage_radius: float = 0.25
    feature_temperature: float = 0.2
    spatial_sigma: float = 0.125
    classification_dropout: float = 0.1

    def validate(self) -> None:
        if self.image_size < 32 or self.image_size % 32 != 0:
            raise ValueError("image_size must be a multiple of 32 and at least 32")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.patch_size != 16:
            raise ValueError("ResNet-18/ViT-S alignment expects 16-pixel token patches")
        if self.vit_dim % self.vit_heads or self.num_classes <= 1:
            raise ValueError("vit_dim must be divisible by vit_heads; num_classes must exceed 1")
        if self.vit_depth < 1 or self.fpn_dim <= 0 or self.score_hidden <= 0:
            raise ValueError("Network widths and transformer depth must be positive")
        if self.attention_heads != 4 or self.attention_head_dim != 32:
            raise ValueError("Four 32-dimensional local attention heads are required")
        token_count = (self.image_size // self.patch_size) ** 2
        if not 1 <= self.budget <= token_count:
            raise ValueError("Node budget must not exceed the global token count")
        if not 1 <= self.neighbors <= self.budget:
            raise ValueError("neighbors must be between 1 and budget")
        if min(self.attention_radius, self.coverage_radius, self.spatial_sigma, self.feature_temperature) <= 0:
            raise ValueError("Spatial and feature selection settings must be positive")


class ResidualEncoder(nn.Module):
    """Expose ResNet-18 feature maps at strides 4, 8, 16, and 32."""

    def __init__(self) -> None:
        super().__init__()
        network = resnet18(weights=None)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.stage1 = network.layer1
        self.stage2 = network.layer2
        self.stage3 = network.layer3
        self.stage4 = network.layer4

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        level1 = self.stage1(self.stem(images))
        level2 = self.stage2(level1)
        level3 = self.stage3(level2)
        level4 = self.stage4(level3)
        return [level1, level2, level3, level4]


class TransformerBlock(nn.Module):
    """Pre-normalized self-attention and MLP transformer block."""

    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attn = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Linear(dimension * 4, dimension),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class GlobalTokenEncoder(nn.Module):
    """ViT-S/16-style patch encoder with a full 256-token grid at 256 pixels."""

    def __init__(self, config: CPGConfig) -> None:
        super().__init__()
        token_side = config.image_size // config.patch_size
        token_count = token_side**2
        self.patch_embedding = nn.Conv2d(
            3, config.vit_dim, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.positional_embedding = nn.Parameter(torch.zeros(1, token_count, config.vit_dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config.vit_dim, config.vit_heads) for _ in range(config.vit_depth)]
        )
        self.norm = nn.LayerNorm(config.vit_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        patches = patches + self.positional_embedding
        for block in self.blocks:
            patches = block(patches)
        return self.norm(patches)


class PatchScorer(nn.Module):
    """Predict a sigmoid location score from patch-wise mean and maximum features."""

    def __init__(self, input_channels: int, patch_count: int, hidden: int) -> None:
        super().__init__()
        self.patch_count = patch_count
        self.fc1 = nn.Linear(input_channels * 2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, finest_features: torch.Tensor) -> torch.Tensor:
        extent = (self.patch_count, self.patch_count)
        mean_features = F.adaptive_avg_pool2d(finest_features, extent)
        maximum_features = F.adaptive_max_pool2d(finest_features, extent)
        descriptor = torch.cat((mean_features, maximum_features), dim=1)
        descriptor = descriptor.flatten(2).transpose(1, 2)
        return torch.sigmoid(self.fc2(F.gelu(self.fc1(descriptor))).squeeze(-1))


class LocalNodeInteraction(nn.Module):
    """Coordinate-restricted four-head attention from complete feature grids."""

    def __init__(self, local_channels: int, global_channels: int, config: CPGConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.attention_head_dim
        self.neighbors = config.neighbors
        self.radius = config.attention_radius
        self.distance_beta = config.attention_distance_beta
        projected = self.heads * self.head_dim
        self.query = nn.Linear(local_channels, projected)
        self.key = nn.Linear(global_channels, projected)
        self.value = nn.Linear(global_channels, projected)
        self.output = nn.Linear(projected, local_channels)

    def forward(
        self,
        local_grid: torch.Tensor,
        selected_tokens: torch.Tensor,
        selected_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = local_grid.shape
        local_flat = local_grid.flatten(2).transpose(1, 2)
        positions = token_coordinates(height, device=local_grid.device)
        if height != width:
            raise ValueError("Local feature grids must be square")
        distances = torch.cdist(
            positions[None].expand(batch, -1, -1).float(),
            selected_coordinates.float(),
            p=2,
        )
        closest_distance, nearest = torch.topk(
            distances, k=self.neighbors, dim=-1, largest=False, sorted=True
        )
        valid = closest_distance <= self.radius
        has_neighbor = valid.any(dim=-1)
        q = self.query(local_flat).view(batch, height * width, self.heads, self.head_dim)
        k = self.key(selected_tokens).view(batch, -1, self.heads, self.head_dim)
        v = self.value(selected_tokens).view(batch, -1, self.heads, self.head_dim)
        batch_ids = torch.arange(batch, device=local_grid.device)[:, None, None]
        gathered_k = k[batch_ids, nearest]
        gathered_v = v[batch_ids, nearest]
        logits = torch.einsum("bphd,bpnhd->bpnh", q, gathered_k) / math.sqrt(self.head_dim)
        logits = logits - self.distance_beta * closest_distance[..., None].to(logits.dtype)
        logits = logits.masked_fill(~valid[..., None], -1.0e4)
        weights = torch.softmax(logits.float(), dim=2).to(logits.dtype)
        weights = weights * valid[..., None].to(weights.dtype)
        context = torch.einsum("bpnh,bpnhd->bphd", weights, gathered_v)
        messages = self.output(context.reshape(batch, height * width, -1))
        # The gate suppresses both messages and output biases for empty neighborhoods.
        updates = messages * has_neighbor[..., None].to(messages.dtype)
        return (local_flat + updates).transpose(1, 2).reshape(batch, channels, height, width)


class PyramidDecoder(nn.Module):
    """Top-down 128-channel feature pyramid and two task-specific heads."""

    def __init__(self, config: CPGConfig) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(channels, config.fpn_dim, 1) for channels in (64, 128, 256, 512)]
        )
        self.smooth = nn.ModuleList(
            [nn.Conv2d(config.fpn_dim, config.fpn_dim, 3, padding=1) for _ in range(4)]
        )
        self.segmentation = nn.Conv2d(config.fpn_dim, 1, kernel_size=1)
        self.classification = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(config.classification_dropout),
            nn.Linear(config.fpn_dim, config.num_classes),
        )

    def forward(self, multi_scale: list[torch.Tensor], output_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        projected = [layer(tensor) for layer, tensor in zip(self.lateral, multi_scale)]
        pyramid = [torch.empty(0)] * 4
        pyramid[3] = self.smooth[3](projected[3])
        for idx in (2, 1, 0):
            propagated = F.interpolate(pyramid[idx + 1], size=projected[idx].shape[-2:], mode="nearest")
            pyramid[idx] = self.smooth[idx](projected[idx] + propagated)
        seg_logits = self.segmentation(pyramid[0])
        seg_logits = F.interpolate(
            seg_logits, size=(output_size, output_size), mode="bilinear", align_corners=False
        )
        cls_logits = self.classification(pyramid[0])
        return seg_logits, cls_logits


class CPGNet(nn.Module):
    """Full dual-encoder architecture with sparse post-encoding interaction."""

    def __init__(self, config: CPGConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.local_encoder = ResidualEncoder()
        self.global_encoder = GlobalTokenEncoder(config)
        self.scorer = PatchScorer(
            input_channels=64,
            patch_count=config.image_size // config.patch_size,
            hidden=config.score_hidden,
        )
        self.interaction = nn.ModuleList(
            [LocalNodeInteraction(channels, config.vit_dim, config) for channels in (64, 128, 256, 512)]
        )
        self.decoder = PyramidDecoder(config)

    def forward(self, images: torch.Tensor, *, selection_mode: str = "coverage") -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1:] != (
            3, self.config.image_size, self.config.image_size
        ):
            raise ValueError(f"Expected [B,3,{self.config.image_size},{self.config.image_size}]")
        local_maps = self.local_encoder(images)
        global_tokens = self.global_encoder(images)
        # Computing scores and selection in FP32 avoids BF16 ranking artifacts.
        with torch.autocast(device_type=images.device.type, enabled=False):
            scores = self.scorer(local_maps[0].float())
            selected_indices = select_nodes(
                scores,
                global_tokens.float(),
                budget=self.config.budget,
                context_fraction=self.config.context_fraction,
                mode=selection_mode,
                feature_temperature=self.config.feature_temperature,
                spatial_sigma=self.config.spatial_sigma,
                coverage_radius=self.config.coverage_radius,
            )
        batch_indices = torch.arange(images.shape[0], device=images.device)[:, None]
        selected_tokens = global_tokens[batch_indices, selected_indices]
        full_coordinates = token_coordinates(
            self.config.image_size // self.config.patch_size, device=images.device
        )
        selected_coordinates = full_coordinates[selected_indices]
        refined = [
            module(level, selected_tokens, selected_coordinates)
            for module, level in zip(self.interaction, local_maps)
        ]
        segmentation, classification = self.decoder(refined, self.config.image_size)
        return {
            "seg_logits": segmentation,
            "cls_logits": classification,
            "scores": scores,
            "selected_indices": selected_indices,
        }

    def architecture_config(self) -> dict:
        return asdict(self.config)
