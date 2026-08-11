from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import timm

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def image_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


def build_frozen_backbone(name: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity()  # 2048-D global-average-pooled representation
    else:
        aliases = {
            "efficientnet_b0": "efficientnet_b0",
            "convnext_tiny": "convnext_tiny",
            "swin_tiny": "swin_tiny_patch4_window7_224",
        }
        if name not in aliases:
            raise ValueError(f"Unsupported torchvision/timm backbone: {name}")
        model = timm.create_model(aliases[name], pretrained=True, num_classes=0, global_pool="avg")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, device


@torch.inference_mode()
def extract_batches(model, tensors, device: str, batch_size: int = 32) -> np.ndarray:
    chunks = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start:start + batch_size]).to(device)
        z = model(batch)
        if z.ndim > 2:
            z = torch.flatten(z, 1)
        chunks.append(z.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def dinov2_note() -> str:
    return (
        "The manuscript uses DINOv2 ViT-S/14 as a frozen sensitivity backbone. "
        "Its exact loading route can vary across torch.hub/timm releases, so this clean "
        "package does not silently substitute a different checkpoint. Pin the exact "
        "checkpoint used for numerical reproduction before enabling this branch."
    )
