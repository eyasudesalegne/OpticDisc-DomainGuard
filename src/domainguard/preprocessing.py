from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def square_fundus_center_crop(image: np.ndarray) -> np.ndarray:
    """Auditable fallback crop used when no validated disc localization exists."""
    h, w = image.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def bbox_crop(image: np.ndarray, bbox, margin: float = 0.20) -> np.ndarray:
    x1, y1, x2, y2 = map(float, bbox)
    h, w = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    xa, xb = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
    ya, yb = int(max(0, cy - side / 2)), int(min(h, cy + side / 2))
    if xb <= xa or yb <= ya:
        raise ValueError("Invalid optic-disc bounding box")
    return image[ya:yb, xa:xb]


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("Mask contains no positive pixels")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def color_normalize_rgb(image: np.ndarray) -> np.ndarray:
    """Deterministic per-channel percentile normalization for the clean reproduction.

    The manuscript specifies color normalization but not an exact formula. This implementation
    is therefore explicitly a clean reproducibility choice, not claimed as archival code.
    """
    x = image.astype(np.float32)
    out = np.empty_like(x)
    for c in range(3):
        lo, hi = np.percentile(x[..., c], [1, 99])
        if hi <= lo:
            out[..., c] = x[..., c]
        else:
            out[..., c] = np.clip((x[..., c] - lo) * 255.0 / (hi - lo), 0, 255)
    return out.astype(np.uint8)


def clahe_luminance(image: np.ndarray, clip_limit: float = 2.0, tile_grid=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def preprocess(image: np.ndarray, branch: str, size: int = 224, bbox=None, mask=None) -> np.ndarray:
    if branch == "raw_resize":
        x = image
    elif branch == "fallback_disc_color_clahe":
        x = clahe_luminance(color_normalize_rgb(square_fundus_center_crop(image)))
    elif branch == "bbox_disc_color_clahe":
        if bbox is None:
            raise ValueError("bbox branch requires a validated bbox")
        x = clahe_luminance(color_normalize_rgb(bbox_crop(image, bbox)))
    elif branch == "mask_disc_color_clahe":
        if mask is None:
            raise ValueError("mask branch requires a validated mask")
        x = clahe_luminance(color_normalize_rgb(bbox_crop(image, mask_bbox(mask))))
    else:
        raise ValueError(f"Unknown preprocessing branch: {branch}")
    return cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
