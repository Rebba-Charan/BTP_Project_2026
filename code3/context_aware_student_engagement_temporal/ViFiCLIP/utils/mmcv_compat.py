"""Small compatibility layer for the MMCV APIs used by ViFiCLIP.

The original project only uses a limited set of MMCV utilities. Keeping these
helpers local avoids the legacy MMCV package, which has no reliable Python
3.13 wheel.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class Registry:
    def __init__(self, name: str):
        self.name = name
        self.module_dict: dict[str, type] = {}

    def register_module(self, module=None, name: str | None = None):
        def register(cls):
            key = name or cls.__name__
            self.module_dict[key] = cls
            return cls

        return register(module) if module is not None else register

    def get(self, key: str):
        return self.module_dict.get(key)


def build_from_cfg(cfg: dict[str, Any], registry: Registry, default_args=None):
    args = dict(cfg)
    obj_type = args.pop("type")
    obj_cls = registry.get(obj_type) if isinstance(obj_type, str) else obj_type
    if obj_cls is None:
        raise KeyError(f"{obj_type!r} is not registered in {registry.name!r}")
    if default_args:
        for key, value in default_args.items():
            args.setdefault(key, value)
    return obj_cls(**args)


def load(filename: str):
    path = Path(filename)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return json.loads(path.read_text())
    with path.open("rb") as stream:
        return pickle.load(stream)


def dump(obj, filename: str):
    path = Path(filename)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(obj, indent=2))
    else:
        with path.open("wb") as stream:
            pickle.dump(obj, stream)
    return filename


def is_str(value) -> bool:
    return isinstance(value, str)


def is_tuple_of(value, expected_type) -> bool:
    return isinstance(value, tuple) and all(
        isinstance(item, expected_type) for item in value
    )


def imresize(img, size, interpolation=cv2.INTER_LINEAR):
    if isinstance(interpolation, str):
        mode = interpolation.lower()
        interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }.get(mode)
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation mode: {mode!r}")
    return cv2.resize(img, tuple(map(int, size)), interpolation=int(interpolation))


def imflip_(img, direction="horizontal"):
    if direction == "horizontal":
        img[...] = np.flip(img, axis=1)
    elif direction == "vertical":
        img[...] = np.flip(img, axis=0)
    elif direction == "diagonal":
        img[...] = np.flip(np.flip(img, axis=0), axis=1)
    else:
        raise ValueError(f"Unsupported flip direction: {direction}")
    return img


def iminvert(img):
    return 255 - img


def imnormalize_(img, mean, std, to_bgr=False):
    if to_bgr:
        img[...] = img[..., ::-1]
    img[...] = (img - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    return img


def rescale_size(old_size, scale):
    old_w, old_h = old_size
    if isinstance(scale, (tuple, list)):
        target_w, target_h = scale
        ratio = min(target_w / old_w, target_h / old_h)
    else:
        ratio = scale
    return int(old_w * ratio + 0.5), int(old_h * ratio + 0.5)


class FileClient:
    def __init__(self, backend="disk", **kwargs):
        if backend != "disk":
            raise ValueError(f"Only the disk backend is supported, got {backend!r}")

    def get(self, filename: str) -> bytes:
        return Path(filename).read_bytes()
