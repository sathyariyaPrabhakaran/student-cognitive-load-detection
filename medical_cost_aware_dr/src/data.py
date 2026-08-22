from __future__ import annotations

from pathlib import Path
from collections import Counter
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RetinaDataset(Dataset):
    def __init__(self, root: str | Path, samples: list[tuple[str, int]], train: bool = False):
        self.root = Path(root)
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(self.root / path).convert("RGB")
        return self.transform(image), label, path


def discover(root: str | Path):
    root = Path(root)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(classes)}
    samples = []
    for cls in classes:
        for path in (root / cls).rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                samples.append((str(path.relative_to(root)), class_to_idx[cls]))
    if not samples:
        raise ValueError(f"No images found below {root}")
    return samples, classes


def stratified_split(samples, seed=42, train_ratio=.70, val_ratio=.15):
    rng = random.Random(seed)
    by_class = {}
    for item in samples:
        by_class.setdefault(item[1], []).append(item)
    train, val, test = [], [], []
    for items in by_class.values():
        rng.shuffle(items)
        n = len(items)
        a, b = int(n * train_ratio), int(n * (train_ratio + val_ratio))
        train.extend(items[:a]); val.extend(items[a:b]); test.extend(items[b:])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test
