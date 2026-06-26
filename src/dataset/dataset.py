"""Dataset и DataLoader для torchvision-детекторов (Faster R-CNN, SSD, RetinaNet).

Читает JSON-манифест, подготовленный src/dataset/prepare.py, применяет
аугментации (albumentations) и отдаёт данные в формате torchvision detection:
    image  : FloatTensor [3, H, W] в диапазоне [0, 1]
    target : {"boxes": FloatTensor[N,4] (x1,y1,x2,y2), "labels": LongTensor[N]}

Нормализацию НЕ делаем вручную — torchvision-детекторы нормализуют входы
внутри своих трансформов.

ВАЖНО про id классов: torchvision-детекторы резервируют 0 под фон, поэтому
в target.labels мы сохраняем (cls_id + 1). Число классов модели = len(classes) + 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset


def build_transforms(cfg: dict, train: bool) -> A.Compose:
    """Собирает пайплайн аугментаций. На train — полный, на val — только resize."""
    img_size = cfg["data"]["img_size"]
    acfg = cfg.get("augment", {})

    if train:
        tfs = [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0, fill=(114, 114, 114)),
            A.HorizontalFlip(p=acfg.get("hflip", 0.5)),
            A.RandomBrightnessContrast(p=acfg.get("brightness_contrast", 0.2)),
            A.HueSaturationValue(p=acfg.get("hue_sat", 0.2)),
        ]
        if acfg.get("blur", 0) > 0:
            tfs.append(A.Blur(blur_limit=3, p=acfg["blur"]))
        if acfg.get("scale_rotate", 0) > 0:
            tfs.append(A.ShiftScaleRotate(p=acfg["scale_rotate"], rotate_limit=10))
    else:
        tfs = [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0, fill=(114, 114, 114)),
        ]

    tfs.append(ToTensorV2())
    return A.Compose(
        tfs,
        bbox_params=A.BboxParams(
            format="pascal_voc",  # [x1, y1, x2, y2] в пикселях
            label_fields=["labels"],
            min_visibility=0.2,
        ),
    )


class AircraftDetectionDataset(Dataset):
    """Детекционный датасет поверх JSON-манифеста FGVC-Aircraft."""

    def __init__(self, manifest_path: str | Path, cfg: dict, train: bool):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.tf = build_transforms(cfg, train)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        image = np.array(Image.open(rec["image"]).convert("RGB"))
        boxes = rec["boxes"]
        # +1: класс 0 зарезервирован под фон в torchvision-детекторах
        labels = [c + 1 for c in rec["labels"]]

        out = self.tf(image=image, bboxes=boxes, labels=labels)
        img_t = out["image"].float() / 255.0
        bxs = out["bboxes"]
        lbs = out["labels"]

        if len(bxs) == 0:  # подстраховка: аугментация выкинула бокс
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(bxs, dtype=torch.float32)
            labels_t = torch.as_tensor(lbs, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }
        return img_t, target


def collate_fn(batch):
    """Детекторы принимают списки изображений и таргетов разной формы."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloaders(cfg: dict):
    """Возвращает (train_loader, val_loader) на основе манифестов."""
    processed = Path(cfg["data"]["processed"])
    tcfg = cfg["train"]

    train_ds = AircraftDetectionDataset(processed / "manifest_train.json", cfg, train=True)
    val_ds = AircraftDetectionDataset(processed / "manifest_val.json", cfg, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=tcfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=tcfg["num_workers"] > 0,
    )
    return train_loader, val_loader


def load_classes(cfg: dict) -> list[str]:
    """Список имён классов в порядке id (из classes.txt)."""
    path = Path(cfg["data"]["processed"]) / "classes.txt"
    return path.read_text(encoding="utf-8").splitlines()
