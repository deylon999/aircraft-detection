"""Обёртка над моделями ultralytics: YOLOv8 и RT-DETR.

У этих моделей собственный высокоуровневый цикл обучения (.train) и
валидации (.val), поэтому они обучаются отдельно от torchvision-детекторов,
но на ТОМ ЖЕ датасете (data/processed/yolo/data.yaml) и с теми же классами.
"""
from __future__ import annotations

# Имя модели -> предобученные веса. Размеры подобраны под 8 ГБ VRAM.
ULTRALYTICS_WEIGHTS = {
    "yolov8": "yolov8s.pt",   # small: хороший баланс скорость/качество
    "rtdetr": "rtdetr-l.pt",  # real-time DETR (трансформер)
}


def build_ultralytics(name: str):
    """Создаёт модель ultralytics с предобученными весами."""
    if name not in ULTRALYTICS_WEIGHTS:
        raise KeyError(f"Неизвестная ultralytics-модель: {name}")
    weights = ULTRALYTICS_WEIGHTS[name]
    if name == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(weights)
    from ultralytics import YOLO
    return YOLO(weights)
