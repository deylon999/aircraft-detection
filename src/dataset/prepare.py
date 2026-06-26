"""Подготовка датасета FGVC-Aircraft для задачи детекции типов самолётов.

Что делает:
  1. Скачивает FGVC-Aircraft (через torchvision) в data/raw, если его нет.
  2. Читает официальные аннотации: тип самолёта (variant) + bounding box.
  3. Фильтрует выбранное подмножество классов (configs/default.yaml -> data.classes).
  4. Сохраняет результат в ДВУХ форматах:
       - YOLO  (data/processed/yolo/...)        -> для YOLOv8 и RT-DETR (ultralytics)
       - JSON-манифест (data/processed/*.json)  -> для torchvision-моделей
         (Faster R-CNN, SSD, RetinaNet) и для нашего Dataset.

Запуск:  python -m src.dataset.prepare --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from src.utils.utils import ensure_dir, get_logger, load_config

log = get_logger("prepare")

# Имя папки, которую создаёт torchvision при распаковке FGVC-Aircraft.
_FGVC_DIRNAME = "fgvc-aircraft-2013b"


def _download_fgvc(raw_root: Path) -> Path:
    """Скачивает FGVC-Aircraft через torchvision и возвращает путь к data-папке."""
    data_dir = raw_root / _FGVC_DIRNAME / "data"
    if data_dir.exists() and (data_dir / "images_box.txt").exists():
        log.info("FGVC-Aircraft уже на месте: %s", data_dir)
        return data_dir

    log.info("Скачиваю FGVC-Aircraft в %s (~2.75 ГБ, это надолго)...", raw_root)
    from torchvision.datasets import FGVCAircraft  # импорт здесь, чтобы не тянуть torch без нужды

    ensure_dir(raw_root)
    # download=True скачает и распакует архив. Сам датасет-объект нам не нужен.
    FGVCAircraft(root=str(raw_root), split="trainval", download=True)
    if not (data_dir / "images_box.txt").exists():
        raise FileNotFoundError(f"Не нашёл images_box.txt в {data_dir} после скачивания.")
    log.info("Скачивание завершено.")
    return data_dir


def _read_boxes(data_dir: Path) -> dict[str, tuple[int, int, int, int]]:
    """images_box.txt: '<image_id> <xmin> <ymin> <xmax> <ymax>' (1-индексация, включительно)."""
    boxes: dict[str, tuple[int, int, int, int]] = {}
    with open(data_dir / "images_box.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            img_id = parts[0]
            xmin, ymin, xmax, ymax = map(int, parts[1:5])
            boxes[img_id] = (xmin, ymin, xmax, ymax)
    return boxes


def _read_variant_split(data_dir: Path, split: str) -> list[tuple[str, str]]:
    """images_variant_<split>.txt: '<image_id> <class name>' (имя класса может содержать пробелы)."""
    path = data_dir / f"images_variant_{split}.txt"
    items: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            img_id, _, cls = line.partition(" ")
            items.append((img_id, cls.strip()))
    return items


def _build_class_map(
    items_by_split: dict[str, list[tuple[str, str]]],
    selected: list[str] | None,
) -> dict[str, int]:
    """Строит отображение имя_класса -> id. None => все встретившиеся классы."""
    if selected:
        missing = []
        all_present = {c for items in items_by_split.values() for _, c in items}
        for c in selected:
            if c not in all_present:
                missing.append(c)
        if missing:
            log.warning("Этих классов НЕТ в датасете (проверьте имена в variants.txt): %s", missing)
        classes = [c for c in selected if c in all_present]
    else:
        classes = sorted({c for items in items_by_split.values() for _, c in items})
    return {c: i for i, c in enumerate(classes)}


def _yolo_line(box: tuple[int, int, int, int], w: int, h: int, cls_id: int) -> str:
    """Конвертирует абсолютный бокс в нормализованный YOLO-формат."""
    xmin, ymin, xmax, ymax = box
    xc = (xmin + xmax) / 2.0 / w
    yc = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def prepare(config_path: str) -> None:
    cfg = load_config(config_path)
    dcfg = cfg["data"]
    raw_root = Path(dcfg["root"])
    out_root = ensure_dir(Path(dcfg["processed"]))

    data_dir = _download_fgvc(raw_root)
    images_dir = data_dir / "images"
    boxes = _read_boxes(data_dir)

    # Сопоставление "наш split" -> "официальный split FGVC"
    split_map = {"train": dcfg["splits"]["train"], "val": dcfg["splits"]["val"]}
    items_by_split = {
        our: _read_variant_split(data_dir, fgvc) for our, fgvc in split_map.items()
    }

    class_map = _build_class_map(items_by_split, dcfg.get("classes"))
    if not class_map:
        raise RuntimeError("Список классов пуст — проверьте configs/default.yaml -> data.classes")
    log.info("Используется классов: %d", len(class_map))

    # Сохраняем список классов (id -> name), порядок важен для всех моделей.
    classes_sorted = [c for c, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    (out_root / "classes.txt").write_text("\n".join(classes_sorted), encoding="utf-8")

    # --- директории YOLO ---
    yolo_root = ensure_dir(out_root / "yolo")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        ensure_dir(yolo_root / sub)

    stats: dict[str, Counter] = {"train": Counter(), "val": Counter()}

    for split, items in items_by_split.items():
        manifest: list[dict] = []
        n_skipped = 0
        for img_id, cls in items:
            if cls not in class_map:
                continue
            cls_id = class_map[cls]
            img_path = images_dir / f"{img_id}.jpg"
            if not img_path.exists() or img_id not in boxes:
                n_skipped += 1
                continue

            with Image.open(img_path) as im:
                w, h = im.size
            xmin, ymin, xmax, ymax = boxes[img_id]
            # подстраховка от выхода за границы
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            # 1) YOLO: копируем картинку + пишем .txt
            dst_img = yolo_root / f"images/{split}/{img_id}.jpg"
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            label_path = yolo_root / f"labels/{split}/{img_id}.txt"
            label_path.write_text(_yolo_line((xmin, ymin, xmax, ymax), w, h, cls_id), encoding="utf-8")

            # 2) JSON-манифест для torchvision-моделей
            manifest.append(
                {
                    "image": str(img_path.resolve()),
                    "width": w,
                    "height": h,
                    "boxes": [[xmin, ymin, xmax, ymax]],  # формат [x1,y1,x2,y2]
                    "labels": [cls_id],
                }
            )
            stats[split][cls] += 1

        manifest_path = out_root / f"manifest_{split}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("%s: %d изображений (пропущено %d) -> %s", split, len(manifest), n_skipped, manifest_path.name)

    # --- data.yaml для ultralytics (YOLOv8 / RT-DETR) ---
    data_yaml = (
        f"# Автогенерация из {config_path}. Не редактировать вручную.\n"
        f"path: {yolo_root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(classes_sorted)}\n"
        f"names: {classes_sorted}\n"
    )
    (yolo_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    log.info("YOLO data.yaml -> %s", (yolo_root / "data.yaml"))

    # --- сводка по классам ---
    log.info("Распределение по классам (train / val):")
    for c in classes_sorted:
        log.info("  %-12s  %4d / %4d", c, stats["train"][c], stats["val"][c])
    log.info("Готово. Подготовлено в %s", out_root.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Подготовка FGVC-Aircraft для детекции")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    prepare(args.config)
