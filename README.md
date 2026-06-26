# Детекция типов гражданских самолётов: сравнение 5 моделей

Учебная практика БВТ — проект полного цикла в области компьютерного зрения.
Решается задача **детектирования самолётов на изображениях и видео с определением
типа воздушного судна** (например, A320 vs A321 vs Boeing 737) на датасете
**FGVC-Aircraft**. Сравниваются **пять современных моделей детекции**.

## Сравниваемые модели

| Модель | Тип архитектуры | Backbone |
|--------|-----------------|----------|
| YOLOv8 | anchor-free, одностадийная | CSPDarknet |
| Faster R-CNN | двухстадийная | ResNet-50 + FPN |
| SSD300 | одностадийная, anchor-based | VGG-16 |
| RetinaNet | одностадийная, focal loss | ResNet-50 + FPN |
| RT-DETR | трансформер, real-time | ResNet-50 |

Все модели дообучаются (fine-tuning) из весов, предобученных на COCO.

## Структура проекта

```
cv-project/
├── configs/            # YAML-конфиги экспериментов (гиперпараметры, пути)
│   └── default.yaml
├── data/
│   ├── raw/            # FGVC-Aircraft (скачивается автоматически)
│   └── processed/      # конвертированные аннотации (YOLO / COCO)
├── src/
│   ├── dataset/        # загрузка, конвертация, аугментации, Dataset
│   ├── models/         # 5 моделей детекции + общий интерфейс
│   ├── training/       # цикл обучения, логирование, чекпойнты
│   ├── evaluation/     # метрики (mAP, Precision, Recall, F1)
│   └── utils/          # визуализация, инференс на фото/видео
├── notebooks/          # исследовательский анализ данных (EDA)
├── results/            # графики, логи, чекпойнты
└── main.py             # точка входа (train / eval / predict)
```

## Установка

```bash
# 1. PyTorch с CUDA под вашу GPU (пример: RTX 4060 Ti, CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. остальные зависимости
pip install -r requirements.txt

# 3. проверка, что GPU виден
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Использование

```bash
# Подготовка данных (скачивание FGVC + конвертация аннотаций)
python main.py --mode prepare --config configs/default.yaml

# Обучение одной модели
python main.py --mode train --model yolov8 --config configs/default.yaml

# Обучение всех 5 моделей
python main.py --mode train --model all --config configs/default.yaml

# Оценка и сравнение моделей (таблицы + графики в results/)
python main.py --mode eval --model all --config configs/default.yaml

# Инференс на фото
python main.py --mode predict --model yolov8 --source path/to/image.jpg

# Инференс на видео
python main.py --mode predict --model yolov8 --source path/to/video.mp4
```

## Воспроизводимость

- Все гиперпараметры зафиксированы в `configs/default.yaml`.
- `seed` фиксируется для `random`, `numpy`, `torch`.
- Сплиты train/val/test берутся из официального разбиения FGVC-Aircraft.

## Датасет

[FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) — 10 200
изображений, 100 типов самолётов, у каждого изображения есть тип и bounding box.
В проекте используется подмножество из 15 распространённых гражданских
авиалайнеров (список — в `configs/default.yaml`).
