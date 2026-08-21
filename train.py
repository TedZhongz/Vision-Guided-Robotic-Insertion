# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from common import load_config, resolve

# PyCharm 直接运行时，在当前文件里改这些硬编码参数即可。
TRAIN_WEIGHTS = r"D:\Positioning\yolo11n-seg.pt"
TRAIN_EPOCHS = None
TRAIN_IMGSZ = None
TRAIN_BATCH = None
TRAIN_DEVICE = None
TRAIN_NAME = "exp"
TRAIN_RESUME = False


def main():
    cfg = load_config()
    t = cfg["train"]

    weights = Path(TRAIN_WEIGHTS) if TRAIN_WEIGHTS else resolve(cfg["paths"]["weights"])
    if not weights.exists():
        raise SystemExit(
            f"[错误] 找不到预训练权重: {weights}\n"
            "请确认权重文件路径。"
        )

    data_yaml = resolve(cfg["paths"]["dataset"]) / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"[错误] 找不到数据集配置: {data_yaml}\n"
            "请先运行 prepare_dataset.py。"
        )

    epochs = TRAIN_EPOCHS if TRAIN_EPOCHS is not None else t["epochs"]
    imgsz = TRAIN_IMGSZ if TRAIN_IMGSZ is not None else t["imgsz"]
    batch = TRAIN_BATCH if TRAIN_BATCH is not None else t["batch"]
    device_raw = TRAIN_DEVICE if TRAIN_DEVICE is not None else t["device"]
    device = device_raw if str(device_raw).lower() == "cpu" else int(device_raw)

    from ultralytics import YOLO

    print(f"权重: {weights}")
    print(f"数据: {data_yaml}")
    print(f"参数: epochs={epochs} imgsz={imgsz} batch={batch} device={device}")

    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=t.get("workers", 0),
        device=device,
        project=str(resolve(cfg["paths"]["runs"]) / "segment"),
        name=TRAIN_NAME,
        resume=TRAIN_RESUME,
        plots=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print("=" * 60)
    print(f"训练完成，最佳权重: {best}")


if __name__ == "__main__":
    main()
