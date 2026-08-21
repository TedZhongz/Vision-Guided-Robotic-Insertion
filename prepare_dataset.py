# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import yaml

from common import IMG_EXTS, find_image, load_config, resolve

SPLITS = ("train", "val", "test")


def collect_annotation_classes(ann_dir: Path) -> list[str]:
    json_dir = ann_dir / "labels" / "json"
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"[错误] {json_dir} 下没有 JSON 标注文件。")

    classes: list[str] = []
    seen: set[str] = set()
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if not label or label in seen:
                continue
            seen.add(label)
            classes.append(label)

    if not classes:
        raise SystemExit("[错误] 标注结果中没有可用类别。")
    return classes


def validate_label_ids(pairs: list[tuple[Path, Path]], classes: list[str]) -> None:
    max_valid_id = len(classes) - 1
    for _, txt_path in pairs:
        for line_no, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                class_id = int(float(parts[0]))
            except (ValueError, IndexError):
                raise SystemExit(f"[错误] 标签文件格式异常: {txt_path} 第 {line_no} 行")
            if class_id < 0 or class_id > max_valid_id:
                raise SystemExit(
                    f"[错误] 标签文件中的类别 id 超出范围: {txt_path} 第 {line_no} 行 -> {class_id}，"
                    f"但根据标注结果只识别到 {len(classes)} 个类别: {classes}"
                )


def collect_pairs(ann_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = ann_dir / "images"
    txt_dir = ann_dir / "labels" / "txt"

    pairs = []
    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"[错误] {txt_dir} 下没有标签文件，请先运行 json_to_txt.py。")

    no_image = []
    for tf in txt_files:
        img = find_image(images_dir, tf.stem)
        if img is None:
            no_image.append(tf.name)
            continue
        pairs.append((img, tf))

    labeled_stems = {tf.stem for _, tf in pairs}
    no_label = [
        p.name
        for ext in IMG_EXTS
        for p in images_dir.glob(f"*{ext}")
        if p.stem not in labeled_stems
    ]

    for name in no_image:
        print(f"警告: 标签 {name} 找不到同名图片，已跳过")
    if no_label:
        print(f"提示: 有 {len(no_label)} 张图片没有标签，不参与训练: {no_label[:5]}")
    return pairs


def split_pairs(pairs: list, r_train: float, r_val: float, r_test: float, seed: int) -> dict[str, list]:
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    if n == 0:
        raise SystemExit("[错误] 没有可用的图片和标签配对。")

    n_val = max(1, round(n * r_val)) if n >= 2 else 0
    n_test = max(1, round(n * r_test)) if r_test > 0 and n >= 3 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        n_test = 0
        n_val = max(0, n - 1)
        n_train = n - n_val - n_test

    return {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }


def build_dataset(dataset_dir: Path, split: dict[str, list], classes: list[str]):
    for s in SPLITS:
        for sub in ("images", "labels"):
            d = dataset_dir / sub / s
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    counts = {}
    for s in SPLITS:
        for img, tf in split[s]:
            shutil.copy2(img, dataset_dir / "images" / s / img.name)
            shutil.copy2(tf, dataset_dir / "labels" / s / tf.name)
        counts[s] = len(split[s])

    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
    }
    with open(dataset_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    for s in SPLITS:
        lines = [str((dataset_dir / "images" / s / img.name).resolve()) for img, _ in split[s]]
        (dataset_dir / f"{s}.txt").write_text("\n".join(lines), encoding="utf-8")

    return counts


def main():
    cfg = load_config()
    ann_dir = resolve(cfg["paths"]["annotation"])
    dataset_dir = resolve(cfg["paths"]["dataset"])

    classes = collect_annotation_classes(ann_dir)
    pairs = collect_pairs(ann_dir)
    validate_label_ids(pairs, classes)
    print(f"共收集到 {len(pairs)} 组图片和标签配对")
    print(f"标注类别: {classes}")

    sp = cfg["split"]
    split = split_pairs(pairs, sp["train"], sp["val"], sp["test"], sp["seed"])
    counts = build_dataset(dataset_dir, split, classes)

    print("-" * 50)
    print(f"数据集已生成: {dataset_dir}")
    for s in SPLITS:
        print(f"{s:<6}: {counts[s]}")
    print(f"类别: {classes}")


if __name__ == "__main__":
    main()
