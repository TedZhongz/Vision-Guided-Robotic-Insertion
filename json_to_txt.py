# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

from common import find_image, load_config, resolve


def collect_annotation_classes(json_dir: Path) -> list[str]:
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


def shape_to_polygon(shape: dict) -> list | None:
    shape_type = shape.get("shape_type", "polygon")
    points = shape.get("points", [])
    if shape_type == "polygon":
        return points
    if shape_type == "rectangle":
        (x1, y1), (x2, y2) = points[0], points[1]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return None


def clean_polygon(points: list, width: int, height: int) -> list[float]:
    cleaned = []
    for x, y in points:
        nx = min(max(x / width, 0.0), 1.0)
        ny = min(max(y / height, 0.0), 1.0)
        if not cleaned or cleaned[-2:] != [nx, ny]:
            cleaned.extend([nx, ny])
    if len(cleaned) >= 6 and cleaned[:2] == cleaned[-2:]:
        cleaned = cleaned[:-2]
    return cleaned if len(cleaned) >= 6 else []


def convert_one(json_path: Path, class_to_id: dict[str, int], images_dir: Path, txt_dir: Path) -> tuple[int, list[str]]:
    warnings = []
    data = json.loads(json_path.read_text(encoding="utf-8"))

    width, height = data["imageWidth"], data["imageHeight"]

    if find_image(images_dir, json_path.stem) is None:
        warnings.append(f"未在 images 目录找到同名图片: {json_path.stem}.*")

    lines = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if not label:
            warnings.append("发现空类别标注，已跳过")
            continue

        polygon = shape_to_polygon(shape)
        if polygon is None:
            warnings.append(f"标注类型 '{shape.get('shape_type')}' 不支持实例分割，已跳过")
            continue

        norm = clean_polygon(polygon, width, height)
        if not norm:
            warnings.append(f"类别 '{label}' 的多边形顶点不足，已跳过")
            continue

        lines.append(f"{class_to_id[label]} " + " ".join(f"{v:.6f}" for v in norm))

    txt_path = txt_dir / f"{json_path.stem}.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines), warnings


def main():
    cfg = load_config()
    ann_dir = resolve(cfg["paths"]["annotation"])
    json_dir = ann_dir / "labels" / "json"
    txt_dir = ann_dir / "labels" / "txt"
    images_dir = ann_dir / "images"
    txt_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"[错误] {json_dir} 下没有 JSON 标注文件。")
        return

    classes = collect_annotation_classes(json_dir)
    class_to_id = {name: i for i, name in enumerate(classes)}
    print(f"类别映射: {class_to_id}")

    total_instances, n_ok, n_empty = 0, 0, 0
    for jf in json_files:
        try:
            n, warns = convert_one(jf, class_to_id, images_dir, txt_dir)
        except Exception as e:
            print(f"[失败] {jf.name}: {e}")
            continue
        total_instances += n
        n_ok += 1
        if n == 0:
            n_empty += 1
        status = f"{n} 个实例" if n > 0 else "空标注"
        print(f"[OK] {jf.name} -> {status}")
        for w in warns:
            print(f"  警告: {w}")

    print("-" * 50)
    print(f"完成: 共处理 {n_ok} 个 JSON，实例总数 {total_instances}，空标注文件 {n_empty} 个")
    print(f"输出目录: {txt_dir}")


if __name__ == "__main__":
    main()
