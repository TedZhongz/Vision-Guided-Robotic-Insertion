# -*- coding: utf-8 -*-

import os
import json

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# 硬编码参数
# =========================

DATASET_FOLDER = r"D:\Positioning\9点标定"
COLOR_FOLDER = r"D:\Positioning\9点标定\color"
DEPTH_FOLDER = r"D:\Positioning\9点标定\depth"
MODEL_PATH = r"D:\Positioning\runs\segment\exp5\weights\best.pt"

# 按当前要求，深度图原值直接按 mm 使用，
# 并且不再使用相机主点做正负偏移，输出全正坐标。
FX = 615.0
FY = 615.0
CX = 320.0
CY = 240.0
DEPTH_SCALE = 1.0

YOLO_CONFIDENCE = 0.25
XYZ_SCALE_X = 1.0
XYZ_SCALE_Y = 1.0
XYZ_SCALE_Z = 1.0

OUTPUT_TXT = r"D:\Positioning\9点标定\roi_center_points.txt"

SETTINGS_FILE = r"D:\Positioning\main_settings.json"

BOX3D_X_MIN = 0.0
BOX3D_X_MAX = 100000.0
BOX3D_Y_MIN = 0.0
BOX3D_Y_MAX = 100000.0
BOX3D_Z_MIN = 10.0
BOX3D_Z_MAX = 400.0


def build_depth_path(color_name):
    if color_name.startswith("color_"):
        depth_name = color_name.replace("color_", "depth_", 1)
    else:
        depth_name = color_name
    return os.path.join(DEPTH_FOLDER, depth_name)


def load_box3d_bounds():
    x_min = BOX3D_X_MIN
    x_max = BOX3D_X_MAX
    y_min = BOX3D_Y_MIN
    y_max = BOX3D_Y_MAX
    z_min = BOX3D_Z_MIN
    z_max = BOX3D_Z_MAX

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)

            x_min = float(settings.get("box_x_min", x_min))
            x_max = float(settings.get("box_x_max", x_max))
            y_min = float(settings.get("box_y_min", y_min))
            y_max = float(settings.get("box_y_max", y_max))
            z_min = float(settings.get("box_z_min", z_min))
            z_max = float(settings.get("box_z_max", z_max))
        except Exception:
            print("提示：main_settings.json 读取失败，使用默认 BOX3D 参数。")

    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    if z_min > z_max:
        z_min, z_max = z_max, z_min

    return x_min, x_max, y_min, y_max, z_min, z_max


def load_depth_image(depth_path, target_width, target_height):
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return None

    if len(depth_raw.shape) == 3:
        channel_count = depth_raw.shape[2]
        if channel_count == 3:
            depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
        elif channel_count == 4:
            depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGRA2GRAY)
        elif channel_count == 1:
            depth_raw = depth_raw[:, :, 0]

    if depth_raw.shape[1] != target_width or depth_raw.shape[0] != target_height:
        depth_raw = cv2.resize(depth_raw, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    depth_mm = depth_raw.astype(np.float32) * DEPTH_SCALE
    return depth_mm


def get_union_mask(result, image_width, image_height):
    if result.masks is None or result.boxes is None:
        return None

    union_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for i in range(len(result.boxes)):
        mask = result.masks.data[i].cpu().numpy()
        if mask.shape[0] != image_height or mask.shape[1] != image_width:
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

        mask_binary = np.zeros((image_height, image_width), dtype=np.uint8)
        mask_binary[mask > 0.5] = 255
        union_mask = cv2.bitwise_or(union_mask, mask_binary)

    if np.count_nonzero(union_mask) == 0:
        return None

    return union_mask


def build_roi_points(depth_mm, mask, box3d_bounds):
    height = depth_mm.shape[0]
    width = depth_mm.shape[1]

    uu, vv = np.meshgrid(np.arange(width), np.arange(height))

    z = depth_mm
    valid = (z > 0.0) & (mask > 0)

    if not np.any(valid):
        return None

    x = uu * z / FX
    y = vv * z / FY

    x = x * 1.0
    y = y * 1.0
    z = z * 1.0

    if box3d_bounds is not None:
        box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = box3d_bounds
        valid = valid & (x >= box_x_min) & (x <= box_x_max)
        valid = valid & (y >= box_y_min) & (y <= box_y_max)
        valid = valid & (z >= box_z_min) & (z <= box_z_max)

    if not np.any(valid):
        return None

    points = np.stack((x, y, z), axis=-1)
    points = points[valid]

    if len(points) == 0:
        return None

    return points


def calculate_center_xyz(points):
    center_x = float(np.mean(points[:, 0]))
    center_y = float(np.mean(points[:, 1]))
    center_z = float(np.mean(points[:, 2]))
    return center_x, center_y, center_z


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("错误：模型文件不存在：%s" % MODEL_PATH)

    if not os.path.isdir(COLOR_FOLDER):
        raise SystemExit("错误：彩色图文件夹不存在：%s" % COLOR_FOLDER)

    if not os.path.isdir(DEPTH_FOLDER):
        raise SystemExit("错误：深度图文件夹不存在：%s" % DEPTH_FOLDER)

    print("数据集文件夹：", DATASET_FOLDER)
    print("彩色图文件夹：", COLOR_FOLDER)
    print("深度图文件夹：", DEPTH_FOLDER)
    print("模型文件：", MODEL_PATH)
    print("输出文件：", OUTPUT_TXT)

    box3d_bounds = load_box3d_bounds()
    print("BOX3D X 范围：%.3f 到 %.3f" % (box3d_bounds[0], box3d_bounds[1]))
    print("BOX3D Y 范围：%.3f 到 %.3f" % (box3d_bounds[2], box3d_bounds[3]))
    print("BOX3D Z 范围：%.3f 到 %.3f" % (box3d_bounds[4], box3d_bounds[5]))

    model = YOLO(MODEL_PATH)

    color_names = os.listdir(COLOR_FOLDER)
    color_names.sort()

    result_lines = []
    result_lines.append("X\tY\tZ")

    for color_name in color_names:
        lower_name = color_name.lower()
        if not lower_name.endswith(".png") and not lower_name.endswith(".jpg") and not lower_name.endswith(".jpeg") and not lower_name.endswith(".bmp"):
            continue

        color_path = os.path.join(COLOR_FOLDER, color_name)
        depth_path = build_depth_path(color_name)

        if not os.path.exists(depth_path):
            print("跳过，没有对应深度图：", color_name)
            continue

        color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
        if color_bgr is None:
            print("跳过，彩色图读取失败：", color_name)
            continue

        depth_mm = load_depth_image(depth_path, color_bgr.shape[1], color_bgr.shape[0])
        if depth_mm is None:
            print("跳过，深度图读取失败：", color_name)
            continue

        results = model.predict(color_bgr, conf=YOLO_CONFIDENCE, verbose=False)
        if len(results) == 0:
            print("跳过，未返回分割结果：", color_name)
            continue

        result = results[0]
        union_mask = get_union_mask(result, color_bgr.shape[1], color_bgr.shape[0])
        if union_mask is None:
            print("跳过，未检测到 ROI：", color_name)
            continue

        roi_points = build_roi_points(depth_mm, union_mask, box3d_bounds)
        if roi_points is None:
            print("跳过，ROI 在 BOX3D 内没有有效 3D 点：", color_name)
            continue

        center_x, center_y, center_z = calculate_center_xyz(roi_points)

        line = "%.2f\t%.2f\t%.2f" % (center_x, center_y, center_z)
        result_lines.append(line)

        print("图像：%s" % color_name)
        print("图像坐标 X,Y,Z(mm)：%.2f\t%.2f\t%.2f" % (center_x, center_y, center_z))
        print("-" * 60)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for line in result_lines:
            f.write(line + "\n")

    print("完成，结果已保存到：", OUTPUT_TXT)


if __name__ == "__main__":
    main()
