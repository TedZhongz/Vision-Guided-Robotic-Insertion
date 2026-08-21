# -*- coding: utf-8 -*-

import json
import os

import cv2
import numpy as np


# =========================
# 硬编码路径
# =========================

ROBOT_POINTS_FILE = r"D:\Positioning\9点标定\机械臂坐标.txt"
IMAGE_POINTS_FILE = r"D:\Positioning\9点标定\roi_center_points.txt"

OUTPUT_JSON_FILE = r"D:\Positioning\9点标定\opencv_3d_calibration_result.json"
OUTPUT_TRANSFORM_FILE = r"D:\Positioning\9点标定\opencv_3d_transform_matrix.txt"
OUTPUT_VERIFY_FILE = r"D:\Positioning\9点标定\opencv_3d_verify_points.txt"

ERROR_THRESHOLD_MM = 20.0


def read_points_file(file_path):
    if not os.path.exists(file_path):
        raise SystemExit("错误：文件不存在：%s" % file_path)

    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "":
                continue
            lines.append(line)

    if len(lines) < 2:
        raise SystemExit("错误：文件没有有效点数据：%s" % file_path)

    points = []
    for i in range(1, len(lines)):
        line = lines[i]
        parts = line.replace(",", "\t").split()
        if len(parts) != 3:
            raise SystemExit("错误：第 %d 行不是 3 个坐标：%s" % (i + 1, line))

        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
        points.append([x, y, z])

    return np.array(points, dtype=np.float32)


def build_transform_matrix(affine_matrix, scale):
    matrix_3x4 = np.array(affine_matrix, dtype=np.float64)
    matrix_3x4[:, 0:3] = matrix_3x4[:, 0:3] * scale

    transform = np.eye(4, dtype=np.float64)
    transform[0:3, 0:4] = matrix_3x4
    return transform


def transform_points(points, transform):
    result = []

    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]

        p = np.array([x, y, z, 1.0], dtype=np.float64)
        q = np.dot(transform, p)
        result.append([float(q[0]), float(q[1]), float(q[2])])

    return np.array(result, dtype=np.float64)


def calculate_errors(pred_points, target_points):
    errors = []

    for i in range(len(pred_points)):
        dx = pred_points[i][0] - target_points[i][0]
        dy = pred_points[i][1] - target_points[i][1]
        dz = pred_points[i][2] - target_points[i][2]
        error = (dx * dx + dy * dy + dz * dz) ** 0.5
        errors.append(error)

    return np.array(errors, dtype=np.float64)


def run_calibration(image_points, robot_points):
    src = image_points.reshape(-1, 1, 3)
    dst = robot_points.reshape(-1, 1, 3)

    affine_matrix, scale = cv2.estimateAffine3D(src, dst, force_rotation=True)

    if affine_matrix is None:
        raise SystemExit("错误：OpenCV 3D 点标定失败。")

    scale = float(scale)
    transform = build_transform_matrix(affine_matrix, scale)
    pred_points = transform_points(image_points, transform)
    errors = calculate_errors(pred_points, robot_points)

    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))

    return transform, scale, pred_points, errors, mean_error, max_error


def save_transform_matrix(file_path, transform):
    lines = []

    for i in range(4):
        line = "%.8f\t%.8f\t%.8f\t%.8f" % (
            transform[i][0],
            transform[i][1],
            transform[i][2],
            transform[i][3],
        )
        lines.append(line)

    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def save_verify_points(file_path, image_points, robot_points, pred_points, errors):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("图像X\t图像Y\t图像Z\t机械臂X\t机械臂Y\t机械臂Z\t预测X\t预测Y\t预测Z\t误差mm\n")

        for i in range(len(image_points)):
            line = "%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f" % (
                image_points[i][0],
                image_points[i][1],
                image_points[i][2],
                robot_points[i][0],
                robot_points[i][1],
                robot_points[i][2],
                pred_points[i][0],
                pred_points[i][1],
                pred_points[i][2],
                errors[i],
            )
            f.write(line + "\n")


def build_filtered_points(image_points, robot_points, errors, threshold):
    keep_image_points = []
    keep_robot_points = []
    removed_indexes = []

    for i in range(len(errors)):
        if errors[i] <= threshold:
            keep_image_points.append(image_points[i].tolist())
            keep_robot_points.append(robot_points[i].tolist())
        else:
            removed_indexes.append(i + 1)

    keep_image_points = np.array(keep_image_points, dtype=np.float32)
    keep_robot_points = np.array(keep_robot_points, dtype=np.float32)

    return keep_image_points, keep_robot_points, removed_indexes


def main():
    print("机械臂坐标文件：", ROBOT_POINTS_FILE)
    print("图像坐标文件：", IMAGE_POINTS_FILE)
    print("误差剔除阈值(mm)：", ERROR_THRESHOLD_MM)

    robot_points = read_points_file(ROBOT_POINTS_FILE)
    image_points = read_points_file(IMAGE_POINTS_FILE)

    if len(robot_points) != len(image_points):
        raise SystemExit(
            "错误：两组点数量不一致。机械臂点数=%d，图像点数=%d"
            % (len(robot_points), len(image_points))
        )

    if len(robot_points) < 3:
        raise SystemExit("错误：3D 标定至少需要 3 组点。")

    print("原始点数量：", len(robot_points))

    transform_1, scale_1, pred_points_1, errors_1, mean_error_1, max_error_1 = run_calibration(
        image_points, robot_points
    )

    print("=" * 60)
    print("第一次标定结果")
    print("尺度因子：%.8f" % scale_1)
    print("平均误差(mm)：%.4f" % mean_error_1)
    print("最大误差(mm)：%.4f" % max_error_1)

    keep_image_points, keep_robot_points, removed_indexes = build_filtered_points(
        image_points, robot_points, errors_1, ERROR_THRESHOLD_MM
    )

    print("剔除的点序号：", removed_indexes)
    print("剔除后的点数量：", len(keep_robot_points))

    if len(keep_robot_points) < 3:
        raise SystemExit("错误：剔除后剩余点少于 3 个，无法继续标定。")

    transform_2, scale_2, pred_points_2, errors_2, mean_error_2, max_error_2 = run_calibration(
        keep_image_points, keep_robot_points
    )

    print("=" * 60)
    print("第二次标定结果（剔除误差大于 10mm 的点后）")
    print("尺度因子：%.8f" % scale_2)
    print("平均误差(mm)：%.4f" % mean_error_2)
    print("最大误差(mm)：%.4f" % max_error_2)
    print("变换矩阵：")
    print(transform_2)

    save_transform_matrix(OUTPUT_TRANSFORM_FILE, transform_2)
    save_verify_points(OUTPUT_VERIFY_FILE, keep_image_points, keep_robot_points, pred_points_2, errors_2)

    result_data = {
        "error_threshold_mm": ERROR_THRESHOLD_MM,
        "first_calibration": {
            "image_points": image_points.tolist(),
            "robot_points": robot_points.tolist(),
            "transform_matrix_4x4": transform_1.tolist(),
            "scale": scale_1,
            "mean_error_mm": mean_error_1,
            "max_error_mm": max_error_1,
            "errors_mm": errors_1.tolist(),
        },
        "removed_point_indexes": removed_indexes,
        "second_calibration": {
            "image_points": keep_image_points.tolist(),
            "robot_points": keep_robot_points.tolist(),
            "transform_matrix_4x4": transform_2.tolist(),
            "scale": scale_2,
            "mean_error_mm": mean_error_2,
            "max_error_mm": max_error_2,
            "errors_mm": errors_2.tolist(),
        },
    }

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("结果文件：", OUTPUT_JSON_FILE)
    print("矩阵文件：", OUTPUT_TRANSFORM_FILE)
    print("验证文件：", OUTPUT_VERIFY_FILE)


if __name__ == "__main__":
    main()
