# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
from ultralytics import YOLO


# 直接在这里修改硬编码路径
MODEL_PATH = r"D:\Positioning\runs\segment\exp5\weights\best.pt"
IMAGE_FOLDER = r"D:\Positioning\02_annotation\images"

# 推理参数
CONFIDENCE = 0.25
DEVICE = 0

# 结果保存路径
SAVE_PROJECT = r"D:\Positioning\runs\predict"
SAVE_NAME = "infer"


def mask_centroid(mask):
    m = mask.astype(np.uint8)
    result = cv2.moments(m)
    if result["m00"] <= 0:
        return None

    x = result["m10"] / result["m00"]
    y = result["m01"] / result["m00"]
    return x, y


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("错误：模型文件不存在：%s" % MODEL_PATH)

    if not os.path.exists(IMAGE_FOLDER):
        raise SystemExit("错误：测试图像文件夹不存在：%s" % IMAGE_FOLDER)

    print("模型路径：", MODEL_PATH)
    print("图像文件夹：", IMAGE_FOLDER)
    print("置信度：", CONFIDENCE)
    print("设备：", DEVICE)

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=IMAGE_FOLDER,
        conf=CONFIDENCE,
        device=DEVICE,
        save=True,
        project=SAVE_PROJECT,
        name=SAVE_NAME,
        exist_ok=True,
    )

    total = 0

    for result in results:
        image_name = os.path.basename(result.path)

        if result.boxes is None or len(result.boxes) == 0:
            print("[%s] 未检测到目标" % image_name)
            continue

        for i in range(len(result.boxes)):
            box = result.boxes[i]
            class_id = int(box.cls)
            class_name = result.names[class_id]
            score = float(box.conf)

            xyxy = box.xyxy[0].tolist()
            x1 = xyxy[0]
            y1 = xyxy[1]
            x2 = xyxy[2]
            y2 = xyxy[3]

            center_text = "无掩膜"
            if result.masks is not None:
                mask = result.masks.data[i].cpu().numpy()
                center = mask_centroid(mask)
                if center is not None:
                    center_text = "(%.1f, %.1f)" % (center[0], center[1])

            print(
                "[%s] %s conf=%.3f bbox=(%.0f,%.0f,%.0f,%.0f) 质心=%s"
                % (image_name, class_name, score, x1, y1, x2, y2, center_text)
            )

            total = total + 1

    print("=" * 60)
    print("共检测到 %d 个目标，结果保存在 %s\\%s\\" % (total, SAVE_PROJECT, SAVE_NAME))


if __name__ == "__main__":
    main()
