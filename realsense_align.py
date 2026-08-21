import pyrealsense2 as rs
import numpy as np
import cv2
import os


# =========================
# 1. 创建保存文件夹
# =========================
os.makedirs("dataset/color", exist_ok=True)
os.makedirs("dataset/depth", exist_ok=True)


# =========================
# 2. 自动寻找下一个编号
# =========================
existing_files = os.listdir("dataset/color")
existing_numbers = []

for filename in existing_files:
    if filename.startswith("color_") and filename.endswith(".png"):

        number = filename.replace("color_", "").replace(".png", "")

        if number.isdigit():
            existing_numbers.append(int(number))


if existing_numbers:
    save_count = max(existing_numbers) + 1
else:
    save_count = 1


print(f"Next save number: {save_count:04d}")


# =========================
# 3. 创建 RealSense pipeline
# =========================
pipeline = rs.pipeline()
config = rs.config()

# Depth
config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    30
)

# RGB
config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)


# 启动相机
pipeline.start(config)


# =========================
# 4. Depth 对齐到 RGB
# =========================
align = rs.align(rs.stream.color)


print("Camera started.")
print("Press S to save RGB + aligned depth.")
print("Press Q to quit.")


try:

    while True:

        # 获取帧
        frames = pipeline.wait_for_frames()

        # Depth -> RGB 对齐
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue


        # 转换成 numpy
        depth_image = np.asanyarray(
            aligned_depth_frame.get_data()
        )

        color_image = np.asanyarray(
            color_frame.get_data()
        )


        # =========================
        # 5. 深度伪彩色，仅供显示
        # =========================
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(
                depth_image,
                alpha=0.03
            ),
            cv2.COLORMAP_JET
        )


        # RGB + Depth 拼接
        images = np.hstack(
            (color_image, depth_colormap)
        )


        cv2.putText(
            images,
            f"S: Save #{save_count:04d}    Q: Quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "RGB | Aligned Depth",
            images
        )


        # =========================
        # 6. 键盘控制
        # =========================
        key = cv2.waitKey(1) & 0xFF


        # 按 S 保存
        if key == ord("s"):

            color_path = (
                f"dataset/color/color_{save_count:04d}.png"
            )

            depth_path = (
                f"dataset/depth/depth_{save_count:04d}.png"
            )


            # 保存 RGB
            cv2.imwrite(
                color_path,
                color_image
            )


            # 保存 16-bit aligned depth
            cv2.imwrite(
                depth_path,
                depth_image
            )


            print()
            print(f"Saved #{save_count:04d}")
            print(f"RGB:   {color_path}")
            print(f"Depth: {depth_path}")


            save_count += 1


        # 按 Q 退出
        elif key == ord("q"):
            break


finally:

    pipeline.stop()
    cv2.destroyAllWindows()

    print("Camera stopped.")
