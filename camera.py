# -*- coding: utf-8 -*-
"""
camera.py —— 相机采集模块（双目 / 深度相机抽象层）

统一的相机接口：
    read() -> (color_bgr, depth_m)
        color_bgr: 彩色图 (H, W, 3) uint8 BGR
        depth_m  : 深度图 (H, W) float32，单位：米，与彩色图已对齐
    读取失败返回 None

内置三种实现，由 config.yaml 的 camera.type 选择：
    mock     : 回放本地图片（无需硬件，用于调试整个流程）
    realsense: Intel RealSense 深度相机（需要 pip 安装 pyrealsense2）
    opencv   : 两个普通相机分别当彩色/深度源（深度图需自行保证对齐）

扩展自己的相机时：继承 CameraBase 实现 read()/release()，
并在 create_camera() 中注册即可。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from common import IMG_EXTS


class CameraBase:
    """相机基类"""

    def read(self) -> tuple[np.ndarray, np.ndarray] | None:
        raise NotImplementedError

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# ----------------------------------------------------------------------
# Mock 相机：回放本地图片，用于无硬件时调试流程
# ----------------------------------------------------------------------

class MockCamera(CameraBase):
    def __init__(self, image_dir: Path, depth_dir: Path | None = None,
                 depth_scale: float = 0.001, default_depth_m: float = 0.6):
        self.image_dir = Path(image_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.depth_scale = depth_scale
        self.default_depth_m = default_depth_m
        self.files = sorted(p for ext in IMG_EXTS
                            for p in self.image_dir.glob(f"*{ext}"))
        self.index = 0
        if not self.files:
            print(f"[提示] Mock 相机图片目录为空: {self.image_dir}")

    def read(self):
        if not self.files:
            return None
        img_path = self.files[self.index % len(self.files)]
        self.index += 1
        color = cv2.imread(str(img_path))
        if color is None:
            return None

        # 优先读取同名 16bit 深度图（png），否则用固定深度
        depth = None
        if self.depth_dir is not None:
            depth_path = self.depth_dir / f"{img_path.stem}.png"
            if depth_path.exists():
                raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    depth = raw.astype(np.float32) * self.depth_scale
        if depth is None:
            depth = np.full(color.shape[:2], self.default_depth_m, dtype=np.float32)

        # 深度图与彩色图尺寸不一致时，缩放到彩色图尺寸
        if depth.shape[:2] != color.shape[:2]:
            depth = cv2.resize(depth, (color.shape[1], color.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        return color, depth


# ----------------------------------------------------------------------
# RealSense 深度相机
# ----------------------------------------------------------------------

class RealsenseCamera(CameraBase):
    def __init__(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise SystemExit("[错误] 未安装 pyrealsense2，无法使用 realsense 模式。\n"
                             "如需使用请安装: pip install pyrealsense2")
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)   # 深度图对齐到彩色图

    def read(self):
        rs = self.rs
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None
        color = np.asanyarray(color_frame.get_data())            # BGR
        units = depth_frame.get_units()                          # 每单位对应米数
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * units
        return color, depth

    def release(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ----------------------------------------------------------------------
# OpenCV 普通相机（彩色 + 深度各一路）
# ----------------------------------------------------------------------

class OpenCVCamera(CameraBase):
    def __init__(self, color_index: int, depth_index: int, depth_scale: float):
        self.depth_scale = depth_scale
        self.cap_color = cv2.VideoCapture(color_index)
        self.cap_depth = cv2.VideoCapture(depth_index)
        if not self.cap_color.isOpened():
            raise SystemExit(f"[错误] 无法打开彩色相机 index={color_index}")
        if not self.cap_depth.isOpened():
            raise SystemExit(f"[错误] 无法打开深度相机 index={depth_index}")

    def read(self):
        ok_c, color = self.cap_color.read()
        ok_d, depth_raw = self.cap_depth.read()
        if not ok_c or not ok_d:
            return None
        # 16bit 原始深度图 × scale 换算为米；8bit 图假定已是米制
        if depth_raw.dtype == np.uint16:
            depth = depth_raw.astype(np.float32) * self.depth_scale
        else:
            depth = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY).astype(np.float32) \
                if depth_raw.ndim == 3 else depth_raw.astype(np.float32)
        if depth.shape[:2] != color.shape[:2]:
            depth = cv2.resize(depth, (color.shape[1], color.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        return color, depth

    def release(self):
        self.cap_color.release()
        self.cap_depth.release()


# ----------------------------------------------------------------------
# 工厂函数
# ----------------------------------------------------------------------

def create_camera(cfg: dict) -> CameraBase:
    """根据 config.yaml 中 camera.type 创建相机对象"""
    from common import resolve

    cam_cfg = cfg["camera"]
    cam_type = cam_cfg.get("type", "mock").lower()

    if cam_type == "mock":
        depth_dir = cam_cfg.get("mock_depth_dir") or None
        return MockCamera(
            image_dir=resolve(cam_cfg["mock_image_dir"]),
            depth_dir=resolve(depth_dir) if depth_dir else None,
            depth_scale=float(cam_cfg.get("depth_scale", 0.001)),
            default_depth_m=float(cam_cfg.get("mock_default_depth_m", 0.6)),
        )
    if cam_type == "realsense":
        return RealsenseCamera()
    if cam_type == "opencv":
        return OpenCVCamera(
            color_index=int(cam_cfg.get("opencv_color_index", 0)),
            depth_index=int(cam_cfg.get("opencv_depth_index", 1)),
            depth_scale=float(cam_cfg.get("depth_scale", 0.001)),
        )
    raise SystemExit(f"[错误] 不支持的相机类型: {cam_type} "
                     f"(可选: mock / realsense / opencv)")
