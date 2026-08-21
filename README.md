# Vision-Guided Robotic Insertion

基于 RGB-D 相机、Ultralytics YOLO 实例分割、三维点云、空间标定和 TCP 通信的机器人视觉定位项目。

This repository contains work developed as part of an MSc Robotics project. It is intended to present the project design, Python implementation and system workflow for academic review and portfolio use.

## Overview

本项目从 Intel RealSense 相机或本地 RGB-D 图像中获取彩色与深度数据，通过 YOLO 实例分割提取目标区域，再结合相机内参生成目标区域的三维点云。程序计算目标区域的三维中心，通过标定矩阵转换为机器人坐标，并将坐标发送给配套的 `frrjiftest` 程序。

当前 Python 项目负责：

- RGB-D 图像采集与对齐
- 本地 RGB-D 数据回放
- YOLO 实例分割
- ROI 三维点云生成与显示
- 目标中心坐标计算
- 相机坐标到机器人坐标的转换
- 向 `frrjiftest` 发送机器人坐标

机器人控制器连接和实际运动由配套 EXE 与机器人系统完成，不属于 Python 视觉代码本身。

## Motivation

项目目标是建立一套从视觉检测到机器人坐标输出的完整定位流程，使机器人能够根据 RGB-D 视觉结果获得目标的三维位置。项目同时包含数据准备、实例分割训练、相机采集、三维标定、点云显示和机器人通信等环节。

## System Overview

```text
Intel RealSense / Local RGB-D Images
                  |
                  v
        RGB-D Acquisition and Alignment
                  |
                  v
         YOLO Instance Segmentation
                  |
                  v
       ROI Mask + Depth + Intrinsics
                  |
                  v
       3D Point Cloud / Center XYZ
                  |
                  v
       Calibration Transformation
                  |
                  v
          Robot Coordinate XYZ
                  |
                  v
 main.py TCP Server (127.0.0.1:5678)
                  |
                  v
      frrjiftest.exe TCP Client
                  |
                  v
          Robot Controller / Motion
```

## Communication Architecture

当前视觉程序与 EXE 使用本机 TCP 通信：

```text
Host: 127.0.0.1
Port: 5678
```

`127.0.0.1` 是本机回环地址，表示 `main.py` 和 `frrjiftest` 在同一台计算机上通信。当前关系为：

- `main.py`：TCP Server
- `frrjiftest`：TCP Client

坐标使用 ASCII 文本发送，格式为：

```text
X:123.456;Y:234.567;Z:345.678
```

`config.yaml` 中的 `robot` TCP 配置块是早期版本或预留的机器人直连配置。当前 PyQt5 主程序没有读取该配置块；它不是 `main.py` 与 EXE 的当前通信地址。

## Main Features

- Intel RealSense 彩色与深度图同步采集
- Depth-to-Color 对齐
- 本地 `color/` 与 `depth/` 文件夹回放模式
- Ultralytics YOLO 实例分割
- 分割区域联合 Mask 生成
- 基于深度和相机内参的三维反投影
- 三维 ROI 范围过滤
- PyVista 点云可视化
- ROI 中心 XYZ 计算
- 4 x 4 标定矩阵坐标转换
- PyQt5 图形界面
- 与机器人配套 EXE 的 TCP 坐标通信
- LabelMe JSON 到 YOLO segmentation 标签转换
- YOLO 数据集划分、训练和推理

## Hardware

- Intel RealSense D435I RGB-D camera
- Windows PC
- Robot system: `Universal Robots UR5e.`
- Network connection between the companion EXE and robot controller

## Software

- Python 3.10 or later
- NumPy
- OpenCV
- PyYAML
- Ultralytics YOLO
- PyQt5
- Intel RealSense SDK / `pyrealsense2`
- PyVista
- `pyvistaqt`

## Project Structure

```text
Positioning/
|-- main.py                         # Main PyQt5 vision and TCP application
|-- main_settings.json              # Local GUI state; not published
|-- camera.py                       # Camera abstraction and RGB-D acquisition
|-- realsense_align.py              # RealSense alignment and capture utility
|-- extract_roi_center_points.py    # ROI point extraction for calibration
|-- calibrate_3d_points_opencv.py   # 3D affine calibration utility
|-- calibration.py                  # 2D/3D calibration functions
|-- infer.py                        # YOLO inference utility
|-- train.py                        # YOLO training entry point
|-- json_to_txt.py                  # LabelMe JSON to YOLO segmentation labels
|-- prepare_dataset.py              # Dataset preparation and splitting
|-- common.py                       # Shared path and configuration helpers
|-- config.yaml                     # Legacy/training configuration
|-- calibration/                    # Public calibration templates
|-- 01_images_raw/                  # Private raw images; not published
|-- 02_annotation/                  # Private images and annotations; not published
|-- 03_dataset/                     # Private training dataset; not published
|-- dataset/                        # Private RGB-D captures; not published
|-- 9点标定/                         # Private calibration measurements; not published
|-- runs/                           # Training and inference outputs; not published by default
`-- output/                         # Runtime output; not published
```

## Installation

Clone or download the repository, create a Python environment, and install the direct Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy opencv-python PyYAML ultralytics PyQt5 pyrealsense2 pyvista pyvistaqt
```

The Intel RealSense device and SDK must be correctly installed before using camera mode.

Some scripts currently contain project-specific absolute Windows paths. They are retained to preserve the exact project implementation and may need to be configured for another computer before use.

## Usage

### Main visual application

Run:

```powershell
python main.py
```

The GUI supports:

- Camera mode using an Intel RealSense device
- Local mode using paired `color/` and `depth/` folders
- YOLO model selection
- RGB, depth and 3D point-cloud visualization
- ROI coordinate calculation
- Automatic TCP coordinate transmission

### 本人实际操作方法

以下是本项目实际连接机器人并执行自动运动时采用的操作顺序：

1. 打开并运行 `main.py`。
2. 在 `main.py` 界面中选择“相机模式”。
3. 打开 `frrjiftest0817.exe`。
4. 在 EXE 中先点击“监听”，使 EXE 与机器人建立连接。
5. 点击“启动客户端”，使 EXE 连接到视觉程序的 `127.0.0.1:5678` TCP Server。
6. 点击“开始自动移动”，使机器人系统进入等待视觉坐标的自动运动状态。
7. 返回 `main.py`，点击“启动”。
8. `main.py` 获取相机画面，完成目标分割、三维中心计算和坐标转换，然后向 EXE 发送一组机器人坐标。
9. EXE 接收到坐标后，机器人根据该坐标自动运动。

当前相机模式的程序逻辑在每次启动相机后最多实际发送第一组坐标，以避免连续帧重复触发机器人运动。

> **Safety warning:** Robot motion must only be performed by trained personnel. Before enabling automatic movement, verify the coordinate transformation, robot workspace, tool configuration and motion speed; keep the emergency stop accessible and ensure no person is inside the robot operating area.

### Dataset preparation and training

The supporting training workflow is:

```text
LabelMe annotation
       |
       v
json_to_txt.py
       |
       v
prepare_dataset.py
       |
       v
train.py
       |
       v
infer.py
```

The scripts retain their original project-specific settings and file paths. No source code has been reorganized for publication.

## Calibration

The repository contains scripts for:

- 2D pixel-to-robot calibration
- 3D camera-to-robot affine calibration
- ROI center point extraction
- Transformation error calculation and verification
- Loading a 4 x 4 transformation matrix in the main application

Only calibration templates are intended for the public repository. Actual calibration images, camera coordinates, robot coordinates, transformation matrices and verification results are excluded.

## Dataset

> The original experimental RGB-D images, annotations and robot measurement data are not included in this public repository.

These files are excluded because of project data management requirements, privacy considerations and repository size. No public dataset download link is currently provided.

## Models

The project uses Ultralytics YOLO models for instance segmentation. The local development directory contains both pretrained and trained weights.

- Pretrained models originate from Ultralytics.
- Project-trained weights were produced using private experimental data.
- The public repository includes the Ultralytics `yolo11n-seg.pt` pretrained weight used by the training script.
- The public repository also includes `runs/segment/exp5/weights/best.pt`, the trained weight selected by the current main application.

Ultralytics software and models are subject to the Ultralytics AGPL-3.0 or Enterprise licensing terms. Users should review the applicable license before reuse.

## Executable / Third-party Components

`frrjiftest0817.exe` is a third-party companion component and was not developed as part of this MSc project. Permission to redistribute and use the executable has been confirmed with its author.

The executable is used to:

- Connect to the robot system
- Connect as a TCP client to `main.py`
- Receive the calculated `X/Y/Z` coordinates
- Support automatic robot movement based on the received coordinates

[Download `frrjiftest0817.exe` from GitHub Releases](https://github.com/tedzhong27149/Vision-Guided-Robotic-Insertion/releases). Repository access is required while the project remains private.

## Results

The implemented system demonstrates the complete software path from RGB-D acquisition and instance segmentation to calibrated robot-coordinate transmission.

`TODO: Add final experimental accuracy, repeatability and robot motion results that are approved for public release.`

## Repository Usage

Visitors may use this repository to:

- Read the Python source code
- Understand the RGB-D vision and robot-coordinate workflow
- Review the system architecture and calibration method
- Clone or download the public project files
- Download approved binary components from GitHub Releases

Private experimental data and measurements are intentionally excluded.

## Academic Project

This repository contains work developed as part of an MSc Robotics project.

The repository is provided for academic presentation, technical review and portfolio purposes.

## License and Attribution

- Project code license: GNU Affero General Public License v3.0 (`AGPL-3.0`). See `LICENSE`.
- Ultralytics YOLO: subject to the applicable Ultralytics license.
- `frrjiftest0817.exe`: third-party component redistributed with the author's permission.
