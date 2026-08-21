# -*- coding: utf-8 -*-

import os
import json
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtNetwork import QAbstractSocket, QHostAddress, QTcpServer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


WINDOW_TITLE = "ROI Point Cloud Viewer"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
DEPTH_VIS_ALPHA = 0.03

DEFAULT_DATASET_FOLDER = r"D:\Positioning\dataset"
DEFAULT_MODEL_PATH = r"D:\Positioning\runs\segment\exp5\weights\best.pt"

LOCAL_FX = 615.0
LOCAL_FY = 615.0
LOCAL_CX = 320.0
LOCAL_CY = 240.0
LOCAL_DEPTH_SCALE = 1.0

POINT_STRIDE = 4
POINT_MAX_DEPTH_M = 3000.0
POINT_SIZE = 2.0
ROI_POINT_SIZE = 4.0

XYZ_SCALE_X = 1.0
XYZ_SCALE_Y = 1.0
XYZ_SCALE_Z = 1.0

YOLO_CONFIDENCE = 0.25

SETTINGS_FILE = r"D:\Positioning\main_settings.json"
CALIBRATION_TRANSFORM_FILE = r"D:\Positioning\9点标定\opencv_3d_transform_matrix.txt"

TCP_SERVER_HOST = "127.0.0.1"
TCP_SERVER_PORT = 5678
AUTO_SEND_TCP = True

BOX3D_X_MIN = 0.0
BOX3D_X_MAX = 100000.0
BOX3D_Y_MIN = 0.0
BOX3D_Y_MAX = 100000.0
BOX3D_Z_MIN = 10.0
BOX3D_Z_MAX = 400.0


class FrameBundle:
    def __init__(self, color_bgr, depth_raw, depth_m, fx, fy, cx, cy):
        self.color_bgr = color_bgr
        self.depth_raw = depth_raw
        self.depth_m = depth_m
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


class RealSenseReader:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.align = None
        self.depth_scale = 1.0

    def start(self):
        if self.pipeline is not None:
            return

        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = 1.0

    def read(self):
        if self.pipeline is None or self.align is None:
            return None

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self.depth_scale

        intr = color_frame.profile.as_video_stream_profile().intrinsics

        return FrameBundle(
            color_bgr,
            depth_raw,
            depth_m,
            float(intr.fx),
            float(intr.fy),
            float(intr.ppx),
            float(intr.ppy),
        )

    def stop(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None
                self.align = None


class SyncedGraphicsView(QGraphicsView):
    def __init__(self):
        QGraphicsView.__init__(self)
        self.partner = None
        self.syncing = False

        self.scene_obj = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene_obj.addItem(self.pixmap_item)
        self.setScene(self.scene_obj)

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(480, 360)

    def set_partner(self, partner):
        self.partner = partner

    def set_image(self, image_bgr):
        height = image_bgr.shape[0]
        width = image_bgr.shape[1]
        qimage = QImage(image_bgr.data, width, height, image_bgr.strides[0], QImage.Format_BGR888).copy()
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        self.scene_obj.setSceneRect(0, 0, width, height)

    def wheelEvent(self, event):
        if self.pixmap_item.pixmap().isNull():
            return

        if event.angleDelta().y() > 0:
            factor = 1.15
        else:
            factor = 1.0 / 1.15

        self.scale(factor, factor)
        self.sync_partner()

    def mouseReleaseEvent(self, event):
        QGraphicsView.mouseReleaseEvent(self, event)
        self.sync_partner()

    def resizeEvent(self, event):
        QGraphicsView.resizeEvent(self, event)
        self.sync_partner()

    def sync_partner(self):
        if self.partner is None:
            return
        if self.syncing:
            return

        self.partner.apply_view_state(
            self.transform(),
            self.horizontalScrollBar().value(),
            self.verticalScrollBar().value(),
        )

    def apply_view_state(self, transform, h_value, v_value):
        self.syncing = True
        try:
            self.setTransform(transform)
            self.horizontalScrollBar().setValue(h_value)
            self.verticalScrollBar().setValue(v_value)
        finally:
            self.syncing = False

    def reset_view(self):
        if self.scene_obj.sceneRect().isNull():
            return
        self.fitInView(self.scene_obj.sceneRect(), Qt.KeepAspectRatio)
        self.sync_partner()


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1900, 1020)

        self.reader = RealSenseReader(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stream)

        self.last_frame = None
        self.last_mask = None
        self.last_roi_center = None
        self.last_robot_center = None
        self.last_roi_count = 0
        self.last_tcp_error_message = ""
        self.tcp_success_notice_shown = False
        self.pending_robot_message = ""
        self.camera_first_message = ""
        self.camera_coordinate_sent = False
        self.camera_tcp_write_count = 0
        self.tcp_server = QTcpServer(self)
        self.tcp_server.newConnection.connect(self.accept_frrjiftest_connection)
        self.frrjiftest_client = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.last_fps_frame_count = 0
        self.saved_local_image_name = ""

        self.yolo_model = None
        self.loaded_model_path = ""
        self.calibration_transform = load_calibration_transform(CALIBRATION_TRANSFORM_FILE)

        self.plotter = None
        self.pv = None
        self.plotter_host = QWidget()
        self.plotter_layout = QVBoxLayout(self.plotter_host)
        self.plotter_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter_placeholder = QLabel("点云窗口未初始化。\n首次显示点云时再加载，这样启动更快。")
        self.plotter_placeholder.setAlignment(Qt.AlignCenter)
        self.plotter_layout.addWidget(self.plotter_placeholder)

        self.color_view = SyncedGraphicsView()
        self.depth_view = SyncedGraphicsView()
        self.color_view.set_partner(self.depth_view)
        self.depth_view.set_partner(self.color_view)

        self.status_label = QLabel("未启动")
        self.status_label.setMinimumWidth(380)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("RealSense 相机")
        self.mode_combo.addItem("本地图片")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        self.dataset_folder_edit = QLineEdit()
        self.dataset_folder_edit.setText(DEFAULT_DATASET_FOLDER)

        self.select_dataset_button = QPushButton("选择文件夹")
        self.select_dataset_button.clicked.connect(self.select_dataset_folder)

        self.local_image_combo = QComboBox()
        self.local_image_combo.currentIndexChanged.connect(self.on_local_image_changed)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setText(DEFAULT_MODEL_PATH)

        self.load_model_button = QPushButton("加载模型")
        self.load_model_button.clicked.connect(self.load_yolo_model)

        self.x_scale_spin = self.build_double_spin(XYZ_SCALE_X, 0.01, 100.0, 0.1)
        self.y_scale_spin = self.build_double_spin(XYZ_SCALE_Y, 0.01, 100.0, 0.1)
        self.z_scale_spin = self.build_double_spin(XYZ_SCALE_Z, 0.01, 100.0, 0.1)
        self.max_depth_spin = self.build_double_spin(POINT_MAX_DEPTH_M, 1.0, 10000.0, 10.0)
        self.point_size_spin = self.build_double_spin(POINT_SIZE, 1.0, 10.0, 0.5)
        self.box_x_min_spin = self.build_double_spin(BOX3D_X_MIN, 0.0, 100000.0, 10.0)
        self.box_x_max_spin = self.build_double_spin(BOX3D_X_MAX, 0.0, 100000.0, 10.0)
        self.box_y_min_spin = self.build_double_spin(BOX3D_Y_MIN, 0.0, 100000.0, 10.0)
        self.box_y_max_spin = self.build_double_spin(BOX3D_Y_MAX, 0.0, 100000.0, 10.0)
        self.box_z_min_spin = self.build_double_spin(BOX3D_Z_MIN, 0.0, 100000.0, 10.0)
        self.box_z_max_spin = self.build_double_spin(BOX3D_Z_MAX, 0.0, 100000.0, 10.0)

        self.auto_send_tcp_checkbox = QCheckBox("自动发送到 frrjiftest.exe")
        self.auto_send_tcp_checkbox.setChecked(AUTO_SEND_TCP)
        self.tcp_send_status_label = QLabel("未发送")
        self.tcp_send_status_label.setMinimumWidth(260)

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 20)
        self.stride_spin.setValue(POINT_STRIDE)

        self.load_settings()

        self.x_scale_spin.valueChanged.connect(self.refresh_point_cloud)
        self.y_scale_spin.valueChanged.connect(self.refresh_point_cloud)
        self.z_scale_spin.valueChanged.connect(self.refresh_point_cloud)
        self.max_depth_spin.valueChanged.connect(self.refresh_point_cloud)
        self.point_size_spin.valueChanged.connect(self.refresh_point_cloud)
        self.stride_spin.valueChanged.connect(self.refresh_point_cloud)
        self.box_x_min_spin.valueChanged.connect(self.refresh_point_cloud)
        self.box_x_max_spin.valueChanged.connect(self.refresh_point_cloud)
        self.box_y_min_spin.valueChanged.connect(self.refresh_point_cloud)
        self.box_y_max_spin.valueChanged.connect(self.refresh_point_cloud)
        self.box_z_min_spin.valueChanged.connect(self.refresh_point_cloud)
        self.box_z_max_spin.valueChanged.connect(self.refresh_point_cloud)

        self.dataset_folder_edit.textChanged.connect(self.save_settings)
        self.model_path_edit.textChanged.connect(self.save_settings)
        self.mode_combo.currentIndexChanged.connect(self.save_settings)
        self.x_scale_spin.valueChanged.connect(self.save_settings)
        self.y_scale_spin.valueChanged.connect(self.save_settings)
        self.z_scale_spin.valueChanged.connect(self.save_settings)
        self.max_depth_spin.valueChanged.connect(self.save_settings)
        self.point_size_spin.valueChanged.connect(self.save_settings)
        self.stride_spin.valueChanged.connect(self.save_settings)
        self.box_x_min_spin.valueChanged.connect(self.save_settings)
        self.box_x_max_spin.valueChanged.connect(self.save_settings)
        self.box_y_min_spin.valueChanged.connect(self.save_settings)
        self.box_y_max_spin.valueChanged.connect(self.save_settings)
        self.box_z_min_spin.valueChanged.connect(self.save_settings)
        self.box_z_max_spin.valueChanged.connect(self.save_settings)
        self.local_image_combo.currentIndexChanged.connect(self.save_settings)
        self.auto_send_tcp_checkbox.stateChanged.connect(self.on_auto_send_tcp_changed)


        self.roi_center_x_label = self.build_value_label("--")
        self.roi_center_y_label = self.build_value_label("--")
        self.roi_center_z_label = self.build_value_label("--")
        self.robot_center_x_label = self.build_value_label("--")
        self.robot_center_y_label = self.build_value_label("--")
        self.robot_center_z_label = self.build_value_label("--")
        self.roi_count_label = self.build_value_label("0")

        self.start_button = QPushButton("启动")
        self.stop_button = QPushButton("停止")
        self.reset_zoom_button = QPushButton("重置缩放")
        self.reload_local_button = QPushButton("刷新本地列表")

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.reset_zoom_button.clicked.connect(self.reset_image_views)
        self.reload_local_button.clicked.connect(self.load_local_file_list)

        self.build_ui()
        self.apply_styles()
        self.load_local_file_list()
        self.start_tcp_server()

        if self.is_local_mode():
            self.status_label.setText("已启动，本地图片模式。未自动打开相机。")
        else:
            self.status_label.setText("已启动，RealSense 模式。点击“启动”打开相机。")

    def build_double_spin(self, value, min_value, max_value, step):
        spin = QDoubleSpinBox()
        spin.setRange(min_value, max_value)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setValue(value)
        return spin

    def build_value_label(self, text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setObjectName("ValueCard")
        return label

    def build_info_card(self, title, value_label):
        card = QFrame()
        card.setObjectName("InfoCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        title_label = QLabel(title)
        title_label.setObjectName("InfoTitle")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return card

    def load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            return

        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception:
            return

        self.mode_combo.blockSignals(True)
        self.dataset_folder_edit.blockSignals(True)
        self.model_path_edit.blockSignals(True)
        self.x_scale_spin.blockSignals(True)
        self.y_scale_spin.blockSignals(True)
        self.z_scale_spin.blockSignals(True)
        self.max_depth_spin.blockSignals(True)
        self.point_size_spin.blockSignals(True)
        self.stride_spin.blockSignals(True)
        self.box_x_min_spin.blockSignals(True)
        self.box_x_max_spin.blockSignals(True)
        self.box_y_min_spin.blockSignals(True)
        self.box_y_max_spin.blockSignals(True)
        self.box_z_min_spin.blockSignals(True)
        self.box_z_max_spin.blockSignals(True)
        self.auto_send_tcp_checkbox.blockSignals(True)

        try:
            self.mode_combo.setCurrentIndex(1)
            self.saved_local_image_name = str(settings.get("local_image_name", ""))
            self.dataset_folder_edit.setText(str(settings.get("dataset_folder", self.dataset_folder_edit.text())))
            self.model_path_edit.setText(str(settings.get("model_path", self.model_path_edit.text())))
            self.x_scale_spin.setValue(float(settings.get("x_scale", self.x_scale_spin.value())))
            self.y_scale_spin.setValue(float(settings.get("y_scale", self.y_scale_spin.value())))
            self.z_scale_spin.setValue(float(settings.get("z_scale", self.z_scale_spin.value())))
            self.max_depth_spin.setValue(float(settings.get("max_depth", self.max_depth_spin.value())))
            self.point_size_spin.setValue(float(settings.get("point_size", self.point_size_spin.value())))
            self.stride_spin.setValue(int(settings.get("stride", self.stride_spin.value())))
            self.box_x_min_spin.setValue(float(settings.get("box_x_min", self.box_x_min_spin.value())))
            self.box_x_max_spin.setValue(float(settings.get("box_x_max", self.box_x_max_spin.value())))
            self.box_y_min_spin.setValue(float(settings.get("box_y_min", self.box_y_min_spin.value())))
            self.box_y_max_spin.setValue(float(settings.get("box_y_max", self.box_y_max_spin.value())))
            self.box_z_min_spin.setValue(float(settings.get("box_z_min", self.box_z_min_spin.value())))
            self.box_z_max_spin.setValue(float(settings.get("box_z_max", self.box_z_max_spin.value())))
            self.auto_send_tcp_checkbox.setChecked(bool(settings.get("auto_send_tcp", self.auto_send_tcp_checkbox.isChecked())))
        finally:
            self.mode_combo.blockSignals(False)
            self.dataset_folder_edit.blockSignals(False)
            self.model_path_edit.blockSignals(False)
            self.x_scale_spin.blockSignals(False)
            self.y_scale_spin.blockSignals(False)
            self.z_scale_spin.blockSignals(False)
            self.max_depth_spin.blockSignals(False)
            self.point_size_spin.blockSignals(False)
            self.stride_spin.blockSignals(False)
            self.box_x_min_spin.blockSignals(False)
            self.box_x_max_spin.blockSignals(False)
            self.box_y_min_spin.blockSignals(False)
            self.box_y_max_spin.blockSignals(False)
            self.box_z_min_spin.blockSignals(False)
            self.box_z_max_spin.blockSignals(False)
            self.auto_send_tcp_checkbox.blockSignals(False)

    def save_settings(self, value=None):
        settings = {
            "mode_index": self.mode_combo.currentIndex(),
            "local_image_name": self.local_image_combo.currentText(),
            "dataset_folder": self.dataset_folder_edit.text(),
            "model_path": self.model_path_edit.text(),
            "x_scale": self.x_scale_spin.value(),
            "y_scale": self.y_scale_spin.value(),
            "z_scale": self.z_scale_spin.value(),
            "max_depth": self.max_depth_spin.value(),
            "point_size": self.point_size_spin.value(),
            "stride": self.stride_spin.value(),
            "box_x_min": self.box_x_min_spin.value(),
            "box_x_max": self.box_x_max_spin.value(),
            "box_y_min": self.box_y_min_spin.value(),
            "box_y_max": self.box_y_max_spin.value(),
            "box_z_min": self.box_z_min_spin.value(),
            "box_z_max": self.box_z_max_spin.value(),
            "auto_send_tcp": self.auto_send_tcp_checkbox.isChecked(),
        }

        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

    def on_auto_send_tcp_changed(self, value=None):
        self.save_settings()
        if not self.auto_send_tcp_checkbox.isChecked():
            self.pending_robot_message = ""

    def start_tcp_server(self):
        if self.tcp_server.isListening():
            return True

        address = QHostAddress(TCP_SERVER_HOST)
        ok = self.tcp_server.listen(address, TCP_SERVER_PORT)
        if ok:
            self.tcp_send_status_label.setText(
                "等待 frrjiftest.exe 连接\n%s:%d" % (TCP_SERVER_HOST, TCP_SERVER_PORT)
            )
            return True

        error_text = self.tcp_server.errorString()
        self.tcp_send_status_label.setText("监听失败\n%s" % error_text)
        self.status_label.setText("TCP Server 启动失败：%s" % error_text)
        QMessageBox.warning(
            self,
            "TCP Server 启动失败",
            "无法监听 %s:%d。\n\n"
            "请先关闭 NetAssist 或其他占用 5678 端口的程序，然后重新启动 main.py。\n\n%s"
            % (TCP_SERVER_HOST, TCP_SERVER_PORT, error_text),
        )
        return False

    def accept_frrjiftest_connection(self):
        while self.tcp_server.hasPendingConnections():
            new_client = self.tcp_server.nextPendingConnection()

            if self.frrjiftest_client is not None:
                self.frrjiftest_client.close()

            self.frrjiftest_client = new_client
            self.frrjiftest_client.disconnected.connect(self.on_frrjiftest_disconnected)
            self.last_tcp_error_message = ""
            self.tcp_send_status_label.setText("frrjiftest.exe 已连接")
            self.status_label.setText("frrjiftest.exe 已连接，可以发送机械臂坐标")

            if self.auto_send_tcp_checkbox.isChecked() and self.pending_robot_message != "":
                pending_text = self.pending_robot_message
                ok, message = self.send_tcp_text(pending_text)
                if ok:
                    self.tcp_send_status_label.setText(
                        "连接成功，已自动发送\n%s" % pending_text
                    )
                    self.status_label.setText(
                        "已发送到 frrjiftest.exe：%s" % pending_text
                    )
                    self.pending_robot_message = ""
                    if not self.is_local_mode() and pending_text == self.camera_first_message:
                        self.camera_coordinate_sent = True
                else:
                    self.tcp_send_status_label.setText("发送失败\n%s" % message)

    def on_frrjiftest_disconnected(self):
        disconnected_client = self.sender()
        if disconnected_client == self.frrjiftest_client:
            self.frrjiftest_client = None

        self.tcp_send_status_label.setText(
            "等待 frrjiftest.exe 连接\n%s:%d" % (TCP_SERVER_HOST, TCP_SERVER_PORT)
        )

    def close_tcp_server(self):
        if self.frrjiftest_client is not None:
            self.frrjiftest_client.close()
            self.frrjiftest_client = None

        if self.tcp_server.isListening():
            self.tcp_server.close()

    def send_tcp_text(self, text):
        if not self.tcp_server.isListening():
            return False, "main.py 的 TCP Server 未启动"

        # 相机模式每次启动最多只允许真正写入一次 TCP 数据。
        if not self.is_local_mode() and self.camera_tcp_write_count >= 1:
            return True, "本次相机启动已经发送过第一组坐标"

        if self.frrjiftest_client is None:
            return False, "frrjiftest.exe 未连接，请在该程序中点击“启动客户端”"

        if self.frrjiftest_client.state() != QAbstractSocket.ConnectedState:
            self.frrjiftest_client = None
            return False, "frrjiftest.exe 连接已断开，请重新点击“启动客户端”"

        # 严格使用英文冒号和英文分号，不添加其他分隔符。
        send_data = text.encode("ascii")
        write_count = self.frrjiftest_client.write(send_data)
        self.frrjiftest_client.flush()

        if write_count == -1:
            return False, self.frrjiftest_client.errorString()

        if not self.is_local_mode():
            self.camera_tcp_write_count = self.camera_tcp_write_count + 1

        return True, "发送成功"

    def get_box3d_bounds(self):
        x_min = self.box_x_min_spin.value()
        x_max = self.box_x_max_spin.value()
        y_min = self.box_y_min_spin.value()
        y_max = self.box_y_max_spin.value()
        z_min = self.box_z_min_spin.value()
        z_max = self.box_z_max_spin.value()

        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        if z_min > z_max:
            z_min, z_max = z_max, z_min

        return x_min, x_max, y_min, y_max, z_min, z_max

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(12)

        source_group = QGroupBox("数据源")
        source_layout = QGridLayout(source_group)
        source_layout.addWidget(QLabel("模式"), 0, 0)
        source_layout.addWidget(self.mode_combo, 0, 1)
        source_layout.addWidget(QLabel("数据集文件夹"), 1, 0)
        source_layout.addWidget(self.dataset_folder_edit, 1, 1)
        source_layout.addWidget(self.select_dataset_button, 1, 2)
        source_layout.addWidget(QLabel("本地彩色图"), 2, 0)
        source_layout.addWidget(self.local_image_combo, 2, 1, 1, 2)
        source_layout.addWidget(self.reload_local_button, 3, 2)

        model_group = QGroupBox("模型设置")
        model_layout = QGridLayout(model_group)
        model_layout.addWidget(QLabel("模型路径"), 0, 0)
        model_layout.addWidget(self.model_path_edit, 0, 1)
        model_layout.addWidget(self.load_model_button, 0, 2)
        model_layout.addWidget(QLabel("说明"), 1, 0)
        model_layout.addWidget(QLabel("彩色图做实例分割，深度图同步叠加分割结果。"), 1, 1, 1, 2)

        point_group = QGroupBox("点云参数")
        point_layout = QGridLayout(point_group)
        point_layout.addWidget(QLabel("X 比例"), 0, 0)
        point_layout.addWidget(self.x_scale_spin, 0, 1)
        point_layout.addWidget(QLabel("Y 比例"), 0, 2)
        point_layout.addWidget(self.y_scale_spin, 0, 3)
        point_layout.addWidget(QLabel("Z 比例"), 0, 4)
        point_layout.addWidget(self.z_scale_spin, 0, 5)
        point_layout.addWidget(QLabel("最大深度(mm)"), 1, 0)
        point_layout.addWidget(self.max_depth_spin, 1, 1)
        point_layout.addWidget(QLabel("点采样步长"), 1, 2)
        point_layout.addWidget(self.stride_spin, 1, 3)
        point_layout.addWidget(QLabel("整图点大小"), 1, 4)
        point_layout.addWidget(self.point_size_spin, 1, 5)

        box_group = QGroupBox("BOX3D ROI")
        box_layout = QGridLayout(box_group)
        box_layout.addWidget(QLabel("X 最小"), 0, 0)
        box_layout.addWidget(self.box_x_min_spin, 0, 1)
        box_layout.addWidget(QLabel("X 最大"), 0, 2)
        box_layout.addWidget(self.box_x_max_spin, 0, 3)
        box_layout.addWidget(QLabel("Y 最小"), 1, 0)
        box_layout.addWidget(self.box_y_min_spin, 1, 1)
        box_layout.addWidget(QLabel("Y 最大"), 1, 2)
        box_layout.addWidget(self.box_y_max_spin, 1, 3)
        box_layout.addWidget(QLabel("Z 最小"), 2, 0)
        box_layout.addWidget(self.box_z_min_spin, 2, 1)
        box_layout.addWidget(QLabel("Z 最大"), 2, 2)
        box_layout.addWidget(self.box_z_max_spin, 2, 3)

        roi_group = QGroupBox("ROI 中心坐标")
        roi_layout = QHBoxLayout(roi_group)
        roi_layout.addWidget(self.build_info_card("中心 X (mm)", self.roi_center_x_label))
        roi_layout.addWidget(self.build_info_card("中心 Y (mm)", self.roi_center_y_label))
        roi_layout.addWidget(self.build_info_card("中心 Z (mm)", self.roi_center_z_label))
        roi_layout.addWidget(self.build_info_card("ROI 点数", self.roi_count_label))

        robot_coord_group = QGroupBox("转换后的机械臂坐标")
        robot_coord_layout = QHBoxLayout(robot_coord_group)
        robot_coord_layout.addWidget(self.build_info_card("机械臂 X (mm)", self.robot_center_x_label))
        robot_coord_layout.addWidget(self.build_info_card("机械臂 Y (mm)", self.robot_center_y_label))
        robot_coord_layout.addWidget(self.build_info_card("机械臂 Z (mm)", self.robot_center_z_label))
        robot_coord_layout.addWidget(self.auto_send_tcp_checkbox)
        robot_coord_layout.addWidget(self.build_info_card("发送状态", self.tcp_send_status_label))

        top_left_layout = QVBoxLayout()
        top_left_layout.addWidget(source_group)
        top_left_layout.addWidget(model_group)

        top_right_layout = QVBoxLayout()
        top_right_layout.addWidget(point_group)
        top_right_layout.addWidget(box_group)
        top_right_layout.addWidget(roi_group)
        top_right_layout.addWidget(robot_coord_group)

        top_layout.addLayout(top_left_layout, 3)
        top_layout.addLayout(top_right_layout, 4)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.reset_zoom_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.status_label)

        image_splitter = QSplitter(Qt.Horizontal)
        image_splitter.addWidget(self.wrap_widget("彩色图", self.color_view))
        image_splitter.addWidget(self.wrap_widget("深度图", self.depth_view))
        image_splitter.setSizes([700, 700])

        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.addWidget(image_splitter)
        content_splitter.addWidget(self.wrap_widget("点云", self.plotter_host))
        content_splitter.setSizes([1200, 720])

        main_layout = QVBoxLayout(root)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(content_splitter, 1)

    def apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                font-size: 17px;
            }
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 1px solid #cfd8dc;
                border-radius: 10px;
                margin-top: 10px;
                background: #f8fafc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px 0 6px;
            }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
                min-height: 40px;
                border: 1px solid #c7d0d9;
                border-radius: 6px;
                padding-left: 8px;
                background: white;
            }
            QPushButton {
                min-height: 42px;
                padding: 0 18px;
                border-radius: 6px;
                background: #1976d2;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1565c0;
            }
            QFrame#InfoCard {
                border: 1px solid #d9e2ec;
                border-radius: 10px;
                background: white;
            }
            QLabel#InfoTitle {
                color: #607d8b;
                font-size: 15px;
            }
            QLabel#ValueCard {
                font-size: 26px;
                font-weight: bold;
                color: #0f172a;
            }
            """
        )

    def wrap_widget(self, title, widget):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(widget, 1)
        return panel

    def ensure_plotter(self):
        if self.plotter is not None:
            return True

        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
        except Exception as exc:
            QMessageBox.warning(self, "错误", "点云窗口初始化失败：\n%s" % exc)
            return False

        self.pv = pv
        self.plotter = QtInteractor(self.plotter_host)

        self.clear_layout(self.plotter_layout)
        self.plotter_layout.addWidget(self.plotter)
        self.plotter.set_background("#101418")
        self.plotter.add_axes()
        self.plotter.show_grid(color="gray")
        return True

    def clear_layout(self, layout):
        while layout.count() > 0:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def ensure_yolo_model(self):
        model_path = self.model_path_edit.text().strip()

        if model_path == "":
            return False

        if not os.path.exists(model_path):
            self.status_label.setText("模型文件不存在：%s" % model_path)
            return False

        if self.yolo_model is not None and self.loaded_model_path == model_path:
            return True

        try:
            from ultralytics import YOLO
        except Exception as exc:
            QMessageBox.warning(self, "错误", "导入 YOLO 失败：\n%s" % exc)
            return False

        try:
            self.yolo_model = YOLO(model_path)
            self.loaded_model_path = model_path
            self.status_label.setText("模型已加载：%s" % model_path)
            return True
        except Exception as exc:
            QMessageBox.warning(self, "错误", "加载模型失败：\n%s" % exc)
            return False

    def load_yolo_model(self):
        if self.ensure_yolo_model() and self.last_frame is not None:
            self.process_frame(self.last_frame)

    def is_local_mode(self):
        return self.mode_combo.currentIndex() == 1

    def on_mode_changed(self):
        if self.is_local_mode():
            self.stop_stream()
            self.show_current_local_image()
            self.status_label.setText("当前模式：本地图片")
        else:
            self.status_label.setText("当前模式：RealSense，相机未自动启动。点击“启动”。")

    def get_dataset_folder(self):
        return self.dataset_folder_edit.text().strip()

    def get_color_folder(self):
        return os.path.join(self.get_dataset_folder(), "color")

    def get_depth_folder(self):
        return os.path.join(self.get_dataset_folder(), "depth")

    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据集文件夹", self.get_dataset_folder())
        if folder == "":
            return

        self.dataset_folder_edit.setText(folder)
        self.load_local_file_list()
        if self.is_local_mode():
            self.show_current_local_image()

    def load_local_file_list(self):
        self.local_image_combo.blockSignals(True)
        self.local_image_combo.clear()

        color_folder = self.get_color_folder()
        if os.path.isdir(color_folder):
            names = os.listdir(color_folder)
            names.sort()
            for name in names:
                full_path = os.path.join(color_folder, name)
                if os.path.isfile(full_path):
                    lower_name = name.lower()
                    if lower_name.endswith(".png") or lower_name.endswith(".jpg") or lower_name.endswith(".jpeg") or lower_name.endswith(".bmp"):
                        self.local_image_combo.addItem(name)

        if self.saved_local_image_name != "":
            image_index = self.local_image_combo.findText(self.saved_local_image_name)
            if image_index >= 0:
                self.local_image_combo.setCurrentIndex(image_index)

        self.local_image_combo.blockSignals(False)

    def on_local_image_changed(self):
        if self.is_local_mode():
            self.show_current_local_image()

    def build_local_depth_path(self, color_name):
        if color_name.startswith("color_"):
            depth_name = color_name.replace("color_", "depth_", 1)
        else:
            depth_name = color_name
        return os.path.join(self.get_depth_folder(), depth_name)

    def load_local_frame(self, color_name):
        color_path = os.path.join(self.get_color_folder(), color_name)
        depth_path = self.build_local_depth_path(color_name)

        if not os.path.exists(color_path):
            QMessageBox.warning(self, "错误", "彩色图不存在：\n%s" % color_path)
            return None

        if not os.path.exists(depth_path):
            QMessageBox.warning(self, "错误", "没有找到对应深度图：\n%s" % depth_path)
            return None

        color_bgr = read_image_with_chinese_path(color_path, cv2.IMREAD_COLOR)
        depth_raw = read_image_with_chinese_path(depth_path, cv2.IMREAD_UNCHANGED)

        if color_bgr is None:
            QMessageBox.warning(self, "错误", "彩色图读取失败：\n%s" % color_path)
            return None

        if depth_raw is None:
            QMessageBox.warning(self, "错误", "深度图读取失败：\n%s" % depth_path)
            return None

        if len(depth_raw.shape) == 3:
            channel_count = depth_raw.shape[2]
            if channel_count == 3:
                depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
            elif channel_count == 4:
                depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGRA2GRAY)
            elif channel_count == 1:
                depth_raw = depth_raw[:, :, 0]

        if depth_raw.shape[0] != color_bgr.shape[0] or depth_raw.shape[1] != color_bgr.shape[1]:
            depth_raw = cv2.resize(depth_raw, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        depth_m = depth_raw.astype(np.float32) * LOCAL_DEPTH_SCALE

        return FrameBundle(
            color_bgr,
            depth_raw,
            depth_m,
            LOCAL_FX,
            LOCAL_FY,
            LOCAL_CX,
            LOCAL_CY,
        )

    def show_current_local_image(self):
        if self.local_image_combo.count() == 0:
            self.status_label.setText("本地图片列表为空：%s" % self.get_dataset_folder())
            self.update_roi_info(None, 0)
            return

        color_name = self.local_image_combo.currentText()
        frame = self.load_local_frame(color_name)
        if frame is None:
            self.update_roi_info(None, 0)
            return

        self.status_label.setText("本地图片模式：%s" % color_name)
        self.process_frame(frame)

    def start_stream(self):
        if not self.start_tcp_server():
            return

        if self.is_local_mode():
            self.show_current_local_image()
            return

        if self.timer.isActive():
            return

        self.pending_robot_message = ""
        self.camera_first_message = ""
        self.camera_coordinate_sent = False
        self.camera_tcp_write_count = 0

        try:
            self.reader.start()
        except Exception as exc:
            QMessageBox.critical(self, "启动失败", "RealSense 启动失败：\n%s" % exc)
            return

        self.frame_count = 0
        self.last_fps_frame_count = 0
        self.last_fps_time = time.time()
        self.timer.start(30)
        self.status_label.setText("相机已启动")

    def stop_stream(self):
        self.timer.stop()
        self.reader.stop()
        if self.is_local_mode():
            self.status_label.setText("当前模式：本地图片")
        else:
            self.status_label.setText("相机已停止")

    def reset_image_views(self):
        self.color_view.reset_view()
        self.depth_view.reset_view()

    def update_stream(self):
        frame = self.reader.read()
        if frame is None:
            return

        self.frame_count = self.frame_count + 1
        self.process_frame(frame)

        elapsed = time.time() - self.last_fps_time
        if elapsed >= 1.0:
            fps = (self.frame_count - self.last_fps_frame_count) / elapsed
            self.status_label.setText(
                "相机已启动 | 分辨率 %dx%d | FPS %.1f"
                % (frame.color_bgr.shape[1], frame.color_bgr.shape[0], fps)
            )
            self.last_fps_time = time.time()
            self.last_fps_frame_count = self.frame_count

    def process_frame(self, frame):
        self.last_frame = frame

        color_image = frame.color_bgr.copy()
        depth_image = cv2.applyColorMap(
            cv2.convertScaleAbs(frame.depth_raw, alpha=DEPTH_VIS_ALPHA),
            cv2.COLORMAP_JET,
        )

        mask = None
        if self.ensure_yolo_model():
            mask, color_image, depth_image = self.run_segmentation(frame.color_bgr, depth_image)

        self.last_mask = mask
        self.color_view.set_image(color_image)
        self.depth_view.set_image(depth_image)
        self.refresh_point_cloud()

    def run_segmentation(self, color_bgr, depth_vis):
        union_mask = np.zeros((color_bgr.shape[0], color_bgr.shape[1]), dtype=np.uint8)
        color_draw = color_bgr.copy()
        depth_draw = depth_vis.copy()

        try:
            results = self.yolo_model.predict(color_bgr, conf=YOLO_CONFIDENCE, verbose=False)
        except Exception as exc:
            self.status_label.setText("分割失败：%s" % exc)
            return None, color_draw, depth_draw

        if len(results) == 0:
            return None, color_draw, depth_draw

        result = results[0]
        if result.masks is None or result.boxes is None:
            return None, color_draw, depth_draw

        for i in range(len(result.boxes)):
            box = result.boxes[i]
            class_id = int(box.cls)
            class_name = result.names[class_id]
            score = float(box.conf)

            mask = result.masks.data[i].cpu().numpy()
            if mask.shape[0] != color_bgr.shape[0] or mask.shape[1] != color_bgr.shape[1]:
                mask = cv2.resize(mask, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            mask_binary = np.zeros_like(union_mask)
            mask_binary[mask > 0.5] = 255
            union_mask = cv2.bitwise_or(union_mask, mask_binary)

            overlay_color = np.zeros_like(color_draw)
            overlay_color[:, :, 1] = 180
            color_draw = np.where(
                mask_binary[:, :, None] > 0,
                cv2.addWeighted(color_draw, 0.6, overlay_color, 0.4, 0),
                color_draw,
            )

            overlay_depth = np.zeros_like(depth_draw)
            overlay_depth[:, :, 2] = 255
            depth_draw = np.where(
                mask_binary[:, :, None] > 0,
                cv2.addWeighted(depth_draw, 0.6, overlay_depth, 0.4, 0),
                depth_draw,
            )

            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color_draw, contours, -1, (0, 255, 0), 2)
            cv2.drawContours(depth_draw, contours, -1, (255, 255, 255), 2)

            xyxy = box.xyxy[0].tolist()
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            text = "%s %.2f" % (class_name, score)
            cv2.putText(color_draw, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(depth_draw, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if np.count_nonzero(union_mask) == 0:
            return None, color_draw, depth_draw

        return union_mask, color_draw, depth_draw

    def refresh_point_cloud(self):
        if self.last_frame is None:
            return

        if not self.ensure_plotter():
            return

        box3d_bounds = self.get_box3d_bounds()

        all_points, all_colors = build_point_cloud(
            self.last_frame,
            self.stride_spin.value(),
            self.max_depth_spin.value(),
            self.x_scale_spin.value(),
            self.y_scale_spin.value(),
            self.z_scale_spin.value(),
            None,
            False,
            box3d_bounds,
        )

        roi_points, roi_colors = build_point_cloud(
            self.last_frame,
            self.stride_spin.value(),
            self.max_depth_spin.value(),
            self.x_scale_spin.value(),
            self.y_scale_spin.value(),
            self.z_scale_spin.value(),
            self.last_mask,
            True,
            box3d_bounds,
        )

        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.show_grid(color="gray")

        if len(all_points) > 0:
            all_cloud = self.pv.PolyData(all_points)
            all_cloud["rgb"] = all_colors
            self.plotter.add_points(
                all_cloud,
                scalars="rgb",
                rgb=True,
                point_size=self.point_size_spin.value(),
                render_points_as_spheres=True,
            )

        roi_center_points = build_roi_center_points(
            self.last_frame,
            self.last_mask,
            1.0,
            1.0,
            1.0,
            box3d_bounds,
        )

        if len(roi_points) > 0:
            roi_cloud = self.pv.PolyData(roi_points)
            roi_cloud["rgb"] = roi_colors
            self.plotter.add_points(
                roi_cloud,
                scalars="rgb",
                rgb=True,
                point_size=ROI_POINT_SIZE,
                render_points_as_spheres=True,
            )

            min_x = float(np.min(roi_points[:, 0]))
            max_x = float(np.max(roi_points[:, 0]))
            min_y = float(np.min(roi_points[:, 1]))
            max_y = float(np.max(roi_points[:, 1]))
            min_z = float(np.min(roi_points[:, 2]))
            max_z = float(np.max(roi_points[:, 2]))

            box = self.pv.Box(bounds=(min_x, max_x, min_y, max_y, min_z, max_z))
            self.plotter.add_mesh(
                box,
                color=(0.0, 1.0, 0.0),
                style="wireframe",
                line_width=2,
            )

        if roi_center_points is not None and len(roi_center_points) > 0:
            center = calculate_center_xyz(roi_center_points)
            self.update_roi_info(center, len(roi_center_points))
        else:
            self.update_roi_info(None, 0)

        self.plotter.reset_camera()
        self.plotter.render()

    def update_roi_info(self, center, point_count):
        self.last_roi_center = center
        self.last_roi_count = point_count

        if center is None:
            if self.is_local_mode():
                self.pending_robot_message = ""
            self.roi_center_x_label.setText("--")
            self.roi_center_y_label.setText("--")
            self.roi_center_z_label.setText("--")
            self.robot_center_x_label.setText("--")
            self.robot_center_y_label.setText("--")
            self.robot_center_z_label.setText("--")
            self.tcp_send_status_label.setText("未发送：坐标为空")
            self.roi_count_label.setText("0")
            self.last_robot_center = None
            return

        self.roi_center_x_label.setText("%.3f" % center[0])
        self.roi_center_y_label.setText("%.3f" % center[1])
        self.roi_center_z_label.setText("%.3f" % center[2])
        self.roi_count_label.setText(str(point_count))

        if self.calibration_transform is None:
            self.calibration_transform = load_calibration_transform(CALIBRATION_TRANSFORM_FILE)

        robot_center = transform_image_point_to_robot(center, self.calibration_transform)
        self.last_robot_center = robot_center

        if robot_center is None:
            if self.is_local_mode():
                self.pending_robot_message = ""
            self.robot_center_x_label.setText("--")
            self.robot_center_y_label.setText("--")
            self.robot_center_z_label.setText("--")
            self.tcp_send_status_label.setText("未发送：机械臂坐标为空")
        else:
            self.robot_center_x_label.setText("%.3f" % robot_center[0])
            self.robot_center_y_label.setText("%.3f" % robot_center[1])
            self.robot_center_z_label.setText("%.3f" % robot_center[2])

            if self.auto_send_tcp_checkbox.isChecked():
                send_text = format_robot_point_message(robot_center)
                should_send = True

                if not self.is_local_mode():
                    if self.camera_first_message == "":
                        self.camera_first_message = send_text
                        self.pending_robot_message = send_text

                    send_text = self.camera_first_message

                    if self.camera_coordinate_sent:
                        should_send = False
                        self.tcp_send_status_label.setText(
                            "本次相机启动已发送第一组坐标\n%s" % send_text
                        )
                else:
                    self.pending_robot_message = send_text

                if should_send:
                    ok, message = self.send_tcp_text(send_text)
                    if ok:
                        self.pending_robot_message = ""
                        if not self.is_local_mode():
                            self.camera_coordinate_sent = True

                        self.tcp_send_status_label.setText("发送成功\n%s" % send_text)
                        self.status_label.setText("已发送到 frrjiftest.exe：%s" % send_text)
                        self.last_tcp_error_message = ""

                        if not self.tcp_success_notice_shown:
                            self.tcp_success_notice_shown = True
                            QMessageBox.information(
                                self,
                                "坐标发送成功",
                                "已发送到 frrjiftest.exe：\n%s\n\n"
                                "请在 frrjiftest.exe 的“接收内容”中查看。" % send_text,
                            )
                    else:
                        self.tcp_send_status_label.setText("发送失败\n%s" % message)
                        self.status_label.setText("发送失败：%s" % message)

                        if message != self.last_tcp_error_message:
                            self.last_tcp_error_message = message
                            QMessageBox.warning(
                                self,
                                "网络发送失败",
                                "机械臂坐标已经计算完成，但没有发送成功。\n\n%s" % message,
                            )
            else:
                self.tcp_send_status_label.setText("未发送：自动发送关闭")

    def closeEvent(self, event):
        self.save_settings()
        self.close_tcp_server()
        self.stop_stream()
        QMainWindow.closeEvent(self, event)


def build_point_cloud(frame, stride, max_depth_m, scale_x, scale_y, scale_z, mask, roi_mode, box3d_bounds):
    depth = frame.depth_m[::stride, ::stride]
    color = frame.color_bgr[::stride, ::stride]

    height = depth.shape[0]
    width = depth.shape[1]

    uu, vv = np.meshgrid(np.arange(width) * stride, np.arange(height) * stride)
    z = depth
    valid = (z > 0.0) & (z <= max_depth_m)

    if mask is not None:
        mask_small = mask[::stride, ::stride]
        valid = valid & (mask_small > 0)

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    x = uu * z / frame.fx
    y = vv * z / frame.fy

    x = x * scale_x
    y = y * scale_y
    z = z * scale_z

    if box3d_bounds is not None:
        box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = box3d_bounds
        valid = valid & (x >= box_x_min) & (x <= box_x_max)
        valid = valid & (y >= box_y_min) & (y <= box_y_max)
        valid = valid & (z >= box_z_min) & (z <= box_z_max)

    points = np.stack((x, y, z), axis=-1)
    points = points[valid].astype(np.float32)

    colors = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    colors = colors[valid]

    if roi_mode and len(colors) > 0:
        colors = np.zeros_like(colors)
        colors[:, 0] = 60
        colors[:, 1] = 255
        colors[:, 2] = 60

    return points, colors


def build_roi_center_points(frame, mask, scale_x, scale_y, scale_z, box3d_bounds):
    if mask is None:
        return None

    depth = frame.depth_m
    height = depth.shape[0]
    width = depth.shape[1]

    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    valid = (z > 0.0) & (mask > 0)

    if not np.any(valid):
        return None

    x = uu * z / frame.fx
    y = vv * z / frame.fy

    x = x * scale_x
    y = y * scale_y
    z = z * scale_z

    if box3d_bounds is not None:
        box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = box3d_bounds
        valid = valid & (x >= box_x_min) & (x <= box_x_max)
        valid = valid & (y >= box_y_min) & (y <= box_y_max)
        valid = valid & (z >= box_z_min) & (z <= box_z_max)

    if not np.any(valid):
        return None

    points = np.stack((x, y, z), axis=-1)
    points = points[valid].astype(np.float32)

    if len(points) == 0:
        return None

    return points


def calculate_center_xyz(points):
    if len(points) == 0:
        return None

    center_x = float(np.mean(points[:, 0]))
    center_y = float(np.mean(points[:, 1]))
    center_z = float(np.mean(points[:, 2]))
    return center_x, center_y, center_z


def load_calibration_transform(file_path):
    if not os.path.exists(file_path):
        return None

    rows = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line == "":
                    continue

                parts = line.replace(",", "\t").split()
                if len(parts) != 4:
                    return None

                row = []
                for i in range(4):
                    row.append(float(parts[i]))
                rows.append(row)
    except Exception:
        return None

    if len(rows) != 4:
        return None

    return np.array(rows, dtype=np.float64)


def transform_image_point_to_robot(image_point, transform):
    if image_point is None:
        return None

    if transform is None:
        return None

    x = float(image_point[0])
    y = float(image_point[1])
    z = float(image_point[2])

    p = np.array([x, y, z, 1.0], dtype=np.float64)
    q = np.dot(transform, p)

    return float(q[0]), float(q[1]), float(q[2])


def format_robot_point_message(robot_point):
    x = float(robot_point[0])
    y = float(robot_point[1])
    z = float(robot_point[2])
    return "X:%.3f;Y:%.3f;Z:%.3f" % (x, y, z)


def read_image_with_chinese_path(file_path, flags):
    if not os.path.exists(file_path):
        return None

    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        image = cv2.imdecode(data, flags)
        return image
    except Exception:
        return None


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
