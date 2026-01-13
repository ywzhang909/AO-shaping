from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QLabel, QTabWidget
)
from PySide6.QtCore import Signal


class ControlPanel(QWidget):
    """控制面板类"""
    
    startRequested = Signal()
    stopRequested = Signal()
    resetRequested = Signal()
    
    def __init__(self, simulation_manager, parent=None):
        super().__init__(parent)
        self.simulation_manager = simulation_manager
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标题
        title_label = QLabel('控制面板')
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # 创建选项卡控件
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # 算法设置选项卡
        algorithm_tab = QWidget()
        algorithm_layout = QVBoxLayout(algorithm_tab)
        
        # 创建算法选择组
        algorithm_group = QGroupBox("算法选择")
        algorithm_form = QFormLayout(algorithm_group)
        
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "波前优化 (wf)",
            "轴向光束优化 (pib)",
            "组合优化 (combine)",
            "贝叶斯优化 (bayes-opt)",
            "启发式搜索 (heuristic)"
        ])
        algorithm_form.addRow("算法:", self.algorithm_combo)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["pso", "ga", "sa", "de"])
        self.method_combo.setCurrentText("pso")
        algorithm_form.addRow("启发式方法:", self.method_combo)
        
        algorithm_layout.addWidget(algorithm_group)
        
        # 创建参数设置组
        params_group = QGroupBox("参数设置")
        params_form = QFormLayout(params_group)
        
        # 通用参数
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.setValue(20000)
        params_form.addRow("迭代次数:", self.epochs_spin)
        
        self.debug_checkbox = QCheckBox("调试模式")
        params_form.addRow("调试:", self.debug_checkbox)
        
        # 实时优化选项
        # TODO : 如果设备没有连接，则无法启用实时优化模式
        self.realtime_optimization_checkbox = QCheckBox("实时PIB优化模式")
        self.realtime_optimization_checkbox.setChecked(True)  # 默认启用
        params_form.addRow("实时优化:", self.realtime_optimization_checkbox)
        
        # 波前优化参数
        self.wfs_res_combo = QComboBox()
        self.wfs_res_combo.addItems(["768", "512"])
        self.wfs_res_combo.setCurrentText("768")
        params_form.addRow("WFS分辨率:", self.wfs_res_combo)
        
        self.pupil_diameter_spin = QDoubleSpinBox()
        self.pupil_diameter_spin.setRange(0.1, 10.0)
        self.pupil_diameter_spin.setValue(2.7)
        self.pupil_diameter_spin.setSingleStep(0.1)
        params_form.addRow("瞳孔直径:", self.pupil_diameter_spin)
        
        self.early_stop_threshold_spin = QDoubleSpinBox()
        self.early_stop_threshold_spin.setRange(0.0, 1.0)
        self.early_stop_threshold_spin.setValue(0.0)
        self.early_stop_threshold_spin.setSingleStep(0.01)
        params_form.addRow("早停阈值:", self.early_stop_threshold_spin)
        
        algorithm_layout.addWidget(params_group)
        tab_widget.addTab(algorithm_tab, "算法设置")
        
        # CCD设置选项卡
        ccd_tab = QWidget()
        ccd_layout = QVBoxLayout(ccd_tab)
        
        ccd_group = QGroupBox("CCD相机设置")
        ccd_form = QFormLayout(ccd_group)
        
        self.cam_id_spin = QSpinBox()
        self.cam_id_spin.setRange(0, 10)
        self.cam_id_spin.setValue(0)
        ccd_form.addRow("相机ID:", self.cam_id_spin)
        
        self.exposure_time_spin = QSpinBox()
        self.exposure_time_spin.setRange(1, 10000)
        self.exposure_time_spin.setValue(60)
        ccd_form.addRow("曝光时间(ms):", self.exposure_time_spin)
        
        self.cam_size_spin = QSpinBox()
        self.cam_size_spin.setRange(50, 1000)
        self.cam_size_spin.setValue(200)
        ccd_form.addRow("相机窗口大小:", self.cam_size_spin)
        
        self.target_max_brightness_spin = QSpinBox()
        self.target_max_brightness_spin.setRange(0, 255)
        self.target_max_brightness_spin.setValue(90)
        ccd_form.addRow("目标最大亮度:", self.target_max_brightness_spin)
        
        ccd_layout.addWidget(ccd_group)
        tab_widget.addTab(ccd_tab, "CCD设置")
        
        # 变形镜设置选项卡
        dm_tab = QWidget()
        dm_layout = QVBoxLayout(dm_tab)
        
        dm_group = QGroupBox("变形镜设置")
        dm_form = QFormLayout(dm_group)
        
        self.delta_spin = QDoubleSpinBox()
        self.delta_spin.setRange(0.1, 10.0)
        self.delta_spin.setValue(2.0)
        self.delta_spin.setSingleStep(0.1)
        dm_form.addRow("步长(delta):", self.delta_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0, 10.0)
        self.lr_spin.setValue(0.0)
        self.lr_spin.setSingleStep(0.1)
        dm_form.addRow("学习率(lr):", self.lr_spin)
        
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setValue(0.0)
        self.weight_decay_spin.setSingleStep(0.01)
        dm_form.addRow("权重衰减:", self.weight_decay_spin)
        
        self.shrink_iter_spin = QSpinBox()
        self.shrink_iter_spin.setRange(0, 10000)
        self.shrink_iter_spin.setValue(300)
        dm_form.addRow("收缩迭代:", self.shrink_iter_spin)
        
        self.shrink_ratio_spin = QDoubleSpinBox()
        self.shrink_ratio_spin.setRange(0.1, 1.0)
        self.shrink_ratio_spin.setValue(0.8)
        self.shrink_ratio_spin.setSingleStep(0.1)
        dm_form.addRow("收缩比率:", self.shrink_ratio_spin)
        
        self.dm_unit_mask_combo = QComboBox()
        self.dm_unit_mask_combo.addItems(["all", "inner", "outer"])
        self.dm_unit_mask_combo.setCurrentText("all")
        dm_form.addRow("单元掩码:", self.dm_unit_mask_combo)
        
        dm_layout.addWidget(dm_group)
        tab_widget.addTab(dm_tab, "变形镜设置")
        
        # 优化器设置选项卡
        optimizer_tab = QWidget()
        optimizer_layout = QVBoxLayout(optimizer_tab)
        
        optimizer_group = QGroupBox("优化器设置")
        optimizer_form = QFormLayout(optimizer_group)
        
        self.r_bucket_spin = QDoubleSpinBox()
        self.r_bucket_spin.setRange(0, 100)
        self.r_bucket_spin.setValue(0)
        self.r_bucket_spin.setSingleStep(0.1)
        optimizer_form.addRow("半径桶大小:", self.r_bucket_spin)
        
        self.center_combo = QComboBox()
        self.center_combo.addItems(["mass", "max", "shape"])
        self.center_combo.setCurrentText("mass")
        optimizer_form.addRow("中心位置:", self.center_combo)
        
        optimizer_layout.addWidget(optimizer_group)
        tab_widget.addTab(optimizer_tab, "优化器设置")
        
        # 贝叶斯优化设置选项卡
        bayes_tab = QWidget()
        bayes_layout = QVBoxLayout(bayes_tab)
        
        bayes_group = QGroupBox("贝叶斯优化设置")
        bayes_form = QFormLayout(bayes_group)
        
        self.n_calls_spin = QSpinBox()
        self.n_calls_spin.setRange(1, 1000)
        self.n_calls_spin.setValue(30)
        bayes_form.addRow("调用次数:", self.n_calls_spin)
        
        self.lr_min_spin = QDoubleSpinBox()
        self.lr_min_spin.setRange(0.01, 10.0)
        self.lr_min_spin.setValue(0.1)
        self.lr_min_spin.setSingleStep(0.1)
        bayes_form.addRow("学习率最小值:", self.lr_min_spin)
        
        self.lr_max_spin = QDoubleSpinBox()
        self.lr_max_spin.setRange(0.01, 10.0)
        self.lr_max_spin.setValue(5.0)
        self.lr_max_spin.setSingleStep(0.1)
        bayes_form.addRow("学习率最大值:", self.lr_max_spin)
        
        self.delta_min_spin = QDoubleSpinBox()
        self.delta_min_spin.setRange(0.01, 10.0)
        self.delta_min_spin.setValue(0.1)
        self.delta_min_spin.setSingleStep(0.1)
        bayes_form.addRow("Delta最小值:", self.delta_min_spin)
        
        self.delta_max_spin = QDoubleSpinBox()
        self.delta_max_spin.setRange(0.01, 10.0)
        self.delta_max_spin.setValue(5.0)
        self.delta_max_spin.setSingleStep(0.1)
        bayes_form.addRow("Delta最大值:", self.delta_max_spin)
        
        self.method_type_combo = QComboBox()
        self.method_type_combo.addItems(["bayes", "grid"])
        self.method_type_combo.setCurrentText("bayes")
        bayes_form.addRow("优化方法:", self.method_type_combo)
        
        bayes_layout.addWidget(bayes_group)
        tab_widget.addTab(bayes_tab, "贝叶斯优化")
        
        # 创建操作按钮组
        buttons_group = QGroupBox("操作")
        buttons_layout = QVBoxLayout(buttons_group)
        
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.on_start_clicked)
        buttons_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.on_stop_clicked)
        buttons_layout.addWidget(self.stop_button)
        
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.on_reset_clicked)
        buttons_layout.addWidget(self.reset_button)
        
        layout.addWidget(buttons_group)
        
    def on_start_clicked(self):
        """处理开始按钮点击事件"""
        self.startRequested.emit()
        
    def on_stop_clicked(self):
        """处理停止按钮点击事件"""
        self.stopRequested.emit()
        
    def on_reset_clicked(self):
        """处理重置按钮点击事件"""
        self.resetRequested.emit()
        
    def get_parameters(self):
        """获取当前设置的参数"""
        return {
            "algorithm": self.algorithm_combo.currentText(),
            "method": self.method_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "debug": self.debug_checkbox.isChecked(),
            "realtime_optimization": self.realtime_optimization_checkbox.isChecked(),
            "wfs_res": self.wfs_res_combo.currentText(),
            "pupil_diameter": self.pupil_diameter_spin.value(),
            "early_stop_threshold": self.early_stop_threshold_spin.value(),
            "cam_id": self.cam_id_spin.value(),
            "exposure_time_ms": self.exposure_time_spin.value(),
            "cam_size": self.cam_size_spin.value(),
            "target_max_brightness": self.target_max_brightness_spin.value(),
            "delta": self.delta_spin.value(),
            "lr": self.lr_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "shrink_iter": self.shrink_iter_spin.value(),
            "shrink_ratio": self.shrink_ratio_spin.value(),
            "dm_unit_mask": self.dm_unit_mask_combo.currentText(),
            "r_bucket": self.r_bucket_spin.value(),
            "center": self.center_combo.currentText(),
            "n_calls": self.n_calls_spin.value(),
            "lr_min": self.lr_min_spin.value(),
            "lr_max": self.lr_max_spin.value(),
            "delta_min": self.delta_min_spin.value(),
            "delta_max": self.delta_max_spin.value(),
            "method_type": self.method_type_combo.currentText(),
            # 添加其他可能需要的参数
            "dir": "data",
            "load_file": "rms",
            "wf_epochs": 8000,
            "rms_threshold": 0.12,
            "grid_lr_steps": 5,
            "grid_delta_steps": 5
        }
