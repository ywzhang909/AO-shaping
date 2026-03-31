from collections import deque
from dataclasses import dataclass, field
import inspect
import os

import tqdm
import numpy as np
import matplotlib.pylab as plt

from ao_shaping.drivers import CameraStreamManager, NlightDM
from ao_shaping.algorithm.adam import AdaMOD, Adam, AdamW, Base, Muno, MunoW, SGD
from ao_shaping.utils import ImageVoltagesDisplay, logger, Recorder
from ao_shaping.utils.spots_calc import centroid, radius
from ao_shaping.algorithm.target_func import ImageTargetFunc

# adam parameters
beta1 = 0.9
beta2 = 0.99
beta3 = 0.9999

# metropolis parameters
METROPOLIS_ALPHA = 0.8

# camera parameters
CAM_SAMPLE_ITER = 1
ADVISE_EXPOSURE_TIME_BRIGHTNESS = int(255 / 3)
TEST_EXPOSURE_TIME_BRIGHTNESS = 220
IDEAL_SPOT_RADIUS = int(os.environ.get("IDEAL_SPOT_RADIUS", 6))

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = True

OPTIMIZER_MAP = {
    "adam": Adam,
    "adamw": AdamW,
    "adamod": AdaMOD,
    "sgd": SGD,
    "muno": Muno,
    "munow": MunoW,
}


@dataclass
class TabuMemory:
    """Short-term tabu memory for already explored suboptimal candidates."""

    capacity: int
    quantization: float
    _queue: deque[tuple[int, ...]] = field(init=False, default_factory=deque)
    _keys: set[tuple[int, ...]] = field(init=False, default_factory=set)

    def make_key(self, voltages: np.ndarray) -> tuple[int, ...]:
        scale = max(float(self.quantization), 1e-6)
        return tuple(
            np.round(np.asarray(voltages, dtype=np.float64) / scale).astype(int)
        )

    def contains(self, voltages: np.ndarray) -> bool:
        if self.capacity <= 0:
            return False
        return self.make_key(voltages) in self._keys

    def add(self, voltages: np.ndarray) -> None:
        if self.capacity <= 0:
            return
        key = self.make_key(voltages)
        if key in self._keys:
            return
        self._queue.append(key)
        self._keys.add(key)
        while len(self._queue) > self.capacity:
            expired = self._queue.popleft()
            self._keys.discard(expired)


@dataclass
class AdaptiveSearchState:
    """State for adaptive neighborhood search around a local optimum."""

    radius: float
    min_radius: float
    max_radius: float
    expand_ratio: float
    shrink_ratio: float
    improvement_tol: float

    def update_radius(self, improved: bool) -> float:
        if improved:
            next_radius = self.radius * self.shrink_ratio
        else:
            next_radius = self.radius * self.expand_ratio
        self.radius = float(np.clip(next_radius, self.min_radius, self.max_radius))
        return self.radius


def _create_optimizer(optimizer_type: str, dim: int, lr: float, **kwargs) -> Base:
    """Create the configured optimizer while filtering unsupported kwargs."""
    optimizer_cls = OPTIMIZER_MAP.get(optimizer_type.lower(), AdaMOD)
    filtered_kwargs = {}
    signature = inspect.signature(optimizer_cls.__init__)
    for key, value in kwargs.items():
        if key in signature.parameters:
            filtered_kwargs[key] = value
    return optimizer_cls(dim, lr=lr, **filtered_kwargs)


def _extract_optimizer_momentum(optimizer: Base) -> np.ndarray | float | None:
    momentum = getattr(optimizer, "m", None)
    if isinstance(momentum, np.ndarray):
        return momentum.copy()
    return momentum


def _reset_optimizer_state(optimizer: Base) -> None:
    """Reset optimizer momentum after a large search jump."""
    optimizer.t = 0
    for attr in ("m", "v", "v_max"):
        value = getattr(optimizer, attr, None)
        if isinstance(value, np.ndarray):
            value[...] = 0
    if hasattr(optimizer, "s"):
        optimizer.s = 0.0


def _generate_search_candidates(
    anchor_v: np.ndarray,
    radius_scale: float,
    n_samples: int,
    dm_unit_mask: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate mixed dense/sparse perturbations around a local optimum."""
    candidates: list[np.ndarray] = []
    mask = np.asarray(dm_unit_mask, dtype=np.float64)
    radius_scale = max(float(radius_scale), 1e-6)
    for sample_id in range(max(int(n_samples), 1)):
        if sample_id % 2 == 0:
            perturbation = rng.normal(0.0, radius_scale, size=anchor_v.shape)
        else:
            signs = (
                rng.binomial(1, 0.5, size=anchor_v.shape).astype(np.float64) * 2.0 - 1.0
            )
            magnitudes = rng.uniform(
                radius_scale * 0.35, radius_scale, size=anchor_v.shape
            )
            sparse_mask = rng.binomial(1, 0.35, size=anchor_v.shape).astype(np.float64)
            perturbation = signs * magnitudes * sparse_mask
        candidates.append(anchor_v + perturbation * mask)
    return candidates


def _should_trigger_adaptive_search(
    epoch: int,
    enabled: bool,
    warmup: int,
    interval: int,
    patience: int,
    last_best_epoch: int,
) -> bool:
    if not enabled:
        return False
    if epoch < max(int(warmup), 1):
        return False
    if interval <= 0 or epoch % int(interval) != 0:
        return False
    return (epoch - last_best_epoch) >= max(int(patience), 0)


def learning_schedule(
    power_radius: float,
    ideal_r: float = IDEAL_SPOT_RADIUS,
    gradient_history: list[float] | None = None,
    pib_history: list[float] | None = None,
    epoch: int = 0,
) -> tuple[float, float]:
    """
    动态学习率调度器：根据功率半径和目标函数收敛状态动态调整学习率和扰动幅度。

    Args:
        power_radius (float): 当前功率半径。
        ideal_r (float): 理想光斑半径。
        gradient_history (list[float] | None): 最近梯度幅值的历史记录（用于检测收敛状态）。
        pib_history (list[float] | None): 最近 PIB 值的历史记录（用于检测收敛状态）。
        epoch (int): 当前迭代轮次。

    Returns:
        tuple[float, float]: (lr, delta) 学习率和扰动幅度。
    """
    # 基础参数：根据 power_radius 分段返回
    if power_radius <= ideal_r:
        base_lr, base_delta = 1.5, 1
    elif power_radius <= 2 * ideal_r:
        base_lr, base_delta = 2, 2
    elif power_radius <= 3 * ideal_r:
        base_lr, base_delta = 2.5, 3
    elif power_radius <= 4 * ideal_r:
        base_lr, base_delta = 3, 4
    elif power_radius <= 5 * ideal_r:
        base_lr, base_delta = 4.5, 5
    else:
        base_lr, base_delta = 6, 5

    # 如果没有收敛状态历史，直接返回基础参数
    if gradient_history is None or pib_history is None or len(gradient_history) < 5:
        return base_lr, base_delta

    # 收敛状态检测
    recent_grads = (
        list(gradient_history[-10:])
        if len(gradient_history) >= 10
        else gradient_history
    )
    recent_pibs = list(pib_history[-10:]) if len(pib_history) >= 10 else pib_history

    # 计算梯度变化趋势（方差越小越稳定）
    grad_mean = np.mean(recent_grads)
    grad_std = np.std(recent_grads) if len(recent_grads) > 1 else 0
    grad_cv = grad_std / (grad_mean + 1e-8)  # 变异系数

    # 计算 PIB 变化率
    pib_mean = np.mean(recent_pibs)
    pib_std = np.std(recent_pibs) if len(recent_pibs) > 1 else 0
    pib_trend = (
        (recent_pibs[-1] - recent_pibs[0]) / (len(recent_pibs) + 1e-8)
        if len(recent_pibs) > 1
        else 0
    )

    # 动态调整因子
    lr_factor = 1.0
    delta_factor = 1.0

    # 情况1：梯度方差小（收敛稳定）-> 减小 lr 和 delta 以精细调整
    if grad_cv < 0.1:
        lr_factor = 0.5
        delta_factor = 0.5
    # 情况2：梯度方差中等（正常波动）-> 保持
    elif grad_cv < 0.3:
        lr_factor = 0.8
        delta_factor = 0.8
    # 情况3：梯度方差大（震荡）-> 大幅减小 lr
    elif grad_cv > 0.8:
        lr_factor = 0.3
        delta_factor = 1.2  # 增加探索

    # 情况4：PIB 长时间无明显提升（可能陷入局部最优）
    if abs(pib_trend) < 1e-5 and pib_std < 0.01:
        # 增加探索：提高 delta
        delta_factor = max(delta_factor, 1.5)
        lr_factor = min(lr_factor, 0.7)
    # 情况5：PIB 持续下降（发散）-> 立即减小 lr
    elif pib_trend < -0.001:
        lr_factor = 0.4
        delta_factor = 0.6
    # 情况6：PIB 持续上升（正常收敛）-> 保持或微调
    elif pib_trend > 0.001:
        lr_factor = min(lr_factor, 1.1)

    # 早期 epoch 使用较大的学习率（warmup 效果）
    if epoch < 20:
        lr_factor *= 1.2
        delta_factor *= 1.1

    # 限制调整范围
    lr_factor = np.clip(lr_factor, 0.2, 2.0)
    delta_factor = np.clip(delta_factor, 0.3, 2.5)

    final_lr = base_lr * lr_factor
    final_delta = base_delta * delta_factor

    return final_lr, final_delta


def optimize_pib(
    center,
    epochs,
    r_bucket=0,
    delta: float = 1,
    lr: float = 0,
    exposure_time_ms: int = 80,
    shrink_iter: int = 0,
    shrink_ratio: float = 0.9,
    cam_id=0,
    show: bool = False,
    init_v=[],
    cam_size=250,
    target_max_brightness=40,
    dm_unit_mask=None,
    dm_neibor_diff=200,
    dm_max_voltage=None,
    dm_min_voltage=None,
    optimizer_type: str = "adamod",
    enable_adaptive_search: bool = False,
    search_interval: int = 120,
    search_warmup: int = 200,
    search_patience: int = 100,
    search_samples: int = 8,
    search_radius: float | None = None,
    search_min_radius: float | None = None,
    search_max_radius: float | None = None,
    search_expand_ratio: float = 1.4,
    search_shrink_ratio: float = 0.75,
    search_improvement_tol: float = 1e-4,
    tabu_memory_size: int = 128,
    tabu_quantization: float = 2.0,
    search_anchor: str = "best",
    random_seed: int | None = None,
    objective: str = "pib",
    **kwargs,
):
    """优化PIB（Power in Bucket）

    Args:
        center (str or tuple): 中心位置。
        epochs (int): 迭代次数。
        r_bucket (float): 桶半径。如果设置为0，则根据功率半径自动调整。
        delta (float): 分布参数。
        lr (float): 学习率。如果设置为0，则根据功率半径自动调整。
        exposure_time_ms (int): 曝光时间（毫秒）。如果设置为0，则自动曝光。
        shrink_iter (int): 收缩迭代次数。如果设置为0，则不进行收缩。
        shrink_ratio (float): 收缩比例。
        cam_id (int): 相机ID。
        show (bool): 是否显示图像。
        init_v (list): 初始电压。
        cam_size (int): 相机图像大小。
        target_max_brightness (float): 目标最大亮度。如果设置为0，则迭代过程中不自动调整曝光时间。
        dm_unit_mask (list): DM单元掩码。
        dm_neibor_diff (float): DM邻居电压差。
        dm_max_voltage (float): DM最大电压。
        dm_min_voltage (float): DM最小电压。
        optimizer_type (str): 梯度阶段使用的优化器类型，支持 Adam/AdaMOD/SGD/Muno 系列。
        enable_adaptive_search (bool): 是否启用局部最优后的自适应邻域搜索。
        search_interval (int): 邻域搜索触发间隔。
        search_warmup (int): 启动邻域搜索前的最小迭代数。
        search_patience (int): 最佳 PIB 无提升时，等待多少轮后触发搜索。
        search_samples (int): 每次邻域搜索评估的候选解数量。
        search_radius (float | None): 初始邻域搜索半径。
        search_min_radius (float | None): 邻域搜索最小半径。
        search_max_radius (float | None): 邻域搜索最大半径。
        search_expand_ratio (float): 搜索失败时的邻域扩张系数。
        search_shrink_ratio (float): 搜索成功时的邻域收缩系数。
        search_improvement_tol (float): 判定改进的最小 PIB 增益。
        tabu_memory_size (int): 禁忌表容量。
        tabu_quantization (float): 电压量化步长，用于禁忌表去重。
        search_anchor (str): 邻域搜索起点，可选 "best" 或 "current"。
        random_seed (int | None): 随机种子。
        objective (str): 优化目标函数，可选 'pib'(最大化), 'radiu'(最小化半径), 'avg_radiu'(最大化平均半径)。
        **kwargs: 其他参数。

    """

    delta = abs(delta)
    epochs = int(epochs)
    rng = np.random.default_rng(random_seed)
    search_anchor = search_anchor.lower()
    if search_anchor not in {"best", "current"}:
        raise ValueError("search_anchor must be 'best' or 'current'")
    if objective not in ("pib", "radiu", "avg_radiu"):
        raise ValueError(
            f"objective must be one of ('pib', 'radiu', 'avg_radiu'), got {objective}"
        )

    # 优化目标模式映射: pib和avg_radiu最大化, radiu最小化
    objective_mode = "max" if objective in ("pib", "avg_radiu") else "min"
    recorder = Recorder(mark=objective, mode=objective_mode)

    # 历史记录用于收敛状态检测
    _gradient_history: list[float] = []
    _pib_history: list[float] = []
    _max_history_len = 50  # 保持最近50次记录

    with (
        CameraStreamManager(
            cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False
        ) as cam,
        NlightDM(
            keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT, max_neibor_diff=dm_neibor_diff
        ) as dm,
    ):
        if dm_unit_mask is None:
            dm_unit_mask = dm.default_dm_unit_mask
            if dm_unit_mask[0]:
                logger.warning(
                    "dm_unit_mask[0] is True, which means the first unit is active."
                )

        if init_v is None or len(init_v) == 0:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 0.5)

        _img = cam.autoset_exposure_time_ms(
            target_max_brightness=TEST_EXPOSURE_TIME_BRIGHTNESS
        )

        def intellij_center(img):
            (h, w) = img.shape
            margin = int(IDEAL_SPOT_RADIUS)
            # 如果中心不是空洞，使用质心而非形心;如果中间存在空洞使用形心，否则质心
            center = centroid(
                np.where(
                    img > np.max(img[: max(int(h // 50), 2), : max(int(w // 50), 2)]),
                    1,
                    0,
                )
            )
            (cx, cy) = center
            if np.all(
                img[cy - margin : cy + margin, cx - margin : cx + margin]
                >= np.max(img) * 0.4
            ):  # 中心不是空洞
                center = centroid(img)
            return center

        if center is None:
            center = intellij_center(_img)
        elif isinstance(center, str):
            _img = cam.get_numpy_image(10)
            if center == "mass":
                #FIX: wrong
                center = centroid(_img)
            elif center == "max":
                center = np.unravel_index(np.argmax(_img), _img.shape)[::-1]
            elif center == "shape":
                (h, w) = _img.shape
                center = centroid(
                    np.where(
                        _img
                        > np.max(_img[: max(int(h // 50), 2), : max(int(w // 50), 2)]),
                        1,
                        0,
                    )
                )
            else:
                raise ValueError(f"known center: {center}")

        else:
            center = center

        if show:
            plt.imshow(_img, cmap="gray")
            plt.scatter(x=center[0], y=center[1], c="red", s=5)
            plt.show()

        logger.info(
            f"Centroid brightness : {_img[center[::-1]]}@{center}, Max brightness: {np.max(_img)} @ {cam.exposure_time}ms"
        )

        img_size = (cam_size, cam_size)
        img_size, center = cam.reset_window(center, img_size)
        logger.info(f"reset window center @ {center}")
        if exposure_time_ms > 0:
            cam.exposure_time = exposure_time_ms
            init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        elif 0 < target_max_brightness < 255 and target_max_brightness > 0:
            init_img = cam.autoset_exposure_time_ms(
                target_max_brightness=target_max_brightness, twice_valid=True
            )
        else:
            init_img = cam.autoset_exposure_time_ms(
                target_max_brightness=ADVISE_EXPOSURE_TIME_BRIGHTNESS, twice_valid=True
            )
        logger.debug(
            f"Inital Image Max brightness: {np.max(init_img)} @ {cam.exposure_time}ms"
        )
        img_size = init_img.shape[::-1]
        if r_bucket <= 0:
            _w, _h = img_size
            r_bucket = ImageTargetFunc(_w, _h, center).radius(init_img, energy=0.99)
            r_bucket = min(r_bucket, cam_size // 2) * shrink_ratio
            _fix_bucket = False
            logger.info(f"Use dynamic radiu @ {r_bucket}")
        else:
            _fix_bucket = True

        if (
            shrink_ratio <= 0
            or np.isclose(shrink_ratio, 1.0)
            or r_bucket <= IDEAL_SPOT_RADIUS
        ):
            update_iter = max(1, epochs)
        else:
            shrink_span = np.log(IDEAL_SPOT_RADIUS / r_bucket) / np.log(shrink_ratio)
            if np.isfinite(shrink_span) and shrink_span > 0:
                update_iter = max(1, int(epochs * 0.8 // shrink_span))
            else:
                update_iter = max(1, epochs)
        _init_r = r_bucket

        if show:
            window = ImageVoltagesDisplay(img_size)
            window.init_window()

        target_func = ImageTargetFunc.build_from_init_image(init_img)

        # 根据优化目标创建对应的计算函数
        def test_pib(img):
            return target_func.pib(img, IDEAL_SPOT_RADIUS)[1]
        to_min = 1
        if objective == "pib":

            def calc_objective(img):
                pib, pib_ratio = target_func.pib(img, r_bucket)
                to_min = -1
                return pib, pib_ratio
        elif objective == "radiu":

            def calc_objective(img):
                r = target_func.radius(img, energy=0.99)
                return r, 0.0  # 返回(值, ratio)保持与其他目标一致
        elif objective == "avg_radiu":

            def calc_objective(img):
                return target_func.avg_radius(img, moment=1.0)

        j, pib_ratio = calc_objective(init_img)

        optimizer = _create_optimizer(
            optimizer_type=optimizer_type,
            dim=dm.DM_Num,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            **kwargs,
        )
        if lr == 0:
            optimizer.lr, delta = learning_schedule(
                radius(init_img, center=center, energy=0.8),
                gradient_history=_gradient_history,
                pib_history=_pib_history,
                epoch=0,
            )

        adaptive_search_state = AdaptiveSearchState(
            radius=float(
                search_radius if search_radius is not None else max(delta * 2.0, 1.0)
            ),
            min_radius=float(
                search_min_radius
                if search_min_radius is not None
                else max(delta * 0.5, 0.5)
            ),
            max_radius=float(
                search_max_radius
                if search_max_radius is not None
                else max(delta * 6.0, 12.0)
            ),
            expand_ratio=float(search_expand_ratio),
            shrink_ratio=float(search_shrink_ratio),
            improvement_tol=float(search_improvement_tol),
        )
        tabu_memory = TabuMemory(
            capacity=int(tabu_memory_size),
            quantization=float(tabu_quantization),
        )

        best_objective = float(test_pib(init_img))
        best_j = float(j)
        best_objective_ratio = float(pib_ratio)
        best_v = _init_v.copy()
        best_img = init_img.copy()
        last_best_epoch = 0

        def evaluate_candidate(voltages: np.ndarray) -> dict[str, np.ndarray | float]:
            dm.send_voltages(voltages)
            candidate_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
            candidate_j, candidate_ratio = calc_objective(candidate_img)
            return {
                "J": float(candidate_j),
                objective: float(test_objective(candidate_img)),
                "ratio": float(candidate_ratio),
                "img": candidate_img,
                "max_brt": float(np.max(candidate_img)),
            }

        def run_adaptive_search(
            epoch: int, current_v: np.ndarray, current_objective: float
        ) -> dict | None:
            logger.info(f"tabu search start @ {epoch}")
            anchor_v = best_v.copy() if search_anchor == "best" else current_v.copy()
            anchor_objective = (
                best_objective if search_anchor == "best" else float(current_objective)
            )
            candidates = _generate_search_candidates(
                anchor_v=anchor_v,
                radius_scale=adaptive_search_state.radius,
                n_samples=search_samples,
                dm_unit_mask=np.asarray(dm_unit_mask),
                rng=rng,
            )

            best_candidate: dict | None = None
            tabu_hits = 0
            safe_rejects = 0
            evaluated = 0

            for candidate in candidates:
                candidate = np.clip(candidate, dm.V_Min, dm.V_Max)
                if tabu_memory.contains(candidate):
                    tabu_hits += 1
                    continue
                if not dm.check_dm_unit_grad_safe(candidate):
                    safe_rejects += 1
                    tabu_memory.add(candidate)
                    continue

                candidate_eval = evaluate_candidate(candidate)
                evaluated += 1
                improved = (
                    candidate_eval[objective]
                    > anchor_objective + adaptive_search_state.improvement_tol
                )
                if improved and (
                    best_candidate is None
                    or candidate_eval[objective] > best_candidate[objective]
                ):
                    best_candidate = {
                        "voltages": candidate.copy(),
                        **candidate_eval,
                    }
                else:
                    tabu_memory.add(candidate)

            if best_candidate is None:
                adaptive_search_state.update_radius(improved=False)
                return {
                    "accepted": False,
                    "tabu_hits": tabu_hits,
                    "safe_rejects": safe_rejects,
                    "evaluated": evaluated,
                    "radius": adaptive_search_state.radius,
                    "anchor": search_anchor,
                }

            tabu_memory.add(anchor_v)
            adaptive_search_state.update_radius(improved=True)
            best_candidate.update(
                {
                    "accepted": True,
                    "tabu_hits": tabu_hits,
                    "safe_rejects": safe_rejects,
                    "evaluated": evaluated,
                    "radius": adaptive_search_state.radius,
                    "anchor": search_anchor,
                    "_epoch": epoch,
                }
            )
            return best_candidate

        recorder.append(
            {
                "J": j,
                objective: test_pib(init_img),
                "_p%": pib_ratio,
                "_max_r": _init_r,
                "_v": _init_v,
                "_img": init_img,
                "_diff": 0,
                "lr": optimizer.lr,
                "r": r_bucket,
                "delta": delta,
                "_epoch": 0,
                "exp_t": cam.exposure_time,
                "max_brt": np.max(init_img),
                "_grad": np.zeros_like(_init_v),
                "optimizer": optimizer_type,
                f"best_{objective}": best_objective,
                "search_radius": adaptive_search_state.radius,
                "tabu_size": len(tabu_memory._queue),
            }
        )
        with tqdm.tqdm(total=epochs, desc=f"iter {epochs}", dynamic_ncols=True) as bar:
            for epoch in range(1, epochs + 1):
                disturb_v = rng.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta * dm_unit_mask

                dm.send_voltages(_init_v + disturb_v)
                pos_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                pos_obj, pos_obj_ratio = calc_objective(pos_img)

                dm.send_voltages(_init_v - disturb_v)
                neg_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                neg_obj, neg_obj_ratio = calc_objective(neg_img)

                if show:
                    if not window.render(
                        pos_img,
                        _init_v,
                        dm.V_Min,
                        dm.V_Max,
                        center,
                        r_bucket,
                        f"{epoch}",
                    ):
                        break

                max_brightness = max([np.max(pos_img), np.max(neg_img)])
                if max_brightness == 255 and exposure_time_ms == 0:
                    _resample_img = cam.autoset_exposure_time_ms(
                        target_max_brightness, twice_valid=False
                    )
                    optimizer.scale_momentum(np.sum(_resample_img) / np.sum(pos_img))

                # if exposure_time_ms > 0 and max_brightness == 255:
                #     # 固定曝光时，如果过曝，则使用pib_ratio计算梯度
                #     optimizer.scale_momentum(neg_pib_ratio / neg_j)
                #     pos_j, neg_j = pos_pib_ratio, neg_pib_ratio
                # else:
                #     pos_j, neg_j = pos_pib, neg_pib
                pos_j, neg_j = pos_obj, neg_obj
                diff = (pos_j - neg_j) * to_min
                gradient = diff * disturb_v
                update = optimizer.update(gradient)
                _to_update_v = np.clip(_init_v - update, dm.V_Min, dm.V_Max)
                if dm.check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
                else:
                    logger.warning(
                        f"相邻单元压差大于{dm.max_neibor_diff}，放弃本次结果"
                    )

                objective_val, objective_ratio = (
                    test_pib(pos_img),
                    (pos_obj_ratio + neg_obj_ratio) / 2,
                )
                J = (pos_j + neg_j) / 2

                if epoch % update_iter == update_iter - 1:
                    _init_r = max(_init_r * shrink_ratio, IDEAL_SPOT_RADIUS)

                if (
                    (
                        epoch % update_iter == update_iter - 1
                        or (shrink_iter > 0 and epoch % shrink_iter == shrink_iter - 1)
                        or objective_ratio >= 0.99
                    )
                    and not _fix_bucket
                    and objective_val > 0
                ):
                    power_radio = radius(pos_img, center=center, energy=0.8)
                    _pr = power_radio * shrink_ratio
                    _r = max(r_bucket * shrink_ratio + 1, IDEAL_SPOT_RADIUS, r_bucket)
                    r_bucket = min(_r, _pr, _init_r)
                    if lr == 0:
                        # 记录梯度幅值和目标值到历史记录
                        _grad_mag = float(np.linalg.norm(gradient))
                        _gradient_history.append(_grad_mag)
                        _pib_history.append(float(objective_val))
                        # 保持历史记录长度限制
                        if len(_gradient_history) > _max_history_len:
                            _gradient_history.pop(0)
                            _pib_history.pop(0)
                        optimizer.lr, delta = learning_schedule(
                            power_radius=r_bucket,
                            gradient_history=_gradient_history,
                            pib_history=_pib_history,
                            epoch=epoch,
                        )

                if (
                    objective_val
                    > best_objective + adaptive_search_state.improvement_tol
                ):
                    best_objective = float(objective_val)
                    best_j = float(J)
                    best_objective_ratio = float(objective_ratio)
                    best_v = _init_v.copy()
                    best_img = pos_img.copy()
                    last_best_epoch = epoch

                log = {
                    "J": J,
                    "_p%": objective_ratio,
                    "_max_r": _init_r,
                    "pib": objective_val,
                    "_diff": diff,
                    "lr": optimizer.lr,
                    "r": r_bucket,
                    "delta": delta,
                    "_epoch": epoch,
                    "_v": _init_v,
                    "_img": pos_img,
                    "exp_t": cam.exposure_time,
                    "max_brt": max_brightness,
                    "_grad": gradient,
                    "_opt_m": _extract_optimizer_momentum(optimizer),
                    "search_radius": adaptive_search_state.radius,
                    "tabu_size": len(tabu_memory._queue),
                    "search_accept": False,
                    "search_eval": 0,
                    "search_tabu_hits": 0,
                    "search_safe_rejects": 0,
                }
                recorder.append(log)

                if _should_trigger_adaptive_search(
                    epoch=epoch,
                    enabled=enable_adaptive_search,
                    warmup=search_warmup,
                    interval=search_interval,
                    patience=search_patience,
                    last_best_epoch=last_best_epoch,
                ):
                    search_result = run_adaptive_search(
                        epoch=epoch, current_v=_init_v, current_objective=objective_val
                    )
                    if search_result is not None:
                        log["search_eval"] = search_result["evaluated"]
                        log["search_tabu_hits"] = search_result["tabu_hits"]
                        log["search_safe_rejects"] = search_result["safe_rejects"]
                        log["search_radius"] = search_result["radius"]
                        log["tabu_size"] = len(tabu_memory._queue)
                        if search_result["accepted"]:
                            _init_v = search_result["voltages"].copy()
                            _reset_optimizer_state(optimizer)
                            log["search_accept"] = True
                            if (
                                search_result[objective]
                                > best_objective + adaptive_search_state.improvement_tol
                            ):
                                best_v = _init_v.copy()
                                best_objective = float(search_result[objective])
                                best_j = float(search_result["J"])
                                best_objective_ratio = float(search_result["ratio"])
                                best_img = search_result["img"].copy()
                                last_best_epoch = epoch
                            recorder.append(
                                {
                                    "J": search_result["J"],
                                    "_p%": search_result["ratio"],
                                    "_max_r": _init_r,
                                    objective: search_result[objective],
                                    "_diff": 0.0,
                                    "lr": optimizer.lr,
                                    "r": r_bucket,
                                    "delta": delta,
                                    "_epoch": epoch,
                                    "_v": _init_v.copy(),
                                    "_img": search_result["img"],
                                    "exp_t": cam.exposure_time,
                                    "max_brt": search_result["max_brt"],
                                    "_grad": np.zeros_like(_init_v),
                                    "_opt_m": _extract_optimizer_momentum(optimizer),
                                    "optimizer": optimizer_type,
                                    f"best_{objective}": best_objective,
                                    "search_radius": adaptive_search_state.radius,
                                    "tabu_size": len(tabu_memory._queue),
                                    "search_accept": True,
                                    "search_eval": search_result["evaluated"],
                                    "search_tabu_hits": search_result["tabu_hits"],
                                    "search_safe_rejects": search_result[
                                        "safe_rejects"
                                    ],
                                    "search_anchor": search_result["anchor"],
                                }
                            )
                        else:
                            dm.send_voltages(_init_v)

                bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                bar.update(1)
        if show:
            window.close()
        return recorder
