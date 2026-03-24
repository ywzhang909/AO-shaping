from collections import deque
from dataclasses import dataclass, field
import inspect
import os

import tqdm
import numpy as np

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
ADVISE_EXPOSURE_TIME_BRIGHTNESS = int(255/3)
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
        return tuple(np.round(np.asarray(voltages, dtype=np.float64) / scale).astype(int))

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
            signs = rng.binomial(1, 0.5, size=anchor_v.shape).astype(np.float64) * 2.0 - 1.0
            magnitudes = rng.uniform(radius_scale * 0.35, radius_scale, size=anchor_v.shape)
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


def learning_schedule(power_radius, ideal_r=IDEAL_SPOT_RADIUS) -> tuple[float, float]:
    '''
    Learning schedule for power radio.
    If power radio is less than or equal to ideal_r, return power radio.
    Otherwise, return power radio raised to the power of epoch divided by update_iter.

    Args:
        epoch (int): Current epoch.
        power_radio (float): Power radio.
        ideal_r (float): Ideal spot radius.
    return:
        lr (float): Learning rate.
        delta (float): distribution.
    '''
    if power_radius <= ideal_r:
        return 1.5, 1
    elif power_radius <= 2*ideal_r:
        return 2, 2
    elif power_radius <= 3*ideal_r:
        return 2.5, 3
    elif power_radius <= 4*ideal_r:
        return 3, 4
    elif power_radius <= 5*ideal_r:
        return 4.5, 5
    else:
        return 6, 5

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
    **kwargs
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
        **kwargs: 其他参数。

    """

    delta = abs(delta)
    epochs = int(epochs)
    rng = np.random.default_rng(random_seed)
    search_anchor = search_anchor.lower()
    if search_anchor not in {"best", "current"}:
        raise ValueError("search_anchor must be 'best' or 'current'")

    recorder = Recorder(mark="pib", mode="max")
    
    with CameraStreamManager(cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False) as cam,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT, max_neibor_diff=dm_neibor_diff) as dm:
        if dm_unit_mask is None:
            dm_unit_mask = dm.default_dm_unit_mask
            if dm_unit_mask[0]:
                logger.warning("dm_unit_mask[0] is True, which means the first unit is active.")
        
        if init_v is None or len(init_v) == 0:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 0.5)

        _img = cam.autoset_exposure_time_ms(target_max_brightness=TEST_EXPOSURE_TIME_BRIGHTNESS)
        def intellij_center(img):
            (h,w) = img.shape
            margin = int(IDEAL_SPOT_RADIUS)
            # 如果中心不是空洞，使用质心而非形心;如果中间存在空洞使用形心，否则质心
            center = centroid(np.where(img > np.max(img[:max(int(h//50),2),:max(int(w//50),2)])
                                       , 1, 0))
            (cx, cy) = center
            if np.all(img[cy-margin: cy+margin, cx-margin: cx+margin] >= np.max(img) * 0.4): # 中心不是空洞
                center = centroid(img)
            return center

        if center is None:
            center = intellij_center(_img)
        elif isinstance(center, str):
            _img = cam.get_numpy_image(10)
            if center == "mass":
                center = centroid(_img)
            elif center == 'max':
                center = np.unravel_index(np.argmax(_img), _img.shape)[::-1]
            elif center == 'shape':
                (h,w) = _img.shape
                center = centroid(
                    np.where(_img > np.max(_img[:max(int(h//50),2),:max(int(w//50),2)]), 1, 0))
            else:
                raise ValueError(f"known center: {center}")

        else:
            center = center    
        logger.info(f"Centroid: {center}, Max brightness: {np.max(_img)} @ {cam.exposure_time}ms")

        img_size = (cam_size, cam_size)
        img_size, center = cam.reset_window(center, img_size)

        if exposure_time_ms > 0:
            cam.exposure_time = exposure_time_ms
            init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        elif 0<target_max_brightness<255 and target_max_brightness > 0:
            init_img = cam.autoset_exposure_time_ms(
                target_max_brightness=target_max_brightness, twice_valid=True)
        else:
            init_img = cam.autoset_exposure_time_ms(
                target_max_brightness=ADVISE_EXPOSURE_TIME_BRIGHTNESS, twice_valid=True)
        logger.debug(f"Inital Image Max brightness: {np.max(init_img)} @ {cam.exposure_time}ms")
        img_size = init_img.shape[::-1]
        if r_bucket <= 0:
            _w, _h = img_size
            r_bucket = ImageTargetFunc(_w, _h, center).radius(init_img, energy=0.99)
            r_bucket = min(r_bucket, cam_size//2) * shrink_ratio
            _fix_bucket = False
        else:
            _fix_bucket = True

        if shrink_ratio <= 0 or np.isclose(shrink_ratio, 1.0) or r_bucket <= IDEAL_SPOT_RADIUS:
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
        def calc_pib(img):
            #
            return target_func.pib(img, r_bucket)
            
            # TODO 根据环围半径找出边缘梯度最大的阶数
            # r, nr = target_func.avg_radius(img, moment=0.5)
            # return -r, -nr
            
            # r = target_func.radius(img)
            # return -r, 0
            
        
        def test_pib(img):
            return target_func.pib(img, IDEAL_SPOT_RADIUS)[1]

        j, pib_ratio = calc_pib(init_img)

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
            optimizer.lr, delta = learning_schedule(radius(init_img, center=center, energy=0.8))

        adaptive_search_state = AdaptiveSearchState(
            radius=float(search_radius if search_radius is not None else max(delta * 2.0, 1.0)),
            min_radius=float(search_min_radius if search_min_radius is not None else max(delta * 0.5, 0.5)),
            max_radius=float(search_max_radius if search_max_radius is not None else max(delta * 6.0, 12.0)),
            expand_ratio=float(search_expand_ratio),
            shrink_ratio=float(search_shrink_ratio),
            improvement_tol=float(search_improvement_tol),
        )
        tabu_memory = TabuMemory(
            capacity=int(tabu_memory_size),
            quantization=float(tabu_quantization),
        )

        best_pib = float(test_pib(init_img))
        best_j = float(j)
        best_pib_ratio = float(pib_ratio)
        best_v = _init_v.copy()
        best_img = init_img.copy()
        last_best_epoch = 0

        def evaluate_candidate(voltages: np.ndarray) -> dict[str, np.ndarray | float]:
            dm.send_voltages(voltages)
            candidate_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
            candidate_j, candidate_ratio = calc_pib(candidate_img)
            return {
                "J": float(candidate_j),
                "pib": float(test_pib(candidate_img)),
                "ratio": float(candidate_ratio),
                "img": candidate_img,
                "max_brt": float(np.max(candidate_img)),
            }

        def run_adaptive_search(epoch: int, current_v: np.ndarray, current_pib: float) -> dict | None:
            anchor_v = best_v.copy() if search_anchor == "best" else current_v.copy()
            anchor_pib = best_pib if search_anchor == "best" else float(current_pib)
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
                candidate = np.clip(candidate, dm.min_voltage, dm.max_voltage)
                if tabu_memory.contains(candidate):
                    tabu_hits += 1
                    continue
                if not dm.check_dm_unit_grad_safe(candidate):
                    safe_rejects += 1
                    tabu_memory.add(candidate)
                    continue

                candidate_eval = evaluate_candidate(candidate)
                evaluated += 1
                improved = candidate_eval["pib"] > anchor_pib + adaptive_search_state.improvement_tol
                if improved and (
                    best_candidate is None or candidate_eval["pib"] > best_candidate["pib"]
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
                "pib": test_pib(init_img),
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
                "best_pib": best_pib,
                "search_radius": adaptive_search_state.radius,
                "tabu_size": len(tabu_memory._queue),
            }
        )
        with tqdm.tqdm(
            total=epochs, desc=f"iter {epochs}", dynamic_ncols=True
        ) as bar:
            for epoch in range(1,epochs+1):
                disturb_v = rng.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta * dm_unit_mask

                dm.send_voltages(_init_v + disturb_v)
                pos_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                pos_pib, pos_pib_ratio = calc_pib(pos_img)

                dm.send_voltages(_init_v - disturb_v)
                neg_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                neg_pib, neg_pib_ratio = calc_pib(neg_img)
                
                if show:
                    if not window.render(
                        pos_img, _init_v, dm.V_Min, dm.V_Max, center, r_bucket, f"{epoch}"
                    ):
                        break
                
                max_brightness = max([np.max(pos_img), np.max(neg_img)])
                if max_brightness == 255 and exposure_time_ms == 0:
                    _resample_img = cam.autoset_exposure_time_ms(target_max_brightness, twice_valid=False)
                    optimizer.scale_momentum(np.sum(_resample_img) / np.sum(pos_img))
                    
                # if exposure_time_ms > 0 and max_brightness == 255:
                #     # 固定曝光时，如果过曝，则使用pib_ratio计算梯度
                #     optimizer.scale_momentum(neg_pib_ratio / neg_j)
                #     pos_j, neg_j = pos_pib_ratio, neg_pib_ratio
                # else:
                #     pos_j, neg_j = pos_pib, neg_pib
                pos_j, neg_j = pos_pib, neg_pib
                diff = pos_j - neg_j
                gradient = -diff * disturb_v
                update = optimizer.update(gradient)
                _to_update_v = np.clip(_init_v - update, dm.V_Min, dm.V_Max)
                if dm.check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
                else:
                    logger.warning(f"相邻单元压差大于{dm.max_neibor_diff}，放弃本次结果")

                pib, pib_ratio = test_pib(pos_img), (pos_pib_ratio+neg_pib_ratio)/2
                J = (pos_j + neg_j) / 2

                if epoch % update_iter == update_iter - 1:
                    _init_r = max(_init_r * shrink_ratio, IDEAL_SPOT_RADIUS)

                if (epoch % update_iter == update_iter - 1 or
                     (shrink_iter > 0 and epoch % shrink_iter == shrink_iter - 1) or
                     pib_ratio >= 0.99) and not _fix_bucket and pib>0:
                    power_radio = radius(pos_img, center=center, energy=0.8)
                    _pr = power_radio * shrink_ratio
                    _r = max(r_bucket*shrink_ratio+1, IDEAL_SPOT_RADIUS, r_bucket)
                    r_bucket = min(_r, _pr, _init_r)
                    if lr == 0:
                        optimizer.lr, delta = learning_schedule(r_bucket)

                if pib > best_pib + adaptive_search_state.improvement_tol:
                    best_pib = float(pib)
                    best_j = float(J)
                    best_pib_ratio = float(pib_ratio)
                    best_v = _init_v.copy()
                    best_img = pos_img.copy()
                    last_best_epoch = epoch

                log = {
                    "J": J,
                    "_p%": pib_ratio,
                    "_max_r": _init_r,
                    "pib": pib,
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
                    "optimizer": optimizer_type,
                    "best_pib": best_pib,
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
                    search_result = run_adaptive_search(epoch=epoch, current_v=_init_v, current_pib=pib)
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
                            if search_result["pib"] > best_pib + adaptive_search_state.improvement_tol:
                                best_v = _init_v.copy()
                                best_pib = float(search_result["pib"])
                                best_j = float(search_result["J"])
                                best_pib_ratio = float(search_result["ratio"])
                                best_img = search_result["img"].copy()
                                last_best_epoch = epoch
                            recorder.append(
                                {
                                    "J": search_result["J"],
                                    "_p%": search_result["ratio"],
                                    "_max_r": _init_r,
                                    "pib": search_result["pib"],
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
                                    "best_pib": best_pib,
                                    "search_radius": adaptive_search_state.radius,
                                    "tabu_size": len(tabu_memory._queue),
                                    "search_accept": True,
                                    "search_eval": search_result["evaluated"],
                                    "search_tabu_hits": search_result["tabu_hits"],
                                    "search_safe_rejects": search_result["safe_rejects"],
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

