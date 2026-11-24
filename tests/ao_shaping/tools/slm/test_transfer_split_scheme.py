"""#254 T2 契约: 三层分组防泄漏 + 按轮预注册划分 + final_test 封存层.

泄漏口径 (validation_quality 排除 sealed) 与 `probes(split_scheme="rounds")` 的
预注册映射都必须在这里钉死: 上游实现任何一处回退成 `!= train` / 逐索引交错,
本文件即红。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.adaptation import (
    SEALED_SPLITS, SELECTION_SPLIT, final_test_quality, validation_quality,
)


def _entry(values):
    return {"metrics": [dict(zip(("correlation", "relative_l2", "cv_abs_error"), v))
                        for v in values]}


# ---------------------------------------------------------------------------
# validation_quality / final_test_quality 口径
# ---------------------------------------------------------------------------


def test_validation_quality_reads_only_the_selection_tier():
    rows = [dict(split="train"), dict(split="validation"), dict(split="validation"),
            dict(split="test"), dict(split="final_test"), dict(split="calibration")]
    quality = validation_quality(_entry([(0.0, 1.0, 0.5), (0.9, 0.1, 0.01),
                                         (0.5, 0.5, 0.2), (0.0, 1.0, 0.4),
                                         (0.0, 1.0, 0.4), (0.0, 1.0, 0.4)]), rows)
    assert quality["correlation"] == pytest.approx(0.7)   # 只有两条 validation
    assert quality["relative_l2"] == pytest.approx(0.3)


def test_validation_quality_raises_without_a_selection_tier():
    rows = [dict(split="train"), dict(split="test"), dict(split="final_test")]
    with pytest.raises(ValueError, match="validation phases required"):
        validation_quality(_entry([(0.9, 0.1, 0.0)] * 3), rows)


def test_final_test_quality_scores_only_the_sealed_tier():
    rows = [dict(split="train"), dict(split="validation"), dict(split="final_test")]
    quality = final_test_quality(_entry([(0.9, 0.1, 0.01), (0.5, 0.5, 0.2),
                                         (0.2, 0.8, 0.3)]), rows)
    assert quality["correlation"] == pytest.approx(0.2)
    assert quality["score"] == pytest.approx(1 - 0.2 + 0.8)
    with pytest.raises(ValueError, match="final_test phases required"):
        final_test_quality(_entry([(0.9, 0.1, 0.01)]), [dict(split="validation")])


def test_sealed_vocabulary():
    assert SELECTION_SPLIT == "validation"
    assert SEALED_SPLITS == frozenset({"test", "final_test"})


# ---------------------------------------------------------------------------
# load_dataset: sealed 行永不折叠进训练集, final_test 只在显式打分时装载
# ---------------------------------------------------------------------------


def _dataset(tmp_path):
    from ao_shaping.tools.slm.transfer.capture import to_panel

    size = 8
    rows, samples = [], {}
    base = np.full((40, 40), 5, np.float32)

    def add(label, split, spot, phase_value):
        name = f"sample_{len(rows):04d}.npz"
        raw = np.stack([base, base])
        phase = np.full((size, size), phase_value, np.float32)
        np.savez_compressed(tmp_path/name, raw_frames=raw, frame=raw.mean(0),
                            model_phase=phase, panel_phase=to_panel(phase),
                            panel_gray=np.zeros((1200, 1920), np.uint16))
        samples[name] = dict(model_phase=phase, raw_frames=raw)
        rows.append(dict(file=name, label=label, split=split, exposure_ms=1.,
                         peak=float(base.max()), spot_xy=list(spot),
                         exposure_setting=dict(status="ok")))

    add("reference_start", "calibration", (20., 20.), 0.)
    for axis, shift in (("x", (6., 0.)), ("y", (0., 6.))):
        add(f"ramp_{axis}_64_1", "calibration", (20.+shift[0], 20.+shift[1]), 0.)
        add(f"ramp_{axis}_64_-1", "calibration", (20.-shift[0], 20.-shift[1]), 0.)
    add("probe_000_a", "train", (20., 20.), .1)
    add("probe_003_b", "validation", (20., 20.), .2)
    add("probe_009_c", "test", (20., 20.), .3)
    add("probe_021_d", "final_test", (20., 20.), .4)
    (tmp_path/"manifest.json").write_text(json.dumps(rows))
    (tmp_path/"config.json").write_text(json.dumps(dict(grid=size)))
    return samples


def test_load_dataset_excludes_final_test_unless_scoring(tmp_path):
    from ao_shaping.tools.slm.transfer import fit

    samples = _dataset(tmp_path)
    selected, *_ , train = fit.load_dataset(tmp_path, 8, 10, samples=samples)
    assert [row["label"] for row in selected] == ["probe_000_a", "probe_003_b", "probe_009_c"]
    assert train.tolist() == [True, False, False]

    scored, *_ , scored_train = fit.load_dataset(
        tmp_path, 8, 10, samples=samples, score_final_test=True)
    assert scored[-1]["label"] == "probe_021_d"
    assert scored_train.tolist() == [True, False, False, False]


def test_train_all_fold_never_touches_sealed(tmp_path):
    from ao_shaping.tools.slm.transfer import fit

    samples = _dataset(tmp_path)
    selected, *_ , train = fit.load_dataset(
        tmp_path, 8, 10, samples=samples, train_all=True, score_final_test=True)
    by_split = {row["split"]: flag for row, flag in zip(selected, train)}
    assert by_split == {"train": True, "validation": True, "test": False,
                        "final_test": False}


# ---------------------------------------------------------------------------
# probes(): legacy 不变量 + rounds 预注册
# ---------------------------------------------------------------------------


def test_legacy_scheme_is_byte_identical_to_history():
    from ao_shaping.tools.slm.transfer import capture

    got = list(capture.probes(128, 12, 102319))
    # Leading/trailing calibration (reference_start/end + 8 ramps) then the
    # historical index%6 interleave: 3 train + 3 validation, twice.
    calib = [row for row in got if row[1] == "calibration"]
    probe_splits = [split for label, split, _ in got if label.startswith("probe_")]
    assert len(calib) == 10  # reference_start + 2 axes x 2 periods x 2 signs + reference_end
    assert probe_splits == ["train", "train", "train", "validation", "validation",
                            "validation"] * 2


def test_rounds_scheme_structure():
    from ao_shaping.tools.slm.transfer import capture

    n_rounds, seed = 6, 102319
    mapping = capture.assign_round_splits(n_rounds, seed)
    tiers = [mapping[r] for r in range(n_rounds)]
    assert tiers.count("final_test") == 1
    assert tiers.count("validation") == 2      # ceil(5/3)
    assert tiers.count("train") == 3
    assert capture.assign_round_splits(n_rounds, seed) == mapping   # 预注册确定性
    assert set(mapping.values()) == {"train", "validation", "final_test"}

    probes = list(capture.probes(64, 36, seed, split_scheme="rounds"))
    probe_rows = [(label, split) for label, split, _ in probes
                  if label.startswith("probe_")]
    for round_index in range(n_rounds):
        tier = {split for label, split in probe_rows
                if capture.probe_round_of_label(label) == round_index}
        assert len(tier) == 1, f"round {round_index} straddles tiers: {tier}"
        assert tier == {mapping[round_index]}

    with pytest.raises(ValueError, match="4 rounds"):
        capture.assign_round_splits(3, seed)
    with pytest.raises(ValueError, match="final_test_rounds"):
        capture.assign_round_splits(6, seed, final_test_rounds=4)


def test_probe_round_of_label():
    from ao_shaping.tools.slm.transfer.capture import probe_round_of_label

    assert probe_round_of_label("reference_start") is None
    assert probe_round_of_label("ramp_x_64_1") is None
    assert probe_round_of_label("probe_005_zernike") == 0
    assert probe_round_of_label("probe_006_smooth") == 1
    assert probe_round_of_label("probe_071_bandlimited") == 11


def test_subset_fit_rejects_every_sealed_tier(tmp_path):
    from ao_shaping.tools.slm.transfer.uncertainty import fit_recorded_subset

    row = dict(split="final_test")
    with pytest.raises(ValueError, match="sealed test/final_test"):
        fit_recorded_subset(tmp_path/"out", None, [], [row], [], {},
                            iterations=1, span=4, device="cpu")
