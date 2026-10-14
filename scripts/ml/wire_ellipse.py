"""Wire the ellipse gap into LossConfig / composite_loss / CLI.

Follows the existing per-sample accumulation pattern exactly: each enabled term appends
``weight * value`` to ``per_sample``, so the composite stays inspectable rather than one
opaque scalar.
"""

from pathlib import Path

LOSSES = Path("src/ml/zernike/losses.py")
TRAIN = Path("src/ml/zernike/train_amp.py")


def sub(text: str, old: str, new: str, path: str) -> str:
    assert text.count(old) == 1, f"{path}: {text.count(old)} matches for {old[:90]!r}"
    return text.replace(old, new, 1)


t = LOSSES.read_text(encoding="utf-8")

t = sub(
    t,
    "    w_spot_moment: float = 0.0",
    "    w_spot_moment: float = 0.0\n"
    "    #: Weight on the anchored ellipse-fit gap -- the five second-moment-ellipse\n"
    "    #: parameters (centroid x/y, var_x, var_y, covariance). Strictly richer than\n"
    "    #: ``w_spot_moment``: a 2 px shift scores 0.0898 here and exactly 0.0 on the radial\n"
    "    #: term, which is blind to both position and orientation.\n"
    "    w_ellipse: float = 0.0",
    "losses.py",
)

t = sub(
    t,
    "    if cfg.w_spot_moment:\n"
    "        moment = spot_moment_gap_term(pred, target, mask)\n"
    '        out["spot_moment"] = moment\n'
    "        per_sample.append(cfg.w_spot_moment * moment)\n",
    "    if cfg.w_ellipse:\n"
    "        ellipse = ellipse_gap_term(pred, target, mask)\n"
    '        out["ellipse"] = ellipse["ellipse"]\n'
    '        per_sample.append(cfg.w_ellipse * ellipse["ellipse"])\n'
    "\n"
    "    if cfg.w_spot_moment:\n"
    "        moment = spot_moment_gap_term(pred, target, mask)\n"
    '        out["spot_moment"] = moment\n'
    "        per_sample.append(cfg.w_spot_moment * moment)\n",
    "losses.py",
)

t = sub(
    t,
    '    "spot_moment_gap_term",',
    '    "spot_moment_gap_term",\n    "ellipse_parameters",\n    "ellipse_gap_term",',
    "losses.py",
)
LOSSES.write_text(t, encoding="utf-8")
print("losses.py: w_ellipse field, composite accumulation, __all__")

t = TRAIN.read_text(encoding="utf-8")
t = sub(
    t,
    '    parser.add_argument(\n        "--target-size-frac"',
    '    parser.add_argument(\n'
    '        "--w-ellipse",\n'
    '        type=float,\n'
    '        default=0.0,\n'
    '        help=(\n'
    '            "Weight on the anchored ellipse-fit gap (centroid x/y, var_x, var_y, "\n'
    '            "covariance). Richer than --w-spot-moment: sees spot position and tilt, "\n'
    '            "which the radial term scores as zero."\n'
    '        ),\n'
    '    )\n'
    '    parser.add_argument(\n        "--target-size-frac"',
    "train_amp.py",
)
t = sub(
    t,
    "            w_mse=1.0, w_shape_gap=1.0, w_spot_moment=args.w_spot_moment\n",
    "            w_mse=1.0,\n"
    "            w_shape_gap=1.0,\n"
    "            w_spot_moment=args.w_spot_moment,\n"
    "            w_ellipse=args.w_ellipse,\n",
    "train_amp.py",
)
t = sub(
    t,
    '    if cfg.loss == "physical" or cfg.loss_weights.w_spot_moment > 0.0:',
    "    if (\n"
    '        cfg.loss == "physical"\n'
    "        or cfg.loss_weights.w_spot_moment > 0.0\n"
    "        or cfg.loss_weights.w_ellipse > 0.0\n"
    "    ):",
    "train_amp.py",
)
TRAIN.write_text(t, encoding="utf-8")
print("train_amp.py: --w-ellipse flag, config, mask gate")