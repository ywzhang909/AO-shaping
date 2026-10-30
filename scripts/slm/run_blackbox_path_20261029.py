"""One-session current-path capture/adaptation with raw evidence and device mutex.

Defaults reproduce the first 2026-10-29 session (``--initial-phase plane``,
8 updates/block).  Every call takes the bench mutex, records the outcome next to
the report, and leaves the run directory named by a UTC-stamped tag so two
sessions can be compared without overwriting each other.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import traceback

from ao_shaping.tools.kernels.device_mutex import DeviceLock
from ao_shaping.tools.slm.transfer.workflow import TransferSession, run_session


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', default='blackbox_path')
    p.add_argument('--initial-phase', default='plane', choices=('plane', 'gs', 'random'))
    p.add_argument('--updates', type=int, default=8)
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--count', type=int, default=12)
    p.add_argument('--iterations', type=int, default=120)
    p.add_argument('--side', type=float, default=200.0)
    p.add_argument('--exposure-steps', type=int, default=8)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--dataset', default=None)
    p.add_argument('--no-calibration', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path('data/shape_trials') / f'{args.tag}_{stamp}'
    status = Path('report/shape_trials/blackbox_path/artifacts') / f'execution_{args.tag}.json'
    status.parent.mkdir(parents=True, exist_ok=True)
    record = {'out': str(out), 'tag': args.tag, 'args': vars(args),
              'started': datetime.now().isoformat(), 'status': 'starting'}
    status.write_text(json.dumps(record, indent=2), encoding='utf-8')
    lock = DeviceLock.default()
    if not lock.acquire(timeout=0):
        record.update(status='blocked', holder=str(lock.holder()))
        status.write_text(json.dumps(record, indent=2), encoding='utf-8')
        raise RuntimeError(f'Device busy: {record}')
    try:
        config = TransferSession(
            out=out, shapes=('square', 'triangle', 'rectangle'),
            count=args.count, iterations=args.iterations,
            blocks=args.blocks, updates=args.updates, frames=3, side=args.side,
            period=64.,
            calibration=None if not args.no_calibration else False,
            adaptation='full', abba=True, save_dataset=True,
            initial_phase=args.initial_phase, exposure_max_steps=args.exposure_steps,
            adaptive_step=True, adaptive_exposure=True, window_span=420,
            window_step=2., control_grid=128, fft_size=2048, bit_depth=12,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            dataset=Path(args.dataset) if args.dataset else None,
        )
        record['status'] = 'running'
        status.write_text(json.dumps(record, indent=2), encoding='utf-8')
        result = run_session(config)
        record.update(status='completed', result=result)
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        lock.release()
        record.update(finished=datetime.now().isoformat(), mutex_released=True)
        status.write_text(json.dumps(record, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
