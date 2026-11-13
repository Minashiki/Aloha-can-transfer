"""The validation tracker must fail closed on NaN and Inf."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "m2_validate.py"
        spec = importlib.util.spec_from_file_location("m2_validate_under_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _MODULE = module
    return _MODULE


def test_tracker_rejects_nonfinite_samples() -> None:
    tracker_cls = _module().Tracker
    samples = (
        lambda tracker: tracker.add([np.nan], [1.0]),
        lambda tracker: tracker.add([1.0], [np.nan]),
        lambda tracker: tracker.add([np.inf], [np.inf]),
        lambda tracker: tracker.add_scalar(np.nan),
    )
    for apply in samples:
        tracker = tracker_cls()
        apply(tracker)
        assert tracker.nonfinite
        assert tracker.ok is False

    over = tracker_cls()
    over.add(np.array([1.0]), np.array([0.0]))
    assert over.nonfinite is False
    assert over.ok is False

    good = tracker_cls()
    good.add(np.array([1.0]), np.array([1.0]))
    good.add_scalar(0.0)
    assert good.nonfinite is False
    assert good.ok is True


def test_nonfinite_sample_does_not_erase_a_finite_error() -> None:
    tracker = _module().Tracker()
    tracker.add(np.array([1.0]), np.array([0.0]))
    tracker.add(np.array([np.nan]), np.array([1.0]))
    assert tracker.nonfinite
    assert tracker.ok is False
    assert tracker.max_abs == 1.0
    assert np.isfinite(tracker.max_abs)


def test_condition_number_labels_match_the_denominator_floor() -> None:
    module = _module()
    assert module.COND_DENOM_FLOOR == 1e-16
    assert module.COND_ESTIMATE_NAME == "分母下限 1e-16 的条件数估计"
    assert module.SINGULAR_LABEL == "理论上为无穷大"
