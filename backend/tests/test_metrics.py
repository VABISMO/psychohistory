"""Verification tests for the evaluation metrics (ROADMAP section I).

Pure standard library, runnable without pytest:

    backend/venv/bin/python backend/tests/test_metrics.py
"""
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation import metrics as M


def test_brier_score_known_values():
    assert M.brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert M.brier_score([0.0, 1.0], [1, 0]) == 1.0
    assert abs(M.brier_score([0.5, 0.5], [1, 0]) - 0.25) < 1e-12
    assert abs(M.brier_score([0.8], [1]) - 0.04) < 1e-12


def test_brier_skill_score():
    # perfect forecast vs base-rate reference
    assert M.brier_skill_score([1.0, 0.0], [1, 0]) == 1.0
    # forecasting exactly the base rate has zero skill
    ys = [1, 1, 0, 0]
    ps = [0.5, 0.5, 0.5, 0.5]
    assert abs(M.brier_skill_score(ps, ys)) < 1e-12
    # worse than base rate -> negative skill
    assert M.brier_skill_score([0.0, 0.0, 1.0, 1.0], ys) < 0.0


def test_log_loss_known_values():
    # p=0.5 on a 1/0 pair -> ln 2
    assert abs(M.log_loss([0.5, 0.5], [1, 0]) - math.log(2)) < 1e-12
    # perfect is ~0 (clipped)
    assert M.log_loss([1.0, 0.0], [1, 0]) < 1e-9
    # confident and wrong is heavily penalised
    assert M.log_loss([0.0, 1.0], [1, 0]) > 30.0


def test_reliability_curve_and_ece():
    # perfectly calibrated within bins: forecast 0.5 for half-positive samples
    ps = [0.5, 0.5] + [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    ys = [1, 0] + [1] * 9 + [0]
    ece = M.expected_calibration_error(ps, ys, n_bins=10)
    assert ece < 0.06, ece
    for mp, of, cnt in M.reliability_curve(ps, ys, n_bins=10):
        assert 0.0 <= mp <= 1.0 and 0.0 <= of <= 1.0 and cnt > 0


def test_ece_zero_when_perfectly_calibrated():
    # bin 0.2 has exactly 20% positives, bin 0.8 has exactly 80% -> ECE = 0
    ps = [0.2] * 10 + [0.8] * 10
    ys = [1] * 2 + [0] * 8 + [1] * 8 + [0] * 2
    assert abs(M.expected_calibration_error(ps, ys, n_bins=10)) < 1e-12
    assert abs(M.maximum_calibration_error(ps, ys, n_bins=10)) < 1e-12


def test_brier_decomposition_identity():
    ps = [0.1, 0.4, 0.6, 0.9, 0.3, 0.7, 0.8, 0.2]
    ys = [0, 1, 1, 1, 0, 1, 0, 0]
    d = M.brier_decomposition(ps, ys, n_bins=10)
    reconstructed = d["reliability"] - d["resolution"] + d["uncertainty"]
    assert abs(reconstructed - d["brier"]) < 1e-9, (reconstructed, d["brier"])


def test_resolution_and_sharpness():
    # constant forecast -> zero resolution and zero sharpness
    ps = [0.5, 0.5, 0.5, 0.5]
    ys = [1, 0, 1, 0]
    assert abs(M.resolution(ps, ys, 5)) < 1e-12
    assert abs(M.sharpness(ps, ys)) < 1e-12
    # informative forecasts -> positive sharpness
    assert M.sharpness([0.1, 0.9], [0, 1]) > 0.0


def test_roc_auc_known_values():
    assert M.roc_auc([1.0, 0.9, 0.1, 0.0], [1, 1, 0, 0]) == 1.0
    assert M.roc_auc([0.0, 0.1, 0.9, 1.0], [1, 1, 0, 0]) == 0.0
    # all forecasts tied -> 0.5
    assert abs(M.roc_auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) - 0.5) < 1e-12
    # single positive/negative
    assert M.roc_auc([0.8, 0.2], [1, 0]) == 1.0


def test_lead_time_errors():
    out = M.lead_time_errors([2027.0, 2024.0], [2026.0, 2026.0])
    assert abs(out["mae"] - 1.5) < 1e-12
    assert abs(out["bias"] - (-0.5)) < 1e-12
    out2 = M.lead_time_errors([2028.0], [2025.0])
    assert out2["bias"] == 3.0


def test_pit_histogram():
    assert M.pit_histogram([0.05, 0.95], n_bins=10) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert sum(M.pit_histogram([0.1 * i for i in range(10)], 10)) == 10


def test_validation_errors():
    for fn in (
        lambda: M.brier_score([], []),
        lambda: M.brier_score([0.5], [1, 0]),
        lambda: M.brier_score([1.5], [1]),
        lambda: M.brier_score([0.5], [2]),
        lambda: M.lead_time_errors([], []),
    ):
        try:
            fn()
            assert False, "debía lanzar ValueError"
        except ValueError:
            pass


def test_summarize_bundle():
    s = M.summarize([0.8, 0.2, 0.6, 0.4], [1, 0, 1, 0], n_bins=4)
    for key in ["n", "base_rate", "brier", "brier_skill", "log_loss", "ece",
                "mce", "resolution", "sharpness", "auc"]:
        assert key in s, key
    assert s["n"] == 4
    assert abs(s["auc"] - 1.0) < 1e-9


if __name__ == "__main__":
    import inspect

    tests = [
        (name, fn)
        for name, fn in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if name.startswith("test_")
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(1 if failed else 0)
