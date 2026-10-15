"""Verification tests for the evaluation package (dataset + scoreboard).

Run:  backend/venv/bin/python backend/tests/test_evaluation.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.dataset import DATASET, by_id
from evaluation import scoreboard as SB
from evaluation import metrics as M


def test_dataset_is_wellformed():
    assert len(DATASET) >= 20
    ids = [r["id"] for r in DATASET]
    assert len(ids) == len(set(ids)), "los id deben ser unicos"
    for r in DATASET:
        assert set(["id", "question", "category", "as_of", "deadline", "resolved_on", "outcome"]).issubset(r)
        assert r["outcome"] in (0, 1)
        assert r["as_of"] <= r["deadline"]
        if r["outcome"] == 1:
            assert r["resolved_on"] is not None
            assert r["as_of"] <= r["resolved_on"] <= r["deadline"]


def test_by_id():
    m = by_id()
    assert m["uk_2024"]["outcome"] == 1
    assert len(m) == len(DATASET)


def _rec(rid, cat, as_of, resolved, outcome):
    return dict(id=rid, question="q", category=cat, as_of=as_of,
                deadline=as_of[:4] + "-12-31", resolved_on=resolved, outcome=outcome)


def test_walkforward_category_base_rate():
    recs = [
        _rec("a", "X", "2010-01-01", "2009-06-01", 1),
        _rec("b", "X", "2011-01-01", "2010-06-01", 0),
        _rec("c", "X", "2012-01-01", "2011-06-01", 1),
    ]
    # for 'c' the two earlier-resolved X questions give (1+0)/2 = 0.5
    assert abs(SB.walkforward_category_base_rate(recs[2], recs) - 0.5) < 1e-12
    # for the earliest there is no prior data -> 0.5
    assert SB.walkforward_category_base_rate(recs[0], recs) == 0.5
    # falls back to overall when the category has no history
    recs2 = recs + [_rec("d", "Y", "2013-01-01", None, 0)]
    # earlier resolved for 'd' = a,b,c -> (1+0+1)/3 = 0.666...
    assert abs(SB.walkforward_category_base_rate(recs2[3], recs2) - (2 / 3)) < 1e-12


def test_build_baseline_shapes():
    base = SB.build_baseline(DATASET)
    for name in SB.STRATEGIES:
        assert name in base
        assert set(base[name].keys()) == set(r["id"] for r in DATASET)
        for p in base[name].values():
            assert 0.0 <= p <= 1.0


def test_score_uses_outcomes():
    recs = [
        _rec("a", "X", "2010-01-01", "2009-06-01", 1),
        _rec("b", "X", "2011-01-01", "2010-06-01", 0),
    ]
    preds = {"a": {"p": 1.0}, "b": {"p": 0.0}}
    summary, ps, ys = SB.score("perfect", preds, recs, n_bins=5)
    assert summary["brier"] == 0.0
    assert summary["auc"] == 1.0
    assert ps == [1.0, 0.0] and ys == [1, 0]


def test_load_predictions_roundtrip():
    data = [{"id": "uk_2024", "p": 0.72}, {"id": "us_2020", "p": 0.9, "predicted_date": "2020-11-01"}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(data, fh)
        path = fh.name
    try:
        loaded = SB.load_predictions(path)
        assert loaded["uk_2024"]["p"] == 0.72
        assert loaded["us_2020"]["predicted_date"] == "2020-11-01"
    finally:
        os.unlink(path)


def test_lead_time_uses_resolved_dates():
    recs = [
        _rec("a", "X", "2020-01-01", "2020-01-01", 1),  # resolved at 2020.0
    ]
    preds = {"a": {"p": 0.5, "predicted_date": "2021-01-01"}}  # ~2021.0
    lt = SB.lead_time(preds, recs)
    assert lt is not None
    assert abs(lt["bias"] - 1.0) < 0.01


def test_metrics_summarize_consistent_with_direct():
    recs = [_rec(str(i), "X", f"{2010+i}-01-01", None, i % 2) for i in range(8)]
    preds = {r["id"]: {"p": 0.5 + 0.4 * (r["outcome"] - 0.5)} for r in recs}
    s, ps, ys = SB.score("m", preds, recs, n_bins=4)
    assert abs(s["brier"] - M.brier_score(ps, ys)) < 1e-9


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
