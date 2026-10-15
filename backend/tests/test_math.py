"""Tests del motor de psicohistoria: matematicas, normalizacion y endpoints."""
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.services.predictor import (
    _compute_probability,
    _build_tree,
    _repair_and_load,
    _brace_extract_objects,
    _group_events,
    _build_event_from_raw,
    _assemble_tree,
    _normalize_scale,
    _normalize_trend,
    _normalize_bool,
    _normalize_year,
    _compute_coherence,
    _phi,
)
from psychohistory.models import Event


def test_phi_is_valid_cdf():
    assert 0.0 <= _phi(-10) <= 1.0
    assert 0.0 <= _phi(0) <= 1.0
    assert abs(_phi(0) - 0.5) < 1e-9
    assert _phi(10) > 0.999
    assert _phi(-10) < 0.001
    assert _phi(-1) < _phi(0) < _phi(1)


def test_compute_probability_bounds():
    for mag in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for trend in ["rising", "stable", "declining", "volatile"]:
            for radiant in [True, False]:
                for depth in [0, 1, 3, 5]:
                    s, mx, mn = _compute_probability(mag, trend, radiant, depth)
                    for p in (s, mx, mn):
                        assert 0.0 <= p <= 1.0, f"fuera de rango: {p}"


def test_compute_probability_ordering():
    """max debe ser >= min siempre (la rama optimista no puede ser menor que la pesimista)."""
    for mag in [0.1, 0.5, 0.9]:
        for trend in ["rising", "stable", "declining", "volatile"]:
            for radiant in [True, False]:
                s, mx, mn = _compute_probability(mag, trend, radiant, 1)
                assert mx >= mn, f"max {mx} < min {mn}"


def test_radiant_increases_same_probability():
    s_no, _, _ = _compute_probability(0.6, "stable", False, 1)
    s_yes, _, _ = _compute_probability(0.6, "stable", True, 1)
    assert s_yes > s_no, "un radiant debe elevar la probabilidad de continuidad"


def test_depth_decay_reduces_probability():
    s0, _, _ = _compute_probability(0.8, "rising", False, 0)
    s5, _, _ = _compute_probability(0.8, "rising", False, 5)
    assert s5 < s0, "a mayor profundidad, menor certeza (decaimiento exponencial)"


def test_build_tree_structure():
    for depth in [1, 2, 3]:
        nodes = _build_tree("test prompt", depth)
        assert len(nodes) == 3, "debe haber 3 ramas raiz (same/max/min)"

        def count(ns):
            return sum(1 + count(n.children) for n in ns)

        expected = sum(3 ** d for d in range(1, depth + 1))
        assert count(nodes) == expected, f"depth {depth}: {count(nodes)} != {expected}"

        def maxdepth(ns):
            return 0 if not ns else 1 + max(maxdepth(n.children) for n in ns)

        assert maxdepth(nodes) == depth

        assert [n.branch_type for n in nodes] == ["same", "max", "min"]


def test_build_tree_links_parents():
    nodes = _build_tree("test", 2)
    for root in nodes:
        assert root.parent_id is None
        for child in root.children:
            assert child.parent_id == root.id
            assert child.depth == 2


def test_coherence_range():
    nodes = _build_tree("test", 3)
    c = _compute_coherence(nodes)
    assert 0.0 <= c <= 1.0


def test_repair_valid_json():
    parsed = _repair_and_load('{"summary": "hola", "events": [{"name": "a"}, {"name": "b"}]}')
    assert parsed["summary"] == "hola"
    assert len(parsed["events"]) == 2


def test_repair_json_list():
    parsed = _repair_and_load('[{"name": "a"}, {"name": "b"}]')
    assert len(parsed["events"]) == 2


def test_repair_truncated_json_salvages_events():
    """Un JSON truncado no debe perder los eventos completos ya generados."""
    truncated = (
        '{"summary": "resumen", "events": ['
        '{"level": 1, "name": "Uno", "description": "d1"}, '
        '{"level": 1, "name": "Dos", "description": "d2"}, '
        '{"level": 1, "name": "Tre'
    )
    parsed = _repair_and_load(truncated)
    names = [e.get("name") for e in parsed["events"]]
    assert "Uno" in names and "Dos" in names
    assert parsed["summary"] == "resumen"


def test_brace_extract_objects_nested():
    text = '{"events": [{"name": "a"}, {"name": "b"}], "summary": "x"}'
    objs = _brace_extract_objects(text)
    names = [o.get("name") for o in objs if "name" in o]
    assert names == ["a", "b"]


def test_group_events_by_level_field():
    raw = (
        [{"level": 1, "name": f"a{i}"} for i in range(3)]
        + [{"level": 2, "name": f"b{i}"} for i in range(9)]
    )
    grouped = _group_events(raw, 2)
    assert len(grouped[1]) == 3
    assert len(grouped[2]) == 9


def test_group_events_positional_fallback():
    raw = [{"name": f"e{i}"} for i in range(12)]
    grouped = _group_events(raw, 2)
    assert len(grouped[1]) == 3
    assert len(grouped[2]) == 9
    assert grouped[1][0]["name"] == "e0"
    assert grouped[2][0]["name"] == "e3"


def test_build_event_from_raw_fields():
    ev = _build_event_from_raw(
        {
            "name": "PP wins",
            "description": "PP wins with 38%",
            "date": "Nov 2029",
            "duration": "3 months",
            "year": 2029,
            "magnitude": 8,
            "trend": "asc",
            "radiant": "true",
        },
        "spain 2029",
        0,
        1,
    )
    assert ev.name == "PP wins"
    assert ev.date == "Nov 2029"
    assert ev.duration == "3 months"
    assert ev.magnitude == 0.8
    assert ev.trend == "rising"
    assert ev.radiant is True


def test_normalize_scale():
    assert _normalize_scale(0.5) == 0.5
    assert _normalize_scale(7) == 0.7
    assert _normalize_scale(70) == 0.7
    assert _normalize_scale(1.5) == 0.15
    assert _normalize_scale(-1) == 0.0
    assert _normalize_scale(999) == 1.0
    assert _normalize_scale(None) == 0.5
    assert _normalize_scale("0.42") == 0.42


def test_normalize_trend():
    assert _normalize_trend("rising") == "rising"
    assert _normalize_trend("subiendo") == "rising"
    assert _normalize_trend("bajando") == "declining"
    assert _normalize_trend("estable") == "stable"
    assert _normalize_trend("volátil") == "volatile"
    assert _normalize_trend("desconocido") == "stable"
    assert _normalize_trend(None) == "stable"


def test_normalize_bool():
    assert _normalize_bool(True) is True
    assert _normalize_bool(False) is False
    assert _normalize_bool("true") is True
    assert _normalize_bool("SI") is True
    assert _normalize_bool("no") is False
    assert _normalize_bool("Caracas") is False
    assert _normalize_bool(1) is True
    assert _normalize_bool(0) is False


def test_normalize_year():
    assert _normalize_year(2030, 0) == 2030
    assert _normalize_year("2035", 0) == 2035
    assert _normalize_year(1800, 2) == 2028
    assert _normalize_year(None, 3) == 2029
    assert _normalize_year("abc", 1) == 2027


def _make_event(name: str, desc: str) -> Event:
    return Event(
        name=name,
        description=desc,
        probability=0.5,
        magnitude=0.5,
        trend="stable",
        radiant=False,
        year=2030,
    )


def test_assemble_tree_counts():
    flat = {
        1: [_make_event(f"r{i}", f"desc r{i}") for i in range(3)],
        2: [_make_event(f"l2_{i}", f"desc l2_{i}") for i in range(9)],
        3: [_make_event(f"l3_{i}", f"desc l3_{i}") for i in range(27)],
    }
    roots = _assemble_tree(flat, 3, "escenario")
    assert len(roots) == 3

    def count(ns):
        return sum(1 + count(n.children) for n in ns)

    assert count(roots) == 39


def test_assemble_tree_uses_llm_descriptions():
    flat = {1: [_make_event("NombreReal", "Descripcion concreta de lo que pasa")]}
    roots = _assemble_tree(flat, 1, "escenario")
    assert roots[0].event.name == "NombreReal"
    assert "Descripcion concreta" in roots[0].event.description


def test_assemble_tree_fills_gaps_with_math():
    # Solo se provee nivel 1; el nivel 2 debe rellenarse por matematica
    flat = {1: [_make_event("r0", "d0"), _make_event("r1", "d1"), _make_event("r2", "d2")]}
    roots = _assemble_tree(flat, 2, "escenario")
    assert len(roots) == 3
    for root in roots:
        assert len(root.children) == 3
        for child in root.children:
            assert child.event.description, "los nodos rellenados deben tener descripcion"


def test_assemble_tree_branch_types():
    flat = {1: [_make_event(f"r{i}", "d") for i in range(3)]}
    roots = _assemble_tree(flat, 2, "escenario")
    assert [r.branch_type for r in roots] == ["same", "max", "min"]
    for root in roots:
        assert [c.branch_type for c in root.children] == ["same", "max", "min"]


def test_assemble_tree_parent_links():
    flat = {1: [_make_event(f"r{i}", "d") for i in range(3)]}
    roots = _assemble_tree(flat, 2, "escenario")
    for root in roots:
        for child in root.children:
            assert child.parent_id == root.id


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
