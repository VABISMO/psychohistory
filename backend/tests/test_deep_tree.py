"""Deep-tree node quality (ROADMAP: readable nodes at every depth).

Covers the two defects seen at depth 4:
  * templated levels repeating parent names ("X — SAME (D3) — MAX (D4)");
  * nodes carrying an actor's name with no description of what happens.

Run:  backend/venv/bin/python backend/tests/test_deep_tree.py
"""
import inspect
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.models import Actor, Event
from psychohistory.services import predictor as P

PROMPT = "Spain calls a snap general election after nationwide housing protests"


def _walk(nodes, out=None):
    out = [] if out is None else out
    for n in nodes:
        out.append(n)
        _walk(n.children, out)
    return out


def _event(name, description="", actors=None):
    return Event(
        name=name,
        description=description,
        probability=0.5,
        magnitude=0.5,
        trend="stable",
        year=2029,
        actors=actors or [],
    )


def test_build_tree_depth4_is_unique_and_described():
    random.seed(1)
    nodes = _walk(P._build_tree(PROMPT, 4))
    assert len(nodes) == sum(3 ** d for d in range(1, 5)) == 120

    names = [n.event.name for n in nodes]
    assert len(set(names)) == len(names), "deep nodes must not repeat names"
    for n in nodes:
        assert n.event.description.strip(), f"empty description: {n.event.name!r}"
        assert len(n.event.name) <= 120


def test_build_tree_has_no_chain_suffixes():
    random.seed(2)
    for n in _walk(P._build_tree(PROMPT, 4)):
        assert " — SAME (D" not in n.event.name
        assert " — MAX (D" not in n.event.name
        assert "(D3)" not in n.event.name
        assert not n.event.description.startswith("Follow-on from")


def test_build_tree_synthetic_nodes_carry_actors_and_prose():
    random.seed(3)
    nodes = _walk(P._build_tree(PROMPT, 3))
    deep = [n for n in nodes if n.depth >= 3]
    assert deep
    for n in deep:
        e = n.event
        assert e.actors, f"no actor on {e.name!r}"
        assert e.actors[0].name in e.description, (
            f"description does not say what happens for {e.name!r}: {e.description!r}"
        )
        assert 0.0 <= e.probability <= 1.0
        assert e.trend in ("rising", "stable", "declining", "volatile")


def test_assemble_tree_repairs_bare_name_with_no_description():
    """The reported bug: a node showing a person's name and nothing else."""
    flat = {
        1: [
            _event(
                "Snap General Election",
                "Sanchez calls a snap election for 29 Nov 2026.",
                [Actor(name="Pedro Sanchez", role="leader")],
            ),
            _event("Pedro Sanchez", ""),
            _event("Santiago Abascal", ""),
        ],
    }
    roots = P._assemble_tree(flat, 4, PROMPT)
    nodes = _walk(roots)
    assert len(nodes) == 120
    for n in nodes:
        e = n.event
        assert e.description.strip(), f"still empty: {e.name!r}"
        assert e.name != "Pedro Sanchez"
        assert e.name != "Santiago Abascal"
        assert e.actors, f"repaired node lost its actor: {e.name!r}"
    # the good LLM event is untouched
    assert roots[0].event.name == "Snap General Election"
    assert "snap election for 29 Nov" in roots[0].event.description


def test_assemble_tree_keeps_llm_headline_and_description():
    flat = {1: [_event("Housing Reform Failure", "The reform bill is rejected by parliament.")]}
    roots = P._assemble_tree(flat, 1, PROMPT)
    assert roots[0].event.name == "Housing Reform Failure"
    assert roots[0].event.description == "The reform bill is rejected by parliament."


def test_assemble_tree_rebuilds_duplicate_headlines():
    flat = {
        1: [
            _event("Snap General Election", "First scenario description."),
            _event("Snap General Election", "Second scenario description."),
            _event("Third Scenario", "Third scenario description."),
        ],
    }
    roots = P._assemble_tree(flat, 2, PROMPT)
    names = [n.event.name for n in _walk(roots)]
    assert len(names) == len(set(names)), "duplicate LLM headlines must be rebuilt"


def test_assemble_tree_structure_at_depth4():
    flat = {1: [_event(f"Scenario {i}", f"description {i}") for i in range(3)]}
    roots = P._assemble_tree(flat, 4, PROMPT)
    assert len(roots) == 3
    assert [r.branch_type for r in roots] == ["same", "max", "min"]

    def depth4(nodes):
        if nodes[0].children and nodes[0].children[0].children:
            return depth4(nodes[0].children)
        return nodes

    nodes = _walk(roots)
    assert [n.depth for n in nodes].count(4) == 81
    for n in nodes:
        if n.parent_id is not None:
            assert n.event.description.strip()


def test_child_topic_of_llm_headline_is_the_headline():
    assert P._child_topic("", "Housing Reform Failure") == "Housing Reform Failure"
    assert P._child_topic("inherited", "") == "inherited"


def test_clean_topic_strips_chain_suffixes():
    assert P._clean_topic("Housing Reform Failure — SAME (D3) — MAX (D4)") == (
        "Housing Reform Failure"
    )
    assert len(P._clean_topic("Word " * 60).split()) <= 6


def test_extract_names_finds_recurring_person_names():
    text = "Pedro Sanchez called a snap election. Pedro Sanchez faces the vote."
    names = P._extract_names([text], min_hits=2)
    assert "Pedro Sanchez" in names
    assert "Housing" not in names


def test_actor_pool_prefers_llm_actors():
    ev = _event("x", "y", [Actor(name="Santiago Abascal", role="leader")])
    pool = P._actor_pool([ev], documents=None, prompt="Spain election")
    assert pool[0] == "Santiago Abascal"


def test_prompt_rules_forbids_empty_descriptions():
    source = inspect.getsource(P._llm_flat_tree)
    assert "is a failure" in source
    assert "NEVER put a" in source
    assert "Do not repeat a headline" in source


if __name__ == "__main__":
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
