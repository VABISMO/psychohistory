"""JSON-schema constrained decoding for the prediction tree (ROADMAP section F).

Ollama can be given a JSON *schema* in the ``format`` field (``format`` accepts a
schema object, not just the string ``"json"``). Constraining the grammar means
the model can only emit structurally valid output, so the regex "salvage" parser
becomes a fallback rather than the main line of defence.

This module builds that schema from the requested depth and validates a parsed
response against it, so the caller can log when the constraint was violated.
"""
from typing import Dict, List, Optional

TRENDS = ["rising", "stable", "declining", "volatile"]


def build_tree_schema(semantic_depth: int, include_actors: bool = True) -> Dict:
    """JSON schema for the flat tree: a summary + a list of events.

    ``semantic_depth`` bounds the ``level`` field; the number of events is left
    loose (minItems) because Ollama handles large exact counts poorly and the
    predictor already pads/truncates per level.
    """
    n_min = sum(3 ** lvl for lvl in range(1, max(1, semantic_depth) + 1))

    actor_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "role": {
                "type": "string",
                "enum": [
                    "leader", "head_of_state", "prime_minister", "party",
                    "general", "central_banker", "ceo", "union",
                    "institution", "other",
                ],
            },
            "influence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
            "tenure_years": {"type": "number", "minimum": 0.0},
            "stress": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["name"],
    }

    event_props = {
        "level": {"type": "integer", "minimum": 1, "maximum": max(1, semantic_depth)},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "magnitude": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "trend": {"type": "string", "enum": TRENDS},
        "radiant": {"type": "boolean"},
        "year": {"type": "integer", "minimum": 2000, "maximum": 2200},
        "date": {"type": "string"},
        "duration": {"type": "string"},
    }
    if include_actors:
        event_props["actors"] = {"type": "array", "items": actor_schema}

    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "events": {
                "type": "array",
                "minItems": n_min,
                "items": {
                    "type": "object",
                    "properties": event_props,
                    "required": ["level", "name", "description"],
                },
            },
        },
        "required": ["summary", "events"],
    }


def validate_flat_tree(parsed: object, semantic_depth: int) -> List[str]:
    """Return a list of human-readable schema violations (empty == valid).

    Deliberately lenient: the goal is to detect structural problems so the caller
    can log/repair, not to reject otherwise-usable output.
    """
    errors: List[str] = []
    if not isinstance(parsed, dict):
        return ["root is not an object"]
    if not isinstance(parsed.get("summary"), str) or not parsed.get("summary", "").strip():
        errors.append("summary missing or empty")
    events = parsed.get("events")
    if not isinstance(events, list):
        return errors + ["events is not a list"]
    if not events:
        errors.append("events list is empty")
    for i, e in enumerate(events):
        if not isinstance(e, dict):
            errors.append(f"event {i} is not an object")
            continue
        lvl = e.get("level")
        if not isinstance(lvl, int) or not (1 <= lvl <= max(1, semantic_depth)):
            errors.append(f"event {i}: bad level {lvl!r}")
        if not str(e.get("name") or "").strip():
            errors.append(f"event {i}: missing name")
        if not str(e.get("description") or "").strip():
            errors.append(f"event {i}: missing description")
        tr = e.get("trend")
        if tr is not None and tr not in TRENDS:
            errors.append(f"event {i}: bad trend {tr!r}")
        actors = e.get("actors")
        if actors is not None and not isinstance(actors, list):
            errors.append(f"event {i}: actors is not a list")
    return errors
