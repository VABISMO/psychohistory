import httpx
import json
import math
import os
import random
import re
import uuid
import time
from typing import List, Optional, Tuple
from ..models import PredictionResult, PredictionNode, Event, TemporalInfo, Actor
from ..config_db import coll
from .limiter import ollama_bottleneck, with_retries
from . import temporal as temporal_math
from . import coupling as coupling_math
from . import actors as actors_math
from . import uncertainty as uncertainty_math
from . import schema as schema_math

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3"

# ROADMAP section A: temporal / rare-event enrichment. Additive and flag-gated;
# set PSYCHO_TEMPORAL=0 to disable and restore the exact previous behaviour.
TEMPORAL_ENABLED = os.getenv("PSYCHO_TEMPORAL", "1").lower() not in ("0", "false", "no", "off")
EPOCH_YEAR = 2026


PREFERRED_MODELS = ["llama3", "llama3:latest", "llama3.1", "gemma4:latest", "gemma4"]

# When enabled, a child's probability is conditioned on its parent's (section C).
COUPLING_ENABLED = os.getenv("PSYCHO_COUPLING", "1").lower() not in ("0", "false", "no", "off")

# ROADMAP section B: named actors parsed from the model and their leverage used
# to nudge branch probabilities. Set PSYCHO_ACTORS=0 for the previous behaviour.
ACTORS_ENABLED = os.getenv("PSYCHO_ACTORS", "1").lower() not in ("0", "false", "no", "off")
ACTOR_SHIFT_STRENGTH = 0.5

# ROADMAP section D: a Beta credible interval around each event's probability.
UNCERTAINTY_ENABLED = os.getenv("PSYCHO_UNCERTAINTY", "1").lower() not in ("0", "false", "no", "off")
UNCERTAINTY_STRENGTH = 6.0

# ROADMAP section F: constrain decoding with a JSON schema (Ollama structured
# outputs). PSYCHO_SCHEMA=0 falls back to the plain "json" format.
SCHEMA_ENABLED = os.getenv("PSYCHO_SCHEMA", "1").lower() not in ("0", "false", "no", "off")


def _get_available_models() -> List[str]:
    try:
        resp = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def _pick_model(available: List[str]) -> str:
    """Pick the fastest/most suitable available model."""
    for pref in PREFERRED_MODELS:
        if pref in available:
            return pref
    for m in available:
        if "mimo" not in m:
            return m
    return available[0] if available else DEFAULT_MODEL


def _phi(x: float) -> float:
    """Approximate standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _compute_probability(magnitude: float, trend: str, radiant: bool, depth: int) -> tuple:
    C_t = 0.5 + 0.3 * magnitude
    C_c = 0.5
    sigma_C = 0.15

    base_same = _phi((C_t - C_c) / sigma_C)
    base_max = base_same * (0.7 + 0.2 * (1.0 - depth * 0.15))
    base_min = max(0.05, base_same * (0.15 + 0.1 * depth * 0.1))

    if trend == "rising":
        base_same = min(1.0, base_same + 0.15)
        base_max = min(1.0, base_max + 0.2)
    elif trend == "declining":
        base_same = max(0.0, base_same - 0.1)
        base_min = min(1.0, base_min + 0.15)
    elif trend == "volatile":
        base_same *= 0.7
        base_max = min(1.0, base_max + 0.1)
        base_min = min(1.0, base_min + 0.1)

    if radiant:
        base_same = min(1.0, base_same + 0.25)
        base_max = min(1.0, base_max + 0.15)
        base_min = max(0.0, base_min - 0.1)

    depth_decay = math.exp(-0.3 * depth)
    base_same *= depth_decay
    base_max *= depth_decay
    base_min *= depth_decay

    return (
        round(max(0.0, min(1.0, base_same)), 4),
        round(max(0.0, min(1.0, base_max)), 4),
        round(max(0.0, min(1.0, base_min)), 4),
    )


def _finite_or_none(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _attach_temporal(event: Event) -> Event:
    """Attach temporal / rare-event statistics to an event (ROADMAP section A).

    Purely additive: it never changes the existing probability, magnitude or
    branch structure. Disabled when PSYCHO_TEMPORAL=0. Failures are swallowed so
    the predictor can never break because of the temporal layer.
    """
    if not TEMPORAL_ENABLED or event.temporal is not None:
        return event
    try:
        horizon = max(1.0, float(event.year) - EPOCH_YEAR)
        assessment = temporal_math.assess(
            base_prob=event.probability,
            trend=event.trend,
            magnitude=event.magnitude,
            horizon=horizon,
            age=0.0,
        )
        event.temporal = TemporalInfo(
            **{k: _finite_or_none(v) for k, v in assessment.items()}
        )
    except Exception as e:
        print(f"[PREDICT] temporal assess skipped ({type(e).__name__}: {e})")
    return event


def _attach_uncertainty(event: Event) -> Event:
    """Attach a 90% credible interval to the event probability (ROADMAP section D).

    A Beta prior is built from the point probability (as base rate) with a modest
    pseudo-sample strength, and the equal-tailed 90% interval is read off. Purely
    additive; disabled with PSYCHO_UNCERTAINTY=0. Failures are swallowed.
    """
    if not UNCERTAINTY_ENABLED or event.probability_ci is not None:
        return event
    try:
        a, b = uncertainty_math.beta_prior_from_base_rate(
            event.probability, UNCERTAINTY_STRENGTH
        )
        lo, hi = uncertainty_math.credible_interval(a, b, 0.9)
        event.probability_ci = [round(lo, 4), round(hi, 4)]
    except Exception as e:
        print(f"[PREDICT] uncertainty assess skipped ({type(e).__name__}: {e})")
    return event


BRANCH_EN = {
    "same": "continuation of the current trend",
    "max": "high-probability (optimistic) scenario",
    "min": "low-probability (pessimistic) scenario",
}

TREND_EN = {
    "rising": "rising",
    "stable": "stable",
    "declining": "declining",
    "volatile": "volatile",
}


def _generate_event(prompt: str, depth: int, parent_event: Optional[str] = None, branch: str = "same") -> Event:
    trends = ["rising", "stable", "declining", "volatile"]
    trend = random.choice(trends)
    magnitude = round(random.uniform(0.2, 0.95), 3)
    radiant = magnitude > 0.75 and trend in ("rising", "stable")
    year = 2026 + depth + random.randint(0, depth * 2)
    branch_en = BRANCH_EN.get(branch, branch)

    if parent_event:
        name = f"{parent_event} — {branch.upper()} (D{depth})"
        description = (
            f"Follow-on from \"{parent_event}\": {branch_en}. "
            f"Trend {TREND_EN.get(trend, trend)}, estimated impact {magnitude:.2f} around {year}."
        )
    else:
        name = f"Root prediction: {prompt[:60]}"
        description = (
            f"Root scenario for \"{prompt[:80]}\": {branch_en}. "
            f"Trend {TREND_EN.get(trend, trend)}, magnitude {magnitude:.2f}, horizon {year}."
        )

    ev = Event(
        name=name,
        description=description,
        probability=round(magnitude * 0.8, 4),
        magnitude=magnitude,
        trend=trend,
        radiant=radiant,
        year=year,
        date=f"{year}",
        duration="months",
    )
    _attach_temporal(ev)
    _attach_uncertainty(ev)
    return ev


def _build_tree(prompt: str, depth: int, parent_id: str = "", parent_event: str = "", current_depth: int = 0) -> List[PredictionNode]:
    if current_depth >= depth:
        return []

    nodes = []
    for branch_type in ["same", "max", "min"]:
        node_id = str(uuid.uuid4())[:8]
        event = _generate_event(prompt, current_depth + 1, parent_event or None, branch_type)
        s_p, _, _ = _compute_probability(event.magnitude, event.trend, event.radiant, current_depth)

        node = PredictionNode(
            id=node_id,
            depth=current_depth + 1,
            parent_id=parent_id or None,
            branch_type=branch_type,
            event=event,
            stability_index=s_p,
            children=[],
        )
        node.children = _build_tree(prompt, depth, node_id, event.name, current_depth + 1)
        nodes.append(node)

    return nodes


def _compute_coherence(nodes: List[PredictionNode]) -> float:
    if not nodes:
        return 0.5
    all_stab = []

    def collect(n: PredictionNode):
        all_stab.append(n.stability_index)
        for c in n.children:
            collect(c)

    for n in nodes:
        collect(n)
    return round(sum(all_stab) / len(all_stab), 4)


def _apply_coupling(nodes: List[PredictionNode]) -> None:
    """Condition each node's probability on its parent's (ROADMAP section C).

    Walks the tree parents-first so a child sees its parent's already-adjusted
    value. Disabled with PSYCHO_COUPLING=0, which restores the independent model.
    """
    if not COUPLING_ENABLED or not nodes:
        return

    def walk(parent: PredictionNode) -> None:
        for child in parent.children:
            child.stability_index = round(
                coupling_math.adjust_for_parent(
                    child.stability_index, parent.stability_index, child.branch_type
                ),
                4,
            )
            walk(child)

    for root in nodes:
        walk(root)


def _apply_actor_influence(nodes: List[PredictionNode]) -> None:
    """Let named actors nudge their event's probability (ROADMAP section B).

    Only events that actually carry named actors are affected, so the math
    fallback (which has none) is untouched. The strongest actor on the event sets
    the weight; the optimistic/continuation branches are pushed up and the
    pessimistic branch down. Disabled with PSYCHO_ACTORS=0.
    """
    if not ACTORS_ENABLED or not nodes:
        return

    def walk(n: PredictionNode) -> None:
        acts = n.event.actors
        if acts:
            weight = max(
                actors_math.influence_weight(a.influence, a.role) for a in acts
            )
            direction = -1.0 if n.branch_type == "min" else 1.0
            n.stability_index = round(
                actors_math.shift_probability(
                    n.stability_index, weight, direction, ACTOR_SHIFT_STRENGTH
                ),
                4,
            )
        for c in n.children:
            walk(c)

    for root in nodes:
        walk(root)


def _fallback_summary(prompt: str, roots: List[PredictionNode]) -> str:
    names = [r.event.name for r in roots[:3]]
    joined = "; ".join(names) if names else "no dominant scenario"
    return (
        f"Most plausible trajectory for \"{prompt}\": the model identifies {joined}. "
        "The probability of a continuation of the current trend is moderate to high, "
        "with the optimistic branch concentrating around the strongest institutional "
        "actors and the pessimistic branch around volatility and fragmentation."
    )


def generate_prediction_fallback(prompt: str, depth: int, context: List[str]) -> PredictionResult:
    """Generate a prediction locally with the mathematical model (no Ollama)."""
    print("[PREDICT] Generating prediction with local mathematical model")
    graph_nodes = _build_tree(prompt, depth)
    _apply_coupling(graph_nodes)
    _apply_actor_influence(graph_nodes)
    coherence = _compute_coherence(graph_nodes)

    return PredictionResult(
        prediction_id=str(uuid.uuid4()),
        prompt_used=prompt,
        prediction_depth=depth,
        coherence_index=coherence,
        summary=_fallback_summary(prompt, graph_nodes),
        graph_nodes=graph_nodes,
        context_used=context,
        model_used="mathematical-fallback",
        timestamp=time.time(),
    )


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize_scale(value, default: float = 0.5) -> float:
    """Normalize magnitudes that may come in 0-1, 0-10 or 0-100."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v > 1.0:
        if v <= 10.0:
            v = v / 10.0
        elif v <= 100.0:
            v = v / 100.0
        else:
            v = 1.0
    return _clamp01(v)


TREND_MAP = {
    "rising": "rising",
    "subiendo": "rising",
    "alza": "rising",
    "asc": "rising",
    "ascendente": "rising",
    "creciente": "rising",
    "up": "rising",
    "stable": "stable",
    "estable": "stable",
    "estavel": "stable",
    "flat": "stable",
    "declining": "declining",
    "bajando": "declining",
    "baja": "declining",
    "desc": "declining",
    "descendente": "declining",
    "decreciente": "declining",
    "down": "declining",
    "volatile": "volatile",
    "volatil": "volatile",
    "inestable": "volatile",
}

TRUTHY = {"true", "si", "sí", "yes", "1", "verdadero", "alto", "high"}


def _normalize_trend(value) -> str:
    if not isinstance(value, str):
        return "stable"
    import unicodedata

    key = "".join(
        c for c in unicodedata.normalize("NFD", value.strip().lower())
        if unicodedata.category(c) != "Mn"
    ).strip()
    return TREND_MAP.get(key, "stable")


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY
    return False


def _normalize_year(value, depth: int) -> int:
    try:
        y = int(float(value))
        if 2020 <= y <= 2200:
            return y
    except (TypeError, ValueError):
        pass
    return 2026 + depth


def _brace_extract_objects(text: str) -> list:
    """Extract every balanced {...} object containing a 'name' key.

    Robust against truncated JSON: complete inner objects are recovered even if
    the enclosing document never closes.
    """
    objs: list = []
    stack: list = []
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}":
            if stack:
                start = stack.pop()
                sub = text[start:i + 1]
                try:
                    obj = json.loads(sub)
                except Exception:
                    continue
                if isinstance(obj, dict) and any(k in obj for k in ("name", "nombre", "event", "evento")):
                    objs.append(obj)
    return objs


def _repair_and_load(text: str) -> dict:
    """Parse the LLM output, salvaging as much as possible from malformed JSON."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {"summary": "", "events": [e for e in parsed if isinstance(e, dict)]}
        if isinstance(parsed, dict):
            events = parsed.get("events") or parsed.get("eventos") or parsed.get("event_list")
            if not isinstance(events, list):
                # some models nest lists by level
                events = []
                for v in parsed.values():
                    if isinstance(v, list):
                        events.extend([e for e in v if isinstance(e, dict)])
            summary = parsed.get("summary") or parsed.get("resumen") or parsed.get("global") or ""
            return {"summary": str(summary), "events": events}
    except Exception:
        pass

    summary = ""
    m = re.search(r'"(?:summary|resumen)"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.S)
    if m:
        try:
            summary = json.loads('"' + m.group(1) + '"')
        except Exception:
            summary = m.group(1)
    return {"summary": summary, "events": _brace_extract_objects(text)}


def _parse_actors(raw: dict) -> List[Actor]:
    """Parse a list of named actors from raw LLM JSON (ROADMAP section B)."""
    if not ACTORS_ENABLED:
        return []
    items = raw.get("actors") or raw.get("actores") or []
    if isinstance(items, dict):
        items = [items]
    out: List[Actor] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if isinstance(it, str):
            it = {"name": it}
        if not isinstance(it, dict):
            continue
        name = it.get("name") or it.get("nombre")
        if not name:
            continue
        age = it.get("age") or it.get("edad")
        tenure = it.get("tenure_years") or it.get("tenure") or it.get("anios")
        stress = it.get("stress") or it.get("presion")
        try:
            actor = Actor(
                name=str(name)[:120],
                role=str(it.get("role") or it.get("rol") or "other")[:40],
                influence=_normalize_scale(it.get("influence") or it.get("influencia"), 0.5),
                age=int(age) if age not in (None, "") else None,
                tenure_years=float(tenure) if tenure not in (None, "") else None,
                stress=_normalize_scale(stress, 0.0),
            )
        except (TypeError, ValueError):
            continue
        out.append(actor)
    return out[:6]


def _build_event_from_raw(raw: dict, prompt: str, index: int, depth: int) -> Event:
    name = (
        raw.get("name")
        or raw.get("nombre")
        or raw.get("event")
        or raw.get("evento")
        or f"Prediction {index + 1}: {prompt[:50]}"
    )
    description = raw.get("description") or raw.get("descripcion") or raw.get("detalle") or ""
    date = raw.get("date") or raw.get("fecha") or raw.get("when") or ""
    duration = raw.get("duration") or raw.get("duracion") or raw.get("length") or ""
    year = _normalize_year(
        raw.get("year") or raw.get("anio") or raw.get("año"), depth
    )
    if not date:
        date = str(year)
    return Event(
        name=str(name)[:200],
        description=str(description)[:600],
        probability=_normalize_scale(raw.get("probability") or raw.get("probabilidad"), 0.5),
        magnitude=_normalize_scale(raw.get("magnitude") or raw.get("magnitud"), 0.5),
        trend=_normalize_trend(raw.get("trend") or raw.get("tendencia")),
        radiant=_normalize_bool(raw.get("radiant") or raw.get("radiante")),
        year=year,
        date=str(date)[:40],
        duration=str(duration)[:40],
        actors=_parse_actors(raw),
    )


def _group_events(raw_events: list, semantic_depth: int) -> dict:
    """Group raw event dicts by level, using the 'level' field or positional order."""
    flat: dict = {lvl: [] for lvl in range(1, semantic_depth + 1)}
    if not raw_events:
        return flat

    def has_level(o: dict) -> bool:
        v = o.get("level") if "level" in o else o.get("nivel")
        try:
            return 1 <= int(v) <= semantic_depth
        except (TypeError, ValueError):
            return False

    if all(has_level(o) for o in raw_events):
        for o in raw_events:
            lvl = int(o.get("level") if "level" in o else o.get("nivel"))
            flat[lvl].append(o)
    else:
        idx = 0
        for lvl in range(1, semantic_depth + 1):
            n = 3 ** lvl
            flat[lvl] = raw_events[idx:idx + n]
            idx += n
    return flat


SEMANTIC_MAX_DEPTH = 2


async def _llm_flat_tree(
    prompt: str,
    depth: int,
    context: List[str],
    model: str,
    constraints: Optional[List[str]] = None,
) -> Tuple[dict, str]:
    """Build the semantic tree with a single LLM call.

    Returns (flat_levels, summary) where flat_levels maps level -> [Event, ...].
    """
    semantic_depth = min(depth, SEMANTIC_MAX_DEPTH)
    context_lines = [c.strip()[:150] for c in context[:5] if c and c.strip()]
    context_str = "\n".join(context_lines) or "No external context available."

    limit_lines = [c.strip()[:220] for c in (constraints or [])[:6] if c and c.strip()]
    if limit_lines:
        limits_block = "\n".join(f"- {c}" for c in limit_lines)
    else:
        limits_block = (
            "- No verified bound was found online. Therefore do NOT state any precise "
            "absolute number (seats, votes, share). Use relative or clearly-approximate "
            "language instead of inventing a figure."
        )

    n1 = 3 ** 1
    n2 = 3 ** 2
    actors_field = (
        ', "actors": [{"name": "...", "role": "leader", "influence": 0.7, '
        '"age": 62, "tenure_years": 4}]'
        if ACTORS_ENABLED else ""
    )
    schema_hint = (
        '{\n'
        '  "summary": "one paragraph (3-5 sentences) with the most plausible global '
        'prediction: who wins, who loses, the key societal events and their dates",\n'
        '  "events": [\n'
        '    {"level": 1, "name": "...", "description": "...", "magnitude": 0.0, '
        '"trend": "rising", "radiant": true, "year": 2029, "date": "Nov 2029", '
        '"duration": "6 months"' + actors_field + '}\n'
        '  ]\n'
        '}'
    )

    actors_rule = (
        "- For each event include 1-3 REAL named \"actors\" that drive it, each "
        "with a role (leader|party|general|central_banker|ceo|union|institution) "
        "and an influence 0.0-1.0. Use real people's approximate age and tenure.\n"
        if ACTORS_ENABLED else ""
    )

    system_prompt = (
        "You are an Asimov-style Psychohistory engine. You project plausible futures "
        "from real facts, real named actors and collective trends. "
        "Respond with VALID JSON ONLY, no markdown, no extra text.\n\n"
        "Output schema (exactly these keys):\n"
        f"{schema_hint}\n\n"
        "The \"events\" array is FLAT and in order. It must contain EXACTLY "
        f"{n1} events with \"level\": 1, followed by {n2} events with \"level\": 2.\n"
        "Structure grouping:\n"
        "- The 3 level-1 events are independent base scenarios: "
        "[continuation, HIGHEST-probability, LOWEST-probability].\n"
        "- The 9 level-2 events come grouped in threes: the first 3 derive from "
        "level-1 event 1, the next 3 from level-1 event 2, the last 3 from level-1 "
        "event 3. Within each group of 3 the order is "
        "[continuation, HIGHEST, LOWEST].\n\n"
        "CONTENT RULES (critical):\n"
        "- Write everything in ENGLISH.\n"
        "- Name REAL, verifiable actors: political parties, leaders, governments, "
        "countries, institutions, companies, unions — adapted to the scenario.\n"
        + actors_rule +
        "- Each description must say what concretely happens: who wins, who loses, "
        "what decision is taken, and the consequence. NO generic filler.\n"
        "- \"date\": approximate month/year it occurs (e.g. \"Nov 2029\").\n"
        "- \"duration\": how long it lasts (e.g. \"3 months\", \"2 years\").\n"
        "- \"magnitude\": 0.0-1.0 impact. \"trend\": rising|stable|declining|volatile.\n"
        "- \"radiant\": true only for highly inevitable events.\n"
        "- Keep each description to 12-18 words. Be concise and information-dense.\n\n"
        "NUMERICAL INTEGRITY RULES (never break these):\n"
        "- You MUST base every number on the REAL-WORLD LIMITS listed in the user "
        "message, which were fetched from live web search. Treat them as hard caps.\n"
        "- NEVER produce a figure that exceeds an established total or maximum. "
        "Example of what is FORBIDDEN: 'a party wins 400 seats' in a chamber that "
        "only has 350 seats in total.\n"
        "- Any per-actor quantity (seats, votes, share, members, budget) must be "
        "consistent with the retrieved total and must not exceed 100% of it.\n"
        "- If several actors are given, their values should be jointly plausible "
        "against the same total.\n"
        "- If no verified bound was retrieved, do NOT invent absolute numbers; "
        "describe the direction and relative standing instead.\n"
        "- Never invent world records or historical maxima that contradict the "
        "retrieved limits."
    )

    user_msg = (
        f"Scenario to project: {prompt}\n"
        f"Depth: {semantic_depth} levels.\n\n"
        f"Real retrieved context (current news/data):\n{context_str}\n\n"
        f"REAL-WORLD LIMITS, TOTALS AND RECORDS fetched from the web — OBEY AS HARD CAPS:\n"
        f"{limits_block}\n\n"
        f"Return exactly {n1 + n2} events: {n1} with level 1, {n2} with level 2, "
        "plus the \"summary\" paragraph. Base the events on the context and keep every "
        "number within the limits above. Name who wins and the societal consequences."
    )

    total_nodes = sum(3 ** l for l in range(1, semantic_depth + 1))
    num_predict = min(1500, max(900, total_nodes * 110))

    # ROADMAP section F: constrain the decoder with a JSON schema so output is
    # valid by construction; the salvage parser stays as a fallback.
    output_format: object = "json"
    if SCHEMA_ENABLED:
        try:
            output_format = schema_math.build_tree_schema(semantic_depth, include_actors=ACTORS_ENABLED)
        except Exception as e:
            print(f"[PREDICT] schema build failed, using plain json ({type(e).__name__}: {e})")
            output_format = "json"

    async def _call_ollama() -> dict:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": model,
                    "prompt": user_msg,
                    "system": system_prompt,
                    "format": output_format,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": 0.7,
                        "num_predict": num_predict,
                        "num_ctx": 4096,
                        "num_thread": 12,
                    },
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "{}")
            return _repair_and_load(raw)

    async with ollama_bottleneck:
        parsed = await with_retries(
            _call_ollama,
            attempts=2,
            base_delay=3.0,
            on_retry=lambda n, e: print(
                f"[PREDICT] Ollama retry {n}/2 ({type(e).__name__})"
            ),
        )

    if SCHEMA_ENABLED:
        problems = schema_math.validate_flat_tree(parsed, semantic_depth)
        if problems:
            print(f"[PREDICT] schema warnings ({len(problems)}): {problems[:3]}")

    grouped = _group_events(parsed.get("events", []), semantic_depth)
    flat: dict = {}
    for lvl in range(1, semantic_depth + 1):
        flat[lvl] = [
            _build_event_from_raw(e, prompt, i, lvl)
            for i, e in enumerate(grouped.get(lvl, []))
        ]

    if not flat.get(1):
        raise ValueError("LLM returned no level-1 events")

    print(
        "[PREDICT] Semantic tree: "
        + ", ".join(f"n{l}={len(flat.get(l, []))}/{3 ** l}" for l in range(1, semantic_depth + 1))
        + f", summary={len(parsed.get('summary', ''))} chars"
    )
    return flat, parsed.get("summary", "")


def _event_to_node(event: Event, level: int, parent_id: Optional[str], branch_type: str) -> PredictionNode:
    _attach_temporal(event)
    _attach_uncertainty(event)
    s_p, _, _ = _compute_probability(event.magnitude, event.trend, event.radiant, level - 1)
    return PredictionNode(
        id=str(uuid.uuid4())[:8],
        depth=level,
        parent_id=parent_id,
        branch_type=branch_type,
        event=event,
        stability_index=s_p,
        children=[],
    )


def _assemble_tree(flat: dict, depth: int, prompt: str) -> List[PredictionNode]:
    """Assemble the tree from per-level events, filling gaps with math."""
    branch_types = ["same", "max", "min"]
    prev: List[PredictionNode] = []
    roots: List[PredictionNode] = []

    for level in range(1, depth + 1):
        expected = 3 ** level
        events = list(flat.get(level, []))
        cur: List[PredictionNode] = []
        for i in range(expected):
            branch = branch_types[i % 3]
            parent = prev[i // 3] if prev else None
            if i < len(events):
                event = events[i]
            else:
                event = _generate_event(
                    prompt, level, parent.event.name if parent else None, branch
                )
            node = _event_to_node(event, level, parent.id if parent else None, branch)
            if parent:
                parent.children.append(node)
            cur.append(node)
        if level == 1:
            roots = cur
        prev = cur

    return roots


async def generate_prediction_ollama(
    prompt: str,
    depth: int,
    context: List[str],
    model: str = DEFAULT_MODEL,
    constraints: Optional[List[str]] = None,
) -> PredictionResult:
    """Try to generate a prediction via Ollama. Fall back to the math model on failure."""
    try:
        models = _get_available_models()
        if not models:
            print("[PREDICT] Ollama not responding, using local mathematical model")
            return generate_prediction_fallback(prompt, depth, context)

        model = model if model in models else _pick_model(models)
        print(f"[PREDICT] Using Ollama model: {model} (available: {models})")

        flat, summary = await _llm_flat_tree(prompt, depth, context, model, constraints)
        graph_nodes = _assemble_tree(flat, depth, prompt)
        _apply_coupling(graph_nodes)
        _apply_actor_influence(graph_nodes)

        if not summary:
            summary = _fallback_summary(prompt, graph_nodes)

        coherence = _compute_coherence(graph_nodes)
        print(f"[PREDICT] Ollama prediction OK: {len(graph_nodes)} roots, coherence={coherence}")
        return PredictionResult(
            prediction_id=str(uuid.uuid4()),
            prompt_used=prompt,
            prediction_depth=depth,
            coherence_index=coherence,
            summary=summary,
            graph_nodes=graph_nodes,
            context_used=context,
            model_used=model,
            timestamp=time.time(),
        )

    except Exception as e:
        print(f"[PREDICT] Ollama error ({type(e).__name__}: {e}), using mathematical fallback")
        return generate_prediction_fallback(prompt, depth, context)


async def save_prediction(result: PredictionResult):
    try:
        coll.insert_one(result.model_dump())
        print(f"[DB] Prediction saved: {result.prediction_id}")
    except Exception as e:
        print(f"[DB] Save error: {e}")
