from typing import List, Optional
from pydantic import BaseModel, Field


class TemporalInfo(BaseModel):
    """Temporal / rare-event statistics attached to an event (ROADMAP section A).

    All fields are optional and default to ``None`` so that older stored
    predictions and clients remain valid, and so non-finite values (e.g. an
    infinite Pareto mean) can be omitted safely from JSON.
    """
    occurrence_prob: Optional[float] = Field(default=None, description="P(occurs within horizon)")
    shape_k: Optional[float] = Field(default=None, description="Weibull hazard shape (>1 = overdue)")
    scale_lambda: Optional[float] = Field(default=None, description="Weibull scale (years)")
    survival_horizon: Optional[float] = Field(default=None, description="S(horizon)")
    hazard_now: Optional[float] = Field(default=None, description="Instantaneous hazard at time 0")
    median_onset: Optional[float] = Field(default=None, description="Median time to onset (years)")
    mean_onset: Optional[float] = Field(default=None, description="Mean time to onset (years)")
    conditional_next: Optional[float] = Field(default=None, description="P(occurs in next window | not yet)")
    overdue_factor: Optional[float] = Field(default=None, description="Conditional vs baseline (>1 = overdue)")
    tail_alpha: Optional[float] = Field(default=None, description="Pareto tail index (small = heavy tail)")
    severity_mean: Optional[float] = Field(default=None, description="Mean severity (impact units)")
    tail_es: Optional[float] = Field(default=None, description="Expected shortfall of severity")
    risk_score: Optional[float] = Field(default=None, description="P(occurs) * severity * magnitude")


class Actor(BaseModel):
    """A first-class actor — leader, party, general, banker, CEO (ROADMAP section B)."""
    name: str = Field(description="Real named actor")
    role: str = Field(default="other", description="leader | party | general | central_banker | ceo | ...")
    influence: float = Field(default=0.5, ge=0.0, le=1.0, description="Leverage 0-1")
    age: Optional[int] = Field(default=None, description="Age in years, if a person")
    tenure_years: Optional[float] = Field(default=None, description="Years in current role")
    stress: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Political stress 0-1")


class Citation(BaseModel):
    """A retrieved evidence source (ROADMAP section E)."""
    title: str = Field(description="Source title")
    url: str = Field(default="", description="Source URL, if available")
    snippet: str = Field(default="", description="Short quoted evidence")
    source: str = Field(default="", description="Publisher / origin (e.g. NEWSDATA, WIKIPEDIA)")
    retrieved_at: str = Field(default="", description="ISO date the source was retrieved")



class Event(BaseModel):
    name: str = Field(description="Short concrete event headline")
    description: str = Field(description="Concrete description naming real actors and outcome")
    probability: float = Field(ge=0.0, le=1.0, description="Base probability 0-1")
    magnitude: float = Field(ge=0.0, le=1.0, description="Impact magnitude 0-1")
    trend: str = Field(description="Trend: rising | stable | declining | volatile")
    radiant: bool = Field(default=False, description="Whether it is a radiant (inevitable) event")
    year: int = Field(description="Estimated year of the event")
    date: str = Field(default="", description="Approximate date it occurs, e.g. 'Nov 2029'")
    duration: str = Field(default="", description="How long it lasts, e.g. '6 months'")
    temporal: Optional[TemporalInfo] = Field(default=None, description="Temporal / rare-event statistics")
    probability_ci: Optional[List[float]] = Field(default=None, description="90% credible interval for probability [low, high]")
    actors: List[Actor] = Field(default_factory=list, description="Named actors driving this event")
    citations: List[Citation] = Field(default_factory=list, description="Retrieved sources supporting this event")


class PredictionBranch(BaseModel):
    """A prediction branch with 3 probabilities"""
    same_probability: float = Field(ge=0.0, le=1.0)
    max_probability: float = Field(ge=0.0, le=1.0)
    min_probability: float = Field(ge=0.0, le=1.0)
    event: Event


class PredictionNode(BaseModel):
    """Node of the prediction tree"""
    id: str
    depth: int
    parent_id: Optional[str] = None
    branch_type: str = Field(description="same | max | min")
    event: Event
    stability_index: float = Field(ge=0.0, le=1.0, description="Lambda index 0-1")
    children: List["PredictionNode"] = []


class PredictionResult(BaseModel):
    prediction_id: str
    prompt_used: str
    prediction_depth: int
    coherence_index: float = Field(ge=0.0, le=1.0)
    summary: str = Field(default="", description="Global narrative of the most plausible future")
    graph_nodes: List[PredictionNode]
    context_used: List[str]
    sources: List[Citation] = Field(default_factory=list, description="All retrieved evidence sources")
    model_used: str
    timestamp: float


PredictionNode.model_rebuild()
