from typing import List, Optional
from pydantic import BaseModel, Field


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
    model_used: str
    timestamp: float


PredictionNode.model_rebuild()
