from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .config_db import coll
from .services.search import retrieve_context, retrieve_constraints, retrieve_sources
from .services.predictor import generate_prediction_ollama, save_prediction
from .services.limiter import general_limiter, predict_limiter

app = FastAPI(title="Psychohistory · Future Graph Dynamics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Throttles requests per client (IP) to protect the CPU-bound LLM."""
    client = request.client.host if request.client else "unknown"
    is_predict = request.method == "POST" and request.url.path.startswith("/predict")
    limiter = predict_limiter if is_predict else general_limiter

    if not limiter.check(client):
        retry_after = limiter.retry_after(client)
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Too many requests. Please wait a few seconds before retrying.",
                "retry_after": round(retry_after, 1),
            },
            headers={
                "Retry-After": str(int(retry_after) + 1),
                "Access-Control-Allow-Origin": "*",
            },
        )
    return await call_next(request)


class UserSourceInput(BaseModel):
    """A source supplied by the user for a single prediction (ROADMAP section E)."""
    title: str = ""
    url: str = ""
    text: str = ""


class PredictionRequest(BaseModel):
    prompt: str
    prediction_depth: int = 3
    model: str = "llama3"
    extra_sources: List[UserSourceInput] = Field(default_factory=list)


def _normalize_user_sources(items: List[UserSourceInput]) -> List[dict]:
    """Turn user-supplied sources into the citation records used downstream.

    Each becomes a first-class citation tagged ``source="USER"`` so it flows into
    grounding, per-event attribution and the global sources list exactly like a
    retrieved source.
    """
    out: List[dict] = []
    for s in items:
        title = (s.title or "").strip()
        url = (s.url or "").strip()
        text = (s.text or "").strip()
        if not (title or url or text):
            continue
        if not title:
            title = (url or text)[:80]
        out.append({
            "title": title[:200],
            "url": url[:500],
            "snippet": text[:300],
            "source": "USER",
        })
    return out[:20]


@app.get("/")
def read_root():
    return {"service": "Psychohistory · Future Graph Dynamics API", "status": "online"}


@app.get("/models")
async def list_models():
    from .services.predictor import _get_available_models, _pick_model

    models = _get_available_models()
    if models:
        best = _pick_model(models)
        ordered = [best] + [m for m in models if m != best]
    else:
        ordered = []
    return {"models": ordered, "default": ordered[0] if ordered else None, "ollama_available": len(models) > 0}


@app.post("/predict")
async def predict(request: PredictionRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="The prompt cannot be empty")

    depth = max(1, min(request.prediction_depth, 5))

    try:
        import asyncio as _asyncio

        user_sources = _normalize_user_sources(request.extra_sources)
        context, constraints, sources = await _asyncio.gather(
            retrieve_context(request.prompt),
            retrieve_constraints(request.prompt),
            retrieve_sources(request.prompt),
        )
        # User-provided sources are merged first so they rank as first-class
        # evidence alongside the freshly retrieved ones.
        all_sources = user_sources + sources
        result = await generate_prediction_ollama(
            prompt=request.prompt,
            depth=depth,
            context=context,
            model=request.model,
            constraints=constraints,
            sources=all_sources,
            user_sources=user_sources,
        )
        await save_prediction(result)
        return {"status": "success", "data": result.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prediction: {str(e)}")


@app.get("/predictions")
async def list_predictions(limit: int = 20):
    docs = list(coll.find({}, {"_id": 0}).limit(limit))
    return {"count": len(docs), "predictions": docs}


@app.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    doc = coll.find_one({"prediction_id": prediction_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"status": "success", "data": doc}
