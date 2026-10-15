from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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


class PredictionRequest(BaseModel):
    prompt: str
    prediction_depth: int = 3
    model: str = "llama3"


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

        context, constraints, sources = await _asyncio.gather(
            retrieve_context(request.prompt),
            retrieve_constraints(request.prompt),
            retrieve_sources(request.prompt),
        )
        result = await generate_prediction_ollama(
            prompt=request.prompt,
            depth=depth,
            context=context,
            model=request.model,
            constraints=constraints,
            sources=sources,
        )
        await save_prediction(result)
        return {"status": "success", "data": result.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prediction: {str(e)}")


@app.get("/predictions")
async def list_predictions(limit: int = 20):
    docs = list(coll.find({}, {"_id": 0}).limit(limit))
    return {"count": len(docs), "predictions": docs}
