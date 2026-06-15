"""
API HTTP compatible OpenAI (/v1/chat/completions, /v1/models)
au-dessus de l'Engine.
"""

from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import Engine
from .models import list_models, best_model


app = FastAPI(title="llm-runtime", description="API compatible OpenAI pour llama.cpp")

_engine: Engine | None = None
_model_path: Path | None = None

# llama_cpp.Llama n'est pas thread-safe pour des appels concurrents :
# on sérialise les générations sur l'unique instance.
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    """Charge le modèle à la demande (lazy load), une seule fois."""
    global _engine, _model_path

    if _engine is not None:
        return _engine

    model_path = os.environ.get("LLM_RUNTIME_MODEL")
    if model_path:
        _model_path = Path(model_path)
    else:
        info = best_model(available_memory_gb=12.0, local_dirs=["models"])
        if info is None:
            raise RuntimeError("Aucun modèle disponible (ni Ollama, ni ./models)")
        _model_path = info.path

    _engine = Engine(_model_path)
    return _engine


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


@app.get("/v1/models")
def get_models():
    models = list_models(local_dirs=["models"])
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "owned_by": m.source,
            }
            for m in models
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    try:
        engine = get_engine()
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    if request.model and request.model not in (engine.model_path.stem, engine.model_path.name):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Modèle '{request.model}' non disponible. "
                f"Ce serveur sert uniquement '{engine.model_path.name}'."
            ),
        )

    messages = [m.model_dump() for m in request.messages]

    if request.stream:
        def event_stream():
            with _engine_lock:
                for chunk in engine.chat(
                    messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=True,
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    with _engine_lock:
        return engine.chat(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=False,
        )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _engine is not None}
