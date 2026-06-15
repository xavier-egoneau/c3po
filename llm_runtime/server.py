"""
API HTTP compatible OpenAI (/v1/chat/completions, /v1/models)
au-dessus de l'Engine.

Portée : serveur **mono-modèle, mono-utilisateur**. Un seul Engine global est chargé à la
fois ; les générations sont sérialisées par `_engine_lock` (llama.cpp n'est pas thread-safe).
Le swap de modèle à la requête (façon Ollama) convient à un usage séquentiel : sous des
clients concurrents demandant des modèles *différents*, les requêtes ne crashent pas (chaque
génération garde une référence à son Engine), mais elles se sérialisent et font « thrasher »
le chargement (X puis Y puis X…). Pour du multi-modèle réellement concurrent, il faudrait un
pool d'Engines / plusieurs process — hors périmètre actuel.
"""

from __future__ import annotations
import gc
import json
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import Engine
from .hardware import detect_hardware
from .models import find_model, list_models, best_model


app = FastAPI(title="llm-runtime", description="API compatible OpenAI pour llama.cpp")

_engine: Engine | None = None
_model_path: Path | None = None

# llama_cpp.Llama n'est pas thread-safe pour des appels concurrents :
# on sérialise les générations sur l'unique instance.
_engine_lock = threading.Lock()


def _engine_overrides_from_env() -> dict:
    """Leviers d'inférence transmis par `c3po serve` via l'environnement."""
    out: dict = {}
    if "LLM_RUNTIME_CTX" in os.environ:
        out["n_ctx"] = int(os.environ["LLM_RUNTIME_CTX"])
    if "LLM_RUNTIME_N_GPU_LAYERS" in os.environ:
        out["n_gpu_layers"] = int(os.environ["LLM_RUNTIME_N_GPU_LAYERS"])
    if "LLM_RUNTIME_THREADS" in os.environ:
        out["n_threads"] = int(os.environ["LLM_RUNTIME_THREADS"])
    if "LLM_RUNTIME_FLASH_ATTN" in os.environ:
        out["flash_attn"] = os.environ["LLM_RUNTIME_FLASH_ATTN"] == "1"
    if "LLM_RUNTIME_KV_TYPE" in os.environ:
        out["kv_type"] = os.environ["LLM_RUNTIME_KV_TYPE"]
    if os.environ.get("LLM_RUNTIME_SPECULATIVE") == "1":
        out["speculative"] = True
    return out


def get_engine(model_query: str | None = None) -> Engine:
    """
    Retourne le moteur servant `model_query`, en chargeant ou en remplaçant
    le modèle actuellement chargé si besoin (façon Ollama).

    - `model_query` fourni : résolu via `find_model()`. Si différent du modèle
      chargé, l'ancien moteur est libéré et le nouveau est chargé (swap).
    - `model_query is None` : si un moteur est déjà chargé, on le garde tel
      quel ; sinon on choisit via `LLM_RUNTIME_MODEL` ou `best_model()`.
    """
    global _engine, _model_path

    if model_query is not None:
        target_path = find_model(model_query).path
    elif _engine is not None:
        return _engine
    else:
        model_path = os.environ.get("LLM_RUNTIME_MODEL")
        if model_path:
            target_path = Path(model_path)
        else:
            info = best_model(detect_hardware().gpu_memory_gb)
            if info is None:
                raise RuntimeError("Aucun modèle disponible (ni Ollama, ni ~/.c3po/models, ni ./models)")
            target_path = info.path

    if _engine is not None and target_path == _model_path:
        return _engine

    if _engine is not None:
        del _engine
        gc.collect()

    _model_path = target_path
    _engine = Engine(_model_path, **_engine_overrides_from_env())
    return _engine


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None  # None → jusqu'à l'EOS / la limite de contexte
    temperature: float = 0.7
    stream: bool = False


@app.get("/v1/models")
def get_models():
    models = list_models()
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
        with _engine_lock:
            engine = get_engine(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(status_code=503, detail=str(e))

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
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "model": _model_path.name if _model_path else None,
    }
