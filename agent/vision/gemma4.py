"""
Primitive expérimentale Gemma 4 : image -> observation structurée.

Ce module reste hors API stable. Il sert à tester le sidecar vision de c3po-agent
avec c3po-core, sans dépendance Ollama.
"""

from __future__ import annotations
import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from agent.structured import FieldSpec, extract_json_object, validate_object
from llm_runtime.gguf import model_shape
from llm_runtime.hardware import detect_hardware
from llm_runtime.models import ModelInfo, find_model, models_dir
from llm_runtime.params import compute_params


DEFAULT_MODEL = "gemma-4-E2B-it-qat-UD-Q4_K_XL"
DEFAULT_PROMPT = (
    "Analyse l'image et réponds uniquement en JSON valide avec ces clés : "
    "caption, ocr, objects, layout, uncertainties. "
    "ocr doit contenir uniquement le texte visible dans l'image, jamais le texte de cette "
    "consigne ; utilise une liste vide si aucun texte n'est visible. "
    "Utilise des listes pour ocr, objects et uncertainties. Si tu ne sais pas, indique-le."
)
VISION_OBSERVATION_SCHEMA = {
    "caption": str,
    "ocr": list,
    "objects": list,
    "layout": FieldSpec((str, list), required=True),
    "uncertainties": list,
}


@dataclass
class VisionObservationResult:
    raw: str
    parsed: dict | None
    errors: list[str]
    model_path: str
    mmproj_path: str


def observe_image(
    image_path: str | Path,
    model: str = DEFAULT_MODEL,
    mmproj: str | Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    n_ctx: int = 4096,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> VisionObservationResult:
    """
    Produit une observation structurée d'une image avec Gemma 4 + mmproj.

    Charge le modèle dans le process courant. Pour un pipeline multi-modèle CUDA,
    préférer plus tard un runner subprocess éphémère.
    """
    image = Path(image_path).expanduser()
    if not image.exists():
        raise FileNotFoundError(f"Image introuvable : {image}")

    model_info = find_model(model)
    mmproj_path = Path(mmproj).expanduser() if mmproj else find_mmproj(model_info.path)
    if mmproj_path is None or not mmproj_path.exists():
        raise FileNotFoundError(
            "mmproj introuvable. Téléchargez-le avec : "
            "c3po load unsloth/gemma-4-E2B-it-qat-GGUF:Q4_K_XL --mmproj"
        )

    content = _run_gemma4_vision(
        image=image,
        model_info=model_info,
        mmproj_path=mmproj_path,
        prompt=prompt,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    parsed = parse_jsonish(content)
    return VisionObservationResult(
        raw=content,
        parsed=parsed,
        errors=validate_object(parsed, VISION_OBSERVATION_SCHEMA),
        model_path=str(model_info.path),
        mmproj_path=str(mmproj_path),
    )


def _run_gemma4_vision(
    image: Path,
    model_info: ModelInfo,
    mmproj_path: Path,
    prompt: str,
    n_ctx: int,
    max_tokens: int,
    temperature: float,
) -> str:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Gemma4ChatHandler

    profile = detect_hardware()
    shape = model_shape(model_info.path)
    params = compute_params(
        profile,
        n_ctx=n_ctx,
        model_size_gb=model_info.size_gb,
        n_layers=shape["n_layers"],
        n_ctx_train=shape["n_ctx_train"],
        n_embd=shape["n_embd"],
        n_heads=shape["n_heads"],
        n_kv_heads=shape["n_kv_heads"],
    )

    handler = Gemma4ChatHandler(clip_model_path=str(mmproj_path), verbose=False, use_gpu=True)
    llm = Llama(
        model_path=str(model_info.path),
        n_gpu_layers=params.n_gpu_layers,
        n_threads=params.n_threads,
        n_ctx=params.n_ctx,
        flash_attn=params.use_flash_attn,
        chat_handler=handler,
        verbose=False,
    )
    result = llm.create_chat_completion(
        messages=vision_messages(image, prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    return result["choices"][0]["message"]["content"]


def vision_messages(image_path: str | Path, prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri(image_path)}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def find_mmproj(model_path: str | Path) -> Path | None:
    model = Path(model_path)
    candidates = [p for p in model.parent.glob("*mmproj*.gguf") if p.is_file()]
    if not candidates:
        candidates = [p for p in models_dir().glob("*mmproj*.gguf") if p.is_file()]
    if not candidates:
        return None
    return sorted(candidates, key=_mmproj_sort_key)[0]


def data_uri(path: str | Path) -> str:
    image = Path(path)
    mime = mimetypes.guess_type(image.name)[0] or "image/png"
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_jsonish(text: str) -> dict | None:
    return extract_json_object(text)


def format_observation(result: VisionObservationResult) -> str:
    if result.parsed is not None:
        return json.dumps(result.parsed, ensure_ascii=False, indent=2)
    return result.raw


def _mmproj_sort_key(path: Path) -> tuple[int, int, str]:
    preference = {"BF16": 0, "F16": 1, "Q8_0": 2, "F32": 3}
    name = path.stem.upper()
    rank = next((v for q, v in preference.items() if q in name), 99)
    return (rank, path.stat().st_size, path.name)
