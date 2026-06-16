"""
Moteur d'inférence : wrapper autour de llama-cpp-python.
Prend un modèle GGUF et génère du texte avec les bons paramètres.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Iterator

from .hardware import detect_hardware, HardwareProfile
from .params import compute_params, apply_overrides, InferenceParams
from .gguf import model_shape
from .instances import check_memory_pressure, register_instance


class Engine:
    """
    Charge un modèle GGUF et expose une interface simple pour générer du texte.
    Le hardware est détecté automatiquement si aucun profil n'est fourni.
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        profile: HardwareProfile | None = None,
        n_gpu_layers: int | None = None,
        n_threads: int | None = None,
        flash_attn: bool | None = None,
        kv_type: str | None = None,
        speculative: bool = False,
        repeat_penalty: float | None = None,
        force: bool = False,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        self.profile = profile or detect_hardware()
        self.size_gb = self.model_path.stat().st_size / (1024 ** 3)

        shape = model_shape(self.model_path)
        self.params = compute_params(
            self.profile,
            n_ctx=n_ctx,
            model_size_gb=self.size_gb,
            n_layers=shape["n_layers"],
            n_ctx_train=shape["n_ctx_train"],
            n_embd=shape["n_embd"],
            n_heads=shape["n_heads"],
            n_kv_heads=shape["n_kv_heads"],
            kv_type=kv_type,  # affecte le calcul de fit → passe par compute_params
        )
        # Autres leviers explicites (CLI) par-dessus les valeurs calculées.
        apply_overrides(
            self.params,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            use_flash_attn=flash_attn,
        )
        # Un KV cache quantifié exige la flash attention dans llama.cpp.
        if self.params.kv_type != "f16":
            self.params.use_flash_attn = True
            print(
                f"KV cache en {self.params.kv_type} (flash attention activée).",
                file=sys.stderr,
            )

        self.params.speculative = speculative
        if repeat_penalty is not None:
            self.params.repeat_penalty = repeat_penalty

        warning = check_memory_pressure(self.size_gb, self.profile.gpu_memory_gb)
        if warning:
            forced = force or os.environ.get("C3PO_FORCE") == "1"
            if forced:
                print(warning, file=sys.stderr)
                print("(C3PO_FORCE : chargement forcé malgré la pression mémoire.)",
                      file=sys.stderr)
            else:
                raise RuntimeError(
                    warning + "\n\nChargement bloqué pour éviter un dépassement mémoire. "
                    "Ferme une instance active, ou force avec C3PO_FORCE=1."
                )

        self._llm = self._load_model()
        register_instance(self.model_path.name, self.size_gb)

    def _load_model(self):
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python n'est pas installé.\n"
                "Installe-le avec : pip install llama-cpp-python"
            )

        kv_ggml = {
            "f16": llama_cpp.GGML_TYPE_F16,
            "q8_0": llama_cpp.GGML_TYPE_Q8_0,
            "q4_0": llama_cpp.GGML_TYPE_Q4_0,
        }[self.params.kv_type]

        # Prompt-lookup decoding : devine les prochains tokens en cherchant des n-grammes
        # déjà présents dans le contexte. Aucun modèle draft ni VRAM en plus.
        draft_model = None
        if self.params.speculative:
            from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
            draft_model = LlamaPromptLookupDecoding(max_ngram_size=2, num_pred_tokens=10)

        return Llama(
            model_path=str(self.model_path),
            n_gpu_layers=self.params.n_gpu_layers,
            n_threads=self.params.n_threads,
            n_ctx=self.params.n_ctx,
            flash_attn=self.params.use_flash_attn,
            type_k=kv_ggml,
            type_v=kv_ggml,
            draft_model=draft_model,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Génère du texte à partir d'un prompt.

        max_tokens=None → génère jusqu'à l'EOS (fin naturelle) ou la limite de contexte.
        stream=True  → retourne un itérateur de tokens au fil de l'eau
        stream=False → retourne le texte complet (défaut)
        """
        if stream:
            return self._stream(prompt, max_tokens, temperature)
        else:
            return self._complete(prompt, max_tokens, temperature)

    def _complete(self, prompt: str, max_tokens: int | None, temperature: float) -> str:
        result = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=self.params.repeat_penalty,
            echo=False,
        )
        return result["choices"][0]["text"]

    def _stream(
        self, prompt: str, max_tokens: int | None, temperature: float
    ) -> Iterator[str]:
        for chunk in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=self.params.repeat_penalty,
            stream=True,
            echo=False,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """
        Génère une réponse de chat au format OpenAI à partir d'une liste
        de messages ({"role": ..., "content": ...}).

        max_tokens=None → génère jusqu'à l'EOS (fin naturelle) ou la limite de contexte.
        Le chat template embarqué dans le GGUF est utilisé automatiquement.
        """
        return self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=self.params.repeat_penalty,
            stream=stream,
        )

    def count_text_tokens(self, text: str) -> int:
        """Estime le nombre de tokens d'un texte avec le tokenizer du modèle chargé."""
        try:
            return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))
        except TypeError:
            return len(self._llm.tokenize(text.encode("utf-8")))
        except Exception:
            # Repli grossier : utile pour les tests/mocks, pas pour piloter la prod.
            return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        """
        Estime la place prise par une liste de messages de chat.

        On ne matérialise pas le chat template exact de chaque modèle ; cette estimation
        sert de garde-fou pour compacter avant d'approcher la limite de contexte.
        """
        rendered = "\n".join(
            f"<{m.get('role', 'user')}>\n{m.get('content', '')}\n</{m.get('role', 'user')}>"
            for m in messages
        )
        return self.count_text_tokens(rendered)

    def __repr__(self) -> str:
        return (
            f"Engine(\n"
            f"  model={self.model_path.name}\n"
            f"  {self.profile}\n"
            f"  ---\n"
            f"  {self.params}\n"
            f")"
        )
