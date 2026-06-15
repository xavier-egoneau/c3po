"""
Moteur d'inférence : wrapper autour de llama-cpp-python.
Prend un modèle GGUF et génère du texte avec les bons paramètres.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Iterator

from .hardware import detect_hardware, HardwareProfile
from .params import compute_params, InferenceParams
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
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        self.profile = profile or detect_hardware()
        self.params = compute_params(self.profile, n_ctx=n_ctx)
        self.size_gb = self.model_path.stat().st_size / (1024 ** 3)

        warning = check_memory_pressure(self.size_gb, self.profile.gpu_memory_gb)
        if warning:
            print(warning, file=sys.stderr)

        self._llm = self._load_model()
        register_instance(self.model_path.name, self.size_gb)

    def _load_model(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python n'est pas installé.\n"
                "Installe-le avec : pip install llama-cpp-python"
            )

        return Llama(
            model_path=str(self.model_path),
            n_gpu_layers=self.params.n_gpu_layers,
            n_threads=self.params.n_threads,
            n_ctx=self.params.n_ctx,
            use_flash_attn=self.params.use_flash_attn,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Génère du texte à partir d'un prompt.

        stream=True  → retourne un itérateur de tokens au fil de l'eau
        stream=False → retourne le texte complet (défaut)
        """
        if stream:
            return self._stream(prompt, max_tokens, temperature)
        else:
            return self._complete(prompt, max_tokens, temperature)

    def _complete(self, prompt: str, max_tokens: int, temperature: float) -> str:
        result = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False,
        )
        return result["choices"][0]["text"]

    def _stream(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> Iterator[str]:
        for chunk in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            echo=False,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """
        Génère une réponse de chat au format OpenAI à partir d'une liste
        de messages ({"role": ..., "content": ...}).

        Le chat template embarqué dans le GGUF est utilisé automatiquement.
        """
        return self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    def __repr__(self) -> str:
        return (
            f"Engine(\n"
            f"  model={self.model_path.name}\n"
            f"  {self.profile}\n"
            f"  ---\n"
            f"  {self.params}\n"
            f")"
        )
