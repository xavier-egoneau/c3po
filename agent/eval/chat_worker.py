"""Worker de génération : charge UN modèle, génère une réponse, écrit le résultat, sort.

Isolation par process. Raison : alterner/recharger des modèles GPU in-process crashe sur
CUDA (ggml_cuda_error dans mul_mat_q / fattn). Un process par appel = chaque modèle vit
seul et tout est rendu proprement à la sortie. Lent (rechargement par appel) mais stable.

Usage : python -m agent.eval.chat_worker <requete.json> <sortie.txt>
  requete.json = {"model", "messages", "temperature"?, "max_tokens"?}
  la réponse texte est écrite dans <sortie.txt> (jamais sur stdout, pollué par les logs llama).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent.eval.solvers import engine_chat


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out_path = Path(argv[1])

    chat = engine_chat(
        request["model"],
        temperature=request.get("temperature", 0.2),
        max_tokens=request.get("max_tokens", 4096),
    )
    try:
        reply = chat(request["messages"])
    finally:
        close = getattr(chat, "close", None)
        if callable(close):
            close()
    out_path.write_text(reply, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
