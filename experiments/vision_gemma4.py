#!/usr/bin/env python3
"""
Wrapper CLI expérimental pour la primitive agent.vision.gemma4.
"""

from __future__ import annotations
import argparse

from agent.vision.gemma4 import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    format_observation,
    observe_image,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spike vision Gemma 4 : image -> observation structurée"
    )
    parser.add_argument("image", help="Chemin de l'image à analyser")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Modèle Gemma 4 GGUF (défaut: {DEFAULT_MODEL})")
    parser.add_argument("--mmproj", default=None,
                        help="Chemin du mmproj (défaut: recherche dans ~/.c3po/models)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    result = observe_image(
        image_path=args.image,
        model=args.model,
        mmproj=args.mmproj,
        prompt=args.prompt,
        n_ctx=args.ctx,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(format_observation(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
