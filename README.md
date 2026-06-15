# c3po — llm-runtime

Orchestrateur Python autour de [llama.cpp](https://github.com/ggerganov/llama.cpp), avec
détection automatique du hardware (Apple Silicon / Metal ou Nvidia / CUDA) et calcul des
paramètres d'inférence optimaux. Un peu comme Ollama, mais avec les leviers exposés plutôt
que cachés.

## Installation

```bash
python3 -m pip install --user -e ".[dev]"
```

## Utilisation

```bash
# Liste les modèles disponibles (~/.c3po/models + ./models + Ollama) et le hardware détecté
c3po list

# Affiche le profil hardware et les paramètres d'inférence calculés
c3po info

# Cherche sur Hugging Face les modèles GGUF qui tiennent dans ta VRAM
c3po search qwen2.5            # éligibles seulement
c3po search "llama 3" --all   # tout, avec colonne FIT

# Télécharge un modèle GGUF depuis Hugging Face (quant auto selon la VRAM)
c3po load bartowski/Qwen2.5-7B-Instruct-GGUF          # quant choisie selon la VRAM
c3po load bartowski/Qwen2.5-7B-Instruct-GGUF:Q5_K_M   # quant forcée

# Métadonnées d'un modèle + benchmark (chargement, tok/s) sur ce hardware
c3po stats <modèle>

# Chat interactif
c3po run <modèle>

# Serveur HTTP compatible OpenAI (GET /v1/models, POST /v1/chat/completions, GET /health)
c3po serve [<modèle>] [--port 8000]

# Traitement batch parallèle (un worker par fichier)
c3po batch <modèle> --input fichier1.txt fichier2.txt --prompt "Résume : {content}" --output results.json
```

Les modèles téléchargés via `c3po load` vont dans `~/.c3po/models` (surchargeable via
`C3PO_MODELS_DIR`). Vous pouvez aussi déposer des `.gguf` dans `./models/` (non versionnés,
voir `.gitignore`) — les deux dossiers sont scannés.

## Développement

```bash
python3 -m pytest tests/ -v
pre-commit install
```

Voir [CONTEXT.md](CONTEXT.md) pour le détail de l'architecture et des choix techniques.
