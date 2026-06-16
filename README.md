# c3po — llm-runtime

Orchestrateur Python autour de [llama.cpp](https://github.com/ggerganov/llama.cpp), avec
détection automatique du hardware (Apple Silicon / Metal ou Nvidia / CUDA) et calcul des
paramètres d'inférence optimaux. Un peu comme Ollama, mais avec les leviers exposés plutôt
que cachés.

## Installation

```bash
python3 -m pip install --user -e ".[dev]"
```

**Apple Silicon (Metal)** : le backend Metal est activé par défaut, rien de plus à faire.

**Nvidia (CUDA)** : `llama-cpp-python` doit être compilé avec le backend CUDA (nécessite le
CUDA toolkit `nvcc` + `cmake`) — les wheels précompilées sont trop anciennes pour les modèles
récents :

```bash
sudo apt install -y nvidia-cuda-toolkit          # fournit nvcc (ou repo NVIDIA)
python3 -m pip install cmake
CMAKE_ARGS="-DGGML_CUDA=on" python3 -m pip install --no-cache-dir llama-cpp-python
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
c3po search mistral --limit 30  # inspecter plus de repos (défaut : 20)

# Télécharge un modèle GGUF depuis Hugging Face (quant auto selon la VRAM)
c3po load bartowski/Qwen2.5-7B-Instruct-GGUF          # quant choisie selon la VRAM
c3po load bartowski/Qwen2.5-7B-Instruct-GGUF:Q5_K_M   # quant forcée

# Métadonnées d'un modèle + benchmark (chargement, tok/s) sur ce hardware
c3po stats <modèle>

# Chat interactif
c3po run <modèle>
# L'historique est compacté automatiquement quand il approche 95% du contexte du modèle.

# Leviers d'inférence exposés (sinon auto-calculés selon le hardware) — sur run/stats/serve/batch
c3po run <modèle> --ctx 8192 --n-gpu-layers 20 --threads 6 --no-flash-attn
# Quantization du KV cache (réduit la VRAM sur long contexte). Auto : F16, ou Q8 si besoin
# pour faire tenir le contexte ; --kv-type q4 force le mode le plus compact (qualité moindre).
c3po run <modèle> --ctx 32768 --kv-type q8

# Speculative decoding (prompt-lookup) : accélère les sorties qui recopient l'entrée
# (code, RAG, édition) — sortie identique, juste plus rapide. À éviter sur du texte créatif.
c3po run <modèle> --speculative

# Pénalité de répétition (défaut 1.1 ; monter si le modèle boucle ; 1.0 = aucune)
c3po run <modèle> --repeat-penalty 1.3

# Serveur HTTP compatible OpenAI (GET /v1/models, POST /v1/chat/completions, GET /health)
# Écoute sur 127.0.0.1 par défaut ; --host 0.0.0.0 pour exposer sur le réseau (sans auth !)
c3po serve [<modèle>] [--port 8000] [--host 127.0.0.1]

# Traitement batch (file séquentielle sur une seule instance modèle)
c3po batch <modèle> --input fichier1.txt fichier2.txt --prompt "Résume : {content}" \
    --output results.json [--ctx 2048] [--max-tokens N]
```

Par défaut, `run`/`serve`/`batch` génèrent jusqu'à la fin de la réponse (ou la limite de
contexte) — `--max-tokens` ne sert qu'à borner volontairement.

Les modèles téléchargés via `c3po load` vont dans `~/.c3po/models` (surchargeable via
`C3PO_MODELS_DIR`). Vous pouvez aussi déposer des `.gguf` dans `./models/` (non versionnés,
voir `.gitignore`) — les deux dossiers sont scannés.

Les commandes qui chargent un modèle (`run`, `serve`, `stats`, `batch`) arrêtent d'abord les
autres instances c3po actives. Le projet privilégie une seule instance modèle vivante à la fois :
moins de copies en VRAM/RAM, calculs mémoire plus prévisibles, moins de risques d'OOM.

## Développement

```bash
python3 -m pytest tests/ -v
pre-commit install
```

Voir [CONTEXT.md](CONTEXT.md) pour le détail de l'architecture et des choix techniques.
