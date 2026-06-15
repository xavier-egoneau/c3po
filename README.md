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
# Liste les modèles disponibles (./models + Ollama) et le profil hardware détecté
c3po list

# Affiche le profil hardware et les paramètres d'inférence calculés
c3po info

# Chat interactif
c3po run <modèle>

# Serveur HTTP compatible OpenAI (GET /v1/models, POST /v1/chat/completions, GET /health)
c3po serve [<modèle>] [--port 8000]

# Traitement batch parallèle (un worker par fichier)
c3po batch <modèle> --input fichier1.txt fichier2.txt --prompt "Résume : {content}" --output results.json
```

Placez vos fichiers `.gguf` dans `./models/` (non versionnés, voir `.gitignore`).

## Développement

```bash
python3 -m pytest tests/ -v
pre-commit install
```

Voir [CONTEXT.md](CONTEXT.md) pour le détail de l'architecture et des choix techniques.
